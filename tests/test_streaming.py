import polars as pl
import pytest
from polspec import Bound, ColRule, ColSpec, ForeignKey, FrameSpec, col


class StreamDataSource(FrameSpec):
    id = ColSpec(dtype=pl.Int64, bounds=Bound(1, 1_000_000), nullable=False)
    category = ColSpec(
        dtype=pl.Enum(["alpha", "beta", "gamma"]),
        nullable=False,
    )
    score = ColSpec(
        dtype=pl.Float64,
        bounds=Bound(0.0, 100.0),
        nullable=True,
        rules=(
            ColRule(
                when=col("category") == "alpha",
                choices=[99.5],
            ),
        ),
    )
    comment = ColSpec(dtype=pl.String, nullable=True)
    active = ColSpec(dtype=pl.Boolean, nullable=False)


def test_generate_lazy():
    lf = StreamDataSource.generate(200, lazy=True, seed=42)
    assert isinstance(lf, pl.LazyFrame)
    assert lf.collect_schema() == StreamDataSource.schema()

    df = lf.collect()
    assert isinstance(df, pl.DataFrame)
    assert df.height == 200
    assert df.schema == StreamDataSource.schema()

    # Verify rules applied
    alpha_scores = df.filter(pl.col("category") == "alpha")["score"].drop_nulls()
    assert (alpha_scores == 99.5).all()


def test_generate_batches_random():
    batches = list(StreamDataSource.generate_batches(550, batch_size=200, seed=123))
    assert len(batches) == 3
    assert [b.height for b in batches] == [200, 200, 150]

    for batch in batches:
        assert batch.schema == StreamDataSource.schema()
        alpha_scores = batch.filter(pl.col("category") == "alpha")["score"].drop_nulls()
        assert (alpha_scores == 99.5).all()

    combined = pl.concat(batches)
    assert combined.height == 550


def test_generate_batches_cartesian():
    batches = list(
        StreamDataSource.generate_batches(
            300, batch_size=100, method="cartesian", seed=42
        )
    )
    assert len(batches) >= 1
    total_rows = sum(b.height for b in batches)
    assert total_rows >= 300

    for batch in batches:
        assert batch.schema == StreamDataSource.schema()


def test_generate_batches_deterministic():
    batches_1 = list(StreamDataSource.generate_batches(500, batch_size=150, seed=999))
    batches_2 = list(StreamDataSource.generate_batches(500, batch_size=150, seed=999))
    assert len(batches_1) == len(batches_2)
    for b1, b2 in zip(batches_1, batches_2, strict=True):
        assert b1.equals(b2)


def test_generate_batches_empty_and_invalid():
    assert list(StreamDataSource.generate_batches(0, batch_size=100)) == []

    with pytest.raises(ValueError, match="n must be >= 0"):
        list(StreamDataSource.generate_batches(-1, batch_size=100))

    with pytest.raises(ValueError, match="batch_size must be > 0"):
        list(StreamDataSource.generate_batches(100, batch_size=0))


def test_sink_parquet(tmp_path):
    parquet_path = tmp_path / "output.parquet"
    StreamDataSource.sink_parquet(
        parquet_path,
        n=750,
        batch_size=250,
        compression="zstd",
        seed=42,
    )

    assert parquet_path.exists()
    df = pl.read_parquet(parquet_path)
    assert df.height == 750
    assert df.schema == StreamDataSource.schema()

    # Rule check
    alpha_scores = df.filter(pl.col("category") == "alpha")["score"].drop_nulls()
    assert (alpha_scores == 99.5).all()


def test_sink_parquet_empty(tmp_path):
    parquet_path = tmp_path / "empty.parquet"
    StreamDataSource.sink_parquet(parquet_path, n=0)

    assert parquet_path.exists()
    df = pl.read_parquet(parquet_path)
    assert df.height == 0
    assert df.schema == StreamDataSource.schema()


def test_sink_csv(tmp_path):
    csv_path = tmp_path / "output.csv"
    StreamDataSource.sink_csv(
        csv_path,
        n=600,
        batch_size=200,
        seed=42,
    )

    assert csv_path.exists()
    df = pl.read_csv(csv_path, schema=StreamDataSource.schema())
    assert df.height == 600
    assert df.schema == StreamDataSource.schema()


def test_sink_csv_no_header_and_empty(tmp_path):
    csv_path = tmp_path / "no_header.csv"
    StreamDataSource.sink_csv(
        csv_path,
        n=100,
        batch_size=50,
        include_header=False,
        seed=42,
    )
    assert csv_path.exists()
    lines = csv_path.read_text().strip().split("\n")
    assert len(lines) == 100

    empty_csv = tmp_path / "empty.csv"
    StreamDataSource.sink_csv(empty_csv, n=0, include_header=True)
    assert empty_csv.exists()
    empty_lines = empty_csv.read_text().strip().split("\n")
    assert len(empty_lines) == 1  # only header line


def test_sink_ipc(tmp_path):
    ipc_path = tmp_path / "output.feather"
    StreamDataSource.sink_ipc(
        ipc_path,
        n=800,
        batch_size=250,
        compression="zstd",
        seed=42,
    )

    assert ipc_path.exists()
    df = pl.read_ipc(ipc_path)
    assert df.height == 800
    assert df.schema == StreamDataSource.schema()


def test_sink_ipc_empty(tmp_path):
    ipc_path = tmp_path / "empty.feather"
    StreamDataSource.sink_ipc(ipc_path, n=0)

    assert ipc_path.exists()
    df = pl.read_ipc(ipc_path)
    assert df.height == 0
    assert df.schema == StreamDataSource.schema()


def test_sink_ndjson(tmp_path):
    ndjson_path = tmp_path / "output.ndjson"
    StreamDataSource.sink_ndjson(
        ndjson_path,
        n=500,
        batch_size=150,
        seed=42,
    )

    assert ndjson_path.exists()
    df = pl.read_ndjson(ndjson_path, schema=StreamDataSource.schema())
    assert df.height == 500
    assert df.schema == StreamDataSource.schema()


def test_sink_nested_directory_creation(tmp_path):
    nested_path = tmp_path / "subdir1" / "subdir2" / "test.parquet"
    StreamDataSource.sink_parquet(nested_path, n=50, batch_size=25)
    assert nested_path.exists()
    df = pl.read_parquet(nested_path)
    assert df.height == 50


def test_a_lazy_parent_is_collected_once_not_once_per_batch(monkeypatch):
    """Resolving references belongs above the batch loop, not inside it.

    Each batch is its own `generate()` call, and reference resolution used to
    sit inside that call -- so a `LazyFrame` parent was collected once per
    batch. On a scan-backed parent that is the whole file re-read, however
    many batches there are.
    """
    from polspec import generation

    class Parent(FrameSpec):
        id = ColSpec(pl.Int64, bounds=(1, 1_000), unique=True)

    class Child(FrameSpec):
        id = ColSpec(pl.Int64, bounds=(1, 1_000))
        __foreign_keys__ = [ForeignKey("id", references=Parent)]

    parent = Parent.generate(200, seed=1)
    collected = 0
    real_collect = generation.to_eager

    def counting_collect(frame):
        nonlocal collected
        if isinstance(frame, pl.LazyFrame):
            collected += 1
        return real_collect(frame)

    monkeypatch.setattr(generation, "to_eager", counting_collect)

    batches = list(
        Child.generate_batches(
            500, batch_size=50, seed=2, references={Parent: parent.lazy()}
        )
    )
    assert len(batches) == 10
    assert collected == 1

    # And the frames are the same whichever form the parent arrived in.
    eager = list(
        Child.generate_batches(500, batch_size=50, seed=2, references={Parent: parent})
    )
    assert all(a.equals(b) for a, b in zip(batches, eager, strict=True))


# ---------------------------------------------------------------------------
# A batch is a window onto one frame
# ---------------------------------------------------------------------------


class PlainSource(FrameSpec):
    """Columns no pass rewrites, one per engine kind, so the whole frame and
    its batches can be compared row for row."""

    i = ColSpec(pl.Int64, bounds=(0, 1000), nullable=True, null_probability=0.2)
    f = ColSpec(pl.Float64)
    s = ColSpec(pl.String, string_length=(3, 12))
    e = ColSpec(pl.Enum(["a", "b", "c"]))
    d = ColSpec(pl.Date)
    m = ColSpec(pl.String, format="email")
    b = ColSpec(pl.Boolean)
    tags = ColSpec(pl.List(pl.Int64), list_length=(0, 3))


@pytest.mark.parametrize("batch_size", [997, 10_000, 65_536, 100_000, 131_072, 300_000])
def test_batches_are_windows_onto_the_whole_frame(batch_size):
    """`pl.concat(generate_batches(n, batch_size=b, seed=s))` equals
    `generate(n, seed=s)` for every `b`, for a column no pass rewrites:
    the engine numbers its chunks from the batch's offset. A List column's
    lengths are a window too; its elements are drawn per batch.
    """
    n = 300_000
    whole = PlainSource.generate(n, seed=5)
    parts = pl.concat(
        list(PlainSource.generate_batches(n, batch_size=batch_size, seed=5))
    )
    scalar = [c for c in whole.columns if c != "tags"]
    assert parts.select(scalar).equals(whole.select(scalar))
    assert parts["tags"].list.len().equals(whole["tags"].list.len())


def test_what_is_drawn_per_batch_is_deterministic_but_not_the_whole_frames():
    """Rules, foreign keys and composite keys are passes over a batch, so a
    batch's draws for them are its own -- the same every run, and distinct
    from the next batch's."""
    n = 2_000
    runs = [
        pl.concat(list(StreamDataSource.generate_batches(n, batch_size=500, seed=3)))
        for _ in range(2)
    ]
    assert runs[0].equals(runs[1])
    whole = StreamDataSource.generate(n, seed=3)
    assert (
        runs[0]
        .select("id", "category", "active")
        .equals(whole.select("id", "category", "active"))
    )
    ruled = pl.concat(
        list(StreamDataSource.generate_batches(n, batch_size=500, seed=3))
    )
    first, second = ruled.slice(0, 500), ruled.slice(500, 500)
    assert not first["score"].equals(second["score"])


def test_a_batch_seed_no_longer_depends_on_how_many_came_before():
    n, b = 200_000, 65_536
    batches = list(PlainSource.generate_batches(n, batch_size=b, seed=8))
    third = PlainSource.generate(n, seed=8).slice(2 * b, b)
    assert batches[2].select("i", "f", "s").equals(third.select("i", "f", "s"))
