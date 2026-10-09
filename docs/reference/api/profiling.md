# Profiling

Inferring a spec from data you already have, and generating a stand-in for
it. See [Fake data from real data](../../how-to/synthesizing.md) for the
workflow, and what carries over from the source.

## synthesize

::: polspec.synthesize

## profile

::: polspec.profile

## profile_dataframe (deprecated)

Deprecated in 0.18.0, removed in 1.0: `profile` takes the same switches. The
call that gives the same columns is
`profile(df, weights=False, shape=False, detect_unique=False).columns`.

::: polspec.profile_dataframe
