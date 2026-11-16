//! Putting a column together from the parts the engine drew for it.
//!
//! Python draws a `List` column as two columns, each a window onto the frame
//! like any other: each cell's length, and every element of every cell in one
//! flat column. A `Decimal` column is drawn as its physical integer. Joining
//! the parts back into the declared column is cheap in Rust and dear in
//! Python, where Polars offers no way to build a list from offsets; so it is
//! done here, without touching a value.

use polars::prelude::*;
use polars_arrow::array::{LIST_VALUES_NAME, ListArray};
use polars_arrow::offset::OffsetsBuffer;

/// A `List` column named `name`: `elements` cut into cells, cell `i` holding
/// the next `lengths[i]` of them. A null length is a null cell, holding none.
///
/// The elements keep their dtype, logical or not: a list holds its values as
/// their physical arrays, with the logical dtype on the column, which is how
/// Polars builds one itself.
pub fn list_from_lengths(
    name: &str,
    lengths: &Series,
    elements: &Series,
) -> Result<Series, String> {
    let lengths = lengths.cast(&DataType::Int64).map_err(|e| e.to_string())?;
    let lengths = lengths.i64().map_err(|e| e.to_string())?.rechunk();
    let mut offsets: Vec<i64> = Vec::with_capacity(lengths.len() + 1);
    let mut end: i64 = 0;
    offsets.push(0);
    for length in lengths.iter() {
        let length = length.unwrap_or(0);
        if length < 0 {
            return Err(format!(
                "List column '{name}' was drawn a negative length {length}"
            ));
        }
        end += length;
        offsets.push(end);
    }
    let values = elements.rechunk();
    if end as usize != values.len() {
        return Err(format!(
            "List column '{name}' has {} element(s) for lengths totalling {end}",
            values.len()
        ));
    }
    let array = values
        .chunks()
        .first()
        .cloned()
        .ok_or_else(|| format!("List column '{name}' has no element array"))?;
    let validity = lengths
        .downcast_iter()
        .next()
        .and_then(|chunk| chunk.validity().cloned());
    let dtype = ArrowDataType::LargeList(Box::new(ArrowField::new(
        LIST_VALUES_NAME,
        array.dtype().clone(),
        true,
    )));
    let offsets = OffsetsBuffer::try_from(offsets).map_err(|e| e.to_string())?;
    let list =
        ListArray::<i64>::try_new(dtype, offsets, array, validity).map_err(|e| e.to_string())?;
    // SAFETY: the one chunk is a large list whose values are `values`' own
    // array, the physical form of `values.dtype()`, the dtype declared here.
    let column = unsafe {
        ListChunked::from_chunks_and_dtype(
            PlSmallStr::from(name),
            vec![list.boxed()],
            DataType::List(Box::new(values.dtype().clone())),
        )
    };
    Ok(column.into_series())
}

/// A `Decimal(precision, scale)` column from its physical integers: the
/// integer `v` is the decimal `v / 10^scale`.
pub fn decimal_from_physical(
    raw: &Series,
    precision: usize,
    scale: usize,
) -> Result<Series, String> {
    raw.cast(&DataType::Int128)
        .and_then(|wide| wide.into_decimal(precision, scale))
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lengths(values: &[Option<i64>]) -> Series {
        Series::new("len".into(), values)
    }

    #[test]
    fn cells_take_their_lengths_in_order_and_a_null_length_is_a_null_cell() {
        let elements = Series::new("e".into(), &[1i64, 2, 3, 4, 5, 6]);
        let list = list_from_lengths(
            "c",
            &lengths(&[Some(2), Some(0), None, Some(3), Some(1)]),
            &elements,
        )
        .unwrap();
        let cells = list.list().unwrap();
        let as_vecs: Vec<Option<Vec<i64>>> = (0..cells.len())
            .map(|i| {
                cells
                    .get_as_series(i)
                    .map(|cell| cell.i64().unwrap().into_no_null_iter().collect())
            })
            .collect();
        assert_eq!(
            as_vecs,
            vec![
                Some(vec![1, 2]),
                Some(vec![]),
                None,
                Some(vec![3, 4, 5]),
                Some(vec![6])
            ]
        );
        assert_eq!(list.name().as_str(), "c");
    }

    #[test]
    fn a_logical_dtype_is_kept() {
        let elements = Series::new("e".into(), &[0i32, 1, 2])
            .cast(&DataType::Date)
            .unwrap();
        let list = list_from_lengths("d", &lengths(&[Some(1), Some(2)]), &elements).unwrap();
        assert_eq!(list.dtype(), &DataType::List(Box::new(DataType::Date)));
    }

    #[test]
    fn elements_in_several_chunks_are_one_array() {
        let mut elements = Series::new("e".into(), &[1i64, 2]);
        elements.append(&Series::new("e".into(), &[3i64])).unwrap();
        assert_eq!(elements.n_chunks(), 2);
        let list = list_from_lengths("c", &lengths(&[Some(1), Some(2)]), &elements).unwrap();
        assert_eq!(list.list().unwrap().get_as_series(1).unwrap().len(), 2);
    }

    #[test]
    fn a_length_mismatch_is_refused_by_name() {
        let elements = Series::new("e".into(), &[1i64]);
        let err = list_from_lengths("tags", &lengths(&[Some(2)]), &elements).unwrap_err();
        assert!(err.contains("'tags'"), "{err}");
    }

    #[test]
    fn no_cells_and_no_elements_is_an_empty_list() {
        let elements = Series::new_empty("e".into(), &DataType::String);
        let list = list_from_lengths("c", &lengths(&[]), &elements).unwrap();
        assert_eq!(list.len(), 0);
        assert_eq!(list.dtype(), &DataType::List(Box::new(DataType::String)));
    }

    #[test]
    fn a_decimal_is_its_integer_over_ten_to_the_scale() {
        let raw = Series::new("d".into(), &[Some(12345i64), None, Some(-5)]);
        let decimal = decimal_from_physical(&raw, 12, 2).unwrap();
        assert_eq!(decimal.dtype(), &DataType::Decimal(12, 2));
        let text = decimal.cast(&DataType::String).unwrap();
        let text = text.str().unwrap();
        let text: Vec<Option<String>> = (0..text.len())
            .map(|i| text.get(i).map(str::to_string))
            .collect();
        assert_eq!(
            text,
            vec![Some("123.45".into()), None, Some("-0.05".into())]
        );
    }
}
