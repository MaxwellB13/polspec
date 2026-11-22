//! A string array built straight from its views.
//!
//! Polars backs a string column with a view array: sixteen bytes a row,
//! holding a string of up to twelve bytes inline and, for a longer one, its
//! length, a four-byte prefix and where its bytes sit in a data buffer.
//! `StringChunkedBuilder` does that one value at a time behind an `Option`
//! and a growth check per row; a generated column knows every value is
//! present or null and valid UTF-8, so `Views` writes the views and the data
//! itself, and hands Polars the finished array.

use polars_arrow::array::{Utf8ViewArray, View};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::ArrowDataType;
use polars_buffer::Buffer;

/// The most bytes one data buffer holds before the next is started: a view
/// addresses its bytes by a `u32` offset, and Polars' own builder keeps its
/// buffers to tens of megabytes.
const BUFFER_BYTES: usize = 32 << 20;

/// One task's strings, in row order.
pub(crate) struct Views {
    views: Vec<View>,
    buffers: Vec<Buffer<u8>>,
    data: Vec<u8>,
    total_bytes: usize,
    total_buffers: usize,
    /// Set bits for present rows, when the column draws nulls.
    validity: Option<Vec<u8>>,
}

impl Views {
    /// Room for `rows` views and `data` bytes of strings too long to inline.
    pub(crate) fn with_capacity(rows: usize, nullable: bool, data: usize) -> Self {
        Views {
            views: Vec::with_capacity(rows),
            buffers: Vec::new(),
            data: Vec::with_capacity(data.min(BUFFER_BYTES)),
            total_bytes: 0,
            total_buffers: 0,
            validity: nullable.then(|| vec![0u8; rows.div_ceil(8)]),
        }
    }

    /// Appends a present value.
    #[inline(always)]
    pub(crate) fn push(&mut self, value: &str) {
        let bytes = value.as_bytes();
        let row = self.views.len();
        if let Some(bits) = self.validity.as_mut() {
            bits[row / 8] |= 1 << (row % 8);
        }
        self.total_bytes += bytes.len();
        let view = if bytes.len() <= View::MAX_INLINE_SIZE as usize {
            // SAFETY: the length was checked just above.
            unsafe { View::new_inline_unchecked(bytes) }
        } else {
            if self.data.len() + bytes.len() > BUFFER_BYTES && !self.data.is_empty() {
                self.roll();
            }
            let offset = self.data.len() as u32;
            self.data.extend_from_slice(bytes);
            // SAFETY: longer than the inline size, so the bytes live in
            // buffer `self.buffers.len()`, which `finish` or `roll` pushes.
            unsafe { View::new_noninline_unchecked(bytes, self.buffers.len() as u32, offset) }
        };
        self.views.push(view);
    }

    /// Appends the present value `padded[at..at + len]`, which must be valid
    /// UTF-8, where `padded` holds at least 16 bytes from `at`. A value short
    /// enough to inline is read as one 16-byte word and masked to its length,
    /// with no copy of a variable length -- the padding past it zero, as
    /// Polars compares an inline view as a whole.
    #[inline(always)]
    pub(crate) fn push_padded(&mut self, padded: &[u8], at: usize, len: usize) {
        if len > View::MAX_INLINE_SIZE as usize {
            // SAFETY: the caller guarantees the bytes are UTF-8.
            return self.push(unsafe { std::str::from_utf8_unchecked(&padded[at..at + len]) });
        }
        let row = self.views.len();
        if let Some(bits) = self.validity.as_mut() {
            bits[row / 8] |= 1 << (row % 8);
        }
        self.total_bytes += len;
        let word: [u8; 16] = padded[at..at + 16].try_into().expect("16 bytes of padding");
        let data = u128::from_le_bytes(word) & ((1u128 << (8 * len)) - 1);
        self.views.push(View::from(len as u128 | (data << 32)));
    }

    /// `bytes` more bytes at the end of the current data buffer, for a
    /// caller to fill and then point views at with `push_stored`; a new
    /// buffer is started first if they would not fit. Returns where they
    /// start in that buffer.
    pub(crate) fn reserve_stored(&mut self, bytes: usize) -> (usize, &mut [u8]) {
        if self.data.len() + bytes > BUFFER_BYTES && !self.data.is_empty() {
            self.roll();
        }
        let start = self.data.len();
        self.data.resize(start + bytes, 0);
        (start, &mut self.data[start..])
    }

    /// Gives back the last `bytes` reserved: scratch a fill wrote past the
    /// end of what it kept.
    pub(crate) fn release_stored(&mut self, bytes: usize) {
        self.data.truncate(self.data.len() - bytes);
    }

    /// Appends a present value already in the current data buffer at
    /// `offset`, from `reserve_stored`: longer than the inline size, and
    /// valid UTF-8.
    #[inline(always)]
    pub(crate) fn push_stored(&mut self, offset: usize, len: usize) {
        debug_assert!(len > View::MAX_INLINE_SIZE as usize);
        let row = self.views.len();
        if let Some(bits) = self.validity.as_mut() {
            bits[row / 8] |= 1 << (row % 8);
        }
        self.total_bytes += len;
        let bytes = &self.data[offset..offset + len];
        // SAFETY: longer than the inline size, and in the current buffer,
        // which `finish` or `roll` pushes as buffer `self.buffers.len()`.
        let view = unsafe {
            View::new_noninline_unchecked(bytes, self.buffers.len() as u32, offset as u32)
        };
        self.views.push(view);
    }

    /// Appends a null: an empty view, its validity bit left unset.
    #[inline(always)]
    pub(crate) fn push_null(&mut self) {
        debug_assert!(self.validity.is_some(), "a null in a column without nulls");
        self.views.push(View::default());
    }

    fn roll(&mut self) {
        let data = std::mem::take(&mut self.data);
        self.total_buffers += data.len();
        self.buffers.push(Buffer::from(data));
    }

    pub(crate) fn finish(mut self) -> Utf8ViewArray {
        if !self.data.is_empty() {
            self.roll();
        }
        let rows = self.views.len();
        let validity = self.validity.map(|bits| Bitmap::from_u8_vec(bits, rows));
        // SAFETY: every value pushed was a `&str`, and every non-inline view
        // addresses bytes of the buffer it names, pushed above.
        unsafe {
            Utf8ViewArray::new_unchecked(
                ArrowDataType::Utf8View,
                Buffer::from(self.views),
                Buffer::from(self.buffers),
                validity,
                Some(self.total_bytes),
                self.total_buffers,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    fn column(values: &[Option<&str>]) -> StringChunked {
        let nullable = values.iter().any(Option::is_none);
        let mut views = Views::with_capacity(values.len(), nullable, 0);
        for value in values {
            match value {
                Some(v) => views.push(v),
                None => views.push_null(),
            }
        }
        StringChunked::with_chunk(PlSmallStr::from("c"), views.finish())
    }

    #[test]
    fn views_hold_what_a_builder_would() {
        let long = "a string well past twelve bytes";
        let values = [
            Some("short"),
            None,
            Some(""),
            Some(long),
            Some("exactly12byt"),
            None,
        ];
        let built = column(&values);
        let expected = StringChunked::from_iter_options(PlSmallStr::from("c"), values.into_iter());
        assert!(built.equal_missing(&expected).all());
        assert_eq!(built.null_count(), 2);
        // Reads that go through the data buffers and the cached byte total.
        assert_eq!(built.get(3), Some(long));
        let array = built.downcast_iter().next().unwrap();
        assert_eq!(array.total_bytes_len(), 5 + long.len() + 12);
    }

    #[test]
    fn a_padded_push_is_the_view_a_plain_push_makes() {
        let text = b"abcdefghijklmnopqrstuvwxyz0123456789";
        let mut padded = text.to_vec();
        padded.extend_from_slice(&[0xAA; 16]); // padding that must not leak in
        for at in [0, 3, 10] {
            for len in 0..=20 {
                let value = std::str::from_utf8(&text[at..at + len]).unwrap();
                let mut plain = Views::with_capacity(1, false, 0);
                plain.push(value);
                let mut fast = Views::with_capacity(1, false, 0);
                fast.push_padded(&padded, at, len);
                let (plain, fast) = (plain.finish(), fast.finish());
                assert_eq!(
                    plain.views()[0].as_u128(),
                    fast.views()[0].as_u128(),
                    "{at}+{len}"
                );
                assert_eq!(fast.value(0), value);
            }
        }
    }

    #[test]
    fn stored_values_are_the_bytes_reserved_for_them() {
        let mut views = Views::with_capacity(3, true, 0);
        let (start, room) = views.reserve_stored(40);
        room.copy_from_slice(b"twenty bytes of textand twenty more.....");
        views.push_stored(start, 20);
        views.push_null();
        views.push_stored(start + 20, 20);
        let column = StringChunked::with_chunk(PlSmallStr::from("c"), views.finish());
        assert_eq!(column.get(0), Some("twenty bytes of text"));
        assert_eq!(column.get(1), None);
        assert_eq!(column.get(2), Some("and twenty more....."));
    }

    #[test]
    fn a_long_column_spans_several_buffers() {
        let value = "x".repeat(1 << 20);
        let values: Vec<Option<&str>> = (0..70).map(|_| Some(value.as_str())).collect();
        let built = column(&values);
        let array = built.downcast_iter().next().unwrap();
        assert!(array.data_buffers().len() > 1);
        assert_eq!(built.get(69), Some(value.as_str()));
        assert_eq!(built.len(), 70);
    }
}
