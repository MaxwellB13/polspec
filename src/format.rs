//! Strings with a shape: a `Template` and the sampler that fills one.
//!
//! The engine knows nothing about emails or UUIDs. Python owns every named
//! format -- `polspec.formats` -- and hands each column a `Template`: a run of
//! parts, each either a literal, a stretch of characters drawn from an
//! alphabet, or one value from a list. That is enough to spell every format
//! polspec ships, and it keeps the knowledge of what a format *is* in one
//! Python file beside the validator that checks it, which is what stops the
//! two sides drifting.
//!
//! A `Template` is built once per column, at plan time, so a bad part is
//! refused with the column's name before any row is drawn.
//!
//! A template's values are also enumerable: `decode` maps an index to a
//! value, reading one mixed-radix digit per part, and is a bijection on
//! `[0, cardinality())`. That is what lets a unique string column be a
//! permutation of its value space rather than a draw against a set.

use rand::Rng;
use rand::distr::{Distribution as _, Uniform};
use rand_xoshiro::Xoshiro256PlusPlus;

/// One piece of a templated string.
#[derive(Clone, Debug, PartialEq)]
pub enum Part {
    /// Emitted as written.
    Literal(String),
    /// `min..=max` characters, each drawn uniformly from `alphabet`.
    Chars {
        alphabet: Vec<u8>,
        min: usize,
        max: usize,
    },
    /// One of `values`, uniformly.
    OneOf(Vec<String>),
}

/// The raw form a part crosses the Python boundary in: `(kind, strings, lo, hi)`.
///
/// One tuple shape for every kind, so pyo3 extracts a `Vec` of them with no
/// custom conversion. `strings` is the literal text, the alphabet, or the
/// values; `lo`/`hi` are the length range and are read only by `chars`.
pub type RawPart = (String, Vec<String>, usize, usize);

/// A run of parts, compiled and checked.
#[derive(Clone, Debug, PartialEq)]
pub struct Template {
    parts: Vec<Part>,
    /// An upper bound on the bytes one value can take, so a scratch buffer can
    /// be sized once per chunk instead of grown per row.
    max_bytes: usize,
}

impl Template {
    /// Compiles the raw parts, naming `column` in every refusal.
    pub fn compile(raw: &[RawPart], column: &str) -> Result<Self, String> {
        if raw.is_empty() {
            return Err(format!("Column '{column}' has an empty format template"));
        }
        let mut parts = Vec::with_capacity(raw.len());
        let mut max_bytes = 0usize;
        for (index, (kind, strings, lo, hi)) in raw.iter().enumerate() {
            let part = match kind.as_str() {
                "lit" => {
                    let text = strings.first().cloned().ok_or_else(|| {
                        format!("Column '{column}': template part {index} ('lit') has no text")
                    })?;
                    max_bytes += text.len();
                    Part::Literal(text)
                }
                "chars" => {
                    let alphabet = strings.first().ok_or_else(|| {
                        format!(
                            "Column '{column}': template part {index} ('chars') has no alphabet"
                        )
                    })?;
                    if alphabet.is_empty() {
                        return Err(format!(
                            "Column '{column}': template part {index} ('chars') has an empty alphabet"
                        ));
                    }
                    if !alphabet.is_ascii() {
                        return Err(format!(
                            "Column '{column}': template part {index} ('chars') alphabet must be ASCII"
                        ));
                    }
                    if *lo > *hi || *hi == 0 {
                        return Err(format!(
                            "Column '{column}': template part {index} ('chars') needs 0 < min <= max, got {lo}..{hi}"
                        ));
                    }
                    max_bytes += *hi;
                    Part::Chars {
                        alphabet: alphabet.as_bytes().to_vec(),
                        min: *lo,
                        max: *hi,
                    }
                }
                "one_of" => {
                    if strings.is_empty() {
                        return Err(format!(
                            "Column '{column}': template part {index} ('one_of') has no values"
                        ));
                    }
                    max_bytes += strings.iter().map(String::len).max().unwrap_or(0);
                    Part::OneOf(strings.clone())
                }
                other => {
                    return Err(format!(
                        "Column '{column}': template part {index} has unknown kind '{other}'; \
                         expected 'lit', 'chars' or 'one_of'"
                    ));
                }
            };
            parts.push(part);
        }
        Ok(Template { parts, max_bytes })
    }

    /// The largest value this template can produce, in bytes.
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// A template of one run of `min..=max` characters from `alphabet`: a
    /// plain string column's value space, so it decodes like any format.
    pub fn run(alphabet: &[u8], min: usize, max: usize) -> Self {
        Template {
            parts: vec![Part::Chars {
                alphabet: alphabet.to_vec(),
                min,
                max: max.max(min),
            }],
            max_bytes: max.max(min),
        }
    }

    /// How many distinct values this template can produce, saturating at
    /// `u128::MAX`: a `chars` part of width `w` over an alphabet of `a`
    /// contributes `a^w` for each length in its range.
    pub fn cardinality(&self) -> u128 {
        self.parts.iter().fold(1u128, |total, part| {
            total.saturating_mul(part.cardinality())
        })
    }

    /// Writes the value at `index` into `out`, which is cleared first.
    ///
    /// Each part reads one digit of `index` in its own radix, its
    /// cardinality: a literal none, a one-of the value's position, a run of
    /// characters a width bucket and then a number in base `alphabet.len()`,
    /// written at that width. Distinct indices below `cardinality()` give
    /// distinct values. A saturated radix is read as the digit it is -- the
    /// index is below it, so the digit is the whole index -- and stays
    /// injective, since no two indices can reach the same digits.
    pub fn decode(&self, mut index: u128, out: &mut Vec<u8>) {
        out.clear();
        for part in &self.parts {
            match part {
                Part::Literal(text) => out.extend_from_slice(text.as_bytes()),
                Part::OneOf(values) => {
                    let radix = values.len() as u128;
                    out.extend_from_slice(values[(index % radix) as usize].as_bytes());
                    index /= radix;
                }
                Part::Chars { alphabet, min, max } => {
                    let radix = part.cardinality();
                    let mut digit = index % radix;
                    index /= radix;
                    let a = alphabet.len() as u128;
                    let mut width = *min;
                    loop {
                        let bucket = a.saturating_pow(width as u32);
                        if digit < bucket || width == *max {
                            break;
                        }
                        digit -= bucket;
                        width += 1;
                    }
                    let start = out.len();
                    out.resize(start + width, alphabet[0]);
                    for slot in out[start..].iter_mut().rev() {
                        *slot = alphabet[(digit % a) as usize];
                        digit /= a;
                    }
                }
            }
        }
    }

    /// A sampler over this template, with its distributions prepared once.
    pub fn sampler(&self) -> TemplateSampler<'_> {
        let parts = self
            .parts
            .iter()
            .map(|part| match part {
                Part::Literal(text) => PartSampler::Literal(text.as_bytes()),
                Part::Chars { alphabet, min, max } => PartSampler::Chars {
                    alphabet,
                    // An alphabet is non-empty and ASCII, so these ranges are valid.
                    index: Uniform::new(0, alphabet.len()).expect("non-empty alphabet"),
                    len: (min < max)
                        .then(|| Uniform::new_inclusive(*min, *max).expect("min <= max")),
                    min: *min,
                },
                Part::OneOf(values) => PartSampler::OneOf {
                    values,
                    index: Uniform::new(0, values.len()).expect("non-empty values"),
                },
            })
            .collect();
        TemplateSampler { parts }
    }
}

impl Part {
    /// How many distinct strings this part contributes, saturating.
    fn cardinality(&self) -> u128 {
        match self {
            Part::Literal(_) => 1,
            Part::OneOf(values) => values.len() as u128,
            Part::Chars { alphabet, min, max } => {
                let a = alphabet.len() as u128;
                (*min..=*max)
                    .map(|w| a.saturating_pow(w as u32))
                    .fold(0u128, |acc, x| acc.saturating_add(x))
            }
        }
    }
}

enum PartSampler<'a> {
    Literal(&'a [u8]),
    Chars {
        alphabet: &'a [u8],
        index: Uniform<usize>,
        len: Option<Uniform<usize>>,
        min: usize,
    },
    OneOf {
        values: &'a [String],
        index: Uniform<usize>,
    },
}

/// Fills a byte buffer with one value after another.
pub struct TemplateSampler<'a> {
    parts: Vec<PartSampler<'a>>,
}

impl TemplateSampler<'_> {
    /// Writes one value into `out`, which is cleared first.
    #[inline]
    pub fn fill<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut Vec<u8>) {
        out.clear();
        for part in &self.parts {
            match part {
                PartSampler::Literal(text) => out.extend_from_slice(text),
                PartSampler::Chars {
                    alphabet,
                    index,
                    len,
                    min,
                } => {
                    let width = len.as_ref().map_or(*min, |d| d.sample(rng));
                    for _ in 0..width {
                        out.push(alphabet[index.sample(rng)]);
                    }
                }
                PartSampler::OneOf { values, index } => {
                    out.extend_from_slice(values[index.sample(rng)].as_bytes());
                }
            }
        }
    }

    /// One value as a `String`.
    pub fn draw(&self, rng: &mut Xoshiro256PlusPlus, scratch: &mut Vec<u8>) -> String {
        self.fill(rng, scratch);
        // Every part is ASCII or a String, so the buffer is valid UTF-8.
        String::from_utf8(scratch.clone()).expect("template output is UTF-8")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn raw(kind: &str, strings: &[&str], lo: usize, hi: usize) -> RawPart {
        (
            kind.to_string(),
            strings.iter().map(|s| s.to_string()).collect(),
            lo,
            hi,
        )
    }

    #[test]
    fn a_template_fills_each_part_in_order() {
        let t = Template::compile(
            &[
                raw("chars", &["ab"], 2, 2),
                raw("lit", &["-"], 0, 0),
                raw("one_of", &["x", "yy"], 0, 0),
            ],
            "c",
        )
        .unwrap();
        let sampler = t.sampler();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        let mut scratch = Vec::with_capacity(t.max_bytes());
        for _ in 0..50 {
            let s = sampler.draw(&mut rng, &mut scratch);
            let (head, tail) = s.split_once('-').unwrap();
            assert_eq!(head.len(), 2);
            assert!(head.bytes().all(|b| b == b'a' || b == b'b'), "{s}");
            assert!(tail == "x" || tail == "yy", "{s}");
        }
    }

    #[test]
    fn cardinality_multiplies_the_parts() {
        let t = Template::compile(
            &[
                raw("chars", &["abc"], 1, 2),     // 3 + 9
                raw("lit", &["-"], 0, 0),         // 1
                raw("one_of", &["x", "y"], 0, 0), // 2
            ],
            "c",
        )
        .unwrap();
        assert_eq!(t.cardinality(), 24);
        assert_eq!(t.max_bytes(), 2 + 1 + 1);
    }

    #[test]
    fn a_bad_part_names_the_column_and_the_part() {
        let err = Template::compile(&[raw("chars", &[""], 1, 2)], "email").unwrap_err();
        assert!(err.contains("'email'") && err.contains("part 0"), "{err}");
        let err = Template::compile(&[raw("chars", &["ab"], 3, 2)], "c").unwrap_err();
        assert!(err.contains("min <= max"), "{err}");
        let err = Template::compile(&[raw("shout", &["!"], 0, 0)], "c").unwrap_err();
        assert!(err.contains("unknown kind 'shout'"), "{err}");
        assert!(Template::compile(&[], "c").is_err());
    }

    fn decoded(t: &Template) -> Vec<String> {
        let mut out = Vec::new();
        (0..t.cardinality())
            .map(|i| {
                t.decode(i, &mut out);
                String::from_utf8(out.clone()).unwrap()
            })
            .collect()
    }

    #[test]
    fn decoding_is_a_bijection_onto_the_values() {
        // 12 widths-and-letters x 1 literal x 2 choices: every index gives a
        // different value, and every value the template can make appears.
        let t = Template::compile(
            &[
                raw("chars", &["abc"], 1, 2),
                raw("lit", &["-"], 0, 0),
                raw("one_of", &["x", "yy"], 0, 0),
            ],
            "c",
        )
        .unwrap();
        let values = decoded(&t);
        assert_eq!(values.len(), 24);
        let distinct: std::collections::HashSet<&String> = values.iter().collect();
        assert_eq!(distinct.len(), 24);
        for v in &values {
            let (head, tail) = v.split_once('-').unwrap();
            assert!((1..=2).contains(&head.len()) && head.bytes().all(|b| b"abc".contains(&b)));
            assert!(tail == "x" || tail == "yy", "{v}");
        }
    }

    #[test]
    fn a_run_holds_every_string_its_lengths_allow_including_the_empty_one() {
        let values = decoded(&Template::run(b"ab", 0, 2));
        let mut sorted = values.clone();
        sorted.sort();
        assert_eq!(sorted, ["", "a", "aa", "ab", "b", "ba", "bb"]);
    }

    #[test]
    fn a_saturated_space_still_decodes_distinct_values() {
        // 62 letters up to 40 wide: far past u128, so the radix saturates.
        let t = Template::run(
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
            1,
            40,
        );
        assert_eq!(t.cardinality(), u128::MAX);
        let mut out = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for i in [0, 1, 61, 62, 1 << 64, u128::MAX - 1, u128::MAX / 3] {
            t.decode(i, &mut out);
            assert!(out.len() <= 40);
            assert!(seen.insert(out.clone()), "index {i} repeated a value");
        }
    }

    #[test]
    fn the_same_seed_gives_the_same_value() {
        let t = Template::compile(&[raw("chars", &["0123456789"], 4, 8)], "c").unwrap();
        let s = t.sampler();
        let mut a = Xoshiro256PlusPlus::seed_from_u64(7);
        let mut b = Xoshiro256PlusPlus::seed_from_u64(7);
        let mut buf = Vec::new();
        assert_eq!(s.draw(&mut a, &mut buf), s.draw(&mut b, &mut buf));
    }
}
