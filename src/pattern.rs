//! Strings from a pattern: a random walk through the pattern's automaton.
//!
//! Polars validates a `pattern` with the `regex` crate. Generation compiles
//! the same pattern with `regex-automata` -- the engine underneath `regex` --
//! into a DFA, so what is generated and what is validated read the pattern's
//! syntax and meaning alike.
//!
//! A value is a full match of the pattern, which `str.contains` also
//! accepts. The DFA runs over *units*: printable ASCII characters, and only
//! when no ASCII string can match, the pattern's own non-ASCII characters
//! too. Its states are compacted once per pattern, and a table records, for
//! each state and each remaining length, whether an accepting end is still
//! reachable. A row then picks a length among those the pattern can fill,
//! and walks: at each step it chooses uniformly among the units that keep an
//! accepting end in reach, so the walk never dead-ends and needs no retry.
//!
//! A unique column numbers the strings instead: for each state and length,
//! how many unit strings end in a match from there, so an index in
//! `[0, cardinality())` decodes to one string and distinct indices to
//! distinct strings -- what a keyed permutation needs, as a format's
//! template provides it.
//!
//! Lengths are counted in characters, as `string_length` and validation
//! count them. A pattern out of reach -- an automaton too large, a Unicode
//! word boundary, no match within the lengths allowed -- is an `Err` naming
//! why, and the column keeps the text it drew before 1.4.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};

use crate::format::Enumerable;

use rand::RngExt;
use rand_xoshiro::Xoshiro256PlusPlus;
use regex_automata::dfa::{Automaton, StartKind, dense};
use regex_automata::util::primitives::StateID;
use regex_automata::util::start;
use regex_automata::{Anchored, MatchKind};
use regex_syntax::hir::{Class, Hir, HirKind};

use crate::sample::CHARSET;

/// The most memory one pattern's DFA may take, in bytes.
const DFA_SIZE_LIMIT: usize = 4 << 20;
/// The most states a pattern may reach over its units.
const MAX_STATES: usize = 4_096;
/// The longest value generated, in characters.
pub const MAX_LENGTH: usize = 4_096;
/// How far past its shortest match an unbounded repeat runs, when no
/// `string_length` says otherwise.
pub const UNBOUNDED_EXTRA: usize = 16;
/// The most non-ASCII characters taken from a pattern for its units.
const MAX_EXTRA_UNITS: usize = 512;
/// The most taken from any one range of a class.
const PER_RANGE: u32 = 8;
/// The most counts a unique pattern may keep: states times lengths.
const MAX_COUNTS: usize = 4 << 20;

/// The units leaving one state for the same next state.
#[derive(Debug)]
struct Group {
    next: u32,
    units: Vec<u16>,
}

/// A compiled pattern, ready to fill values.
#[derive(Debug)]
pub struct PatternSampler {
    units: Vec<Box<str>>,
    /// Each unit's one byte, when every unit is a single ASCII character --
    /// almost every pattern -- so a step writes a byte, not a string.
    ascii: Option<Vec<u8>>,
    /// Per state, the groups of units out of it, in a fixed order.
    groups: Vec<Vec<Group>>,
    /// `viable[k]`: the states from which exactly `k` more units can end in
    /// a match, as a bitset.
    viable: Vec<Vec<u64>>,
    /// The lengths a value may have, in characters, ascending.
    lengths: Vec<u32>,
    start: u32,
    max_bytes: usize,
    /// For a unique column, computed when first asked for: `counts[k][s]`,
    /// how many strings of exactly `k` units end in a match from state `s`,
    /// saturating.
    counts: OnceLock<Result<Vec<Vec<u128>>, String>>,
}

impl PatternSampler {
    fn viable(&self, remaining: usize, state: u32) -> bool {
        self.viable[remaining][state as usize / 64] >> (state % 64) & 1 == 1
    }

    /// An upper bound on one value's bytes.
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// The counts behind `cardinality` and `decode`; an error when there
    /// would be too many to keep.
    pub fn counts(&self) -> Result<&Vec<Vec<u128>>, String> {
        self.counts
            .get_or_init(|| {
                let longest = *self.lengths.last().expect("at least one length") as usize;
                if self.groups.len().saturating_mul(longest + 1) > MAX_COUNTS {
                    return Err(
                        "it has too many states and lengths to number its values for \
                         unique=True"
                            .to_string(),
                    );
                }
                let mut counts: Vec<Vec<u128>> = Vec::with_capacity(longest + 1);
                counts.push(
                    (0..self.groups.len() as u32)
                        .map(|s| u128::from(self.viable(0, s)))
                        .collect(),
                );
                for k in 1..=longest {
                    let below = &counts[k - 1];
                    let layer = self
                        .groups
                        .iter()
                        .map(|groups| {
                            groups.iter().fold(0u128, |total, g| {
                                total.saturating_add(
                                    (g.units.len() as u128).saturating_mul(below[g.next as usize]),
                                )
                            })
                        })
                        .collect();
                    counts.push(layer);
                }
                Ok(counts)
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    /// Writes one value into `out`.
    /// `out` holds valid UTF-8 afterwards: every unit is a `str`.
    pub fn fill(&self, rng: &mut Xoshiro256PlusPlus, out: &mut Vec<u8>) {
        out.clear();
        let length = match self.lengths.len() {
            1 => self.lengths[0],
            n => self.lengths[rng.random_range(0..n as u32) as usize],
        } as usize;
        let write = |out: &mut Vec<u8>, unit: u16| match &self.ascii {
            Some(bytes) => out.push(bytes[unit as usize]),
            None => out.extend_from_slice(self.units[unit as usize].as_bytes()),
        };
        let mut state = self.start;
        for remaining in (0..length).rev() {
            let groups = &self.groups[state as usize];
            // A state the walk reaches has a group that can still finish in
            // `remaining` steps -- the step before chose it for that -- so a
            // state with one group needs no viability check: the draw is
            // the one the general path below makes.
            if let [group] = groups.as_slice() {
                let pick = rng.random_range(0..group.units.len() as u32);
                write(out, group.units[pick as usize]);
                state = group.next;
                continue;
            }
            let total: u32 = groups
                .iter()
                .filter(|g| self.viable(remaining, g.next))
                .map(|g| g.units.len() as u32)
                .sum();
            let mut pick = rng.random_range(0..total);
            for group in groups.iter().filter(|g| self.viable(remaining, g.next)) {
                let size = group.units.len() as u32;
                if pick < size {
                    write(out, group.units[pick as usize]);
                    state = group.next;
                    break;
                }
                pick -= size;
            }
        }
    }
}

impl Enumerable for PatternSampler {
    /// How many distinct values the pattern has at its lengths, saturating.
    /// Zero until `counts` has succeeded; a unique plan asks for it first.
    fn cardinality(&self) -> u128 {
        let Ok(counts) = self.counts() else {
            return 0;
        };
        self.lengths.iter().fold(0u128, |total, &k| {
            total.saturating_add(counts[k as usize][self.start as usize])
        })
    }

    /// The value at `index`: its length by the counts of the shorter ones,
    /// then each unit by the counts of what can follow it. Distinct indices
    /// below `cardinality()` give distinct values; a saturated count reads
    /// as the number it is, as a template's saturated radix does.
    fn decode(&self, mut index: u128, out: &mut Vec<u8>) {
        out.clear();
        let counts = self.counts().expect("counted before decoding");
        let mut length = *self.lengths.last().expect("at least one length") as usize;
        for &k in &self.lengths {
            let here = counts[k as usize][self.start as usize];
            if index < here {
                length = k as usize;
                break;
            }
            index -= here;
        }
        let mut state = self.start;
        for remaining in (0..length).rev() {
            let below = &counts[remaining];
            for group in &self.groups[state as usize] {
                let each = below[group.next as usize];
                if each == 0 {
                    continue;
                }
                let block = (group.units.len() as u128).saturating_mul(each);
                if index < block {
                    let unit = (index / each) as usize;
                    index %= each;
                    out.extend_from_slice(self.units[group.units[unit] as usize].as_bytes());
                    state = group.next;
                    break;
                }
                index -= block;
            }
        }
    }

    fn max_bytes(&self) -> usize {
        self.max_bytes
    }
}

fn dfa(pattern: &str, start_kind: StartKind) -> Result<dense::DFA<Vec<u32>>, String> {
    dense::Builder::new()
        .configure(
            dense::Config::new()
                .start_kind(start_kind)
                .match_kind(MatchKind::All)
                .dfa_size_limit(Some(DFA_SIZE_LIMIT))
                .determinize_size_limit(Some(DFA_SIZE_LIMIT)),
        )
        .build(pattern)
        .map_err(|err| describe_build_error(&err.to_string()))
}

fn describe_build_error(err: &str) -> String {
    if err.contains("word bound") {
        "it uses a Unicode word boundary, which the generator cannot follow \
         (an ASCII one, (?-u:\\b), it can)"
            .to_string()
    } else if err.contains("exceeded") || err.contains("limit") {
        "its automaton is too large to generate from".to_string()
    } else {
        format!("it does not compile: {err}")
    }
}

/// The non-ASCII characters a pattern names: its literals, and the start of
/// each range of its classes.
fn extra_units(hir: &Hir) -> Vec<Box<str>> {
    fn walk(hir: &Hir, out: &mut Vec<char>) {
        if out.len() >= MAX_EXTRA_UNITS {
            return;
        }
        match hir.kind() {
            HirKind::Literal(literal) => {
                if let Ok(text) = std::str::from_utf8(&literal.0) {
                    out.extend(text.chars());
                }
            }
            HirKind::Class(Class::Unicode(class)) => {
                for range in class.ranges() {
                    let (lo, hi) = (range.start() as u32, range.end() as u32);
                    let lo = lo.max(0x80);
                    if lo > hi {
                        continue;
                    }
                    out.extend((lo..=hi.min(lo + PER_RANGE - 1)).filter_map(char::from_u32));
                }
            }
            HirKind::Repetition(rep) => walk(&rep.sub, out),
            HirKind::Capture(cap) => walk(&cap.sub, out),
            HirKind::Concat(subs) | HirKind::Alternation(subs) => {
                for sub in subs {
                    walk(sub, out);
                }
            }
            HirKind::Empty | HirKind::Look(_) | HirKind::Class(Class::Bytes(_)) => {}
        }
    }
    let mut chars = Vec::new();
    walk(hir, &mut chars);
    let mut seen = HashSet::new();
    chars
        .into_iter()
        .filter(|c| !c.is_ascii() && seen.insert(*c))
        .take(MAX_EXTRA_UNITS)
        .map(|c| c.to_string().into_boxed_str())
        .collect()
}

fn ascii_units() -> Vec<Box<str>> {
    (0x20u8..=0x7E)
        .map(|b| (b as char).to_string().into_boxed_str())
        .collect()
}

/// The automaton over `units`: groups out of each state, and which states
/// accept at the end of the text.
type Compacted = (Vec<Vec<Group>>, Vec<bool>, u32);

fn compact(dfa: &dense::DFA<Vec<u32>>, units: &[Box<str>]) -> Result<Compacted, String> {
    let start_id = dfa
        .start_state(&start::Config::new().anchored(Anchored::Yes))
        .map_err(|err| format!("it has no start: {err}"))?;
    let mut index: HashMap<StateID, u32> = HashMap::from([(start_id, 0)]);
    let mut order = vec![start_id];
    let mut queue = VecDeque::from([start_id]);
    let mut groups: Vec<Vec<Group>> = Vec::new();
    let mut accept: Vec<bool> = Vec::new();
    while let Some(id) = queue.pop_front() {
        accept.push(dfa.is_match_state(dfa.next_eoi_state(id)));
        let mut by_next: Vec<Group> = Vec::new();
        for (u, unit) in units.iter().enumerate() {
            let mut next = id;
            for &byte in unit.as_bytes() {
                next = dfa.next_state(next, byte);
                if dfa.is_dead_state(next) || dfa.is_quit_state(next) {
                    break;
                }
            }
            if dfa.is_dead_state(next) || dfa.is_quit_state(next) {
                continue;
            }
            let next = match index.get(&next) {
                Some(&i) => i,
                None => {
                    if order.len() >= MAX_STATES {
                        return Err("its automaton is too large to generate from".into());
                    }
                    let i = order.len() as u32;
                    index.insert(next, i);
                    order.push(next);
                    queue.push_back(next);
                    i
                }
            };
            match by_next.iter_mut().find(|g| g.next == next) {
                Some(group) => group.units.push(u as u16),
                None => by_next.push(Group {
                    next,
                    units: vec![u as u16],
                }),
            }
        }
        groups.push(by_next);
    }
    Ok((groups, accept, 0))
}

/// The bitset of states that can end in a match in exactly one more unit
/// than the states `after` can.
fn step(groups: &[Vec<Group>], after: &[u64]) -> Vec<u64> {
    let mut out = vec![0u64; after.len()];
    for (state, out_groups) in groups.iter().enumerate() {
        if out_groups
            .iter()
            .any(|g| after[g.next as usize / 64] >> (g.next % 64) & 1 == 1)
        {
            out[state / 64] |= 1 << (state % 64);
        }
    }
    out
}

fn bitset(flags: &[bool]) -> Vec<u64> {
    let mut out = vec![0u64; flags.len().div_ceil(64)];
    for (i, &on) in flags.iter().enumerate() {
        if on {
            out[i / 64] |= 1 << (i % 64);
        }
    }
    out
}

fn has(bits: &[u64], state: u32) -> bool {
    bits[state as usize / 64] >> (state % 64) & 1 == 1
}

fn build_over(
    dfa: &dense::DFA<Vec<u32>>,
    units: Vec<Box<str>>,
    lengths: Option<(usize, usize)>,
    longest: Option<usize>,
) -> Result<PatternSampler, String> {
    let (groups, accept, start) = compact(dfa, &units)?;
    let mut viable = vec![bitset(&accept)];
    // The range of lengths: as declared, or from the shortest match to the
    // pattern's longest, or 16 past the shortest when it has no longest.
    let (lo, hi) = match lengths {
        Some((lo, hi)) => (lo, hi.min(MAX_LENGTH)),
        None => {
            let mut shortest = None;
            for k in 0..=MAX_LENGTH {
                if k > 0 {
                    let next = step(&groups, &viable[k - 1]);
                    viable.push(next);
                }
                if has(&viable[k], start) {
                    shortest = Some(k);
                    break;
                }
                if viable[k].iter().all(|&w| w == 0) {
                    break;
                }
            }
            let shortest = shortest.ok_or("it matches no string the generator can write")?;
            let hi = match longest {
                Some(m) => m.min(MAX_LENGTH),
                None => (shortest + UNBOUNDED_EXTRA).min(MAX_LENGTH),
            };
            (shortest, hi.max(shortest))
        }
    };
    while viable.len() <= hi {
        let next = step(&groups, viable.last().expect("at least one layer"));
        viable.push(next);
    }
    let fits: Vec<u32> = (lo..=hi)
        .filter(|&k| has(&viable[k], start))
        .map(|k| k as u32)
        .collect();
    if fits.is_empty() {
        return Err(format!(
            "no string it matches is {lo} to {hi} characters long"
        ));
    }
    let widest = units.iter().map(|u| u.len()).max().unwrap_or(1);
    let ascii = units
        .iter()
        .all(|u| u.len() == 1)
        .then(|| units.iter().map(|u| u.as_bytes()[0]).collect());
    Ok(PatternSampler {
        max_bytes: hi * widest,
        units,
        ascii,
        groups,
        viable,
        lengths: fits,
        start,
        counts: OnceLock::new(),
    })
}

/// Compiles `pattern` for generation, between `lengths` characters when
/// given.
fn build(pattern: &str, lengths: Option<(usize, usize)>) -> Result<PatternSampler, String> {
    if let Some((lo, hi)) = lengths
        && lo > hi
    {
        return Err(format!("its lengths {lo} to {hi} are empty"));
    }
    let hir = regex_syntax::ParserBuilder::new()
        .build()
        .parse(pattern)
        .map_err(|err| format!("it does not compile: {err}"))?;
    // An upper bound in bytes is one in characters too.
    let longest = hir.properties().maximum_len();
    let dfa = dfa(pattern, StartKind::Anchored)?;
    match build_over(&dfa, ascii_units(), lengths, longest) {
        Ok(sampler) => Ok(sampler),
        Err(ascii_err) => {
            let extra = extra_units(&hir);
            if extra.is_empty() {
                return Err(ascii_err);
            }
            let mut units = ascii_units();
            units.extend(extra);
            build_over(&dfa, units, lengths, longest)
        }
    }
}

type Key = (String, Option<(usize, usize)>);
type Compiled = Result<Arc<PatternSampler>, String>;

/// A pattern's sampler, compiled once per pattern and lengths.
pub fn sampler(
    pattern: &str,
    lengths: Option<(usize, usize)>,
) -> Result<Arc<PatternSampler>, String> {
    static CACHE: OnceLock<Mutex<HashMap<Key, Compiled>>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    let key = (pattern.to_string(), lengths);
    if let Some(found) = cache.lock().expect("pattern cache").get(&key) {
        return found.clone();
    }
    let built = build(pattern, lengths).map(Arc::new);
    cache
        .lock()
        .expect("pattern cache")
        .insert(key, built.clone());
    built
}

/// Whether every value the plain string generator draws -- `CHARSET` text,
/// `lo` to `hi` characters -- already passes `pattern` as validation reads
/// it, anywhere in the text. Such a pattern keeps that output exactly, since
/// a seed's output changes only where it broke its own spec. Decided on the
/// automaton, not by sampling.
pub fn plain_text_always_passes(pattern: &str, lo: usize, hi: usize) -> bool {
    let Ok(dfa) = dfa(pattern, StartKind::Unanchored) else {
        return false;
    };
    let Ok(start) = dfa.start_state(&start::Config::new().anchored(Anchored::No)) else {
        return false;
    };
    // The states of the strings of each length that have not matched yet.
    let mut layer: HashSet<StateID> = HashSet::from([start]);
    for k in 0..=hi {
        if k >= lo
            && layer
                .iter()
                .any(|&s| !dfa.is_match_state(dfa.next_eoi_state(s)))
        {
            return false;
        }
        if k == hi || layer.is_empty() {
            break;
        }
        let mut next_layer = HashSet::new();
        for &state in &layer {
            for &byte in CHARSET {
                let next = dfa.next_state(state, byte);
                if dfa.is_match_state(next) {
                    continue; // matched: every extension passes
                }
                if dfa.is_dead_state(next) || dfa.is_quit_state(next) {
                    return false; // can never match, and has a length in range
                }
                next_layer.insert(next);
            }
        }
        layer = next_layer;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use regex_automata::meta::Regex;

    fn draws(pattern: &str, lengths: Option<(usize, usize)>, n: usize) -> Vec<String> {
        let sampler = sampler(pattern, lengths).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        let mut out = Vec::new();
        (0..n)
            .map(|_| {
                sampler.fill(&mut rng, &mut out);
                String::from_utf8(out.clone()).expect("a pattern's value is UTF-8")
            })
            .collect()
    }

    #[test]
    fn every_value_matches_its_pattern() {
        for pattern in [
            r"^ORD-[0-9]{6}$",
            r"^[A-Z]{1,2}[0-9][A-Z0-9]? [0-9][A-Z]{2}$",
            r"(cat|dog)s?",
            r"^\w+@\w+\.(com|org)$",
            r"^[^aeiou]{3,5}$",
            r"^a+b*$",
            r"^(?i)hello$",
            r"^é+$",
        ] {
            let re = Regex::new(pattern).unwrap();
            for value in draws(pattern, None, 500) {
                assert!(re.is_match(&value), "{pattern} drew {value:?}");
            }
        }
    }

    #[test]
    fn lengths_are_characters_and_held() {
        for value in draws(r"^[a-z]+$", Some((3, 6)), 500) {
            assert!((3..=6).contains(&value.chars().count()), "{value:?}");
        }
        for value in draws(r"^é+$", Some((2, 2)), 50) {
            assert_eq!(value, "éé");
        }
    }

    #[test]
    fn an_unbounded_repeat_runs_sixteen_past_its_shortest() {
        let lens: HashSet<usize> = draws(r"^ab+$", None, 2_000)
            .iter()
            .map(|v| v.len())
            .collect();
        assert_eq!(*lens.iter().min().unwrap(), 2);
        assert_eq!(*lens.iter().max().unwrap(), 2 + UNBOUNDED_EXTRA);
    }

    #[test]
    fn ascii_wherever_the_pattern_allows_it() {
        for value in draws(r"^\w{4}$", None, 500) {
            assert!(value.is_ascii(), "{value:?}");
        }
    }

    #[test]
    fn out_of_reach_says_why() {
        assert!(
            sampler(r"\bword\b", None)
                .unwrap_err()
                .contains("word boundary")
        );
        assert!(
            sampler(r"^[0-9]{6}$", Some((2, 3)))
                .unwrap_err()
                .contains("2 to 3 characters")
        );
        assert!(
            sampler(r"(", None)
                .unwrap_err()
                .contains("does not compile")
        );
    }

    #[test]
    fn a_pattern_numbers_its_values_one_to_one() {
        let sampler = sampler(r"^[ab]{1,3}$", None).unwrap();
        assert_eq!(sampler.cardinality(), 2 + 4 + 8);
        let re = Regex::new(r"^[ab]{1,3}$").unwrap();
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for index in 0..sampler.cardinality() {
            sampler.decode(index, &mut out);
            let value = String::from_utf8(out.clone()).unwrap();
            assert!(re.is_match(&value), "{value:?}");
            assert!(seen.insert(value), "index {index} repeated a value");
        }
    }

    #[test]
    fn numbering_reads_alternations_and_non_ascii() {
        for pattern in [r"^(cat|dog)s?$", r"^é{1,2}x?$", r"^ORD-[0-9]{2}$"] {
            let sampler = sampler(pattern, None).unwrap();
            let re = Regex::new(pattern).unwrap();
            let mut seen = HashSet::new();
            let mut out = Vec::new();
            for index in 0..sampler.cardinality() {
                sampler.decode(index, &mut out);
                let value = String::from_utf8(out.clone()).unwrap();
                assert!(re.is_match(&value), "{pattern} decoded {value:?}");
                assert!(seen.insert(value));
            }
        }
        assert_eq!(sampler(r"^(cat|dog)s?$", None).unwrap().cardinality(), 4);
        assert_eq!(sampler(r"^ORD-[0-9]{2}$", None).unwrap().cardinality(), 100);
    }

    #[test]
    fn a_huge_space_saturates_and_still_decodes_distinctly() {
        let sampler = sampler(r"^.{30}$", None).unwrap();
        assert_eq!(sampler.cardinality(), u128::MAX);
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        for index in [0, 1, 2, u128::MAX / 2, u128::MAX - 1] {
            sampler.decode(index, &mut out);
            assert_eq!(out.len(), 30);
            assert!(seen.insert(out.clone()));
        }
    }

    #[test]
    fn plain_text_that_always_passes_is_recognised_exactly() {
        assert!(plain_text_always_passes(r"^[A-Za-z0-9]+$", 5, 15));
        assert!(plain_text_always_passes(r"[A-Za-z0-9]", 5, 15));
        assert!(plain_text_always_passes(r"^\w{5,15}$", 5, 15));
        assert!(!plain_text_always_passes(r"^[A-Za-z0-9]{5,14}$", 5, 15));
        assert!(!plain_text_always_passes(r"[a-z]", 5, 15));
        assert!(!plain_text_always_passes(r"^ORD-\d{6}$", 5, 15));
    }
}

#[cfg(test)]
mod timing {
    //! `cargo test --release --lib pattern::timing -- --ignored --nocapture`:
    //! a pattern row's cost on one thread.
    use super::*;
    use rand::SeedableRng;
    use std::hint::black_box;

    #[test]
    #[ignore]
    fn timing_pattern_parts() {
        let n = 1_000_000;
        for pattern in [r"^ORD-[0-9]{6}$", r"^[A-Z]{2}-\d{4,6}(-(EU|US|UK))?$"] {
            let s = sampler(pattern, None).unwrap();
            let time = |f: &mut dyn FnMut() -> u64| {
                black_box(f());
                (0..5)
                    .map(|_| {
                        let t = std::time::Instant::now();
                        black_box(f());
                        t.elapsed().as_secs_f64() * 1e9 / n as f64
                    })
                    .fold(f64::MAX, f64::min)
            };
            let mut out = Vec::new();
            let full = time(&mut || {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
                (0..n)
                    .map(|_| {
                        s.fill(&mut rng, &mut out);
                        out.len() as u64
                    })
                    .sum()
            });
            println!("{pattern:<36} fill {full:>6.2} ns/row");
        }
    }
}
