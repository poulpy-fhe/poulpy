//! Enforcement test for the operation contracts of [`poulpy_hal::api`].
//!
//! Every public trait of `src/api` documents its operations with a structured
//! block, fenced as `text`, whose lines are `key` followed by a value:
//!
//! ```text
//! op         vec_znx_add(res, res_col, a, a_col, b, b_col)
//! class      basis
//! mutation   out-of-place
//! definition res[res_col,j] = a[a_col,j] + b[b_col,j]; other columns of res are unchanged
//! domain     res, a, b: VecZnx read at one shared base2k
//! ensures    the selected output column is the limbwise sum
//! test       test_vec_znx_add_matches_reference
//! ```
//!
//! This test reads the sources and checks that the blocks are there, that they
//! carry the lines their class requires, and that every test they name exists
//! in `src/test_suite`. It is the mechanical half of the specification; the
//! vocabulary the values use is defined in the `api` module documentation.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

/// Keys every contract block must carry.
const REQUIRED: [&str; 6] = ["op", "class", "mutation", "domain", "ensures", "test"];
/// Keys a block may carry in addition to [`REQUIRED`].
const OPTIONAL: [&str; 5] = ["definition", "requires", "fallback", "override", "sparse"];
/// Accepted values of the `class` line.
const CLASSES: [&str; 4] = ["basis", "variant", "derived", "support"];
/// Accepted values of the `mutation` line.
const MUTATIONS: [&str; 4] = ["out-of-place", "in-place", "accumulate", "none"];

/// One parsed contract block, with the source location it was found at.
struct Contract {
    file: String,
    line: usize,
    /// `(key, value)` in source order; a key never repeats within a block.
    fields: Vec<(String, String)>,
}

impl Contract {
    fn get(&self, key: &str) -> Option<&str> {
        self.fields.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }

    fn at(&self) -> String {
        format!("{}:{}", self.file, self.line)
    }
}

fn api_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/api")
}

fn test_suite_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/test_suite")
}

fn rust_files(dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    for entry in fs::read_dir(dir).expect("readable directory") {
        let path: PathBuf = entry.expect("readable entry").path();
        if path.is_dir() {
            out.extend(rust_files(&path));
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
    out.sort();
    out
}

/// Strips the `///` or `//!` marker from a documentation line, or returns
/// `None` when the line is not documentation.
fn doc_body(line: &str) -> Option<&str> {
    let trimmed: &str = line.trim_start();
    for marker in ["///", "//!"] {
        if let Some(rest) = trimmed.strip_prefix(marker) {
            return Some(rest.strip_prefix(' ').unwrap_or(rest));
        }
    }
    None
}

/// Splits a contract line into its key and value, accepting continuation lines
/// (indented, no key) by returning `None`.
fn split_field(body: &str) -> Option<(String, String)> {
    if body.starts_with(' ') {
        return None;
    }
    let mut parts = body.splitn(2, char::is_whitespace);
    let key: &str = parts.next()?;
    let value: &str = parts.next()?.trim();
    Some((key.to_string(), value.to_string()))
}

/// Parses every contract block of one file, keyed by the trait it belongs to.
///
/// A block found in the documentation directly above a `pub trait` line belongs
/// to that trait; any other block belongs to the trait whose body encloses it.
fn contracts_of(path: &Path) -> Vec<(String, Contract)> {
    let text: String = fs::read_to_string(path).expect("readable source file");
    let file: String = path.file_name().expect("named file").to_string_lossy().into_owned();
    let lines: Vec<&str> = text.lines().collect();

    let trait_lines: Vec<(usize, String)> = lines
        .iter()
        .enumerate()
        .filter_map(|(i, line)| {
            let name: &str = line.strip_prefix("pub trait ")?;
            let name: &str = name.split(|c: char| !(c.is_alphanumeric() || c == '_')).next()?;
            Some((i, name.to_string()))
        })
        .collect();

    let mut out: Vec<(String, Contract)> = Vec::new();
    let mut i: usize = 0;
    while i < lines.len() {
        // A block opens on a documentation line that is exactly the text fence
        // and whose next documentation line starts with `op`.
        let opens: bool = doc_body(lines[i]).is_some_and(|b| b.trim() == "```text")
            && doc_body(lines.get(i + 1).copied().unwrap_or("")).is_some_and(|b| b.starts_with("op "));
        if !opens {
            i += 1;
            continue;
        }
        let start: usize = i;
        let mut fields: Vec<(String, String)> = Vec::new();
        i += 1;
        while let Some(body) = doc_body(lines[i]) {
            if body.trim() == "```" {
                break;
            }
            if let Some((key, value)) = split_field(body) {
                fields.push((key, value));
            }
            i += 1;
        }
        let end: usize = i;

        // Attach to the trait below when only documentation, attributes and
        // blank lines separate the two; otherwise to the enclosing trait.
        let next: Option<&(usize, String)> = trait_lines.iter().find(|(l, _)| *l > end);
        let attaches_below: bool = next.is_some_and(|(l, _)| {
            lines[end + 1..*l].iter().all(|line| {
                let t: &str = line.trim();
                t.is_empty() || t.starts_with("///") || t.starts_with("#[")
            })
        });
        let owner: Option<&String> = if attaches_below {
            next.map(|(_, name)| name)
        } else {
            trait_lines.iter().rev().find(|(l, _)| *l < start).map(|(_, name)| name)
        };
        if let Some(owner) = owner {
            out.push((
                owner.clone(),
                Contract {
                    file: file.clone(),
                    line: start + 1,
                    fields,
                },
            ));
        }
        i += 1;
    }
    out
}

/// Every `pub fn` of `src/test_suite`, the set the `test` lines draw from.
fn test_suite_functions() -> BTreeSet<String> {
    let mut out: BTreeSet<String> = BTreeSet::new();
    for path in rust_files(&test_suite_dir()) {
        let text: String = fs::read_to_string(&path).expect("readable source file");
        for line in text.lines() {
            let Some(rest) = line.trim_start().strip_prefix("pub fn ") else {
                continue;
            };
            let name: &str = rest
                .split(|c: char| !(c.is_alphanumeric() || c == '_'))
                .next()
                .unwrap_or_default();
            if !name.is_empty() {
                out.insert(name.to_string());
            }
        }
    }
    out
}

#[test]
fn every_api_trait_carries_a_contract() {
    let suite: BTreeSet<String> = test_suite_functions();
    assert!(
        suite.len() > 50,
        "the test suite scan found only {} functions, the parser is broken",
        suite.len()
    );

    let mut errors: Vec<String> = Vec::new();
    for path in rust_files(&api_dir()) {
        let text: String = fs::read_to_string(&path).expect("readable source file");
        let file: String = path.file_name().expect("named file").to_string_lossy().into_owned();
        let contracts: Vec<(String, Contract)> = contracts_of(&path);

        for line in text.lines() {
            let Some(rest) = line.strip_prefix("pub trait ") else {
                continue;
            };
            let name: &str = rest
                .split(|c: char| !(c.is_alphanumeric() || c == '_'))
                .next()
                .unwrap_or_default();
            if !contracts.iter().any(|(owner, _)| owner == name) {
                errors.push(format!("{file}: `pub trait {name}` has no contract block"));
            }
        }

        for (owner, contract) in &contracts {
            check(owner, contract, &suite, &mut errors);
        }
    }

    assert!(
        errors.is_empty(),
        "{} contract violations:\n{}",
        errors.len(),
        errors.join("\n")
    );
}

fn check(owner: &str, contract: &Contract, suite: &BTreeSet<String>, errors: &mut Vec<String>) {
    let at: String = contract.at();
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for (key, value) in &contract.fields {
        if value.trim().is_empty() {
            errors.push(format!("{at}: {owner}: empty contract line `{key}`"));
        }
        if !REQUIRED.contains(&key.as_str()) && !OPTIONAL.contains(&key.as_str()) {
            errors.push(format!("{at}: {owner}: unknown contract line `{key}`"));
        }
        if !seen.insert(key.as_str()) {
            errors.push(format!("{at}: {owner}: duplicate contract line `{key}`"));
        }
    }
    for key in REQUIRED {
        if !seen.contains(key) {
            errors.push(format!("{at}: {owner}: missing contract line `{key}`"));
        }
    }

    let class: &str = contract.get("class").unwrap_or_default();
    if !CLASSES.contains(&class) {
        errors.push(format!("{at}: {owner}: class `{class}` is none of {}", CLASSES.join(", ")));
    }
    let mutation: &str = contract.get("mutation").unwrap_or_default();
    if !MUTATIONS.contains(&mutation) {
        errors.push(format!(
            "{at}: {owner}: mutation `{mutation}` is none of {}",
            MUTATIONS.join(", ")
        ));
    }

    // Every computing operation defines its result; support operations do not.
    // Duplicate keys are rejected above, so presence also implies uniqueness.
    let takes_scratch: bool = contract.get("requires").is_some_and(|r| r.contains("_tmp_bytes"));
    if matches!(class, "basis" | "derived" | "variant") && !seen.contains("definition") {
        errors.push(format!("{at}: {owner}: class `{class}` needs a `definition` line"));
    }
    if class == "support" && seen.contains("definition") {
        errors.push(format!("{at}: {owner}: support class must not have a `definition` line"));
    }
    if class == "derived" || (class == "variant" && takes_scratch) {
        for key in ["fallback", "override"] {
            if !seen.contains(key) {
                errors.push(format!("{at}: {owner}: class `{class}` needs a `{key}` line"));
            }
        }
    }

    for name in contract
        .get("test")
        .unwrap_or_default()
        .split([',', ' '])
        .filter(|n| !n.is_empty())
    {
        if name == "none" {
            if class != "support" {
                errors.push(format!("{at}: {owner}: only a support trait may leave `test` at `none`"));
            }
            continue;
        }
        if !suite.contains(name) {
            errors.push(format!("{at}: {owner}: `test {name}` is not a `pub fn` of src/test_suite"));
        }
    }
}

#[test]
fn computing_and_support_definition_rules() {
    let suite = BTreeSet::from(["test_operation".to_string()]);
    for class in CLASSES {
        for definitions in [vec![], vec!["res = 0"], vec!["res = 0", "res = 1"], vec![""]] {
            let mut contract = Contract {
                file: "example.rs".into(),
                line: 1,
                fields: vec![
                    ("op".into(), "operation(res)".into()),
                    ("class".into(), class.into()),
                    (
                        "mutation".into(),
                        if class == "support" { "none" } else { "out-of-place" }.into(),
                    ),
                    ("domain".into(), "res: an integer".into()),
                    ("ensures".into(), "res is zero".into()),
                    ("test".into(), "test_operation".into()),
                ],
            };
            if class == "derived" {
                contract
                    .fields
                    .extend([("fallback".into(), "zero(res)".into()), ("override".into(), "allowed".into())]);
            }
            for definition in &definitions {
                contract.fields.push(("definition".into(), (*definition).into()));
            }
            let mut errors = Vec::new();
            check("Example", &contract, &suite, &mut errors);
            let valid = if class == "support" {
                definitions.is_empty()
            } else {
                definitions == ["res = 0"]
            };
            assert_eq!(
                errors.is_empty(),
                valid,
                "class={class}, definitions={definitions:?}: {errors:?}"
            );
        }
    }
}
