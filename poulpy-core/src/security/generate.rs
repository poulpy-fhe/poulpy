//! Rebuilds `rlwe_params_128.rs` from the vendored rlwe-params table, see `third_party/rlwe-params/README.md`.
//!
//! The test fails when the committed file differs from what the table gives, and rewrites it when
//! `POULPY_REGENERATE_SECURITY_TABLE` is set.

use std::{fmt::Write, fs};

use serde_json::Value;

const TABLE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/third_party/rlwe-params");
const GENERATED: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/security/rlwe_params_128.rs");
const REGENERATE: &str = "POULPY_REGENERATE_SECURITY_TABLE";

#[test]
fn the_committed_table_matches_the_vendored_one() {
    let expected = render().unwrap_or_else(|err| panic!("{TABLE_DIR}: {err}"));
    if std::env::var_os(REGENERATE).is_some() {
        fs::write(GENERATED, &expected).unwrap_or_else(|err| panic!("{GENERATED}: {err}"));
        return;
    }
    let committed = fs::read_to_string(GENERATED)
        .unwrap_or_else(|err| panic!("{GENERATED}: {err}"))
        .replace("\r\n", "\n");
    if let Some((line, (committed, expected))) = committed.lines().zip(expected.lines()).enumerate().find(|(_, (c, e))| c != e) {
        panic!(
            "{GENERATED}:{} is stale, `{committed}` where the table gives `{expected}`. Run `{REGENERATE}=1 cargo test -p poulpy-core --lib security::generate`.",
            line + 1
        );
    }
    assert_eq!(
        committed.lines().count(),
        expected.lines().count(),
        "{GENERATED} is stale. Run `{REGENERATE}=1 cargo test -p poulpy-core --lib security::generate`."
    );
}

/// The source of `rlwe_params_128.rs`.
fn render() -> Result<String, String> {
    let read = |name: &str| fs::read_to_string(format!("{TABLE_DIR}/{name}")).map_err(|err| format!("{name}: {err}"));
    let version = read("VERSION")?;
    let version = version.trim();
    if version.is_empty() {
        return Err("VERSION is empty".into());
    }
    let table: Value = serde_json::from_str(&read("128bits_security.json")?).map_err(|err| err.to_string())?;
    let cells = parse_table(&table, 128)?;

    let mut out = String::new();
    let mut line = |text: &str| writeln!(out, "{text}").unwrap();
    line("//! The 128-bit table of rlwe-params, as the cells [`super`] reads.");
    line("//!");
    line("//! Generated from `third_party/rlwe-params` by the test in `generate.rs`, not edited by hand.");
    line("");
    line("use super::{Cell, TableSecret};");
    line("");
    line("/// Release of rlwe-params the table was copied from.");
    line(&format!("pub(super) const TABLE_VERSION: &str = {version:?};"));
    line("");
    line("/// Cells of the 128-bit table: ring degree exponent, secret, largest `log2(q)`.");
    line("#[rustfmt::skip]");
    line("pub(super) static CELLS_128: &[Cell] = &[");
    for (log_n, distribution, max_log_q) in cells {
        line(&format!(
            "    Cell {{ log_n: {log_n}, secret: TableSecret::{distribution}, max_log_q: {max_log_q:?} }},"
        ));
    }
    line("];");
    Ok(out)
}

/// The cells of a table for `target` bits: `(log_n, distribution, max_log_q)`,
/// the secret distribution as a `TableSecret` expression. Cells of a distribution
/// the library does not sample are left out (see [`parse_distribution`]).
fn parse_table(table: &Value, target: u64) -> Result<Vec<(u64, String, Option<u64>)>, String> {
    let field = |name: &str| table.get(name).ok_or(format!("missing `{name}`"));
    if field("format")?.as_u64() != Some(1) {
        return Err(format!("unsupported format {}", field("format")?));
    }
    if field("target")?.as_u64() != Some(target) {
        return Err(format!("target {} where {target} was expected", field("target")?));
    }
    let error = field("error")?;
    if error.get("kind").and_then(Value::as_str) != Some("gaussian") || error.get("sigma").and_then(Value::as_str) != Some("3.2")
    {
        return Err(format!("error {error} is not the Gaussian of sigma 3.2 the library samples"));
    }
    if !field("samples")?.is_null() {
        return Err(format!(
            "{} samples where the table must hold for unlimited samples",
            field("samples")?
        ));
    }

    let mut cells: Vec<(u64, String, Option<u64>)> = Vec::new();
    for cell in field("cells")?.as_array().ok_or("`cells` is not an array")? {
        let log_n = cell
            .get("log_n")
            .and_then(Value::as_u64)
            .ok_or(format!("cell {cell} has no `log_n`"))?;
        let distribution = cell.get("secret").ok_or(format!("cell {cell} has no `secret`"))?;
        let Some(distribution) = parse_distribution(distribution)? else {
            continue;
        };
        // `max_log_q` is a bound only where the search ended on it: at the last
        // passing modulus, or at the upper end of the search. A cell without a
        // result leaves its secret unsupported at that degree.
        let max_log_q = match cell.get("status").and_then(Value::as_str) {
            Some("boundary" | "at_least_upper") => Some(
                cell.get("max_log_q")
                    .and_then(Value::as_u64)
                    .ok_or(format!("cell {cell} has a result but no `max_log_q`"))?,
            ),
            _ => None,
        };
        if cells.iter().any(|(n, d, _)| *n == log_n && *d == distribution) {
            return Err(format!("two cells for {distribution} at log_n {log_n}"));
        }
        cells.push((log_n, distribution, max_log_q));
    }
    Ok(cells)
}

/// The `TableSecret` of a cell's secret distribution, or `None` for one the
/// library does not sample (a Gaussian secret, a kind or a range it does not
/// know), which leaves it without a bound. An error for a malformed one.
fn parse_distribution(distribution: &Value) -> Result<Option<String>, String> {
    let int = |name: &str| {
        distribution
            .get(name)
            .and_then(Value::as_i64)
            .ok_or(format!("distribution {distribution} has no `{name}`"))
    };
    let parsed = match distribution.get("kind").and_then(Value::as_str) {
        Some("ternary") => ((int("min")?, int("max")?) == (-1, 1)).then(|| "Ternary".to_string()),
        Some("binary") => ((int("min")?, int("max")?) == (0, 1)).then(|| "Binary".to_string()),
        Some("sparse_ternary") => {
            let h = int("h")?;
            (h > 0 && int("ones")? + int("minus_ones")? == h).then(|| format!("SparseTernary({h})"))
        }
        Some("sparse_binary") => {
            let h = int("h")?;
            (h > 0 && int("ones")? == h && int("minus_ones")? == 0).then(|| format!("SparseBinary({h})"))
        }
        Some(_) => None,
        None => return Err(format!("distribution {distribution} has no `kind`")),
    };
    Ok(parsed)
}
