//! Turns the vendored rlwe-params table into the cells of `security`.
//!
//! The table is parsed here, so a malformed or mismatched table fails the build
//! and the library carries no JSON parser. See `third_party/rlwe-params/README.md`.

use std::{env, fmt::Write, fs, path::Path};

use serde_json::Value;

const TABLE_DIR: &str = "third_party/rlwe-params";
const TABLE_128: &str = "128bits_security.json";

fn main() {
    let table_path = Path::new(TABLE_DIR).join(TABLE_128);
    let version_path = Path::new(TABLE_DIR).join("VERSION");
    println!("cargo::rerun-if-changed={}", table_path.display());
    println!("cargo::rerun-if-changed={}", version_path.display());

    let version = fs::read_to_string(&version_path).unwrap_or_else(|err| panic!("{}: {err}", version_path.display()));
    let version = version.trim();
    assert!(!version.is_empty(), "{} is empty", version_path.display());
    let text = fs::read_to_string(&table_path).unwrap_or_else(|err| panic!("{}: {err}", table_path.display()));
    let table: Value = serde_json::from_str(&text).unwrap_or_else(|err| panic!("{}: {err}", table_path.display()));
    let cells = parse_table(&table, 128).unwrap_or_else(|err| panic!("{}: {err}", table_path.display()));

    let mut out = String::new();
    writeln!(out, "/// Release of rlwe-params the tables were copied from.").unwrap();
    writeln!(out, "const TABLE_VERSION: &str = {version:?};").unwrap();
    writeln!(
        out,
        "/// Cells of the 128-bit table: ring degree exponent, secret, largest `log2(q)`."
    )
    .unwrap();
    writeln!(out, "static CELLS_128: &[Cell] = &[").unwrap();
    for (log_n, distribution, max_log_q) in cells {
        writeln!(
            out,
            "    Cell {{ log_n: {log_n}, secret: TableSecret::{distribution}, max_log_q: {max_log_q:?} }},"
        )
        .unwrap();
    }
    writeln!(out, "];").unwrap();
    let dest = Path::new(&env::var("OUT_DIR").unwrap()).join("rlwe_params.rs");
    fs::write(&dest, out).unwrap_or_else(|err| panic!("{}: {err}", dest.display()));
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
