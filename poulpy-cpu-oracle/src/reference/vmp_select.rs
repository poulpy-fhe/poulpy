use crate::layouts::{Backend, VmpPMatBackendMut, VmpPMatBackendRef};

/// Rejects a selection the kernel must not be handed: mismatched prepared
/// shapes or [`PrepareHint`](crate::layouts::PrepareHint)s, a truncation that widens, or a last row that
/// is outside `a` or whose index overflows. Extraction copies representation
/// bytes, so both matrices must name the same representation.
///
/// Every kernel entry point calls it first, so the kernel may index without
/// bounds checks in release.
pub fn assert_extractable<BE: Backend>(
    res: &VmpPMatBackendMut<'_, BE>,
    a: &VmpPMatBackendRef<'_, BE>,
    first_row: usize,
    row_step: usize,
) {
    assert!(row_step > 0, "row_step must be positive");
    assert_eq!(
        res.hint(),
        a.hint(),
        "vmp_extract_selected_rows: res and a must carry the same PrepareHint ({:?} != {:?})",
        res.hint(),
        a.hint()
    );
    assert_eq!(res.n(), a.n(), "res.n(): {} != a.n(): {}", res.n(), a.n());
    assert_eq!(
        res.cols_in(),
        a.cols_in(),
        "res.cols_in(): {} != a.cols_in(): {}",
        res.cols_in(),
        a.cols_in()
    );
    assert_eq!(
        res.cols_out(),
        a.cols_out(),
        "res.cols_out(): {} != a.cols_out(): {}",
        res.cols_out(),
        a.cols_out()
    );
    assert!(res.size() <= a.size(), "res.size(): {} > a.size(): {}", res.size(), a.size());
    let Some(rows) = res.rows().checked_sub(1) else {
        return;
    };
    let last_row: Option<usize> = rows.checked_mul(row_step).and_then(|o| o.checked_add(first_row));
    assert!(
        last_row.is_some_and(|last| last < a.rows()),
        "selected rows {first_row}..={:?} step {row_step} exceed a.rows(): {}",
        last_row,
        a.rows()
    );
}
