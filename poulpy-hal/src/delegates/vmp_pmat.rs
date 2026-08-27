use crate::{
    api::{
        VmpApplyDft, VmpApplyDftTmpBytes, VmpApplyDftToDft, VmpApplyDftToDftAccumulate, VmpApplyDftToDftAccumulateTmpBytes,
        VmpApplyDftToDftTmpBytes, VmpExtractSelectedRows, VmpPMatAlloc, VmpPMatBytesOf, VmpPrepare, VmpPrepareTmpBytes, VmpZero,
    },
    layouts::{
        Backend, MatZnxBackendRef, Module, ScratchArena, VecZnxBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef,
        VecZnxDftToBackendMut, VmpPMatBackendMut, VmpPMatBackendRef, VmpPMatOwned,
    },
    oep::HalVmpImpl,
};

macro_rules! impl_vmp_delegate {
    ($trait:ty, $($body:item)+) => {
        impl<B> $trait for Module<B>
        where
            B: Backend<ZnxWord = i64> + HalVmpImpl<B>,
        {
            $($body)+
        }
    };
}

impl<B: Backend> VmpPMatAlloc<B> for Module<B> {
    fn vmp_pmat_alloc(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> VmpPMatOwned<B> {
        VmpPMatOwned::<B>::alloc(self.n(), rows, cols_in, cols_out, size)
    }
}

impl<B: Backend> VmpPMatBytesOf for Module<B> {
    fn bytes_of_vmp_pmat(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        B::bytes_of_vmp_pmat(self.n(), rows, cols_in, cols_out, size)
    }
}

impl_vmp_delegate!(
    VmpPrepareTmpBytes,
    fn vmp_prepare_tmp_bytes(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        B::vmp_prepare_tmp_bytes(self, rows, cols_in, cols_out, size)
    }
);

impl_vmp_delegate!(
    VmpPrepare<B>,
    fn vmp_prepare(&self, res: &mut VmpPMatBackendMut<'_, B>, a: &MatZnxBackendRef<'_, B>, scratch: &mut ScratchArena<'_, B>) {
        B::vmp_prepare(self, res, a, scratch);
    }
);

impl_vmp_delegate!(
    VmpApplyDftTmpBytes,
    fn vmp_apply_dft_tmp_bytes(
        &self,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize {
        B::vmp_apply_dft_tmp_bytes(self, res_size, a_size, b_rows, b_cols_in, b_cols_out, b_size)
    }
);

impl_vmp_delegate!(
    VmpApplyDft<B>,
    fn vmp_apply_dft<R>(
        &self,
        res: &mut R,
        a: &VecZnxBackendRef<'_, B>,
        b: &VmpPMatBackendRef<'_, B>,
        scratch: &mut ScratchArena<'_, B>,
    ) where
        R: VecZnxDftToBackendMut<B>,
    {
        B::vmp_apply_dft(self, res, a, b, scratch)
    }
);

impl_vmp_delegate!(
    VmpApplyDftToDftTmpBytes,
    fn vmp_apply_dft_to_dft_tmp_bytes(
        &self,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize {
        B::vmp_apply_dft_to_dft_tmp_bytes(self, res_size, a_size, b_rows, b_cols_in, b_cols_out, b_size)
    }
);

impl_vmp_delegate!(
    VmpApplyDftToDft<B>,
    fn vmp_apply_dft_to_dft(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        a: &VecZnxDftBackendRef<'_, B>,
        b: &VmpPMatBackendRef<'_, B>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vmp_apply_dft_to_dft(self, res, a, b, limb_offset, scratch)
    }
);

impl_vmp_delegate!(
    VmpApplyDftToDftAccumulateTmpBytes,
    fn vmp_apply_dft_to_dft_accumulate_tmp_bytes(
        &self,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize {
        B::vmp_apply_dft_to_dft_accumulate_tmp_bytes(self, res_size, a_size, b_rows, b_cols_in, b_cols_out, b_size)
    }
);

impl_vmp_delegate!(
    VmpApplyDftToDftAccumulate<B>,
    fn vmp_apply_dft_to_dft_accumulate(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        a: &VecZnxDftBackendRef<'_, B>,
        b: &VmpPMatBackendRef<'_, B>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::vmp_apply_dft_to_dft_accumulate(self, res, a, b, limb_offset, scratch);
    }
);

impl_vmp_delegate!(
    VmpExtractSelectedRows<B>,
    fn vmp_extract_selected_rows(
        &self,
        res: &mut VmpPMatBackendMut<'_, B>,
        a: &VmpPMatBackendRef<'_, B>,
        first_row: usize,
        row_step: usize,
    ) {
        assert_extractable(res, a, first_row, row_step);
        B::vmp_extract_selected_rows(self, res, a, first_row, row_step);
    }
);

/// Rejects a selection the backend kernel must not be handed: mismatched
/// prepared shapes, a truncation that widens, or a last row that is outside
/// `a` or whose index overflows.
///
/// Enforced here rather than per backend so a kernel may index without bounds
/// checks in release, and so a new backend inherits the contract.
fn assert_extractable<B: Backend>(
    res: &VmpPMatBackendMut<'_, B>,
    a: &VmpPMatBackendRef<'_, B>,
    first_row: usize,
    row_step: usize,
) {
    assert!(row_step > 0, "row_step must be positive");
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

impl_vmp_delegate!(
    VmpZero<B>,
    fn vmp_zero(&self, res: &mut VmpPMatBackendMut<'_, B>) {
        B::vmp_zero(self, res);
    }
);
