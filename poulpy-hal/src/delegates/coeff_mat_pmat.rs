use crate::{
    api::{
        CoeffMatApplyBig, CoeffMatApplyBigTmpBytes, CoeffMatPMatAlloc, CoeffMatPMatBytesOf, CoeffMatPrepare,
        CoeffMatPrepareTmpBytes,
    },
    layouts::{
        Backend, CoeffMatPMat, CoeffMatPMatBackendMut, CoeffMatPMatBackendRef, CoeffMatPMatOwned, Module, ScratchArena,
        VecZnxBackendRef, VecZnxBigBackendMut,
    },
    oep::HalCoeffMatImpl,
};

impl<B> CoeffMatPMatAlloc<B> for Module<B>
where
    B: Backend,
{
    fn coeff_mat_pmat_alloc(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> CoeffMatPMatOwned<B> {
        CoeffMatPMat::alloc(self.n(), rows, cols_in, cols_out, size)
    }
}

impl<B> CoeffMatPMatBytesOf for Module<B>
where
    B: Backend,
{
    fn bytes_of_coeff_mat_pmat(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        CoeffMatPMat::<B::OwnedBuf, B>::bytes_of(self.n(), rows, cols_in, cols_out, size)
    }
}

impl<B> CoeffMatPrepareTmpBytes for Module<B>
where
    B: Backend + HalCoeffMatImpl<B>,
{
    fn coeff_mat_prepare_tmp_bytes(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize {
        B::coeff_mat_prepare_tmp_bytes(self, rows, cols_in, cols_out, size)
    }
}

impl<B> CoeffMatPrepare<B> for Module<B>
where
    B: Backend + HalCoeffMatImpl<B>,
{
    fn coeff_mat_prepare(
        &self,
        res: &mut CoeffMatPMatBackendMut<'_, B>,
        matrix: &VecZnxBackendRef<'_, B>,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::coeff_mat_prepare(self, res, matrix, scratch)
    }
}

impl<B> CoeffMatApplyBigTmpBytes for Module<B>
where
    B: Backend + HalCoeffMatImpl<B>,
{
    fn coeff_mat_apply_big_tmp_bytes(&self, rows_in: usize, rows_out: usize) -> usize {
        B::coeff_mat_apply_big_tmp_bytes(self, rows_in, rows_out)
    }
}

impl<B> CoeffMatApplyBig<B> for Module<B>
where
    B: Backend + HalCoeffMatImpl<B>,
{
    fn coeff_mat_apply_big(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_limb: usize,
        pmat: &CoeffMatPMatBackendRef<'_, B>,
        pmat_limb: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        a_limb: usize,
        rows_in: usize,
        rows_out: usize,
        scratch: &mut ScratchArena<'_, B>,
    ) {
        B::coeff_mat_apply_big(
            self, res, res_limb, pmat, pmat_limb, a, a_col, a_limb, rows_in, rows_out, scratch,
        )
    }
}
