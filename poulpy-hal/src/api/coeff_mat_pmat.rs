use crate::layouts::{
    Backend, CoeffMatPMatBackendMut, CoeffMatPMatBackendRef, CoeffMatPMatOwned, ScratchArena, VecZnxBackendRef,
    VecZnxBigBackendMut,
};

pub trait CoeffMatPMatAlloc<B: Backend> {
    fn coeff_mat_pmat_alloc(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> CoeffMatPMatOwned<B>;
}

pub trait CoeffMatPMatBytesOf {
    fn bytes_of_coeff_mat_pmat(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize;
}

pub trait CoeffMatPrepareTmpBytes {
    fn coeff_mat_prepare_tmp_bytes(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize;
}

pub trait CoeffMatPrepare<B: Backend> {
    fn coeff_mat_prepare(
        &self,
        res: &mut CoeffMatPMatBackendMut<'_, B>,
        matrix: &VecZnxBackendRef<'_, B>,
        scratch: &mut ScratchArena<'_, B>,
    );
}

pub trait CoeffMatApplyBigTmpBytes {
    fn coeff_mat_apply_big_tmp_bytes(&self, rows_in: usize, rows_out: usize) -> usize;
}

#[allow(clippy::too_many_arguments)]
pub trait CoeffMatApplyBig<B: Backend> {
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
    );
}
