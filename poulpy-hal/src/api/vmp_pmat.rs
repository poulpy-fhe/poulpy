use crate::layouts::{
    Backend, MatZnxBackendRef, ScratchArena, VecZnxBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef, VecZnxDftToBackendMut,
    VmpPMatBackendMut, VmpPMatBackendRef, VmpPMatOwned,
};

/// Allocates a [`VmpPMat`](crate::layouts::VmpPMat).
pub trait VmpPMatAlloc<B: Backend> {
    fn vmp_pmat_alloc(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> VmpPMatOwned<B>;
}

/// Returns the byte size required for a [`VmpPMat`](crate::layouts::VmpPMat).
pub trait VmpPMatBytesOf {
    fn bytes_of_vmp_pmat(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize;
}

/// Returns scratch bytes required for [`VmpPrepare`].
pub trait VmpPrepareTmpBytes {
    fn vmp_prepare_tmp_bytes(&self, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize;
}

/// Prepares a coefficient-domain [`MatZnx`](crate::layouts::MatZnx) into a
/// DFT-domain [`VmpPMat`](crate::layouts::VmpPMat).
pub trait VmpPrepare<B: Backend> {
    fn vmp_prepare<'s>(
        &self,
        pmat: &mut VmpPMatBackendMut<'_, B>,
        mat: &MatZnxBackendRef<'_, B>,
        scratch: &mut ScratchArena<'s, B>,
    );
}

#[allow(clippy::too_many_arguments)]
/// Returns scratch bytes required for [`VmpApplyDft`].
pub trait VmpApplyDftTmpBytes {
    fn vmp_apply_dft_tmp_bytes(
        &self,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize;
}

/// Applies the vector-matrix product `VecZnx x VmpPMat -> VecZnxDft`.
pub trait VmpApplyDft<B: Backend> {
    fn vmp_apply_dft<'s, R>(
        &self,
        res: &mut R,
        a: &VecZnxBackendRef<'_, B>,
        pmat: &VmpPMatBackendRef<'_, B>,
        scratch: &mut ScratchArena<'s, B>,
    ) where
        R: VecZnxDftToBackendMut<B>;
}

#[allow(clippy::too_many_arguments)]
/// Returns scratch bytes required for [`VmpApplyDftToDft`].
pub trait VmpApplyDftToDftTmpBytes {
    fn vmp_apply_dft_to_dft_tmp_bytes(
        &self,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize;
}

#[allow(clippy::too_many_arguments)]
/// Returns scratch bytes required for [`VmpApplyDftToDftAccumulate`].
pub trait VmpApplyDftToDftAccumulateTmpBytes {
    fn vmp_apply_dft_to_dft_accumulate_tmp_bytes(
        &self,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize;
}

pub trait VmpApplyDftToDft<B: Backend> {
    /// Applies the vector matrix product [crate::layouts::VecZnxDft] x [crate::layouts::VmpPMat].
    ///
    /// A vector matrix product numerically equivalent to a sum of [crate::api::SvpApplyDft],
    /// where each [crate::layouts::SvpPPol] is a limb of the input [crate::layouts::VecZnx] in DFT,
    /// and each vector a [crate::layouts::VecZnxDft] (row) of the [crate::layouts::VmpPMat].
    ///
    /// As such, given an input [crate::layouts::VecZnx] of `i` size and a [crate::layouts::VmpPMat] of `i` rows and
    /// `j` size, the output is a [crate::layouts::VecZnx] of `j` size.
    ///
    /// If there is a mismatch between the dimensions the largest valid ones are used.
    ///
    /// ```text
    /// |a b c d| x |e f g| = (a * |e f g| + b * |h i j| + c * |k l m|) = |n o p|
    ///             |h i j|
    ///             |k l m|
    /// ```
    /// where each element is a [crate::layouts::VecZnxDft].
    ///
    /// # Arguments
    ///
    /// * `c`: the output of the vector matrix product, as a [crate::layouts::VecZnxDft].
    /// * `a`: the left operand [crate::layouts::VecZnxDft] of the vector matrix product.
    /// * `b`: the right operand [crate::layouts::VmpPMat] of the vector matrix product.
    /// * `buf`: scratch space, the size can be obtained with [VmpApplyDftToDftTmpBytes::vmp_apply_dft_to_dft_tmp_bytes].
    fn vmp_apply_dft_to_dft<'s, 'r>(
        &self,
        res: &mut VecZnxDftBackendMut<'r, B>,
        a: &VecZnxDftBackendRef<'_, B>,
        pmat: &VmpPMatBackendRef<'_, B>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'s, B>,
    );
}

pub trait VmpApplyDftToDftAccumulate<B: Backend> {
    /// Fused `res += a · pmat`, shifted by `limb_offset` limbs.
    fn vmp_apply_dft_to_dft_accumulate<'s, 'r>(
        &self,
        res: &mut VecZnxDftBackendMut<'r, B>,
        a: &VecZnxDftBackendRef<'_, B>,
        pmat: &VmpPMatBackendRef<'_, B>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'s, B>,
    );
}

/// Zeroes all entries of a [`VmpPMat`](crate::layouts::VmpPMat).
pub trait VmpZero<B: Backend> {
    fn vmp_zero(&self, res: &mut VmpPMatBackendMut<'_, B>);
}
