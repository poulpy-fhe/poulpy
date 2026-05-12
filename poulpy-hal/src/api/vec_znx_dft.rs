use crate::layouts::{
    Backend, ScratchArena, VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDftBackendMut, VecZnxDftBackendRef, VecZnxDftOwned,
};

/// Allocates a [`VecZnxDft`](crate::layouts::VecZnxDft).
pub trait VecZnxDftAlloc<B: Backend> {
    fn vec_znx_dft_alloc(&self, cols: usize, size: usize) -> VecZnxDftOwned<B>;
}

/// Wraps a byte buffer into a [`VecZnxDft`](crate::layouts::VecZnxDft).
pub trait VecZnxDftFromBytes<B: Backend> {
    fn vec_znx_dft_from_bytes(&self, cols: usize, size: usize, bytes: Vec<u8>) -> VecZnxDftOwned<B>;
}

/// Returns the byte size required for a [`VecZnxDft`](crate::layouts::VecZnxDft).
pub trait VecZnxDftBytesOf {
    fn bytes_of_vec_znx_dft(&self, cols: usize, size: usize) -> usize;
}

/// Applies the forward DFT to a coefficient-domain [`VecZnx`](crate::layouts::VecZnx),
/// storing the result in a [`VecZnxDft`](crate::layouts::VecZnxDft).
///
/// The `step` and `offset` parameters select which limbs of the input
/// are transformed: limbs `offset, offset + step, offset + 2*step, ...`.
pub trait VecZnxDftApply<B: Backend> {
    fn vec_znx_dft_apply<'a>(
        &self,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
    );
}

/// Returns scratch bytes required for [`VecZnxIdftApply`].
pub trait VecZnxIdftApplyTmpBytes {
    fn vec_znx_idft_apply_tmp_bytes(&self) -> usize;
}

/// Applies the inverse DFT, converting a [`VecZnxDft`](crate::layouts::VecZnxDft)
/// into a [`VecZnxBig`](crate::layouts::VecZnxBig) (extended precision).
pub trait VecZnxIdftApply<B: Backend> {
    fn vec_znx_idft_apply<'s>(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    );
}

/// Inverse DFT using `a` as temporary storage (avoids extra scratch).
pub trait VecZnxIdftApplyTmpA<B: Backend> {
    fn vec_znx_idft_apply_tmpa(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &mut VecZnxDftBackendMut<'_, B>,
        a_col: usize,
    );
}

/// Element-wise addition of two [`VecZnxDft`](crate::layouts::VecZnxDft) vectors.
pub trait VecZnxDftAddInto<B: Backend> {
    fn vec_znx_dft_add_into(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, B>,
        b_col: usize,
    );
}

/// In-place addition in DFT domain: `res += a`.
pub trait VecZnxDftAddAssign<B: Backend> {
    fn vec_znx_dft_add_assign(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, B>,
        a_col: usize,
    );
}

/// In-place scaled addition in DFT domain: `res += a * a_scale`.
pub trait VecZnxDftAddScaledAssign<B: Backend> {
    fn vec_znx_dft_add_scaled_assign(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, B>,
        a_col: usize,
        a_scale: i64,
    );
}

/// Element-wise subtraction of two [`VecZnxDft`](crate::layouts::VecZnxDft) vectors.
pub trait VecZnxDftSub<B: Backend> {
    fn vec_znx_dft_sub(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, B>,
        b_col: usize,
    );
}

/// In-place subtraction in DFT domain: `res -= a`.
pub trait VecZnxDftSubAssign<B: Backend> {
    fn vec_znx_dft_sub_assign(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, B>,
        a_col: usize,
    );
}

/// In-place negated subtraction in DFT domain: `res = a - res`.
pub trait VecZnxDftSubNegateAssign<B: Backend> {
    fn vec_znx_dft_sub_negate_assign(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, B>,
        a_col: usize,
    );
}

/// Copies selected limbs from one [`VecZnxDft`](crate::layouts::VecZnxDft) to another.
///
/// The `step` and `offset` parameters select which limbs are copied.
pub trait VecZnxDftCopy<B: Backend> {
    fn vec_znx_dft_copy(
        &self,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, B>,
        a_col: usize,
    );
}

/// Zeroes all limbs of the selected column in DFT domain.
pub trait VecZnxDftZero<B: Backend> {
    fn vec_znx_dft_zero(&self, res: &mut VecZnxDftBackendMut<'_, B>, res_col: usize);
}
