use crate::layouts::{
    Backend, Data, Scratch, VecZnxBig, VecZnxBigToMut, VecZnxDft, VecZnxDftOwned, VecZnxDftToMut, VecZnxDftToRef, VecZnxToRef,
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
    fn vec_znx_dft_apply<R, A>(&self, step: usize, offset: usize, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxToRef;
}

/// Returns scratch bytes required for [`VecZnxIdftApply`].
pub trait VecZnxIdftApplyTmpBytes {
    fn vec_znx_idft_apply_tmp_bytes(&self) -> usize;
}

/// Applies the inverse DFT, converting a [`VecZnxDft`](crate::layouts::VecZnxDft)
/// into a [`VecZnxBig`](crate::layouts::VecZnxBig) (extended precision).
pub trait VecZnxIdftApply<B: Backend> {
    fn vec_znx_idft_apply<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, scratch: &mut Scratch<B>)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxDftToRef<B>;
}

/// Inverse DFT using `a` as temporary storage (avoids extra scratch).
pub trait VecZnxIdftApplyTmpA<B: Backend> {
    fn vec_znx_idft_apply_tmpa<R, A>(&self, res: &mut R, res_col: usize, a: &mut A, a_col: usize)
    where
        R: VecZnxBigToMut<B>,
        A: VecZnxDftToMut<B>;
}

/// Inverse DFT consuming the input DFT vector and reinterpreting its
/// buffer as a [`VecZnxBig`](crate::layouts::VecZnxBig).
pub trait VecZnxIdftApplyConsume<B: Backend> {
    fn vec_znx_idft_apply_consume<D: Data>(&self, a: VecZnxDft<D, B>) -> VecZnxBig<D, B>
    where
        VecZnxDft<D, B>: VecZnxDftToMut<B>;
}

/// Element-wise addition of two [`VecZnxDft`](crate::layouts::VecZnxDft) vectors.
pub trait VecZnxDftAddInto<B: Backend> {
    fn vec_znx_dft_add_into<R, A, D>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &D, b_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
        D: VecZnxDftToRef<B>;
}

/// In-place addition in DFT domain: `res += a`.
pub trait VecZnxDftAddAssign<B: Backend> {
    fn vec_znx_dft_add_assign<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>;
}

/// In-place scaled addition in DFT domain: `res += a * a_scale`.
pub trait VecZnxDftAddScaledAssign<B: Backend> {
    fn vec_znx_dft_add_scaled_assign<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, a_scale: i64)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>;
}

/// Element-wise subtraction of two [`VecZnxDft`](crate::layouts::VecZnxDft) vectors.
pub trait VecZnxDftSub<B: Backend> {
    fn vec_znx_dft_sub<R, A, D>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize, b: &D, b_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>,
        D: VecZnxDftToRef<B>;
}

/// In-place subtraction in DFT domain: `res -= a`.
pub trait VecZnxDftSubAssign<B: Backend> {
    fn vec_znx_dft_sub_assign<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>;
}

/// In-place negated subtraction in DFT domain: `res = a - res`.
pub trait VecZnxDftSubNegateAssign<B: Backend> {
    fn vec_znx_dft_sub_negate_assign<R, A>(&self, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>;
}

/// Copies selected limbs from one [`VecZnxDft`](crate::layouts::VecZnxDft) to another.
///
/// The `step` and `offset` parameters select which limbs are copied.
pub trait VecZnxDftCopy<B: Backend> {
    fn vec_znx_dft_copy<R, A>(&self, step: usize, offset: usize, res: &mut R, res_col: usize, a: &A, a_col: usize)
    where
        R: VecZnxDftToMut<B>,
        A: VecZnxDftToRef<B>;
}

/// Zeroes all limbs of the selected column in DFT domain.
pub trait VecZnxDftZero<B: Backend> {
    fn vec_znx_dft_zero<R>(&self, res: &mut R, res_col: usize)
    where
        R: VecZnxDftToMut<B>;
}
