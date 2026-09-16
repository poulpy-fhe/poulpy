use crate::layouts::{
    Backend, ScratchArena, VecZnxBackendMut, VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDftBackendMut, VecZnxDftBackendRef,
    VecZnxDftOwned,
};

/// Allocates a [`VecZnxDft`](crate::layouts::VecZnxDft).
///
/// ```text
/// op         vec_znx_dft_alloc(cols, size)
/// class      support
/// mutation   none
/// domain     cols >= 1, size >= 1
/// ensures    returns an owned degree-N VecZnxDft with cols columns and size limbs; its contents are unspecified
/// test       none
/// ```
pub trait VecZnxDftAlloc<B: Backend> {
    fn vec_znx_dft_alloc(&self, cols: usize, size: usize) -> VecZnxDftOwned<B>;
}

/// Returns the byte size required for a [`VecZnxDft`](crate::layouts::VecZnxDft).
///
/// ```text
/// op         bytes_of_vec_znx_dft(cols, size)
/// class      support
/// mutation   none
/// domain     cols >= 1, size >= 1
/// ensures    returns the bytes required for a degree-N VecZnxDft with cols columns and size limbs
/// test       test_word_compat_dft_bytes
/// ```
pub trait VecZnxDftBytesOf {
    fn bytes_of_vec_znx_dft(&self, cols: usize, size: usize) -> usize;
}

/// Applies the forward DFT to a coefficient-domain [`VecZnx`](crate::layouts::VecZnx),
/// storing the result in a [`VecZnxDft`](crate::layouts::VecZnxDft).
///
/// The `step` and `offset` parameters select which limbs of the input
/// are transformed: limbs `offset, offset + step, offset + 2*step, ...`.
///
/// ```text
/// op         vec_znx_dft_apply(step, offset, res, res_col, a, a_col)
/// class      basis
/// mutation   out-of-place
/// definition idft(res)[res_col,j] = a[a_col,offset + j * step]; other columns of res are unchanged
/// domain     res: a VecZnxDft; a: a dense VecZnx of the module degree; step >= 1; offset + j * step fits usize for every 0 <= j < min(res.size(), ceil(a.size() / step))
/// ensures    the selected source limbs are transformed; result limbs whose source index is at least a.size() are zero
/// test       test_vec_znx_dft_apply, test_vec_znx_idft_apply, test_vec_znx_dft_step_zero_rejected
/// ```
pub trait VecZnxDftApply<B: Backend> {
    fn vec_znx_dft_apply(
        &self,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    );
}

/// Returns scratch bytes required for [`VecZnxIdftApply`].
///
/// ```text
/// op         vec_znx_idft_apply_tmp_bytes()
/// class      support
/// mutation   none
/// domain     -
/// ensures    returns the scratch bytes required by vec_znx_idft_apply, independent of the limb count
/// test       test_vec_znx_idft_apply
/// ```
pub trait VecZnxIdftApplyTmpBytes {
    fn vec_znx_idft_apply_tmp_bytes(&self) -> usize;
}

/// Applies the inverse DFT, converting a [`VecZnxDft`](crate::layouts::VecZnxDft)
/// into a [`VecZnxBig`](crate::layouts::VecZnxBig) (extended precision).
///
/// ```text
/// op         vec_znx_idft_apply(res, res_col, a, a_col, scratch)
/// class      basis
/// mutation   out-of-place
/// definition res[res_col,j] = idft(a)[a_col,j]; other columns of res are unchanged
/// domain     res: a dense VecZnxBig; a: a VecZnxDft of the same degree; a is read, not clobbered
/// requires   scratch >= vec_znx_idft_apply_tmp_bytes()
/// ensures    the selected column is inverse-transformed into big words; result limbs from a.size() onward are zero and a is unchanged
/// test       test_vec_znx_idft_apply, test_vec_znx_idft_apply_alloc
/// ```
pub trait VecZnxIdftApply<B: Backend> {
    fn vec_znx_idft_apply(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Inverse DFT using `a` as temporary storage (avoids extra scratch).
///
/// ```text
/// op         vec_znx_idft_apply_tmpa(res, res_col, a, a_col)
/// class      variant
/// mutation   out-of-place
/// definition res[res_col,j] = idft(old(a))[a_col,j]; a[a_col] is unspecified afterwards; columns of res other than res_col and columns of a other than a_col are unchanged
/// domain     res: a dense VecZnxBig; a: a VecZnxDft of the same degree, taken mutably
/// ensures    the pre-call selected source column is inverse-transformed into big words; result limbs from a.size() onward are zero
/// test       test_vec_znx_idft_apply_tmpa
/// ```
pub trait VecZnxIdftApplyTmpA<B: Backend> {
    fn vec_znx_idft_apply_tmpa(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &mut VecZnxDftBackendMut<'_, B>,
        a_col: usize,
    );
}

/// Returns scratch bytes required for [`VecZnxIdftNormalizeConsume`].
///
/// ```text
/// op         vec_znx_idft_normalize_consume_tmp_bytes(res_size, a_size)
/// class      support
/// mutation   none
/// domain     res_size, a_size: the destination's and the source's limb counts
/// ensures    returns the scratch bytes required by vec_znx_idft_normalize_consume for these limb counts
/// test       test_vec_znx_idft_normalize_consume
/// ```
pub trait VecZnxIdftNormalizeConsumeTmpBytes {
    fn vec_znx_idft_normalize_consume_tmp_bytes(&self, res_size: usize, a_size: usize) -> usize;
}

/// Inverse DFT fused with normalization at precision `res_k`, consuming `a[a_col]`.
/// The optional addend contributes only its first `a.size()` limbs.
///
/// ```text
/// op         vec_znx_idft_normalize_consume(res, res_base2k, res_k, res_col, a, a_col, a_base2k, addend, scratch)
/// class      derived
/// mutation   out-of-place
/// definition res[res_col] = canon(sum_{0 <= t < a.size()} (idft(old(a))[a_col,t] + addend_limb(addend,t)) * 2^(-a_base2k * (t + 1)), res_base2k, res_k, res.size()); a[a_col] is unspecified afterwards; columns of res other than res_col and columns of a other than a_col are unchanged
/// domain     res: a VecZnx; a: a VecZnxDft taken mutably; addend: an optional VecZnx column read at a_base2k; all operands have the module degree; res_k <= res.size() * res_base2k; normalization input and radix bounds apply
/// requires   scratch >= vec_znx_idft_normalize_consume_tmp_bytes(res.size(), a.size())
/// ensures    the inverse-transformed source and the first a.size() limbs of the optional addend are read at a_base2k, rounded once at precision res_k, and represented canonically at res_base2k modulo 1
/// fallback   inverse-transform into an a.size()-limb VecZnxBig, add the optional selected column, then normalize
/// override   allowed, with vec_znx_idft_normalize_consume_tmp_bytes
/// test       test_vec_znx_idft_normalize_consume, test_vec_znx_idft_normalize_consume_derived
/// ```
pub trait VecZnxIdftNormalizeConsume<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_idft_normalize_consume(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_base2k: usize,
        res_k: usize,
        res_col: usize,
        a: &mut VecZnxDftBackendMut<'_, B>,
        a_col: usize,
        a_base2k: usize,
        addend: Option<(&VecZnxBackendRef<'_, B>, usize)>,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Element-wise addition of two [`VecZnxDft`](crate::layouts::VecZnxDft) vectors.
///
/// ```text
/// op         vec_znx_dft_add(res, res_col, a, a_col, b, b_col)
/// class      basis
/// mutation   out-of-place
/// definition idft(res)[res_col,j] = idft(a)[a_col,j] + idft(b)[b_col,j]; other columns of res are unchanged
/// domain     res, a, b: VecZnxDft of the same degree
/// ensures    the selected columns are added limb by limb with zero extension and destination truncation
/// test       test_vec_znx_dft_add
/// ```
pub trait VecZnxDftAdd<B: Backend> {
    fn vec_znx_dft_add(
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
///
/// ```text
/// op         vec_znx_dft_add_assign(res, res_col, a, a_col)
/// class      variant
/// mutation   in-place
/// definition idft(res)[res_col,j] = idft(old(res))[res_col,j] + idft(a)[a_col,j]; result limbs from a.size() onward and other columns of res are unchanged
/// domain     res, a: VecZnxDft of the same degree
/// ensures    the selected result column gains the selected source column limb by limb
/// test       test_vec_znx_dft_add_assign
/// ```
pub trait VecZnxDftAddAssign<B: Backend> {
    fn vec_znx_dft_add_assign(
        &self,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, B>,
        a_col: usize,
    );
}

/// Element-wise subtraction of two [`VecZnxDft`](crate::layouts::VecZnxDft) vectors.
///
/// ```text
/// op         vec_znx_dft_sub(res, res_col, a, a_col, b, b_col)
/// class      basis
/// mutation   out-of-place
/// definition idft(res)[res_col,j] = idft(a)[a_col,j] - idft(b)[b_col,j]; other columns of res are unchanged
/// domain     res, a, b: VecZnxDft of the same degree
/// ensures    the selected columns are subtracted limb by limb with zero extension and destination truncation
/// test       test_vec_znx_dft_sub
/// ```
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
///
/// ```text
/// op         vec_znx_dft_sub_assign(res, res_col, a, a_col)
/// class      variant
/// mutation   in-place
/// definition idft(res)[res_col,j] = idft(old(res))[res_col,j] - idft(a)[a_col,j]; result limbs from a.size() onward and other columns of res are unchanged
/// domain     res, a: VecZnxDft of the same degree
/// ensures    the selected result column loses the selected source column limb by limb
/// test       test_vec_znx_dft_sub_assign
/// ```
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
///
/// ```text
/// op         vec_znx_dft_sub_negate_assign(res, res_col, a, a_col)
/// class      variant
/// mutation   in-place
/// definition idft(res)[res_col,j] = idft(a)[a_col,j] - idft(old(res))[res_col,j]; other columns of res are unchanged
/// domain     res, a: VecZnxDft of the same degree
/// ensures    the pre-call selected result column is subtracted from the selected source column; result limbs from a.size() onward are negated
/// test       test_vec_znx_dft_sub_negate_assign
/// ```
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
///
/// ```text
/// op         vec_znx_dft_copy(step, offset, res, res_col, a, a_col)
/// class      basis
/// mutation   out-of-place
/// definition idft(res)[res_col,j] = idft(a)[a_col,offset + j * step]; other columns of res are unchanged
/// domain     res, a: VecZnxDft of the same degree; step >= 1; offset + j * step fits usize for every 0 <= j < min(res.size(), ceil(a.size() / step))
/// ensures    the selected source limbs are copied; result limbs whose source index is at least a.size() are zero
/// test       test_vec_znx_dft_copy, test_vec_znx_dft_step_zero_rejected
/// ```
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
///
/// ```text
/// op         vec_znx_dft_zero(res, res_col)
/// class      basis
/// mutation   out-of-place
/// definition idft(res)[res_col,j] = 0; other columns of res are unchanged
/// domain     res: a VecZnxDft
/// ensures    every limb of the selected result column reads as zero
/// test       test_vec_znx_dft_zero
/// ```
pub trait VecZnxDftZero<B: Backend> {
    fn vec_znx_dft_zero(&self, res: &mut VecZnxDftBackendMut<'_, B>, res_col: usize);
}

/// Builds a backend-specific permutation plan that implements the DFT-domain
/// automorphism `tau_p: X -> X^p` for odd `p`. The plan captures the
/// slot permutation, plus any implementation bookkeeping, and is reusable
/// across columns and limbs.
///
/// The associated `Plan` type is the only point in the public API where
/// the backend leaks its automorphism representation. Callers that want to
/// keep plans backend-agnostic must own
/// `<Module<B> as VecZnxDftAutomorphismPlan<B>>::Plan`.
///
/// ```text
/// op         vec_znx_dft_automorphism_plan(p)
/// class      support
/// mutation   none
/// domain     p odd
/// ensures    returns a reusable plan for the substitution X -> X^p in R_N
/// test       test_vec_znx_dft_automorphism
/// ```
pub trait VecZnxDftAutomorphismPlan<B: Backend> {
    type Plan;

    fn vec_znx_dft_automorphism_plan(&self, p: i64) -> Self::Plan;
}

/// Returns scratch bytes required by
/// [`VecZnxDftAutomorphism::vec_znx_dft_automorphism_add_with_plan`].
///
/// ```text
/// op         vec_znx_dft_automorphism_add_with_plan_tmp_bytes(res_size, a_size)
/// class      support
/// mutation   none
/// domain     res_size, a_size: the destination's and the source's limb counts
/// ensures    returns the scratch bytes required by vec_znx_dft_automorphism_add_with_plan for these limb counts
/// test       test_vec_znx_dft_automorphism_add
/// ```
pub trait VecZnxDftAutomorphismAddWithPlanTmpBytes {
    fn vec_znx_dft_automorphism_add_with_plan_tmp_bytes(&self, res_size: usize, a_size: usize) -> usize;
}

/// Applies a precomputed DFT-domain automorphism plan to `a`, writing the
/// result into `res` (out-of-place).
///
/// ```text
/// op         vec_znx_dft_automorphism_with_plan(plan, res, res_col, a, a_col)
/// class      basis
/// mutation   out-of-place
/// definition idft(res)[res_col,j] = sum_{0 <= i < N} idft(a)[a_col,j,i] * X^(i * plan.p) in R_N; other columns of res are unchanged
/// domain     res, a: VecZnxDft of the module degree; plan: built by vec_znx_dft_automorphism_plan for this degree and an odd exponent
/// ensures    the planned substitution is applied to the selected source column limb by limb; result limbs from a.size() onward are zero
/// test       test_vec_znx_dft_automorphism
/// ```
pub trait VecZnxDftAutomorphism<B: Backend>: VecZnxDftAutomorphismPlan<B> {
    fn vec_znx_dft_automorphism_with_plan(
        &self,
        plan: &Self::Plan,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, B>,
        a_col: usize,
    );

    /// `res[res_col] += automorphism(a[a_col])` over `min(res.size(), a.size())` limbs;
    /// res limbs beyond that are left untouched.
    ///
    /// `scratch` must hold at least
    /// [`vec_znx_dft_automorphism_add_with_plan_tmp_bytes`](VecZnxDftAutomorphismAddWithPlanTmpBytes::vec_znx_dft_automorphism_add_with_plan_tmp_bytes)
    /// on the same sizes: the derived body carves one
    /// `min(res.size(), a.size())`-limb, one-column `VecZnxDft` for the
    /// rotated operand.
    /// ```text
    /// op         vec_znx_dft_automorphism_add_with_plan(plan, res, res_col, a, a_col, scratch)
    /// class      derived
    /// mutation   accumulate
    /// definition idft(res)[res_col,j] = idft(old(res))[res_col,j] + sum_{0 <= i < N} idft(a)[a_col,j,i] * X^(i * plan.p) in R_N; result limbs from a.size() onward and other columns of res are unchanged
    /// domain     res, a: VecZnxDft of the module degree; plan: built by vec_znx_dft_automorphism_plan for this degree and an odd exponent
    /// requires   scratch >= vec_znx_dft_automorphism_add_with_plan_tmp_bytes(res.size(), a.size())
    /// ensures    the selected result column gains the planned substitution of the selected source column limb by limb
    /// fallback   apply the plan into a one-column VecZnxDft with min(res.size(), a.size()) limbs, then add its column 0 into res[res_col]
    /// override   allowed, with vec_znx_dft_automorphism_add_with_plan_tmp_bytes
    /// test       test_vec_znx_dft_automorphism_add, test_vec_znx_dft_automorphism_add_with_plan_derived
    /// ```
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_dft_automorphism_add_with_plan(
        &self,
        plan: &Self::Plan,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );

    /// Convenience: build the plan and apply in one call. Prefer
    /// [`vec_znx_dft_automorphism_with_plan`](Self::vec_znx_dft_automorphism_with_plan)
    /// when the same `p` is used repeatedly.
    ///
    /// ```text
    /// op         vec_znx_dft_automorphism(p, res, res_col, a, a_col)
    /// class      derived
    /// mutation   out-of-place
    /// definition idft(res)[res_col,j] = sum_{0 <= i < N} idft(a)[a_col,j,i] * X^(i * p) in R_N; other columns of res are unchanged
    /// domain     res, a: VecZnxDft of the module degree; p odd
    /// ensures    the substitution X -> X^p is applied to the selected source column limb by limb; result limbs from a.size() onward are zero
    /// fallback   build the plan for p, apply it, then drop it
    /// override   allowed, scratch-free
    /// test       test_vec_znx_dft_automorphism, test_vec_znx_dft_automorphism_derived
    /// ```
    fn vec_znx_dft_automorphism(
        &self,
        p: i64,
        res: &mut VecZnxDftBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, B>,
        a_col: usize,
    );
}
