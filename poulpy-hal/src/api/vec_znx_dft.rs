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
/// ensures    returns an owned degree-N VecZnxDft of those dimensions in the backend's representation, which is opaque; its contents are unspecified
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
/// ensures    returns the byte size of such a VecZnxDft in this backend's representation, the amount take_vec_znx_dft_scratch carves
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
/// definition idft(res[res_col])[j] = a[a_col][offset + j * step]
/// domain     res: a VecZnxDft; a: a dense VecZnx of the module degree; step >= 1
/// ensures    limb j of res[res_col] is the forward transform of limb offset + j * step of a[a_col], for as many limbs as res holds; limbs whose source is past a.size() are zero. The representation is opaque, so the statement is on the observable: idft(res) reproduces those limbs of a
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
/// ensures    returns the scratch bytes vec_znx_idft_apply needs, one ring element wide and independent of the limb count
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
/// definition res[res_col][j] = idft(a[a_col])[j]
/// domain     res: a VecZnxBig; a: a VecZnxDft of the same degree; a is read, not clobbered
/// requires   scratch >= vec_znx_idft_apply_tmp_bytes()
/// ensures    res[res_col] is the inverse transform of a[a_col], limb by limb, in big words; limbs of res past a.size() are zero. idft(dft(x)) = x is the round trip the DFT-domain contracts are stated through
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
/// definition vec_znx_idft_apply(res, res_col, a, a_col)
/// domain     res: a VecZnxBig; a: a VecZnxDft of the same degree, taken mutably
/// ensures    res[res_col] holds idft(a[a_col]) as for vec_znx_idft_apply, and a[a_col] is left unspecified. Kept beside the basis form because it carries temporary-lifetime information a composition loses: the caller states that a is dead, which buys the backend the scratch
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
/// ensures    returns the scratch bytes vec_znx_idft_normalize_consume needs: the a_size-limb VecZnxBig the inverse transform lands in, plus the big normalization's own carry
/// test       test_vec_znx_idft_normalize_consume
/// ```
pub trait VecZnxIdftNormalizeConsumeTmpBytes {
    fn vec_znx_idft_normalize_consume_tmp_bytes(&self, res_size: usize, a_size: usize) -> usize;
}

/// Inverse DFT fused with normalization: `res[res_col] = normalize(idft(a[a_col]) + addend)`,
/// clobbering `a[a_col]`, at precision `res_k`.
///
/// ```text
/// op         vec_znx_idft_normalize_consume(res, res_base2k, res_k, res_col, a, a_col, a_base2k, addend, scratch)
/// class      derived
/// mutation   out-of-place
/// definition [[res]]_res_base2k = rnd([[idft(a[a_col])]]_a_base2k + sum_{j < a.size()} addend[addend_col][j] * 2^(-a_base2k * (j + 1)), res_k) (mod 1), canonical at res_base2k and res_k; the sum is 0 when no addend is given
/// domain     res: a VecZnx; a: a VecZnxDft taken mutably; addend: an optional VecZnx column read at a_base2k
/// requires   scratch >= vec_znx_idft_normalize_consume_tmp_bytes(res.size(), a.size())
/// ensures    [[res]] = [[idft(a)]] + [[addend]] at radix res_base2k, rounded once at precision res_k and canonical there, the addend read over its first a.size() limbs; a[a_col] is left unspecified
/// fallback   OEP default body: idft_apply_tmpa into a carved VecZnxBig, the optional add_small_assign, then big_normalize
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
/// definition idft(res[res_col])[j] = idft(a[a_col])[j] + idft(b[b_col])[j]
/// domain     res, a, b: VecZnxDft of the same degree
/// ensures    idft(res[res_col]) = idft(a[a_col]) + idft(b[b_col]) limb by limb, the image of the ring addition; operands shorter than res are zero-extended and every limb of res is written
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
/// definition vec_znx_dft_add(res, res_col, res, res_col, a, a_col)
/// domain     res, a: VecZnxDft of the same degree
/// ensures    idft(res[res_col]) gains idft(a[a_col]) limb by limb; limbs of res past a.size() keep their value
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
/// definition idft(res[res_col])[j] = idft(a[a_col])[j] - idft(b[b_col])[j]
/// domain     res, a, b: VecZnxDft of the same degree
/// ensures    idft(res[res_col]) = idft(a[a_col]) - idft(b[b_col]) limb by limb; operands shorter than res are zero-extended and every limb of res is written
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
/// definition vec_znx_dft_sub(res, res_col, res, res_col, a, a_col)
/// domain     res, a: VecZnxDft of the same degree
/// ensures    idft(res[res_col]) loses idft(a[a_col]) limb by limb; limbs of res past a.size() keep their value
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
/// definition vec_znx_dft_sub(res, res_col, a, a_col, res, res_col)
/// domain     res, a: VecZnxDft of the same degree
/// ensures    idft(res[res_col]) = idft(a[a_col]) - idft(res[res_col]) limb by limb; limbs of res past a.size() are negated in place
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
/// definition idft(res[res_col])[j] = idft(a[a_col])[offset + j * step]
/// domain     res, a: VecZnxDft of the same degree; step >= 1
/// ensures    limb j of res[res_col] holds limb offset + j * step of a[a_col], for as many limbs as res holds; limbs whose source is past a.size() are zero. It moves representation bytes, so both operands are the same backend's
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
/// definition idft(res[res_col])[j] = 0
/// domain     res: a VecZnxDft
/// ensures    every limb of res[res_col] is the representation of zero, so idft(res) = 0; the other columns are untouched
/// test       test_vec_znx_dft_zero
/// ```
pub trait VecZnxDftZero<B: Backend> {
    fn vec_znx_dft_zero(&self, res: &mut VecZnxDftBackendMut<'_, B>, res_col: usize);
}

/// Builds a backend-specific permutation plan that implements the DFT-domain
/// automorphism `tau_p: X -> X^p` for odd `p`. The plan captures the
/// slot↔slot permutation (plus any backend-specific bookkeeping such as a
/// half-spectrum conjugate flag) and is reusable across columns and limbs.
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
/// ensures    returns the backend's plan for tau_p in the DFT domain, reusable across columns and limbs. The Plan type is the one place the public API lets a backend's automorphism representation show, so a caller that wants to stay backend-agnostic owns it as the associated type
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
/// ensures    returns the scratch bytes vec_znx_dft_automorphism_add_with_plan needs: one min(res_size, a_size)-limb, one-column VecZnxDft for the rotated operand
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
/// definition idft(res[res_col])[j] = tau_p(idft(a[a_col])[j]), p the one the plan was built for
/// domain     res, a: VecZnxDft of the module degree; plan: built by vec_znx_dft_automorphism_plan for an odd p
/// ensures    idft(res[res_col]) = tau_p(idft(a[a_col])) limb by limb; limbs of res past a.size() are zero
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
    #[allow(clippy::too_many_arguments)]
    /// ```text
    /// op         vec_znx_dft_automorphism_add_with_plan(plan, res, res_col, a, a_col, scratch)
    /// class      derived
    /// mutation   accumulate
    /// definition vec_znx_dft_add_assign(res, res_col, vec_znx_dft_automorphism_with_plan(plan, tmp, a, a_col), 0)
    /// domain     res, a: VecZnxDft of the module degree
    /// requires   scratch >= vec_znx_dft_automorphism_add_with_plan_tmp_bytes(res.size(), a.size())
    /// ensures    res[res_col] gains tau_p(a[a_col]) over min(res.size(), a.size()) limbs; the limbs of res past that are untouched
    /// fallback   OEP default body: the automorphism into a carved one-column VecZnxDft, then dft_add_assign
    /// override   allowed, with vec_znx_dft_automorphism_add_with_plan_tmp_bytes
    /// test       test_vec_znx_dft_automorphism_add, test_vec_znx_dft_automorphism_add_with_plan_derived
    /// ```
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
    /// definition vec_znx_dft_automorphism_with_plan(vec_znx_dft_automorphism_plan(p), res, res_col, a, a_col)
    /// domain     res, a: VecZnxDft of the module degree; p odd
    /// ensures    as for vec_znx_dft_automorphism_with_plan, with the plan built and dropped inside the call; prefer the plan form when one p is reused
    /// fallback   OEP default body: build the plan, apply it, drop it
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
