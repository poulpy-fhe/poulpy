use crate::{
    layouts::{Backend, ScalarZnxBackendMut, ScalarZnxBackendRef, ScratchArena, VecZnxBackendMut, VecZnxBackendRef},
    source::Source,
};

/// ```text
/// op         vec_znx_normalize_tmp_bytes()
/// class      support
/// mutation   none
/// domain     -
/// ensures    returns the scratch bytes vec_znx_normalize and vec_znx_normalize_assign need; the carry chain is one ring element wide, so the answer does not depend on the operand sizes
/// exact      not an arithmetic operation
/// test       test_vec_znx_normalize
/// ```
pub trait VecZnxNormalizeTmpBytes {
    /// Returns the minimum number of bytes necessary for normalization.
    fn vec_znx_normalize_tmp_bytes(&self) -> usize;
}

/// ```text
/// op         vec_znx_zero(res, res_col)
/// class      basis
/// mutation   out-of-place
/// domain     res: a VecZnx or a window of one; res_col < res.cols()
/// ensures    every limb of res[res_col] is zero, so [[res]] = 0; the other columns are untouched
/// exact      exact
/// test       test_vec_znx_zero_matches_wrapper, test_vec_znx_window_ops
/// ```
pub trait VecZnxZero<B: Backend> {
    fn vec_znx_zero(&self, res: &mut VecZnxBackendMut<'_, B>, res_col: usize);
}

/// Converts a column to centered digits, rounding once at the destination precision.
///
/// For i64 backends, each input limb coefficient must lie in `[-2^62, 2^62]`,
/// and both radix widths must lie in `1..=62`. These bounds leave room for
/// shifted digits and propagated carries. They are caller preconditions;
/// normalization does not scan the input to validate them.
///
/// ```text
/// op         vec_znx_normalize(res, res_base2k, res_k, res_offset, res_col, a, a_base2k, a_col, scratch)
/// class      basis
/// mutation   out-of-place
/// domain     res, a: VecZnx or windows of one, read at res_base2k and a_base2k; both radix widths in 1..=62; every input digit in [-2^62, 2^62]
/// requires   scratch >= vec_znx_normalize_tmp_bytes()
/// ensures    [[res]]_res_base2k = [[a]]_a_base2k * 2^res_offset, rounded once at precision res_k and canonical there; every limb of res[res_col] is written and the limbs past res_k are zero
/// exact      exact
/// test       test_vec_znx_normalize, test_vec_znx_window_normalize_ops
/// ```
pub trait VecZnxNormalize<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    /// Normalizes the selected column of `a` at `res_k` bits into `res`.
    fn vec_znx_normalize(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_base2k: usize,
        res_k: usize,
        res_offset: i64,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// In-place normalization with the input and radix bounds of [`VecZnxNormalize`].
///
/// ```text
/// op         vec_znx_normalize_assign(base2k, k, a_offset, a, a_col, scratch)
/// class      variant
/// mutation   in-place
/// definition vec_znx_normalize(a, base2k, k, a_offset, a_col, a, base2k, a_col)
/// domain     a: a VecZnx or a window of one, read at base2k, with the input and radix bounds of VecZnxNormalize; a_offset <= 0, and a_offset != 0 also requires k == a.size() * base2k, since a positive offset would write a limb before reading it
/// requires   scratch >= vec_znx_normalize_tmp_bytes()
/// ensures    [[a]] after the call is [[a]] before the call times 2^a_offset, rounded once at precision k and canonical there
/// fallback   none, the kernel is required of every backend
/// override   required
/// exact      exact
/// test       test_vec_znx_normalize_assign, test_vec_znx_window_normalize_ops
/// ```
pub trait VecZnxNormalizeAssign<B: Backend> {
    fn vec_znx_normalize_assign(
        &self,
        base2k: usize,
        k: usize,
        a_offset: i64,
        a: &mut VecZnxBackendMut<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// ```text
/// op         vec_znx_add(res, res_col, a, a_col, b, b_col)
/// class      basis
/// mutation   out-of-place
/// domain     res, a, b: VecZnx or windows of one, read at one shared base2k
/// ensures    res[res_col] = a[a_col] + b[b_col] limb by limb; operands shorter than res are zero-extended, every limb of res is written, and the digits are not renormalized
/// sparse     a and b are the sparse-capable slots: a degree-n operand, n dividing N, stands for switch_ring_{n->N} of itself (4.5). Implemented in PR7 (#266); until then the kernels require the module degree
/// exact      exact
/// test       test_vec_znx_add_matches_reference, test_vec_znx_window_ops
/// ```
pub trait VecZnxAdd<B: Backend> {
    /// Adds the selected backend-native column of `a` to the selected backend-native column of `b`.
    fn vec_znx_add(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, B>,
        b_col: usize,
    );
}

/// ```text
/// op         vec_znx_add_assign(res, res_col, a, a_col)
/// class      variant
/// mutation   in-place
/// definition vec_znx_add(res, res_col, res, res_col, a, a_col)
/// domain     res, a: VecZnx or windows of one, read at one shared base2k
/// ensures    res[res_col] += a[a_col] limb by limb; limbs of res past a.size() add the zero-extension of a and so keep their value; the digits are not renormalized
/// sparse     a is the sparse-capable slot, as for vec_znx_add (4.5)
/// exact      exact
/// test       test_vec_znx_add_assign, test_vec_znx_add_assign_matches_wrapper
/// ```
pub trait VecZnxAddAssign<B: Backend> {
    fn vec_znx_add_assign(&self, res: &mut VecZnxBackendMut<'_, B>, res_col: usize, a: &VecZnxBackendRef<'_, B>, a_col: usize);
}

/// ```text
/// op         vec_znx_add_scalar_assign(res, res_col, res_limb, a, a_col)
/// class      derived
/// mutation   accumulate
/// definition vec_znx_add_assign(window_limbs(res, res_limb, 1, 1), res_col, a as a one-limb VecZnx, a_col)
/// domain     res: a VecZnx; res_limb < res.size(); a: a ScalarZnx of the module degree
/// ensures    limb res_limb of res[res_col] gains the coefficients of a[a_col]; every other limb of res is untouched; the digits are not renormalized
/// fallback   OEP default body: an add_assign on the one-limb window of res, the scalar read as a one-limb VecZnx
/// override   allowed, scratch-free
/// exact      exact
/// test       test_vec_znx_add_scalar_assign, test_vec_znx_add_scalar_assign_derived
/// ```
pub trait VecZnxAddScalarAssign<B: Backend> {
    fn vec_znx_add_scalar_assign(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        res_limb: usize,
        a: &ScalarZnxBackendRef<'_, B>,
        a_col: usize,
    );
}

/// ```text
/// op         vec_znx_sub(res, res_col, a, a_col, b, b_col)
/// class      basis
/// mutation   out-of-place
/// domain     res, a, b: VecZnx or windows of one, read at one shared base2k
/// ensures    res[res_col] = a[a_col] - b[b_col] limb by limb; operands shorter than res are zero-extended, every limb of res is written, and the digits are not renormalized
/// sparse     a and b are the sparse-capable slots, as for vec_znx_add (4.5)
/// exact      exact
/// test       test_vec_znx_sub, test_vec_znx_window_ops
/// ```
pub trait VecZnxSub<B: Backend> {
    fn vec_znx_sub(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, B>,
        b_col: usize,
    );
}

/// ```text
/// op         vec_znx_sub_assign(res, res_col, a, a_col)
/// class      variant
/// mutation   in-place
/// definition vec_znx_sub(res, res_col, res, res_col, a, a_col)
/// domain     res, a: VecZnx or windows of one, read at one shared base2k
/// ensures    res[res_col] -= a[a_col] limb by limb; limbs of res past a.size() keep their value; the digits are not renormalized
/// sparse     a is the sparse-capable slot, as for vec_znx_add (4.5)
/// exact      exact
/// test       test_vec_znx_sub_assign
/// ```
pub trait VecZnxSubAssign<B: Backend> {
    fn vec_znx_sub_assign(&self, res: &mut VecZnxBackendMut<'_, B>, res_col: usize, a: &VecZnxBackendRef<'_, B>, a_col: usize);
}

/// ```text
/// op         vec_znx_sub_negate_assign(res, res_col, a, a_col)
/// class      variant
/// mutation   in-place
/// definition vec_znx_sub(res, res_col, a, a_col, res, res_col)
/// domain     res, a: VecZnx or windows of one, read at one shared base2k
/// ensures    res[res_col] = a[a_col] - res[res_col] limb by limb; limbs of res past a.size() are negated in place; the digits are not renormalized
/// sparse     a is the sparse-capable slot, as for vec_znx_add (4.5)
/// exact      exact
/// test       test_vec_znx_sub_negate_assign
/// ```
pub trait VecZnxSubNegateAssign<B: Backend> {
    fn vec_znx_sub_negate_assign(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    );
}

/// ```text
/// op         vec_znx_negate(res, res_col, a, a_col)
/// class      basis
/// mutation   out-of-place
/// domain     res, a: VecZnx or windows of one
/// ensures    res[res_col] = -a[a_col] limb by limb; limbs of res past a.size() are zero
/// exact      exact
/// test       test_vec_znx_negate, test_vec_znx_negate_matches_wrapper
/// ```
pub trait VecZnxNegate<B: Backend> {
    fn vec_znx_negate(&self, res: &mut VecZnxBackendMut<'_, B>, res_col: usize, a: &VecZnxBackendRef<'_, B>, a_col: usize);
}

/// ```text
/// op         vec_znx_negate_assign(a, a_col)
/// class      variant
/// mutation   in-place
/// definition vec_znx_negate(a, a_col, a, a_col)
/// domain     a: a VecZnx or a window of one
/// ensures    every limb of a[a_col] is negated in place
/// exact      exact
/// test       test_vec_znx_negate_assign, test_vec_znx_negate_assign_matches_wrapper
/// ```
pub trait VecZnxNegateAssign<B: Backend> {
    fn vec_znx_negate_assign(&self, a: &mut VecZnxBackendMut<'_, B>, a_col: usize);
}

/// Returns scratch bytes required for the left-shift family:
/// [`VecZnxLsh::vec_znx_lsh`], [`VecZnxLshAdd::vec_znx_lsh_add`],
/// [`VecZnxLshSub::vec_znx_lsh_sub`] and
/// [`VecZnxLshAssign::vec_znx_lsh_assign`], on a destination of `res_size`
/// limbs.
///
/// ```text
/// op         vec_znx_lsh_tmp_bytes(res_size)
/// class      support
/// mutation   none
/// domain     res_size: the destination's limb count
/// ensures    returns the scratch bytes the whole left-shift family needs on that destination, the largest of the four default bodies: one res_size-limb VecZnx plus the normalization carry. It over-reports for the plain shift, never under-reports
/// exact      not an arithmetic operation
/// test       test_vec_znx_lsh
/// ```
pub trait VecZnxLshTmpBytes {
    fn vec_znx_lsh_tmp_bytes(&self, res_size: usize) -> usize;
}

/// ```text
/// op         vec_znx_lsh(base2k, k, res, res_col, a, a_col, scratch)
/// class      derived
/// mutation   out-of-place
/// definition vec_znx_normalize(res, base2k, res.size() * base2k, +k, res_col, a, base2k, a_col)
/// domain     res, a: VecZnx or windows of one, read at one shared base2k
/// requires   scratch >= vec_znx_lsh_tmp_bytes(res.size())
/// ensures    [[res]] = [[a]] * 2^k at precision res.size() * base2k, canonical there. A destination shorter than the source is not a truncation: the bits res cannot hold are rounded into its last limb jointly, as the normalization does
/// fallback   OEP default body: the normalization with offset = +k
/// override   allowed, with vec_znx_lsh_tmp_bytes
/// exact      exact
/// test       test_vec_znx_lsh, test_vec_znx_lsh_derived
/// ```
pub trait VecZnxLsh<B: Backend> {
    /// `res[res_col] = a[a_col] * 2^k`, both operands at the same radix: the
    /// operation is exactly `vec_znx_normalize` with `res_base2k = a_base2k =
    /// base2k`, `res_k = res.size() * base2k` and `offset = +k`.
    ///
    /// A destination shorter than the source is not a truncation: the limbs of
    /// the shifted value that `res` cannot hold are rounded into its last limb
    /// jointly, as the normalization does.
    /// A `k` beyond `a.size() * base2k` has shifted every bit of `a` past the
    /// integer part, so `res` is zeroed whatever its own width.
    ///
    /// Scratch is the family's
    /// [`vec_znx_lsh_tmp_bytes`](VecZnxLshTmpBytes::vec_znx_lsh_tmp_bytes) on
    /// `res.size()`.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_lsh(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// ```text
/// op         vec_znx_lsh_add(base2k, k, res, res_col, a, a_col, scratch)
/// class      derived
/// mutation   accumulate
/// definition vec_znx_add_assign(res, res_col, vec_znx_lsh(base2k, k, tmp, a, a_col), 0), tmp of res.size() limbs
/// domain     res, a: VecZnx or windows of one, read at one shared base2k
/// requires   scratch >= vec_znx_lsh_tmp_bytes(res.size())
/// ensures    res[res_col] gains the canonical vec_znx_lsh of a on a res.size()-limb destination. The addend is normalized, the sum is not: a digit of res may end one bit past the base2k range, and a caller that needs a canonical res normalizes it
/// fallback   OEP default body: the shift into a res.size()-limb temporary, then add_assign
/// override   allowed, with vec_znx_lsh_tmp_bytes
/// exact      exact
/// test       test_vec_znx_lsh_add_derived
/// ```
pub trait VecZnxLshAdd<B: Backend> {
    /// `res[res_col] += a[a_col] * 2^k`, both operands at the same radix: the
    /// addend is exactly [`vec_znx_lsh`](VecZnxLsh::vec_znx_lsh) on a
    /// `res.size()`-limb destination, the shift followed by a normalize at
    /// `base2k`.
    ///
    /// An addend shorter than the source is not a truncation: the limbs of the
    /// shifted value that `res.size()` cannot hold are rounded into the last
    /// one jointly, as the normalization does.
    ///
    /// Scratch is the family's
    /// [`vec_znx_lsh_tmp_bytes`](VecZnxLshTmpBytes::vec_znx_lsh_tmp_bytes) on
    /// `res.size()`.
    ///
    /// Normalization contract: the shifted operand `a` is normalized on the fly
    /// (its own inter-limb carries are propagated), so the addend is in the
    /// canonical `base2k` digit range. The addition into `res` is **not**
    /// re-normalized: adding a normalized digit onto an already-normalized `res`
    /// limb can leave that limb one bit beyond the `base2k` range. Callers that
    /// require a normalized `res` afterwards must normalize it themselves; this
    /// op alone does not restore the digit contract on `res`.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_lsh_add(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Returns scratch bytes required for the right-shift family:
/// [`VecZnxRsh::vec_znx_rsh`], [`VecZnxRshAdd::vec_znx_rsh_add`],
/// [`VecZnxRshSub::vec_znx_rsh_sub`] and
/// [`VecZnxRshAssign::vec_znx_rsh_assign`], on a destination of `res_size`
/// limbs.
///
/// ```text
/// op         vec_znx_rsh_tmp_bytes(res_size)
/// class      support
/// mutation   none
/// domain     res_size: the destination's limb count
/// ensures    returns the scratch bytes the whole right-shift family needs on that destination, the largest of the four default bodies: one res_size-limb VecZnx plus the normalization carry
/// exact      not an arithmetic operation
/// test       test_vec_znx_rsh
/// ```
pub trait VecZnxRshTmpBytes {
    fn vec_znx_rsh_tmp_bytes(&self, res_size: usize) -> usize;
}

/// ```text
/// op         vec_znx_rsh(base2k, k, res, res_col, a, a_col, scratch)
/// class      derived
/// mutation   out-of-place
/// definition vec_znx_normalize(res, base2k, res.size() * base2k, -k, res_col, a, base2k, a_col)
/// domain     res, a: VecZnx or windows of one, read at one shared base2k
/// requires   scratch >= vec_znx_rsh_tmp_bytes(res.size())
/// ensures    [[res]] = [[a]] / 2^k at precision res.size() * base2k, canonical there; the bits res cannot hold are rounded into its last limb jointly. A k beyond res.size() * base2k has moved every bit past the last limb, so res is zero
/// fallback   OEP default body: the normalization with offset = -k
/// override   allowed, with vec_znx_rsh_tmp_bytes
/// exact      exact
/// test       test_vec_znx_rsh, test_vec_znx_rsh_derived
/// ```
pub trait VecZnxRsh<B: Backend> {
    /// `res[res_col] = a[a_col] / 2^k`, both operands at the same radix: the
    /// operation is exactly `vec_znx_normalize` with `res_base2k = a_base2k =
    /// base2k`, `res_k = res.size() * base2k` and `offset = -k`.
    ///
    /// A destination shorter than the source is not a truncation: the limbs of
    /// the shifted value that `res` cannot hold are rounded into its last limb
    /// jointly, as the normalization does. A `k` beyond `res.size() * base2k`
    /// has shifted every bit past the destination's last limb, so `res` is
    /// zero.
    ///
    /// Scratch is the family's
    /// [`vec_znx_rsh_tmp_bytes`](VecZnxRshTmpBytes::vec_znx_rsh_tmp_bytes) on
    /// `res.size()`.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// ```text
/// op         vec_znx_rsh_add(base2k, k, res, res_col, a, a_col, scratch)
/// class      derived
/// mutation   accumulate
/// definition vec_znx_add_assign(res, res_col, vec_znx_rsh(base2k, k, tmp, a, a_col), 0), tmp of res.size() limbs
/// domain     res, a: VecZnx or windows of one, read at one shared base2k
/// requires   scratch >= vec_znx_rsh_tmp_bytes(res.size())
/// ensures    res[res_col] gains the canonical vec_znx_rsh of a on a res.size()-limb destination; a k beyond res.size() * base2k adds nothing. The addend is normalized, the sum is not
/// fallback   OEP default body: the shift into a res.size()-limb temporary, then add_assign
/// override   allowed, with vec_znx_rsh_tmp_bytes
/// exact      exact
/// test       test_vec_znx_rsh_add_derived
/// ```
pub trait VecZnxRshAdd<B: Backend> {
    /// `res[res_col] += a[a_col] / 2^k`, both operands at the same radix: the
    /// addend is exactly [`vec_znx_rsh`](VecZnxRsh::vec_znx_rsh) on a
    /// `res.size()`-limb destination, the shift followed by a normalize at
    /// `base2k`.
    ///
    /// An addend shorter than the source is not a truncation: the limbs of the
    /// shifted value that `res.size()` cannot hold are rounded into the last
    /// one jointly, as the normalization does. A `k` beyond
    /// `res.size() * base2k` has shifted every bit past the last limb, so
    /// nothing is added.
    ///
    /// Scratch is the family's
    /// [`vec_znx_rsh_tmp_bytes`](VecZnxRshTmpBytes::vec_znx_rsh_tmp_bytes) on
    /// `res.size()`.
    ///
    /// Normalization contract: the shifted operand `a` is normalized on the fly
    /// (its own inter-limb carries are propagated), so the addend is in the
    /// canonical `base2k` digit range. The addition into `res` is **not**
    /// re-normalized: adding a normalized digit onto an already-normalized `res`
    /// limb can leave that limb one bit beyond the `base2k` range. Callers that
    /// require a normalized `res` afterwards must normalize it themselves; this
    /// op alone does not restore the digit contract on `res`.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh_add(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// ```text
/// op         vec_znx_lsh_sub(base2k, k, res, res_col, a, a_col, scratch)
/// class      derived
/// mutation   accumulate
/// definition vec_znx_sub_assign(res, res_col, vec_znx_lsh(base2k, k, tmp, a, a_col), 0), tmp of res.size() limbs
/// domain     res, a: VecZnx or windows of one, read at one shared base2k
/// requires   scratch >= vec_znx_lsh_tmp_bytes(res.size())
/// ensures    res[res_col] loses the canonical vec_znx_lsh of a on a res.size()-limb destination. The subtrahend is normalized, the difference is not
/// fallback   OEP default body: the shift into a res.size()-limb temporary, then sub_assign
/// override   allowed, with vec_znx_lsh_tmp_bytes
/// exact      exact
/// test       test_vec_znx_lsh_sub_derived
/// ```
pub trait VecZnxLshSub<B: Backend> {
    /// `res[res_col] -= a[a_col] * 2^k`, both operands at the same radix: the
    /// subtrahend is exactly [`vec_znx_lsh`](VecZnxLsh::vec_znx_lsh) on a
    /// `res.size()`-limb destination, the shift followed by a normalize at
    /// `base2k`.
    ///
    /// A subtrahend shorter than the source is not a truncation: the limbs of
    /// the shifted value that `res.size()` cannot hold are rounded into the
    /// last one jointly, as the normalization does.
    ///
    /// Scratch is the family's
    /// [`vec_znx_lsh_tmp_bytes`](VecZnxLshTmpBytes::vec_znx_lsh_tmp_bytes) on
    /// `res.size()`.
    ///
    /// Normalization contract: as for
    /// [`vec_znx_lsh_add`](VecZnxLshAdd::vec_znx_lsh_add), the subtraction into
    /// `res` is **not** re-normalized; callers that need a normalized `res`
    /// afterwards must normalize it themselves.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_lsh_sub(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// ```text
/// op         vec_znx_rsh_sub(base2k, k, res, res_col, a, a_col, scratch)
/// class      derived
/// mutation   accumulate
/// definition vec_znx_sub_assign(res, res_col, vec_znx_rsh(base2k, k, tmp, a, a_col), 0), tmp of res.size() limbs
/// domain     res, a: VecZnx or windows of one, read at one shared base2k
/// requires   scratch >= vec_znx_rsh_tmp_bytes(res.size())
/// ensures    res[res_col] loses the canonical vec_znx_rsh of a on a res.size()-limb destination; a k beyond res.size() * base2k subtracts nothing. The subtrahend is normalized, the difference is not
/// fallback   OEP default body: the shift into a res.size()-limb temporary, then sub_assign
/// override   allowed, with vec_znx_rsh_tmp_bytes
/// exact      exact
/// test       test_vec_znx_rsh_sub_derived
/// ```
pub trait VecZnxRshSub<B: Backend> {
    /// `res[res_col] -= a[a_col] / 2^k`, both operands at the same radix: the
    /// subtrahend is exactly [`vec_znx_rsh`](VecZnxRsh::vec_znx_rsh) on a
    /// `res.size()`-limb destination, the shift followed by a normalize at
    /// `base2k`.
    ///
    /// A subtrahend shorter than the source is not a truncation: the limbs of
    /// the shifted value that `res.size()` cannot hold are rounded into the
    /// last one jointly, as the normalization does. A `k` beyond
    /// `res.size() * base2k` has shifted every bit past the last limb, so
    /// nothing is subtracted.
    ///
    /// Scratch is the family's
    /// [`vec_znx_rsh_tmp_bytes`](VecZnxRshTmpBytes::vec_znx_rsh_tmp_bytes) on
    /// `res.size()`.
    ///
    /// Normalization contract: as for
    /// [`vec_znx_rsh_add`](VecZnxRshAdd::vec_znx_rsh_add), the subtraction into
    /// `res` is **not** re-normalized; callers that need a normalized `res`
    /// afterwards must normalize it themselves.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh_sub(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// ```text
/// op         vec_znx_lsh_assign(base2k, k, a, a_col, scratch)
/// class      derived
/// mutation   in-place
/// definition vec_znx_lsh(base2k, k, a, a_col, a, a_col) over a.size() limbs
/// domain     a: a VecZnx or a window of one, read at base2k
/// requires   scratch >= vec_znx_lsh_tmp_bytes(a.size())
/// ensures    [[a]] after the call is [[a]] before the call times 2^k, canonical at a.size() * base2k; the other columns of a are untouched
/// fallback   OEP default body: the shift into an a.size()-limb temporary, then copy back, since a positive offset cannot be taken in place
/// override   allowed, with vec_znx_lsh_tmp_bytes; the CPU families override it with a fused kernel
/// exact      exact
/// test       test_vec_znx_lsh_assign, test_vec_znx_lsh_assign_derived
/// ```
pub trait VecZnxLshAssign<B: Backend> {
    /// `a[a_col] *= 2^k`, in place on the one selected column: the operation
    /// is exactly [`vec_znx_lsh`](VecZnxLsh::vec_znx_lsh) with `a` as both
    /// source and destination, the shift followed by a normalize at `base2k`
    /// over `a.size()` limbs. The other columns of `a` are untouched.
    ///
    /// Scratch is the family's
    /// [`vec_znx_lsh_tmp_bytes`](VecZnxLshTmpBytes::vec_znx_lsh_tmp_bytes) on
    /// `a.size()`.
    fn vec_znx_lsh_assign(
        &self,
        base2k: usize,
        k: usize,
        a: &mut VecZnxBackendMut<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// ```text
/// op         vec_znx_rsh_assign(base2k, k, a, a_col, scratch)
/// class      derived
/// mutation   in-place
/// definition vec_znx_normalize_assign(base2k, a.size() * base2k, -k, a, a_col)
/// domain     a: a VecZnx or a window of one, read at base2k
/// requires   scratch >= vec_znx_rsh_tmp_bytes(a.size())
/// ensures    [[a]] after the call is [[a]] before the call divided by 2^k, canonical at a.size() * base2k; a k beyond a.size() * base2k zeroes the column; the other columns of a are untouched
/// fallback   OEP default body: one in-place normalization with offset = -k, no temporary
/// override   allowed, with vec_znx_rsh_tmp_bytes
/// exact      exact
/// test       test_vec_znx_rsh_assign, test_vec_znx_rsh_assign_derived
/// ```
pub trait VecZnxRshAssign<B: Backend> {
    /// `a[a_col] /= 2^k`, in place on the one selected column: the operation
    /// is exactly [`vec_znx_rsh`](VecZnxRsh::vec_znx_rsh) with `a` as both
    /// source and destination, the shift followed by a normalize at `base2k`
    /// over `a.size()` limbs. A `k` beyond `a.size() * base2k` has shifted
    /// every bit past the last limb, so the column is zeroed. The other
    /// columns of `a` are untouched.
    ///
    /// Scratch is the family's
    /// [`vec_znx_rsh_tmp_bytes`](VecZnxRshTmpBytes::vec_znx_rsh_tmp_bytes) on
    /// `a.size()`.
    fn vec_znx_rsh_assign(
        &self,
        base2k: usize,
        k: usize,
        a: &mut VecZnxBackendMut<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// ```text
/// op         vec_znx_rotate(p, res, res_col, a, a_col)
/// class      basis
/// mutation   out-of-place
/// domain     res, a: dense VecZnx of the module degree; a ring operation, so windows are rejected
/// ensures    res[res_col] = X^p * a[a_col] in Z[X]/(X^N + 1), limb by limb, p taken modulo 2N; limbs of res past a.size() are zero
/// exact      exact
/// test       test_vec_znx_rotate
/// ```
pub trait VecZnxRotate<B: Backend> {
    /// Multiplies the selected column of `a` by X^p and stores the result in `res_col` of `res`.
    fn vec_znx_rotate(
        &self,
        p: i64,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    );
}

/// ```text
/// op         vec_znx_rotate_assign_tmp_bytes()
/// class      support
/// mutation   none
/// domain     -
/// ensures    returns the scratch bytes vec_znx_rotate_assign needs, one ring element wide and independent of the limb count
/// exact      not an arithmetic operation
/// test       test_vec_znx_rotate_assign
/// ```
pub trait VecZnxRotateAssignTmpBytes {
    fn vec_znx_rotate_assign_tmp_bytes(&self) -> usize;
}

/// ```text
/// op         vec_znx_rotate_assign(p, a, a_col, scratch)
/// class      variant
/// mutation   in-place
/// definition vec_znx_rotate(p, a, a_col, a, a_col)
/// domain     a: a dense VecZnx of the module degree
/// requires   scratch >= vec_znx_rotate_assign_tmp_bytes()
/// ensures    a[a_col] = X^p * a[a_col]; the other columns are untouched
/// fallback   none, the kernel is required of every backend
/// override   required
/// exact      exact
/// test       test_vec_znx_rotate_assign
/// ```
pub trait VecZnxRotateAssign<B: Backend> {
    /// Multiplies the selected column of `a` by X^p in place.
    fn vec_znx_rotate_assign(&self, p: i64, a: &mut VecZnxBackendMut<'_, B>, a_col: usize, scratch: &mut ScratchArena<'_, B>);
}

/// ```text
/// op         vec_znx_automorphism(k, res, res_col, a, a_col)
/// class      basis
/// mutation   out-of-place
/// domain     res, a: dense VecZnx of the module degree; k odd
/// ensures    res[res_col] = tau_k(a[a_col]), the ring automorphism X -> X^k, limb by limb; limbs of res past a.size() are zero
/// exact      exact
/// test       test_vec_znx_automorphism
/// ```
pub trait VecZnxAutomorphism<B: Backend> {
    /// Applies the automorphism X -> X^k on the selected column of `a` and stores the result in `res_col` of `res`.
    fn vec_znx_automorphism(
        &self,
        k: i64,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    );
}

/// ```text
/// op         vec_znx_automorphism_assign_tmp_bytes()
/// class      support
/// mutation   none
/// domain     -
/// ensures    returns the scratch bytes vec_znx_automorphism_assign needs, one ring element wide and independent of the limb count
/// exact      not an arithmetic operation
/// test       test_vec_znx_automorphism_assign
/// ```
pub trait VecZnxAutomorphismAssignTmpBytes {
    fn vec_znx_automorphism_assign_tmp_bytes(&self) -> usize;
}

/// ```text
/// op         vec_znx_automorphism_assign(k, res, res_col, scratch)
/// class      variant
/// mutation   in-place
/// definition vec_znx_automorphism(k, res, res_col, res, res_col)
/// domain     res: a dense VecZnx of the module degree; k odd
/// requires   scratch >= vec_znx_automorphism_assign_tmp_bytes()
/// ensures    res[res_col] = tau_k(res[res_col]); the other columns are untouched
/// fallback   none, the kernel is required of every backend
/// override   required
/// exact      exact
/// test       test_vec_znx_automorphism_assign
/// ```
pub trait VecZnxAutomorphismAssign<B: Backend> {
    /// Applies the automorphism X -> X^k on the selected column of `res` in place.
    fn vec_znx_automorphism_assign(
        &self,
        k: i64,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// ```text
/// op         scalar_znx_automorphism(k, res, res_col, a, a_col)
/// class      basis
/// mutation   out-of-place
/// domain     res, a: ScalarZnx of the module degree; k odd
/// ensures    res[res_col] = tau_k(a[a_col]), the ring automorphism X -> X^k on the single limb
/// exact      exact
/// test       test_scalar_znx_automorphism
/// ```
pub trait ScalarZnxAutomorphism<B: Backend> {
    /// Applies the automorphism X -> X^k on the selected column of `a` and stores the result in `res_col` of `res`.
    fn scalar_znx_automorphism(
        &self,
        k: i64,
        res: &mut ScalarZnxBackendMut<'_, B>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, B>,
        a_col: usize,
    );
}

/// Multiplies the selected column by `(X^p - 1)` in `Z[X]/(X^N + 1)`.
///
/// ```text
/// op         vec_znx_mul_xp_minus_one(p, res, res_col, a, a_col)
/// class      derived
/// mutation   out-of-place
/// definition vec_znx_sub(res, res_col, vec_znx_rotate(p, a, a_col), a, a_col)
/// domain     res, a: dense VecZnx of the module degree
/// ensures    res[res_col] = (X^p - 1) * a[a_col] in Z[X]/(X^N + 1), limb by limb; the digits are not renormalized
/// fallback   OEP default body: rotate into res, then subtract the unrotated operand in place
/// override   allowed, scratch-free
/// exact      exact
/// test       test_vec_znx_mul_xp_minus_one, test_vec_znx_mul_xp_minus_one_derived
/// ```
pub trait VecZnxMulXpMinusOne<B: Backend> {
    fn vec_znx_mul_xp_minus_one(
        &self,
        p: i64,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    );
}

/// Returns scratch bytes required for
/// [`VecZnxMulXpMinusOneAssign::vec_znx_mul_xp_minus_one_assign`] on a
/// destination of `size` limbs.
///
/// ```text
/// op         vec_znx_mul_xp_minus_one_assign_tmp_bytes(size)
/// class      support
/// mutation   none
/// domain     size: the destination's limb count
/// ensures    returns the scratch bytes vec_znx_mul_xp_minus_one_assign needs, one size-limb VecZnx: vec_znx_rotate is a ring operation and rejects a windowed view, so the per-limb form is not available
/// exact      not an arithmetic operation
/// test       test_vec_znx_mul_xp_minus_one_assign
/// ```
pub trait VecZnxMulXpMinusOneAssignTmpBytes {
    fn vec_znx_mul_xp_minus_one_assign_tmp_bytes(&self, size: usize) -> usize;
}

/// ```text
/// op         vec_znx_mul_xp_minus_one_assign(p, res, res_col, scratch)
/// class      derived
/// mutation   in-place
/// definition vec_znx_mul_xp_minus_one(p, res, res_col, res, res_col)
/// domain     res: a dense VecZnx of the module degree
/// requires   scratch >= vec_znx_mul_xp_minus_one_assign_tmp_bytes(res.size())
/// ensures    res[res_col] = (X^p - 1) * res[res_col]; the other columns are untouched; the digits are not renormalized
/// fallback   OEP default body: the product into a res.size()-limb temporary, then copy back
/// override   allowed, with vec_znx_mul_xp_minus_one_assign_tmp_bytes; the CPU families override it with a fused kernel
/// exact      exact
/// test       test_vec_znx_mul_xp_minus_one_assign, test_vec_znx_mul_xp_minus_one_assign_derived
/// ```
pub trait VecZnxMulXpMinusOneAssign<B: Backend> {
    fn vec_znx_mul_xp_minus_one_assign(
        &self,
        p: i64,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Switches ring degree between `a` and `res` by truncation or zero-padding.
///
/// ```text
/// op         vec_znx_switch_ring(res, res_col, a, a_col)
/// class      basis
/// mutation   out-of-place
/// domain     res, a: dense VecZnx whose degrees divide one another
/// ensures    res[res_col] is a[a_col] moved between the two degrees, limb by limb: growing by a factor g sends coefficient j of a to coefficient j * g of res and zeroes the rest, shrinking by g keeps the coefficients of a at multiples of g. Limbs of res past a.size() are zero. Growing is the ring embedding a degree-n operand denotes
/// exact      exact
/// test       test_vec_znx_switch_ring, test_vec_znx_switch_ring_matches_wrapper
/// ```
pub trait VecZnxSwitchRing<B: Backend> {
    fn vec_znx_switch_ring(&self, res: &mut VecZnxBackendMut<'_, B>, res_col: usize, a: &VecZnxBackendRef<'_, B>, a_col: usize);
}

/// ```text
/// op         vec_znx_copy(res, res_col, a, a_col)
/// class      basis
/// mutation   out-of-place
/// domain     res, a: VecZnx or windows of one, of equal degree
/// ensures    res[res_col] holds a[a_col] limb by limb; limbs of res past a.size() are zero
/// exact      exact
/// test       test_vec_znx_copy, test_vec_znx_copy_matches_wrapper, test_vec_znx_window_ops
/// ```
pub trait VecZnxCopy<B: Backend> {
    fn vec_znx_copy(&self, res: &mut VecZnxBackendMut<'_, B>, res_col: usize, a: &VecZnxBackendRef<'_, B>, a_col: usize);
}

/// ```text
/// op         vec_znx_fill_uniform_source(base2k, k, res, res_col, source)
/// class      basis
/// mutation   out-of-place
/// domain     res: a dense VecZnx; base2k in 1..=62; source: the caller's pseudorandom stream
/// ensures    [[res]] is uniform over the torus at precision k and canonical there: the digits are uniform in the centered base2k range, the low (-k) mod base2k bits of the last live limb are zero and the limbs past k are zero. The byte-to-digit mapping is fixed, so a device generator can reproduce the stream bit for bit
/// exact      exact
/// test       test_vec_znx_fill_uniform
/// ```
pub trait VecZnxFillUniformSource<B: Backend> {
    /// Fills a column with a uniform `k`-bit torus value in base `2^base2k`.
    ///
    /// Unused low bits in the last live limb and limbs above `k` are zeroed.
    fn vec_znx_fill_uniform_source(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        source: &mut Source,
    );
}
