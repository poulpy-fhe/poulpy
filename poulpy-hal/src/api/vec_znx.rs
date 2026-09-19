use crate::{
    layouts::{Backend, ScalarZnxBackendMut, ScalarZnxBackendRef, ScratchArena, VecZnxBackendMut, VecZnxBackendRef},
    source::Source,
};

/// Scratch requirement of the normalization operations.
///
/// ```text
/// op         vec_znx_normalize_tmp_bytes()
/// class      support
/// mutation   none
/// domain     -
/// ensures    returns the scratch bytes vec_znx_normalize and vec_znx_normalize_assign require, independently of operand limb counts
/// test       test_vec_znx_normalize
/// ```
pub trait VecZnxNormalizeTmpBytes {
    /// Returns the scratch byte count the normalization operations require.
    fn vec_znx_normalize_tmp_bytes(&self) -> usize;
}

/// Zeroing of one column of a vector.
///
/// ```text
/// op         vec_znx_zero(res, res_col)
/// class      basis
/// mutation   out-of-place
/// definition res[res_col,j] = 0; the other columns of res are untouched
/// domain     res: a VecZnx or a window of one; res_col < res.cols()
/// ensures    every limb of res[res_col] is zero; the other columns are untouched
/// test       test_vec_znx_zero_matches_wrapper, test_vec_znx_window_ops
/// ```
pub trait VecZnxZero<B: Backend> {
    /// Writes zero into every limb of `res[res_col]`.
    fn vec_znx_zero(&self, res: &mut VecZnxBackendMut<'_, B>, res_col: usize);
}

/// Normalization of a column at a destination radix and precision, with a power-of-two offset.
///
/// ```text
/// op         vec_znx_normalize(res, res_base2k, res_k, res_offset, res_col, a, a_base2k, a_col, scratch)
/// class      basis
/// mutation   out-of-place
/// definition res[res_col] = canon([[a[a_col]]]_a_base2k * 2^res_offset, res_base2k, res_k, res.size()); the other columns of res are untouched
/// domain     res, a: VecZnx or windows of one, read at res_base2k and a_base2k; both radix widths in 1..=62; res_k <= res.size() * res_base2k; every input digit in [-2^62, 2^62]
/// requires   scratch >= vec_znx_normalize_tmp_bytes()
/// ensures    res[res_col] is canonical at radix res_base2k and precision res_k and congruent modulo 1 to rnd([[a[a_col]]]_a_base2k * 2^res_offset, res_k); every limb of the selected column is written
/// test       test_vec_znx_normalize, test_vec_znx_window_normalize_ops
/// ```
pub trait VecZnxNormalize<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    /// Writes `a[a_col]`, scaled by `2^res_offset` and normalized at `res_base2k` and `res_k`, into `res[res_col]`.
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

/// In-place normalization of a column at a radix and precision, with a power-of-two offset.
///
/// ```text
/// op         vec_znx_normalize_assign(base2k, k, a_offset, a, a_col, scratch)
/// class      variant
/// mutation   in-place
/// definition a[a_col] = canon([[old(a)[a_col]]]_base2k * 2^a_offset, base2k, k, a.size()); the other columns of a are untouched
/// domain     a: a VecZnx or a window of one, read at base2k; base2k in 1..=62; every input digit in [-2^62, 2^62]; k <= a.size() * base2k; a_offset <= 0; a_offset != 0 requires k == a.size() * base2k
/// requires   scratch >= vec_znx_normalize_tmp_bytes()
/// ensures    a[a_col] is canonical at radix base2k and precision k and congruent modulo 1 to rnd([[old(a)[a_col]]]_base2k * 2^a_offset, k)
/// fallback   none; required
/// override   required
/// test       test_vec_znx_normalize_assign, test_vec_znx_window_normalize_ops
/// ```
pub trait VecZnxNormalizeAssign<B: Backend> {
    /// Replaces `a[a_col]` with its value scaled by `2^a_offset` and normalized at `base2k` and `k`.
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

/// Limbwise sum of two columns.
///
/// ```text
/// op         vec_znx_add(res, res_col, a, a_col, b, b_col)
/// class      basis
/// mutation   out-of-place
/// definition res[res_col,j] = a[a_col,j] + b[b_col,j]; the other columns of res are untouched
/// domain     res, a, b: VecZnx or windows of one, read at one shared base2k
/// ensures    res[res_col] = a[a_col] + b[b_col] limb by limb; operands shorter than res are zero-extended, every limb of res is written, and the digits are not renormalized
/// sparse     a and b are the sparse-capable slots: a dense degree-n operand, n dividing N, stands for switch_ring_{n->N} of itself; a window in these slots has the width of res
/// test       test_vec_znx_add_matches_reference, test_vec_znx_window_ops, test_vec_znx_sparse_add_sub
/// ```
pub trait VecZnxAdd<B: Backend> {
    /// Writes the sum of `a[a_col]` and `b[b_col]` into `res[res_col]`.
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

/// In-place limbwise addition of a column.
///
/// ```text
/// op         vec_znx_add_assign(res, res_col, a, a_col)
/// class      variant
/// mutation   in-place
/// definition res[res_col,j] = old(res)[res_col,j] + a[a_col,j]; limbs from a.size() onward and the other columns of res are untouched
/// domain     res, a: VecZnx or windows of one, read at one shared base2k
/// ensures    res[res_col] gains a[a_col] limb by limb; limbs of res past a.size() are untouched; the digits are not renormalized
/// sparse     a is the sparse-capable slot, as for vec_znx_add
/// test       test_vec_znx_add_assign, test_vec_znx_add_assign_matches_wrapper, test_vec_znx_sparse_add_sub
/// ```
pub trait VecZnxAddAssign<B: Backend> {
    /// Adds `a[a_col]` into `res[res_col]`.
    fn vec_znx_add_assign(&self, res: &mut VecZnxBackendMut<'_, B>, res_col: usize, a: &VecZnxBackendRef<'_, B>, a_col: usize);
}

/// Addition of a scalar polynomial into one limb of a column.
///
/// ```text
/// op         vec_znx_add_scalar_assign(res, res_col, res_limb, a, a_col)
/// class      derived
/// mutation   accumulate
/// definition res[res_col,res_limb] = old(res)[res_col,res_limb] + a[a_col]; all other limbs and columns of res are untouched
/// domain     res: a VecZnx; res_limb < res.size(); a: a ScalarZnx of degree N
/// ensures    limb res_limb of res[res_col] gains the coefficients of a[a_col]; every other limb of res is untouched; the digits are not renormalized
/// fallback   default body: add_assign on the one-limb window of res, with a read as a one-limb VecZnx
/// override   allowed, scratch-free
/// test       test_vec_znx_add_scalar_assign, test_vec_znx_add_scalar_assign_derived
/// ```
pub trait VecZnxAddScalarAssign<B: Backend> {
    /// Adds `a[a_col]` into limb `res_limb` of `res[res_col]`.
    fn vec_znx_add_scalar_assign(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        res_limb: usize,
        a: &ScalarZnxBackendRef<'_, B>,
        a_col: usize,
    );
}

/// Limbwise difference of two columns.
///
/// ```text
/// op         vec_znx_sub(res, res_col, a, a_col, b, b_col)
/// class      basis
/// mutation   out-of-place
/// definition res[res_col,j] = a[a_col,j] - b[b_col,j]; the other columns of res are untouched
/// domain     res, a, b: VecZnx or windows of one, read at one shared base2k
/// ensures    res[res_col] = a[a_col] - b[b_col] limb by limb; operands shorter than res are zero-extended, every limb of res is written, and the digits are not renormalized
/// sparse     a and b are the sparse-capable slots, as for vec_znx_add
/// test       test_vec_znx_sub, test_vec_znx_window_ops, test_vec_znx_sparse_add_sub
/// ```
pub trait VecZnxSub<B: Backend> {
    /// Writes the difference of `a[a_col]` and `b[b_col]` into `res[res_col]`.
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

/// In-place limbwise subtraction of a column.
///
/// ```text
/// op         vec_znx_sub_assign(res, res_col, a, a_col)
/// class      variant
/// mutation   in-place
/// definition res[res_col,j] = old(res)[res_col,j] - a[a_col,j]; limbs from a.size() onward and the other columns of res are untouched
/// domain     res, a: VecZnx or windows of one, read at one shared base2k
/// ensures    res[res_col] loses a[a_col] limb by limb; limbs from a.size() onward are untouched; the digits are not renormalized
/// sparse     a is the sparse-capable slot, as for vec_znx_add
/// test       test_vec_znx_sub_assign, test_vec_znx_sparse_add_sub
/// ```
pub trait VecZnxSubAssign<B: Backend> {
    /// Subtracts `a[a_col]` from `res[res_col]`.
    fn vec_znx_sub_assign(&self, res: &mut VecZnxBackendMut<'_, B>, res_col: usize, a: &VecZnxBackendRef<'_, B>, a_col: usize);
}

/// In-place limbwise subtraction of the destination from a column.
///
/// ```text
/// op         vec_znx_sub_negate_assign(res, res_col, a, a_col)
/// class      variant
/// mutation   in-place
/// definition res[res_col,j] = a[a_col,j] - old(res)[res_col,j]; the other columns of res are untouched
/// domain     res, a: VecZnx or windows of one, read at one shared base2k
/// ensures    res[res_col] holds a[a_col] minus its pre-call value limb by limb; limbs from a.size() onward are negated in place; the digits are not renormalized
/// sparse     a is the sparse-capable slot, as for vec_znx_add
/// test       test_vec_znx_sub_negate_assign, test_vec_znx_sparse_add_sub
/// ```
pub trait VecZnxSubNegateAssign<B: Backend> {
    /// Replaces `res[res_col]` with `a[a_col]` minus its pre-call value.
    fn vec_znx_sub_negate_assign(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    );
}

/// Limbwise negation of a column.
///
/// ```text
/// op         vec_znx_negate(res, res_col, a, a_col)
/// class      basis
/// mutation   out-of-place
/// definition res[res_col,j] = -a[a_col,j]; the other columns of res are untouched
/// domain     res, a: VecZnx or windows of one
/// ensures    res[res_col] = -a[a_col] limb by limb; limbs of res past a.size() are zero
/// test       test_vec_znx_negate, test_vec_znx_negate_matches_wrapper
/// ```
pub trait VecZnxNegate<B: Backend> {
    /// Writes the negation of `a[a_col]` into `res[res_col]`.
    fn vec_znx_negate(&self, res: &mut VecZnxBackendMut<'_, B>, res_col: usize, a: &VecZnxBackendRef<'_, B>, a_col: usize);
}

/// In-place limbwise negation of a column.
///
/// ```text
/// op         vec_znx_negate_assign(a, a_col)
/// class      variant
/// mutation   in-place
/// definition a[a_col,j] = -old(a)[a_col,j]; the other columns of a are untouched
/// domain     a: a VecZnx or a window of one
/// ensures    every limb of a[a_col] is negated in place
/// test       test_vec_znx_negate_assign, test_vec_znx_negate_assign_matches_wrapper
/// ```
pub trait VecZnxNegateAssign<B: Backend> {
    /// Negates every limb of `a[a_col]` in place.
    fn vec_znx_negate_assign(&self, a: &mut VecZnxBackendMut<'_, B>, a_col: usize);
}

/// Scratch requirement of the left-shift operations.
///
/// ```text
/// op         vec_znx_lsh_tmp_bytes(res_size)
/// class      support
/// mutation   none
/// domain     res_size: the destination's limb count
/// ensures    returns sufficient scratch bytes for vec_znx_lsh, vec_znx_lsh_add, vec_znx_lsh_sub and vec_znx_lsh_assign on a res_size-limb destination
/// test       test_vec_znx_lsh
/// ```
pub trait VecZnxLshTmpBytes {
    /// Returns the scratch byte count the left-shift operations require on a destination of `res_size` limbs.
    fn vec_znx_lsh_tmp_bytes(&self, res_size: usize) -> usize;
}

/// Left shift of a column by a power of two, normalized into the destination.
///
/// ```text
/// op         vec_znx_lsh(base2k, k, res, res_col, a, a_col, scratch)
/// class      derived
/// mutation   out-of-place
/// definition res[res_col] = canon([[a[a_col]]]_base2k * 2^k, base2k, res.size() * base2k, res.size()); the other columns of res are untouched
/// domain     res, a: VecZnx or windows of equal visible degree, read at base2k; base2k in 1..=62; every digit of a in [-2^62, 2^62]
/// requires   scratch >= vec_znx_lsh_tmp_bytes(res.size())
/// ensures    res[res_col] is canonical at radix base2k and precision res.size() * base2k and congruent modulo 1 to rnd([[a[a_col]]]_base2k * 2^k, res.size() * base2k)
/// fallback   default body: normalization with offset = +k
/// override   allowed, with vec_znx_lsh_tmp_bytes
/// test       test_vec_znx_lsh, test_vec_znx_lsh_derived
/// ```
pub trait VecZnxLsh<B: Backend> {
    /// Writes `a[a_col]` shifted up by `k` bits and normalized at `base2k` into `res[res_col]`.
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

/// Accumulation of a left-shifted column into a destination.
///
/// ```text
/// op         vec_znx_lsh_add(base2k, k, res, res_col, a, a_col, scratch)
/// class      derived
/// mutation   accumulate
/// definition res[res_col,j] = old(res)[res_col,j] + canon([[a[a_col]]]_base2k * 2^k, base2k, res.size() * base2k, res.size())[j]; the other columns of res are untouched
/// domain     res, a: VecZnx or windows of one, read at base2k; base2k in 1..=62; every digit of a in [-2^62, 2^62]
/// requires   scratch >= vec_znx_lsh_tmp_bytes(res.size())
/// ensures    res[res_col] gains the canonical vec_znx_lsh of a[a_col] on a res.size()-limb destination; the result is not renormalized
/// fallback   default body: shift into a res.size()-limb temporary of a's degree, then add_assign
/// override   allowed, with vec_znx_lsh_tmp_bytes
/// sparse     a is the sparse-capable slot, as for vec_znx_add_assign
/// test       test_vec_znx_lsh_add_derived, test_vec_znx_sparse_add_sub
/// ```
pub trait VecZnxLshAdd<B: Backend> {
    /// Adds `a[a_col]`, shifted up by `k` bits and normalized at `base2k`, into `res[res_col]`.
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

/// Scratch requirement of the right-shift operations.
///
/// ```text
/// op         vec_znx_rsh_tmp_bytes(res_size)
/// class      support
/// mutation   none
/// domain     res_size: the destination's limb count
/// ensures    returns sufficient scratch bytes for vec_znx_rsh, vec_znx_rsh_add, vec_znx_rsh_sub and vec_znx_rsh_assign on a res_size-limb destination
/// test       test_vec_znx_rsh
/// ```
pub trait VecZnxRshTmpBytes {
    /// Returns the scratch byte count the right-shift operations require on a destination of `res_size` limbs.
    fn vec_znx_rsh_tmp_bytes(&self, res_size: usize) -> usize;
}

/// Right shift of a column by a power of two, normalized into the destination.
///
/// ```text
/// op         vec_znx_rsh(base2k, k, res, res_col, a, a_col, scratch)
/// class      derived
/// mutation   out-of-place
/// definition res[res_col] = canon([[a[a_col]]]_base2k / 2^k, base2k, res.size() * base2k, res.size()); the other columns of res are untouched
/// domain     res, a: VecZnx or windows of equal visible degree, read at base2k; base2k in 1..=62; every digit of a in [-2^62, 2^62]
/// requires   scratch >= vec_znx_rsh_tmp_bytes(res.size())
/// ensures    res[res_col] is canonical at radix base2k and precision res.size() * base2k and congruent modulo 1 to rnd([[a[a_col]]]_base2k / 2^k, res.size() * base2k)
/// fallback   default body: normalization with offset = -k
/// override   allowed, with vec_znx_rsh_tmp_bytes
/// test       test_vec_znx_rsh, test_vec_znx_rsh_derived
/// ```
pub trait VecZnxRsh<B: Backend> {
    /// Writes `a[a_col]` shifted down by `k` bits and normalized at `base2k` into `res[res_col]`.
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

/// Accumulation of a right-shifted column into a destination.
///
/// ```text
/// op         vec_znx_rsh_add(base2k, k, res, res_col, a, a_col, scratch)
/// class      derived
/// mutation   accumulate
/// definition res[res_col,j] = old(res)[res_col,j] + canon([[a[a_col]]]_base2k / 2^k, base2k, res.size() * base2k, res.size())[j]; the other columns of res are untouched
/// domain     res, a: VecZnx or windows of one, read at base2k; base2k in 1..=62; every digit of a in [-2^62, 2^62]
/// requires   scratch >= vec_znx_rsh_tmp_bytes(res.size())
/// ensures    res[res_col] gains the canonical vec_znx_rsh of a[a_col] on a res.size()-limb destination; the result is not renormalized
/// fallback   default body: shift into a res.size()-limb temporary of a's degree, then add_assign
/// override   allowed, with vec_znx_rsh_tmp_bytes
/// sparse     a is the sparse-capable slot, as for vec_znx_add_assign
/// test       test_vec_znx_rsh_add_derived, test_vec_znx_sparse_add_sub
/// ```
pub trait VecZnxRshAdd<B: Backend> {
    /// Adds `a[a_col]`, shifted down by `k` bits and normalized at `base2k`, into `res[res_col]`.
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

/// Subtraction of a left-shifted column from a destination.
///
/// ```text
/// op         vec_znx_lsh_sub(base2k, k, res, res_col, a, a_col, scratch)
/// class      derived
/// mutation   accumulate
/// definition res[res_col,j] = old(res)[res_col,j] - canon([[a[a_col]]]_base2k * 2^k, base2k, res.size() * base2k, res.size())[j]; the other columns of res are untouched
/// domain     res, a: VecZnx or windows of one, read at base2k; base2k in 1..=62; every digit of a in [-2^62, 2^62]
/// requires   scratch >= vec_znx_lsh_tmp_bytes(res.size())
/// ensures    res[res_col] loses the canonical vec_znx_lsh of a[a_col] on a res.size()-limb destination; the result is not renormalized
/// fallback   default body: shift into a res.size()-limb temporary of a's degree, then sub_assign
/// override   allowed, with vec_znx_lsh_tmp_bytes
/// sparse     a is the sparse-capable slot, as for vec_znx_sub_assign
/// test       test_vec_znx_lsh_sub_derived, test_vec_znx_sparse_add_sub
/// ```
pub trait VecZnxLshSub<B: Backend> {
    /// Subtracts `a[a_col]`, shifted up by `k` bits and normalized at `base2k`, from `res[res_col]`.
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

/// Subtraction of a right-shifted column from a destination.
///
/// ```text
/// op         vec_znx_rsh_sub(base2k, k, res, res_col, a, a_col, scratch)
/// class      derived
/// mutation   accumulate
/// definition res[res_col,j] = old(res)[res_col,j] - canon([[a[a_col]]]_base2k / 2^k, base2k, res.size() * base2k, res.size())[j]; the other columns of res are untouched
/// domain     res, a: VecZnx or windows of one, read at base2k; base2k in 1..=62; every digit of a in [-2^62, 2^62]
/// requires   scratch >= vec_znx_rsh_tmp_bytes(res.size())
/// ensures    res[res_col] loses the canonical vec_znx_rsh of a[a_col] on a res.size()-limb destination; the result is not renormalized
/// fallback   default body: shift into a res.size()-limb temporary of a's degree, then sub_assign
/// override   allowed, with vec_znx_rsh_tmp_bytes
/// sparse     a is the sparse-capable slot, as for vec_znx_sub_assign
/// test       test_vec_znx_rsh_sub_derived, test_vec_znx_sparse_add_sub
/// ```
pub trait VecZnxRshSub<B: Backend> {
    /// Subtracts `a[a_col]`, shifted down by `k` bits and normalized at `base2k`, from `res[res_col]`.
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

/// In-place left shift of a column by a power of two, normalized at its own width.
///
/// ```text
/// op         vec_znx_lsh_assign(base2k, k, a, a_col, scratch)
/// class      derived
/// mutation   in-place
/// definition a[a_col] = canon([[old(a)[a_col]]]_base2k * 2^k, base2k, a.size() * base2k, a.size()); the other columns of a are untouched
/// domain     a: a VecZnx or a window of one, read at base2k; base2k in 1..=62; every input digit in [-2^62, 2^62]
/// requires   scratch >= vec_znx_lsh_tmp_bytes(a.size())
/// ensures    a[a_col] is canonical at radix base2k and precision a.size() * base2k and congruent modulo 1 to rnd([[old(a)[a_col]]]_base2k * 2^k, a.size() * base2k); the other columns are untouched
/// fallback   default body: shift into an a.size()-limb temporary, then copy back
/// override   allowed, with vec_znx_lsh_tmp_bytes
/// test       test_vec_znx_lsh_assign, test_vec_znx_lsh_assign_derived
/// ```
pub trait VecZnxLshAssign<B: Backend> {
    /// Replaces `a[a_col]` with its value shifted up by `k` bits and normalized at `base2k`.
    fn vec_znx_lsh_assign(
        &self,
        base2k: usize,
        k: usize,
        a: &mut VecZnxBackendMut<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// In-place right shift of a column by a power of two, normalized at its own width.
///
/// ```text
/// op         vec_znx_rsh_assign(base2k, k, a, a_col, scratch)
/// class      derived
/// mutation   in-place
/// definition a[a_col] = canon([[old(a)[a_col]]]_base2k / 2^k, base2k, a.size() * base2k, a.size()); the other columns of a are untouched
/// domain     a: a VecZnx or a window of one, read at base2k; base2k in 1..=62; every input digit in [-2^62, 2^62]
/// requires   scratch >= vec_znx_rsh_tmp_bytes(a.size())
/// ensures    a[a_col] is canonical at radix base2k and precision a.size() * base2k and congruent modulo 1 to rnd([[old(a)[a_col]]]_base2k / 2^k, a.size() * base2k); the other columns are untouched
/// fallback   default body: in-place normalization with offset = -k
/// override   allowed, with vec_znx_rsh_tmp_bytes
/// test       test_vec_znx_rsh_assign, test_vec_znx_rsh_assign_derived
/// ```
pub trait VecZnxRshAssign<B: Backend> {
    /// Replaces `a[a_col]` with its value shifted down by `k` bits and normalized at `base2k`.
    fn vec_znx_rsh_assign(
        &self,
        base2k: usize,
        k: usize,
        a: &mut VecZnxBackendMut<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Multiplication of a column by a power of `X` in the ring.
///
/// ```text
/// op         vec_znx_rotate(p, res, res_col, a, a_col)
/// class      basis
/// mutation   out-of-place
/// definition res[res_col,j] = X^p * a[a_col,j] in R_N; the other columns of res are untouched
/// domain     res, a: dense VecZnx of degree N; windows are rejected
/// ensures    res[res_col] = X^p * a[a_col] in Z[X]/(X^N + 1), limb by limb, p taken modulo 2N; limbs of res past a.size() are zero
/// test       test_vec_znx_rotate
/// ```
pub trait VecZnxRotate<B: Backend> {
    /// Writes `X^p` times `a[a_col]` into `res[res_col]`.
    fn vec_znx_rotate(
        &self,
        p: i64,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    );
}

/// Scratch requirement of the in-place rotation.
///
/// ```text
/// op         vec_znx_rotate_assign_tmp_bytes()
/// class      support
/// mutation   none
/// domain     -
/// ensures    returns the scratch bytes vec_znx_rotate_assign needs, one ring element wide and independent of the limb count
/// test       test_vec_znx_rotate_assign
/// ```
pub trait VecZnxRotateAssignTmpBytes {
    /// Returns the scratch byte count the in-place rotation requires.
    fn vec_znx_rotate_assign_tmp_bytes(&self) -> usize;
}

/// In-place multiplication of a column by a power of `X` in the ring.
///
/// ```text
/// op         vec_znx_rotate_assign(p, a, a_col, scratch)
/// class      variant
/// mutation   in-place
/// definition a[a_col,j] = X^p * old(a)[a_col,j] in R_N; the other columns of a are untouched
/// domain     a: a dense VecZnx of degree N
/// requires   scratch >= vec_znx_rotate_assign_tmp_bytes()
/// ensures    each limb of a[a_col] is X^p times its pre-call value in R_N; the other columns are untouched
/// fallback   none; required
/// override   required
/// test       test_vec_znx_rotate_assign
/// ```
pub trait VecZnxRotateAssign<B: Backend> {
    /// Multiplies every limb of `a[a_col]` by `X^p` in place.
    fn vec_znx_rotate_assign(&self, p: i64, a: &mut VecZnxBackendMut<'_, B>, a_col: usize, scratch: &mut ScratchArena<'_, B>);
}

/// Substitution `X -> X^k` on a column.
///
/// ```text
/// op         vec_znx_automorphism(k, res, res_col, a, a_col)
/// class      basis
/// mutation   out-of-place
/// definition res[res_col,j] = sum_{0 <= i < a.n()} a[a_col,j,i] * X^(k * i) in R_N; the other columns of res are untouched
/// domain     res, a: dense VecZnx of degree N; k odd
/// ensures    each limb of res[res_col] is the image of a[a_col] under X -> X^k; limbs of res past a.size() are zero
/// test       test_vec_znx_automorphism
/// ```
pub trait VecZnxAutomorphism<B: Backend> {
    /// Writes the image of `a[a_col]` under `X -> X^k` into `res[res_col]`.
    fn vec_znx_automorphism(
        &self,
        k: i64,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    );
}

/// Scratch requirement of the in-place automorphism.
///
/// ```text
/// op         vec_znx_automorphism_assign_tmp_bytes()
/// class      support
/// mutation   none
/// domain     -
/// ensures    returns the scratch bytes vec_znx_automorphism_assign needs, one ring element wide and independent of the limb count
/// test       test_vec_znx_automorphism_assign
/// ```
pub trait VecZnxAutomorphismAssignTmpBytes {
    /// Returns the scratch byte count the in-place automorphism requires.
    fn vec_znx_automorphism_assign_tmp_bytes(&self) -> usize;
}

/// In-place substitution `X -> X^k` on a column.
///
/// ```text
/// op         vec_znx_automorphism_assign(k, res, res_col, scratch)
/// class      variant
/// mutation   in-place
/// definition res[res_col,j] = sum_{0 <= i < res.n()} old(res)[res_col,j,i] * X^(k * i) in R_N; the other columns of res are untouched
/// domain     res: a dense VecZnx of degree N; k odd
/// requires   scratch >= vec_znx_automorphism_assign_tmp_bytes()
/// ensures    each limb of res[res_col] is the image of its pre-call value under X -> X^k; the other columns are untouched
/// fallback   none; required
/// override   required
/// test       test_vec_znx_automorphism_assign
/// ```
pub trait VecZnxAutomorphismAssign<B: Backend> {
    /// Replaces `res[res_col]` with its image under `X -> X^k`.
    fn vec_znx_automorphism_assign(
        &self,
        k: i64,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Substitution `X -> X^k` on a column of a scalar vector.
///
/// ```text
/// op         scalar_znx_automorphism(k, res, res_col, a, a_col)
/// class      variant
/// mutation   out-of-place
/// definition res[res_col] = sum_{0 <= i < a.n()} a[a_col,0,i] * X^(k * i) in R_N; the other columns of res are untouched
/// domain     res, a: ScalarZnx of degree N; k odd
/// ensures    res[res_col] is the image of a[a_col] under X -> X^k; the other columns are untouched
/// test       test_scalar_znx_automorphism
/// ```
pub trait ScalarZnxAutomorphism<B: Backend> {
    /// Writes the image of `a[a_col]` under `X -> X^k` into `res[res_col]`.
    fn scalar_znx_automorphism(
        &self,
        k: i64,
        res: &mut ScalarZnxBackendMut<'_, B>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, B>,
        a_col: usize,
    );
}

/// Multiplication of a column by `X^p - 1` in the ring.
///
/// ```text
/// op         vec_znx_mul_xp_minus_one(p, res, res_col, a, a_col)
/// class      derived
/// mutation   out-of-place
/// definition res[res_col,j] = (X^p - 1) * a[a_col,j] in R_N; the other columns of res are untouched
/// domain     res, a: dense VecZnx of degree N
/// ensures    res[res_col] = (X^p - 1) * a[a_col] in Z[X]/(X^N + 1), limb by limb; the digits are not renormalized
/// fallback   default body: rotate into res, then subtract the unrotated operand in place
/// override   allowed, scratch-free
/// test       test_vec_znx_mul_xp_minus_one, test_vec_znx_mul_xp_minus_one_derived
/// ```
pub trait VecZnxMulXpMinusOne<B: Backend> {
    /// Writes `X^p - 1` times `a[a_col]` into `res[res_col]`.
    fn vec_znx_mul_xp_minus_one(
        &self,
        p: i64,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    );
}

/// Scratch requirement of the in-place multiplication by `X^p - 1`.
///
/// ```text
/// op         vec_znx_mul_xp_minus_one_assign_tmp_bytes(size)
/// class      support
/// mutation   none
/// domain     size: the destination's limb count
/// ensures    returns sufficient scratch bytes for vec_znx_mul_xp_minus_one_assign on a size-limb destination
/// test       test_vec_znx_mul_xp_minus_one_assign
/// ```
pub trait VecZnxMulXpMinusOneAssignTmpBytes {
    /// Returns the scratch byte count the in-place multiplication by `X^p - 1` requires on a destination of `size` limbs.
    fn vec_znx_mul_xp_minus_one_assign_tmp_bytes(&self, size: usize) -> usize;
}

/// In-place multiplication of a column by `X^p - 1` in the ring.
///
/// ```text
/// op         vec_znx_mul_xp_minus_one_assign(p, res, res_col, scratch)
/// class      derived
/// mutation   in-place
/// definition res[res_col,j] = (X^p - 1) * old(res)[res_col,j] in R_N; the other columns of res are untouched
/// domain     res: a dense VecZnx of degree N
/// requires   scratch >= vec_znx_mul_xp_minus_one_assign_tmp_bytes(res.size())
/// ensures    each limb of res[res_col] is (X^p - 1) times its pre-call value in R_N; the other columns are untouched; the digits are not renormalized
/// fallback   default body: the product into a res.size()-limb temporary, then copy back
/// override   allowed, with vec_znx_mul_xp_minus_one_assign_tmp_bytes
/// test       test_vec_znx_mul_xp_minus_one_assign, test_vec_znx_mul_xp_minus_one_assign_derived
/// ```
pub trait VecZnxMulXpMinusOneAssign<B: Backend> {
    /// Multiplies every limb of `res[res_col]` by `X^p - 1` in place.
    fn vec_znx_mul_xp_minus_one_assign(
        &self,
        p: i64,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Ring degree switch of a column, by coefficient insertion or selection.
///
/// ```text
/// op         vec_znx_switch_ring(res, res_col, a, a_col)
/// class      basis
/// mutation   out-of-place
/// definition for 0 <= i < res.n(), res[res_col,j,i] = a[a_col,j,i * (a.n() / res.n())] if a.n() >= res.n(); otherwise res[res_col,j,i] = a[a_col,j,i / (res.n() / a.n())] if i mod (res.n() / a.n()) = 0, and 0 otherwise; the other columns of res are untouched
/// domain     res, a: dense VecZnx whose degrees divide one another
/// ensures    growing inserts zero coefficients between the coefficients of a[a_col]; shrinking retains only source coefficients at multiples of the degree ratio; limbs from a.size() onward are zero; the other columns are untouched
/// test       test_vec_znx_switch_ring, test_vec_znx_switch_ring_matches_wrapper
/// ```
pub trait VecZnxSwitchRing<B: Backend> {
    /// Writes `a[a_col]` at the ring degree of `res` into `res[res_col]`.
    fn vec_znx_switch_ring(&self, res: &mut VecZnxBackendMut<'_, B>, res_col: usize, a: &VecZnxBackendRef<'_, B>, a_col: usize);
}

/// Copy of one column into another.
///
/// ```text
/// op         vec_znx_copy(res, res_col, a, a_col)
/// class      basis
/// mutation   out-of-place
/// definition res[res_col,j] = a[a_col,j]; the other columns of res are untouched
/// domain     res, a: VecZnx or windows of one, of equal degree
/// ensures    res[res_col] holds a[a_col] limb by limb; limbs of res past a.size() are zero
/// test       test_vec_znx_copy, test_vec_znx_copy_matches_wrapper, test_vec_znx_window_ops
/// ```
pub trait VecZnxCopy<B: Backend> {
    /// Writes `a[a_col]` into `res[res_col]`.
    fn vec_znx_copy(&self, res: &mut VecZnxBackendMut<'_, B>, res_col: usize, a: &VecZnxBackendRef<'_, B>, a_col: usize);
}

/// Filling of a column with a uniform torus value drawn from a pseudorandom stream.
///
/// ```text
/// op         vec_znx_fill_uniform_source(base2k, k, res, res_col, source)
/// class      basis
/// mutation   out-of-place
/// definition for 0 <= i < res.n(), res[res_col,j,i] = 0 if j >= live(k, base2k), 2^pad(k, base2k) * floor((draw(reseed(source), j * res.n() + i, base2k) - 2^(base2k - 1)) / 2^pad(k, base2k)) if j = live(k, base2k) - 1, and draw(reseed(source), j * res.n() + i, base2k) - 2^(base2k - 1) otherwise; the other columns of res are untouched
/// domain     res: a dense VecZnx; base2k in 1..=62; 0 < k <= res.size() * base2k; source: the caller's pseudorandom stream
/// ensures    res[res_col] is uniform over the torus at precision k and canonical at radix base2k; source advances by the 32 bytes of one seed, and reseed(source) by live(k, base2k) * res.n() draws in increasing limb order, then increasing coefficient order
/// test       test_vec_znx_fill_uniform
/// ```
pub trait VecZnxFillUniformSource<B: Backend> {
    /// Writes a uniform torus value of precision `k`, canonical at radix `base2k`, into `res[res_col]`, drawing from `source`.
    fn vec_znx_fill_uniform_source(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        source: &mut Source,
    );
}
