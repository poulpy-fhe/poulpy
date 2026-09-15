use crate::layouts::{
    Backend, ScalarZnxBackendRef, ScratchArena, VecZnxBackendMut, VecZnxBackendRef, VecZnxBigBackendMut, VecZnxBigBackendRef,
    VecZnxBigOwned,
};

/// Converts a coefficient-domain [`VecZnx`](crate::layouts::VecZnx) column
/// into a [`VecZnxBig`](crate::layouts::VecZnxBig) column.
///
/// ```text
/// op         vec_znx_big_from_small(res, res_col, a, a_col)
/// class      basis
/// mutation   out-of-place
/// domain     res: a VecZnxBig or a window of one; a: a VecZnx or a window of one, of the same degree
/// ensures    res[res_col] holds a[a_col] limb by limb, each coefficient widened to a big word; limbs of res past a.size() are zero, so [[res]] = [[a]] at the shared radix
/// exact      exact
/// test       test_vec_znx_big_from_small, test_vec_znx_big_window_ops
/// ```
pub trait VecZnxBigFromSmall<B: Backend> {
    fn vec_znx_big_from_small(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    );
}

/// Allocates as [crate::layouts::VecZnxBig].
///
/// ```text
/// op         vec_znx_big_alloc(cols, size)
/// class      support
/// mutation   none
/// domain     cols >= 1, size >= 1
/// ensures    returns an owned degree-N VecZnxBig of those dimensions in the backend's memory; its contents are unspecified
/// exact      not an arithmetic operation
/// test       none
/// ```
pub trait VecZnxBigAlloc<B: Backend> {
    fn vec_znx_big_alloc(&self, cols: usize, size: usize) -> VecZnxBigOwned<B>;
}

/// Returns the size in bytes to allocate a [crate::layouts::VecZnxBig].
///
/// ```text
/// op         bytes_of_vec_znx_big(cols, size) / bytes_of_vec_znx_big_n(n, cols, size)
/// class      support
/// mutation   none
/// domain     cols >= 1, size >= 1; the `_n` form takes a degree other than the module's
/// ensures    returns the byte size of such a VecZnxBig in this backend's representation, the amount take_vec_znx_big_scratch carves
/// exact      not an arithmetic operation
/// test       none
/// ```
pub trait VecZnxBigBytesOf {
    fn bytes_of_vec_znx_big(&self, cols: usize, size: usize) -> usize;

    fn bytes_of_vec_znx_big_n(&self, n: usize, cols: usize, size: usize) -> usize;
}

/// ```text
/// op         vec_znx_big_add(res, res_col, a, a_col, b, b_col)
/// class      basis
/// mutation   out-of-place
/// domain     res, a, b: VecZnxBig or windows of one
/// ensures    res[res_col] = a[a_col] + b[b_col] limb by limb; operands shorter than res are zero-extended and every limb of res is written. Big words wrap, so the caller keeps the sum inside the backend's big-word bound
/// sparse     a and b are the sparse-capable slots: a degree-n operand, n dividing N, stands for switch_ring_{n->N} of itself (4.5). The kernels read it with a stride (#266)
/// exact      exact
/// test       test_vec_znx_big_add, test_vec_znx_big_sparse_add_sub
/// ```
pub trait VecZnxBigAdd<B: Backend> {
    /// Adds `a` and `b` into `res`.
    fn vec_znx_big_add(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxBigBackendRef<'_, B>,
        b_col: usize,
    );
}

/// ```text
/// op         vec_znx_big_add_assign(res, res_col, a, a_col)
/// class      variant
/// mutation   in-place
/// definition vec_znx_big_add(res, res_col, res, res_col, a, a_col)
/// domain     res, a: VecZnxBig or windows of one
/// ensures    res[res_col] += a[a_col] limb by limb; limbs of res past a.size() keep their value
/// sparse     a is the sparse-capable slot, as for vec_znx_big_add (4.5)
/// exact      exact
/// test       test_vec_znx_big_add_assign, test_vec_znx_big_sparse_add_sub
/// ```
pub trait VecZnxBigAddAssign<B: Backend> {
    /// Adds `a` into `res`.
    fn vec_znx_big_add_assign(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, B>,
        a_col: usize,
    );
}

/// ```text
/// op         vec_znx_big_add_small(res, res_col, a, a_col, b, b_col)
/// class      derived
/// mutation   out-of-place
/// definition vec_znx_big_add(res, res_col, a, a_col, vec_znx_big_from_small(b, b_col), 0)
/// domain     res, a: VecZnxBig or windows of one; b: a VecZnx or a window of one; res distinct from a and b
/// ensures    res[res_col] = a[a_col] + b[b_col]: limbs only b reaches hold b, limbs only a reaches hold a, limbs neither reaches are zero
/// sparse     b is the sparse-capable slot, as for vec_znx_big_add (4.5)
/// fallback   OEP default body: promote b into res with vec_znx_big_from_small, then fold a in with add_assign. Scratch-free, and it writes res before reading the operands, so res must not alias them
/// override   allowed, scratch-free; the CPU families override it with a fused kernel, which also tolerates aliasing
/// exact      exact
/// test       test_vec_znx_big_add_small, test_vec_znx_big_add_small_derived, test_vec_znx_big_sparse_add_sub
/// ```
pub trait VecZnxBigAddSmall<B: Backend> {
    /// Adds `a` and the coefficient-domain `b` into `res`.
    fn vec_znx_big_add_small(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, B>,
        b_col: usize,
    );
}

/// ```text
/// op         vec_znx_big_add_small_assign(res, res_col, a, a_col)
/// class      variant
/// mutation   in-place
/// definition vec_znx_big_add_small(res, res_col, res, res_col, a, a_col)
/// domain     res: a VecZnxBig or a window of one; a: a VecZnx or a window of one
/// ensures    res[res_col] += a[a_col] limb by limb; limbs of res past a.size() keep their value
/// sparse     a is the sparse-capable slot, as for vec_znx_big_add (4.5)
/// exact      exact
/// test       test_vec_znx_big_add_small_assign, test_vec_znx_big_sparse_add_sub
/// ```
pub trait VecZnxBigAddSmallAssign<B: Backend> {
    /// Adds the coefficient-domain `a` into `res`.
    fn vec_znx_big_add_small_assign(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    );
}

/// ```text
/// op         vec_znx_big_sub(res, res_col, a, a_col, b, b_col)
/// class      basis
/// mutation   out-of-place
/// domain     res, a, b: VecZnxBig or windows of one
/// ensures    res[res_col] = a[a_col] - b[b_col] limb by limb; operands shorter than res are zero-extended and every limb of res is written
/// sparse     a and b are the sparse-capable slots, as for vec_znx_big_add (4.5)
/// exact      exact
/// test       test_vec_znx_big_sub, test_vec_znx_big_sparse_add_sub
/// ```
pub trait VecZnxBigSub<B: Backend> {
    /// Subtracts `b` from `a` into `res`.
    fn vec_znx_big_sub(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxBigBackendRef<'_, B>,
        b_col: usize,
    );
}

/// ```text
/// op         vec_znx_big_sub_assign(res, res_col, a, a_col)
/// class      variant
/// mutation   in-place
/// definition vec_znx_big_sub(res, res_col, res, res_col, a, a_col)
/// domain     res, a: VecZnxBig or windows of one
/// ensures    res[res_col] -= a[a_col] limb by limb; limbs of res past a.size() keep their value
/// sparse     a is the sparse-capable slot, as for vec_znx_big_add (4.5)
/// exact      exact
/// test       test_vec_znx_big_sub_assign, test_vec_znx_big_sparse_add_sub
/// ```
pub trait VecZnxBigSubAssign<B: Backend> {
    /// Subtracts `a` from `res`.
    fn vec_znx_big_sub_assign(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, B>,
        a_col: usize,
    );
}

/// ```text
/// op         vec_znx_big_sub_negate_assign(res, res_col, a, a_col)
/// class      variant
/// mutation   in-place
/// definition vec_znx_big_sub(res, res_col, a, a_col, res, res_col)
/// domain     res, a: VecZnxBig or windows of one
/// ensures    res[res_col] = a[a_col] - res[res_col] limb by limb; limbs of res past a.size() are negated in place
/// sparse     a is the sparse-capable slot, as for vec_znx_big_add (4.5)
/// exact      exact
/// test       test_vec_znx_big_sub_negate_assign, test_vec_znx_big_sparse_add_sub
/// ```
pub trait VecZnxBigSubNegateAssign<B: Backend> {
    /// Subtracts `res` from `a` into `res`.
    fn vec_znx_big_sub_negate_assign(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, B>,
        a_col: usize,
    );
}

/// ```text
/// op         vec_znx_big_sub_small_a(res, res_col, a, a_col, b, b_col)
/// class      derived
/// mutation   out-of-place
/// definition vec_znx_big_sub(res, res_col, vec_znx_big_from_small(a, a_col), 0, b, b_col)
/// domain     res, b: VecZnxBig or windows of one; a: a VecZnx or a window of one, the coefficient-domain operand; res distinct from a and b
/// ensures    res[res_col] = a[a_col] - b[b_col]: limbs only a reaches hold a, limbs only b reaches hold -b, limbs neither reaches are zero
/// sparse     a is the sparse-capable slot, as for vec_znx_big_add (4.5)
/// fallback   OEP default body: promote a into res with vec_znx_big_from_small, then subtract b in place. Scratch-free, and it writes res before reading the operands, so res must not alias them
/// override   allowed, scratch-free; the CPU families override it with a fused kernel, which also tolerates aliasing
/// exact      exact
/// test       test_vec_znx_big_sub_small_a, test_vec_znx_big_sub_small_a_derived, test_vec_znx_big_sparse_add_sub
/// ```
pub trait VecZnxBigSubSmallA<B: Backend> {
    /// Subtracts `b` from the coefficient-domain `a` into `res`.
    fn vec_znx_big_sub_small_a(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxBigBackendRef<'_, B>,
        b_col: usize,
    );
}

/// ```text
/// op         vec_znx_big_sub_small_assign(res, res_col, a, a_col)
/// class      variant
/// mutation   in-place
/// definition vec_znx_big_sub_small_b(res, res_col, res, res_col, a, a_col): res is the big operand, a the small one
/// domain     res: a VecZnxBig or a window of one; a: a VecZnx or a window of one
/// ensures    res[res_col] -= a[a_col] limb by limb; limbs of res past a.size() keep their value
/// sparse     a is the sparse-capable slot, as for vec_znx_big_add (4.5)
/// exact      exact
/// test       test_vec_znx_big_sub_small_a_assign, test_vec_znx_big_sparse_add_sub
/// ```
pub trait VecZnxBigSubSmallAssign<B: Backend> {
    /// Subtracts the coefficient-domain `a` from `res`.
    fn vec_znx_big_sub_small_assign(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    );
}

/// ```text
/// op         vec_znx_big_sub_small_b(res, res_col, a, a_col, b, b_col)
/// class      derived
/// mutation   out-of-place
/// definition vec_znx_big_sub(res, res_col, a, a_col, vec_znx_big_from_small(b, b_col), 0)
/// domain     res, a: VecZnxBig or windows of one; b: a VecZnx or a window of one, the coefficient-domain operand; res distinct from a and b
/// ensures    res[res_col] = a[a_col] - b[b_col]: limbs only a reaches hold a, limbs only b reaches hold -b, limbs neither reaches are zero
/// sparse     b is the sparse-capable slot, as for vec_znx_big_add (4.5)
/// fallback   OEP default body: promote b into res with vec_znx_big_from_small, then negate it against a with sub_negate_assign. Scratch-free, and it writes res before reading the operands, so res must not alias them
/// override   allowed, scratch-free; the CPU families override it with a fused kernel, which also tolerates aliasing
/// exact      exact
/// test       test_vec_znx_big_sub_small_b, test_vec_znx_big_sub_small_b_derived, test_vec_znx_big_sparse_add_sub
/// ```
pub trait VecZnxBigSubSmallB<B: Backend> {
    /// Subtracts the coefficient-domain `b` from `a` into `res`.
    fn vec_znx_big_sub_small_b(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, B>,
        b_col: usize,
    );
}

/// ```text
/// op         vec_znx_big_sub_small_negate_assign(res, res_col, a, a_col)
/// class      variant
/// mutation   in-place
/// definition vec_znx_big_sub_small_a(res, res_col, a, a_col, res, res_col): a is the small operand, res the big one
/// domain     res: a VecZnxBig or a window of one; a: a VecZnx or a window of one
/// ensures    res[res_col] = a[a_col] - res[res_col] limb by limb; limbs of res past a.size() are negated in place
/// sparse     a is the sparse-capable slot, as for vec_znx_big_add (4.5)
/// exact      exact
/// test       test_vec_znx_big_sub_small_b_assign, test_vec_znx_big_sparse_add_sub
/// ```
pub trait VecZnxBigSubSmallNegateAssign<B: Backend> {
    /// Subtracts `res` from the coefficient-domain `a` into `res`.
    fn vec_znx_big_sub_small_negate_assign(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    );
}

/// Sums coefficients from a selected [`VecZnxBig`](crate::layouts::VecZnxBig)
/// column and stores each limb's result in one destination coefficient.
///
/// ```text
/// op         vec_znx_big_inner_sum(res, res_col, res_coeff, a, a_col)
/// class      basis
/// mutation   out-of-place
/// domain     res, a: dense VecZnxBig of any degree, the module's does not enter; res_coeff < res.n(); res.size() <= a.size()
/// ensures    for every limb of res, coefficient res_coeff of res[res_col] is the wrapping sum of all n coefficients of that limb of a[a_col]; every other coefficient of res is untouched. This is the trace-like reduction the LWE path uses, not a ring operation
/// exact      exact modulo the big word, which wraps
/// test       test_vec_znx_big_inner_sum
/// ```
pub trait VecZnxBigInnerSum<B: Backend> {
    fn vec_znx_big_inner_sum(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        res_coeff: usize,
        a: &VecZnxBigBackendRef<'_, B>,
        a_col: usize,
    );
}

/// Computes a coefficient-wise linear combination of [`VecZnx`](crate::layouts::VecZnx)
/// columns with scalar weights:
/// `res[res_col][k] = sum_{j < cols} a[j][k] * weights[weights_col][j]`.
#[allow(clippy::too_many_arguments)]
/// ```text
/// op         vec_znx_big_col_weighted_sum(res, res_col, a, weights, weights_col, cols, coeffs)
/// class      basis
/// mutation   out-of-place
/// domain     res: a dense VecZnxBig; a: a dense VecZnx; both of any degree, the module's does not enter; weights: a ScalarZnx with cols <= weights.n() and weights_col < weights.cols(); coeffs <= min(a.n(), res.n()); res.size() <= a.size()
/// ensures    res[res_col][limb][k] = sum over col < cols of a[col][limb][k] * weights[weights_col][0][col], for every k < coeffs and every limb of res; the coefficients from coeffs on are zero. Coefficient-wise, not a ring product
/// exact      exact modulo the big word, which wraps
/// test       test_vec_znx_big_col_weighted_sum
/// ```
pub trait VecZnxBigColWeightedSum<B: Backend> {
    fn vec_znx_big_col_weighted_sum<'r, 'a, 'b>(
        &self,
        res: &mut VecZnxBigBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        weights: &ScalarZnxBackendRef<'b, B>,
        weights_col: usize,
        cols: usize,
        coeffs: usize,
    );
}

/// Computes the element-wise (Hadamard) product `res[k] = a[k] * b[k]` for all `k`
/// and stores each product as a [`BigWord`](Backend::BigWord) value in `res`.
/// Use [`VecZnxBigInnerSum`] afterwards to reduce to a single scalar.
///
/// ```text
/// op         vec_znx_scalar_product(res, res_col, a, a_col, b, b_col)
/// class      basis
/// mutation   out-of-place
/// domain     res: a dense VecZnxBig with res.n() >= a.n() and res.size() <= a.size(); a: a dense VecZnx of any degree, the module's does not enter; b: a ScalarZnx of a's degree
/// ensures    res[res_col][limb][k] = a[a_col][limb][k] * b[b_col][0][k] for every k < a.n() and every limb of res, the coefficient-wise Hadamard product widened to big words. Follow with vec_znx_big_inner_sum to reduce it to one coefficient
/// exact      exact modulo the big word, which wraps
/// test       test_vec_znx_scalar_product
/// ```
pub trait VecZnxScalarProduct<B: Backend> {
    fn vec_znx_scalar_product(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        b: &ScalarZnxBackendRef<'_, B>,
        b_col: usize,
    );
}

/// Negates the selected column of `a` and stores the result in `res`.
///
/// ```text
/// op         vec_znx_big_negate(res, res_col, a, a_col)
/// class      basis
/// mutation   out-of-place
/// domain     res, a: VecZnxBig or windows of one
/// ensures    res[res_col] = -a[a_col] limb by limb; limbs of res past a.size() are zero
/// exact      exact
/// test       test_vec_znx_big_negate
/// ```
pub trait VecZnxBigNegate<B: Backend> {
    fn vec_znx_big_negate(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, B>,
        a_col: usize,
    );
}

/// Negates the selected column of `res` in-place.
///
/// ```text
/// op         vec_znx_big_negate_assign(res, res_col)
/// class      variant
/// mutation   in-place
/// definition vec_znx_big_negate(res, res_col, res, res_col)
/// domain     res: a VecZnxBig or a window of one
/// ensures    every limb of res[res_col] is negated in place
/// exact      exact
/// test       test_vec_znx_big_negate_assign
/// ```
pub trait VecZnxBigNegateAssign<B: Backend> {
    fn vec_znx_big_negate_assign(&self, res: &mut VecZnxBigBackendMut<'_, B>, res_col: usize);
}

/// Returns scratch bytes required for [`VecZnxBigNormalize`].
///
/// ```text
/// op         vec_znx_big_normalize_tmp_bytes()
/// class      support
/// mutation   none
/// domain     -
/// ensures    returns the scratch bytes vec_znx_big_normalize needs; the carry chain is one ring element wide, so the answer does not depend on the operand sizes
/// exact      not an arithmetic operation
/// test       test_vec_znx_big_normalize
/// ```
pub trait VecZnxBigNormalizeTmpBytes {
    fn vec_znx_big_normalize_tmp_bytes(&self) -> usize;
}

#[allow(clippy::too_many_arguments)]
/// Normalizes a [`VecZnxBig`](crate::layouts::VecZnxBig) into a coefficient-domain
/// [`VecZnx`](crate::layouts::VecZnx) with the target base, precision, and offset.
///
/// For i64 big words, the input and radix bounds of [`super::VecZnxNormalize`]
/// apply. For i128 big words, input coefficients must lie in `[-2^126, 2^126]`,
/// with input radix width in `1..=127` and output radix width in `1..=64`.
/// NTT4x30 IDFT coefficients have the tighter bound `abs(a) < 2^119`.
/// Additions before normalization must preserve the applicable coefficient
/// bound. These are caller preconditions and are not checked by an input scan.
///
/// ```text
/// op         vec_znx_big_normalize(res, res_base2k, res_k, res_offset, res_col, a, a_base2k, a_col, scratch)
/// class      basis
/// mutation   out-of-place
/// domain     res: a VecZnx or a window of one; a: a VecZnxBig or a window of one; the input and radix bounds stated above, which depend on the backend's big word
/// requires   scratch >= vec_znx_big_normalize_tmp_bytes()
/// ensures    [[res]]_res_base2k = [[a]]_a_base2k * 2^res_offset, rounded once at precision res_k and canonical there; every limb of res[res_col] is written and the limbs past res_k are zero. This is the one operation that leaves the big domain
/// exact      exact
/// test       test_vec_znx_big_normalize, test_vec_znx_big_window_normalize
/// ```
pub trait VecZnxBigNormalize<B: Backend> {
    fn vec_znx_big_normalize(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_base2k: usize,
        res_k: usize,
        res_offset: i64,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, B>,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Returns scratch bytes required for in-place automorphism on [`VecZnxBig`](crate::layouts::VecZnxBig).
///
/// ```text
/// op         vec_znx_big_automorphism_assign_tmp_bytes()
/// class      support
/// mutation   none
/// domain     -
/// ensures    returns the scratch bytes vec_znx_big_automorphism_assign needs, one ring element wide and independent of the limb count
/// exact      not an arithmetic operation
/// test       test_vec_znx_big_automorphism_assign
/// ```
pub trait VecZnxBigAutomorphismAssignTmpBytes {
    fn vec_znx_big_automorphism_assign_tmp_bytes(&self) -> usize;
}

/// ```text
/// op         vec_znx_big_automorphism(p, res, res_col, a, a_col)
/// class      basis
/// mutation   out-of-place
/// domain     res, a: dense VecZnxBig of the module degree; p odd
/// ensures    res[res_col] = tau_p(a[a_col]), the ring automorphism X -> X^p, limb by limb; limbs of res past a.size() are zero
/// exact      exact
/// test       test_vec_znx_big_automorphism
/// ```
pub trait VecZnxBigAutomorphism<B: Backend> {
    /// Applies the automorphism X -> X^p on `a` and stores the result in `res`.
    fn vec_znx_big_automorphism(
        &self,
        p: i64,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, B>,
        a_col: usize,
    );
}

/// ```text
/// op         vec_znx_big_automorphism_assign(p, res, res_col, scratch)
/// class      variant
/// mutation   in-place
/// definition vec_znx_big_automorphism(p, res, res_col, res, res_col)
/// domain     res: a dense VecZnxBig of the module degree; p odd
/// requires   scratch >= vec_znx_big_automorphism_assign_tmp_bytes()
/// ensures    res[res_col] = tau_p(res[res_col]); the other columns are untouched
/// fallback   none, the kernel is required of every backend
/// override   required
/// exact      exact
/// test       test_vec_znx_big_automorphism_assign
/// ```
pub trait VecZnxBigAutomorphismAssign<B: Backend> {
    /// Applies the automorphism X -> X^p on `res` in place.
    fn vec_znx_big_automorphism_assign(
        &self,
        p: i64,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}
