use crate::layouts::{
    Backend, ScalarZnxBackendRef, ScratchArena, VecZnxBackendMut, VecZnxBackendRef, VecZnxBigBackendMut, VecZnxBigBackendRef,
    VecZnxBigOwned,
};

/// Conversion of a coefficient-domain vector column into a big-word vector column.
///
/// ```text
/// op         vec_znx_big_from_small(res, res_col, a, a_col)
/// class      basis
/// mutation   out-of-place
/// definition res[res_col,j] = a[a_col,j]; other columns of res are unchanged
/// domain     res: a VecZnxBig or a window of one; a: a VecZnx or a window of one, of the same degree or of a power-of-two degree dividing it
/// ensures    the selected column is copied with zero extension or truncation; [[res[res_col]]]_b = [[a[a_col]]]_b for every radix width b when res.size() >= a.size()
/// sparse     a is the sparse-capable slot, as for vec_znx_big_add
/// test       test_vec_znx_big_from_small, test_vec_znx_big_window_ops, test_vec_znx_big_sparse_add_sub
/// ```
pub trait VecZnxBigFromSmall<B: Backend> {
    /// Writes column `a_col` of `a` into column `res_col` of `res`.
    fn vec_znx_big_from_small(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    );
}

/// Allocation of a big-word vector.
///
/// ```text
/// op         vec_znx_big_alloc(cols, size)
/// class      support
/// mutation   none
/// domain     cols >= 1, size >= 1
/// ensures    returns an owned degree-N VecZnxBig of those dimensions in the backend's memory; its contents are unspecified
/// test       none
/// ```
pub trait VecZnxBigAlloc<B: Backend> {
    /// Returns an owned big-word vector with `cols` columns and `size` limbs.
    fn vec_znx_big_alloc(&self, cols: usize, size: usize) -> VecZnxBigOwned<B>;
}

/// Byte size of a big-word vector.
///
/// ```text
/// op         bytes_of_vec_znx_big(cols, size) / bytes_of_vec_znx_big_n(n, cols, size)
/// class      support
/// mutation   none
/// domain     cols >= 1, size >= 1; the `_n` form takes a degree other than the module's
/// ensures    returns the byte size of such a VecZnxBig in this backend's representation, the amount take_vec_znx_big_scratch carves
/// test       none
/// ```
pub trait VecZnxBigBytesOf {
    /// Returns the byte size of a big-word vector of degree N with `cols` columns and `size` limbs.
    fn bytes_of_vec_znx_big(&self, cols: usize, size: usize) -> usize;

    /// Returns the byte size of a degree-`n` big-word vector with `cols` columns and `size` limbs.
    fn bytes_of_vec_znx_big_n(&self, n: usize, cols: usize, size: usize) -> usize;
}

/// Sum of two big-word vectors.
///
/// ```text
/// op         vec_znx_big_add(res, res_col, a, a_col, b, b_col)
/// class      basis
/// mutation   out-of-place
/// definition res[res_col,j] = a[a_col,j] + b[b_col,j]; other columns of res are unchanged
/// domain     res, a, b: VecZnxBig or windows of one; every sum stays inside the big word
/// ensures    the selected output column contains the limbwise sum with zero extension or truncation
/// sparse     a and b are the sparse-capable slots: a dense degree-n operand, n dividing N, stands for switch_ring_{n->N} of itself; a window in these slots has the width of res
/// test       test_vec_znx_big_add, test_vec_znx_big_sparse_add_sub
/// ```
pub trait VecZnxBigAdd<B: Backend> {
    /// Writes `a + b` into `res`.
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

/// In-place sum of a big-word vector into another.
///
/// ```text
/// op         vec_znx_big_add_assign(res, res_col, a, a_col)
/// class      variant
/// mutation   in-place
/// definition res[res_col,j] = old(res)[res_col,j] + a[a_col,j]; limbs a.size() <= j < res.size() and other columns of res are unchanged
/// domain     res, a: VecZnxBig or windows of one
/// ensures    res[res_col] += a[a_col] limb by limb; limbs of res past a.size() keep their value
/// sparse     a is the sparse-capable slot, as for vec_znx_big_add
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

/// Sum of a big-word vector and a coefficient-domain vector.
///
/// ```text
/// op         vec_znx_big_add_small(res, res_col, a, a_col, b, b_col)
/// class      derived
/// mutation   out-of-place
/// definition res[res_col,j] = a[a_col,j] + b[b_col,j]; other columns of res are unchanged
/// domain     res, a: VecZnxBig or windows of one; b: a VecZnx or a window of one
/// ensures    res[res_col] = a[a_col] + b[b_col]: limbs only b reaches hold b, limbs only a reaches hold a, limbs neither reaches are zero
/// sparse     b is the sparse-capable slot, as for vec_znx_big_add
/// fallback   vec_znx_big_from_small(res, res_col, b, b_col); vec_znx_big_add_assign(res, res_col, a, a_col)
/// override   allowed, scratch-free
/// test       test_vec_znx_big_add_small, test_vec_znx_big_add_small_derived, test_vec_znx_big_sparse_add_sub
/// ```
pub trait VecZnxBigAddSmall<B: Backend> {
    /// Writes the sum of `a` and the coefficient-domain `b` into `res`.
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

/// In-place sum of a coefficient-domain vector into a big-word vector.
///
/// ```text
/// op         vec_znx_big_add_small_assign(res, res_col, a, a_col)
/// class      variant
/// mutation   in-place
/// definition res[res_col,j] = old(res)[res_col,j] + a[a_col,j]; limbs a.size() <= j < res.size() and other columns of res are unchanged
/// domain     res: a VecZnxBig or a window of one; a: a VecZnx or a window of one
/// ensures    res[res_col] += a[a_col] limb by limb; limbs of res past a.size() keep their value
/// sparse     a is the sparse-capable slot, as for vec_znx_big_add
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

/// Difference of two big-word vectors.
///
/// ```text
/// op         vec_znx_big_sub(res, res_col, a, a_col, b, b_col)
/// class      basis
/// mutation   out-of-place
/// definition res[res_col,j] = a[a_col,j] - b[b_col,j]; other columns of res are unchanged
/// domain     res, a, b: VecZnxBig or windows of one; every difference stays inside the big word
/// ensures    res[res_col] = a[a_col] - b[b_col] limb by limb; operands shorter than res are zero-extended and every limb of res is written
/// sparse     a and b are the sparse-capable slots, as for vec_znx_big_add
/// test       test_vec_znx_big_sub, test_vec_znx_big_sparse_add_sub
/// ```
pub trait VecZnxBigSub<B: Backend> {
    /// Writes `a - b` into `res`.
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

/// In-place difference of two big-word vectors, the destination as the minuend.
///
/// ```text
/// op         vec_znx_big_sub_assign(res, res_col, a, a_col)
/// class      variant
/// mutation   in-place
/// definition res[res_col,j] = old(res)[res_col,j] - a[a_col,j]; limbs a.size() <= j < res.size() and other columns of res are unchanged
/// domain     res, a: VecZnxBig or windows of one
/// ensures    res[res_col] -= a[a_col] limb by limb; limbs of res past a.size() keep their value
/// sparse     a is the sparse-capable slot, as for vec_znx_big_add
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

/// In-place difference of two big-word vectors, the destination as the subtrahend.
///
/// ```text
/// op         vec_znx_big_sub_negate_assign(res, res_col, a, a_col)
/// class      variant
/// mutation   in-place
/// definition res[res_col,j] = a[a_col,j] - old(res)[res_col,j]; other columns of res are unchanged
/// domain     res, a: VecZnxBig or windows of one
/// ensures    res[res_col] = a[a_col] - res[res_col] limb by limb; limbs of res past a.size() are negated in place
/// sparse     a is the sparse-capable slot, as for vec_znx_big_add
/// test       test_vec_znx_big_sub_negate_assign, test_vec_znx_big_sparse_add_sub
/// ```
pub trait VecZnxBigSubNegateAssign<B: Backend> {
    /// Writes `a - res` into `res`.
    fn vec_znx_big_sub_negate_assign(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, B>,
        a_col: usize,
    );
}

/// Difference of a coefficient-domain vector and a big-word vector.
///
/// ```text
/// op         vec_znx_big_sub_small_a(res, res_col, a, a_col, b, b_col)
/// class      derived
/// mutation   out-of-place
/// definition res[res_col,j] = a[a_col,j] - b[b_col,j]; other columns of res are unchanged
/// domain     res, b: VecZnxBig or windows of one; a: a VecZnx or a window of one, the coefficient-domain operand
/// ensures    res[res_col] = a[a_col] - b[b_col]: limbs only a reaches hold a, limbs only b reaches hold -b, limbs neither reaches are zero
/// sparse     a is the sparse-capable slot, as for vec_znx_big_add
/// fallback   vec_znx_big_from_small(res, res_col, a, a_col); vec_znx_big_sub_assign(res, res_col, b, b_col)
/// override   allowed, scratch-free
/// test       test_vec_znx_big_sub_small_a, test_vec_znx_big_sub_small_a_derived, test_vec_znx_big_sparse_add_sub
/// ```
pub trait VecZnxBigSubSmallA<B: Backend> {
    /// Writes the difference of the coefficient-domain `a` and `b` into `res`.
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

/// In-place difference of a big-word vector and a coefficient-domain vector, the destination as the minuend.
///
/// ```text
/// op         vec_znx_big_sub_small_assign(res, res_col, a, a_col)
/// class      variant
/// mutation   in-place
/// definition res[res_col,j] = old(res)[res_col,j] - a[a_col,j]; limbs a.size() <= j < res.size() and other columns of res are unchanged
/// domain     res: a VecZnxBig or a window of one; a: a VecZnx or a window of one
/// ensures    res[res_col] -= a[a_col] limb by limb; limbs of res past a.size() keep their value
/// sparse     a is the sparse-capable slot, as for vec_znx_big_add
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

/// Difference of a big-word vector and a coefficient-domain vector.
///
/// ```text
/// op         vec_znx_big_sub_small_b(res, res_col, a, a_col, b, b_col)
/// class      derived
/// mutation   out-of-place
/// definition res[res_col,j] = a[a_col,j] - b[b_col,j]; other columns of res are unchanged
/// domain     res, a: VecZnxBig or windows of one; b: a VecZnx or a window of one, the coefficient-domain operand
/// ensures    res[res_col] = a[a_col] - b[b_col]: limbs only a reaches hold a, limbs only b reaches hold -b, limbs neither reaches are zero
/// sparse     b is the sparse-capable slot, as for vec_znx_big_add
/// fallback   vec_znx_big_from_small(res, res_col, b, b_col); vec_znx_big_sub_negate_assign(res, res_col, a, a_col)
/// override   allowed, scratch-free
/// test       test_vec_znx_big_sub_small_b, test_vec_znx_big_sub_small_b_derived, test_vec_znx_big_sparse_add_sub
/// ```
pub trait VecZnxBigSubSmallB<B: Backend> {
    /// Writes the difference of `a` and the coefficient-domain `b` into `res`.
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

/// In-place difference of a coefficient-domain vector and a big-word vector, the destination as the subtrahend.
///
/// ```text
/// op         vec_znx_big_sub_small_negate_assign(res, res_col, a, a_col)
/// class      variant
/// mutation   in-place
/// definition res[res_col,j] = a[a_col,j] - old(res)[res_col,j]; other columns of res are unchanged
/// domain     res: a VecZnxBig or a window of one; a: a VecZnx or a window of one
/// ensures    res[res_col] = a[a_col] - res[res_col] limb by limb; limbs of res past a.size() are negated in place
/// sparse     a is the sparse-capable slot, as for vec_znx_big_add
/// test       test_vec_znx_big_sub_small_b_assign, test_vec_znx_big_sparse_add_sub
/// ```
pub trait VecZnxBigSubSmallNegateAssign<B: Backend> {
    /// Writes the difference of the coefficient-domain `a` and `res` into `res`.
    fn vec_znx_big_sub_small_negate_assign(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    );
}

/// Sum of the coefficients of a big-word vector column, one destination coefficient per limb.
///
/// ```text
/// op         vec_znx_big_inner_sum(res, res_col, res_coeff, a, a_col)
/// class      basis
/// mutation   out-of-place
/// definition res[res_col,j,res_coeff] = wrap(sum_{0 <= i < a.n()} a[a_col,j,i]); coefficients i != res_coeff with 0 <= i < res.n() and other columns of res are unchanged
/// domain     res, a: dense VecZnxBig of any degree, the module's does not enter; res_coeff < res.n(); res.size() <= a.size()
/// ensures    only coefficient res_coeff in the selected column is written in each limb
/// test       test_vec_znx_big_inner_sum
/// ```
pub trait VecZnxBigInnerSum<B: Backend> {
    /// Writes the coefficient sum of each limb of `a` into coefficient `res_coeff` of `res`.
    fn vec_znx_big_inner_sum(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        res_coeff: usize,
        a: &VecZnxBigBackendRef<'_, B>,
        a_col: usize,
    );
}

#[allow(clippy::too_many_arguments)]
/// Weighted sum of the columns of a coefficient-domain vector into a big-word vector, coefficient by coefficient.
///
/// ```text
/// op         vec_znx_big_col_weighted_sum(res, res_col, a, weights, weights_col, cols, coeffs)
/// class      basis
/// mutation   out-of-place
/// definition res[res_col,j,i] = wrap(sum_{0 <= c < cols} a[c,j,i] * weights[weights_col,0,c]) for 0 <= i < coeffs; res[res_col,j,i] = 0 for coeffs <= i < res.n(); other columns of res are unchanged
/// domain     res: a dense VecZnxBig; a: a dense VecZnx; res and a of any degree, the module's does not enter; weights: a ScalarZnx; cols <= min(a.cols(), weights.n()); weights_col < weights.cols(); coeffs <= min(a.n(), res.n()); res.size() <= a.size()
/// ensures    every coefficient in the selected output column is written; coefficients from coeffs onward are zero
/// test       test_vec_znx_big_col_weighted_sum
/// ```
pub trait VecZnxBigColWeightedSum<B: Backend> {
    /// Writes into `res` the sum of the first `cols` columns of `a` weighted by `weights`, over the first `coeffs` coefficients.
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

/// Coefficient-wise product of a coefficient-domain vector and a scalar vector, in big words.
///
/// ```text
/// op         vec_znx_scalar_product(res, res_col, a, a_col, b, b_col)
/// class      basis
/// mutation   out-of-place
/// definition res[res_col,j,i] = wrap(a[a_col,j,i] * b[b_col,0,i]) for 0 <= i < a.n(); coefficients a.n() <= i < res.n() and other columns of res are unchanged
/// domain     res: a dense VecZnxBig with res.n() >= a.n() and res.size() <= a.size(); a: a dense VecZnx of any degree, the module's does not enter; b: a ScalarZnx of a's degree
/// ensures    only the first a.n() coefficients in each selected output limb are written
/// test       test_vec_znx_scalar_product
/// ```
pub trait VecZnxScalarProduct<B: Backend> {
    /// Writes the coefficient-wise product of `a` and `b` into `res`.
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

/// Negation of a big-word vector.
///
/// ```text
/// op         vec_znx_big_negate(res, res_col, a, a_col)
/// class      basis
/// mutation   out-of-place
/// definition res[res_col,j] = -a[a_col,j]; other columns of res are unchanged
/// domain     res, a: VecZnxBig or windows of one
/// ensures    res[res_col] = -a[a_col] limb by limb; limbs of res past a.size() are zero
/// test       test_vec_znx_big_negate
/// ```
pub trait VecZnxBigNegate<B: Backend> {
    /// Writes `-a` into `res`.
    fn vec_znx_big_negate(
        &self,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, B>,
        a_col: usize,
    );
}

/// In-place negation of a big-word vector.
///
/// ```text
/// op         vec_znx_big_negate_assign(res, res_col)
/// class      variant
/// mutation   in-place
/// definition res[res_col,j] = -old(res)[res_col,j]; other columns of res are unchanged
/// domain     res: a VecZnxBig or a window of one
/// ensures    every limb of res[res_col] is negated in place
/// test       test_vec_znx_big_negate_assign
/// ```
pub trait VecZnxBigNegateAssign<B: Backend> {
    /// Negates column `res_col` of `res`.
    fn vec_znx_big_negate_assign(&self, res: &mut VecZnxBigBackendMut<'_, B>, res_col: usize);
}

/// Scratch size of the normalization of a big-word vector.
///
/// ```text
/// op         vec_znx_big_normalize_tmp_bytes()
/// class      support
/// mutation   none
/// domain     -
/// ensures    returns the scratch bytes required by vec_znx_big_normalize, independent of operand limb counts
/// test       test_vec_znx_big_normalize
/// ```
pub trait VecZnxBigNormalizeTmpBytes {
    /// Returns the scratch byte size that [`VecZnxBigNormalize`] requires.
    fn vec_znx_big_normalize_tmp_bytes(&self) -> usize;
}

#[allow(clippy::too_many_arguments)]
/// Normalization of a big-word vector into a coefficient-domain vector at a target base, precision and offset.
///
/// ```text
/// op         vec_znx_big_normalize(res, res_base2k, res_k, res_offset, res_col, a, a_base2k, a_col, scratch)
/// class      basis
/// mutation   out-of-place
/// definition res[res_col] = canon([[a[a_col]]]_a_base2k * 2^res_offset, res_base2k, res_k, res.size()); other columns of res are unchanged
/// domain     res: a VecZnx or a window of one; a: a VecZnxBig or a window of one, of the same degree; res_k <= res.size() * res_base2k; for every 0 <= j < a.size() and 0 <= i < a.n(): with a 64-bit big coefficient width in a, abs(a[a_col,j,i]) <= 2^62 and a_base2k and res_base2k in 1..=62; with a 128-bit one, abs(a[a_col,j,i]) <= 2^126, 1 <= a_base2k <= 127 and 1 <= res_base2k <= 64
/// requires   scratch >= vec_znx_big_normalize_tmp_bytes()
/// ensures    the selected output column is canonical at res_base2k and res_k and congruent modulo 1 to rnd([[a[a_col]]]_a_base2k * 2^res_offset, res_k)
/// test       test_vec_znx_big_normalize, test_vec_znx_big_window_normalize
/// ```
pub trait VecZnxBigNormalize<B: Backend> {
    /// Writes column `a_col` of `a`, scaled by `2^res_offset` and normalized at `res_base2k` and `res_k`, into `res`.
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

/// Scratch size of the in-place automorphism of a big-word vector.
///
/// ```text
/// op         vec_znx_big_automorphism_assign_tmp_bytes()
/// class      support
/// mutation   none
/// domain     -
/// ensures    returns the scratch bytes required by vec_znx_big_automorphism_assign, independent of the limb count
/// test       test_vec_znx_big_automorphism_assign
/// ```
pub trait VecZnxBigAutomorphismAssignTmpBytes {
    /// Returns the scratch byte size that [`VecZnxBigAutomorphismAssign`] requires.
    fn vec_znx_big_automorphism_assign_tmp_bytes(&self) -> usize;
}

/// Automorphism `X -> X^p` of a big-word vector.
///
/// ```text
/// op         vec_znx_big_automorphism(p, res, res_col, a, a_col)
/// class      basis
/// mutation   out-of-place
/// definition res[res_col,j] = sum_{0 <= i < a.n()} a[a_col,j,i] * X^(p*i) in R_N; other columns of res are unchanged
/// domain     res, a: dense VecZnxBig of degree N; p odd
/// ensures    the selected output column is the limbwise automorphism with zero extension or truncation
/// test       test_vec_znx_big_automorphism
/// ```
pub trait VecZnxBigAutomorphism<B: Backend> {
    /// Writes the automorphism `X -> X^p` of `a` into `res`.
    fn vec_znx_big_automorphism(
        &self,
        p: i64,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBigBackendRef<'_, B>,
        a_col: usize,
    );
}

/// In-place automorphism `X -> X^p` of a big-word vector.
///
/// ```text
/// op         vec_znx_big_automorphism_assign(p, res, res_col, scratch)
/// class      variant
/// mutation   in-place
/// definition res[res_col,j] = sum_{0 <= i < res.n()} old(res)[res_col,j,i] * X^(p*i) in R_N; other columns of res are unchanged
/// domain     res: a dense VecZnxBig of degree N; p odd
/// requires   scratch >= vec_znx_big_automorphism_assign_tmp_bytes()
/// ensures    the selected output column is the limbwise automorphism of its pre-call value
/// fallback   none
/// override   required
/// test       test_vec_znx_big_automorphism_assign
/// ```
pub trait VecZnxBigAutomorphismAssign<B: Backend> {
    /// Replaces column `res_col` of `res` by its automorphism `X -> X^p`.
    fn vec_znx_big_automorphism_assign(
        &self,
        p: i64,
        res: &mut VecZnxBigBackendMut<'_, B>,
        res_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}
