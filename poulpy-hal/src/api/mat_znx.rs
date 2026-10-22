use crate::layouts::{Backend, MatZnxBackendMut, MatZnxBackendRef, ScratchArena};

/// In-place entrywise addition of a matrix.
///
/// ```text
/// op         mat_znx_add_assign(res, a)
/// class      derived
/// mutation   in-place
/// definition res[i,c,d,j] = old(res)[i,c,d,j] + a[i,c,d,j] for every row i, input column c and output column d
/// domain     res, a: MatZnx of one degree with the same rows, cols_in, cols_out and size, read at one shared base2k
/// ensures    every entry of res gains the matching entry of a limb by limb; the digits are not renormalized
/// fallback   default body: vec_znx_add_assign on every output column of every (row, input column) entry
/// override   allowed, scratch-free
/// test       test_mat_znx_add_assign
/// ```
pub trait MatZnxAddAssign<B: Backend> {
    /// Adds every entry of `a` into the matching entry of `res`.
    fn mat_znx_add_assign(&self, res: &mut MatZnxBackendMut<'_, B>, a: &MatZnxBackendRef<'_, B>);
}

/// Copy of one output column of every entry of a matrix into another.
///
/// ```text
/// op         mat_znx_copy(res, res_col, a, a_col)
/// class      derived
/// mutation   out-of-place
/// definition res[i,c,res_col,j] = a[i,c,a_col,j] for every row i < res.rows() and input column c, with a[i,c,a_col,j] = 0 for i >= a.rows(); the other output columns of res are untouched
/// domain     res, a: MatZnx of one degree with the same cols_in
/// ensures    res[i,c][res_col] holds a[i,c][a_col] limb by limb; rows of res past a.rows() and limbs of res past a.size() are zero; rows of a past res.rows() are not read
/// fallback   default body: vec_znx_copy on every (row, input column) entry the two share, vec_znx_zero on the rows of res past a.rows()
/// override   allowed, scratch-free
/// test       test_mat_znx_copy
/// ```
pub trait MatZnxCopy<B: Backend> {
    /// Writes `a[i,c][a_col]` into `res[i,c][res_col]` for every entry of `res`, zero past the rows of `a`.
    fn mat_znx_copy(&self, res: &mut MatZnxBackendMut<'_, B>, res_col: usize, a: &MatZnxBackendRef<'_, B>, a_col: usize);
}

/// Scratch requirement of the matrix normalization operations.
///
/// ```text
/// op         mat_znx_normalize_tmp_bytes()
/// class      support
/// mutation   none
/// domain     -
/// ensures    returns the scratch bytes mat_znx_normalize and mat_znx_normalize_assign require, independently of operand shapes
/// test       test_mat_znx_normalize
/// ```
pub trait MatZnxNormalizeTmpBytes {
    /// Returns the scratch byte count the matrix normalization operations require.
    fn mat_znx_normalize_tmp_bytes(&self) -> usize;
}

/// Normalization of one output column of every entry of a matrix at a destination radix and precision, with a
/// power-of-two offset.
///
/// ```text
/// op         mat_znx_normalize(res, res_base2k, res_k, res_offset, res_col, a, a_base2k, a_col, scratch)
/// class      derived
/// mutation   out-of-place
/// definition res[i,c][res_col] = canon([[a[i,c][a_col]]]_a_base2k * 2^res_offset, res_base2k, res_k, res.size()) for every row i < res.rows() and input column c, with a[i,c][a_col] = 0 for i >= a.rows(); the other output columns of res are untouched
/// domain     res, a: MatZnx of one degree with the same cols_in, read at res_base2k and a_base2k; both radix widths in 1..=62; res_k <= res.size() * res_base2k; every input digit in [-2^62, 2^62]
/// requires   scratch >= mat_znx_normalize_tmp_bytes()
/// ensures    res[i,c][res_col] is canonical at radix res_base2k and precision res_k and congruent modulo 1 to rnd([[a[i,c][a_col]]]_a_base2k * 2^res_offset, res_k) for every entry of res; rows of res past a.rows() are zero; rows of a past res.rows() are not read
/// fallback   default body: vec_znx_normalize on every (row, input column) entry the two share, vec_znx_zero on the rows of res past a.rows()
/// override   allowed
/// test       test_mat_znx_normalize
/// ```
pub trait MatZnxNormalize<B: Backend> {
    /// Writes `a[i,c][a_col]`, scaled by `2^res_offset` and normalized at `res_base2k` and `res_k`, into
    /// `res[i,c][res_col]` for every entry of `res`, zero past the rows of `a`.
    #[allow(clippy::too_many_arguments)]
    fn mat_znx_normalize(
        &self,
        res: &mut MatZnxBackendMut<'_, B>,
        res_base2k: usize,
        res_k: usize,
        res_offset: i64,
        res_col: usize,
        a: &MatZnxBackendRef<'_, B>,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// In-place normalization of one output column of every entry of a matrix at a radix and precision, with a
/// power-of-two offset.
///
/// ```text
/// op         mat_znx_normalize_assign(base2k, k, a_offset, a, a_col, scratch)
/// class      derived
/// mutation   in-place
/// definition a[i,c][a_col] = canon([[old(a)[i,c][a_col]]]_base2k * 2^a_offset, base2k, k, a.size()) for every row i and input column c; the other output columns of a are untouched
/// domain     a: a MatZnx read at base2k; base2k in 1..=62; every input digit in [-2^62, 2^62]; k <= a.size() * base2k; a_offset <= 0; a_offset != 0 requires k == a.size() * base2k
/// requires   scratch >= mat_znx_normalize_tmp_bytes()
/// ensures    a[i,c][a_col] is canonical at radix base2k and precision k and congruent modulo 1 to rnd([[old(a)[i,c][a_col]]]_base2k * 2^a_offset, k) for every entry
/// fallback   default body: vec_znx_normalize_assign on every (row, input column) entry
/// override   allowed
/// test       test_mat_znx_normalize
/// ```
pub trait MatZnxNormalizeAssign<B: Backend> {
    /// Replaces `a[i,c][a_col]` with its value scaled by `2^a_offset` and normalized at `base2k` and `k`, for every
    /// entry.
    fn mat_znx_normalize_assign(
        &self,
        base2k: usize,
        k: usize,
        a_offset: i64,
        a: &mut MatZnxBackendMut<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}
