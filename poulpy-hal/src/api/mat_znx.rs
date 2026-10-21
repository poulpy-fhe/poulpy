use crate::layouts::{Backend, MatZnxBackendMut, MatZnxBackendRef};

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
