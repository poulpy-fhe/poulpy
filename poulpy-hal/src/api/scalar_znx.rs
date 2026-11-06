use crate::layouts::{Backend, ScalarZnxBackendMut, ScalarZnxBackendRef};

/// Substitution `X -> X^k` on a column of a scalar vector.
///
/// ```text
/// op         scalar_znx_automorphism(k, res, res_col, a, a_col)
/// class      derived
/// mutation   out-of-place
/// definition res[res_col] = sigma_k(a[a_col,0]) in R_N; the other columns of res are untouched
/// domain     res, a: ScalarZnx of degree N; k odd
/// ensures    res[res_col] is the image of a[a_col] under X -> X^k; the other columns are untouched
/// fallback   default body: vec_znx_automorphism(k, res, res_col, a, a_col) with res and a read as one-limb VecZnx
/// override   allowed, scratch-free
/// test       test_scalar_znx_automorphism, test_scalar_znx_automorphism_derived
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

/// Embedding of a conjugate-invariant scalar column into the standard ring of twice its degree.
///
/// ```text
/// op         scalar_znx_ci_embed(res, res_col, a, a_col)
/// class      derived
/// mutation   out-of-place
/// definition res[res_col] = a[a_col,0] + sum_{0<i<N} a[a_col,i] (X^i + X^-i) in Z[X]/(X^2N + 1); the other columns of res are untouched
/// domain     standard module of degree at least 2N; res: ScalarZnx of degree 2N; a: ScalarZnx of degree N
/// ensures    res[res_col] is the image of a[a_col] in the standard ring
/// fallback   default body: vec_znx_ci_embed(res, res_col, a, a_col) with res and a read as one-limb VecZnx
/// override   allowed, scratch-free
/// test       test_scalar_znx_ci_embed, test_scalar_znx_ci_embed_derived
/// ```
pub trait ScalarZnxCIEmbed<B: Backend> {
    /// Writes the embedded `a[a_col]` into `res[res_col]`.
    fn scalar_znx_ci_embed(
        &self,
        res: &mut ScalarZnxBackendMut<'_, B>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, B>,
        a_col: usize,
    );
}
