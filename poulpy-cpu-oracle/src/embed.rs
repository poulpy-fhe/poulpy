//! Degree embedding for sparse-capable slots.
//!
//! A degree-`n` operand under a degree-`N` call stands for its image
//! `p(X^(N/n))`, that is `switch_ring_{n->N}` of itself. The oracle
//! materializes that image on the heap and runs the dense kernel on it.

use poulpy_hal::{
    AlignedBuf, alloc_aligned,
    layouts::{
        Backend, VecZnx, VecZnxBackendRef, VecZnxBig, VecZnxBigBackendRef, VecZnxBigToBackendRef, VecZnxToBackendRef, ZnxView,
        ZnxViewMut, assert_dense,
    },
};

fn gap(n: usize, a_n: usize) -> usize {
    assert!(
        a_n.is_power_of_two() && a_n <= n && n.is_multiple_of(a_n),
        "operand degree {a_n} does not divide call degree {n}"
    );
    n / a_n
}

fn embed_into<A: ZnxView, R: ZnxViewMut<Scalar = A::Scalar>>(res: &mut R, a: &A) {
    let gap = gap(res.n(), a.n());
    for col in 0..a.cols() {
        for j in 0..a.size() {
            let dst = res.at_mut(col, j);
            for (i, &x) in a.at(col, j).iter().enumerate() {
                dst[i * gap] = x;
            }
        }
    }
}

/// Calls `f` with `a` at degree `n`, embedding it first when it is sparse.
pub(crate) fn with_vec_znx<BE, T>(a: &VecZnxBackendRef<'_, BE>, n: usize, f: impl FnOnce(&VecZnxBackendRef<'_, BE>) -> T) -> T
where
    BE: Backend<ZnxWord = i64>,
    VecZnx<AlignedBuf, i64>: VecZnxToBackendRef<BE>,
    for<'x> VecZnxBackendRef<'x, BE>: ZnxView<Scalar = i64>,
{
    if a.n() == n {
        return f(a);
    }
    assert_dense(a, "sparse operand");
    let bytes = n * a.cols() * a.size() * size_of::<i64>();
    let mut dense = VecZnx::<AlignedBuf, i64>::from_data(alloc_aligned::<u8>(bytes), n, a.cols(), a.size());
    embed_into(&mut dense, a);
    f(&dense.to_backend_ref())
}

/// Calls `f` with `a` at degree `n`, embedding it first when it is sparse.
pub(crate) fn with_vec_znx_big<BE, T>(
    a: &VecZnxBigBackendRef<'_, BE>,
    n: usize,
    f: impl FnOnce(&VecZnxBigBackendRef<'_, BE>) -> T,
) -> T
where
    BE: Backend,
    VecZnxBig<AlignedBuf, BE::BigWord, BE>: VecZnxBigToBackendRef<BE> + ZnxViewMut<Scalar = BE::BigWord>,
    for<'x> VecZnxBigBackendRef<'x, BE>: ZnxView<Scalar = BE::BigWord>,
{
    if a.n() == n {
        return f(a);
    }
    assert_dense(a, "sparse operand");
    let bytes = n * a.cols() * a.size() * size_of::<BE::BigWord>();
    let mut dense = VecZnxBig::<AlignedBuf, BE::BigWord, BE>::from_data(alloc_aligned::<u8>(bytes), n, a.cols(), a.size());
    embed_into(&mut dense, a);
    f(&dense.to_backend_ref())
}
