use super::{TestBackend, TestParams, download_mat_znx, upload_mat_znx};
use crate::{
    AlignedBuf,
    api::{MatZnxAddAssign, MatZnxAlloc, VecZnxFillUniformSource},
    layouts::{MatZnx, MatZnxAtBackendMut, MatZnxToBackendMut, MatZnxToBackendRef, Module},
    oep::mat_znx_add_assign_derived,
    source::Source,
};

/// `mat_znx_add_assign` on both backends and the derived default body against the entrywise sum, over matrices of
/// several shapes, and its rejection of a shape mismatch before any entry is written.
pub fn test_mat_znx_add_assign<BR: TestBackend, BT: TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: MatZnxAddAssign<BR> + MatZnxAlloc<BR> + VecZnxFillUniformSource<BR>,
    Module<BT>: MatZnxAddAssign<BT>,
{
    let base2k: usize = params.base2k;
    let mut source: Source = Source::new([0u8; 32]);
    let mut fill = |rows: usize, cols_in: usize, cols_out: usize, size: usize| -> MatZnx<AlignedBuf, i64> {
        let mut mat = module_ref.mat_znx_alloc(params.n, rows, cols_in, cols_out, size);
        for row in 0..rows {
            for col_in in 0..cols_in {
                for col in 0..cols_out {
                    module_ref.vec_znx_fill_uniform_source(
                        base2k,
                        size * base2k,
                        &mut MatZnxAtBackendMut::<BR>::at_backend_mut(&mut mat, row, col_in),
                        col,
                        &mut source,
                    );
                }
            }
        }
        download_mat_znx::<BR>(&mat)
    };

    for (rows, cols_in, cols_out, size) in [(1, 1, 1, 1), (2, 1, 2, 3), (3, 2, 3, 2)] {
        let (res, a) = (fill(rows, cols_in, cols_out, size), fill(rows, cols_in, cols_out, size));
        let mut want = res.clone();
        for (w, x) in want.raw_mut().iter_mut().zip(a.raw()) {
            *w += x;
        }

        let (mut res_ref, a_ref) = (upload_mat_znx::<BR>(&res), upload_mat_znx::<BR>(&a));
        module_ref.mat_znx_add_assign(
            &mut MatZnxToBackendMut::<BR>::to_backend_mut(&mut res_ref),
            &MatZnxToBackendRef::<BR>::to_backend_ref(&a_ref),
        );
        let (mut res_test, a_test) = (upload_mat_znx::<BT>(&res), upload_mat_znx::<BT>(&a));
        module_test.mat_znx_add_assign(
            &mut MatZnxToBackendMut::<BT>::to_backend_mut(&mut res_test),
            &MatZnxToBackendRef::<BT>::to_backend_ref(&a_test),
        );
        let (mut res_derived, a_derived) = (upload_mat_znx::<BT>(&res), upload_mat_znx::<BT>(&a));
        mat_znx_add_assign_derived::<BT>(
            module_test,
            &mut MatZnxToBackendMut::<BT>::to_backend_mut(&mut res_derived),
            &MatZnxToBackendRef::<BT>::to_backend_ref(&a_derived),
        );

        assert!(
            download_mat_znx::<BR>(&res_ref) == want,
            "reference backend: not the entrywise sum"
        );
        assert!(
            download_mat_znx::<BT>(&res_test) == want,
            "backend under test: not the entrywise sum"
        );
        assert!(
            download_mat_znx::<BT>(&res_derived) == want,
            "derived default body: not the entrywise sum"
        );
        assert!(
            download_mat_znx::<BR>(&a_ref) == a,
            "the reference backend changed the operand"
        );
        assert!(
            download_mat_znx::<BT>(&a_test) == a,
            "the backend under test changed the operand"
        );
        assert!(
            download_mat_znx::<BT>(&a_derived) == a,
            "the derived default body changed the operand"
        );
    }

    let (res, a) = (fill(2, 1, 1, 1), fill(1, 1, 1, 1));
    let (mut res_test, a_test) = (upload_mat_znx::<BT>(&res), upload_mat_znx::<BT>(&a));
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        module_test.mat_znx_add_assign(
            &mut MatZnxToBackendMut::<BT>::to_backend_mut(&mut res_test),
            &MatZnxToBackendRef::<BT>::to_backend_ref(&a_test),
        );
    }));
    std::panic::set_hook(hook);
    assert!(caught.is_err(), "mat_znx_add_assign accepted operands of different rows");
    assert!(
        download_mat_znx::<BT>(&res_test) == res,
        "a rejected mat_znx_add_assign wrote res"
    );
}
