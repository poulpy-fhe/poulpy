use super::{TestBackend, TestParams, download_mat_znx, upload_mat_znx};
use crate::{
    AlignedBuf,
    api::{MatZnxAddAssign, MatZnxAlloc, VecZnxFillUniformSource},
    layouts::{MatZnx, MatZnxAtBackendMut, MatZnxToBackendMut, MatZnxToBackendRef, Module},
    source::Source,
};

/// `mat_znx_add_assign` on both backends against the entrywise sum, over matrices of several shapes.
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

    for (rows, cols_in, cols_out, size) in [(1, 1, 1, 1), (2, 1, 2, 3), (3, 2, 3, 2)] {
        let mut fill = || -> MatZnx<AlignedBuf, i64> {
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
        let (res, a) = (fill(), fill());
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

        assert!(
            download_mat_znx::<BR>(&res_ref) == want,
            "reference backend: not the entrywise sum"
        );
        assert!(
            download_mat_znx::<BT>(&res_test) == want,
            "backend under test: not the entrywise sum"
        );
        assert!(download_mat_znx::<BT>(&a_test) == a, "the operand changed");
    }
}
