use super::{TestBackend, TestParams, download_mat_znx, upload_mat_znx};
use crate::test_suite::{PoisonedArena, poisoned_arena};
use crate::{
    AlignedBuf,
    api::{
        MatZnxAddAssign, MatZnxAlloc, MatZnxCopy, MatZnxNormalize, MatZnxNormalizeAssign, MatZnxNormalizeTmpBytes, VecZnxCopy,
        VecZnxFillUniformSource, VecZnxNormalize, VecZnxNormalizeAssign, VecZnxNormalizeTmpBytes, VecZnxZero,
    },
    layouts::{MatZnx, MatZnxAtBackendMut, MatZnxAtBackendRef, MatZnxToBackendMut, MatZnxToBackendRef, Module},
    oep::{mat_znx_add_assign_derived, mat_znx_copy_derived, mat_znx_normalize_assign_derived, mat_znx_normalize_derived},
    source::Source,
};

/// A host copy of a `rows x cols_in x cols_out` matrix of `size` limbs, every digit uniform at `base2k`.
fn random_mat<BR: TestBackend>(
    module_ref: &Module<BR>,
    params: &TestParams,
    (rows, cols_in, cols_out, size): (usize, usize, usize, usize),
    source: &mut Source,
) -> MatZnx<AlignedBuf, i64>
where
    Module<BR>: MatZnxAlloc<BR> + VecZnxFillUniformSource<BR>,
{
    let mut mat = module_ref.mat_znx_alloc(params.n, rows, cols_in, cols_out, size);
    for row in 0..rows {
        for col_in in 0..cols_in {
            for col in 0..cols_out {
                module_ref.vec_znx_fill_uniform_source(
                    params.base2k,
                    size * params.base2k,
                    &mut MatZnxAtBackendMut::<BR>::at_backend_mut(&mut mat, row, col_in),
                    col,
                    source,
                );
            }
        }
    }
    download_mat_znx::<BR>(&mat)
}

/// Runs `call` with the panic message silenced and returns whether it panicked.
fn panics(call: impl FnOnce()) -> bool {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(call));
    std::panic::set_hook(hook);
    caught.is_err()
}

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
    let mut source: Source = Source::new([0u8; 32]);

    for shape in [(1, 1, 1, 1), (2, 1, 2, 3), (3, 2, 3, 2)] {
        let (res, a) = (
            random_mat(module_ref, params, shape, &mut source),
            random_mat(module_ref, params, shape, &mut source),
        );
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

    let (res, a) = (
        random_mat(module_ref, params, (2, 1, 1, 1), &mut source),
        random_mat(module_ref, params, (1, 1, 1, 1), &mut source),
    );
    let (mut res_test, a_test) = (upload_mat_znx::<BT>(&res), upload_mat_znx::<BT>(&a));
    assert!(
        panics(|| module_test.mat_znx_add_assign(
            &mut MatZnxToBackendMut::<BT>::to_backend_mut(&mut res_test),
            &MatZnxToBackendRef::<BT>::to_backend_ref(&a_test),
        )),
        "mat_znx_add_assign accepted operands of different rows"
    );
    assert!(
        download_mat_znx::<BT>(&res_test) == res,
        "a rejected mat_znx_add_assign wrote res"
    );
}

/// `mat_znx_copy` on both backends and the derived default body against `vec_znx_copy` on every entry the operands
/// share and `vec_znx_zero` past the rows of `a`, over matrices of fewer, as many and more rows and limbs than `a`, one
/// output column at a time, and the rejection of an input-column mismatch before any entry is written.
pub fn test_mat_znx_copy<BR: TestBackend, BT: TestBackend>(params: &TestParams, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: MatZnxCopy<BR> + VecZnxCopy<BR> + VecZnxZero<BR> + MatZnxAlloc<BR> + VecZnxFillUniformSource<BR>,
    Module<BT>: MatZnxCopy<BT>,
{
    let mut source: Source = Source::new([2u8; 32]);

    for ((res_rows, res_size), (rows, cols_in, cols_out, size)) in [
        ((1, 1), (1, 1, 1, 1)),
        ((1, 2), (2, 1, 2, 3)),
        ((3, 3), (2, 2, 2, 2)),
        ((2, 1), (3, 2, 3, 2)),
    ] {
        let a = random_mat(module_ref, params, (rows, cols_in, cols_out, size), &mut source);
        let res = random_mat(module_ref, params, (res_rows, cols_in, cols_out, res_size), &mut source);

        for res_col in 0..cols_out {
            let a_col = cols_out - 1 - res_col;
            let a_ref = upload_mat_znx::<BR>(&a);
            let mut want = upload_mat_znx::<BR>(&res);
            for row in 0..res_rows {
                for col_in in 0..cols_in {
                    let mut entry = MatZnxAtBackendMut::<BR>::at_backend_mut(&mut want, row, col_in);
                    if row < rows {
                        let a_entry = MatZnxAtBackendRef::<BR>::at_backend(&a_ref, row, col_in);
                        module_ref.vec_znx_copy(&mut entry, res_col, &a_entry, a_col);
                    } else {
                        module_ref.vec_znx_zero(&mut entry, res_col);
                    }
                }
            }
            let want = download_mat_znx::<BR>(&want);

            let mut have_ref = upload_mat_znx::<BR>(&res);
            module_ref.mat_znx_copy(
                &mut MatZnxToBackendMut::<BR>::to_backend_mut(&mut have_ref),
                res_col,
                &MatZnxToBackendRef::<BR>::to_backend_ref(&a_ref),
                a_col,
            );
            let a_test = upload_mat_znx::<BT>(&a);
            let mut have_test = upload_mat_znx::<BT>(&res);
            module_test.mat_znx_copy(
                &mut MatZnxToBackendMut::<BT>::to_backend_mut(&mut have_test),
                res_col,
                &MatZnxToBackendRef::<BT>::to_backend_ref(&a_test),
                a_col,
            );
            let mut have_derived = upload_mat_znx::<BT>(&res);
            mat_znx_copy_derived::<BT>(
                module_test,
                &mut MatZnxToBackendMut::<BT>::to_backend_mut(&mut have_derived),
                res_col,
                &MatZnxToBackendRef::<BT>::to_backend_ref(&a_test),
                a_col,
            );
            assert!(
                download_mat_znx::<BR>(&have_ref) == want,
                "reference backend: not the entrywise copy"
            );
            assert!(
                download_mat_znx::<BT>(&have_test) == want,
                "backend under test: not the entrywise copy"
            );
            assert!(
                download_mat_znx::<BT>(&have_derived) == want,
                "derived default body: not the entrywise copy"
            );
            assert!(download_mat_znx::<BT>(&a_test) == a, "mat_znx_copy changed the operand");
        }
    }

    let (res, a) = (
        random_mat(module_ref, params, (1, 2, 1, 1), &mut source),
        random_mat(module_ref, params, (1, 1, 1, 1), &mut source),
    );
    let (mut res_test, a_test) = (upload_mat_znx::<BT>(&res), upload_mat_znx::<BT>(&a));
    assert!(
        panics(|| module_test.mat_znx_copy(
            &mut MatZnxToBackendMut::<BT>::to_backend_mut(&mut res_test),
            0,
            &MatZnxToBackendRef::<BT>::to_backend_ref(&a_test),
            0,
        )),
        "mat_znx_copy accepted operands of different input columns"
    );
    assert!(download_mat_znx::<BT>(&res_test) == res, "a rejected mat_znx_copy wrote res");
}

/// `mat_znx_normalize` and `mat_znx_normalize_assign` on both backends and their derived default bodies against
/// `vec_znx_normalize` and `vec_znx_normalize_assign` on every entry, over non-canonical matrices of several shapes and
/// destinations of fewer, as many and more rows, one output column at a time, and the rejection of an input-column
/// mismatch before any entry is written.
pub fn test_mat_znx_normalize<BR: TestBackend, BT: TestBackend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: MatZnxNormalize<BR>
        + MatZnxNormalizeAssign<BR>
        + MatZnxNormalizeTmpBytes
        + VecZnxNormalize<BR>
        + VecZnxNormalizeAssign<BR>
        + VecZnxNormalizeTmpBytes
        + VecZnxZero<BR>
        + MatZnxAlloc<BR>
        + VecZnxFillUniformSource<BR>,
    Module<BT>: MatZnxNormalize<BT> + MatZnxNormalizeAssign<BT> + MatZnxNormalizeTmpBytes,
{
    let base2k: usize = params.base2k;
    let mut source: Source = Source::new([1u8; 32]);
    let mut scratch_ref: PoisonedArena<BR> = poisoned_arena(module_ref.mat_znx_normalize_tmp_bytes());
    let mut scratch_ref_assign: PoisonedArena<BR> = poisoned_arena(module_ref.mat_znx_normalize_tmp_bytes());
    let mut scratch_ref_oracle: PoisonedArena<BR> = poisoned_arena(module_ref.vec_znx_normalize_tmp_bytes());
    let mut scratch_test: PoisonedArena<BT> = poisoned_arena(module_test.mat_znx_normalize_tmp_bytes());
    let mut scratch_test_assign: PoisonedArena<BT> = poisoned_arena(module_test.mat_znx_normalize_tmp_bytes());
    let mut scratch_test_derived: PoisonedArena<BT> =
        poisoned_arena(crate::oep::mat_znx_normalize_tmp_bytes_derived::<BT>(module_test));
    let mut scratch_test_derived_assign: PoisonedArena<BT> =
        poisoned_arena(crate::oep::mat_znx_normalize_tmp_bytes_derived::<BT>(module_test));

    for (res_rows, (rows, cols_in, cols_out, size)) in
        [(1, (1, 1, 1, 1)), (1, (2, 1, 2, 3)), (3, (3, 2, 3, 2)), (4, (3, 2, 3, 2))]
    {
        let res_shape = (res_rows, cols_in, cols_out, size + 1);
        let res_k = (size + 1) * base2k - 1;
        let mut a = random_mat(module_ref, params, (rows, cols_in, cols_out, size), &mut source);
        let carry = random_mat(module_ref, params, (rows, cols_in, cols_out, size), &mut source);
        for (x, y) in a.raw_mut().iter_mut().zip(carry.raw()) {
            *x += y;
        }
        let res = random_mat(module_ref, params, res_shape, &mut source);

        for res_col in 0..cols_out {
            let a_col = cols_out - 1 - res_col;
            let a_ref = upload_mat_znx::<BR>(&a);
            let mut want = upload_mat_znx::<BR>(&res);
            for row in 0..res_rows {
                for col_in in 0..cols_in {
                    let mut entry = MatZnxAtBackendMut::<BR>::at_backend_mut(&mut want, row, col_in);
                    if row < rows {
                        module_ref.vec_znx_normalize(
                            &mut entry,
                            base2k,
                            res_k,
                            0,
                            res_col,
                            &MatZnxAtBackendRef::<BR>::at_backend(&a_ref, row, col_in),
                            base2k,
                            a_col,
                            &mut scratch_ref_oracle.arena(),
                        );
                    } else {
                        module_ref.vec_znx_zero(&mut entry, res_col);
                    }
                }
            }
            let want = download_mat_znx::<BR>(&want);

            let mut have_ref = upload_mat_znx::<BR>(&res);
            module_ref.mat_znx_normalize(
                &mut MatZnxToBackendMut::<BR>::to_backend_mut(&mut have_ref),
                base2k,
                res_k,
                0,
                res_col,
                &MatZnxToBackendRef::<BR>::to_backend_ref(&a_ref),
                base2k,
                a_col,
                &mut scratch_ref.arena(),
            );
            let a_test = upload_mat_znx::<BT>(&a);
            let mut have_test = upload_mat_znx::<BT>(&res);
            module_test.mat_znx_normalize(
                &mut MatZnxToBackendMut::<BT>::to_backend_mut(&mut have_test),
                base2k,
                res_k,
                0,
                res_col,
                &MatZnxToBackendRef::<BT>::to_backend_ref(&a_test),
                base2k,
                a_col,
                &mut scratch_test.arena(),
            );
            let mut have_derived = upload_mat_znx::<BT>(&res);
            mat_znx_normalize_derived::<BT>(
                module_test,
                &mut MatZnxToBackendMut::<BT>::to_backend_mut(&mut have_derived),
                base2k,
                res_k,
                0,
                res_col,
                &MatZnxToBackendRef::<BT>::to_backend_ref(&a_test),
                base2k,
                a_col,
                &mut scratch_test_derived.arena(),
            );
            assert!(
                download_mat_znx::<BR>(&have_ref) == want,
                "reference backend: not the entrywise normalization"
            );
            assert!(
                download_mat_znx::<BT>(&have_test) == want,
                "backend under test: not the entrywise normalization"
            );
            assert!(
                download_mat_znx::<BT>(&have_derived) == want,
                "derived default body: not the entrywise normalization"
            );
            assert!(download_mat_znx::<BT>(&a_test) == a, "mat_znx_normalize changed the operand");

            let k = size * base2k - 1;
            let mut want = upload_mat_znx::<BR>(&a);
            for row in 0..rows {
                for col_in in 0..cols_in {
                    module_ref.vec_znx_normalize_assign(
                        base2k,
                        k,
                        0,
                        &mut MatZnxAtBackendMut::<BR>::at_backend_mut(&mut want, row, col_in),
                        a_col,
                        &mut scratch_ref_oracle.arena(),
                    );
                }
            }
            let want = download_mat_znx::<BR>(&want);
            let mut have_ref = upload_mat_znx::<BR>(&a);
            module_ref.mat_znx_normalize_assign(
                base2k,
                k,
                0,
                &mut MatZnxToBackendMut::<BR>::to_backend_mut(&mut have_ref),
                a_col,
                &mut scratch_ref_assign.arena(),
            );
            let mut have_test = upload_mat_znx::<BT>(&a);
            module_test.mat_znx_normalize_assign(
                base2k,
                k,
                0,
                &mut MatZnxToBackendMut::<BT>::to_backend_mut(&mut have_test),
                a_col,
                &mut scratch_test_assign.arena(),
            );
            let mut have_derived = upload_mat_znx::<BT>(&a);
            mat_znx_normalize_assign_derived::<BT>(
                module_test,
                base2k,
                k,
                0,
                &mut MatZnxToBackendMut::<BT>::to_backend_mut(&mut have_derived),
                a_col,
                &mut scratch_test_derived_assign.arena(),
            );
            assert!(
                download_mat_znx::<BR>(&have_ref) == want,
                "reference backend: not the entrywise normalization"
            );
            assert!(
                download_mat_znx::<BT>(&have_test) == want,
                "backend under test: not the entrywise normalization"
            );
            assert!(
                download_mat_znx::<BT>(&have_derived) == want,
                "derived default body: not the entrywise normalization"
            );
        }
    }

    let (res, a) = (
        random_mat(module_ref, params, (1, 2, 1, 1), &mut source),
        random_mat(module_ref, params, (1, 1, 1, 1), &mut source),
    );
    let (mut res_test, a_test) = (upload_mat_znx::<BT>(&res), upload_mat_znx::<BT>(&a));
    assert!(
        panics(|| module_test.mat_znx_normalize(
            &mut MatZnxToBackendMut::<BT>::to_backend_mut(&mut res_test),
            base2k,
            base2k,
            0,
            0,
            &MatZnxToBackendRef::<BT>::to_backend_ref(&a_test),
            base2k,
            0,
            &mut scratch_test.arena(),
        )),
        "mat_znx_normalize accepted operands of different input columns"
    );
    assert!(
        download_mat_znx::<BT>(&res_test) == res,
        "a rejected mat_znx_normalize wrote res"
    );
}
