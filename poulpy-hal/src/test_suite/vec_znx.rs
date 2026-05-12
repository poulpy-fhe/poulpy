use super::{
    TestParams, download_scalar_znx, download_vec_znx, scalar_znx_backend_ref, upload_scalar_znx, upload_vec_znx,
    vec_znx_backend_mut, vec_znx_backend_ref,
};
use std::f64::consts::SQRT_2;

use crate::{
    api::{
        ModuleNew, ScalarZnxFillBinaryBlockBackend, ScalarZnxFillBinaryBlockSourceBackend, ScalarZnxFillBinaryHwBackend,
        ScalarZnxFillBinaryHwSourceBackend, ScalarZnxFillBinaryProbBackend, ScalarZnxFillBinaryProbSourceBackend,
        ScalarZnxFillTernaryHwBackend, ScalarZnxFillTernaryHwSourceBackend, ScalarZnxFillTernaryProbBackend,
        ScalarZnxFillTernaryProbSourceBackend, ScratchOwnedAlloc, VecZnxAddAssignBackend, VecZnxAddConstAssignBackend,
        VecZnxAddConstIntoBackend, VecZnxAddIntoBackend, VecZnxAddNormalSourceBackend, VecZnxAddScalarAssignBackend,
        VecZnxAddScalarIntoBackend, VecZnxAutomorphismAssign, VecZnxAutomorphismAssignTmpBytes, VecZnxAutomorphismBackend,
        VecZnxCopyBackend, VecZnxCopyRangeBackend, VecZnxExtractCoeffBackend, VecZnxFillNormalBackend,
        VecZnxFillNormalSourceBackend, VecZnxFillUniformBackend, VecZnxFillUniformSourceBackend, VecZnxLshAddCoeffIntoBackend,
        VecZnxLshAssignBackend, VecZnxLshBackend, VecZnxLshCoeffBackend, VecZnxLshTmpBytes, VecZnxMergeRingsBackend,
        VecZnxMergeRingsTmpBytes, VecZnxMulXpMinusOneAssignBackend, VecZnxMulXpMinusOneAssignTmpBytes,
        VecZnxMulXpMinusOneBackend, VecZnxNegateAssignBackend, VecZnxNegateBackend, VecZnxNormalize,
        VecZnxNormalizeAssignBackend, VecZnxNormalizeCoeffAssignBackend, VecZnxNormalizeCoeffBackend, VecZnxNormalizeTmpBytes,
        VecZnxRotateAssignBackend, VecZnxRotateAssignTmpBytes, VecZnxRotateBackend, VecZnxRshAddCoeffIntoBackend,
        VecZnxRshAssignBackend, VecZnxRshBackend, VecZnxRshCoeffBackend, VecZnxRshSubCoeffIntoBackend, VecZnxRshTmpBytes,
        VecZnxSplitRingBackend, VecZnxSplitRingTmpBytes, VecZnxSubAssignBackend, VecZnxSubBackend, VecZnxSubNegateAssignBackend,
        VecZnxSubScalarAssignBackend, VecZnxSubScalarBackend, VecZnxSwitchRingBackend, VecZnxZeroBackend,
    },
    layouts::{
        DigestU64, FillUniform, HostBytesBackend, Module, NoiseInfos, ScalarZnx, ScalarZnxToBackendMut, ScratchOwned, VecZnx,
        ZnxView, ZnxViewMut, ZnxZero,
    },
    source::Source,
};

pub fn test_vec_znx_zero_backend_matches_wrapper<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    _module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BT>: VecZnxZeroBackend<BT>,
{
    let base2k = params.base2k;
    let _n: usize = module_test.n();
    let cols: usize = 2;
    let mut source: Source = Source::new([5u8; 32]);

    for size in [1, 2, 3, 4] {
        for col_i in 0..cols {
            let mut expected: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, size);
            expected.fill_uniform(base2k, &mut source);
            let mut backend = upload_vec_znx::<BT>(&expected);

            for limb in 0..size {
                expected.at_mut(col_i, limb).fill(0);
            }
            module_test.vec_znx_zero_backend(&mut vec_znx_backend_mut::<BT>(&mut backend), col_i);

            assert_eq!(expected, download_vec_znx::<BT>(&backend));
        }
    }
}

pub fn test_vec_znx_encode_vec_i64() {
    let n: usize = 32;
    let base2k: usize = 17;
    let size: usize = 5;
    let module = crate::layouts::Module::<crate::layouts::HostBytesBackend>::new(n as u64);
    for k in [1, base2k / 2, size * base2k - 5] {
        let mut a: VecZnx<Vec<u8>> = module.vec_znx_alloc(2, size);
        let mut source = Source::new([0u8; 32]);
        let raw: &mut [i64] = a.raw_mut();
        raw.iter_mut().enumerate().for_each(|(i, x)| *x = i as i64);
        (0..a.cols()).for_each(|col_i| {
            let mut have: Vec<i64> = vec![i64::default(); n];
            have.iter_mut().for_each(|x| {
                if k < 64 {
                    *x = source.next_u64n(1 << k, (1 << k) - 1) as i64;
                } else {
                    *x = source.next_i64();
                }
            });
            a.encode_vec_i64(base2k, col_i, k, &have);
            let mut want: Vec<i64> = vec![i64::default(); n];
            a.decode_vec_i64(base2k, col_i, k, &mut want);
            assert_eq!(have, want, "{:?} != {:?}", &have, &want);
        })
    }
}

pub fn test_vec_znx_add_scalar_into<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxAddScalarIntoBackend<BR>,
    Module<BT>: VecZnxAddScalarIntoBackend<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut a: ScalarZnx<Vec<u8>> = module_host.scalar_znx_alloc(cols);
    a.fill_uniform(base2k, &mut source);
    let a_digest = a.digest_u64();
    let a_ref = upload_scalar_znx::<BR>(&a);
    let a_test = upload_scalar_znx::<BT>(&a);

    for a_size in [1, 2, 3, 4] {
        let mut b: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        b.fill_uniform(base2k, &mut source);
        let b_digest: u64 = b.digest_u64();
        let b_ref = upload_vec_znx::<BR>(&b);
        let b_test = upload_vec_znx::<BT>(&b);

        for res_size in [1, 2, 3, 4] {
            let mut rest_ref: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
            let mut res_test: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);

            // Set d to garbage
            rest_ref.fill_uniform(base2k, &mut source);
            res_test.fill_uniform(base2k, &mut source);
            let mut rest_ref_backend = upload_vec_znx::<BR>(&rest_ref);
            let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

            // Reference
            for i in 0..cols {
                module_ref.vec_znx_add_scalar_into_backend(
                    &mut vec_znx_backend_mut::<BR>(&mut rest_ref_backend),
                    i,
                    &scalar_znx_backend_ref::<BR>(&a_ref),
                    i,
                    &vec_znx_backend_ref::<BR>(&b_ref),
                    i,
                    (res_size.min(a_size)) - 1,
                );
                module_test.vec_znx_add_scalar_into_backend(
                    &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                    i,
                    &scalar_znx_backend_ref::<BT>(&a_test),
                    i,
                    &vec_znx_backend_ref::<BT>(&b_test),
                    i,
                    (res_size.min(a_size)) - 1,
                );
            }

            assert_eq!(b.digest_u64(), b_digest);
            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(
                download_vec_znx::<BR>(&rest_ref_backend),
                download_vec_znx::<BT>(&res_test_backend)
            );
        }
    }
}

pub fn test_vec_znx_add_scalar_assign<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxAddScalarAssignBackend<BR>,
    Module<BT>: VecZnxAddScalarAssignBackend<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut b: ScalarZnx<Vec<u8>> = module_host.scalar_znx_alloc(cols);
    b.fill_uniform(base2k, &mut source);
    let b_digest: u64 = b.digest_u64();
    let b_ref = upload_scalar_znx::<BR>(&b);
    let b_test = upload_scalar_znx::<BT>(&b);

    for res_size in [1, 2, 3, 4] {
        let mut rest_ref: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
        let mut res_test: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);

        rest_ref.fill_uniform(base2k, &mut source);
        res_test.raw_mut().copy_from_slice(rest_ref.raw());
        let mut rest_ref_backend = upload_vec_znx::<BR>(&rest_ref);
        let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

        for i in 0..cols {
            module_ref.vec_znx_add_scalar_assign_backend(
                &mut vec_znx_backend_mut::<BR>(&mut rest_ref_backend),
                i,
                res_size - 1,
                &scalar_znx_backend_ref::<BR>(&b_ref),
                i,
            );
            module_test.vec_znx_add_scalar_assign_backend(
                &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                i,
                res_size - 1,
                &scalar_znx_backend_ref::<BT>(&b_test),
                i,
            );
        }

        assert_eq!(b.digest_u64(), b_digest);
        assert_eq!(
            download_vec_znx::<BR>(&rest_ref_backend),
            download_vec_znx::<BT>(&res_test_backend)
        );
    }
}

pub fn test_vec_znx_add_const_into<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxAddConstIntoBackend<BR>,
    Module<BT>: VecZnxAddConstIntoBackend<BT>,
{
    let base2k = params.base2k;
    let n = module_ref.n();
    assert_eq!(n, module_test.n());

    let cols: usize = 2;
    let mut source = Source::new([17u8; 32]);
    let cnst_digits: [i64; 3] = [11, -7, 23];
    let res_coeffs = [0usize, 1usize.min(n - 1), (n / 2).min(n - 1)];
    let mut cnst: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(1, cnst_digits.len());
    cnst.zero();
    for (limb, digit) in cnst_digits.iter().enumerate() {
        for coeff in res_coeffs {
            cnst.at_mut(0, limb)[coeff] = *digit;
        }
    }
    let cnst_ref = upload_vec_znx::<BR>(&cnst);
    let cnst_test = upload_vec_znx::<BT>(&cnst);

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let mut expected: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
            let mut actual: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
            expected.fill_uniform(base2k, &mut source);
            actual.data.copy_from_slice(&expected.data);
            let mut expected_backend = upload_vec_znx::<BR>(&expected);
            let mut actual_backend = upload_vec_znx::<BT>(&actual);

            for col_i in 0..cols {
                let res_limb = col_i.min(res_size - 1);
                let res_coeff = res_coeffs[col_i % res_coeffs.len()];
                module_ref.vec_znx_add_const_into_backend(
                    &mut vec_znx_backend_mut::<BR>(&mut expected_backend),
                    col_i,
                    &vec_znx_backend_ref::<BR>(&a_ref),
                    col_i,
                    &vec_znx_backend_ref::<BR>(&cnst_ref),
                    0,
                    res_coeff,
                    res_limb,
                    res_coeff,
                );
                module_test.vec_znx_add_const_into_backend(
                    &mut vec_znx_backend_mut::<BT>(&mut actual_backend),
                    col_i,
                    &vec_znx_backend_ref::<BT>(&a_test),
                    col_i,
                    &vec_znx_backend_ref::<BT>(&cnst_test),
                    0,
                    res_coeff,
                    res_limb,
                    res_coeff,
                );
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(
                download_vec_znx::<BR>(&expected_backend),
                download_vec_znx::<BT>(&actual_backend)
            );
        }
    }
}

pub fn test_vec_znx_add_const_assign<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxAddConstAssignBackend<BR>,
    Module<BT>: VecZnxAddConstAssignBackend<BT>,
{
    let base2k = params.base2k;
    let n = module_ref.n();
    assert_eq!(n, module_test.n());

    let cols: usize = 2;
    let mut source = Source::new([19u8; 32]);
    let cnst_digits: [i64; 4] = [5, -3, 9, 17];
    let res_coeffs = [0usize, 1usize.min(n - 1), (n / 2).min(n - 1)];
    let mut cnst: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(1, cnst_digits.len());
    cnst.zero();
    for (limb, digit) in cnst_digits.iter().enumerate() {
        for coeff in res_coeffs {
            cnst.at_mut(0, limb)[coeff] = *digit;
        }
    }
    let cnst_ref = upload_vec_znx::<BR>(&cnst);
    let cnst_test = upload_vec_znx::<BT>(&cnst);

    for res_size in [1, 2, 3, 4] {
        let mut expected: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
        let mut actual: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
        expected.fill_uniform(base2k, &mut source);
        actual.data.copy_from_slice(&expected.data);
        let mut expected_backend = upload_vec_znx::<BR>(&expected);
        let mut actual_backend = upload_vec_znx::<BT>(&actual);

        for col_i in 0..cols {
            let res_limb = col_i.min(res_size - 1);
            let res_coeff = res_coeffs[col_i % res_coeffs.len()];
            module_ref.vec_znx_add_const_assign_backend(
                &mut vec_znx_backend_mut::<BR>(&mut expected_backend),
                col_i,
                &vec_znx_backend_ref::<BR>(&cnst_ref),
                0,
                res_coeff,
                res_limb,
                res_coeff,
            );
            module_test.vec_znx_add_const_assign_backend(
                &mut vec_znx_backend_mut::<BT>(&mut actual_backend),
                col_i,
                &vec_znx_backend_ref::<BT>(&cnst_test),
                0,
                res_coeff,
                res_limb,
                res_coeff,
            );
        }

        assert_eq!(
            download_vec_znx::<BR>(&expected_backend),
            download_vec_znx::<BT>(&actual_backend)
        );
    }
}
pub fn test_vec_znx_add_into_backend_matches_reference<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxAddIntoBackend<BR>,
    Module<BT>: VecZnxAddIntoBackend<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let cols: usize = 2;
    let mut source: Source = Source::new([13u8; 32]);

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);

        for b_size in [1, 2, 3, 4] {
            let mut b: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, b_size);
            b.fill_uniform(base2k, &mut source);
            let b_digest: u64 = b.digest_u64();
            let b_ref = upload_vec_znx::<BR>(&b);
            let b_test = upload_vec_znx::<BT>(&b);

            for res_size in [1, 2, 3, 4] {
                let mut wrapper: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
                let mut backend: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);

                wrapper.fill_uniform(base2k, &mut source);
                backend.data.copy_from_slice(&wrapper.data);
                let mut wrapper_backend = upload_vec_znx::<BR>(&wrapper);
                let mut backend_owned = upload_vec_znx::<BT>(&backend);

                for col_i in 0..cols {
                    module_ref.vec_znx_add_into_backend(
                        &mut vec_znx_backend_mut::<BR>(&mut wrapper_backend),
                        col_i,
                        &vec_znx_backend_ref::<BR>(&a_ref),
                        col_i,
                        &vec_znx_backend_ref::<BR>(&b_ref),
                        col_i,
                    );
                    module_test.vec_znx_add_into_backend(
                        &mut vec_znx_backend_mut::<BT>(&mut backend_owned),
                        col_i,
                        &vec_znx_backend_ref::<BT>(&a_test),
                        col_i,
                        &vec_znx_backend_ref::<BT>(&b_test),
                        col_i,
                    );
                }

                assert_eq!(a.digest_u64(), a_digest);
                assert_eq!(b.digest_u64(), b_digest);
                assert_eq!(
                    download_vec_znx::<BR>(&wrapper_backend),
                    download_vec_znx::<BT>(&backend_owned)
                );
            }
        }
    }
}

pub fn test_vec_znx_add_assign<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxAddAssignBackend<BR>,
    Module<BT>: VecZnxAddAssignBackend<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let mut res_ref: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
            let mut res_test: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);

            res_ref.fill_uniform(base2k, &mut source);
            res_test.raw_mut().copy_from_slice(res_ref.raw());
            let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
            let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

            for i in 0..cols {
                module_ref.vec_znx_add_assign_backend(
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                    i,
                    &vec_znx_backend_ref::<BR>(&a_ref),
                    i,
                );
                module_test.vec_znx_add_assign_backend(
                    &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                    i,
                    &vec_znx_backend_ref::<BT>(&a_test),
                    i,
                );
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(
                download_vec_znx::<BR>(&res_ref_backend),
                download_vec_znx::<BT>(&res_test_backend)
            );
        }
    }
}

pub fn test_vec_znx_add_assign_backend_matches_wrapper<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    _module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BT>: VecZnxAddAssignBackend<BT>,
{
    let base2k = params.base2k;
    let _n: usize = module_test.n();
    let cols: usize = 2;
    let mut source: Source = Source::new([14u8; 32]);

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();
        let a_backend = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let mut wrapper: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
            let mut backend: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);

            wrapper.fill_uniform(base2k, &mut source);
            backend.data.copy_from_slice(&wrapper.data);
            let mut wrapper_backend = upload_vec_znx::<BT>(&wrapper);
            let mut backend_backend = upload_vec_znx::<BT>(&backend);

            for col_i in 0..cols {
                module_test.vec_znx_add_assign_backend(
                    &mut vec_znx_backend_mut::<BT>(&mut wrapper_backend),
                    col_i,
                    &vec_znx_backend_ref::<BT>(&a_backend),
                    col_i,
                );
                module_test.vec_znx_add_assign_backend(
                    &mut vec_znx_backend_mut::<BT>(&mut backend_backend),
                    col_i,
                    &vec_znx_backend_ref::<BT>(&a_backend),
                    col_i,
                );
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(
                download_vec_znx::<BT>(&wrapper_backend),
                download_vec_znx::<BT>(&backend_backend)
            );
        }
    }
}

pub fn test_vec_znx_automorphism<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxAutomorphismBackend<BR>,
    Module<BT>: VecZnxAutomorphismBackend<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let res_ref: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
            let res_test: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
            let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
            let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

            let p: i64 = -5;

            // Normalize on c
            for i in 0..cols {
                module_ref.vec_znx_automorphism_backend(
                    p,
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                    i,
                    &vec_znx_backend_ref::<BR>(&a_ref),
                    i,
                );
                module_test.vec_znx_automorphism_backend(
                    p,
                    &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                    i,
                    &vec_znx_backend_ref::<BT>(&a_test),
                    i,
                );
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(
                download_vec_znx::<BR>(&res_ref_backend),
                download_vec_znx::<BT>(&res_test_backend)
            );

            let p: i64 = 5;

            // Normalize on c
            for i in 0..cols {
                module_ref.vec_znx_automorphism_backend(
                    p,
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                    i,
                    &vec_znx_backend_ref::<BR>(&a_ref),
                    i,
                );
                module_test.vec_znx_automorphism_backend(
                    p,
                    &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                    i,
                    &vec_znx_backend_ref::<BT>(&a_test),
                    i,
                );
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(
                download_vec_znx::<BR>(&res_ref_backend),
                download_vec_znx::<BT>(&res_test_backend)
            );
        }
    }
}

pub fn test_vec_znx_automorphism_assign<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxAutomorphismAssign<BR> + VecZnxAutomorphismAssignTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxAutomorphismAssign<BT> + VecZnxAutomorphismAssignTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_automorphism_assign_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_automorphism_assign_tmp_bytes());

    for size in [1, 2, 3, 4] {
        let mut res_ref: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, size);
        let mut res_test: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, size);

        // Fill a with random i64
        res_ref.fill_uniform(base2k, &mut source);
        res_test.raw_mut().copy_from_slice(res_ref.raw());
        let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
        let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

        let p: i64 = -7;

        // Normalize on c
        for i in 0..cols {
            module_ref.vec_znx_automorphism_assign(
                p,
                &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                i,
                &mut scratch_ref.arena(),
            );
            module_test.vec_znx_automorphism_assign(
                p,
                &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                i,
                &mut scratch_test.arena(),
            );
        }

        assert_eq!(
            download_vec_znx::<BR>(&res_ref_backend),
            download_vec_znx::<BT>(&res_test_backend)
        );

        let p: i64 = 7;

        // Normalize on c
        for i in 0..cols {
            module_ref.vec_znx_automorphism_assign(
                p,
                &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                i,
                &mut scratch_ref.arena(),
            );
            module_test.vec_znx_automorphism_assign(
                p,
                &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                i,
                &mut scratch_test.arena(),
            );
        }

        assert_eq!(
            download_vec_znx::<BR>(&res_ref_backend),
            download_vec_znx::<BT>(&res_test_backend)
        );
    }
}

pub fn test_vec_znx_copy<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxCopyBackend<BR>,
    Module<BT>: VecZnxCopyBackend<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let mut res_0: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
            let mut res_1: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);

            // Set d to garbage
            res_0.fill_uniform(base2k, &mut source);
            res_1.fill_uniform(base2k, &mut source);
            let mut res_0_backend = upload_vec_znx::<BR>(&res_0);
            let mut res_1_backend = upload_vec_znx::<BT>(&res_1);

            // Reference
            for i in 0..cols {
                module_ref.vec_znx_copy_backend(
                    &mut vec_znx_backend_mut::<BR>(&mut res_0_backend),
                    i,
                    &vec_znx_backend_ref::<BR>(&a_ref),
                    i,
                );
                module_test.vec_znx_copy_backend(
                    &mut vec_znx_backend_mut::<BT>(&mut res_1_backend),
                    i,
                    &vec_znx_backend_ref::<BT>(&a_test),
                    i,
                );
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(download_vec_znx::<BR>(&res_0_backend), download_vec_znx::<BT>(&res_1_backend));
        }
    }
}

pub fn test_vec_znx_copy_backend_matches_wrapper<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    _module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BT>: VecZnxCopyBackend<BT>,
{
    let base2k = params.base2k;
    let _n: usize = module_test.n();
    let cols: usize = 2;
    let a_col: usize = 0;
    let res_col: usize = 1;
    let mut source: Source = Source::new([3u8; 32]);

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_backend = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let mut wrapper: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
            let mut backend: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
            wrapper.fill_uniform(base2k, &mut source);
            backend.data.copy_from_slice(&wrapper.data);
            let mut wrapper_backend = upload_vec_znx::<BT>(&wrapper);
            let mut backend_backend = upload_vec_znx::<BT>(&backend);

            module_test.vec_znx_copy_backend(
                &mut vec_znx_backend_mut::<BT>(&mut wrapper_backend),
                res_col,
                &vec_znx_backend_ref::<BT>(&a_backend),
                a_col,
            );
            module_test.vec_znx_copy_backend(
                &mut vec_znx_backend_mut::<BT>(&mut backend_backend),
                res_col,
                &vec_znx_backend_ref::<BT>(&a_backend),
                a_col,
            );

            assert_eq!(
                download_vec_znx::<BT>(&wrapper_backend),
                download_vec_znx::<BT>(&backend_backend)
            );
        }
    }
}

pub fn test_vec_znx_copy_range_backend<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    _module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BT>: VecZnxCopyRangeBackend<BT>,
{
    let base2k = params.base2k;
    let n: usize = module_test.n();
    let cols: usize = 2;
    let a_col: usize = 0;
    let res_col: usize = 1;
    let mut source: Source = Source::new([13u8; 32]);

    for a_size in [1, 2, 3, 4] {
        let a_limb = a_size - 1;
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_backend = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let res_limb = res_size - 1;
            let mut expected: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
            let mut actual: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
            expected.fill_uniform(base2k, &mut source);
            actual.data.copy_from_slice(&expected.data);
            let mut actual_backend = upload_vec_znx::<BT>(&actual);

            for (res_offset, a_offset, len) in [(0usize, 0usize, 1usize), (1, 0, 2), (0, 1, 3), (2, 4, 5)] {
                if res_offset + len > n || a_offset + len > n {
                    continue;
                }

                expected.at_mut(res_col, res_limb)[res_offset..res_offset + len]
                    .copy_from_slice(&a.at(a_col, a_limb)[a_offset..a_offset + len]);

                module_test.vec_znx_copy_range_backend(
                    &mut vec_znx_backend_mut::<BT>(&mut actual_backend),
                    res_col,
                    res_limb,
                    res_offset,
                    &vec_znx_backend_ref::<BT>(&a_backend),
                    a_col,
                    a_limb,
                    a_offset,
                    len,
                );
            }

            assert_eq!(expected, download_vec_znx::<BT>(&actual_backend));
        }
    }
}

pub fn test_vec_znx_merge_rings<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxMergeRingsBackend<BR> + ModuleNew<BR> + VecZnxMergeRingsTmpBytes,
    Module<BT>: VecZnxMergeRingsBackend<BT> + ModuleNew<BT> + VecZnxMergeRingsTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;
    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_merge_rings_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_merge_rings_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: [VecZnx<Vec<u8>>; 2] = [VecZnx::alloc(n >> 1, cols, a_size), VecZnx::alloc(n >> 1, cols, a_size)];

        a.iter_mut().for_each(|ai| {
            ai.fill_uniform(base2k, &mut source);
        });

        let a_digests: [u64; 2] = [a[0].digest_u64(), a[1].digest_u64()];
        let a_ref_backend = a.each_ref().map(upload_vec_znx::<BR>);
        let a_test_backend = a.each_ref().map(upload_vec_znx::<BT>);

        for res_size in [1, 2, 3, 4] {
            let mut res_ref: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
            let mut res_test: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);

            res_ref.fill_uniform(base2k, &mut source);
            res_test.fill_uniform(base2k, &mut source);
            let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
            let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

            for i in 0..cols {
                let a_ref: Vec<_> = a_ref_backend.iter().map(vec_znx_backend_ref::<BR>).collect();
                module_ref.vec_znx_merge_rings_backend(
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                    i,
                    &a_ref,
                    i,
                    &mut scratch_ref.arena(),
                );
                let a_test: Vec<_> = a_test_backend.iter().map(vec_znx_backend_ref::<BT>).collect();
                module_test.vec_znx_merge_rings_backend(
                    &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                    i,
                    &a_test,
                    i,
                    &mut scratch_test.arena(),
                );
            }

            assert_eq!([a[0].digest_u64(), a[1].digest_u64()], a_digests);
            assert_eq!(
                download_vec_znx::<BR>(&res_ref_backend),
                download_vec_znx::<BT>(&res_test_backend)
            );
        }
    }
}

pub fn test_vec_znx_mul_xp_minus_one<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxMulXpMinusOneBackend<BR>,
    Module<BT>: VecZnxMulXpMinusOneBackend<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);

        let a_digest: u64 = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let res_ref: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
            let res_test: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
            let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
            let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

            let p: i64 = -5;

            // Normalize on c
            for i in 0..cols {
                module_ref.vec_znx_mul_xp_minus_one_backend(
                    p,
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                    i,
                    &vec_znx_backend_ref::<BR>(&a_ref),
                    i,
                );
                module_test.vec_znx_mul_xp_minus_one_backend(
                    p,
                    &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                    i,
                    &vec_znx_backend_ref::<BT>(&a_test),
                    i,
                );
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(
                download_vec_znx::<BT>(&res_test_backend),
                download_vec_znx::<BR>(&res_ref_backend)
            );

            let p: i64 = 5;

            // Normalize on c
            for i in 0..cols {
                module_ref.vec_znx_mul_xp_minus_one_backend(
                    p,
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                    i,
                    &vec_znx_backend_ref::<BR>(&a_ref),
                    i,
                );
                module_test.vec_znx_mul_xp_minus_one_backend(
                    p,
                    &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                    i,
                    &vec_znx_backend_ref::<BT>(&a_test),
                    i,
                );
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(
                download_vec_znx::<BT>(&res_test_backend),
                download_vec_znx::<BR>(&res_ref_backend)
            );
        }
    }
}

pub fn test_vec_znx_mul_xp_minus_one_assign<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxMulXpMinusOneAssignBackend<BR> + VecZnxMulXpMinusOneAssignTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxMulXpMinusOneAssignBackend<BT> + VecZnxMulXpMinusOneAssignTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_mul_xp_minus_one_assign_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_mul_xp_minus_one_assign_tmp_bytes());

    for size in [1, 2, 3, 4] {
        let mut res_ref: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, size);
        let mut res_test: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, size);

        // Fill a with random i64
        res_ref.fill_uniform(base2k, &mut source);
        res_test.raw_mut().copy_from_slice(res_ref.raw());
        let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
        let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

        let p: i64 = -7;

        for i in 0..cols {
            module_ref.vec_znx_mul_xp_minus_one_assign_backend(
                p,
                &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                i,
                &mut scratch_ref.arena(),
            );
            module_test.vec_znx_mul_xp_minus_one_assign_backend(
                p,
                &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                i,
                &mut scratch_test.arena(),
            );
        }

        assert_eq!(
            download_vec_znx::<BR>(&res_ref_backend),
            download_vec_znx::<BT>(&res_test_backend)
        );

        let p: i64 = 7;

        for i in 0..cols {
            module_ref.vec_znx_mul_xp_minus_one_assign_backend(
                p,
                &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                i,
                &mut scratch_ref.arena(),
            );
            module_test.vec_znx_mul_xp_minus_one_assign_backend(
                p,
                &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                i,
                &mut scratch_test.arena(),
            );
        }

        assert_eq!(
            download_vec_znx::<BR>(&res_ref_backend),
            download_vec_znx::<BT>(&res_test_backend)
        );
    }
}

pub fn test_vec_znx_negate<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxNegateBackend<BR>,
    Module<BT>: VecZnxNegateBackend<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let mut res_ref: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
            let mut res_test: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);

            res_ref.fill_uniform(base2k, &mut source);
            res_test.raw_mut().copy_from_slice(res_ref.raw());
            let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
            let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

            for i in 0..cols {
                module_ref.vec_znx_negate_backend(
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                    i,
                    &vec_znx_backend_ref::<BR>(&a_ref),
                    i,
                );
                module_test.vec_znx_negate_backend(
                    &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                    i,
                    &vec_znx_backend_ref::<BT>(&a_test),
                    i,
                );
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(
                download_vec_znx::<BR>(&res_ref_backend),
                download_vec_znx::<BT>(&res_test_backend)
            );
        }
    }
}

pub fn test_vec_znx_negate_backend_matches_wrapper<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    _module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BT>: VecZnxNegateBackend<BT>,
{
    let base2k = params.base2k;
    let _n: usize = module_test.n();
    let cols: usize = 2;
    let a_col: usize = 0;
    let res_col: usize = 1;
    let mut source: Source = Source::new([6u8; 32]);

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_backend = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let mut wrapper: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
            let mut backend: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
            wrapper.fill_uniform(base2k, &mut source);
            backend.data.copy_from_slice(&wrapper.data);
            let mut wrapper_backend = upload_vec_znx::<BT>(&wrapper);
            let mut backend_backend = upload_vec_znx::<BT>(&backend);

            module_test.vec_znx_negate_backend(
                &mut vec_znx_backend_mut::<BT>(&mut wrapper_backend),
                res_col,
                &vec_znx_backend_ref::<BT>(&a_backend),
                a_col,
            );
            module_test.vec_znx_negate_backend(
                &mut vec_znx_backend_mut::<BT>(&mut backend_backend),
                res_col,
                &vec_znx_backend_ref::<BT>(&a_backend),
                a_col,
            );

            assert_eq!(
                download_vec_znx::<BT>(&wrapper_backend),
                download_vec_znx::<BT>(&backend_backend)
            );
        }
    }
}

pub fn test_vec_znx_negate_assign<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxNegateAssignBackend<BR>,
    Module<BT>: VecZnxNegateAssignBackend<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for res_size in [1, 2, 3, 4] {
        let mut res_ref: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
        let mut res_test: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);

        res_ref.fill_uniform(base2k, &mut source);
        res_test.raw_mut().copy_from_slice(res_ref.raw());
        let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
        let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

        for i in 0..cols {
            module_ref.vec_znx_negate_assign_backend(&mut vec_znx_backend_mut::<BR>(&mut res_ref_backend), i);
            module_test.vec_znx_negate_assign_backend(&mut vec_znx_backend_mut::<BT>(&mut res_test_backend), i);
        }

        assert_eq!(
            download_vec_znx::<BR>(&res_ref_backend),
            download_vec_znx::<BT>(&res_test_backend)
        );
    }
}

pub fn test_vec_znx_negate_assign_backend_matches_wrapper<
    BR: crate::test_suite::TestBackend,
    BT: crate::test_suite::TestBackend,
>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    _module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BT>: VecZnxNegateAssignBackend<BT>,
{
    let base2k = params.base2k;
    let _n: usize = module_test.n();
    let cols: usize = 2;
    let mut source: Source = Source::new([7u8; 32]);

    for res_size in [1, 2, 3, 4] {
        for col_i in 0..cols {
            let mut wrapper: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
            let mut backend: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
            wrapper.fill_uniform(base2k, &mut source);
            backend.data.copy_from_slice(&wrapper.data);
            let mut wrapper_backend = upload_vec_znx::<BT>(&wrapper);
            let mut backend_backend = upload_vec_znx::<BT>(&backend);

            module_test.vec_znx_negate_assign_backend(&mut vec_znx_backend_mut::<BT>(&mut wrapper_backend), col_i);
            module_test.vec_znx_negate_assign_backend(&mut vec_znx_backend_mut::<BT>(&mut backend_backend), col_i);

            assert_eq!(
                download_vec_znx::<BT>(&wrapper_backend),
                download_vec_znx::<BT>(&backend_backend)
            );
        }
    }
}

pub fn test_vec_znx_normalize<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxNormalize<BR> + VecZnxNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxNormalize<BT> + VecZnxNormalizeTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let mut res_ref: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
            let mut res_test: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);

            for res_offset in -(base2k as i64)..=(base2k as i64) {
                // Set d to garbage
                res_ref.fill_uniform(base2k, &mut source);
                res_test.fill_uniform(base2k, &mut source);
                let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
                let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_normalize(
                        &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                        base2k,
                        res_offset,
                        i,
                        &vec_znx_backend_ref::<BR>(&a_ref),
                        base2k,
                        i,
                        &mut scratch_ref.arena(),
                    );
                    module_test.vec_znx_normalize(
                        &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                        base2k,
                        res_offset,
                        i,
                        &vec_znx_backend_ref::<BT>(&a_test),
                        base2k,
                        i,
                        &mut scratch_test.arena(),
                    );
                }

                assert_eq!(a.digest_u64(), a_digest);
                assert_eq!(
                    download_vec_znx::<BR>(&res_ref_backend),
                    download_vec_znx::<BT>(&res_test_backend)
                );
            }
        }
    }
}

pub fn test_vec_znx_normalize_assign<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxNormalizeAssignBackend<BR> + VecZnxNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxNormalizeAssignBackend<BT> + VecZnxNormalizeTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_normalize_tmp_bytes());

    for res_size in [1, 2, 3, 4] {
        let mut res_ref: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
        let mut res_test: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);

        res_ref.fill_uniform(base2k, &mut source);
        res_test.raw_mut().copy_from_slice(res_ref.raw());
        let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
        let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

        // Reference
        for i in 0..cols {
            module_ref.vec_znx_normalize_assign_backend(
                base2k,
                &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                i,
                &mut scratch_ref.arena(),
            );
            module_test.vec_znx_normalize_assign_backend(
                base2k,
                &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                i,
                &mut scratch_test.arena(),
            );
        }

        assert_eq!(
            download_vec_znx::<BR>(&res_ref_backend),
            download_vec_znx::<BT>(&res_test_backend)
        );
    }
}

pub fn test_vec_znx_normalize_coeff_backend<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxNormalizeCoeffBackend<BR> + VecZnxNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxNormalizeCoeffBackend<BT> + VecZnxNormalizeTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    let n = module_ref.n();
    assert_eq!(n, module_test.n());

    let mut source = Source::new([41u8; 32]);
    let cols: usize = 2;
    let coeffs = [0usize, 1usize.min(n - 1), (n / 2).min(n - 1), n - 1];

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            for res_offset in -(base2k as i64)..=(base2k as i64) {
                let expected =
                    VecZnx::from_data_with_max_size(vec![0u8; VecZnx::<Vec<u8>>::bytes_of(1, cols, 4)], 1, cols, res_size, 4);
                let actual =
                    VecZnx::from_data_with_max_size(vec![0u8; VecZnx::<Vec<u8>>::bytes_of(1, cols, 4)], 1, cols, res_size, 4);
                let mut expected_backend = upload_vec_znx::<BR>(&expected);
                let mut actual_backend = upload_vec_znx::<BT>(&actual);

                for col_i in 0..cols {
                    let coeff = coeffs[(col_i + a_size + res_size) % coeffs.len()];
                    module_ref.vec_znx_normalize_coeff_backend(
                        &mut vec_znx_backend_mut::<BR>(&mut expected_backend),
                        base2k,
                        res_offset,
                        col_i,
                        &vec_znx_backend_ref::<BR>(&a_ref),
                        base2k,
                        col_i,
                        coeff,
                        &mut scratch_ref.arena(),
                    );
                    module_test.vec_znx_normalize_coeff_backend(
                        &mut vec_znx_backend_mut::<BT>(&mut actual_backend),
                        base2k,
                        res_offset,
                        col_i,
                        &vec_znx_backend_ref::<BT>(&a_test),
                        base2k,
                        col_i,
                        coeff,
                        &mut scratch_test.arena(),
                    );
                }

                assert_eq!(a.digest_u64(), a_digest);
                assert_eq!(
                    download_vec_znx::<BR>(&expected_backend),
                    download_vec_znx::<BT>(&actual_backend)
                );
            }
        }
    }
}

pub fn test_vec_znx_normalize_coeff_assign_backend<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxNormalizeCoeffAssignBackend<BR> + VecZnxNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxNormalizeCoeffAssignBackend<BT> + VecZnxNormalizeTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    let n = module_ref.n();
    assert_eq!(n, module_test.n());
    let mut source = Source::new([37u8; 32]);
    let cols: usize = 2;
    let coeffs = [0usize, 1usize.min(n - 1), (n / 2).min(n - 1), n - 1];
    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_normalize_tmp_bytes());

    for res_size in [1, 2, 3, 4] {
        let mut expected: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
        let mut actual: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
        expected.fill_uniform(base2k, &mut source);
        actual.raw_mut().copy_from_slice(expected.raw());
        let mut expected_backend = upload_vec_znx::<BR>(&expected);
        let mut actual_backend = upload_vec_znx::<BT>(&actual);

        for col_i in 0..cols {
            let coeff = coeffs[(col_i + res_size) % coeffs.len()];
            module_ref.vec_znx_normalize_coeff_assign_backend(
                base2k,
                &mut vec_znx_backend_mut::<BR>(&mut expected_backend),
                col_i,
                coeff,
                &mut scratch_ref.arena(),
            );
            module_test.vec_znx_normalize_coeff_assign_backend(
                base2k,
                &mut vec_znx_backend_mut::<BT>(&mut actual_backend),
                col_i,
                coeff,
                &mut scratch_test.arena(),
            );
        }

        assert_eq!(
            download_vec_znx::<BR>(&expected_backend),
            download_vec_znx::<BT>(&actual_backend)
        );
    }
}

pub fn test_vec_znx_lsh_coeff_backend<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxLshCoeffBackend<BR> + VecZnxLshTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxLshCoeffBackend<BT> + VecZnxLshTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    let n = module_ref.n();
    assert_eq!(n, module_test.n());
    let mut source = Source::new([43u8; 32]);
    let cols: usize = 2;
    let coeffs = [0usize, 1usize.min(n - 1), (n / 2).min(n - 1), n - 1];
    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_lsh_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_lsh_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);
        for res_size in [1, 2, 3, 4] {
            for k in 0..=(base2k * (a_size + 1)) {
                let expected =
                    VecZnx::from_data_with_max_size(vec![0u8; VecZnx::<Vec<u8>>::bytes_of(1, cols, 4)], 1, cols, res_size, 4);
                let actual =
                    VecZnx::from_data_with_max_size(vec![0u8; VecZnx::<Vec<u8>>::bytes_of(1, cols, 4)], 1, cols, res_size, 4);
                let mut expected_backend = upload_vec_znx::<BR>(&expected);
                let mut actual_backend = upload_vec_znx::<BT>(&actual);
                for col_i in 0..cols {
                    let coeff = coeffs[(col_i + a_size + res_size) % coeffs.len()];
                    module_ref.vec_znx_lsh_coeff_backend(
                        base2k,
                        k,
                        &mut vec_znx_backend_mut::<BR>(&mut expected_backend),
                        col_i,
                        &vec_znx_backend_ref::<BR>(&a_ref),
                        col_i,
                        coeff,
                        &mut scratch_ref.arena(),
                    );
                    module_test.vec_znx_lsh_coeff_backend(
                        base2k,
                        k,
                        &mut vec_znx_backend_mut::<BT>(&mut actual_backend),
                        col_i,
                        &vec_znx_backend_ref::<BT>(&a_test),
                        col_i,
                        coeff,
                        &mut scratch_test.arena(),
                    );
                }
                assert_eq!(a.digest_u64(), a_digest);
                assert_eq!(
                    download_vec_znx::<BR>(&expected_backend),
                    download_vec_znx::<BT>(&actual_backend)
                );
            }
        }
    }
}

pub fn test_vec_znx_lsh_add_coeff_into_backend<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxLshAddCoeffIntoBackend<BR> + VecZnxLshTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxLshAddCoeffIntoBackend<BT> + VecZnxLshTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    let n = module_ref.n();
    assert_eq!(n, module_test.n());
    let mut source = Source::new([47u8; 32]);
    let cols: usize = 2;
    let coeffs = [0usize, 1usize.min(n - 1), (n / 2).min(n - 1), n - 1];
    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_lsh_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_lsh_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);
        for res_size in [1, 2, 3, 4] {
            for k in 0..=(base2k * (a_size + 1)) {
                let mut expected =
                    VecZnx::from_data_with_max_size(vec![0u8; VecZnx::<Vec<u8>>::bytes_of(1, cols, 4)], 1, cols, res_size, 4);
                let mut actual =
                    VecZnx::from_data_with_max_size(vec![0u8; VecZnx::<Vec<u8>>::bytes_of(1, cols, 4)], 1, cols, res_size, 4);
                expected.fill_uniform(base2k, &mut source);
                actual.raw_mut().copy_from_slice(expected.raw());
                let mut expected_backend = upload_vec_znx::<BR>(&expected);
                let mut actual_backend = upload_vec_znx::<BT>(&actual);
                for col_i in 0..cols {
                    let coeff = coeffs[(col_i + a_size + res_size) % coeffs.len()];
                    module_ref.vec_znx_lsh_add_coeff_into_backend(
                        base2k,
                        k,
                        &mut vec_znx_backend_mut::<BR>(&mut expected_backend),
                        col_i,
                        &vec_znx_backend_ref::<BR>(&a_ref),
                        col_i,
                        coeff,
                        &mut scratch_ref.arena(),
                    );
                    module_test.vec_znx_lsh_add_coeff_into_backend(
                        base2k,
                        k,
                        &mut vec_znx_backend_mut::<BT>(&mut actual_backend),
                        col_i,
                        &vec_znx_backend_ref::<BT>(&a_test),
                        col_i,
                        coeff,
                        &mut scratch_test.arena(),
                    );
                }
                assert_eq!(a.digest_u64(), a_digest);
                assert_eq!(
                    download_vec_znx::<BR>(&expected_backend),
                    download_vec_znx::<BT>(&actual_backend)
                );
            }
        }
    }
}

pub fn test_vec_znx_rsh_coeff_backend<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxRshCoeffBackend<BR> + VecZnxRshTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxRshCoeffBackend<BT> + VecZnxRshTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    let n = module_ref.n();
    assert_eq!(n, module_test.n());
    let mut source = Source::new([53u8; 32]);
    let cols: usize = 2;
    let coeffs = [0usize, 1usize.min(n - 1), (n / 2).min(n - 1), n - 1];
    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_rsh_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_rsh_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);
        for res_size in [1, 2, 3, 4] {
            for k in 0..=(base2k * (a_size + 1)) {
                let expected =
                    VecZnx::from_data_with_max_size(vec![0u8; VecZnx::<Vec<u8>>::bytes_of(1, cols, 4)], 1, cols, res_size, 4);
                let actual =
                    VecZnx::from_data_with_max_size(vec![0u8; VecZnx::<Vec<u8>>::bytes_of(1, cols, 4)], 1, cols, res_size, 4);
                let mut expected_backend = upload_vec_znx::<BR>(&expected);
                let mut actual_backend = upload_vec_znx::<BT>(&actual);
                for col_i in 0..cols {
                    let coeff = coeffs[(col_i + a_size + res_size) % coeffs.len()];
                    module_ref.vec_znx_rsh_coeff_backend(
                        base2k,
                        k,
                        &mut vec_znx_backend_mut::<BR>(&mut expected_backend),
                        col_i,
                        &vec_znx_backend_ref::<BR>(&a_ref),
                        col_i,
                        coeff,
                        &mut scratch_ref.arena(),
                    );
                    module_test.vec_znx_rsh_coeff_backend(
                        base2k,
                        k,
                        &mut vec_znx_backend_mut::<BT>(&mut actual_backend),
                        col_i,
                        &vec_znx_backend_ref::<BT>(&a_test),
                        col_i,
                        coeff,
                        &mut scratch_test.arena(),
                    );
                }
                assert_eq!(a.digest_u64(), a_digest);
                assert_eq!(
                    download_vec_znx::<BR>(&expected_backend),
                    download_vec_znx::<BT>(&actual_backend)
                );
            }
        }
    }
}

pub fn test_vec_znx_rsh_add_coeff_into_backend<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxRshAddCoeffIntoBackend<BR> + VecZnxRshTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxRshAddCoeffIntoBackend<BT> + VecZnxRshTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    let n = module_ref.n();
    assert_eq!(n, module_test.n());
    let mut source = Source::new([59u8; 32]);
    let cols: usize = 2;
    let coeffs = [0usize, 1usize.min(n - 1), (n / 2).min(n - 1), n - 1];
    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_rsh_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_rsh_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);
        for res_size in [1, 2, 3, 4] {
            for k in 0..=(base2k * (a_size + 1)) {
                let mut expected =
                    VecZnx::from_data_with_max_size(vec![0u8; VecZnx::<Vec<u8>>::bytes_of(n, cols, 4)], n, cols, res_size, 4);
                let mut actual =
                    VecZnx::from_data_with_max_size(vec![0u8; VecZnx::<Vec<u8>>::bytes_of(n, cols, 4)], n, cols, res_size, 4);
                expected.fill_uniform(base2k, &mut source);
                actual.raw_mut().copy_from_slice(expected.raw());
                let mut expected_backend = upload_vec_znx::<BR>(&expected);
                let mut actual_backend = upload_vec_znx::<BT>(&actual);
                for col_i in 0..cols {
                    let coeff = coeffs[(col_i + a_size + res_size) % coeffs.len()];
                    let res_coeff = coeffs[(col_i + 1 + a_size + res_size) % coeffs.len()];
                    module_ref.vec_znx_rsh_add_coeff_into_backend(
                        base2k,
                        k,
                        &mut vec_znx_backend_mut::<BR>(&mut expected_backend),
                        col_i,
                        &vec_znx_backend_ref::<BR>(&a_ref),
                        col_i,
                        coeff,
                        res_coeff,
                        &mut scratch_ref.arena(),
                    );
                    module_test.vec_znx_rsh_add_coeff_into_backend(
                        base2k,
                        k,
                        &mut vec_znx_backend_mut::<BT>(&mut actual_backend),
                        col_i,
                        &vec_znx_backend_ref::<BT>(&a_test),
                        col_i,
                        coeff,
                        res_coeff,
                        &mut scratch_test.arena(),
                    );
                }
                assert_eq!(a.digest_u64(), a_digest);
                assert_eq!(
                    download_vec_znx::<BR>(&expected_backend),
                    download_vec_znx::<BT>(&actual_backend)
                );
            }
        }
    }
}

pub fn test_vec_znx_rsh_sub_coeff_into_backend<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxRshSubCoeffIntoBackend<BR> + VecZnxRshTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxRshSubCoeffIntoBackend<BT> + VecZnxRshTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    let n = module_ref.n();
    assert_eq!(n, module_test.n());
    let mut source = Source::new([67u8; 32]);
    let cols: usize = 2;
    let coeffs = [0usize, 1usize.min(n - 1), (n / 2).min(n - 1), n - 1];
    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_rsh_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_rsh_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);
        for res_size in [1, 2, 3, 4] {
            for k in 0..=(base2k * (a_size + 1)) {
                let mut expected =
                    VecZnx::from_data_with_max_size(vec![0u8; VecZnx::<Vec<u8>>::bytes_of(n, cols, 4)], n, cols, res_size, 4);
                let mut actual =
                    VecZnx::from_data_with_max_size(vec![0u8; VecZnx::<Vec<u8>>::bytes_of(n, cols, 4)], n, cols, res_size, 4);
                expected.fill_uniform(base2k, &mut source);
                actual.raw_mut().copy_from_slice(expected.raw());
                let mut expected_backend = upload_vec_znx::<BR>(&expected);
                let mut actual_backend = upload_vec_znx::<BT>(&actual);
                for col_i in 0..cols {
                    let coeff = coeffs[(col_i + a_size + res_size) % coeffs.len()];
                    let res_coeff = coeffs[(col_i + 1 + a_size + res_size) % coeffs.len()];
                    module_ref.vec_znx_rsh_sub_coeff_into_backend(
                        base2k,
                        k,
                        &mut vec_znx_backend_mut::<BR>(&mut expected_backend),
                        col_i,
                        &vec_znx_backend_ref::<BR>(&a_ref),
                        col_i,
                        coeff,
                        res_coeff,
                        &mut scratch_ref.arena(),
                    );
                    module_test.vec_znx_rsh_sub_coeff_into_backend(
                        base2k,
                        k,
                        &mut vec_znx_backend_mut::<BT>(&mut actual_backend),
                        col_i,
                        &vec_znx_backend_ref::<BT>(&a_test),
                        col_i,
                        coeff,
                        res_coeff,
                        &mut scratch_test.arena(),
                    );
                }
                assert_eq!(a.digest_u64(), a_digest);
                assert_eq!(
                    download_vec_znx::<BR>(&expected_backend),
                    download_vec_znx::<BT>(&actual_backend)
                );
            }
        }
    }
}

pub fn test_vec_znx_rotate<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxRotateBackend<BR>,
    Module<BT>: VecZnxRotateBackend<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let res_ref: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
            let res_test: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
            let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
            let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

            let p: i64 = -5;

            // Normalize on c
            for i in 0..cols {
                module_ref.vec_znx_rotate_backend(
                    p,
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                    i,
                    &vec_znx_backend_ref::<BR>(&a_ref),
                    i,
                );
                module_test.vec_znx_rotate_backend(
                    p,
                    &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                    i,
                    &vec_znx_backend_ref::<BT>(&a_test),
                    i,
                );
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(
                download_vec_znx::<BR>(&res_ref_backend),
                download_vec_znx::<BT>(&res_test_backend)
            );

            let p: i64 = 5;

            // Normalize on c
            for i in 0..cols {
                module_ref.vec_znx_rotate_backend(
                    p,
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                    i,
                    &vec_znx_backend_ref::<BR>(&a_ref),
                    i,
                );
                module_test.vec_znx_rotate_backend(
                    p,
                    &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                    i,
                    &vec_znx_backend_ref::<BT>(&a_test),
                    i,
                );
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(
                download_vec_znx::<BR>(&res_ref_backend),
                download_vec_znx::<BT>(&res_test_backend)
            );
        }
    }
}

pub fn test_vec_znx_rotate_assign<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxRotateAssignBackend<BR> + VecZnxRotateAssignTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxRotateAssignBackend<BT> + VecZnxRotateAssignTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_rotate_assign_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_rotate_assign_tmp_bytes());

    for size in [1, 2, 3, 4] {
        let mut res_ref: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, size);
        let mut res_test: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, size);

        // Fill a with random i64
        res_ref.fill_uniform(base2k, &mut source);
        res_test.raw_mut().copy_from_slice(res_ref.raw());
        let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
        let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

        let p: i64 = -5;

        // Normalize on c
        for i in 0..cols {
            module_ref.vec_znx_rotate_assign_backend(
                p,
                &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                i,
                &mut scratch_ref.arena(),
            );
            module_test.vec_znx_rotate_assign_backend(
                p,
                &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                i,
                &mut scratch_test.arena(),
            );
        }

        assert_eq!(
            download_vec_znx::<BR>(&res_ref_backend),
            download_vec_znx::<BT>(&res_test_backend)
        );

        let p: i64 = 5;

        // Normalize on c
        for i in 0..cols {
            module_ref.vec_znx_rotate_assign_backend(
                p,
                &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                i,
                &mut scratch_ref.arena(),
            );
            module_test.vec_znx_rotate_assign_backend(
                p,
                &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                i,
                &mut scratch_test.arena(),
            );
        }

        assert_eq!(
            download_vec_znx::<BR>(&res_ref_backend),
            download_vec_znx::<BT>(&res_test_backend)
        );
    }
}

pub fn test_vec_znx_fill_uniform<B: crate::test_suite::TestBackend<OwnedBuf = Vec<u8>>>(_params: &TestParams, module: &Module<B>)
where
    Module<B>: VecZnxFillUniformSourceBackend<B>,
{
    let n: usize = module.n();
    let base2k: usize = 17;
    let size: usize = 5;
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;
    let zero: Vec<i64> = vec![0; n];
    let one_12_sqrt: f64 = 0.28867513459481287;
    (0..cols).for_each(|col_i| {
        let host_init: VecZnx<Vec<u8>> = module.vec_znx_alloc(cols, size);
        let mut a = upload_vec_znx::<B>(&host_init);
        module.vec_znx_fill_uniform_source_backend(base2k, &mut vec_znx_backend_mut::<B>(&mut a), col_i, &mut source);
        let a = download_vec_znx::<B>(&a);
        (0..cols).for_each(|col_j| {
            if col_j != col_i {
                (0..size).for_each(|limb_i| {
                    assert_eq!(a.at(col_j, limb_i), zero);
                })
            } else {
                let std: f64 = a.stats(base2k, col_i).std();
                assert!((std - one_12_sqrt).abs() < 0.01, "std={std} ~!= {one_12_sqrt}",);
            }
        })
    });
}

pub fn test_vec_znx_seed_sampling_matches_source_wrappers<B: crate::test_suite::TestBackend<OwnedBuf = Vec<u8>>>(
    _params: &TestParams,
    module: &Module<B>,
) where
    Module<B>: VecZnxFillUniformSourceBackend<B>
        + VecZnxFillUniformBackend<B>
        + VecZnxFillNormalSourceBackend<B>
        + VecZnxFillNormalBackend<B>,
{
    let _n: usize = module.n();
    let base2k: usize = 17;
    let size: usize = 5;
    let cols: usize = 2;
    let col_i: usize = 1;

    let mut seed_source = Source::new([0u8; 32]);
    let seed = seed_source.new_seed();
    let mut wrapper_source = Source::new([0u8; 32]);

    let host_init = module.vec_znx_alloc(cols, size);
    let mut wrapper_uniform = upload_vec_znx::<B>(&host_init);
    let mut backend_uniform = upload_vec_znx::<B>(&host_init);
    module.vec_znx_fill_uniform_source_backend(
        base2k,
        &mut vec_znx_backend_mut::<B>(&mut wrapper_uniform),
        col_i,
        &mut wrapper_source,
    );
    module.vec_znx_fill_uniform_backend(base2k, &mut vec_znx_backend_mut::<B>(&mut backend_uniform), col_i, seed);
    assert_eq!(
        download_vec_znx::<B>(&wrapper_uniform),
        download_vec_znx::<B>(&backend_uniform)
    );

    let noise_infos = NoiseInfos::new(2 * 17, 3.2, 6.0 * 3.2).unwrap();
    let mut seed_source = Source::new([1u8; 32]);
    let seed = seed_source.new_seed();
    let mut wrapper_source = Source::new([1u8; 32]);

    let host_init = module.vec_znx_alloc(cols, size);
    let mut wrapper_normal = upload_vec_znx::<B>(&host_init);
    let mut backend_normal = upload_vec_znx::<B>(&host_init);
    module.vec_znx_fill_normal_source_backend(
        base2k,
        &mut vec_znx_backend_mut::<B>(&mut wrapper_normal),
        col_i,
        noise_infos,
        &mut wrapper_source,
    );
    module.vec_znx_fill_normal_backend(
        base2k,
        &mut vec_znx_backend_mut::<B>(&mut backend_normal),
        col_i,
        noise_infos,
        seed,
    );
    assert_eq!(download_vec_znx::<B>(&wrapper_normal), download_vec_znx::<B>(&backend_normal));
}

pub fn test_scalar_znx_secret_seed_sampling_matches_source_wrappers<B: crate::test_suite::TestBackend<OwnedBuf = Vec<u8>>>(
    _params: &TestParams,
    module: &Module<B>,
) where
    Module<B>: ScalarZnxFillTernaryHwSourceBackend<B>
        + ScalarZnxFillTernaryHwBackend<B>
        + ScalarZnxFillTernaryProbSourceBackend<B>
        + ScalarZnxFillTernaryProbBackend<B>
        + ScalarZnxFillBinaryHwSourceBackend<B>
        + ScalarZnxFillBinaryHwBackend<B>
        + ScalarZnxFillBinaryProbSourceBackend<B>
        + ScalarZnxFillBinaryProbBackend<B>
        + ScalarZnxFillBinaryBlockSourceBackend<B>
        + ScalarZnxFillBinaryBlockBackend<B>,
{
    let n: usize = module.n();
    let cols: usize = 2;
    let col_i: usize = 1;

    fn check<B, Fw, Fb>(
        module: &Module<B>,
        seed_bytes: [u8; 32],
        _n: usize,
        cols: usize,
        mut fill_wrapper: Fw,
        mut fill_backend: Fb,
    ) where
        B: crate::test_suite::TestBackend<OwnedBuf = Vec<u8>>,
        Fw: FnMut(&mut ScalarZnx<B::OwnedBuf>, &mut Source),
        Fb: FnMut(&mut ScalarZnx<B::OwnedBuf>, [u8; 32]),
    {
        let mut seed_source = Source::new(seed_bytes);
        let seed = seed_source.new_seed();
        let mut wrapper_source = Source::new(seed_bytes);

        let host_init = module.scalar_znx_alloc(cols);
        let mut wrapper = upload_scalar_znx::<B>(&host_init);
        let mut backend = upload_scalar_znx::<B>(&host_init);
        fill_wrapper(&mut wrapper, &mut wrapper_source);
        fill_backend(&mut backend, seed);
        assert_eq!(download_scalar_znx::<B>(&wrapper), download_scalar_znx::<B>(&backend));
    }

    check::<B, _, _>(
        module,
        [2u8; 32],
        n,
        cols,
        move |res, source| {
            module.scalar_znx_fill_ternary_hw_source_backend(
                &mut <ScalarZnx<B::OwnedBuf> as ScalarZnxToBackendMut<B>>::to_backend_mut(res),
                col_i,
                n / 8,
                source,
            )
        },
        move |res, seed| {
            module.scalar_znx_fill_ternary_hw_backend(
                &mut <ScalarZnx<B::OwnedBuf> as ScalarZnxToBackendMut<B>>::to_backend_mut(res),
                col_i,
                n / 8,
                seed,
            )
        },
    );
    check::<B, _, _>(
        module,
        [3u8; 32],
        n,
        cols,
        move |res, source| {
            module.scalar_znx_fill_ternary_prob_source_backend(
                &mut <ScalarZnx<B::OwnedBuf> as ScalarZnxToBackendMut<B>>::to_backend_mut(res),
                col_i,
                0.5,
                source,
            )
        },
        move |res, seed| {
            module.scalar_znx_fill_ternary_prob_backend(
                &mut <ScalarZnx<B::OwnedBuf> as ScalarZnxToBackendMut<B>>::to_backend_mut(res),
                col_i,
                0.5,
                seed,
            )
        },
    );
    check::<B, _, _>(
        module,
        [4u8; 32],
        n,
        cols,
        move |res, source| {
            module.scalar_znx_fill_binary_hw_source_backend(
                &mut <ScalarZnx<B::OwnedBuf> as ScalarZnxToBackendMut<B>>::to_backend_mut(res),
                col_i,
                n / 8,
                source,
            )
        },
        move |res, seed| {
            module.scalar_znx_fill_binary_hw_backend(
                &mut <ScalarZnx<B::OwnedBuf> as ScalarZnxToBackendMut<B>>::to_backend_mut(res),
                col_i,
                n / 8,
                seed,
            )
        },
    );
    check::<B, _, _>(
        module,
        [5u8; 32],
        n,
        cols,
        move |res, source| {
            module.scalar_znx_fill_binary_prob_source_backend(
                &mut <ScalarZnx<B::OwnedBuf> as ScalarZnxToBackendMut<B>>::to_backend_mut(res),
                col_i,
                0.5,
                source,
            )
        },
        move |res, seed| {
            module.scalar_znx_fill_binary_prob_backend(
                &mut <ScalarZnx<B::OwnedBuf> as ScalarZnxToBackendMut<B>>::to_backend_mut(res),
                col_i,
                0.5,
                seed,
            )
        },
    );
    check::<B, _, _>(
        module,
        [6u8; 32],
        n,
        cols,
        move |res, source| {
            module.scalar_znx_fill_binary_block_source_backend(
                &mut <ScalarZnx<B::OwnedBuf> as ScalarZnxToBackendMut<B>>::to_backend_mut(res),
                col_i,
                8,
                source,
            )
        },
        move |res, seed| {
            module.scalar_znx_fill_binary_block_backend(
                &mut <ScalarZnx<B::OwnedBuf> as ScalarZnxToBackendMut<B>>::to_backend_mut(res),
                col_i,
                8,
                seed,
            )
        },
    );
}

pub fn test_vec_znx_fill_normal<B: crate::test_suite::TestBackend<OwnedBuf = Vec<u8>>>(_params: &TestParams, module: &Module<B>)
where
    Module<B>: VecZnxFillNormalSourceBackend<B>,
{
    let n: usize = module.n();
    let base2k: usize = 17;
    let size: usize = 5;
    let noise_infos = NoiseInfos::new(2 * 17, 3.2, 6.0 * 3.2).unwrap();
    let mut source_xe: Source = Source::new([0u8; 32]);
    let cols: usize = 2;
    let zero: Vec<i64> = vec![0; n];
    let k_f64: f64 = (1u64 << noise_infos.k as u64) as f64;
    (0..cols).for_each(|col_i| {
        let host_init: VecZnx<Vec<u8>> = module.vec_znx_alloc(cols, size);
        let mut a = upload_vec_znx::<B>(&host_init);
        module.vec_znx_fill_normal_source_backend(
            base2k,
            &mut vec_znx_backend_mut::<B>(&mut a),
            col_i,
            noise_infos,
            &mut source_xe,
        );
        let a = download_vec_znx::<B>(&a);
        (0..cols).for_each(|col_j| {
            if col_j != col_i {
                (0..size).for_each(|limb_i| {
                    assert_eq!(a.at(col_j, limb_i), zero);
                })
            } else {
                let std: f64 = a.stats(base2k, col_i).std() * k_f64;
                assert!((std - noise_infos.sigma).abs() < 0.1, "std={std} ~!= {}", noise_infos.sigma);
            }
        })
    });
}

pub fn test_vec_znx_add_normal<B: crate::test_suite::TestBackend<OwnedBuf = Vec<u8>>>(_params: &TestParams, module: &Module<B>)
where
    Module<B>: VecZnxFillNormalSourceBackend<B> + VecZnxAddNormalSourceBackend<B>,
{
    let n: usize = module.n();
    let base2k: usize = 17;
    let size: usize = 5;
    let noise_infos = NoiseInfos::new(2 * 17, 3.2, 6.0 * 3.2).unwrap();
    let mut source_xe: Source = Source::new([0u8; 32]);
    let cols: usize = 2;
    let zero: Vec<i64> = vec![0; n];
    let k_f64: f64 = (1u64 << noise_infos.k as u64) as f64;
    let sqrt2: f64 = SQRT_2;
    (0..cols).for_each(|col_i| {
        let host_init: VecZnx<Vec<u8>> = module.vec_znx_alloc(cols, size);
        let mut a = upload_vec_znx::<B>(&host_init);
        module.vec_znx_fill_normal_source_backend(
            base2k,
            &mut vec_znx_backend_mut::<B>(&mut a),
            col_i,
            noise_infos,
            &mut source_xe,
        );
        module.vec_znx_add_normal_source_backend(
            base2k,
            &mut vec_znx_backend_mut::<B>(&mut a),
            col_i,
            noise_infos,
            &mut source_xe,
        );
        let a = download_vec_znx::<B>(&a);
        (0..cols).for_each(|col_j| {
            if col_j != col_i {
                (0..size).for_each(|limb_i| {
                    assert_eq!(a.at(col_j, limb_i), zero);
                })
            } else {
                let std: f64 = a.stats(base2k, col_i).std() * k_f64;
                assert!(
                    (std - noise_infos.sigma * sqrt2).abs() < 0.1,
                    "std={std} ~!= {}",
                    noise_infos.sigma * sqrt2
                );
            }
        })
    });
}

pub fn test_vec_znx_lsh<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxLshBackend<BR> + VecZnxLshTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxLshBackend<BT> + VecZnxLshTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_lsh_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_lsh_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            for k in 0..res_size * base2k {
                let mut res_ref: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
                let mut res_test: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);

                // Set d to garbage
                res_ref.fill_uniform(base2k, &mut source);
                res_test.fill_uniform(base2k, &mut source);
                let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
                let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_lsh_backend(
                        base2k,
                        k,
                        &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                        i,
                        &vec_znx_backend_ref::<BR>(&a_ref),
                        i,
                        &mut scratch_ref.arena(),
                    );
                    module_test.vec_znx_lsh_backend(
                        base2k,
                        k,
                        &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                        i,
                        &vec_znx_backend_ref::<BT>(&a_test),
                        i,
                        &mut scratch_test.arena(),
                    );
                }

                assert_eq!(a.digest_u64(), a_digest);
                assert_eq!(
                    download_vec_znx::<BR>(&res_ref_backend),
                    download_vec_znx::<BT>(&res_test_backend)
                );
            }
        }
    }
}

pub fn test_vec_znx_lsh_assign<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxLshAssignBackend<BR> + VecZnxLshTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxLshAssignBackend<BT> + VecZnxLshTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_lsh_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_lsh_tmp_bytes());

    for res_size in [1, 2, 3, 4] {
        for k in 0..base2k * res_size {
            let mut res_ref: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
            let mut res_test: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);

            res_ref.fill_uniform(base2k, &mut source);
            res_test.raw_mut().copy_from_slice(res_ref.raw());
            let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
            let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

            for i in 0..cols {
                module_ref.vec_znx_lsh_assign_backend(
                    base2k,
                    k,
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                    i,
                    &mut scratch_ref.arena(),
                );
                module_test.vec_znx_lsh_assign_backend(
                    base2k,
                    k,
                    &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                    i,
                    &mut scratch_test.arena(),
                );
            }

            assert_eq!(
                download_vec_znx::<BR>(&res_ref_backend),
                download_vec_znx::<BT>(&res_test_backend)
            );
        }
    }
}

pub fn test_vec_znx_rsh<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxRshBackend<BR> + VecZnxRshTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxRshBackend<BT> + VecZnxRshTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_rsh_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_rsh_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            for k in 0..res_size * base2k {
                let mut res_ref: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
                let mut res_test: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);

                // Set d to garbage
                res_ref.fill_uniform(base2k, &mut source);
                res_test.fill_uniform(base2k, &mut source);
                let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
                let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_rsh_backend(
                        base2k,
                        k,
                        &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                        i,
                        &vec_znx_backend_ref::<BR>(&a_ref),
                        i,
                        &mut scratch_ref.arena(),
                    );
                    module_test.vec_znx_rsh_backend(
                        base2k,
                        k,
                        &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                        i,
                        &vec_znx_backend_ref::<BT>(&a_test),
                        i,
                        &mut scratch_test.arena(),
                    );
                }

                assert_eq!(a.digest_u64(), a_digest);
                assert_eq!(
                    download_vec_znx::<BR>(&res_ref_backend),
                    download_vec_znx::<BT>(&res_test_backend)
                );
            }
        }
    }
}

pub fn test_vec_znx_rsh_assign<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxRshAssignBackend<BR> + VecZnxRshTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxRshAssignBackend<BT> + VecZnxRshTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_rsh_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_rsh_tmp_bytes());

    for res_size in [1, 2, 3, 4] {
        for k in 0..base2k * res_size {
            let mut res_ref: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
            let mut res_test: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);

            res_ref.fill_uniform(base2k, &mut source);
            res_test.raw_mut().copy_from_slice(res_ref.raw());
            let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
            let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

            for i in 0..cols {
                module_ref.vec_znx_rsh_assign_backend(
                    base2k,
                    k,
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                    i,
                    &mut scratch_ref.arena(),
                );
                module_test.vec_znx_rsh_assign_backend(
                    base2k,
                    k,
                    &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                    i,
                    &mut scratch_test.arena(),
                );
            }

            assert_eq!(
                download_vec_znx::<BR>(&res_ref_backend),
                download_vec_znx::<BT>(&res_test_backend)
            );
        }
    }
}

pub fn test_vec_znx_split_ring<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxSplitRingBackend<BR> + ModuleNew<BR> + VecZnxSplitRingTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxSplitRingBackend<BT> + ModuleNew<BT> + VecZnxSplitRingTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;
    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_split_ring_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_split_ring_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let mut res_ref: [VecZnx<Vec<u8>>; 2] =
                [VecZnx::alloc(n >> 1, cols, res_size), VecZnx::alloc(n >> 1, cols, res_size)];

            let mut res_test: [VecZnx<Vec<u8>>; 2] =
                [VecZnx::alloc(n >> 1, cols, res_size), VecZnx::alloc(n >> 1, cols, res_size)];

            res_ref.iter_mut().for_each(|ri| {
                ri.fill_uniform(base2k, &mut source);
            });

            res_test.iter_mut().for_each(|ri| {
                ri.fill_uniform(base2k, &mut source);
            });
            let mut res_ref_backend = res_ref.each_ref().map(upload_vec_znx::<BR>);
            let mut res_test_backend = res_test.each_ref().map(upload_vec_znx::<BT>);

            for i in 0..cols {
                let mut res_ref_backend_refs: Vec<_> = res_ref_backend.iter_mut().map(vec_znx_backend_mut::<BR>).collect();
                module_ref.vec_znx_split_ring_backend(
                    &mut res_ref_backend_refs,
                    i,
                    &vec_znx_backend_ref::<BR>(&a_ref),
                    i,
                    &mut scratch_ref.arena(),
                );
                let mut res_test_backend_refs: Vec<_> = res_test_backend.iter_mut().map(vec_znx_backend_mut::<BT>).collect();
                module_test.vec_znx_split_ring_backend(
                    &mut res_test_backend_refs,
                    i,
                    &vec_znx_backend_ref::<BT>(&a_test),
                    i,
                    &mut scratch_test.arena(),
                );
            }

            assert_eq!(a.digest_u64(), a_digest);

            for (a, b) in res_ref_backend.iter().zip(res_test_backend.iter()) {
                assert_eq!(download_vec_znx::<BR>(a), download_vec_znx::<BT>(b));
            }
        }
    }
}

pub fn test_vec_znx_sub_scalar<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxSubScalarBackend<BR>,
    Module<BT>: VecZnxSubScalarBackend<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut a: ScalarZnx<Vec<u8>> = module_host.scalar_znx_alloc(cols);
    a.fill_uniform(base2k, &mut source);
    let a_digest: u64 = a.digest_u64();
    let a_ref = upload_scalar_znx::<BR>(&a);
    let a_test = upload_scalar_znx::<BT>(&a);

    for b_size in [1, 2, 3, 4] {
        let mut b: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, b_size);
        b.fill_uniform(base2k, &mut source);
        let b_digest: u64 = b.digest_u64();
        let b_ref = upload_vec_znx::<BR>(&b);
        let b_test = upload_vec_znx::<BT>(&b);

        for res_size in [1, 2, 3, 4] {
            let mut res_0: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
            let mut res_1: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);

            // Set d to garbage
            res_0.fill_uniform(base2k, &mut source);
            res_1.fill_uniform(base2k, &mut source);
            let mut res_0_backend = upload_vec_znx::<BR>(&res_0);
            let mut res_1_backend = upload_vec_znx::<BT>(&res_1);

            // Reference
            for i in 0..cols {
                module_ref.vec_znx_sub_scalar_backend(
                    &mut vec_znx_backend_mut::<BR>(&mut res_0_backend),
                    i,
                    &scalar_znx_backend_ref::<BR>(&a_ref),
                    i,
                    &vec_znx_backend_ref::<BR>(&b_ref),
                    i,
                    (res_size.min(b_size)) - 1,
                );
                module_test.vec_znx_sub_scalar_backend(
                    &mut vec_znx_backend_mut::<BT>(&mut res_1_backend),
                    i,
                    &scalar_znx_backend_ref::<BT>(&a_test),
                    i,
                    &vec_znx_backend_ref::<BT>(&b_test),
                    i,
                    (res_size.min(b_size)) - 1,
                );
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(b.digest_u64(), b_digest);
            assert_eq!(download_vec_znx::<BR>(&res_0_backend), download_vec_znx::<BT>(&res_1_backend));
        }
    }
}

pub fn test_vec_znx_sub_scalar_assign<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxSubScalarAssignBackend<BR>,
    Module<BT>: VecZnxSubScalarAssignBackend<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut a: ScalarZnx<Vec<u8>> = module_host.scalar_znx_alloc(cols);
    a.fill_uniform(base2k, &mut source);
    let a_digest: u64 = a.digest_u64();
    let a_ref = upload_scalar_znx::<BR>(&a);
    let a_test = upload_scalar_znx::<BT>(&a);

    for res_size in [1, 2, 3, 4] {
        let mut res_0: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
        let mut res_1: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);

        res_0.fill_uniform(base2k, &mut source);
        res_1.raw_mut().copy_from_slice(res_0.raw());
        let mut res_0_backend = upload_vec_znx::<BR>(&res_0);
        let mut res_1_backend = upload_vec_znx::<BT>(&res_1);

        for i in 0..cols {
            module_ref.vec_znx_sub_scalar_assign_backend(
                &mut vec_znx_backend_mut::<BR>(&mut res_0_backend),
                i,
                res_size - 1,
                &scalar_znx_backend_ref::<BR>(&a_ref),
                i,
            );
            module_test.vec_znx_sub_scalar_assign_backend(
                &mut vec_znx_backend_mut::<BT>(&mut res_1_backend),
                i,
                res_size - 1,
                &scalar_znx_backend_ref::<BT>(&a_test),
                i,
            );
        }

        assert_eq!(a.digest_u64(), a_digest);
        assert_eq!(download_vec_znx::<BR>(&res_0_backend), download_vec_znx::<BT>(&res_1_backend));
    }
}

pub fn test_vec_znx_extract_coeff_backend<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxExtractCoeffBackend<BR>,
    Module<BT>: VecZnxExtractCoeffBackend<BT>,
{
    let base2k = params.base2k;
    let n = module_ref.n();
    assert_eq!(n, module_test.n());

    let cols: usize = 2;
    let mut source = Source::new([23u8; 32]);
    let coeffs = [0usize, 1usize.min(n - 1), (n / 2).min(n - 1), n - 1];

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let expected =
                VecZnx::from_data_with_max_size(vec![0u8; VecZnx::<Vec<u8>>::bytes_of(1, cols, 4)], 1, cols, res_size, 4);
            let actual =
                VecZnx::from_data_with_max_size(vec![0u8; VecZnx::<Vec<u8>>::bytes_of(1, cols, 4)], 1, cols, res_size, 4);
            let mut expected_backend = upload_vec_znx::<BR>(&expected);
            let mut actual_backend = upload_vec_znx::<BT>(&actual);

            for col_i in 0..cols {
                let coeff = coeffs[(col_i + a_size + res_size) % coeffs.len()];
                module_ref.vec_znx_extract_coeff_backend(
                    &mut vec_znx_backend_mut::<BR>(&mut expected_backend),
                    col_i,
                    &vec_znx_backend_ref::<BR>(&a_ref),
                    col_i,
                    coeff,
                );
                module_test.vec_znx_extract_coeff_backend(
                    &mut vec_znx_backend_mut::<BT>(&mut actual_backend),
                    col_i,
                    &vec_znx_backend_ref::<BT>(&a_test),
                    col_i,
                    coeff,
                );
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(
                download_vec_znx::<BR>(&expected_backend),
                download_vec_znx::<BT>(&actual_backend)
            );
        }
    }
}

pub fn test_vec_znx_sub<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxSubBackend<BR>,
    Module<BT>: VecZnxSubBackend<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);

        for b_size in [1, 2, 3, 4] {
            let mut b: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, b_size);
            b.fill_uniform(base2k, &mut source);
            let b_digest: u64 = b.digest_u64();
            let b_ref = upload_vec_znx::<BR>(&b);
            let b_test = upload_vec_znx::<BT>(&b);

            for res_size in [1, 2, 3, 4] {
                let mut res_ref: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
                let mut res_test: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);

                // Set d to garbage
                res_ref.fill_uniform(base2k, &mut source);
                res_test.fill_uniform(base2k, &mut source);
                let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
                let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

                // Reference
                for i in 0..cols {
                    module_test.vec_znx_sub_backend(
                        &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                        i,
                        &vec_znx_backend_ref::<BT>(&a_test),
                        i,
                        &vec_znx_backend_ref::<BT>(&b_test),
                        i,
                    );
                    module_ref.vec_znx_sub_backend(
                        &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                        i,
                        &vec_znx_backend_ref::<BR>(&a_ref),
                        i,
                        &vec_znx_backend_ref::<BR>(&b_ref),
                        i,
                    );
                }

                assert_eq!(a.digest_u64(), a_digest);
                assert_eq!(b.digest_u64(), b_digest);

                assert_eq!(
                    download_vec_znx::<BR>(&res_ref_backend),
                    download_vec_znx::<BT>(&res_test_backend)
                );
            }
        }
    }
}

pub fn test_vec_znx_sub_assign<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxSubAssignBackend<BR>,
    Module<BT>: VecZnxSubAssignBackend<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let mut res_ref: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
            let mut res_test: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);

            res_ref.fill_uniform(base2k, &mut source);
            res_test.raw_mut().copy_from_slice(res_ref.raw());
            let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
            let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

            for i in 0..cols {
                module_test.vec_znx_sub_assign_backend(
                    &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                    i,
                    &vec_znx_backend_ref::<BT>(&a_test),
                    i,
                );
                module_ref.vec_znx_sub_assign_backend(
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                    i,
                    &vec_znx_backend_ref::<BR>(&a_ref),
                    i,
                );
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(
                download_vec_znx::<BR>(&res_ref_backend),
                download_vec_znx::<BT>(&res_test_backend)
            );
        }
    }
}

pub fn test_vec_znx_sub_negate_assign<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxSubNegateAssignBackend<BR>,
    Module<BT>: VecZnxSubNegateAssignBackend<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let mut res_ref: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
            let mut res_test: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);

            res_ref.fill_uniform(base2k, &mut source);
            res_test.raw_mut().copy_from_slice(res_ref.raw());
            let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
            let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

            for i in 0..cols {
                module_test.vec_znx_sub_negate_assign_backend(
                    &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                    i,
                    &vec_znx_backend_ref::<BT>(&a_test),
                    i,
                );
                module_ref.vec_znx_sub_negate_assign_backend(
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                    i,
                    &vec_znx_backend_ref::<BR>(&a_ref),
                    i,
                );
            }

            assert_eq!(a.digest_u64(), a_digest);
            assert_eq!(
                download_vec_znx::<BR>(&res_ref_backend),
                download_vec_znx::<BT>(&res_test_backend)
            );
        }
    }
}

pub fn test_vec_znx_switch_ring<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxSwitchRingBackend<BR> + ModuleNew<BR>,
    Module<BT>: VecZnxSwitchRingBackend<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);

        // Fill a with random i64
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            {
                let mut res_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n << 1, cols, res_size);
                let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n << 1, cols, res_size);

                res_ref.fill_uniform(base2k, &mut source);
                res_test.fill_uniform(base2k, &mut source);
                let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
                let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

                // Normalize on c
                for i in 0..cols {
                    module_ref.vec_znx_switch_ring_backend(
                        &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                        i,
                        &vec_znx_backend_ref::<BR>(&a_ref),
                        i,
                    );
                    module_test.vec_znx_switch_ring_backend(
                        &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                        i,
                        &vec_znx_backend_ref::<BT>(&a_test),
                        i,
                    );
                }

                assert_eq!(a.digest_u64(), a_digest);
                assert_eq!(
                    download_vec_znx::<BR>(&res_ref_backend),
                    download_vec_znx::<BT>(&res_test_backend)
                );
            }

            {
                let mut res_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n >> 1, cols, res_size);
                let mut res_test: VecZnx<Vec<u8>> = VecZnx::alloc(n >> 1, cols, res_size);

                res_ref.fill_uniform(base2k, &mut source);
                res_test.fill_uniform(base2k, &mut source);
                let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
                let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

                // Normalize on c
                for i in 0..cols {
                    module_ref.vec_znx_switch_ring_backend(
                        &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                        i,
                        &vec_znx_backend_ref::<BR>(&a_ref),
                        i,
                    );
                    module_test.vec_znx_switch_ring_backend(
                        &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                        i,
                        &vec_znx_backend_ref::<BT>(&a_test),
                        i,
                    );
                }

                assert_eq!(a.digest_u64(), a_digest);
                assert_eq!(
                    download_vec_znx::<BR>(&res_ref_backend),
                    download_vec_znx::<BT>(&res_test_backend)
                );
            }
        }
    }
}

pub fn test_vec_znx_switch_ring_backend_matches_wrapper<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    _module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: ModuleNew<BR>,
    Module<BT>: VecZnxSwitchRingBackend<BT>,
{
    let base2k = params.base2k;
    let n: usize = module_test.n();
    let cols: usize = 2;
    let a_col: usize = 0;
    let res_col: usize = 1;
    let mut source: Source = Source::new([4u8; 32]);

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_backend = upload_vec_znx::<BT>(&a);

        for res_n in [n >> 1, n << 1] {
            for res_size in [1, 2, 3, 4] {
                let mut wrapper: VecZnx<Vec<u8>> = VecZnx::alloc(res_n, cols, res_size);
                let mut backend: VecZnx<Vec<u8>> = VecZnx::alloc(res_n, cols, res_size);
                wrapper.fill_uniform(base2k, &mut source);
                backend.data.copy_from_slice(&wrapper.data);
                let mut wrapper_backend = upload_vec_znx::<BT>(&wrapper);
                let mut backend_backend = upload_vec_znx::<BT>(&backend);

                module_test.vec_znx_switch_ring_backend(
                    &mut vec_znx_backend_mut::<BT>(&mut wrapper_backend),
                    res_col,
                    &vec_znx_backend_ref::<BT>(&a_backend),
                    a_col,
                );
                module_test.vec_znx_switch_ring_backend(
                    &mut vec_znx_backend_mut::<BT>(&mut backend_backend),
                    res_col,
                    &vec_znx_backend_ref::<BT>(&a_backend),
                    a_col,
                );

                assert_eq!(
                    download_vec_znx::<BT>(&wrapper_backend),
                    download_vec_znx::<BT>(&backend_backend)
                );
            }
        }
    }
}
