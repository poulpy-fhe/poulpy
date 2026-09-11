use std::f64::consts::SQRT_2;

use super::{
    TestParams, download_scalar_znx, download_vec_znx, scalar_znx_backend_mut, scalar_znx_backend_ref, upload_scalar_znx,
    upload_vec_znx, vec_znx_backend_mut, vec_znx_backend_ref,
};

use crate::{
    api::{
        ModuleNew, ScalarZnxAutomorphism, ScratchOwnedAlloc, VecZnxAdd, VecZnxAddAssign, VecZnxAddNormalSource,
        VecZnxAddScalarAssign, VecZnxAutomorphism, VecZnxAutomorphismAssign, VecZnxAutomorphismAssignTmpBytes, VecZnxCopy,
        VecZnxFillUniformSource, VecZnxLsh, VecZnxLshAssign, VecZnxLshTmpBytes, VecZnxMulXpMinusOne, VecZnxMulXpMinusOneAssign,
        VecZnxMulXpMinusOneAssignTmpBytes, VecZnxNegate, VecZnxNegateAssign, VecZnxNormalize, VecZnxNormalizeAssign,
        VecZnxNormalizeTmpBytes, VecZnxRotate, VecZnxRotateAssign, VecZnxRotateAssignTmpBytes, VecZnxRsh, VecZnxRshAssign,
        VecZnxRshTmpBytes, VecZnxSub, VecZnxSubAssign, VecZnxSubNegateAssign, VecZnxSwitchRing, VecZnxZero,
    },
    layouts::{
        DigestU64, FillUniform, HostBytesBackend, HostDataRef, Module, NoiseInfos, ScratchOwned, VecZnx, ZnxView, ZnxViewMut,
    },
    source::Source,
};

fn assert_canonical(a: &VecZnx<impl HostDataRef, i64>, base2k: usize, k: usize) {
    let active_size = k.div_ceil(base2k);
    let half = 1i64 << (base2k - 1);
    for col in 0..a.cols() {
        for limb in 0..active_size.min(a.size()) {
            assert!(
                a.at(col, limb).iter().all(|&digit| (-half..half).contains(&digit)),
                "col {col} limb {limb}: digit outside [-2^{}, 2^{})",
                base2k - 1,
                base2k - 1
            );
        }
        for limb in active_size..a.size() {
            assert!(a.at(col, limb).iter().all(|&digit| digit == 0));
        }
        let padding = (base2k - k % base2k) % base2k;
        if active_size != 0 && padding != 0 {
            let mask = (1u64 << padding) - 1;
            assert!(a.at(col, active_size - 1).iter().all(|&digit| digit as u64 & mask == 0));
        }
    }
}

pub(super) fn cross_normalization_cases(a_size: usize, res_size: usize) -> Vec<(usize, usize, usize, i64)> {
    let mut cases = Vec::new();
    if matches!(a_size, 1 | 4) && matches!(res_size, 1 | 4) {
        for (a_base, res_base) in [(1, 2), (2, 1), (17, 50), (50, 17), (50, 51), (51, 50), (51, 62), (62, 51)] {
            for k in [
                res_size * res_base - res_base / 2 - 1,
                res_size.saturating_sub(1).max(1) * res_base - 1,
            ] {
                for offset in [-(a_base as i64) - 1, 0, a_base as i64 + 1] {
                    cases.push((a_base, res_base, k, offset));
                }
            }
        }
    }
    cases
}

pub fn test_vec_znx_zero_matches_wrapper<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    _module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BT>: VecZnxZero<BT>,
{
    let base2k = params.base2k;
    let _n: usize = module_test.n();
    let cols: usize = 2;
    let mut source: Source = Source::new([5u8; 32]);

    for size in [1, 2, 3, 4] {
        for col_i in 0..cols {
            let mut expected = module_host.vec_znx_alloc(cols, size);
            expected.fill_uniform(base2k, &mut source);
            let mut backend = upload_vec_znx::<BT>(&expected);

            for limb in 0..size {
                expected.at_mut(col_i, limb).fill(0);
            }
            module_test.vec_znx_zero(&mut vec_znx_backend_mut::<BT>(&mut backend), col_i);

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
        let mut a = module.vec_znx_alloc(2, size);
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
            assert_eq!(have, want, "{:?} != {:?}", have, want);
        })
    }
}

pub fn test_vec_znx_add_scalar_assign<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxAddScalarAssign<BR>,
    Module<BT>: VecZnxAddScalarAssign<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut b = module_host.scalar_znx_alloc(cols);
    b.fill_uniform(base2k, &mut source);
    let b_digest: u64 = b.digest_u64();
    let b_ref = upload_scalar_znx::<BR>(&b);
    let b_test = upload_scalar_znx::<BT>(&b);

    for res_size in [1usize, 2, 3, 4] {
        let mut rest_ref = module_host.vec_znx_alloc(cols, res_size);
        let mut res_test = module_host.vec_znx_alloc(cols, res_size);

        rest_ref.fill_uniform(base2k, &mut source);
        res_test.raw_mut().copy_from_slice(rest_ref.raw());
        let mut rest_ref_backend = upload_vec_znx::<BR>(&rest_ref);
        let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

        for i in 0..cols {
            module_ref.vec_znx_add_scalar_assign(
                &mut vec_znx_backend_mut::<BR>(&mut rest_ref_backend),
                i,
                res_size - 1,
                &scalar_znx_backend_ref::<BR>(&b_ref),
                i,
            );
            module_test.vec_znx_add_scalar_assign(
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

pub fn test_vec_znx_add_matches_reference<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxAdd<BR>,
    Module<BT>: VecZnxAdd<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let cols: usize = 2;
    let mut source: Source = Source::new([13u8; 32]);

    for a_size in [1, 2, 3, 4] {
        let mut a = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);

        for b_size in [1, 2, 3, 4] {
            let mut b = module_host.vec_znx_alloc(cols, b_size);
            b.fill_uniform(base2k, &mut source);
            let b_digest: u64 = b.digest_u64();
            let b_ref = upload_vec_znx::<BR>(&b);
            let b_test = upload_vec_znx::<BT>(&b);

            for res_size in [1, 2, 3, 4] {
                let mut wrapper = module_host.vec_znx_alloc(cols, res_size);
                let mut backend = module_host.vec_znx_alloc(cols, res_size);

                wrapper.fill_uniform(base2k, &mut source);
                backend.data_mut().copy_from_slice(wrapper.data());
                let mut wrapper_backend = upload_vec_znx::<BR>(&wrapper);
                let mut backend_owned = upload_vec_znx::<BT>(&backend);

                for col_i in 0..cols {
                    module_ref.vec_znx_add(
                        &mut vec_znx_backend_mut::<BR>(&mut wrapper_backend),
                        col_i,
                        &vec_znx_backend_ref::<BR>(&a_ref),
                        col_i,
                        &vec_znx_backend_ref::<BR>(&b_ref),
                        col_i,
                    );
                    module_test.vec_znx_add(
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
    Module<BR>: VecZnxAddAssign<BR>,
    Module<BT>: VecZnxAddAssign<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let mut res_ref = module_host.vec_znx_alloc(cols, res_size);
            let mut res_test = module_host.vec_znx_alloc(cols, res_size);

            res_ref.fill_uniform(base2k, &mut source);
            res_test.raw_mut().copy_from_slice(res_ref.raw());
            let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
            let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

            for i in 0..cols {
                module_ref.vec_znx_add_assign(
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                    i,
                    &vec_znx_backend_ref::<BR>(&a_ref),
                    i,
                );
                module_test.vec_znx_add_assign(
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

pub fn test_vec_znx_add_assign_matches_wrapper<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    _module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BT>: VecZnxAddAssign<BT>,
{
    let base2k = params.base2k;
    let _n: usize = module_test.n();
    let cols: usize = 2;
    let mut source: Source = Source::new([14u8; 32]);

    for a_size in [1, 2, 3, 4] {
        let mut a = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();
        let a_backend = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let mut wrapper = module_host.vec_znx_alloc(cols, res_size);
            let mut backend = module_host.vec_znx_alloc(cols, res_size);

            wrapper.fill_uniform(base2k, &mut source);
            backend.data_mut().copy_from_slice(wrapper.data());
            let mut wrapper_backend = upload_vec_znx::<BT>(&wrapper);
            let mut backend_backend = upload_vec_znx::<BT>(&backend);

            for col_i in 0..cols {
                module_test.vec_znx_add_assign(
                    &mut vec_znx_backend_mut::<BT>(&mut wrapper_backend),
                    col_i,
                    &vec_znx_backend_ref::<BT>(&a_backend),
                    col_i,
                );
                module_test.vec_znx_add_assign(
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
    Module<BR>: VecZnxAutomorphism<BR>,
    Module<BT>: VecZnxAutomorphism<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let res_ref = module_host.vec_znx_alloc(cols, res_size);
            let res_test = module_host.vec_znx_alloc(cols, res_size);
            let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
            let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

            let p: i64 = -5;

            // Normalize on c
            for i in 0..cols {
                module_ref.vec_znx_automorphism(
                    p,
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                    i,
                    &vec_znx_backend_ref::<BR>(&a_ref),
                    i,
                );
                module_test.vec_znx_automorphism(
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
                module_ref.vec_znx_automorphism(
                    p,
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                    i,
                    &vec_znx_backend_ref::<BR>(&a_ref),
                    i,
                );
                module_test.vec_znx_automorphism(
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
        let mut res_ref = module_host.vec_znx_alloc(cols, size);
        let mut res_test = module_host.vec_znx_alloc(cols, size);

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
    Module<BR>: VecZnxCopy<BR>,
    Module<BT>: VecZnxCopy<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let mut res_0 = module_host.vec_znx_alloc(cols, res_size);
            let mut res_1 = module_host.vec_znx_alloc(cols, res_size);

            // Set d to garbage
            res_0.fill_uniform(base2k, &mut source);
            res_1.fill_uniform(base2k, &mut source);
            let mut res_0_backend = upload_vec_znx::<BR>(&res_0);
            let mut res_1_backend = upload_vec_znx::<BT>(&res_1);

            // Reference
            for i in 0..cols {
                module_ref.vec_znx_copy(
                    &mut vec_znx_backend_mut::<BR>(&mut res_0_backend),
                    i,
                    &vec_znx_backend_ref::<BR>(&a_ref),
                    i,
                );
                module_test.vec_znx_copy(
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

pub fn test_vec_znx_copy_matches_wrapper<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    _module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BT>: VecZnxCopy<BT>,
{
    let base2k = params.base2k;
    let _n: usize = module_test.n();
    let cols: usize = 2;
    let a_col: usize = 0;
    let res_col: usize = 1;
    let mut source: Source = Source::new([3u8; 32]);

    for a_size in [1, 2, 3, 4] {
        let mut a = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_backend = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let mut wrapper = module_host.vec_znx_alloc(cols, res_size);
            let mut backend = module_host.vec_znx_alloc(cols, res_size);
            wrapper.fill_uniform(base2k, &mut source);
            backend.data_mut().copy_from_slice(wrapper.data());
            let mut wrapper_backend = upload_vec_znx::<BT>(&wrapper);
            let mut backend_backend = upload_vec_znx::<BT>(&backend);

            module_test.vec_znx_copy(
                &mut vec_znx_backend_mut::<BT>(&mut wrapper_backend),
                res_col,
                &vec_znx_backend_ref::<BT>(&a_backend),
                a_col,
            );
            module_test.vec_znx_copy(
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

pub fn test_scalar_znx_automorphism<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: ScalarZnxAutomorphism<BR>,
    Module<BT>: ScalarZnxAutomorphism<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut a = module_host.scalar_znx_alloc(cols);
    a.fill_uniform(base2k, &mut source);
    let a_digest: u64 = a.digest_u64();
    let a_ref = upload_scalar_znx::<BR>(&a);
    let a_test = upload_scalar_znx::<BT>(&a);

    let mut res_ref = module_host.scalar_znx_alloc(cols);
    let mut res_test = module_host.scalar_znx_alloc(cols);
    res_ref.fill_uniform(base2k, &mut source);
    res_test.fill_uniform(base2k, &mut source);
    let mut res_ref_backend = upload_scalar_znx::<BR>(&res_ref);
    let mut res_test_backend = upload_scalar_znx::<BT>(&res_test);

    for p in [-5, 5] {
        for i in 0..cols {
            module_ref.scalar_znx_automorphism(
                p,
                &mut scalar_znx_backend_mut::<BR>(&mut res_ref_backend),
                i,
                &scalar_znx_backend_ref::<BR>(&a_ref),
                i,
            );
            module_test.scalar_znx_automorphism(
                p,
                &mut scalar_znx_backend_mut::<BT>(&mut res_test_backend),
                i,
                &scalar_znx_backend_ref::<BT>(&a_test),
                i,
            );
        }

        assert_eq!(a.digest_u64(), a_digest);
        assert_eq!(
            download_scalar_znx::<BR>(&res_ref_backend),
            download_scalar_znx::<BT>(&res_test_backend)
        );
    }
}

pub fn test_vec_znx_mul_xp_minus_one<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxMulXpMinusOne<BR>,
    Module<BT>: VecZnxMulXpMinusOne<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);

        let a_digest: u64 = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let res_ref = module_host.vec_znx_alloc(cols, res_size);
            let res_test = module_host.vec_znx_alloc(cols, res_size);
            let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
            let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

            let p: i64 = -5;

            // Normalize on c
            for i in 0..cols {
                module_ref.vec_znx_mul_xp_minus_one(
                    p,
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                    i,
                    &vec_znx_backend_ref::<BR>(&a_ref),
                    i,
                );
                module_test.vec_znx_mul_xp_minus_one(
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
                module_ref.vec_znx_mul_xp_minus_one(
                    p,
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                    i,
                    &vec_znx_backend_ref::<BR>(&a_ref),
                    i,
                );
                module_test.vec_znx_mul_xp_minus_one(
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
    Module<BR>: VecZnxMulXpMinusOneAssign<BR> + VecZnxMulXpMinusOneAssignTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxMulXpMinusOneAssign<BT> + VecZnxMulXpMinusOneAssignTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_mul_xp_minus_one_assign_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_mul_xp_minus_one_assign_tmp_bytes());

    for size in [1, 2, 3, 4] {
        let mut res_ref = module_host.vec_znx_alloc(cols, size);
        let mut res_test = module_host.vec_znx_alloc(cols, size);

        // Fill a with random i64
        res_ref.fill_uniform(base2k, &mut source);
        res_test.raw_mut().copy_from_slice(res_ref.raw());
        let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
        let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

        let p: i64 = -7;

        for i in 0..cols {
            module_ref.vec_znx_mul_xp_minus_one_assign(
                p,
                &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                i,
                &mut scratch_ref.arena(),
            );
            module_test.vec_znx_mul_xp_minus_one_assign(
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
            module_ref.vec_znx_mul_xp_minus_one_assign(
                p,
                &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                i,
                &mut scratch_ref.arena(),
            );
            module_test.vec_znx_mul_xp_minus_one_assign(
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
    Module<BR>: VecZnxNegate<BR>,
    Module<BT>: VecZnxNegate<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let mut res_ref = module_host.vec_znx_alloc(cols, res_size);
            let mut res_test = module_host.vec_znx_alloc(cols, res_size);

            res_ref.fill_uniform(base2k, &mut source);
            res_test.raw_mut().copy_from_slice(res_ref.raw());
            let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
            let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

            for i in 0..cols {
                module_ref.vec_znx_negate(
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                    i,
                    &vec_znx_backend_ref::<BR>(&a_ref),
                    i,
                );
                module_test.vec_znx_negate(
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

pub fn test_vec_znx_negate_matches_wrapper<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    _module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BT>: VecZnxNegate<BT>,
{
    let base2k = params.base2k;
    let _n: usize = module_test.n();
    let cols: usize = 2;
    let a_col: usize = 0;
    let res_col: usize = 1;
    let mut source: Source = Source::new([6u8; 32]);

    for a_size in [1, 2, 3, 4] {
        let mut a = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_backend = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let mut wrapper = module_host.vec_znx_alloc(cols, res_size);
            let mut backend = module_host.vec_znx_alloc(cols, res_size);
            wrapper.fill_uniform(base2k, &mut source);
            backend.data_mut().copy_from_slice(wrapper.data());
            let mut wrapper_backend = upload_vec_znx::<BT>(&wrapper);
            let mut backend_backend = upload_vec_znx::<BT>(&backend);

            module_test.vec_znx_negate(
                &mut vec_znx_backend_mut::<BT>(&mut wrapper_backend),
                res_col,
                &vec_znx_backend_ref::<BT>(&a_backend),
                a_col,
            );
            module_test.vec_znx_negate(
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
    Module<BR>: VecZnxNegateAssign<BR>,
    Module<BT>: VecZnxNegateAssign<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for res_size in [1, 2, 3, 4] {
        let mut res_ref = module_host.vec_znx_alloc(cols, res_size);
        let mut res_test = module_host.vec_znx_alloc(cols, res_size);

        res_ref.fill_uniform(base2k, &mut source);
        res_test.raw_mut().copy_from_slice(res_ref.raw());
        let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
        let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

        for i in 0..cols {
            module_ref.vec_znx_negate_assign(&mut vec_znx_backend_mut::<BR>(&mut res_ref_backend), i);
            module_test.vec_znx_negate_assign(&mut vec_znx_backend_mut::<BT>(&mut res_test_backend), i);
        }

        assert_eq!(
            download_vec_znx::<BR>(&res_ref_backend),
            download_vec_znx::<BT>(&res_test_backend)
        );
    }
}

pub fn test_vec_znx_negate_assign_matches_wrapper<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    _module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BT>: VecZnxNegateAssign<BT>,
{
    let base2k = params.base2k;
    let _n: usize = module_test.n();
    let cols: usize = 2;
    let mut source: Source = Source::new([7u8; 32]);

    for res_size in [1, 2, 3, 4] {
        for col_i in 0..cols {
            let mut wrapper = module_host.vec_znx_alloc(cols, res_size);
            let mut backend = module_host.vec_znx_alloc(cols, res_size);
            wrapper.fill_uniform(base2k, &mut source);
            backend.data_mut().copy_from_slice(wrapper.data());
            let mut wrapper_backend = upload_vec_znx::<BT>(&wrapper);
            let mut backend_backend = upload_vec_znx::<BT>(&backend);

            module_test.vec_znx_negate_assign(&mut vec_znx_backend_mut::<BT>(&mut wrapper_backend), col_i);
            module_test.vec_znx_negate_assign(&mut vec_znx_backend_mut::<BT>(&mut backend_backend), col_i);

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
        let mut a = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let mut res_ref = module_host.vec_znx_alloc(cols, res_size);
            let mut res_test = module_host.vec_znx_alloc(cols, res_size);

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
                        res_size * base2k,
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
                        res_size * base2k,
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

            let same_base = (base2k, base2k, res_size.saturating_sub(1).max(1) * base2k - 1, 0);
            for (a_base, res_base, res_k, offset) in std::iter::once(same_base).chain(cross_normalization_cases(a_size, res_size))
            {
                let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
                let mut res_test_backend = upload_vec_znx::<BT>(&res_test);
                for i in 0..cols {
                    module_ref.vec_znx_normalize(
                        &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                        res_base,
                        res_k,
                        offset,
                        i,
                        &vec_znx_backend_ref::<BR>(&a_ref),
                        a_base,
                        i,
                        &mut scratch_ref.arena(),
                    );
                    module_test.vec_znx_normalize(
                        &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                        res_base,
                        res_k,
                        offset,
                        i,
                        &vec_znx_backend_ref::<BT>(&a_test),
                        a_base,
                        i,
                        &mut scratch_test.arena(),
                    );
                }
                let res_ref_host = download_vec_znx::<BR>(&res_ref_backend);
                let res_test_host = download_vec_znx::<BT>(&res_test_backend);
                assert_eq!(
                    res_ref_host, res_test_host,
                    "a_base={a_base} res_base={res_base} k={res_k} offset={offset}"
                );
                assert_canonical(&res_test_host, res_base, res_k);
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
    Module<BR>: VecZnxNormalizeAssign<BR> + VecZnxNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxNormalizeAssign<BT> + VecZnxNormalizeTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_normalize_tmp_bytes());

    for res_size in [1usize, 2, 3, 4] {
        let res_k = res_size.saturating_sub(1).max(1) * base2k - 1;
        let mut res_ref = module_host.vec_znx_alloc(cols, res_size);
        let mut res_test = module_host.vec_znx_alloc(cols, res_size);

        res_ref.fill_uniform(base2k, &mut source);
        res_test.raw_mut().copy_from_slice(res_ref.raw());
        let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
        let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

        // Reference
        for i in 0..cols {
            module_ref.vec_znx_normalize_assign(
                base2k,
                res_k,
                &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                i,
                &mut scratch_ref.arena(),
            );
            module_test.vec_znx_normalize_assign(
                base2k,
                res_k,
                &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                i,
                &mut scratch_test.arena(),
            );
        }

        let res_ref_host = download_vec_znx::<BR>(&res_ref_backend);
        let res_test_host = download_vec_znx::<BT>(&res_test_backend);
        assert_eq!(res_ref_host, res_test_host);
        assert_canonical(&res_test_host, base2k, res_k);
    }
}

pub fn test_vec_znx_rotate<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxRotate<BR>,
    Module<BT>: VecZnxRotate<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let res_ref = module_host.vec_znx_alloc(cols, res_size);
            let res_test = module_host.vec_znx_alloc(cols, res_size);
            let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
            let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

            let p: i64 = -5;

            // Normalize on c
            for i in 0..cols {
                module_ref.vec_znx_rotate(
                    p,
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                    i,
                    &vec_znx_backend_ref::<BR>(&a_ref),
                    i,
                );
                module_test.vec_znx_rotate(
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
                module_ref.vec_znx_rotate(
                    p,
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                    i,
                    &vec_znx_backend_ref::<BR>(&a_ref),
                    i,
                );
                module_test.vec_znx_rotate(
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
    Module<BR>: VecZnxRotateAssign<BR> + VecZnxRotateAssignTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxRotateAssign<BT> + VecZnxRotateAssignTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_rotate_assign_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_rotate_assign_tmp_bytes());

    for size in [1, 2, 3, 4] {
        let mut res_ref = module_host.vec_znx_alloc(cols, size);
        let mut res_test = module_host.vec_znx_alloc(cols, size);

        // Fill a with random i64
        res_ref.fill_uniform(base2k, &mut source);
        res_test.raw_mut().copy_from_slice(res_ref.raw());
        let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
        let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

        let p: i64 = -5;

        // Normalize on c
        for i in 0..cols {
            module_ref.vec_znx_rotate_assign(
                p,
                &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                i,
                &mut scratch_ref.arena(),
            );
            module_test.vec_znx_rotate_assign(
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
            module_ref.vec_znx_rotate_assign(
                p,
                &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                i,
                &mut scratch_ref.arena(),
            );
            module_test.vec_znx_rotate_assign(
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

pub fn test_vec_znx_fill_uniform<B: crate::test_suite::TestBackend>(_params: &TestParams, module: &Module<B>)
where
    Module<B>: VecZnxFillUniformSource<B>,
{
    let n: usize = module.n();
    let base2k: usize = 17;
    let size: usize = 5;
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;
    let k = base2k * size;
    let zero: Vec<i64> = vec![0; n];
    let one_12_sqrt: f64 = 0.28867513459481287;
    (0..cols).for_each(|col_i| {
        let host_init = VecZnx::alloc(module.n(), cols, size);
        let mut a = upload_vec_znx::<B>(&host_init);
        module.vec_znx_fill_uniform_source(base2k, k, &mut vec_znx_backend_mut::<B>(&mut a), col_i, &mut source);
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

    let k = 3 * base2k + 1;
    let live_size = k.div_ceil(base2k);
    let low_mask = (1i64 << (base2k - k % base2k)) - 1;
    let mut host_init = VecZnx::alloc(module.n(), cols, size);
    host_init.raw_mut().fill(0xff);
    let mut a = upload_vec_znx::<B>(&host_init);
    module.vec_znx_fill_uniform_source(base2k, k, &mut vec_znx_backend_mut::<B>(&mut a), 0, &mut source);
    let a = download_vec_znx::<B>(&a);
    assert!(a.at(0, live_size - 1).iter().all(|value| value & low_mask == 0));
    for limb in live_size..size {
        assert_eq!(a.at(0, limb), zero);
    }
}

pub fn test_vec_znx_add_normal<B: crate::test_suite::TestBackend>(_params: &TestParams, module: &Module<B>)
where
    Module<B>: VecZnxAddNormalSource<B>,
{
    let n: usize = module.n();
    let base2k: usize = 17;
    let size: usize = 5;
    let noise_infos = NoiseInfos::new(2 * 17 - 3, 3.2, 6.0 * 3.2).unwrap();
    let mut source_xe: Source = Source::new([0u8; 32]);
    let cols: usize = 2;
    let zero: Vec<i64> = vec![0; n];
    let k_f64: f64 = (1u64 << noise_infos.k as u64) as f64;
    let sqrt2: f64 = SQRT_2;
    (0..cols).for_each(|col_i| {
        let host_init = VecZnx::alloc(module.n(), cols, size);
        let mut a = upload_vec_znx::<B>(&host_init);
        module.vec_znx_add_normal_source(
            base2k,
            &mut vec_znx_backend_mut::<B>(&mut a),
            col_i,
            noise_infos,
            &mut source_xe,
        );
        module.vec_znx_add_normal_source(
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
                let (limb, shift) = noise_infos.target_limb_and_shift(base2k);
                let low_mask = (1i64 << shift) - 1;
                assert!(a.at(col_i, limb).iter().all(|value| value & low_mask == 0));
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
    Module<BR>: VecZnxLsh<BR> + VecZnxLshTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxLsh<BT> + VecZnxLshTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_lsh_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_lsh_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            for k in 0..res_size * base2k {
                let mut res_ref = module_host.vec_znx_alloc(cols, res_size);
                let mut res_test = module_host.vec_znx_alloc(cols, res_size);

                // Set d to garbage
                res_ref.fill_uniform(base2k, &mut source);
                res_test.fill_uniform(base2k, &mut source);
                let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
                let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_lsh(
                        base2k,
                        k,
                        &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                        i,
                        &vec_znx_backend_ref::<BR>(&a_ref),
                        i,
                        &mut scratch_ref.arena(),
                    );
                    module_test.vec_znx_lsh(
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
    Module<BR>: VecZnxLshAssign<BR> + VecZnxLshTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxLshAssign<BT> + VecZnxLshTmpBytes,
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
            let mut res_ref = module_host.vec_znx_alloc(cols, res_size);
            let mut res_test = module_host.vec_znx_alloc(cols, res_size);

            res_ref.fill_uniform(base2k, &mut source);
            res_test.raw_mut().copy_from_slice(res_ref.raw());
            let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
            let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

            for i in 0..cols {
                module_ref.vec_znx_lsh_assign(
                    base2k,
                    k,
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                    i,
                    &mut scratch_ref.arena(),
                );
                module_test.vec_znx_lsh_assign(
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
    Module<BR>: VecZnxRsh<BR> + VecZnxRshTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxRsh<BT> + VecZnxRshTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_rsh_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_rsh_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            for k in 0..res_size * base2k {
                let mut res_ref = module_host.vec_znx_alloc(cols, res_size);
                let mut res_test = module_host.vec_znx_alloc(cols, res_size);

                // Set d to garbage
                res_ref.fill_uniform(base2k, &mut source);
                res_test.fill_uniform(base2k, &mut source);
                let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
                let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_rsh(
                        base2k,
                        k,
                        &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                        i,
                        &vec_znx_backend_ref::<BR>(&a_ref),
                        i,
                        &mut scratch_ref.arena(),
                    );
                    module_test.vec_znx_rsh(
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
    Module<BR>: VecZnxRshAssign<BR> + VecZnxRshTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxRshAssign<BT> + VecZnxRshTmpBytes,
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
            let mut res_ref = module_host.vec_znx_alloc(cols, res_size);
            let mut res_test = module_host.vec_znx_alloc(cols, res_size);

            res_ref.fill_uniform(base2k, &mut source);
            res_test.raw_mut().copy_from_slice(res_ref.raw());
            let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
            let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

            for i in 0..cols {
                module_ref.vec_znx_rsh_assign(
                    base2k,
                    k,
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                    i,
                    &mut scratch_ref.arena(),
                );
                module_test.vec_znx_rsh_assign(
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

pub fn test_vec_znx_sub<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxSub<BR>,
    Module<BT>: VecZnxSub<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);

        for b_size in [1, 2, 3, 4] {
            let mut b = module_host.vec_znx_alloc(cols, b_size);
            b.fill_uniform(base2k, &mut source);
            let b_digest: u64 = b.digest_u64();
            let b_ref = upload_vec_znx::<BR>(&b);
            let b_test = upload_vec_znx::<BT>(&b);

            for res_size in [1, 2, 3, 4] {
                let mut res_ref = module_host.vec_znx_alloc(cols, res_size);
                let mut res_test = module_host.vec_znx_alloc(cols, res_size);

                // Set d to garbage
                res_ref.fill_uniform(base2k, &mut source);
                res_test.fill_uniform(base2k, &mut source);
                let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
                let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

                // Reference
                for i in 0..cols {
                    module_test.vec_znx_sub(
                        &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                        i,
                        &vec_znx_backend_ref::<BT>(&a_test),
                        i,
                        &vec_znx_backend_ref::<BT>(&b_test),
                        i,
                    );
                    module_ref.vec_znx_sub(
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
    Module<BR>: VecZnxSubAssign<BR>,
    Module<BT>: VecZnxSubAssign<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let mut res_ref = module_host.vec_znx_alloc(cols, res_size);
            let mut res_test = module_host.vec_znx_alloc(cols, res_size);

            res_ref.fill_uniform(base2k, &mut source);
            res_test.raw_mut().copy_from_slice(res_ref.raw());
            let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
            let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

            for i in 0..cols {
                module_test.vec_znx_sub_assign(
                    &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                    i,
                    &vec_znx_backend_ref::<BT>(&a_test),
                    i,
                );
                module_ref.vec_znx_sub_assign(
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
    Module<BR>: VecZnxSubNegateAssign<BR>,
    Module<BT>: VecZnxSubNegateAssign<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let mut res_ref = module_host.vec_znx_alloc(cols, res_size);
            let mut res_test = module_host.vec_znx_alloc(cols, res_size);

            res_ref.fill_uniform(base2k, &mut source);
            res_test.raw_mut().copy_from_slice(res_ref.raw());
            let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
            let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

            for i in 0..cols {
                module_test.vec_znx_sub_negate_assign(
                    &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                    i,
                    &vec_znx_backend_ref::<BT>(&a_test),
                    i,
                );
                module_ref.vec_znx_sub_negate_assign(
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
    Module<BR>: VecZnxSwitchRing<BR> + ModuleNew<BR>,
    Module<BT>: VecZnxSwitchRing<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a = module_host.vec_znx_alloc(cols, a_size);

        // Fill a with random i64
        a.fill_uniform(base2k, &mut source);
        let a_digest: u64 = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            {
                let mut res_ref = VecZnx::alloc(n << 1, cols, res_size);
                let mut res_test = VecZnx::alloc(n << 1, cols, res_size);

                res_ref.fill_uniform(base2k, &mut source);
                res_test.fill_uniform(base2k, &mut source);
                let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
                let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

                // Normalize on c
                for i in 0..cols {
                    module_ref.vec_znx_switch_ring(
                        &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                        i,
                        &vec_znx_backend_ref::<BR>(&a_ref),
                        i,
                    );
                    module_test.vec_znx_switch_ring(
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
                let mut res_ref = VecZnx::alloc(n >> 1, cols, res_size);
                let mut res_test = VecZnx::alloc(n >> 1, cols, res_size);

                res_ref.fill_uniform(base2k, &mut source);
                res_test.fill_uniform(base2k, &mut source);
                let mut res_ref_backend = upload_vec_znx::<BR>(&res_ref);
                let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

                // Normalize on c
                for i in 0..cols {
                    module_ref.vec_znx_switch_ring(
                        &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                        i,
                        &vec_znx_backend_ref::<BR>(&a_ref),
                        i,
                    );
                    module_test.vec_znx_switch_ring(
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

pub fn test_vec_znx_switch_ring_matches_wrapper<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    _module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: ModuleNew<BR>,
    Module<BT>: VecZnxSwitchRing<BT>,
{
    let base2k = params.base2k;
    let n: usize = module_test.n();
    let cols: usize = 2;
    let a_col: usize = 0;
    let res_col: usize = 1;
    let mut source: Source = Source::new([4u8; 32]);

    for a_size in [1, 2, 3, 4] {
        let mut a = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_backend = upload_vec_znx::<BT>(&a);

        for res_n in [n >> 1, n << 1] {
            for res_size in [1, 2, 3, 4] {
                let mut wrapper = VecZnx::alloc(res_n, cols, res_size);
                let mut backend = VecZnx::alloc(res_n, cols, res_size);
                wrapper.fill_uniform(base2k, &mut source);
                backend.data_mut().copy_from_slice(wrapper.data());
                let mut wrapper_backend = upload_vec_znx::<BT>(&wrapper);
                let mut backend_backend = upload_vec_znx::<BT>(&backend);

                module_test.vec_znx_switch_ring(
                    &mut vec_znx_backend_mut::<BT>(&mut wrapper_backend),
                    res_col,
                    &vec_znx_backend_ref::<BT>(&a_backend),
                    a_col,
                );
                module_test.vec_znx_switch_ring(
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
