use super::{
    TestParams, download_scalar_znx, download_vec_znx, scalar_znx_backend_mut, scalar_znx_backend_ref, upload_scalar_znx,
    upload_vec_znx, vec_znx_backend_mut, vec_znx_backend_ref,
};

use rand_core::Rng;

use crate::{
    api::{
        ScalarZnxAutomorphism, ScratchOwnedAlloc, VecZnxAdd, VecZnxAddAssign, VecZnxAddScalarAssign, VecZnxAutomorphism,
        VecZnxAutomorphismAssign, VecZnxAutomorphismAssignTmpBytes, VecZnxCopy, VecZnxFillUniformSource,
        VecZnxFillUniformSourceAll, VecZnxLsh, VecZnxLshAssign, VecZnxLshTmpBytes, VecZnxMulXpMinusOne,
        VecZnxMulXpMinusOneAssign, VecZnxMulXpMinusOneAssignTmpBytes, VecZnxNegate, VecZnxNegateAssign, VecZnxNormalize,
        VecZnxNormalizeAssign, VecZnxNormalizeTmpBytes, VecZnxRotate, VecZnxRotateAssign, VecZnxRotateAssignTmpBytes, VecZnxRsh,
        VecZnxRshAssign, VecZnxRshTmpBytes, VecZnxSub, VecZnxSubAssign, VecZnxSubNegateAssign, VecZnxSwitchRing, VecZnxZero,
    },
    layouts::{
        DigestU64, HostBytesBackend, HostDataRef, Module, ScalarZnxAsVecZnxBackendMut, ScratchOwned, VecZnx, ZnxView, ZnxViewMut,
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
    _module_host: &Module<HostBytesBackend>,
    _module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BT>: VecZnxZero<BT> + VecZnxFillUniformSourceAll<BT>,
{
    let base2k = params.base2k;
    let _n: usize = params.n;
    let cols: usize = 2;
    let mut source: Source = Source::new([5u8; 32]);

    for size in [1, 2, 3, 4] {
        for col_i in 0..cols {
            let mut backend = module_test.vec_znx_alloc(params.n, cols, size);
            module_test.vec_znx_fill_uniform_source_all(base2k, size * base2k, &mut backend, &mut source);
            let mut expected = download_vec_znx::<BT>(&backend);

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
        let mut a = module.vec_znx_alloc(n, 2, size);
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
    Module<BR>: VecZnxAddScalarAssign<BR> + VecZnxFillUniformSource<BR> + VecZnxFillUniformSourceAll<BR>,
    Module<BT>: VecZnxAddScalarAssign<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut b_ref = module_ref.scalar_znx_alloc(params.n, cols);
    for col in 0..cols {
        module_ref.vec_znx_fill_uniform_source(
            base2k,
            base2k,
            &mut ScalarZnxAsVecZnxBackendMut::<BR>::as_vec_znx_backend_mut(&mut b_ref),
            col,
            &mut source,
        );
    }
    let b = download_scalar_znx::<BR>(&b_ref);
    let b_digest: u64 = b.digest_u64();
    let b_test = upload_scalar_znx::<BT>(&b);

    for res_size in [1usize, 2, 3, 4] {
        let mut rest_ref_backend = module_ref.vec_znx_alloc(params.n, cols, res_size);
        module_ref.vec_znx_fill_uniform_source_all(base2k, res_size * base2k, &mut rest_ref_backend, &mut source);
        let mut res_test = module_host.vec_znx_alloc(params.n, cols, res_size);

        let rest_ref = download_vec_znx::<BR>(&rest_ref_backend);
        res_test.raw_mut().copy_from_slice(rest_ref.raw());
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
    Module<BR>: VecZnxAdd<BR> + VecZnxFillUniformSourceAll<BR>,
    Module<BT>: VecZnxAdd<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let cols: usize = 2;
    let mut source: Source = Source::new([13u8; 32]);

    for a_size in [1, 2, 3, 4] {
        let mut a_ref = module_ref.vec_znx_alloc(params.n, cols, a_size);
        module_ref.vec_znx_fill_uniform_source_all(base2k, a_size * base2k, &mut a_ref, &mut source);
        let a = download_vec_znx::<BR>(&a_ref);
        let a_digest: u64 = a.digest_u64();
        let a_test = upload_vec_znx::<BT>(&a);

        for b_size in [1, 2, 3, 4] {
            let mut b_ref = module_ref.vec_znx_alloc(params.n, cols, b_size);
            module_ref.vec_znx_fill_uniform_source_all(base2k, b_size * base2k, &mut b_ref, &mut source);
            let b = download_vec_znx::<BR>(&b_ref);
            let b_digest: u64 = b.digest_u64();
            let b_test = upload_vec_znx::<BT>(&b);

            for res_size in [1, 2, 3, 4] {
                let mut wrapper_backend = module_ref.vec_znx_alloc(params.n, cols, res_size);
                module_ref.vec_znx_fill_uniform_source_all(base2k, res_size * base2k, &mut wrapper_backend, &mut source);
                let mut backend = module_host.vec_znx_alloc(params.n, cols, res_size);

                let wrapper = download_vec_znx::<BR>(&wrapper_backend);
                backend.data_mut().copy_from_slice(wrapper.data());
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
    Module<BR>: VecZnxAddAssign<BR> + VecZnxFillUniformSourceAll<BR>,
    Module<BT>: VecZnxAddAssign<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a_ref = module_ref.vec_znx_alloc(params.n, cols, a_size);
        module_ref.vec_znx_fill_uniform_source_all(base2k, a_size * base2k, &mut a_ref, &mut source);
        let a = download_vec_znx::<BR>(&a_ref);
        let a_digest: u64 = a.digest_u64();
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let mut res_ref_backend = module_ref.vec_znx_alloc(params.n, cols, res_size);
            module_ref.vec_znx_fill_uniform_source_all(base2k, res_size * base2k, &mut res_ref_backend, &mut source);
            let mut res_test = module_host.vec_znx_alloc(params.n, cols, res_size);

            let res_ref = download_vec_znx::<BR>(&res_ref_backend);
            res_test.raw_mut().copy_from_slice(res_ref.raw());
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
    Module<BT>: VecZnxAddAssign<BT> + VecZnxFillUniformSourceAll<BT>,
{
    let base2k = params.base2k;
    let _n: usize = params.n;
    let cols: usize = 2;
    let mut source: Source = Source::new([14u8; 32]);

    for a_size in [1, 2, 3, 4] {
        let mut a_backend = module_test.vec_znx_alloc(params.n, cols, a_size);
        module_test.vec_znx_fill_uniform_source_all(base2k, a_size * base2k, &mut a_backend, &mut source);
        let a = download_vec_znx::<BT>(&a_backend);
        let a_digest: u64 = a.digest_u64();

        for res_size in [1, 2, 3, 4] {
            let mut wrapper_backend = module_test.vec_znx_alloc(params.n, cols, res_size);
            module_test.vec_znx_fill_uniform_source_all(base2k, res_size * base2k, &mut wrapper_backend, &mut source);
            let mut backend = module_host.vec_znx_alloc(params.n, cols, res_size);

            let wrapper = download_vec_znx::<BT>(&wrapper_backend);
            backend.data_mut().copy_from_slice(wrapper.data());
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
    Module<BR>: VecZnxAutomorphism<BR> + VecZnxFillUniformSourceAll<BR>,
    Module<BT>: VecZnxAutomorphism<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a_ref = module_ref.vec_znx_alloc(params.n, cols, a_size);
        module_ref.vec_znx_fill_uniform_source_all(base2k, a_size * base2k, &mut a_ref, &mut source);
        let a = download_vec_znx::<BR>(&a_ref);
        let a_digest: u64 = a.digest_u64();
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let res_ref = module_host.vec_znx_alloc(params.n, cols, res_size);
            let res_test = module_host.vec_znx_alloc(params.n, cols, res_size);
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
    Module<BR>: VecZnxAutomorphismAssign<BR> + VecZnxAutomorphismAssignTmpBytes + VecZnxFillUniformSourceAll<BR>,
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
        let mut res_ref_backend = module_ref.vec_znx_alloc(params.n, cols, size);
        module_ref.vec_znx_fill_uniform_source_all(base2k, size * base2k, &mut res_ref_backend, &mut source);
        let mut res_test = module_host.vec_znx_alloc(params.n, cols, size);

        let res_ref = download_vec_znx::<BR>(&res_ref_backend);
        res_test.raw_mut().copy_from_slice(res_ref.raw());
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
    _module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxCopy<BR> + VecZnxFillUniformSourceAll<BR>,
    Module<BT>: VecZnxCopy<BT> + VecZnxFillUniformSourceAll<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a_ref = module_ref.vec_znx_alloc(params.n, cols, a_size);
        module_ref.vec_znx_fill_uniform_source_all(base2k, a_size * base2k, &mut a_ref, &mut source);
        let a = download_vec_znx::<BR>(&a_ref);
        let a_digest: u64 = a.digest_u64();
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let mut res_0_backend = module_ref.vec_znx_alloc(params.n, cols, res_size);
            module_ref.vec_znx_fill_uniform_source_all(base2k, res_size * base2k, &mut res_0_backend, &mut source);
            let mut res_1_backend = module_test.vec_znx_alloc(params.n, cols, res_size);
            module_test.vec_znx_fill_uniform_source_all(base2k, res_size * base2k, &mut res_1_backend, &mut source);

            // Set d to garbage

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
    Module<BT>: VecZnxCopy<BT> + VecZnxFillUniformSourceAll<BT>,
{
    let base2k = params.base2k;
    let _n: usize = params.n;
    let cols: usize = 2;
    let a_col: usize = 0;
    let res_col: usize = 1;
    let mut source: Source = Source::new([3u8; 32]);

    for a_size in [1, 2, 3, 4] {
        let mut a_backend = module_test.vec_znx_alloc(params.n, cols, a_size);
        module_test.vec_znx_fill_uniform_source_all(base2k, a_size * base2k, &mut a_backend, &mut source);

        for res_size in [1, 2, 3, 4] {
            let mut wrapper_backend = module_test.vec_znx_alloc(params.n, cols, res_size);
            module_test.vec_znx_fill_uniform_source_all(base2k, res_size * base2k, &mut wrapper_backend, &mut source);
            let mut backend = module_host.vec_znx_alloc(params.n, cols, res_size);
            let wrapper = download_vec_znx::<BT>(&wrapper_backend);
            backend.data_mut().copy_from_slice(wrapper.data());
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
    _module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: ScalarZnxAutomorphism<BR> + VecZnxFillUniformSource<BR>,
    Module<BT>: ScalarZnxAutomorphism<BT> + VecZnxFillUniformSource<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut a_ref = module_ref.scalar_znx_alloc(params.n, cols);
    for col in 0..cols {
        module_ref.vec_znx_fill_uniform_source(
            base2k,
            base2k,
            &mut ScalarZnxAsVecZnxBackendMut::<BR>::as_vec_znx_backend_mut(&mut a_ref),
            col,
            &mut source,
        );
    }
    let a = download_scalar_znx::<BR>(&a_ref);
    let a_digest: u64 = a.digest_u64();
    let a_test = upload_scalar_znx::<BT>(&a);

    let mut res_ref_backend = module_ref.scalar_znx_alloc(params.n, cols);
    let mut res_test_backend = module_test.scalar_znx_alloc(params.n, cols);
    for col in 0..cols {
        module_ref.vec_znx_fill_uniform_source(
            base2k,
            base2k,
            &mut ScalarZnxAsVecZnxBackendMut::<BR>::as_vec_znx_backend_mut(&mut res_ref_backend),
            col,
            &mut source,
        );
    }
    for col in 0..cols {
        module_test.vec_znx_fill_uniform_source(
            base2k,
            base2k,
            &mut ScalarZnxAsVecZnxBackendMut::<BT>::as_vec_znx_backend_mut(&mut res_test_backend),
            col,
            &mut source,
        );
    }

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
    Module<BR>: VecZnxMulXpMinusOne<BR> + VecZnxFillUniformSourceAll<BR>,
    Module<BT>: VecZnxMulXpMinusOne<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a_ref = module_ref.vec_znx_alloc(params.n, cols, a_size);
        module_ref.vec_znx_fill_uniform_source_all(base2k, a_size * base2k, &mut a_ref, &mut source);
        let a = download_vec_znx::<BR>(&a_ref);

        let a_digest: u64 = a.digest_u64();
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let res_ref = module_host.vec_znx_alloc(params.n, cols, res_size);
            let res_test = module_host.vec_znx_alloc(params.n, cols, res_size);
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
    Module<BR>: VecZnxMulXpMinusOneAssign<BR> + VecZnxMulXpMinusOneAssignTmpBytes + VecZnxFillUniformSourceAll<BR>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxMulXpMinusOneAssign<BT> + VecZnxMulXpMinusOneAssignTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_mul_xp_minus_one_assign_tmp_bytes(4));
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_mul_xp_minus_one_assign_tmp_bytes(4));

    for size in [1, 2, 3, 4] {
        let mut res_ref_backend = module_ref.vec_znx_alloc(params.n, cols, size);
        module_ref.vec_znx_fill_uniform_source_all(base2k, size * base2k, &mut res_ref_backend, &mut source);
        let mut res_test = module_host.vec_znx_alloc(params.n, cols, size);

        let res_ref = download_vec_znx::<BR>(&res_ref_backend);
        res_test.raw_mut().copy_from_slice(res_ref.raw());
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
    Module<BR>: VecZnxNegate<BR> + VecZnxFillUniformSourceAll<BR>,
    Module<BT>: VecZnxNegate<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a_ref = module_ref.vec_znx_alloc(params.n, cols, a_size);
        module_ref.vec_znx_fill_uniform_source_all(base2k, a_size * base2k, &mut a_ref, &mut source);
        let a = download_vec_znx::<BR>(&a_ref);
        let a_digest: u64 = a.digest_u64();
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let mut res_ref_backend = module_ref.vec_znx_alloc(params.n, cols, res_size);
            module_ref.vec_znx_fill_uniform_source_all(base2k, res_size * base2k, &mut res_ref_backend, &mut source);
            let mut res_test = module_host.vec_znx_alloc(params.n, cols, res_size);

            let res_ref = download_vec_znx::<BR>(&res_ref_backend);
            res_test.raw_mut().copy_from_slice(res_ref.raw());
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
    Module<BT>: VecZnxNegate<BT> + VecZnxFillUniformSourceAll<BT>,
{
    let base2k = params.base2k;
    let _n: usize = params.n;
    let cols: usize = 2;
    let a_col: usize = 0;
    let res_col: usize = 1;
    let mut source: Source = Source::new([6u8; 32]);

    for a_size in [1, 2, 3, 4] {
        let mut a_backend = module_test.vec_znx_alloc(params.n, cols, a_size);
        module_test.vec_znx_fill_uniform_source_all(base2k, a_size * base2k, &mut a_backend, &mut source);

        for res_size in [1, 2, 3, 4] {
            let mut wrapper_backend = module_test.vec_znx_alloc(params.n, cols, res_size);
            module_test.vec_znx_fill_uniform_source_all(base2k, res_size * base2k, &mut wrapper_backend, &mut source);
            let mut backend = module_host.vec_znx_alloc(params.n, cols, res_size);
            let wrapper = download_vec_znx::<BT>(&wrapper_backend);
            backend.data_mut().copy_from_slice(wrapper.data());
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
    Module<BR>: VecZnxNegateAssign<BR> + VecZnxFillUniformSourceAll<BR>,
    Module<BT>: VecZnxNegateAssign<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for res_size in [1, 2, 3, 4] {
        let mut res_ref_backend = module_ref.vec_znx_alloc(params.n, cols, res_size);
        module_ref.vec_znx_fill_uniform_source_all(base2k, res_size * base2k, &mut res_ref_backend, &mut source);
        let mut res_test = module_host.vec_znx_alloc(params.n, cols, res_size);

        let res_ref = download_vec_znx::<BR>(&res_ref_backend);
        res_test.raw_mut().copy_from_slice(res_ref.raw());
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
    Module<BT>: VecZnxNegateAssign<BT> + VecZnxFillUniformSourceAll<BT>,
{
    let base2k = params.base2k;
    let _n: usize = params.n;
    let cols: usize = 2;
    let mut source: Source = Source::new([7u8; 32]);

    for res_size in [1, 2, 3, 4] {
        for col_i in 0..cols {
            let mut wrapper_backend = module_test.vec_znx_alloc(params.n, cols, res_size);
            module_test.vec_znx_fill_uniform_source_all(base2k, res_size * base2k, &mut wrapper_backend, &mut source);
            let mut backend = module_host.vec_znx_alloc(params.n, cols, res_size);
            let wrapper = download_vec_znx::<BT>(&wrapper_backend);
            backend.data_mut().copy_from_slice(wrapper.data());
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
    _module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxNormalize<BR> + VecZnxNormalizeTmpBytes + VecZnxFillUniformSourceAll<BR>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxNormalize<BT> + VecZnxNormalizeTmpBytes + VecZnxFillUniformSourceAll<BT>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a_ref = module_ref.vec_znx_alloc(params.n, cols, a_size);
        module_ref.vec_znx_fill_uniform_source_all(base2k, a_size * base2k, &mut a_ref, &mut source);
        let a = download_vec_znx::<BR>(&a_ref);
        let a_digest: u64 = a.digest_u64();
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            for res_offset in -(base2k as i64)..=(base2k as i64) {
                // Set d to garbage
                let mut res_ref_backend = module_ref.vec_znx_alloc(params.n, cols, res_size);
                module_ref.vec_znx_fill_uniform_source_all(base2k, res_size * base2k, &mut res_ref_backend, &mut source);
                let mut res_test_backend = module_test.vec_znx_alloc(params.n, cols, res_size);
                module_test.vec_znx_fill_uniform_source_all(base2k, res_size * base2k, &mut res_test_backend, &mut source);

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
                let mut res_ref_backend = module_ref.vec_znx_alloc(params.n, cols, res_size);
                module_ref.vec_znx_fill_uniform_source_all(base2k, res_size * base2k, &mut res_ref_backend, &mut source);
                let mut res_test_backend = module_test.vec_znx_alloc(params.n, cols, res_size);
                module_test.vec_znx_fill_uniform_source_all(base2k, res_size * base2k, &mut res_test_backend, &mut source);
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
    Module<BR>: VecZnxNormalizeAssign<BR> + VecZnxNormalizeTmpBytes + VecZnxFillUniformSourceAll<BR>,
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
        let mut res_ref_backend = module_ref.vec_znx_alloc(params.n, cols, res_size);
        module_ref.vec_znx_fill_uniform_source_all(base2k, res_size * base2k, &mut res_ref_backend, &mut source);
        let mut res_test = module_host.vec_znx_alloc(params.n, cols, res_size);

        let res_ref = download_vec_znx::<BR>(&res_ref_backend);
        res_test.raw_mut().copy_from_slice(res_ref.raw());
        let mut res_test_backend = upload_vec_znx::<BT>(&res_test);

        // Reference
        for i in 0..cols {
            module_ref.vec_znx_normalize_assign(
                base2k,
                res_k,
                0,
                &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                i,
                &mut scratch_ref.arena(),
            );
            module_test.vec_znx_normalize_assign(
                base2k,
                res_k,
                0,
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
    Module<BR>: VecZnxRotate<BR> + VecZnxFillUniformSourceAll<BR>,
    Module<BT>: VecZnxRotate<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a_ref = module_ref.vec_znx_alloc(params.n, cols, a_size);
        module_ref.vec_znx_fill_uniform_source_all(base2k, a_size * base2k, &mut a_ref, &mut source);
        let a = download_vec_znx::<BR>(&a_ref);
        let a_digest: u64 = a.digest_u64();
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let res_ref = module_host.vec_znx_alloc(params.n, cols, res_size);
            let res_test = module_host.vec_znx_alloc(params.n, cols, res_size);
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
    Module<BR>: VecZnxRotateAssign<BR> + VecZnxRotateAssignTmpBytes + VecZnxFillUniformSourceAll<BR>,
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
        let mut res_ref_backend = module_ref.vec_znx_alloc(params.n, cols, size);
        module_ref.vec_znx_fill_uniform_source_all(base2k, size * base2k, &mut res_ref_backend, &mut source);
        let mut res_test = module_host.vec_znx_alloc(params.n, cols, size);

        let res_ref = download_vec_znx::<BR>(&res_ref_backend);
        res_test.raw_mut().copy_from_slice(res_ref.raw());
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

pub fn test_vec_znx_fill_uniform<B: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<B>)
where
    Module<B>: VecZnxFillUniformSource<B>,
{
    let n: usize = params.n;
    let base2k: usize = 17;
    let size: usize = 5;
    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;
    let k = base2k * size;
    let zero: Vec<i64> = vec![0; n];
    let one_12_sqrt: f64 = 0.28867513459481287;
    // The estimate averages the `n` coefficients of one column, so its standard
    // error is `sigma * sqrt(1 / 5n)`; five of those give the historical 0.01 at
    // n = 4096. At a small degree the band is loose, at n = 8 it is wider than the
    // 0.5 a sample standard deviation can reach at all, so the small-degree leg of
    // the sweep is a smoke test and the tight check is the full-degree leg.
    let std_tol: f64 = 5.0 * one_12_sqrt * (0.2 / n as f64).sqrt();
    (0..cols).for_each(|col_i| {
        let host_init = VecZnx::alloc(params.n, cols, size);
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
                assert!(
                    (std - one_12_sqrt).abs() < std_tol,
                    "std={std} ~!= {one_12_sqrt} (tolerance {std_tol})",
                );
            }
        })
    });

    // Verify the specified stream and draw order independently of the backend's
    // sampling helpers. Include the radix endpoints, partial final limbs, and
    // poisoned destination tails and unselected columns.
    for (seed, prefix_len) in [([0u8; 32], 0), ([0xa5u8; 32], 13), (std::array::from_fn(|i| i as u8), 61)] {
        for base2k in [1, 17, 62] {
            for k in [1, base2k, 3 * base2k + 1, size * base2k] {
                for col in 0..cols {
                    let mut expected = VecZnx::alloc(n, cols, size);
                    for (i, value) in expected.raw_mut().iter_mut().enumerate() {
                        *value = i64::MIN + i as i64;
                    }
                    let mut actual = upload_vec_znx::<B>(&expected);
                    let mut source = Source::new(seed);
                    let mut expected_source = Source::new(seed);
                    // Exercise both fresh and previously consumed caller streams.
                    let mut prefix = [0u8; 61];
                    source.fill_bytes(&mut prefix[..prefix_len]);
                    expected_source.fill_bytes(&mut prefix[..prefix_len]);
                    let mut child_seed = [0u8; 32];
                    expected_source.fill_bytes(&mut child_seed);
                    let mut draws = Source::new(child_seed);
                    let live_size = k.div_ceil(base2k);
                    let padding = (base2k - k % base2k) % base2k;
                    let mask = (1u64 << base2k) - 1;
                    let half = 1i64 << (base2k - 1);
                    for limb in 0..size {
                        for value in expected.at_mut(col, limb) {
                            *value = if limb < live_size {
                                let digit = (draws.next_u64() & mask) as i64 - half;
                                if limb == live_size - 1 {
                                    let unit = 1i64 << padding;
                                    digit.div_euclid(unit) * unit
                                } else {
                                    digit
                                }
                            } else {
                                0
                            };
                        }
                    }

                    module.vec_znx_fill_uniform_source(base2k, k, &mut vec_znx_backend_mut::<B>(&mut actual), col, &mut source);
                    assert_eq!(
                        download_vec_znx::<B>(&actual),
                        expected,
                        "uniform sampling differs from the contract: seed={seed:?}, base2k={base2k}, k={k}, col={col}"
                    );
                    // The call consumes exactly one 32-byte seed from its caller,
                    // regardless of the number of generated limbs or coefficients.
                    let mut actual_next = [0u8; 96];
                    let mut expected_next = [0u8; 96];
                    source.fill_bytes(&mut actual_next);
                    expected_source.fill_bytes(&mut expected_next);
                    assert_eq!(
                        actual_next, expected_next,
                        "uniform sampling advanced the caller source incorrectly"
                    );
                }
            }
        }
    }
}

pub fn test_vec_znx_lsh<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    _module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxLsh<BR> + VecZnxLshTmpBytes + VecZnxFillUniformSourceAll<BR>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxLsh<BT> + VecZnxLshTmpBytes + VecZnxFillUniformSourceAll<BT>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_lsh_tmp_bytes(4));
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_lsh_tmp_bytes(4));

    for a_size in [1, 2, 3, 4] {
        let mut a_ref = module_ref.vec_znx_alloc(params.n, cols, a_size);
        module_ref.vec_znx_fill_uniform_source_all(base2k, a_size * base2k, &mut a_ref, &mut source);
        let a = download_vec_znx::<BR>(&a_ref);
        let a_digest: u64 = a.digest_u64();
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            for k in (0..res_size * base2k).chain([
                res_size * base2k,
                res_size * base2k + 1,
                a_size * base2k,
                a_size * base2k + 1,
                usize::MAX,
            ]) {
                let mut res_ref_backend = module_ref.vec_znx_alloc(params.n, cols, res_size);
                module_ref.vec_znx_fill_uniform_source_all(base2k, res_size * base2k, &mut res_ref_backend, &mut source);
                let mut res_test_backend = module_test.vec_znx_alloc(params.n, cols, res_size);
                module_test.vec_znx_fill_uniform_source_all(base2k, res_size * base2k, &mut res_test_backend, &mut source);

                // Set d to garbage

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
                // A left shift by the source width has moved every bit of `a`
                // above the integer part, whatever the destination width.
                if k >= a_size * base2k {
                    assert!(
                        download_vec_znx::<BR>(&res_ref_backend).raw().iter().all(|&x| x == 0),
                        "k = {k} past the source width must zero the destination"
                    );
                }
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
    Module<BR>: VecZnxLshAssign<BR> + VecZnxLshTmpBytes + VecZnxFillUniformSourceAll<BR>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxLshAssign<BT> + VecZnxLshTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_lsh_tmp_bytes(4));
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_lsh_tmp_bytes(4));

    for res_size in [1, 2, 3, 4] {
        for k in (0..res_size * base2k).chain([res_size * base2k, res_size * base2k + 1, usize::MAX]) {
            let mut res_ref_backend = module_ref.vec_znx_alloc(params.n, cols, res_size);
            module_ref.vec_znx_fill_uniform_source_all(base2k, res_size * base2k, &mut res_ref_backend, &mut source);
            let mut res_test = module_host.vec_znx_alloc(params.n, cols, res_size);

            let res_ref = download_vec_znx::<BR>(&res_ref_backend);
            res_test.raw_mut().copy_from_slice(res_ref.raw());
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
            if k >= res_size * base2k {
                assert!(
                    download_vec_znx::<BR>(&res_ref_backend).raw().iter().all(|&x| x == 0),
                    "k = {k} past the width must zero the destination"
                );
            }
        }
    }
}

pub fn test_vec_znx_rsh<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    _module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxRsh<BR> + VecZnxRshTmpBytes + VecZnxFillUniformSourceAll<BR>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxRsh<BT> + VecZnxRshTmpBytes + VecZnxFillUniformSourceAll<BT>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_rsh_tmp_bytes(4));
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_rsh_tmp_bytes(4));

    for a_size in [1, 2, 3, 4] {
        let mut a_ref = module_ref.vec_znx_alloc(params.n, cols, a_size);
        module_ref.vec_znx_fill_uniform_source_all(base2k, a_size * base2k, &mut a_ref, &mut source);
        let a = download_vec_znx::<BR>(&a_ref);
        let a_digest: u64 = a.digest_u64();
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            for k in (0..res_size * base2k).chain([res_size * base2k, res_size * base2k + 1, usize::MAX]) {
                let mut res_ref_backend = module_ref.vec_znx_alloc(params.n, cols, res_size);
                module_ref.vec_znx_fill_uniform_source_all(base2k, res_size * base2k, &mut res_ref_backend, &mut source);
                let mut res_test_backend = module_test.vec_znx_alloc(params.n, cols, res_size);
                module_test.vec_znx_fill_uniform_source_all(base2k, res_size * base2k, &mut res_test_backend, &mut source);

                // Set d to garbage

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
                // At exactly the width a source just past half a unit of the last
                // limb still rounds to one, so zero is only promised past it.
                if k > res_size * base2k {
                    assert!(
                        download_vec_znx::<BR>(&res_ref_backend).raw().iter().all(|&x| x == 0),
                        "k = {k} past the width must zero the destination"
                    );
                }
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
    Module<BR>: VecZnxRshAssign<BR> + VecZnxRshTmpBytes + VecZnxFillUniformSourceAll<BR>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: VecZnxRshAssign<BT> + VecZnxRshTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_rsh_tmp_bytes(4));
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_rsh_tmp_bytes(4));

    for res_size in [1, 2, 3, 4] {
        for k in (0..res_size * base2k).chain([res_size * base2k, res_size * base2k + 1, usize::MAX]) {
            let mut res_ref_backend = module_ref.vec_znx_alloc(params.n, cols, res_size);
            module_ref.vec_znx_fill_uniform_source_all(base2k, res_size * base2k, &mut res_ref_backend, &mut source);
            let mut res_test = module_host.vec_znx_alloc(params.n, cols, res_size);

            let res_ref = download_vec_znx::<BR>(&res_ref_backend);
            res_test.raw_mut().copy_from_slice(res_ref.raw());
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
            // At exactly the width a source just past half a unit of the last
            // limb still rounds to one, so zero is only promised past it.
            if k > res_size * base2k {
                assert!(
                    download_vec_znx::<BR>(&res_ref_backend).raw().iter().all(|&x| x == 0),
                    "k = {k} past the width must zero the destination"
                );
            }
        }
    }
}

pub fn test_vec_znx_sub<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    _module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxSub<BR> + VecZnxFillUniformSourceAll<BR>,
    Module<BT>: VecZnxSub<BT> + VecZnxFillUniformSourceAll<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a_ref = module_ref.vec_znx_alloc(params.n, cols, a_size);
        module_ref.vec_znx_fill_uniform_source_all(base2k, a_size * base2k, &mut a_ref, &mut source);
        let a = download_vec_znx::<BR>(&a_ref);
        let a_digest: u64 = a.digest_u64();
        let a_test = upload_vec_znx::<BT>(&a);

        for b_size in [1, 2, 3, 4] {
            let mut b_ref = module_ref.vec_znx_alloc(params.n, cols, b_size);
            module_ref.vec_znx_fill_uniform_source_all(base2k, b_size * base2k, &mut b_ref, &mut source);
            let b = download_vec_znx::<BR>(&b_ref);
            let b_digest: u64 = b.digest_u64();
            let b_test = upload_vec_znx::<BT>(&b);

            for res_size in [1, 2, 3, 4] {
                let mut res_ref_backend = module_ref.vec_znx_alloc(params.n, cols, res_size);
                module_ref.vec_znx_fill_uniform_source_all(base2k, res_size * base2k, &mut res_ref_backend, &mut source);
                let mut res_test_backend = module_test.vec_znx_alloc(params.n, cols, res_size);
                module_test.vec_znx_fill_uniform_source_all(base2k, res_size * base2k, &mut res_test_backend, &mut source);

                // Set d to garbage

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
    Module<BR>: VecZnxSubAssign<BR> + VecZnxFillUniformSourceAll<BR>,
    Module<BT>: VecZnxSubAssign<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a_ref = module_ref.vec_znx_alloc(params.n, cols, a_size);
        module_ref.vec_znx_fill_uniform_source_all(base2k, a_size * base2k, &mut a_ref, &mut source);
        let a = download_vec_znx::<BR>(&a_ref);
        let a_digest: u64 = a.digest_u64();
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let mut res_ref_backend = module_ref.vec_znx_alloc(params.n, cols, res_size);
            module_ref.vec_znx_fill_uniform_source_all(base2k, res_size * base2k, &mut res_ref_backend, &mut source);
            let mut res_test = module_host.vec_znx_alloc(params.n, cols, res_size);

            let res_ref = download_vec_znx::<BR>(&res_ref_backend);
            res_test.raw_mut().copy_from_slice(res_ref.raw());
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
    Module<BR>: VecZnxSubNegateAssign<BR> + VecZnxFillUniformSourceAll<BR>,
    Module<BT>: VecZnxSubNegateAssign<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a_ref = module_ref.vec_znx_alloc(params.n, cols, a_size);
        module_ref.vec_znx_fill_uniform_source_all(base2k, a_size * base2k, &mut a_ref, &mut source);
        let a = download_vec_znx::<BR>(&a_ref);
        let a_digest: u64 = a.digest_u64();
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let mut res_ref_backend = module_ref.vec_znx_alloc(params.n, cols, res_size);
            module_ref.vec_znx_fill_uniform_source_all(base2k, res_size * base2k, &mut res_ref_backend, &mut source);
            let mut res_test = module_host.vec_znx_alloc(params.n, cols, res_size);

            let res_ref = download_vec_znx::<BR>(&res_ref_backend);
            res_test.raw_mut().copy_from_slice(res_ref.raw());
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
    _module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxSwitchRing<BR> + VecZnxFillUniformSourceAll<BR>,
    Module<BT>: VecZnxSwitchRing<BT> + VecZnxFillUniformSourceAll<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = params.n;

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    for a_size in [1, 2, 3, 4] {
        let mut a_ref = module_ref.vec_znx_alloc(params.n, cols, a_size);
        module_ref.vec_znx_fill_uniform_source_all(base2k, a_size * base2k, &mut a_ref, &mut source);

        let a = download_vec_znx::<BR>(&a_ref);
        let a_digest: u64 = a.digest_u64();
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            {
                let mut res_ref_backend = module_ref.vec_znx_alloc(n << 1, cols, res_size);
                module_ref.vec_znx_fill_uniform_source_all(base2k, res_size * base2k, &mut res_ref_backend, &mut source);
                let mut res_test_backend = module_test.vec_znx_alloc(n << 1, cols, res_size);
                module_test.vec_znx_fill_uniform_source_all(base2k, res_size * base2k, &mut res_test_backend, &mut source);

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
                let mut res_ref_backend = module_ref.vec_znx_alloc(n >> 1, cols, res_size);
                module_ref.vec_znx_fill_uniform_source_all(base2k, res_size * base2k, &mut res_ref_backend, &mut source);
                let mut res_test_backend = module_test.vec_znx_alloc(n >> 1, cols, res_size);
                module_test.vec_znx_fill_uniform_source_all(base2k, res_size * base2k, &mut res_test_backend, &mut source);

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
    _module_host: &Module<HostBytesBackend>,
    _module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BT>: VecZnxSwitchRing<BT> + VecZnxFillUniformSourceAll<BT>,
{
    let base2k = params.base2k;
    let n: usize = params.n;
    let cols: usize = 2;
    let a_col: usize = 0;
    let res_col: usize = 1;
    let mut source: Source = Source::new([4u8; 32]);

    for a_size in [1, 2, 3, 4] {
        let mut a_backend = module_test.vec_znx_alloc(params.n, cols, a_size);
        module_test.vec_znx_fill_uniform_source_all(base2k, a_size * base2k, &mut a_backend, &mut source);

        for res_n in [n >> 1, n << 1] {
            for res_size in [1, 2, 3, 4] {
                let mut wrapper_backend = module_test.vec_znx_alloc(res_n, cols, res_size);
                module_test.vec_znx_fill_uniform_source_all(base2k, res_size * base2k, &mut wrapper_backend, &mut source);
                let mut backend = VecZnx::alloc(res_n, cols, res_size);
                let wrapper = download_vec_znx::<BT>(&wrapper_backend);
                backend.data_mut().copy_from_slice(wrapper.data());
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
