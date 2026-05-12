use super::{TestParams, download_vec_znx, upload_vec_znx, vec_znx_backend_ref};

use crate::{
    api::{
        ScratchOwnedAlloc, VecZnxBigAddAssign, VecZnxBigAddInto, VecZnxBigAddNormal, VecZnxBigAddNormalBackend,
        VecZnxBigAddSmallAssign, VecZnxBigAddSmallIntoBackend, VecZnxBigAlloc, VecZnxBigAutomorphism,
        VecZnxBigAutomorphismAssign, VecZnxBigAutomorphismAssignTmpBytes, VecZnxBigFromSmallBackend, VecZnxBigNegate,
        VecZnxBigNegateAssign, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxBigSub, VecZnxBigSubAssign,
        VecZnxBigSubNegateAssign, VecZnxBigSubSmallABackend, VecZnxBigSubSmallAssign, VecZnxBigSubSmallBBackend,
        VecZnxBigSubSmallNegateAssign,
    },
    layouts::{
        Backend, DigestU64, FillUniform, HostBytesBackend, Module, NoiseInfos, ScratchOwned, VecZnx, VecZnxBig,
        VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxToBackendMut,
    },
    source::Source,
};

type VecZnxBigOwned<BE> = VecZnxBig<<BE as Backend>::OwnedBuf, BE>;

fn big_from_small_host<BE>(module: &Module<BE>, host: &VecZnx<impl crate::layouts::HostDataRef>) -> VecZnxBigOwned<BE>
where
    BE: Backend,
    Module<BE>: VecZnxBigAlloc<BE> + VecZnxBigFromSmallBackend<BE>,
{
    let cols = host.cols();
    let size = host.size();
    let uploaded = upload_vec_znx::<BE>(host);
    let mut res = module.vec_znx_big_alloc(cols, size);
    for j in 0..cols {
        module.vec_znx_big_from_small_backend(&mut res.to_backend_mut(), j, &vec_znx_backend_ref::<BE>(&uploaded), j);
    }
    res
}

fn normalize_big_to_host<BE>(
    module: &Module<BE>,
    base2k: usize,
    backend: &VecZnxBigOwned<BE>,
    scratch: &mut ScratchOwned<BE>,
) -> VecZnx<Vec<u8>>
where
    BE: Backend,
    Module<BE>: VecZnxBigNormalize<BE>,
{
    normalize_big_to_host_with_offset(module, base2k, 0, backend.shape().size(), backend, scratch)
}

fn normalize_big_to_host_with_offset<BE>(
    module: &Module<BE>,
    base2k: usize,
    res_offset: i64,
    res_size: usize,
    backend: &VecZnxBigOwned<BE>,
    scratch: &mut ScratchOwned<BE>,
) -> VecZnx<Vec<u8>>
where
    BE: Backend,
    Module<BE>: VecZnxBigNormalize<BE>,
{
    let shape = backend.shape();
    let mut res_backend = module.vec_znx_alloc(shape.cols(), res_size);
    for j in 0..shape.cols() {
        module.vec_znx_big_normalize(
            &mut <VecZnx<BE::OwnedBuf> as VecZnxToBackendMut<BE>>::to_backend_mut(&mut res_backend),
            base2k,
            res_offset,
            j,
            &backend.to_backend_ref(),
            base2k,
            j,
            &mut scratch.arena(),
        );
    }
    download_vec_znx::<BE>(&res_backend)
}

pub fn test_vec_znx_big_seed_add_normal_matches_source_wrapper<
    BR: crate::test_suite::TestBackend,
    BT: crate::test_suite::TestBackend,
>(
    params: &TestParams,
    _module_host: &Module<HostBytesBackend>,
    _module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BT>: VecZnxBigAddNormal<BT>
        + VecZnxBigAddNormalBackend<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    let size: usize = 5;
    let cols: usize = 2;
    let col_i: usize = 1;
    let noise_infos = NoiseInfos::new(2 * base2k, 3.2, 6.0 * 3.2).unwrap();
    let mut scratch = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    let mut seed_source = Source::new([2u8; 32]);
    let seed = seed_source.new_seed();
    let mut wrapper_source = Source::new([2u8; 32]);

    let mut wrapper: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, size);
    let mut backend: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, size);
    module_test.vec_znx_big_add_normal(base2k, &mut wrapper.to_backend_mut(), col_i, noise_infos, &mut wrapper_source);
    module_test.vec_znx_big_add_normal_backend(base2k, &mut backend.to_backend_mut(), col_i, noise_infos, seed);
    assert_eq!(
        normalize_big_to_host(module_test, base2k, &wrapper, &mut scratch),
        normalize_big_to_host(module_test, base2k, &backend, &mut scratch)
    );
}

pub fn test_vec_znx_big_add_into<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxBigAddInto<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmallBackend<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigAddInto<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmallBackend<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_digest = a.digest_u64();

        let a_ref = big_from_small_host(module_ref, &a);
        let a_test = big_from_small_host(module_test, &a);

        assert_eq!(a.digest_u64(), a_digest);

        for b_size in [1, 2, 3, 4] {
            let mut b: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, b_size);
            b.fill_uniform(base2k, &mut source);
            let b_digest = b.digest_u64();

            let b_ref = big_from_small_host(module_ref, &b);
            let b_test = big_from_small_host(module_test, &b);

            assert_eq!(b.digest_u64(), b_digest);

            for res_size in [1, 2, 3, 4] {
                let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
                let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_big_add_into(
                        &mut res_big_ref.to_backend_mut(),
                        i,
                        &a_ref.to_backend_ref(),
                        i,
                        &b_ref.to_backend_ref(),
                        i,
                    );
                    module_test.vec_znx_big_add_into(
                        &mut res_big_test.to_backend_mut(),
                        i,
                        &a_test.to_backend_ref(),
                        i,
                        &b_test.to_backend_ref(),
                        i,
                    );
                }

                let res_small_ref = normalize_big_to_host(module_ref, base2k, &res_big_ref, &mut scratch_ref);
                let res_small_test = normalize_big_to_host(module_test, base2k, &res_big_test, &mut scratch_test);

                assert_eq!(res_small_ref, res_small_test);
            }
        }
    }
}

pub fn test_vec_znx_big_add_assign<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxBigAddAssign<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmallBackend<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigAddAssign<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmallBackend<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);

        let a_ref = big_from_small_host(module_ref, &a);
        let a_test = big_from_small_host(module_test, &a);

        for res_size in [1, 2, 3, 4] {
            let mut res: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
            res.fill_uniform(base2k, &mut source);

            let mut res_big_ref = big_from_small_host(module_ref, &res);
            let mut res_big_test = big_from_small_host(module_test, &res);

            for i in 0..cols {
                module_ref.vec_znx_big_add_assign(&mut res_big_ref.to_backend_mut(), i, &a_ref.to_backend_ref(), i);
                module_test.vec_znx_big_add_assign(&mut res_big_test.to_backend_mut(), i, &a_test.to_backend_ref(), i);
            }

            let res_small_ref = normalize_big_to_host(module_ref, base2k, &res_big_ref, &mut scratch_ref);
            let res_small_test = normalize_big_to_host(module_test, base2k, &res_big_test, &mut scratch_test);

            assert_eq!(res_small_ref, res_small_test);
        }
    }
}

pub fn test_vec_znx_big_add_small_into<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxBigAddSmallIntoBackend<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmallBackend<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigAddSmallIntoBackend<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmallBackend<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);

        let a_ref = big_from_small_host(module_ref, &a);
        let a_test = big_from_small_host(module_test, &a);

        for b_size in [1, 2, 3, 4] {
            let mut b: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, b_size);
            b.fill_uniform(base2k, &mut source);
            let b_digest: u64 = b.digest_u64();
            let b_ref = upload_vec_znx::<BR>(&b);
            let b_test = upload_vec_znx::<BT>(&b);

            for res_size in [1, 2, 3, 4] {
                let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
                let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_big_add_small_into_backend(
                        &mut res_big_ref.to_backend_mut(),
                        i,
                        &a_ref.to_backend_ref(),
                        i,
                        &vec_znx_backend_ref::<BR>(&b_ref),
                        i,
                    );
                    module_test.vec_znx_big_add_small_into_backend(
                        &mut res_big_test.to_backend_mut(),
                        i,
                        &a_test.to_backend_ref(),
                        i,
                        &vec_znx_backend_ref::<BT>(&b_test),
                        i,
                    );
                }

                assert_eq!(b.digest_u64(), b_digest);
                let res_small_ref = normalize_big_to_host(module_ref, base2k, &res_big_ref, &mut scratch_ref);
                let res_small_test = normalize_big_to_host(module_test, base2k, &res_big_test, &mut scratch_test);

                assert_eq!(res_small_ref, res_small_test);
            }
        }
    }
}

pub fn test_vec_znx_big_add_small_assign<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxBigAddSmallAssign<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmallBackend<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigAddSmallAssign<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmallBackend<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);

        let a_digest: u64 = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let mut res: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
            res.fill_uniform(base2k, &mut source);

            let mut res_big_ref = big_from_small_host(module_ref, &res);
            let mut res_big_test = big_from_small_host(module_test, &res);

            for i in 0..cols {
                module_ref.vec_znx_big_add_small_assign(
                    &mut res_big_ref.to_backend_mut(),
                    i,
                    &vec_znx_backend_ref::<BR>(&a_ref),
                    i,
                );
                module_test.vec_znx_big_add_small_assign(
                    &mut res_big_test.to_backend_mut(),
                    i,
                    &vec_znx_backend_ref::<BT>(&a_test),
                    i,
                );
            }

            assert_eq!(a.digest_u64(), a_digest);

            let res_small_ref = normalize_big_to_host(module_ref, base2k, &res_big_ref, &mut scratch_ref);
            let res_small_test = normalize_big_to_host(module_test, base2k, &res_big_test, &mut scratch_test);

            assert_eq!(res_small_ref, res_small_test);
        }
    }
}

pub fn test_vec_znx_big_automorphism<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxBigAutomorphism<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmallBackend<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigAutomorphism<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmallBackend<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);

        let a_ref = big_from_small_host(module_ref, &a);
        let a_test = big_from_small_host(module_test, &a);

        for res_size in [1, 2, 3, 4] {
            for p in [-5, 5] {
                let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
                let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_big_automorphism(p, &mut res_big_ref.to_backend_mut(), i, &a_ref.to_backend_ref(), i);
                    module_test.vec_znx_big_automorphism(p, &mut res_big_test.to_backend_mut(), i, &a_test.to_backend_ref(), i);
                }

                let res_small_ref = normalize_big_to_host(module_ref, base2k, &res_big_ref, &mut scratch_ref);
                let res_small_test = normalize_big_to_host(module_test, base2k, &res_big_test, &mut scratch_test);

                assert_eq!(res_small_ref, res_small_test);
            }
        }
    }
}

pub fn test_vec_znx_big_automorphism_assign<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxBigAutomorphismAssign<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmallBackend<BR>
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigAutomorphismAssign<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmallBackend<BT>
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(
        module_ref.vec_znx_big_automorphism_assign_tmp_bytes() | module_ref.vec_znx_big_normalize_tmp_bytes(),
    );
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(
        module_test.vec_znx_big_automorphism_assign_tmp_bytes() | module_test.vec_znx_big_normalize_tmp_bytes(),
    );

    for res_size in [1, 2, 3, 4] {
        let mut res: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
        res.fill_uniform(base2k, &mut source);

        for p in [-5, 5] {
            let mut res_big_ref = big_from_small_host(module_ref, &res);
            let mut res_big_test = big_from_small_host(module_test, &res);

            for i in 0..cols {
                module_ref.vec_znx_big_automorphism_assign(p, &mut res_big_ref.to_backend_mut(), i, &mut scratch_ref.arena());
                module_test.vec_znx_big_automorphism_assign(p, &mut res_big_test.to_backend_mut(), i, &mut scratch_test.arena());
            }

            let res_small_ref = normalize_big_to_host(module_ref, base2k, &res_big_ref, &mut scratch_ref);
            let res_small_test = normalize_big_to_host(module_test, base2k, &res_big_test, &mut scratch_test);

            assert_eq!(res_small_ref, res_small_test);
        }
    }
}

pub fn test_vec_znx_big_negate<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxBigNegate<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmallBackend<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigNegate<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmallBackend<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);

        let a_ref = big_from_small_host(module_ref, &a);
        let a_test = big_from_small_host(module_test, &a);

        for res_size in [1, 2, 3, 4] {
            let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
            let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

            // Reference
            for i in 0..cols {
                module_ref.vec_znx_big_negate(&mut res_big_ref.to_backend_mut(), i, &a_ref.to_backend_ref(), i);
                module_test.vec_znx_big_negate(&mut res_big_test.to_backend_mut(), i, &a_test.to_backend_ref(), i);
            }

            let res_small_ref = normalize_big_to_host(module_ref, base2k, &res_big_ref, &mut scratch_ref);
            let res_small_test = normalize_big_to_host(module_test, base2k, &res_big_test, &mut scratch_test);

            assert_eq!(res_small_ref, res_small_test);
        }
    }
}

pub fn test_vec_znx_big_negate_assign<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxBigNegateAssign<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmallBackend<BR>
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigNegateAssign<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmallBackend<BT>
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for res_size in [1, 2, 3, 4] {
        let mut res: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
        res.fill_uniform(base2k, &mut source);

        let mut res_big_ref = big_from_small_host(module_ref, &res);
        let mut res_big_test = big_from_small_host(module_test, &res);

        for i in 0..cols {
            module_ref.vec_znx_big_negate_assign(&mut res_big_ref.to_backend_mut(), i);
            module_test.vec_znx_big_negate_assign(&mut res_big_test.to_backend_mut(), i);
        }

        let res_small_ref = normalize_big_to_host(module_ref, base2k, &res_big_ref, &mut scratch_ref);
        let res_small_test = normalize_big_to_host(module_test, base2k, &res_big_test, &mut scratch_test);

        assert_eq!(res_small_ref, res_small_test);
    }
}

pub fn test_vec_znx_big_normalize<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxBigAlloc<BR>
        + VecZnxBigFromSmallBackend<BR>
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigAlloc<BT>
        + VecZnxBigFromSmallBackend<BT>
        + VecZnxBigAutomorphismAssignTmpBytes
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(
        module_ref.vec_znx_big_automorphism_assign_tmp_bytes() | module_ref.vec_znx_big_normalize_tmp_bytes(),
    );
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(
        module_test.vec_znx_big_automorphism_assign_tmp_bytes() | module_test.vec_znx_big_normalize_tmp_bytes(),
    );

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(63, &mut source);

        let a_ref = big_from_small_host(module_ref, &a);
        let a_test = big_from_small_host(module_test, &a);

        for res_size in [1, 2, 3, 4] {
            for res_offset in -(base2k as i64)..=(base2k as i64) {
                let res_ref =
                    normalize_big_to_host_with_offset(module_ref, base2k, res_offset, res_size, &a_ref, &mut scratch_ref);
                let res_test =
                    normalize_big_to_host_with_offset(module_test, base2k, res_offset, res_size, &a_test, &mut scratch_test);

                assert_eq!(res_ref, res_test);
            }
        }
    }
}

pub fn test_vec_znx_big_sub<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxBigSub<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmallBackend<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigSub<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmallBackend<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);

        let a_ref = big_from_small_host(module_ref, &a);
        let a_test = big_from_small_host(module_test, &a);

        for b_size in [1, 2, 3, 4] {
            let mut b: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, b_size);
            b.fill_uniform(base2k, &mut source);

            let b_ref = big_from_small_host(module_ref, &b);
            let b_test = big_from_small_host(module_test, &b);

            for res_size in [1, 2, 3, 4] {
                let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
                let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_big_sub(
                        &mut res_big_ref.to_backend_mut(),
                        i,
                        &a_ref.to_backend_ref(),
                        i,
                        &b_ref.to_backend_ref(),
                        i,
                    );
                    module_test.vec_znx_big_sub(
                        &mut res_big_test.to_backend_mut(),
                        i,
                        &a_test.to_backend_ref(),
                        i,
                        &b_test.to_backend_ref(),
                        i,
                    );
                }

                let res_small_ref = normalize_big_to_host(module_ref, base2k, &res_big_ref, &mut scratch_ref);
                let res_small_test = normalize_big_to_host(module_test, base2k, &res_big_test, &mut scratch_test);

                assert_eq!(res_small_ref, res_small_test);
            }
        }
    }
}

pub fn test_vec_znx_big_sub_assign<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxBigSubAssign<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmallBackend<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigSubAssign<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmallBackend<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);

        let a_ref = big_from_small_host(module_ref, &a);
        let a_test = big_from_small_host(module_test, &a);

        for res_size in [1, 2, 3, 4] {
            let mut res: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
            res.fill_uniform(base2k, &mut source);

            let mut res_big_ref = big_from_small_host(module_ref, &res);
            let mut res_big_test = big_from_small_host(module_test, &res);

            for i in 0..cols {
                module_ref.vec_znx_big_sub_assign(&mut res_big_ref.to_backend_mut(), i, &a_ref.to_backend_ref(), i);
                module_test.vec_znx_big_sub_assign(&mut res_big_test.to_backend_mut(), i, &a_test.to_backend_ref(), i);
            }

            let res_small_ref = normalize_big_to_host(module_ref, base2k, &res_big_ref, &mut scratch_ref);
            let res_small_test = normalize_big_to_host(module_test, base2k, &res_big_test, &mut scratch_test);

            assert_eq!(res_small_ref, res_small_test);
        }
    }
}

pub fn test_vec_znx_big_sub_negate_assign<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxBigSubNegateAssign<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmallBackend<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigSubNegateAssign<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmallBackend<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);

        let a_ref = big_from_small_host(module_ref, &a);
        let a_test = big_from_small_host(module_test, &a);

        for res_size in [1, 2, 3, 4] {
            let mut res: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
            res.fill_uniform(base2k, &mut source);

            let mut res_big_ref = big_from_small_host(module_ref, &res);
            let mut res_big_test = big_from_small_host(module_test, &res);

            for i in 0..cols {
                module_ref.vec_znx_big_sub_negate_assign(&mut res_big_ref.to_backend_mut(), i, &a_ref.to_backend_ref(), i);
                module_test.vec_znx_big_sub_negate_assign(&mut res_big_test.to_backend_mut(), i, &a_test.to_backend_ref(), i);
            }

            let res_small_ref = normalize_big_to_host(module_ref, base2k, &res_big_ref, &mut scratch_ref);
            let res_small_test = normalize_big_to_host(module_test, base2k, &res_big_test, &mut scratch_test);

            assert_eq!(res_small_ref, res_small_test);
        }
    }
}

pub fn test_vec_znx_big_sub_small_a<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxBigSubSmallABackend<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmallBackend<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigSubSmallABackend<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmallBackend<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);

        let a_ref = big_from_small_host(module_ref, &a);
        let a_test = big_from_small_host(module_test, &a);

        for b_size in [1, 2, 3, 4] {
            let mut b: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, b_size);
            b.fill_uniform(base2k, &mut source);
            let b_ref = upload_vec_znx::<BR>(&b);
            let b_test = upload_vec_znx::<BT>(&b);

            for res_size in [1, 2, 3, 4] {
                let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
                let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_big_sub_small_a_backend(
                        &mut res_big_ref.to_backend_mut(),
                        i,
                        &vec_znx_backend_ref::<BR>(&b_ref),
                        i,
                        &a_ref.to_backend_ref(),
                        i,
                    );
                    module_test.vec_znx_big_sub_small_a_backend(
                        &mut res_big_test.to_backend_mut(),
                        i,
                        &vec_znx_backend_ref::<BT>(&b_test),
                        i,
                        &a_test.to_backend_ref(),
                        i,
                    );
                }

                let res_small_ref = normalize_big_to_host(module_ref, base2k, &res_big_ref, &mut scratch_ref);
                let res_small_test = normalize_big_to_host(module_test, base2k, &res_big_test, &mut scratch_test);

                assert_eq!(res_small_ref, res_small_test);
            }
        }
    }
}

pub fn test_vec_znx_big_sub_small_b<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxBigSubSmallBBackend<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmallBackend<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigSubSmallBBackend<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmallBackend<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);

        let a_ref = big_from_small_host(module_ref, &a);
        let a_test = big_from_small_host(module_test, &a);

        for b_size in [1, 2, 3, 4] {
            let mut b: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, b_size);
            b.fill_uniform(base2k, &mut source);
            let b_ref = upload_vec_znx::<BR>(&b);
            let b_test = upload_vec_znx::<BT>(&b);

            for res_size in [1, 2, 3, 4] {
                let mut res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_big_alloc(cols, res_size);
                let mut res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_big_alloc(cols, res_size);

                // Reference
                for i in 0..cols {
                    module_ref.vec_znx_big_sub_small_b_backend(
                        &mut res_big_ref.to_backend_mut(),
                        i,
                        &a_ref.to_backend_ref(),
                        i,
                        &vec_znx_backend_ref::<BR>(&b_ref),
                        i,
                    );
                    module_test.vec_znx_big_sub_small_b_backend(
                        &mut res_big_test.to_backend_mut(),
                        i,
                        &a_test.to_backend_ref(),
                        i,
                        &vec_znx_backend_ref::<BT>(&b_test),
                        i,
                    );
                }

                let res_small_ref = normalize_big_to_host(module_ref, base2k, &res_big_ref, &mut scratch_ref);
                let res_small_test = normalize_big_to_host(module_test, base2k, &res_big_test, &mut scratch_test);

                assert_eq!(res_small_ref, res_small_test);
            }
        }
    }
}

pub fn test_vec_znx_big_sub_small_a_assign<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxBigSubSmallAssign<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmallBackend<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigSubSmallAssign<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmallBackend<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);

        let a_digest: u64 = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let mut res: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
            res.fill_uniform(base2k, &mut source);

            let mut res_big_ref = big_from_small_host(module_ref, &res);
            let mut res_big_test = big_from_small_host(module_test, &res);

            for i in 0..cols {
                module_ref.vec_znx_big_sub_small_assign(
                    &mut res_big_ref.to_backend_mut(),
                    i,
                    &vec_znx_backend_ref::<BR>(&a_ref),
                    i,
                );
                module_test.vec_znx_big_sub_small_assign(
                    &mut res_big_test.to_backend_mut(),
                    i,
                    &vec_znx_backend_ref::<BT>(&a_test),
                    i,
                );
            }

            assert_eq!(a.digest_u64(), a_digest);

            let res_small_ref = normalize_big_to_host(module_ref, base2k, &res_big_ref, &mut scratch_ref);
            let res_small_test = normalize_big_to_host(module_test, base2k, &res_big_test, &mut scratch_test);

            assert_eq!(res_small_ref, res_small_test);
        }
    }
}

pub fn test_vec_znx_big_sub_small_b_assign<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxBigSubSmallNegateAssign<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigFromSmallBackend<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxBigSubSmallNegateAssign<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigFromSmallBackend<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let mut source: Source = Source::new([0u8; 32]);
    let cols: usize = 2;

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);

        let a_digest: u64 = a.digest_u64();
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            for res_offset in -(base2k as i64)..=(base2k as i64) {
                let mut res: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
                res.fill_uniform(base2k, &mut source);

                let mut res_big_ref = big_from_small_host(module_ref, &res);
                let mut res_big_test = big_from_small_host(module_test, &res);

                for i in 0..cols {
                    module_ref.vec_znx_big_sub_small_negate_assign(
                        &mut res_big_ref.to_backend_mut(),
                        i,
                        &vec_znx_backend_ref::<BR>(&a_ref),
                        i,
                    );
                    module_test.vec_znx_big_sub_small_negate_assign(
                        &mut res_big_test.to_backend_mut(),
                        i,
                        &vec_znx_backend_ref::<BT>(&a_test),
                        i,
                    );
                }

                assert_eq!(a.digest_u64(), a_digest);

                let res_small_ref =
                    normalize_big_to_host_with_offset(module_ref, base2k, res_offset, res_size, &res_big_ref, &mut scratch_ref);
                let res_small_test = normalize_big_to_host_with_offset(
                    module_test,
                    base2k,
                    res_offset,
                    res_size,
                    &res_big_test,
                    &mut scratch_test,
                );

                assert_eq!(res_small_ref, res_small_test);
            }
        }
    }
}
