use super::{TestParams, download_vec_znx, upload_vec_znx};

use crate::{
    api::{
        ScratchOwnedAlloc, VecZnxBigAlloc, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAddAssign, VecZnxDftAddInto,
        VecZnxDftAlloc, VecZnxDftApply, VecZnxDftCopy, VecZnxDftSub, VecZnxDftSubAssign, VecZnxDftSubNegateAssign,
        VecZnxIdftApply, VecZnxIdftApplyTmpA, VecZnxIdftApplyTmpBytes,
    },
    layouts::{
        Backend, FillUniform, HostBytesBackend, Module, ScratchOwned, VecZnx, VecZnxBig, VecZnxBigToBackendMut,
        VecZnxBigToBackendRef, VecZnxDft, VecZnxDftToBackendMut, VecZnxDftToBackendRef, VecZnxToBackendMut, VecZnxToBackendRef,
    },
    source::Source,
};

type VecZnxDftOwned<BE> = VecZnxDft<<BE as Backend>::OwnedBuf, BE>;
type VecZnxBigOwned<BE> = VecZnxBig<<BE as Backend>::OwnedBuf, BE>;

fn idft_into_alloc<BE>(module: &Module<BE>, a: &mut VecZnxDftOwned<BE>) -> VecZnxBigOwned<BE>
where
    BE: Backend,
    Module<BE>: VecZnxBigAlloc<BE> + VecZnxIdftApplyTmpA<BE>,
{
    let cols = a.cols();
    let size = a.size();
    let mut res = module.vec_znx_big_alloc(cols, size);
    for j in 0..cols {
        let mut res_backend = res.to_backend_mut();
        let mut a_backend = a.to_backend_mut();
        module.vec_znx_idft_apply_tmpa(&mut res_backend, j, &mut a_backend, j);
    }
    res
}

fn dft_of_uploaded_vec_znx<BE>(
    module: &Module<BE>,
    host: &VecZnx<impl crate::layouts::HostDataRef>,
    steps: usize,
    offset: usize,
) -> VecZnxDftOwned<BE>
where
    BE: Backend,
    Module<BE>: VecZnxDftAlloc<BE> + VecZnxDftApply<BE>,
{
    let cols = host.cols();
    let size = host.size();
    let backend = upload_vec_znx::<BE>(host);
    let mut out = module.vec_znx_dft_alloc(cols, size);
    for j in 0..cols {
        module.vec_znx_dft_apply(
            steps,
            offset,
            &mut out.to_backend_mut(),
            j,
            &<VecZnx<BE::OwnedBuf> as VecZnxToBackendRef<BE>>::to_backend_ref(&backend),
            j,
        );
    }
    out
}

fn normalize_big_to_host<BE>(
    module: &Module<BE>,
    base2k: usize,
    big: &VecZnxBigOwned<BE>,
    scratch: &mut ScratchOwned<BE>,
) -> VecZnx<Vec<u8>>
where
    BE: Backend,
    Module<BE>: VecZnxBigNormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let mut backend = module.vec_znx_alloc(big.cols(), big.size());
    for j in 0..big.cols() {
        module.vec_znx_big_normalize(
            &mut <VecZnx<BE::OwnedBuf> as VecZnxToBackendMut<BE>>::to_backend_mut(&mut backend),
            base2k,
            0,
            j,
            &big.to_backend_ref(),
            base2k,
            j,
            &mut scratch.arena(),
        );
    }
    download_vec_znx::<BE>(&backend)
}

fn idft_tmpa_to_host<BE>(
    module: &Module<BE>,
    base2k: usize,
    dft: &mut VecZnxDftOwned<BE>,
    scratch: &mut ScratchOwned<BE>,
) -> VecZnx<Vec<u8>>
where
    BE: Backend,
    Module<BE>: VecZnxBigAlloc<BE> + VecZnxIdftApplyTmpA<BE> + VecZnxBigNormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let big = idft_into_alloc(module, dft);
    normalize_big_to_host(module, base2k, &big, scratch)
}

fn idft_apply_to_host<BE>(
    module: &Module<BE>,
    base2k: usize,
    dft: &VecZnxDftOwned<BE>,
    res_size: usize,
    scratch: &mut ScratchOwned<BE>,
) -> VecZnx<Vec<u8>>
where
    BE: Backend,
    Module<BE>: VecZnxBigAlloc<BE> + VecZnxIdftApply<BE> + VecZnxBigNormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let mut big = module.vec_znx_big_alloc(dft.cols(), res_size);
    for j in 0..dft.cols() {
        module.vec_znx_idft_apply(&mut big.to_backend_mut(), j, &dft.to_backend_ref(), j, &mut scratch.arena());
    }
    normalize_big_to_host(module, base2k, &big, scratch)
}

pub fn test_vec_znx_dft_add_into<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxDftAddInto<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxDftApply<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxIdftApplyTmpA<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxDftAddInto<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxDftApply<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxIdftApplyTmpA<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let cols: usize = 2;
    let mut source = Source::new([0u8; 32]);
    let mut scratch_ref = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_dft_ref = dft_of_uploaded_vec_znx(module_ref, &a, 1, 0);
        let a_dft_test = dft_of_uploaded_vec_znx(module_test, &a, 1, 0);

        for b_size in [1, 2, 3, 4] {
            let mut b = module_host.vec_znx_alloc(cols, b_size);
            b.fill_uniform(base2k, &mut source);
            let b_dft_ref = dft_of_uploaded_vec_znx(module_ref, &b, 1, 0);
            let b_dft_test = dft_of_uploaded_vec_znx(module_test, &b, 1, 0);

            for res_size in [1, 2, 3, 4] {
                let res_init = module_host.vec_znx_alloc(cols, res_size);
                let mut res_dft_ref = dft_of_uploaded_vec_znx(module_ref, &res_init, 1, 0);
                let mut res_dft_test = dft_of_uploaded_vec_znx(module_test, &res_init, 1, 0);

                for i in 0..cols {
                    module_ref.vec_znx_dft_add_into(
                        &mut res_dft_ref.to_backend_mut(),
                        i,
                        &a_dft_ref.to_backend_ref(),
                        i,
                        &b_dft_ref.to_backend_ref(),
                        i,
                    );
                    module_test.vec_znx_dft_add_into(
                        &mut res_dft_test.to_backend_mut(),
                        i,
                        &a_dft_test.to_backend_ref(),
                        i,
                        &b_dft_test.to_backend_ref(),
                        i,
                    );
                }

                let res_ref = idft_tmpa_to_host(module_ref, base2k, &mut res_dft_ref, &mut scratch_ref);
                let res_test = idft_tmpa_to_host(module_test, base2k, &mut res_dft_test, &mut scratch_test);
                assert_eq!(res_ref, res_test);
            }
        }
    }
}

pub fn test_vec_znx_dft_add_assign<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxDftAddAssign<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxDftApply<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxIdftApplyTmpA<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxDftAddAssign<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxDftApply<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxIdftApplyTmpA<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let _n = module_ref.n();
    let cols = 2;
    let mut source = Source::new([0u8; 32]);
    let mut scratch_ref = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_dft_ref = dft_of_uploaded_vec_znx(module_ref, &a, 1, 0);
        let a_dft_test = dft_of_uploaded_vec_znx(module_test, &a, 1, 0);

        for _res_size in [1, 2, 3, 4] {
            let mut res = module_host.vec_znx_alloc(cols, a_size);
            res.fill_uniform(base2k, &mut source);
            let mut res_dft_ref = dft_of_uploaded_vec_znx(module_ref, &res, 1, 0);
            let mut res_dft_test = dft_of_uploaded_vec_znx(module_test, &res, 1, 0);

            for i in 0..cols {
                module_ref.vec_znx_dft_add_assign(&mut res_dft_ref.to_backend_mut(), i, &a_dft_ref.to_backend_ref(), i);
                module_test.vec_znx_dft_add_assign(&mut res_dft_test.to_backend_mut(), i, &a_dft_test.to_backend_ref(), i);
            }

            let res_ref = idft_tmpa_to_host(module_ref, base2k, &mut res_dft_ref, &mut scratch_ref);
            let res_test = idft_tmpa_to_host(module_test, base2k, &mut res_dft_test, &mut scratch_test);
            assert_eq!(res_ref, res_test);
        }
    }
}

pub fn test_vec_znx_copy<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxDftCopy<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxDftApply<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxIdftApplyTmpA<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxDftCopy<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxDftApply<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxIdftApplyTmpA<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let _n = module_ref.n();
    let cols = 2;
    let mut source = Source::new([0u8; 32]);
    let mut scratch_ref = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 6, 11] {
        let mut a = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_dft_ref = dft_of_uploaded_vec_znx(module_ref, &a, 1, 0);
        let a_dft_test = dft_of_uploaded_vec_znx(module_test, &a, 1, 0);

        for res_size in [1, 2, 6, 11] {
            for params in [[1, 0], [1, 1], [1, 2], [2, 2]] {
                let steps = params[0];
                let offset = params[1];
                let res_init = module_host.vec_znx_alloc(cols, res_size);
                let mut res_dft_ref = dft_of_uploaded_vec_znx(module_ref, &res_init, 1, 0);
                let mut res_dft_test = dft_of_uploaded_vec_znx(module_test, &res_init, 1, 0);

                for i in 0..cols {
                    module_ref.vec_znx_dft_copy(
                        steps,
                        offset,
                        &mut res_dft_ref.to_backend_mut(),
                        i,
                        &a_dft_ref.to_backend_ref(),
                        i,
                    );
                    module_test.vec_znx_dft_copy(
                        steps,
                        offset,
                        &mut res_dft_test.to_backend_mut(),
                        i,
                        &a_dft_test.to_backend_ref(),
                        i,
                    );
                }

                let res_ref = idft_tmpa_to_host(module_ref, base2k, &mut res_dft_ref, &mut scratch_ref);
                let res_test = idft_tmpa_to_host(module_test, base2k, &mut res_dft_test, &mut scratch_test);
                assert_eq!(res_ref, res_test);
            }
        }
    }
}

pub fn test_vec_znx_idft_apply<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxDftApply<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApply<BR>,
    Module<BT>: VecZnxDftApply<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApply<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let _n = module_ref.n();
    let cols = 2;
    let mut source = Source::new([0u8; 32]);
    let mut scratch_ref = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);

        for res_size in [1, 2, 3, 4] {
            for params in [[1, 0], [1, 1], [1, 2], [2, 2]] {
                let res_dft_ref = dft_of_uploaded_vec_znx(module_ref, &a, params[0], params[1]);
                let res_dft_test = dft_of_uploaded_vec_znx(module_test, &a, params[0], params[1]);
                let res_ref = idft_apply_to_host(module_ref, base2k, &res_dft_ref, res_size, &mut scratch_ref);
                let res_test = idft_apply_to_host(module_test, base2k, &res_dft_test, res_size, &mut scratch_test);
                assert_eq!(res_ref, res_test);
            }
        }
    }
}

pub fn test_vec_znx_idft_apply_tmpa<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxDftApply<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApplyTmpA<BR>,
    Module<BT>: VecZnxDftApply<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApplyTmpA<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let _n = module_ref.n();
    let cols = 2;
    let mut source = Source::new([0u8; 32]);
    let mut scratch_ref = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);

        for _res_size in [1, 2, 3, 4] {
            for params in [[1, 0], [1, 1], [1, 2], [2, 2]] {
                let mut res_dft_ref = dft_of_uploaded_vec_znx(module_ref, &a, params[0], params[1]);
                let mut res_dft_test = dft_of_uploaded_vec_znx(module_test, &a, params[0], params[1]);
                let res_ref = idft_tmpa_to_host(module_ref, base2k, &mut res_dft_ref, &mut scratch_ref);
                let res_test = idft_tmpa_to_host(module_test, base2k, &mut res_dft_test, &mut scratch_test);
                assert_eq!(res_ref, res_test);
            }
        }
    }
}

pub fn test_vec_znx_idft_apply_alloc<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxDftApply<BR>
        + VecZnxIdftApplyTmpBytes
        + VecZnxDftAlloc<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxBigAlloc<BR>
        + VecZnxIdftApplyTmpA<BR>,
    Module<BT>: VecZnxDftApply<BT>
        + VecZnxIdftApplyTmpBytes
        + VecZnxDftAlloc<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxBigAlloc<BT>
        + VecZnxIdftApplyTmpA<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let _n = module_ref.n();
    let cols = 2;
    let mut source = Source::new([0u8; 32]);
    let mut scratch_ref =
        ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes() | module_ref.vec_znx_idft_apply_tmp_bytes());
    let mut scratch_test =
        ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes() | module_test.vec_znx_idft_apply_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);

        for _res_size in [1, 2, 3, 4] {
            for params in [[1, 0], [1, 1], [1, 2], [2, 2]] {
                let mut res_dft_ref = dft_of_uploaded_vec_znx(module_ref, &a, params[0], params[1]);
                let mut res_dft_test = dft_of_uploaded_vec_znx(module_test, &a, params[0], params[1]);
                let res_ref = idft_tmpa_to_host(module_ref, base2k, &mut res_dft_ref, &mut scratch_ref);
                let res_test = idft_tmpa_to_host(module_test, base2k, &mut res_dft_test, &mut scratch_test);
                assert_eq!(res_ref, res_test);
            }
        }
    }
}

pub fn test_vec_znx_dft_sub<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxDftSub<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxDftApply<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxIdftApplyTmpA<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxDftSub<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxDftApply<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxIdftApplyTmpA<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let _n = module_ref.n();
    let cols = 2;
    let mut source = Source::new([0u8; 32]);
    let mut scratch_ref = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_dft_ref = dft_of_uploaded_vec_znx(module_ref, &a, 1, 0);
        let a_dft_test = dft_of_uploaded_vec_znx(module_test, &a, 1, 0);

        for b_size in [1, 2, 3, 4] {
            let mut b = module_host.vec_znx_alloc(cols, b_size);
            b.fill_uniform(base2k, &mut source);
            let b_dft_ref = dft_of_uploaded_vec_znx(module_ref, &b, 1, 0);
            let b_dft_test = dft_of_uploaded_vec_znx(module_test, &b, 1, 0);

            for res_size in [1, 2, 3, 4] {
                let res_init = module_host.vec_znx_alloc(cols, res_size);
                let mut res_dft_ref = dft_of_uploaded_vec_znx(module_ref, &res_init, 1, 0);
                let mut res_dft_test = dft_of_uploaded_vec_znx(module_test, &res_init, 1, 0);

                for i in 0..cols {
                    module_ref.vec_znx_dft_sub(
                        &mut res_dft_ref.to_backend_mut(),
                        i,
                        &a_dft_ref.to_backend_ref(),
                        i,
                        &b_dft_ref.to_backend_ref(),
                        i,
                    );
                    module_test.vec_znx_dft_sub(
                        &mut res_dft_test.to_backend_mut(),
                        i,
                        &a_dft_test.to_backend_ref(),
                        i,
                        &b_dft_test.to_backend_ref(),
                        i,
                    );
                }

                let res_ref = idft_tmpa_to_host(module_ref, base2k, &mut res_dft_ref, &mut scratch_ref);
                let res_test = idft_tmpa_to_host(module_test, base2k, &mut res_dft_test, &mut scratch_test);
                assert_eq!(res_ref, res_test);
            }
        }
    }
}

pub fn test_vec_znx_dft_sub_assign<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxDftSubAssign<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxDftApply<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxIdftApplyTmpA<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxDftSubAssign<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxDftApply<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxIdftApplyTmpA<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let _n = module_ref.n();
    let cols = 2;
    let mut source = Source::new([0u8; 32]);
    let mut scratch_ref = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_dft_ref = dft_of_uploaded_vec_znx(module_ref, &a, 1, 0);
        let a_dft_test = dft_of_uploaded_vec_znx(module_test, &a, 1, 0);

        for _res_size in [1, 2, 3, 4] {
            let mut res = module_host.vec_znx_alloc(cols, a_size);
            res.fill_uniform(base2k, &mut source);
            let mut res_dft_ref = dft_of_uploaded_vec_znx(module_ref, &res, 1, 0);
            let mut res_dft_test = dft_of_uploaded_vec_znx(module_test, &res, 1, 0);

            for i in 0..cols {
                module_ref.vec_znx_dft_sub_assign(&mut res_dft_ref.to_backend_mut(), i, &a_dft_ref.to_backend_ref(), i);
                module_test.vec_znx_dft_sub_assign(&mut res_dft_test.to_backend_mut(), i, &a_dft_test.to_backend_ref(), i);
            }

            let res_ref = idft_tmpa_to_host(module_ref, base2k, &mut res_dft_ref, &mut scratch_ref);
            let res_test = idft_tmpa_to_host(module_test, base2k, &mut res_dft_test, &mut scratch_test);
            assert_eq!(res_ref, res_test);
        }
    }
}

pub fn test_vec_znx_dft_sub_negate_assign<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxDftSubNegateAssign<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxDftApply<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxIdftApplyTmpA<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxDftSubNegateAssign<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxDftApply<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxIdftApplyTmpA<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let _n = module_ref.n();
    let cols = 2;
    let mut source = Source::new([0u8; 32]);
    let mut scratch_ref = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    for a_size in [1, 2, 3, 4] {
        let mut a = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_dft_ref = dft_of_uploaded_vec_znx(module_ref, &a, 1, 0);
        let a_dft_test = dft_of_uploaded_vec_znx(module_test, &a, 1, 0);

        for _res_size in [1, 2, 3, 4] {
            let mut res = module_host.vec_znx_alloc(cols, a_size);
            res.fill_uniform(base2k, &mut source);
            let mut res_dft_ref = dft_of_uploaded_vec_znx(module_ref, &res, 1, 0);
            let mut res_dft_test = dft_of_uploaded_vec_znx(module_test, &res, 1, 0);

            for i in 0..cols {
                module_ref.vec_znx_dft_sub_negate_assign(&mut res_dft_ref.to_backend_mut(), i, &a_dft_ref.to_backend_ref(), i);
                module_test.vec_znx_dft_sub_negate_assign(&mut res_dft_test.to_backend_mut(), i, &a_dft_test.to_backend_ref(), i);
            }

            let res_ref = idft_tmpa_to_host(module_ref, base2k, &mut res_dft_ref, &mut scratch_ref);
            let res_test = idft_tmpa_to_host(module_test, base2k, &mut res_dft_test, &mut scratch_test);
            assert_eq!(res_ref, res_test);
        }
    }
}
