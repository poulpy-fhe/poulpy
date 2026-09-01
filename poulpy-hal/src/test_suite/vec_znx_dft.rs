use super::{TestParams, download_vec_znx, upload_vec_znx, vec_znx_backend_mut, vec_znx_backend_ref};
use crate::layouts::VecZnxBigToBackendMut;
use crate::layouts::VecZnxBigToBackendRef;
use crate::layouts::VecZnxDftToBackendMut;
use crate::layouts::VecZnxDftToBackendRef;

use crate::{
    api::{
        ScratchOwnedAlloc, VecZnxAutomorphismBackend, VecZnxBigAddSmallAssign, VecZnxBigAlloc, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxDftAddAssign, VecZnxDftAddInto, VecZnxDftAlloc, VecZnxDftApply, VecZnxDftAutomorphism,
        VecZnxDftAutomorphismPlan, VecZnxDftCopy, VecZnxDftSub, VecZnxDftSubAssign, VecZnxDftSubNegateAssign, VecZnxIdftApply,
        VecZnxIdftApplyTmpA, VecZnxIdftApplyTmpBytes, VecZnxIdftNormalizeConsume, VecZnxIdftNormalizeConsumeTmpBytes,
    },
    layouts::{FillUniform, HostBytesBackend, Module, ScratchOwned, VecZnx, VecZnxOwned, VecZnxToBackendMut, VecZnxToBackendRef},
    source::Source,
};

use crate::layouts::VecZnxBigOwned;
use crate::layouts::VecZnxDftOwned;

fn idft_into_alloc<BE>(module: &Module<BE>, a: &mut VecZnxDftOwned<BE>) -> VecZnxBigOwned<BE>
where
    BE: crate::test_suite::TestBackend,
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

pub(crate) fn dft_of_uploaded_vec_znx<BE>(
    module: &Module<BE>,
    host: &VecZnx<impl crate::layouts::HostDataRef, i64>,
    steps: usize,
    offset: usize,
) -> VecZnxDftOwned<BE>
where
    BE: crate::test_suite::TestBackend,
    Module<BE>: VecZnxDftAlloc<BE> + VecZnxDftApply<BE>,
{
    let cols = host.cols();
    let size = host.size();
    let backend = upload_vec_znx::<BE, _>(host);
    let mut out = module.vec_znx_dft_alloc(cols, size);
    for j in 0..cols {
        module.vec_znx_dft_apply(
            steps,
            offset,
            &mut out.to_backend_mut(),
            j,
            &<VecZnx<BE::OwnedBuf, BE::ZnxWord> as VecZnxToBackendRef<BE>>::to_backend_ref(&backend),
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
) -> VecZnxOwned<BE::ZnxWord>
where
    BE: crate::test_suite::TestBackend,
    Module<BE>: VecZnxBigNormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let mut backend = module.vec_znx_alloc(big.cols(), big.size());
    for j in 0..big.cols() {
        module.vec_znx_big_normalize(
            &mut <VecZnx<BE::OwnedBuf, BE::ZnxWord> as VecZnxToBackendMut<BE>>::to_backend_mut(&mut backend),
            base2k,
            0,
            j,
            &big.to_backend_ref(),
            base2k,
            j,
            &mut scratch.arena(),
        );
    }
    download_vec_znx::<BE, _>(&backend)
}

fn idft_tmpa_to_host<BE>(
    module: &Module<BE>,
    base2k: usize,
    dft: &mut VecZnxDftOwned<BE>,
    scratch: &mut ScratchOwned<BE>,
) -> VecZnxOwned<BE::ZnxWord>
where
    BE: crate::test_suite::TestBackend,
    Module<BE>: VecZnxBigAlloc<BE> + VecZnxIdftApplyTmpA<BE> + VecZnxBigNormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let big = idft_into_alloc(module, dft);
    normalize_big_to_host(module, base2k, &big, scratch)
}

pub(crate) fn idft_apply_to_host<BE>(
    module: &Module<BE>,
    base2k: usize,
    dft: &VecZnxDftOwned<BE>,
    res_size: usize,
    scratch: &mut ScratchOwned<BE>,
) -> VecZnxOwned<BE::ZnxWord>
where
    BE: crate::test_suite::TestBackend,
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

/// Runs the contract check `IDFT(DFT_aut(DFT(a))) == coeff_aut(a)` on a
/// single backend, for a list of automorphism exponents. Used by
/// [`test_vec_znx_dft_automorphism`] which exercises both backends.
fn contract_check_one_backend<BE>(
    base2k: usize,
    module_host: &Module<HostBytesBackend>,
    module: &Module<BE>,
    scratch: &mut ScratchOwned<BE>,
    cols: usize,
    p_values: &[i64],
) where
    BE: crate::test_suite::TestBackend,
    Module<BE>: VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftAutomorphism<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxAutomorphismBackend<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let mut source = Source::new([0u8; 32]);

    for size in [1, 2, 3, 4] {
        let mut a = module_host.vec_znx_alloc(cols, size);
        a.fill_uniform(base2k, &mut source);

        for &p in p_values {
            // Pipeline A: DFT → automorphism with plan → IDFT → normalize.
            let mut a_dft = dft_of_uploaded_vec_znx(module, &a, 1, 0);
            let mut res_dft = module.vec_znx_dft_alloc(cols, size);
            let plan = module.vec_znx_dft_automorphism_plan(p);
            for j in 0..cols {
                module.vec_znx_dft_automorphism_with_plan(&plan, &mut res_dft.to_backend_mut(), j, &a_dft.to_backend_ref(), j);
            }
            let res_dft_normalized = idft_tmpa_to_host(module, base2k, &mut res_dft, scratch);
            // a_dft is consumed by the pipeline; discard it.
            let _ = idft_tmpa_to_host(module, base2k, &mut a_dft, scratch);

            // Pipeline B: coefficient-domain automorphism on the same backend.
            let a_backend = upload_vec_znx::<BE, _>(&a);
            let res_coeff_backend_host = module_host.vec_znx_alloc(cols, size);
            let mut res_coeff_backend = upload_vec_znx::<BE, _>(&res_coeff_backend_host);
            for j in 0..cols {
                module.vec_znx_automorphism_backend(
                    p,
                    &mut vec_znx_backend_mut::<BE, _>(&mut res_coeff_backend),
                    j,
                    &vec_znx_backend_ref::<BE, _>(&a_backend),
                    j,
                );
            }
            let res_coeff = download_vec_znx::<BE, _>(&res_coeff_backend);

            assert_eq!(
                res_dft_normalized, res_coeff,
                "DFT-domain automorphism != coefficient-domain automorphism for p={p}, size={size}"
            );
        }
    }
}

/// Verifies that for every odd `p`, `IDFT(VecZnxDftAutomorphism(p, DFT(a)))`
/// matches `VecZnxAutomorphism(p, a)` after normalization. Runs the contract
/// on both `module_ref` and `module_test`.
pub fn test_vec_znx_dft_automorphism<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxDftAlloc<BR>
        + VecZnxDftApply<BR>
        + VecZnxDftAutomorphism<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxIdftApplyTmpA<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxAutomorphismBackend<BR>,
    Module<BT>: VecZnxDftAlloc<BT>
        + VecZnxDftApply<BT>
        + VecZnxDftAutomorphism<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxIdftApplyTmpA<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxAutomorphismBackend<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let cols = 2;
    let mut scratch_ref = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    // Cover both residue classes mod 4 to exercise the conj/no-conj arms
    // of the FFT64 plan and a range of orbits under odd-p multiplication.
    let p_values: &[i64] = &[1, 5, 9, 13, 3, 7, 11, 15, -1, -5];

    contract_check_one_backend::<BR>(base2k, module_host, module_ref, &mut scratch_ref, cols, p_values);
    contract_check_one_backend::<BT>(base2k, module_host, module_test, &mut scratch_test, cols, p_values);
}

/// Runs the fused-accumulation check `automorphism_add(res, a) ==
/// add_assign(res, automorphism(a))` on a single backend, including a `res`
/// wider than `a` (limbs beyond `a.size()` must stay untouched).
fn automorphism_add_check_one_backend<BE>(
    base2k: usize,
    module_host: &Module<HostBytesBackend>,
    module: &Module<BE>,
    scratch: &mut ScratchOwned<BE>,
    cols: usize,
    p_values: &[i64],
) where
    BE: crate::test_suite::TestBackend,
    Module<BE>: VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftAutomorphism<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let mut source = Source::new([1u8; 32]);

    for (a_size, res_size) in [(1, 1), (3, 3), (3, 4), (4, 4)] {
        let mut a = module_host.vec_znx_alloc(cols, a_size);
        let mut seed = module_host.vec_znx_alloc(cols, res_size);
        a.fill_uniform(base2k, &mut source);
        seed.fill_uniform(base2k, &mut source);

        for &p in p_values {
            let plan = module.vec_znx_dft_automorphism_plan(p);
            let a_dft = dft_of_uploaded_vec_znx(module, &a, 1, 0);

            // Fused: res += automorphism(a) in one pass.
            let mut res_have = dft_of_uploaded_vec_znx(module, &seed, 1, 0);
            for j in 0..cols {
                module.vec_znx_dft_automorphism_add_with_plan(
                    &plan,
                    &mut res_have.to_backend_mut(),
                    j,
                    &a_dft.to_backend_ref(),
                    j,
                );
            }

            // Composition: automorphism into a temporary, then DFT-domain add.
            let mut res_want = dft_of_uploaded_vec_znx(module, &seed, 1, 0);
            let mut rot = module.vec_znx_dft_alloc(cols, a_size);
            for j in 0..cols {
                module.vec_znx_dft_automorphism_with_plan(&plan, &mut rot.to_backend_mut(), j, &a_dft.to_backend_ref(), j);
                module.vec_znx_dft_add_assign(&mut res_want.to_backend_mut(), j, &rot.to_backend_ref(), j);
            }

            let have = idft_tmpa_to_host(module, base2k, &mut res_have, scratch);
            let want = idft_tmpa_to_host(module, base2k, &mut res_want, scratch);
            assert_eq!(
                want, have,
                "automorphism_add != automorphism + add for p={p}, a_size={a_size}, res_size={res_size}"
            );
        }
    }
}

/// Verifies `vec_znx_dft_automorphism_add_with_plan` against the composition
/// of `vec_znx_dft_automorphism_with_plan` and `vec_znx_dft_add_assign`, on
/// both `module_ref` and `module_test`.
pub fn test_vec_znx_dft_automorphism_add<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxDftAlloc<BR>
        + VecZnxDftApply<BR>
        + VecZnxDftAutomorphism<BR>
        + VecZnxDftAddAssign<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxIdftApplyTmpA<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxDftAlloc<BT>
        + VecZnxDftApply<BT>
        + VecZnxDftAutomorphism<BT>
        + VecZnxDftAddAssign<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxIdftApplyTmpA<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let cols = 2;
    let mut scratch_ref = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    let p_values: &[i64] = &[1, 5, 3, 7, -1, -5];

    automorphism_add_check_one_backend::<BR>(base2k, module_host, module_ref, &mut scratch_ref, cols, p_values);
    automorphism_add_check_one_backend::<BT>(base2k, module_host, module_test, &mut scratch_test, cols, p_values);
}

/// Runs the fused check `idft_normalize_consume(res, a, addend) ==
/// normalize(idft(a) + addend)` on a single backend, over size and base2k
/// combinations and both addend arms.
fn idft_normalize_consume_check_one_backend<BE>(
    base2k: usize,
    module_host: &Module<HostBytesBackend>,
    module: &Module<BE>,
    cols: usize,
) where
    BE: crate::test_suite::TestBackend,
    Module<BE>: VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + crate::api::VecZnxIdftNormalizeConsume<BE>
        + crate::api::VecZnxIdftNormalizeConsumeTmpBytes
        + crate::api::VecZnxBigAddSmallAssign<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let mut source = Source::new([2u8; 32]);

    for (a_size, res_size) in [(1, 1), (4, 4), (4, 3), (3, 4)] {
        for res_base2k in [base2k, base2k - 2] {
            for with_addend in [false, true] {
                let mut a = module_host.vec_znx_alloc(cols, a_size);
                let mut addend = module_host.vec_znx_alloc(1, a_size);
                a.fill_uniform(base2k, &mut source);
                addend.fill_uniform(base2k, &mut source);
                let addend_backend = upload_vec_znx::<BE, _>(&addend);
                let addend_ref = vec_znx_backend_ref::<BE, _>(&addend_backend);

                let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                    module
                        .vec_znx_idft_normalize_consume_tmp_bytes(res_size, a_size)
                        .max(module.vec_znx_idft_apply_tmp_bytes())
                        .max(module.vec_znx_big_normalize_tmp_bytes()),
                );

                for col in 0..cols {
                    // Fused: res = normalize(idft(a) + addend), clobbering a.
                    let mut a_dft = dft_of_uploaded_vec_znx(module, &a, 1, 0);
                    let res_host_template = VecZnx::alloc(module.n(), 1, res_size);
                    let mut res_have_backend = upload_vec_znx::<BE, _>(&res_host_template);
                    module.vec_znx_idft_normalize_consume(
                        &mut vec_znx_backend_mut::<BE, _>(&mut res_have_backend),
                        res_base2k,
                        0,
                        &mut a_dft.to_backend_mut(),
                        col,
                        base2k,
                        with_addend.then_some((&addend_ref, 0)),
                        &mut scratch.arena(),
                    );
                    let res_have = download_vec_znx::<BE, _>(&res_have_backend);

                    // Composition: idft into BIG, small add, normalize.
                    let a_dft = dft_of_uploaded_vec_znx(module, &a, 1, 0);
                    let mut big = module.vec_znx_big_alloc(1, a_size);
                    module.vec_znx_idft_apply(
                        &mut big.to_backend_mut(),
                        0,
                        &a_dft.to_backend_ref(),
                        col,
                        &mut scratch.arena(),
                    );
                    if with_addend {
                        module.vec_znx_big_add_small_assign(&mut big.to_backend_mut(), 0, &addend_ref, 0);
                    }
                    let mut res_want_backend = upload_vec_znx::<BE, _>(&res_host_template);
                    module.vec_znx_big_normalize(
                        &mut vec_znx_backend_mut::<BE, _>(&mut res_want_backend),
                        res_base2k,
                        0,
                        0,
                        &big.to_backend_ref(),
                        base2k,
                        0,
                        &mut scratch.arena(),
                    );
                    let res_want = download_vec_znx::<BE, _>(&res_want_backend);

                    assert_eq!(
                        res_want, res_have,
                        "idft_normalize_consume != idft + add_small + normalize for a_size={a_size}, res_size={res_size}, \
                         res_base2k={res_base2k}, addend={with_addend}, col={col}"
                    );
                }
            }
        }
    }
}

/// Verifies `vec_znx_idft_normalize_consume` against the composition of
/// `vec_znx_idft_apply`, `vec_znx_big_add_small_assign` and
/// `vec_znx_big_normalize`, on both `module_ref` and `module_test`.
pub fn test_vec_znx_idft_normalize_consume<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: VecZnxDftAlloc<BR>
        + VecZnxDftApply<BR>
        + VecZnxIdftApply<BR>
        + VecZnxIdftApplyTmpBytes
        + crate::api::VecZnxIdftNormalizeConsume<BR>
        + crate::api::VecZnxIdftNormalizeConsumeTmpBytes
        + crate::api::VecZnxBigAddSmallAssign<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: VecZnxDftAlloc<BT>
        + VecZnxDftApply<BT>
        + VecZnxIdftApply<BT>
        + VecZnxIdftApplyTmpBytes
        + crate::api::VecZnxIdftNormalizeConsume<BT>
        + crate::api::VecZnxIdftNormalizeConsumeTmpBytes
        + crate::api::VecZnxBigAddSmallAssign<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let cols = 2;

    idft_normalize_consume_check_one_backend::<BR>(base2k, module_host, module_ref, cols);
    idft_normalize_consume_check_one_backend::<BT>(base2k, module_host, module_test, cols);
}
