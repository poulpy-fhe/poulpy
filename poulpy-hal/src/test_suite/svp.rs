use super::{
    TestParams, download_vec_znx, scalar_znx_backend_ref, upload_scalar_znx, upload_vec_znx, vec_znx_backend_mut,
    vec_znx_backend_ref,
};

use crate::{
    api::{
        ScratchOwnedAlloc, SvpApplyDft, SvpApplyDftToDft, SvpApplyDftToDftAssign, SvpPPolAlloc, SvpPrepare, VecZnxBigAlloc,
        VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAlloc, VecZnxDftApply, VecZnxIdftApplyTmpA,
    },
    layouts::{
        Backend, FillUniform, HostBytesBackend, Module, ScalarZnx, ScratchOwned, SvpPPolOwned, SvpPPolToBackendMut,
        SvpPPolToBackendRef, VecZnx, VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxDft, VecZnxDftToBackendMut,
        VecZnxDftToBackendRef,
    },
    source::Source,
};

type VecZnxDftOwned<BE> = VecZnxDft<<BE as Backend>::OwnedBuf, BE>;
type VecZnxBigOwned<BE> = crate::layouts::VecZnxBig<<BE as Backend>::OwnedBuf, BE>;

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

pub fn test_svp_apply_dft<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: SvpPrepare<BR>
        + SvpApplyDft<BR>
        + SvpPPolAlloc<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxIdftApplyTmpA<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: SvpPrepare<BT>
        + SvpApplyDft<BT>
        + SvpPPolAlloc<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxIdftApplyTmpA<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let cols: usize = 2;

    let mut source: Source = Source::new([0u8; 32]);

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    let mut scalar: ScalarZnx<Vec<u8>> = module_host.scalar_znx_alloc(cols);
    scalar.fill_uniform(base2k, &mut source);

    let mut svp_ref: SvpPPolOwned<BR> = module_ref.svp_ppol_alloc(cols);
    let mut svp_test: SvpPPolOwned<BT> = module_test.svp_ppol_alloc(cols);
    let scalar_ref_backend = upload_scalar_znx::<BR>(&scalar);
    let scalar_test_backend = upload_scalar_znx::<BT>(&scalar);

    for j in 0..cols {
        module_ref.svp_prepare(
            &mut svp_ref.to_backend_mut(),
            j,
            &scalar_znx_backend_ref::<BR>(&scalar_ref_backend),
            j,
        );
        module_test.svp_prepare(
            &mut svp_test.to_backend_mut(),
            j,
            &scalar_znx_backend_ref::<BT>(&scalar_test_backend),
            j,
        );
    }

    for a_size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_ref_backend = upload_vec_znx::<BR>(&a);
        let a_test_backend = upload_vec_znx::<BT>(&a);

        for res_size in [1, 2, 3, 4] {
            let mut res_dft_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols, res_size);
            let mut res_dft_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols, res_size);

            for j in 0..cols {
                module_ref.svp_apply_dft(
                    &mut res_dft_ref.to_backend_mut(),
                    j,
                    &svp_ref.to_backend_ref(),
                    j,
                    &vec_znx_backend_ref::<BR>(&a_ref_backend),
                    j,
                );
                module_test.svp_apply_dft(
                    &mut res_dft_test.to_backend_mut(),
                    j,
                    &svp_test.to_backend_ref(),
                    j,
                    &vec_znx_backend_ref::<BT>(&a_test_backend),
                    j,
                );
            }

            let res_big_ref = idft_into_alloc(module_ref, &mut res_dft_ref);
            let res_big_test = idft_into_alloc(module_test, &mut res_dft_test);

            let res_host_template: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
            let mut res_ref_backend = upload_vec_znx::<BR>(&res_host_template);
            let mut res_test_backend = upload_vec_znx::<BT>(&res_host_template);

            for j in 0..cols {
                module_ref.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                    base2k,
                    0,
                    j,
                    &res_big_ref.to_backend_ref(),
                    base2k,
                    j,
                    &mut scratch_ref.arena(),
                );
                module_test.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                    base2k,
                    0,
                    j,
                    &res_big_test.to_backend_ref(),
                    base2k,
                    j,
                    &mut scratch_test.arena(),
                );
            }

            let res_ref = download_vec_znx::<BR>(&res_ref_backend);
            let res_test = download_vec_znx::<BT>(&res_test_backend);
            assert_eq!(res_ref, res_test);
        }
    }
}

pub fn test_svp_apply_dft_to_dft<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: SvpPrepare<BR>
        + SvpApplyDftToDft<BR>
        + SvpPPolAlloc<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxDftApply<BR>
        + VecZnxIdftApplyTmpA<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: SvpPrepare<BT>
        + SvpApplyDftToDft<BT>
        + SvpPPolAlloc<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxDftApply<BT>
        + VecZnxIdftApplyTmpA<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let cols: usize = 2;

    let mut source: Source = Source::new([0u8; 32]);

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    let mut scalar: ScalarZnx<Vec<u8>> = module_host.scalar_znx_alloc(cols);
    scalar.fill_uniform(base2k, &mut source);

    let mut svp_ref: SvpPPolOwned<BR> = module_ref.svp_ppol_alloc(cols);
    let mut svp_test: SvpPPolOwned<BT> = module_test.svp_ppol_alloc(cols);
    let scalar_ref_backend = upload_scalar_znx::<BR>(&scalar);
    let scalar_test_backend = upload_scalar_znx::<BT>(&scalar);

    for j in 0..cols {
        module_ref.svp_prepare(
            &mut svp_ref.to_backend_mut(),
            j,
            &scalar_znx_backend_ref::<BR>(&scalar_ref_backend),
            j,
        );
        module_test.svp_prepare(
            &mut svp_test.to_backend_mut(),
            j,
            &scalar_znx_backend_ref::<BT>(&scalar_test_backend),
            j,
        );
    }

    for a_size in [3] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_ref_backend = upload_vec_znx::<BR>(&a);
        let a_test_backend = upload_vec_znx::<BT>(&a);

        let mut a_dft_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols, a_size);
        let mut a_dft_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols, a_size);

        for j in 0..cols {
            module_ref.vec_znx_dft_apply(
                1,
                0,
                &mut a_dft_ref.to_backend_mut(),
                j,
                &vec_znx_backend_ref::<BR>(&a_ref_backend),
                j,
            );
            module_test.vec_znx_dft_apply(
                1,
                0,
                &mut a_dft_test.to_backend_mut(),
                j,
                &vec_znx_backend_ref::<BT>(&a_test_backend),
                j,
            );
        }

        for res_size in [3] {
            let mut res_dft_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols, res_size);
            let mut res_dft_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols, res_size);

            for j in 0..cols {
                module_ref.svp_apply_dft_to_dft(
                    &mut res_dft_ref.to_backend_mut(),
                    j,
                    &svp_ref.to_backend_ref(),
                    j,
                    &a_dft_ref.to_backend_ref(),
                    j,
                );
                module_test.svp_apply_dft_to_dft(
                    &mut res_dft_test.to_backend_mut(),
                    j,
                    &svp_test.to_backend_ref(),
                    j,
                    &a_dft_test.to_backend_ref(),
                    j,
                );
            }

            let res_big_ref = idft_into_alloc(module_ref, &mut res_dft_ref);
            let res_big_test = idft_into_alloc(module_test, &mut res_dft_test);

            let res_host_template: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
            let mut res_ref_backend = upload_vec_znx::<BR>(&res_host_template);
            let mut res_test_backend = upload_vec_znx::<BT>(&res_host_template);

            for j in 0..cols {
                module_ref.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                    base2k,
                    0,
                    j,
                    &res_big_ref.to_backend_ref(),
                    base2k,
                    j,
                    &mut scratch_ref.arena(),
                );
                module_test.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                    base2k,
                    0,
                    j,
                    &res_big_test.to_backend_ref(),
                    base2k,
                    j,
                    &mut scratch_test.arena(),
                );
            }

            let res_ref = download_vec_znx::<BR>(&res_ref_backend);
            let res_test = download_vec_znx::<BT>(&res_test_backend);
            assert_eq!(res_ref, res_test);
        }
    }
}

pub fn test_svp_apply_dft_to_dft_assign<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: SvpPrepare<BR>
        + SvpApplyDftToDftAssign<BR>
        + SvpPPolAlloc<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxDftApply<BR>
        + VecZnxIdftApplyTmpA<BR>
        + VecZnxBigNormalizeTmpBytes,
    Module<BT>: SvpPrepare<BT>
        + SvpApplyDftToDftAssign<BT>
        + SvpPPolAlloc<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxDftApply<BT>
        + VecZnxIdftApplyTmpA<BT>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let cols: usize = 2;

    let mut source: Source = Source::new([0u8; 32]);

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.vec_znx_big_normalize_tmp_bytes());

    let mut scalar: ScalarZnx<Vec<u8>> = module_host.scalar_znx_alloc(cols);
    scalar.fill_uniform(base2k, &mut source);

    let mut svp_ref: SvpPPolOwned<BR> = module_ref.svp_ppol_alloc(cols);
    let mut svp_test: SvpPPolOwned<BT> = module_test.svp_ppol_alloc(cols);
    let scalar_ref_backend = upload_scalar_znx::<BR>(&scalar);
    let scalar_test_backend = upload_scalar_znx::<BT>(&scalar);

    for j in 0..cols {
        module_ref.svp_prepare(
            &mut svp_ref.to_backend_mut(),
            j,
            &scalar_znx_backend_ref::<BR>(&scalar_ref_backend),
            j,
        );
        module_test.svp_prepare(
            &mut svp_test.to_backend_mut(),
            j,
            &scalar_znx_backend_ref::<BT>(&scalar_test_backend),
            j,
        );
    }

    for res_size in [1, 2, 3, 4] {
        let mut res: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
        res.fill_uniform(base2k, &mut source);
        let res_ref_backend_input = upload_vec_znx::<BR>(&res);
        let res_test_backend_input = upload_vec_znx::<BT>(&res);

        let mut res_dft_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols, res_size);
        let mut res_dft_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols, res_size);

        for j in 0..cols {
            module_ref.vec_znx_dft_apply(
                1,
                0,
                &mut res_dft_ref.to_backend_mut(),
                j,
                &vec_znx_backend_ref::<BR>(&res_ref_backend_input),
                j,
            );
            module_test.vec_znx_dft_apply(
                1,
                0,
                &mut res_dft_test.to_backend_mut(),
                j,
                &vec_znx_backend_ref::<BT>(&res_test_backend_input),
                j,
            );
        }

        for j in 0..cols {
            module_ref.svp_apply_dft_to_dft_assign(&mut res_dft_ref.to_backend_mut(), j, &svp_ref.to_backend_ref(), j);
            module_test.svp_apply_dft_to_dft_assign(&mut res_dft_test.to_backend_mut(), j, &svp_test.to_backend_ref(), j);
        }

        let res_big_ref = idft_into_alloc(module_ref, &mut res_dft_ref);
        let res_big_test = idft_into_alloc(module_test, &mut res_dft_test);

        let res_host_template: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, res_size);
        let mut res_ref_backend = upload_vec_znx::<BR>(&res_host_template);
        let mut res_test_backend = upload_vec_znx::<BT>(&res_host_template);

        for j in 0..cols {
            module_ref.vec_znx_big_normalize(
                &mut vec_znx_backend_mut::<BR>(&mut res_ref_backend),
                base2k,
                0,
                j,
                &res_big_ref.to_backend_ref(),
                base2k,
                j,
                &mut scratch_ref.arena(),
            );
            module_test.vec_znx_big_normalize(
                &mut vec_znx_backend_mut::<BT>(&mut res_test_backend),
                base2k,
                0,
                j,
                &res_big_test.to_backend_ref(),
                base2k,
                j,
                &mut scratch_test.arena(),
            );
        }

        let res_ref = download_vec_znx::<BR>(&res_ref_backend);
        let res_test = download_vec_znx::<BT>(&res_test_backend);
        assert_eq!(res_ref, res_test);
    }
}
