use super::{TestParams, download_vec_znx, upload_mat_znx, upload_vec_znx, vec_znx_backend_mut, vec_znx_backend_ref};
use crate::{
    api::{
        ModuleNew, ScratchOwnedAlloc, VecZnxBigAlloc, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAddAssign,
        VecZnxDftAlloc, VecZnxDftApply, VecZnxDftZero, VecZnxIdftApplyTmpA, VmpApplyDft, VmpApplyDftTmpBytes, VmpApplyDftToDft,
        VmpApplyDftToDftAccumulate, VmpApplyDftToDftAccumulateTmpBytes, VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPrepare,
        VmpPrepareTmpBytes,
    },
    layouts::{
        Backend, DigestU64, FillUniform, HostBytesBackend, MatZnx, MatZnxToBackendRef, Module, ScratchOwned, VecZnx, VecZnxBig,
        VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxDft, VecZnxDftToBackendMut, VecZnxDftToBackendRef, VmpPMat,
        VmpPMatToBackendMut, VmpPMatToBackendRef,
    },
    source::Source,
};

type VecZnxDftOwned<BE> = VecZnxDft<<BE as Backend>::OwnedBuf, BE>;
type VmpPMatOwned<BE> = VmpPMat<<BE as Backend>::OwnedBuf, BE>;
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

pub fn test_vmp_apply_dft<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: ModuleNew<BR>
        + VmpApplyDftTmpBytes
        + VmpApplyDft<BR>
        + VmpPMatAlloc<BR>
        + VecZnxDftAlloc<BR>
        + VmpPrepare<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxIdftApplyTmpA<BR>
        + VecZnxBigNormalize<BR>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: ModuleNew<BT>
        + VmpApplyDftTmpBytes
        + VmpApplyDft<BT>
        + VmpPMatAlloc<BT>
        + VecZnxDftAlloc<BT>
        + VmpPrepare<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxIdftApplyTmpA<BT>
        + VecZnxBigNormalize<BT>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let max_size: usize = 4;
    let max_cols: usize = 2;
    let mut source: Source = Source::new([0u8; 32]);

    let mut scratch_ref: ScratchOwned<BR> =
        ScratchOwned::alloc(module_ref.vmp_apply_dft_tmp_bytes(max_size, max_size, max_size, max_cols, max_cols, max_size));
    let mut scratch_test: ScratchOwned<BT> =
        ScratchOwned::alloc(module_test.vmp_apply_dft_tmp_bytes(max_size, max_size, max_size, max_cols, max_cols, max_size));

    for cols_in in 1..max_cols + 1 {
        for cols_out in 1..max_cols + 1 {
            for size_in in 1..max_size + 1 {
                for size_out in 1..max_size + 1 {
                    let rows: usize = cols_in;

                    let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols_in, size_in);
                    a.fill_uniform(base2k, &mut source);
                    let a_digest: u64 = a.digest_u64();
                    let a_ref_backend = upload_vec_znx::<BR>(&a);
                    let a_test_backend = upload_vec_znx::<BT>(&a);

                    let mut mat: MatZnx<Vec<u8>> = module_host.mat_znx_alloc(rows, cols_in, cols_out, size_out);
                    mat.fill_uniform(base2k, &mut source);
                    let mat_digest: u64 = mat.digest_u64();
                    let mat_ref_backend = upload_mat_znx::<BR>(&mat);
                    let mat_test_backend = upload_mat_znx::<BT>(&mat);

                    let mut pmat_ref: VmpPMatOwned<BR> = module_ref.vmp_pmat_alloc(rows, cols_in, cols_out, size_out);
                    let mut pmat_test: VmpPMatOwned<BT> = module_test.vmp_pmat_alloc(rows, cols_in, cols_out, size_out);

                    module_ref.vmp_prepare(
                        &mut pmat_ref.to_backend_mut(),
                        &<MatZnx<BR::OwnedBuf> as MatZnxToBackendRef<BR>>::to_backend_ref(&mat_ref_backend),
                        &mut scratch_ref.arena(),
                    );
                    module_test.vmp_prepare(
                        &mut pmat_test.to_backend_mut(),
                        &<MatZnx<BT::OwnedBuf> as MatZnxToBackendRef<BT>>::to_backend_ref(&mat_test_backend),
                        &mut scratch_test.arena(),
                    );

                    assert_eq!(mat.digest_u64(), mat_digest);

                    let mut res_dft_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols_out, size_out);
                    let mut res_dft_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols_out, size_out);

                    module_ref.vmp_apply_dft(
                        &mut res_dft_ref,
                        &vec_znx_backend_ref::<BR>(&a_ref_backend),
                        &pmat_ref.to_backend_ref(),
                        &mut scratch_ref.arena(),
                    );
                    module_test.vmp_apply_dft(
                        &mut res_dft_test,
                        &vec_znx_backend_ref::<BT>(&a_test_backend),
                        &pmat_test.to_backend_ref(),
                        &mut scratch_test.arena(),
                    );

                    assert_eq!(a.digest_u64(), a_digest);

                    let res_big_ref = idft_into_alloc(module_ref, &mut res_dft_ref);
                    let res_big_test = idft_into_alloc(module_test, &mut res_dft_test);

                    let res_host_template: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols_out, size_out);
                    let mut res_small_ref_backend = upload_vec_znx::<BR>(&res_host_template);
                    let mut res_small_test_backend = upload_vec_znx::<BT>(&res_host_template);

                    for j in 0..cols_out {
                        module_ref.vec_znx_big_normalize(
                            &mut vec_znx_backend_mut::<BR>(&mut res_small_ref_backend),
                            base2k,
                            0,
                            j,
                            &res_big_ref.to_backend_ref(),
                            base2k,
                            j,
                            &mut scratch_ref.arena(),
                        );
                        module_test.vec_znx_big_normalize(
                            &mut vec_znx_backend_mut::<BT>(&mut res_small_test_backend),
                            base2k,
                            0,
                            j,
                            &res_big_test.to_backend_ref(),
                            base2k,
                            j,
                            &mut scratch_test.arena(),
                        );
                    }

                    let res_small_ref = download_vec_znx::<BR>(&res_small_ref_backend);
                    let res_small_test = download_vec_znx::<BT>(&res_small_test_backend);
                    assert_eq!(res_small_ref, res_small_test);
                }
            }
        }
    }
}

pub fn test_vmp_apply_dft_to_dft<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: ModuleNew<BR>
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDft<BR>
        + VmpPMatAlloc<BR>
        + VecZnxDftAlloc<BR>
        + VmpPrepare<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxIdftApplyTmpA<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxDftApply<BR>
        + VmpPrepareTmpBytes
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: ModuleNew<BT>
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDft<BT>
        + VmpPMatAlloc<BT>
        + VecZnxDftAlloc<BT>
        + VmpPrepare<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxIdftApplyTmpA<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxDftApply<BT>
        + VmpPrepareTmpBytes
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let max_size: usize = 4;
    let max_cols: usize = 2;

    let mut source: Source = Source::new([0u8; 32]);

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(
        module_ref
            .vmp_apply_dft_to_dft_tmp_bytes(max_size, max_size, max_size, max_cols, max_cols, max_size)
            .max(module_ref.vmp_prepare_tmp_bytes(max_size, max_cols, max_cols, max_size))
            .max(module_ref.vec_znx_big_normalize_tmp_bytes()),
    );
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(
        module_test
            .vmp_apply_dft_to_dft_tmp_bytes(max_size, max_size, max_size, max_cols, max_cols, max_size)
            .max(module_test.vmp_prepare_tmp_bytes(max_size, max_cols, max_cols, max_size))
            .max(module_test.vec_znx_big_normalize_tmp_bytes()),
    );

    for cols_in in 1..max_cols + 1 {
        for cols_out in 1..max_cols + 1 {
            for size_in in 1..max_size + 1 {
                for size_out in 1..max_size + 1 {
                    let rows: usize = size_in;

                    let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols_in, size_in);
                    a.fill_uniform(base2k, &mut source);
                    let a_digest: u64 = a.digest_u64();
                    let a_ref_backend = upload_vec_znx::<BR>(&a);
                    let a_test_backend = upload_vec_znx::<BT>(&a);

                    let mut a_dft_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols_in, size_in);
                    let mut a_dft_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols_in, size_in);

                    for j in 0..cols_in {
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

                    assert_eq!(a.digest_u64(), a_digest);

                    let mut mat: MatZnx<Vec<u8>> = module_host.mat_znx_alloc(rows, cols_in, cols_out, size_out);
                    mat.fill_uniform(base2k, &mut source);
                    let mat_digest: u64 = mat.digest_u64();
                    let mat_ref_backend = upload_mat_znx::<BR>(&mat);
                    let mat_test_backend = upload_mat_znx::<BT>(&mat);

                    let mut pmat_ref: VmpPMatOwned<BR> = module_ref.vmp_pmat_alloc(rows, cols_in, cols_out, size_out);
                    let mut pmat_test: VmpPMatOwned<BT> = module_test.vmp_pmat_alloc(rows, cols_in, cols_out, size_out);

                    module_ref.vmp_prepare(
                        &mut pmat_ref.to_backend_mut(),
                        &<MatZnx<BR::OwnedBuf> as MatZnxToBackendRef<BR>>::to_backend_ref(&mat_ref_backend),
                        &mut scratch_ref.arena(),
                    );
                    module_test.vmp_prepare(
                        &mut pmat_test.to_backend_mut(),
                        &<MatZnx<BT::OwnedBuf> as MatZnxToBackendRef<BT>>::to_backend_ref(&mat_test_backend),
                        &mut scratch_test.arena(),
                    );

                    assert_eq!(mat.digest_u64(), mat_digest);

                    let mut res_dft_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols_out, size_out);
                    let mut res_dft_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols_out, size_out);

                    module_ref.vmp_apply_dft_to_dft(
                        &mut res_dft_ref.to_backend_mut(),
                        &a_dft_ref.to_backend_ref(),
                        &pmat_ref.to_backend_ref(),
                        0,
                        &mut scratch_ref.arena(),
                    );
                    module_test.vmp_apply_dft_to_dft(
                        &mut res_dft_test.to_backend_mut(),
                        &a_dft_test.to_backend_ref(),
                        &pmat_test.to_backend_ref(),
                        0,
                        &mut scratch_test.arena(),
                    );

                    let res_big_ref = idft_into_alloc(module_ref, &mut res_dft_ref);
                    let res_big_test = idft_into_alloc(module_test, &mut res_dft_test);

                    let res_host_template: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols_out, size_out);
                    let mut res_small_ref_backend = upload_vec_znx::<BR>(&res_host_template);
                    let mut res_small_test_backend = upload_vec_znx::<BT>(&res_host_template);

                    for j in 0..cols_out {
                        module_ref.vec_znx_big_normalize(
                            &mut vec_znx_backend_mut::<BR>(&mut res_small_ref_backend),
                            base2k,
                            0,
                            j,
                            &res_big_ref.to_backend_ref(),
                            base2k,
                            j,
                            &mut scratch_ref.arena(),
                        );
                        module_test.vec_znx_big_normalize(
                            &mut vec_znx_backend_mut::<BT>(&mut res_small_test_backend),
                            base2k,
                            0,
                            j,
                            &res_big_test.to_backend_ref(),
                            base2k,
                            j,
                            &mut scratch_test.arena(),
                        );
                    }

                    let res_small_ref = download_vec_znx::<BR>(&res_small_ref_backend);
                    let res_small_test = download_vec_znx::<BT>(&res_small_test_backend);
                    assert_eq!(res_small_ref, res_small_test);
                }
            }
        }
    }
}

pub fn test_vmp_apply_dft_to_dft_accumulate<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: ModuleNew<BR>
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDft<BR>
        + VmpApplyDftToDftAccumulateTmpBytes
        + VmpApplyDftToDftAccumulate<BR>
        + VmpPMatAlloc<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxDftAddAssign<BR>
        + VecZnxDftZero<BR>
        + VmpPrepare<BR>
        + VecZnxBigAlloc<BR>
        + VecZnxIdftApplyTmpA<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxDftApply<BR>
        + VmpPrepareTmpBytes
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
    Module<BT>: ModuleNew<BT>
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDft<BT>
        + VmpApplyDftToDftAccumulateTmpBytes
        + VmpApplyDftToDftAccumulate<BT>
        + VmpPMatAlloc<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxDftAddAssign<BT>
        + VecZnxDftZero<BT>
        + VmpPrepare<BT>
        + VecZnxBigAlloc<BT>
        + VecZnxIdftApplyTmpA<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxDftApply<BT>
        + VmpPrepareTmpBytes
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());

    let max_size: usize = 4;
    let max_cols: usize = 2;
    let mut source: Source = Source::new([0u8; 32]);

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(
        module_ref
            .vmp_apply_dft_to_dft_tmp_bytes(max_size, max_size, max_size, max_cols, max_cols, max_size)
            .max(module_ref.vmp_apply_dft_to_dft_accumulate_tmp_bytes(max_size, max_size, max_size, max_cols, max_cols, max_size))
            .max(module_ref.vmp_prepare_tmp_bytes(max_size, max_cols, max_cols, max_size))
            .max(module_ref.vec_znx_big_normalize_tmp_bytes()),
    );
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(
        module_test
            .vmp_apply_dft_to_dft_tmp_bytes(max_size, max_size, max_size, max_cols, max_cols, max_size)
            .max(
                module_test.vmp_apply_dft_to_dft_accumulate_tmp_bytes(max_size, max_size, max_size, max_cols, max_cols, max_size),
            )
            .max(module_test.vmp_prepare_tmp_bytes(max_size, max_cols, max_cols, max_size))
            .max(module_test.vec_znx_big_normalize_tmp_bytes()),
    );

    for cols_in in 1..max_cols + 1 {
        for cols_out in 1..max_cols + 1 {
            for size_in in 1..max_size + 1 {
                for size_out in 1..max_size + 1 {
                    let rows: usize = size_in;

                    let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols_in, size_in);
                    a.fill_uniform(base2k, &mut source);
                    let a_ref_backend = upload_vec_znx::<BR>(&a);
                    let a_test_backend = upload_vec_znx::<BT>(&a);

                    let mut res_init: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols_out, size_out);
                    res_init.fill_uniform(base2k, &mut source);
                    let res_init_ref_backend = upload_vec_znx::<BR>(&res_init);
                    let res_init_test_backend = upload_vec_znx::<BT>(&res_init);

                    let mut mat: MatZnx<Vec<u8>> = module_host.mat_znx_alloc(rows, cols_in, cols_out, size_out);
                    mat.fill_uniform(base2k, &mut source);
                    let mat_ref_backend = upload_mat_znx::<BR>(&mat);
                    let mat_test_backend = upload_mat_znx::<BT>(&mat);

                    let mut a_dft_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols_in, size_in);
                    let mut a_dft_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols_in, size_in);
                    for j in 0..cols_in {
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

                    let mut res_init_dft_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols_out, size_out);
                    let mut res_init_dft_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols_out, size_out);
                    for j in 0..cols_out {
                        module_ref.vec_znx_dft_apply(
                            1,
                            0,
                            &mut res_init_dft_ref.to_backend_mut(),
                            j,
                            &vec_znx_backend_ref::<BR>(&res_init_ref_backend),
                            j,
                        );
                        module_test.vec_znx_dft_apply(
                            1,
                            0,
                            &mut res_init_dft_test.to_backend_mut(),
                            j,
                            &vec_znx_backend_ref::<BT>(&res_init_test_backend),
                            j,
                        );
                    }

                    let mut pmat_ref: VmpPMatOwned<BR> = module_ref.vmp_pmat_alloc(rows, cols_in, cols_out, size_out);
                    let mut pmat_test: VmpPMatOwned<BT> = module_test.vmp_pmat_alloc(rows, cols_in, cols_out, size_out);
                    module_ref.vmp_prepare(
                        &mut pmat_ref.to_backend_mut(),
                        &<MatZnx<BR::OwnedBuf> as MatZnxToBackendRef<BR>>::to_backend_ref(&mat_ref_backend),
                        &mut scratch_ref.arena(),
                    );
                    module_test.vmp_prepare(
                        &mut pmat_test.to_backend_mut(),
                        &<MatZnx<BT::OwnedBuf> as MatZnxToBackendRef<BT>>::to_backend_ref(&mat_test_backend),
                        &mut scratch_test.arena(),
                    );

                    let mut res_apply_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols_out, size_out);
                    let mut res_apply_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols_out, size_out);
                    module_ref.vmp_apply_dft_to_dft(
                        &mut res_apply_ref.to_backend_mut(),
                        &a_dft_ref.to_backend_ref(),
                        &pmat_ref.to_backend_ref(),
                        0,
                        &mut scratch_ref.arena(),
                    );
                    module_test.vmp_apply_dft_to_dft(
                        &mut res_apply_test.to_backend_mut(),
                        &a_dft_test.to_backend_ref(),
                        &pmat_test.to_backend_ref(),
                        0,
                        &mut scratch_test.arena(),
                    );
                    for j in 0..cols_out {
                        module_ref.vec_znx_dft_add_assign(
                            &mut res_apply_ref.to_backend_mut(),
                            j,
                            &res_init_dft_ref.to_backend_ref(),
                            j,
                        );
                        module_test.vec_znx_dft_add_assign(
                            &mut res_apply_test.to_backend_mut(),
                            j,
                            &res_init_dft_test.to_backend_ref(),
                            j,
                        );
                    }

                    let mut res_acc_ref = res_init_dft_ref;
                    let mut res_acc_test = res_init_dft_test;
                    module_ref.vmp_apply_dft_to_dft_accumulate(
                        &mut res_acc_ref.to_backend_mut(),
                        &a_dft_ref.to_backend_ref(),
                        &pmat_ref.to_backend_ref(),
                        0,
                        &mut scratch_ref.arena(),
                    );
                    module_test.vmp_apply_dft_to_dft_accumulate(
                        &mut res_acc_test.to_backend_mut(),
                        &a_dft_test.to_backend_ref(),
                        &pmat_test.to_backend_ref(),
                        0,
                        &mut scratch_test.arena(),
                    );

                    let res_apply_big_ref = idft_into_alloc(module_ref, &mut res_apply_ref);
                    let res_apply_big_test = idft_into_alloc(module_test, &mut res_apply_test);
                    let res_acc_big_ref = idft_into_alloc(module_ref, &mut res_acc_ref);
                    let res_acc_big_test = idft_into_alloc(module_test, &mut res_acc_test);

                    let res_host_template: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols_out, size_out);
                    let mut res_apply_small_ref = upload_vec_znx::<BR>(&res_host_template);
                    let mut res_apply_small_test = upload_vec_znx::<BT>(&res_host_template);
                    let mut res_acc_small_ref = upload_vec_znx::<BR>(&res_host_template);
                    let mut res_acc_small_test = upload_vec_znx::<BT>(&res_host_template);

                    for j in 0..cols_out {
                        module_ref.vec_znx_big_normalize(
                            &mut vec_znx_backend_mut::<BR>(&mut res_apply_small_ref),
                            base2k,
                            0,
                            j,
                            &res_apply_big_ref.to_backend_ref(),
                            base2k,
                            j,
                            &mut scratch_ref.arena(),
                        );
                        module_test.vec_znx_big_normalize(
                            &mut vec_znx_backend_mut::<BT>(&mut res_apply_small_test),
                            base2k,
                            0,
                            j,
                            &res_apply_big_test.to_backend_ref(),
                            base2k,
                            j,
                            &mut scratch_test.arena(),
                        );
                        module_ref.vec_znx_big_normalize(
                            &mut vec_znx_backend_mut::<BR>(&mut res_acc_small_ref),
                            base2k,
                            0,
                            j,
                            &res_acc_big_ref.to_backend_ref(),
                            base2k,
                            j,
                            &mut scratch_ref.arena(),
                        );
                        module_test.vec_znx_big_normalize(
                            &mut vec_znx_backend_mut::<BT>(&mut res_acc_small_test),
                            base2k,
                            0,
                            j,
                            &res_acc_big_test.to_backend_ref(),
                            base2k,
                            j,
                            &mut scratch_test.arena(),
                        );
                    }

                    let res_apply_small_ref_v = download_vec_znx::<BR>(&res_apply_small_ref);
                    let res_apply_small_test_v = download_vec_znx::<BT>(&res_apply_small_test);
                    let res_acc_small_ref_v = download_vec_znx::<BR>(&res_acc_small_ref);
                    let res_acc_small_test_v = download_vec_znx::<BT>(&res_acc_small_test);

                    assert_eq!(res_apply_small_ref_v, res_acc_small_ref_v);
                    assert_eq!(res_apply_small_test_v, res_acc_small_test_v);
                    assert_eq!(res_apply_small_ref_v, res_apply_small_test_v);
                }
            }
        }
    }
}
