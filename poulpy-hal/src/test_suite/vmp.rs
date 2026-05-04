use super::TestParams;
use crate::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAlloc,
        VecZnxDftApply, VecZnxIdftApplyConsume, VmpApplyDft, VmpApplyDftTmpBytes, VmpApplyDftToDft, VmpApplyDftToDftAccumulate,
        VmpApplyDftToDftAccumulateTmpBytes, VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPrepare, VmpPrepareTmpBytes,
    },
    layouts::{
        Backend, DataViewMut, DeviceBuf, DigestU64, FillUniform, MatZnx, Module, ScratchOwned, VecZnx, VecZnxBig, VecZnxDft,
        VmpPMat,
    },
    source::Source,
};
use rand::Rng;

type VecZnxDftOwned<BE> = VecZnxDft<DeviceBuf<BE>, BE>;
type VmpPMatOwned<BE> = VmpPMat<DeviceBuf<BE>, BE>;
type VecZnxBigOwned<BE> = VecZnxBig<DeviceBuf<BE>, BE>;

pub fn test_vmp_apply_dft<BR: Backend, BT: Backend>(params: &TestParams, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: ModuleNew<BR>
        + VmpApplyDftTmpBytes
        + VmpApplyDft<BR>
        + VmpPMatAlloc<BR>
        + VecZnxDftAlloc<BR>
        + VmpPrepare<BR>
        + VecZnxDftAlloc<BR>
        + VecZnxIdftApplyConsume<BR>
        + VecZnxBigNormalize<BR>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    Module<BT>: ModuleNew<BT>
        + VmpApplyDftTmpBytes
        + VmpApplyDft<BT>
        + VmpPMatAlloc<BT>
        + VecZnxDftAlloc<BT>
        + VmpPrepare<BT>
        + VecZnxDftAlloc<BT>
        + VecZnxIdftApplyConsume<BT>
        + VecZnxBigNormalize<BT>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

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

                    let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols_in, size_in);
                    a.fill_uniform(base2k, &mut source);
                    let a_digest: u64 = a.digest_u64();

                    let mut mat: MatZnx<Vec<u8>> = MatZnx::alloc(n, rows, cols_in, cols_out, size_out);
                    mat.fill_uniform(base2k, &mut source);
                    let mat_digest: u64 = mat.digest_u64();

                    let mut pmat_ref: VmpPMatOwned<BR> = module_ref.vmp_pmat_alloc(rows, cols_in, cols_out, size_out);
                    let mut pmat_test: VmpPMatOwned<BT> = module_test.vmp_pmat_alloc(rows, cols_in, cols_out, size_out);

                    module_ref.vmp_prepare(&mut pmat_ref, &mat, scratch_ref.borrow());
                    module_test.vmp_prepare(&mut pmat_test, &mat, scratch_test.borrow());

                    assert_eq!(mat.digest_u64(), mat_digest);

                    let mut res_dft_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols_out, size_out);
                    let mut res_dft_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols_out, size_out);

                    source.fill_bytes(res_dft_ref.data_mut().as_mut());
                    source.fill_bytes(res_dft_test.data_mut().as_mut());

                    module_ref.vmp_apply_dft(&mut res_dft_ref, &a, &pmat_ref, scratch_ref.borrow());
                    module_test.vmp_apply_dft(&mut res_dft_test, &a, &pmat_test, scratch_test.borrow());

                    assert_eq!(a.digest_u64(), a_digest);

                    let res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_idft_apply_consume(res_dft_ref);
                    let res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_idft_apply_consume(res_dft_test);

                    let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols_out, size_out);
                    let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols_out, size_out);

                    let res_ref_digest: u64 = res_big_ref.digest_u64();
                    let res_test_digest: u64 = res_big_test.digest_u64();

                    for j in 0..cols_out {
                        module_ref.vec_znx_big_normalize(
                            &mut res_small_ref,
                            base2k,
                            0,
                            j,
                            &res_big_ref,
                            base2k,
                            j,
                            scratch_ref.borrow(),
                        );
                        module_test.vec_znx_big_normalize(
                            &mut res_small_test,
                            base2k,
                            0,
                            j,
                            &res_big_test,
                            base2k,
                            j,
                            scratch_test.borrow(),
                        );
                    }

                    assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
                    assert_eq!(res_big_test.digest_u64(), res_test_digest);

                    assert_eq!(res_small_ref, res_small_test);
                }
            }
        }
    }
}

pub fn test_vmp_apply_dft_to_dft<BR: Backend, BT: Backend>(params: &TestParams, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    Module<BR>: ModuleNew<BR>
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDft<BR>
        + VmpPMatAlloc<BR>
        + VecZnxDftAlloc<BR>
        + VmpPrepare<BR>
        + VecZnxIdftApplyConsume<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxDftApply<BR>
        + VmpPrepareTmpBytes
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    Module<BT>: ModuleNew<BT>
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDft<BT>
        + VmpPMatAlloc<BT>
        + VecZnxDftAlloc<BT>
        + VmpPrepare<BT>
        + VecZnxIdftApplyConsume<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxDftApply<BT>
        + VmpPrepareTmpBytes
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

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

                    let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols_in, size_in);
                    a.fill_uniform(base2k, &mut source);
                    let a_digest: u64 = a.digest_u64();

                    let mut a_dft_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols_in, size_in);
                    let mut a_dft_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols_in, size_in);

                    for j in 0..cols_in {
                        module_ref.vec_znx_dft_apply(1, 0, &mut a_dft_ref, j, &a, j);
                        module_test.vec_znx_dft_apply(1, 0, &mut a_dft_test, j, &a, j);
                    }

                    assert_eq!(a.digest_u64(), a_digest);

                    let mut mat: MatZnx<Vec<u8>> = MatZnx::alloc(n, rows, cols_in, cols_out, size_out);
                    mat.fill_uniform(base2k, &mut source);
                    let mat_digest: u64 = mat.digest_u64();

                    let mut pmat_ref: VmpPMatOwned<BR> = module_ref.vmp_pmat_alloc(rows, cols_in, cols_out, size_out);
                    let mut pmat_test: VmpPMatOwned<BT> = module_test.vmp_pmat_alloc(rows, cols_in, cols_out, size_out);

                    module_ref.vmp_prepare(&mut pmat_ref, &mat, scratch_ref.borrow());
                    module_test.vmp_prepare(&mut pmat_test, &mat, scratch_test.borrow());

                    assert_eq!(mat.digest_u64(), mat_digest);

                    let mut res_dft_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols_out, size_out);
                    let mut res_dft_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols_out, size_out);

                    source.fill_bytes(res_dft_ref.data_mut().as_mut());
                    source.fill_bytes(res_dft_test.data_mut().as_mut());

                    module_ref.vmp_apply_dft_to_dft(&mut res_dft_ref, &a_dft_ref, &pmat_ref, 0, scratch_ref.borrow());
                    module_test.vmp_apply_dft_to_dft(&mut res_dft_test, &a_dft_test, &pmat_test, 0, scratch_test.borrow());

                    let res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_idft_apply_consume(res_dft_ref);
                    let res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_idft_apply_consume(res_dft_test);

                    let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols_out, size_out);
                    let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols_out, size_out);

                    let res_ref_digest: u64 = res_big_ref.digest_u64();
                    let res_test_digest: u64 = res_big_test.digest_u64();

                    for j in 0..cols_out {
                        module_ref.vec_znx_big_normalize(
                            &mut res_small_ref,
                            base2k,
                            0,
                            j,
                            &res_big_ref,
                            base2k,
                            j,
                            scratch_ref.borrow(),
                        );
                        module_test.vec_znx_big_normalize(
                            &mut res_small_test,
                            base2k,
                            0,
                            j,
                            &res_big_test,
                            base2k,
                            j,
                            scratch_test.borrow(),
                        );
                    }

                    assert_eq!(res_big_ref.digest_u64(), res_ref_digest);
                    assert_eq!(res_big_test.digest_u64(), res_test_digest);

                    assert_eq!(res_small_ref, res_small_test);

                    // Test non-zero limb_offset: verify ref == test for each offset
                    for limb_offset in 1..size_out {
                        let mut res_dft_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols_out, size_out);
                        let mut res_dft_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols_out, size_out);

                        module_ref.vmp_apply_dft_to_dft(
                            &mut res_dft_ref,
                            &a_dft_ref,
                            &pmat_ref,
                            limb_offset,
                            scratch_ref.borrow(),
                        );
                        module_test.vmp_apply_dft_to_dft(
                            &mut res_dft_test,
                            &a_dft_test,
                            &pmat_test,
                            limb_offset,
                            scratch_test.borrow(),
                        );

                        let res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_idft_apply_consume(res_dft_ref);
                        let res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_idft_apply_consume(res_dft_test);

                        let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols_out, size_out);
                        let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols_out, size_out);

                        for j in 0..cols_out {
                            module_ref.vec_znx_big_normalize(
                                &mut res_small_ref,
                                base2k,
                                0,
                                j,
                                &res_big_ref,
                                base2k,
                                j,
                                scratch_ref.borrow(),
                            );
                            module_test.vec_znx_big_normalize(
                                &mut res_small_test,
                                base2k,
                                0,
                                j,
                                &res_big_test,
                                base2k,
                                j,
                                scratch_test.borrow(),
                            );
                        }

                        assert_eq!(res_small_ref, res_small_test);
                    }
                }
            }
        }
    }
}

pub fn test_vmp_apply_dft_to_dft_accumulate<BR: Backend, BT: Backend>(
    params: &TestParams,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    Module<BR>: ModuleNew<BR>
        + VmpApplyDftToDftAccumulateTmpBytes
        + VmpApplyDftToDftAccumulate<BR>
        + VmpPMatAlloc<BR>
        + VecZnxDftAlloc<BR>
        + VmpPrepare<BR>
        + VecZnxIdftApplyConsume<BR>
        + VecZnxBigNormalize<BR>
        + VecZnxDftApply<BR>
        + VmpPrepareTmpBytes
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    Module<BT>: ModuleNew<BT>
        + VmpApplyDftToDftAccumulateTmpBytes
        + VmpApplyDftToDftAccumulate<BT>
        + VmpPMatAlloc<BT>
        + VecZnxDftAlloc<BT>
        + VmpPrepare<BT>
        + VecZnxIdftApplyConsume<BT>
        + VecZnxBigNormalize<BT>
        + VecZnxDftApply<BT>
        + VmpPrepareTmpBytes
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    let base2k = params.base2k;
    assert_eq!(module_ref.n(), module_test.n());
    let n: usize = module_ref.n();

    let max_size: usize = 4;
    let max_cols: usize = 2;

    let mut source: Source = Source::new([0u8; 32]);

    let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(
        module_ref
            .vmp_apply_dft_to_dft_accumulate_tmp_bytes(max_size, max_size, max_size, max_cols, max_cols, max_size)
            .max(module_ref.vmp_prepare_tmp_bytes(max_size, max_cols, max_cols, max_size))
            .max(module_ref.vec_znx_big_normalize_tmp_bytes()),
    );
    let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(
        module_test
            .vmp_apply_dft_to_dft_accumulate_tmp_bytes(max_size, max_size, max_size, max_cols, max_cols, max_size)
            .max(module_test.vmp_prepare_tmp_bytes(max_size, max_cols, max_cols, max_size))
            .max(module_test.vec_znx_big_normalize_tmp_bytes()),
    );

    for cols_in in 1..max_cols + 1 {
        for cols_out in 1..max_cols + 1 {
            for size_in in 1..max_size + 1 {
                for size_out in 1..max_size + 1 {
                    let rows: usize = size_in;

                    let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols_in, size_in);
                    a.fill_uniform(base2k, &mut source);

                    let mut a_dft_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols_in, size_in);
                    let mut a_dft_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols_in, size_in);

                    for j in 0..cols_in {
                        module_ref.vec_znx_dft_apply(1, 0, &mut a_dft_ref, j, &a, j);
                        module_test.vec_znx_dft_apply(1, 0, &mut a_dft_test, j, &a, j);
                    }

                    let mut mat: MatZnx<Vec<u8>> = MatZnx::alloc(n, rows, cols_in, cols_out, size_out);
                    mat.fill_uniform(base2k, &mut source);

                    let mut pmat_ref: VmpPMatOwned<BR> = module_ref.vmp_pmat_alloc(rows, cols_in, cols_out, size_out);
                    let mut pmat_test: VmpPMatOwned<BT> = module_test.vmp_pmat_alloc(rows, cols_in, cols_out, size_out);

                    module_ref.vmp_prepare(&mut pmat_ref, &mat, scratch_ref.borrow());
                    module_test.vmp_prepare(&mut pmat_test, &mat, scratch_test.borrow());

                    // Seed res_dft from a known coefficient-domain VecZnx so both backends
                    // start the accumulation from equivalent DFT-domain values.
                    let mut r: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols_out, size_out);
                    r.fill_uniform(base2k, &mut source);

                    for limb_offset in 0..size_out {
                        let mut res_dft_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(cols_out, size_out);
                        let mut res_dft_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(cols_out, size_out);

                        for j in 0..cols_out {
                            module_ref.vec_znx_dft_apply(1, 0, &mut res_dft_ref, j, &r, j);
                            module_test.vec_znx_dft_apply(1, 0, &mut res_dft_test, j, &r, j);
                        }

                        module_ref.vmp_apply_dft_to_dft_accumulate(
                            &mut res_dft_ref,
                            &a_dft_ref,
                            &pmat_ref,
                            limb_offset,
                            scratch_ref.borrow(),
                        );
                        module_test.vmp_apply_dft_to_dft_accumulate(
                            &mut res_dft_test,
                            &a_dft_test,
                            &pmat_test,
                            limb_offset,
                            scratch_test.borrow(),
                        );

                        let res_big_ref: VecZnxBigOwned<BR> = module_ref.vec_znx_idft_apply_consume(res_dft_ref);
                        let res_big_test: VecZnxBigOwned<BT> = module_test.vec_znx_idft_apply_consume(res_dft_test);

                        let mut res_small_ref: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols_out, size_out);
                        let mut res_small_test: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols_out, size_out);

                        for j in 0..cols_out {
                            module_ref.vec_znx_big_normalize(
                                &mut res_small_ref,
                                base2k,
                                0,
                                j,
                                &res_big_ref,
                                base2k,
                                j,
                                scratch_ref.borrow(),
                            );
                            module_test.vec_znx_big_normalize(
                                &mut res_small_test,
                                base2k,
                                0,
                                j,
                                &res_big_test,
                                base2k,
                                j,
                                scratch_test.borrow(),
                            );
                        }

                        assert_eq!(res_small_ref, res_small_test);
                    }
                }
            }
        }
    }
}
