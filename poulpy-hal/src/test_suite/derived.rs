//! Default-body decomposition pinning for the derived operations.
//!
//! Each test hand-builds the oracle for one derived op by driving the *same*
//! decomposition the op's `poulpy_hal::oep::*_derived` function performs, but
//! through the public api traits on `Module` rather than the OEP dispatch the
//! derived function uses internally, then compares that oracle against the
//! derived function's own output on the same inputs. `test_vmp_apply_dft` in
//! `test_suite/vmp.rs` is the correctness oracle for `vmp_apply_dft` itself
//! (checked bit-for-bit against a second backend there); this test instead
//! pins the *shape* of the default body's decomposition, the column
//! alignment and call order, independently of any backend override. Where a
//! backend does keep an override (ruling R10: `vec_znx_lsh_assign`,
//! `vec_znx_mul_xp_minus_one_assign` and the three small-operand `VecZnxBig`
//! products), the test additionally drives the op through `Module`, so the
//! override is pinned to the same oracle as the default body. Coefficient- and
//! big-domain results are compared bit for bit; DFT-domain results are
//! compared after `idft` and `big_normalize` at the suite's `base2k`, which is
//! how the rest of this suite states DFT-domain equality.
//!
//! Where the two sides are allowed to distribute the same value over
//! different non-canonical digits, the accumulating shifts, the comparison
//! is on canonical forms, as in the `rsh_sub` argument.

use crate::{
    api::{
        CnvPVecAlloc, Convolution, MatZnxAlloc, ModuleN, ScratchOwnedAlloc, ScratchOwnedBorrow, SvpApplyDftToDft, SvpPPolAlloc,
        SvpPrepare, VecZnxAdd, VecZnxAddAssign, VecZnxAddScalarAssign, VecZnxAlloc, VecZnxBigAddAssign, VecZnxBigAddSmall,
        VecZnxBigAddSmallAssign, VecZnxBigAlloc, VecZnxBigFromSmall, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxBigSubAssign, VecZnxBigSubNegateAssign, VecZnxBigSubSmallA, VecZnxBigSubSmallB, VecZnxCopy, VecZnxDftAddAssign,
        VecZnxDftAlloc, VecZnxDftApply, VecZnxDftAutomorphism, VecZnxDftAutomorphismPlan, VecZnxDftZero, VecZnxIdftApplyTmpA,
        VecZnxLsh, VecZnxLshAdd, VecZnxLshAssign, VecZnxLshSub, VecZnxLshTmpBytes, VecZnxMulXpMinusOne,
        VecZnxMulXpMinusOneAssign, VecZnxMulXpMinusOneAssignTmpBytes, VecZnxNormalize, VecZnxNormalizeAssign,
        VecZnxNormalizeTmpBytes, VecZnxRotate, VecZnxRsh, VecZnxRshAdd, VecZnxRshAssign, VecZnxRshSub, VecZnxRshTmpBytes,
        VecZnxSub, VecZnxSubAssign, VmpApplyDft, VmpApplyDftTmpBytes, VmpApplyDftToDft, VmpApplyDftToDftAdd,
        VmpApplyDftToDftAddTmpBytes, VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPrepare, VmpPrepareTmpBytes,
    },
    layouts::{
        CnvDftAccTerm, CnvPVecLOwned, CnvPVecLToBackendMut, CnvPVecLToBackendRef, CnvPVecROwned, CnvPVecRToBackendMut,
        CnvPVecRToBackendRef, FillUniform, HostBytesBackend, MatZnx, MatZnxInfos, MatZnxToBackendRef, Module, PrepareHint,
        ScratchOwned, SvpPPolOwned, SvpPPolToBackendMut, SvpPPolToBackendRef, VecZnx, VecZnxBigToBackendMut,
        VecZnxBigToBackendRef, VecZnxDftToBackendMut, VecZnxDftToBackendRef, VecZnxInfos, VecZnxOwned, VmpPMatToBackendMut,
        VmpPMatToBackendRef, ZnxInfos, ZnxView, ZnxViewMut, vec_znx_backend_mut, vec_znx_backend_ref,
    },
    oep::{
        HalConvolutionImpl, HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl, HalVmpImpl, cnv_apply_dft_add_derived,
        cnv_apply_dft_add_tmp_bytes_derived, cnv_apply_dft_sum_derived, cnv_apply_dft_sum_tmp_bytes_derived,
        cnv_by_const_apply_add_derived, cnv_by_const_apply_add_tmp_bytes_derived, cnv_pairwise_apply_dft_derived,
        cnv_pairwise_apply_dft_tmp_bytes_derived, cnv_prepare_self_derived, cnv_prepare_self_tmp_bytes_derived,
        vmp_apply_dft_derived, vmp_apply_dft_to_dft_add_derived, vmp_apply_dft_to_dft_add_tmp_bytes_derived,
    },
    source::Source,
    test_suite::{
        TestBackend, TestParams, download_vec_znx, scalar_znx_backend_ref, upload_mat_znx, upload_scalar_znx, upload_vec_znx,
    },
};

/// `vmp_apply_dft`: the derived free function's decomposition versus an
/// oracle hand-built from the public api traits.
pub fn test_vmp_apply_dft_derived<BE: TestBackend + HalVmpImpl>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: ModuleN
        + VecZnxAlloc<BE>
        + MatZnxAlloc<BE>
        + VmpPMatAlloc<BE>
        + VmpPrepare<BE>
        + VmpPrepareTmpBytes
        + VmpApplyDft<BE>
        + VmpApplyDftTmpBytes
        + VmpApplyDftToDft<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxDftZero<BE>
        + VecZnxDftApply<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let (rows, cols_in, cols_out, size) = (4usize, 2usize, 2usize, 4usize);
    let mut source: Source = Source::new([0u8; 32]);

    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(params.n as u64);

    let mut a = module_host.vec_znx_alloc(params.n, cols_in, size);
    a.fill_uniform(base2k, &mut source);
    let mut mat = module_host.mat_znx_alloc(params.n, rows, cols_in, cols_out, size);
    mat.fill_uniform(base2k, &mut source);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .vmp_prepare_tmp_bytes(rows, cols_in, cols_out, size)
            .max(module.vmp_apply_dft_tmp_bytes(size, size, rows, cols_in, cols_out, size))
            .max(module.vec_znx_big_normalize_tmp_bytes()),
    );

    let mat_backend = upload_mat_znx::<BE>(&mat);
    let mut pmat = module.vmp_pmat_alloc(params.n, rows, cols_in, cols_out, size, PrepareHint::Reuse);
    module.vmp_prepare(
        &mut pmat.to_backend_mut(),
        &<MatZnx<BE::OwnedBuf, i64> as MatZnxToBackendRef<BE>>::to_backend_ref(&mat_backend),
        &mut scratch.borrow(),
    );

    let a_backend = upload_vec_znx::<BE>(&a);
    let a_ref = vec_znx_backend_ref::<BE>(&a_backend);
    let pmat_ref = pmat.to_backend_ref();

    // Oracle: the same column alignment `vmp_apply_dft_derived` uses, driven
    // through the public api traits instead of the OEP dispatch.
    let a_cols: usize = VecZnxInfos::cols(&a_ref);
    let a_size: usize = ZnxInfos::size(&a_ref);
    let b_rows: usize = MatZnxInfos::rows(&pmat_ref);
    let b_cols_in: usize = MatZnxInfos::cols_in(&pmat_ref);
    let cols_to_copy: usize = a_cols.min(b_cols_in);
    let a_start_col: usize = a_cols - cols_to_copy;
    let a_dft_size: usize = a_size.min(b_rows);
    let offset: usize = b_cols_in - cols_to_copy;

    let mut a_dft_oracle = module.vec_znx_dft_alloc(params.n, b_cols_in, a_dft_size);
    for j in 0..offset {
        module.vec_znx_dft_zero(&mut a_dft_oracle.to_backend_mut(), j);
    }
    for j in 0..cols_to_copy {
        module.vec_znx_dft_apply(1, 0, &mut a_dft_oracle.to_backend_mut(), offset + j, &a_ref, a_start_col + j);
    }

    let mut res_oracle = module.vec_znx_dft_alloc(params.n, cols_out, size);
    module.vmp_apply_dft_to_dft(
        &mut res_oracle.to_backend_mut(),
        &a_dft_oracle.to_backend_ref(),
        &pmat_ref,
        0,
        &mut scratch.borrow(),
    );

    let mut res_derived = module.vec_znx_dft_alloc(params.n, cols_out, size);
    vmp_apply_dft_derived::<BE, _>(module, &mut res_derived, &a_ref, &pmat_ref, &mut scratch.borrow());

    let mut big = module.vec_znx_big_alloc(params.n, 1, size);
    for col in 0..cols_out {
        let want_template = module_host.vec_znx_alloc(params.n, 1, size);
        let have_template = module_host.vec_znx_alloc(params.n, 1, size);
        let mut want_backend = upload_vec_znx::<BE>(&want_template);
        let mut have_backend = upload_vec_znx::<BE>(&have_template);

        module.vec_znx_idft_apply_tmpa(&mut big.to_backend_mut(), 0, &mut res_oracle.to_backend_mut(), col);
        module.vec_znx_big_normalize(
            &mut vec_znx_backend_mut::<BE>(&mut want_backend),
            base2k,
            size * base2k,
            0,
            0,
            &big.to_backend_ref(),
            base2k,
            0,
            &mut scratch.borrow(),
        );
        module.vec_znx_idft_apply_tmpa(&mut big.to_backend_mut(), 0, &mut res_derived.to_backend_mut(), col);
        module.vec_znx_big_normalize(
            &mut vec_znx_backend_mut::<BE>(&mut have_backend),
            base2k,
            size * base2k,
            0,
            0,
            &big.to_backend_ref(),
            base2k,
            0,
            &mut scratch.borrow(),
        );

        let want = download_vec_znx::<BE>(&want_backend);
        let have = download_vec_znx::<BE>(&have_backend);
        assert_eq!(
            want, have,
            "vmp_apply_dft: derived decomposition != hand-built oracle (col {col})"
        );
    }
}

/// `vmp_apply_dft_to_dft_add`: the derived free function's decomposition
/// versus an explicit two-step oracle (`vmp_apply_dft_to_dft` into a fresh
/// accumulator, then `vec_znx_dft_add_assign` per column), and versus the
/// backend's own `module.vmp_apply_dft_to_dft_add` dispatch, the inherited
/// default on backends with no fused override, the fused kernel on backends
/// that have one. All three are compared bit for bit after
/// `vec_znx_idft_apply_tmpa` + `vec_znx_big_normalize`, the same comparison
/// `test_vmp_apply_dft_to_dft_add` in `test_suite/vmp.rs` uses for
/// DFT-domain results. The scratch for the derived call is sized only by
/// `vmp_apply_dft_to_dft_add_tmp_bytes_derived`, so a decomposition that
/// under-reports its own scratch panics here.
pub fn test_vmp_apply_dft_to_dft_add_derived<BE: TestBackend + HalVmpImpl>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: ModuleN
        + VmpPMatAlloc<BE>
        + VmpPrepare<BE>
        + VmpPrepareTmpBytes
        + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDftAdd<BE>
        + VmpApplyDftToDftAddTmpBytes
        + VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    // `cols_out >= 2`, `cols_in != cols_out` on the second shape, and, for two
    // of the three (res_size, a_size) pairs, `res_size != a_size`, the cases
    // the deleted hand-written bodies handled identically (they always staged a
    // `res.size()`-limb, not `a.size()`-limb, accumulator).
    let mat_size: usize = 4;
    let mut source: Source = Source::new([0u8; 32]);

    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(params.n as u64);

    for ((cols_in, cols_out), (res_size, a_size)) in [(2usize, 2usize), (2, 3)]
        .into_iter()
        .flat_map(|cols| [(4usize, 4usize), (3, 4), (4, 3)].map(|sizes| (cols, sizes)))
    {
        let rows: usize = a_size;

        let mut a = module_host.vec_znx_alloc(params.n, cols_in, a_size);
        a.fill_uniform(base2k, &mut source);
        let mut mat = module_host.mat_znx_alloc(params.n, rows, cols_in, cols_out, mat_size);
        mat.fill_uniform(base2k, &mut source);
        let mut res_init = module_host.vec_znx_alloc(params.n, cols_out, res_size);
        res_init.fill_uniform(base2k, &mut source);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .vmp_prepare_tmp_bytes(rows, cols_in, cols_out, mat_size)
                .max(module.vmp_apply_dft_to_dft_tmp_bytes(res_size, a_size, rows, cols_in, cols_out, mat_size))
                .max(module.vmp_apply_dft_to_dft_add_tmp_bytes(res_size, a_size, rows, cols_in, cols_out, mat_size))
                .max(module.vec_znx_big_normalize_tmp_bytes()),
        );

        let mat_backend = upload_mat_znx::<BE>(&mat);
        let mut pmat = module.vmp_pmat_alloc(params.n, rows, cols_in, cols_out, mat_size, PrepareHint::Reuse);
        module.vmp_prepare(
            &mut pmat.to_backend_mut(),
            &<MatZnx<BE::OwnedBuf, i64> as MatZnxToBackendRef<BE>>::to_backend_ref(&mat_backend),
            &mut scratch.borrow(),
        );
        let pmat_ref = pmat.to_backend_ref();

        let a_backend = upload_vec_znx::<BE>(&a);
        let a_ref = vec_znx_backend_ref::<BE>(&a_backend);
        let mut a_dft = module.vec_znx_dft_alloc(params.n, cols_in, a_size);
        for j in 0..cols_in {
            module.vec_znx_dft_apply(1, 0, &mut a_dft.to_backend_mut(), j, &a_ref, j);
        }
        let a_dft_ref = a_dft.to_backend_ref();

        let res_init_backend = upload_vec_znx::<BE>(&res_init);
        let res_init_ref = vec_znx_backend_ref::<BE>(&res_init_backend);

        for limb_offset in [0usize, 1usize] {
            // Three independent copies of the same initial accumulator, one
            // per path, so each can be mutated without disturbing the others.
            let mut res_oracle = module.vec_znx_dft_alloc(params.n, cols_out, res_size);
            let mut res_derived = module.vec_znx_dft_alloc(params.n, cols_out, res_size);
            let mut res_module = module.vec_znx_dft_alloc(params.n, cols_out, res_size);
            for j in 0..cols_out {
                module.vec_znx_dft_apply(1, 0, &mut res_oracle.to_backend_mut(), j, &res_init_ref, j);
                module.vec_znx_dft_apply(1, 0, &mut res_derived.to_backend_mut(), j, &res_init_ref, j);
                module.vec_znx_dft_apply(1, 0, &mut res_module.to_backend_mut(), j, &res_init_ref, j);
            }

            // Oracle: the contract definition of `vmp_apply_dft_to_dft_add`,
            // spelled out through the public api traits, a fresh product,
            // then folded in column by column.
            let mut fresh = module.vec_znx_dft_alloc(params.n, cols_out, res_size);
            module.vmp_apply_dft_to_dft(
                &mut fresh.to_backend_mut(),
                &a_dft_ref,
                &pmat_ref,
                limb_offset,
                &mut scratch.borrow(),
            );
            for j in 0..cols_out {
                module.vec_znx_dft_add_assign(&mut res_oracle.to_backend_mut(), j, &fresh.to_backend_ref(), j);
            }

            // The derived free function, dispatched directly (bypassing the
            // OEP method entirely) with an arena sized only by its own
            // `_tmp_bytes_derived` sibling.
            let mut derived_scratch: ScratchOwned<BE> = ScratchOwned::alloc(vmp_apply_dft_to_dft_add_tmp_bytes_derived::<BE>(
                module, res_size, a_size, rows, cols_in, cols_out, mat_size,
            ));
            vmp_apply_dft_to_dft_add_derived::<BE>(
                module,
                &mut res_derived.to_backend_mut(),
                &a_dft_ref,
                &pmat_ref,
                limb_offset,
                &mut derived_scratch.borrow(),
            );

            // The backend's own dispatch through the OEP method: the
            // inherited default on backends with no fused override, the
            // fused kernel on backends that have one.
            module.vmp_apply_dft_to_dft_add(
                &mut res_module.to_backend_mut(),
                &a_dft_ref,
                &pmat_ref,
                limb_offset,
                &mut scratch.borrow(),
            );

            let mut big = module.vec_znx_big_alloc(params.n, 1, res_size);
            for j in 0..cols_out {
                let want_template = module_host.vec_znx_alloc(params.n, 1, res_size);
                let mut oracle_backend = upload_vec_znx::<BE>(&want_template);
                let mut derived_backend = upload_vec_znx::<BE>(&want_template);
                let mut module_backend = upload_vec_znx::<BE>(&want_template);

                module.vec_znx_idft_apply_tmpa(&mut big.to_backend_mut(), 0, &mut res_oracle.to_backend_mut(), j);
                module.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BE>(&mut oracle_backend),
                    base2k,
                    res_size * base2k,
                    0,
                    0,
                    &big.to_backend_ref(),
                    base2k,
                    0,
                    &mut scratch.borrow(),
                );
                module.vec_znx_idft_apply_tmpa(&mut big.to_backend_mut(), 0, &mut res_derived.to_backend_mut(), j);
                module.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BE>(&mut derived_backend),
                    base2k,
                    res_size * base2k,
                    0,
                    0,
                    &big.to_backend_ref(),
                    base2k,
                    0,
                    &mut scratch.borrow(),
                );
                module.vec_znx_idft_apply_tmpa(&mut big.to_backend_mut(), 0, &mut res_module.to_backend_mut(), j);
                module.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BE>(&mut module_backend),
                    base2k,
                    res_size * base2k,
                    0,
                    0,
                    &big.to_backend_ref(),
                    base2k,
                    0,
                    &mut scratch.borrow(),
                );

                let want = download_vec_znx::<BE>(&oracle_backend);
                let have_derived = download_vec_znx::<BE>(&derived_backend);
                let have_module = download_vec_znx::<BE>(&module_backend);

                assert_eq!(
                    want, have_derived,
                    "vmp_apply_dft_to_dft_add: derived decomposition != two-step oracle (cols_in {cols_in} cols_out {cols_out} res_size {res_size} a_size {a_size} limb_offset {limb_offset} col {j})"
                );
                assert_eq!(
                    want, have_module,
                    "vmp_apply_dft_to_dft_add: module dispatch != two-step oracle (cols_in {cols_in} cols_out {cols_out} res_size {res_size} a_size {a_size} limb_offset {limb_offset} col {j})"
                );
            }
        }
    }
}

/// `vec_znx_lsh`: the OEP default body against the definition it is derived
/// from, written out, `normalize(res_base2k = base2k, res_k = res_size *
/// base2k, res_offset = +k)`. Both sides are canonical, so this is bit for
/// bit. The arena is sized by `vec_znx_lsh_tmp_bytes(res_size)` alone, so a
/// default that under-reports its scratch panics here.
pub fn test_vec_znx_lsh_derived<BE: TestBackend + HalVecZnxImpl>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: VecZnxLsh<BE> + VecZnxLshTmpBytes + VecZnxNormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let mut source: Source = Source::new([0u8; 32]);
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(params.n as u64);

    for a_size in [1usize, 2, 4] {
        let mut a = module_host.vec_znx_alloc(params.n, 1, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_backend = upload_vec_znx::<BE>(&a);

        for res_size in [1usize, 2, 4] {
            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.vec_znx_lsh_tmp_bytes(res_size));

            for k in 0..=(res_size * base2k) {
                let mut res = module_host.vec_znx_alloc(params.n, 1, res_size);
                res.fill_uniform(base2k, &mut source);
                let mut have_backend = upload_vec_znx::<BE>(&res);
                let mut want_backend = upload_vec_znx::<BE>(&res);

                module.vec_znx_lsh(
                    base2k,
                    k,
                    &mut vec_znx_backend_mut::<BE>(&mut have_backend),
                    0,
                    &vec_znx_backend_ref::<BE>(&a_backend),
                    0,
                    &mut scratch.borrow(),
                );

                // Oracle: the contract definition of `lsh`, spelled out.
                module.vec_znx_normalize(
                    &mut vec_znx_backend_mut::<BE>(&mut want_backend),
                    base2k,
                    res_size * base2k,
                    k as i64,
                    0,
                    &vec_znx_backend_ref::<BE>(&a_backend),
                    base2k,
                    0,
                    &mut scratch.borrow(),
                );

                assert_eq!(
                    download_vec_znx::<BE>(&want_backend),
                    download_vec_znx::<BE>(&have_backend),
                    "vec_znx_lsh: default body != normalize(offset) (a_size {a_size} res_size {res_size} k {k})"
                );
            }
        }
    }
}

/// `vec_znx_rsh`: the OEP default body against the definition it is derived
/// from, written out, `normalize(res_base2k = base2k, res_k = res_size *
/// base2k, res_offset = -k)`. Both sides are canonical, so this is bit for
/// bit. The arena is sized by `vec_znx_rsh_tmp_bytes(res_size)` alone, so a
/// default that under-reports its scratch panics here.
pub fn test_vec_znx_rsh_derived<BE: TestBackend + HalVecZnxImpl>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: VecZnxRsh<BE> + VecZnxRshTmpBytes + VecZnxNormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let mut source: Source = Source::new([0u8; 32]);
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(params.n as u64);

    for a_size in [1usize, 2, 4] {
        let mut a = module_host.vec_znx_alloc(params.n, 1, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_backend = upload_vec_znx::<BE>(&a);

        for res_size in [1usize, 2, 4] {
            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.vec_znx_rsh_tmp_bytes(res_size));

            for k in 0..=(res_size * base2k) {
                let mut res = module_host.vec_znx_alloc(params.n, 1, res_size);
                res.fill_uniform(base2k, &mut source);
                let mut have_backend = upload_vec_znx::<BE>(&res);
                let mut want_backend = upload_vec_znx::<BE>(&res);

                module.vec_znx_rsh(
                    base2k,
                    k,
                    &mut vec_znx_backend_mut::<BE>(&mut have_backend),
                    0,
                    &vec_znx_backend_ref::<BE>(&a_backend),
                    0,
                    &mut scratch.borrow(),
                );

                // Oracle: the contract definition of `rsh`, spelled out.
                module.vec_znx_normalize(
                    &mut vec_znx_backend_mut::<BE>(&mut want_backend),
                    base2k,
                    res_size * base2k,
                    -(k as i64),
                    0,
                    &vec_znx_backend_ref::<BE>(&a_backend),
                    base2k,
                    0,
                    &mut scratch.borrow(),
                );

                assert_eq!(
                    download_vec_znx::<BE>(&want_backend),
                    download_vec_znx::<BE>(&have_backend),
                    "vec_znx_rsh: default body != normalize(offset) (a_size {a_size} res_size {res_size} k {k})"
                );
            }
        }
    }

    // Exactly one out-of-domain case: shifting further right than the
    // destination's own precision must annihilate the value.
    {
        let res_size: usize = 2;
        let k: usize = res_size * base2k + 3;
        let mut a = module_host.vec_znx_alloc(params.n, 1, 2);
        a.fill_uniform(base2k, &mut source);
        let a_backend = upload_vec_znx::<BE>(&a);
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.vec_znx_rsh_tmp_bytes(res_size));
        let mut res = module_host.vec_znx_alloc(params.n, 1, res_size);
        res.fill_uniform(base2k, &mut source);
        let mut have_backend = upload_vec_znx::<BE>(&res);
        module.vec_znx_rsh(
            base2k,
            k,
            &mut vec_znx_backend_mut::<BE>(&mut have_backend),
            0,
            &vec_znx_backend_ref::<BE>(&a_backend),
            0,
            &mut scratch.borrow(),
        );
        let have = download_vec_znx::<BE>(&have_backend);
        for limb in 0..res_size {
            assert!(
                have.at(0, limb).iter().all(|&c| c == 0),
                "vec_znx_rsh: shifting past the destination precision must give zero (res_size {res_size} k {k})"
            );
        }
    }
}

/// `vec_znx_lsh_add`: the OEP default body against a two-step oracle built on
/// the api, `tmp = lsh(a, k)` into a fresh buffer, then `res + (a << k)` with the
/// three-operand `vec_znx_add`. The two spread the same value over different
/// non-canonical digits, so the comparison is on canonical forms. The arena is
/// sized by the family's `vec_znx_lsh_tmp_bytes(res_size)`.
pub fn test_vec_znx_lsh_add_derived<BE: TestBackend + HalVecZnxImpl>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: VecZnxLshAdd<BE>
        + VecZnxLsh<BE>
        + VecZnxAdd<BE>
        + VecZnxLshTmpBytes
        + VecZnxNormalizeAssign<BE>
        + VecZnxNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let mut source: Source = Source::new([0u8; 32]);
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(params.n as u64);

    for a_size in [1usize, 2, 4] {
        let mut a = module_host.vec_znx_alloc(params.n, 1, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_backend = upload_vec_znx::<BE>(&a);

        for res_size in [1usize, 2, 4] {
            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                module
                    .vec_znx_lsh_tmp_bytes(res_size)
                    .max(module.vec_znx_normalize_tmp_bytes()),
            );

            for k in 0..=(res_size * base2k) {
                let mut res = module_host.vec_znx_alloc(params.n, 1, res_size);
                res.fill_uniform(base2k, &mut source);
                let orig_backend = upload_vec_znx::<BE>(&res);
                let mut have_backend = upload_vec_znx::<BE>(&res);
                let mut want_backend = upload_vec_znx::<BE>(&res);

                module.vec_znx_lsh_add(
                    base2k,
                    k,
                    &mut vec_znx_backend_mut::<BE>(&mut have_backend),
                    0,
                    &vec_znx_backend_ref::<BE>(&a_backend),
                    0,
                    &mut scratch.borrow(),
                );

                // Oracle: shift into a fresh buffer, then combine on the api.
                let mut tmp_backend = upload_vec_znx::<BE>(&module_host.vec_znx_alloc(params.n, 1, res_size));
                module.vec_znx_lsh(
                    base2k,
                    k,
                    &mut vec_znx_backend_mut::<BE>(&mut tmp_backend),
                    0,
                    &vec_znx_backend_ref::<BE>(&a_backend),
                    0,
                    &mut scratch.borrow(),
                );
                module.vec_znx_add(
                    &mut vec_znx_backend_mut::<BE>(&mut want_backend),
                    0,
                    &vec_znx_backend_ref::<BE>(&orig_backend),
                    0,
                    &vec_znx_backend_ref::<BE>(&tmp_backend),
                    0,
                );

                for backend in [&mut want_backend, &mut have_backend] {
                    module.vec_znx_normalize_assign(
                        base2k,
                        res_size * base2k,
                        0,
                        &mut vec_znx_backend_mut::<BE>(backend),
                        0,
                        &mut scratch.borrow(),
                    );
                }

                assert_eq!(
                    download_vec_znx::<BE>(&want_backend),
                    download_vec_znx::<BE>(&have_backend),
                    "vec_znx_lsh_add: default body != api oracle (a_size {a_size} res_size {res_size} k {k})"
                );
            }
        }
    }
}

/// `vec_znx_lsh_sub`: the OEP default body against a two-step oracle built on
/// the api, `tmp = lsh(a, k)` into a fresh buffer, then `res - (a << k)` with the
/// three-operand `vec_znx_sub`. The two spread the same value over different
/// non-canonical digits, so the comparison is on canonical forms. The arena is
/// sized by the family's `vec_znx_lsh_tmp_bytes(res_size)`.
pub fn test_vec_znx_lsh_sub_derived<BE: TestBackend + HalVecZnxImpl>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: VecZnxLshSub<BE>
        + VecZnxLsh<BE>
        + VecZnxSub<BE>
        + VecZnxLshTmpBytes
        + VecZnxNormalizeAssign<BE>
        + VecZnxNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let mut source: Source = Source::new([0u8; 32]);
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(params.n as u64);

    for a_size in [1usize, 2, 4] {
        let mut a = module_host.vec_znx_alloc(params.n, 1, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_backend = upload_vec_znx::<BE>(&a);

        for res_size in [1usize, 2, 4] {
            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                module
                    .vec_znx_lsh_tmp_bytes(res_size)
                    .max(module.vec_znx_normalize_tmp_bytes()),
            );

            for k in 0..=(res_size * base2k) {
                let mut res = module_host.vec_znx_alloc(params.n, 1, res_size);
                res.fill_uniform(base2k, &mut source);
                let orig_backend = upload_vec_znx::<BE>(&res);
                let mut have_backend = upload_vec_znx::<BE>(&res);
                let mut want_backend = upload_vec_znx::<BE>(&res);

                module.vec_znx_lsh_sub(
                    base2k,
                    k,
                    &mut vec_znx_backend_mut::<BE>(&mut have_backend),
                    0,
                    &vec_znx_backend_ref::<BE>(&a_backend),
                    0,
                    &mut scratch.borrow(),
                );

                // Oracle: shift into a fresh buffer, then combine on the api.
                let mut tmp_backend = upload_vec_znx::<BE>(&module_host.vec_znx_alloc(params.n, 1, res_size));
                module.vec_znx_lsh(
                    base2k,
                    k,
                    &mut vec_znx_backend_mut::<BE>(&mut tmp_backend),
                    0,
                    &vec_znx_backend_ref::<BE>(&a_backend),
                    0,
                    &mut scratch.borrow(),
                );
                module.vec_znx_sub(
                    &mut vec_znx_backend_mut::<BE>(&mut want_backend),
                    0,
                    &vec_znx_backend_ref::<BE>(&orig_backend),
                    0,
                    &vec_znx_backend_ref::<BE>(&tmp_backend),
                    0,
                );

                for backend in [&mut want_backend, &mut have_backend] {
                    module.vec_znx_normalize_assign(
                        base2k,
                        res_size * base2k,
                        0,
                        &mut vec_znx_backend_mut::<BE>(backend),
                        0,
                        &mut scratch.borrow(),
                    );
                }

                assert_eq!(
                    download_vec_znx::<BE>(&want_backend),
                    download_vec_znx::<BE>(&have_backend),
                    "vec_znx_lsh_sub: default body != api oracle (a_size {a_size} res_size {res_size} k {k})"
                );
            }
        }
    }
}

/// `vec_znx_rsh_add`: the OEP default body against a two-step oracle built on
/// the api, `tmp = rsh(a, k)` into a fresh buffer, then `res + (a >> k)` with the
/// three-operand `vec_znx_add`. The two spread the same value over different
/// non-canonical digits, so the comparison is on canonical forms. The arena is
/// sized by the family's `vec_znx_rsh_tmp_bytes(res_size)`.
pub fn test_vec_znx_rsh_add_derived<BE: TestBackend + HalVecZnxImpl>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: VecZnxRshAdd<BE>
        + VecZnxRsh<BE>
        + VecZnxAdd<BE>
        + VecZnxRshTmpBytes
        + VecZnxNormalizeAssign<BE>
        + VecZnxNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let mut source: Source = Source::new([0u8; 32]);
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(params.n as u64);

    for a_size in [1usize, 2, 4] {
        let mut a = module_host.vec_znx_alloc(params.n, 1, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_backend = upload_vec_znx::<BE>(&a);

        for res_size in [1usize, 2, 4] {
            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                module
                    .vec_znx_rsh_tmp_bytes(res_size)
                    .max(module.vec_znx_normalize_tmp_bytes()),
            );

            for k in 0..=(res_size * base2k) {
                let mut res = module_host.vec_znx_alloc(params.n, 1, res_size);
                res.fill_uniform(base2k, &mut source);
                let orig_backend = upload_vec_znx::<BE>(&res);
                let mut have_backend = upload_vec_znx::<BE>(&res);
                let mut want_backend = upload_vec_znx::<BE>(&res);

                module.vec_znx_rsh_add(
                    base2k,
                    k,
                    &mut vec_znx_backend_mut::<BE>(&mut have_backend),
                    0,
                    &vec_znx_backend_ref::<BE>(&a_backend),
                    0,
                    &mut scratch.borrow(),
                );

                // Oracle: shift into a fresh buffer, then combine on the api.
                let mut tmp_backend = upload_vec_znx::<BE>(&module_host.vec_znx_alloc(params.n, 1, res_size));
                module.vec_znx_rsh(
                    base2k,
                    k,
                    &mut vec_znx_backend_mut::<BE>(&mut tmp_backend),
                    0,
                    &vec_znx_backend_ref::<BE>(&a_backend),
                    0,
                    &mut scratch.borrow(),
                );
                module.vec_znx_add(
                    &mut vec_znx_backend_mut::<BE>(&mut want_backend),
                    0,
                    &vec_znx_backend_ref::<BE>(&orig_backend),
                    0,
                    &vec_znx_backend_ref::<BE>(&tmp_backend),
                    0,
                );

                for backend in [&mut want_backend, &mut have_backend] {
                    module.vec_znx_normalize_assign(
                        base2k,
                        res_size * base2k,
                        0,
                        &mut vec_znx_backend_mut::<BE>(backend),
                        0,
                        &mut scratch.borrow(),
                    );
                }

                assert_eq!(
                    download_vec_znx::<BE>(&want_backend),
                    download_vec_znx::<BE>(&have_backend),
                    "vec_znx_rsh_add: default body != api oracle (a_size {a_size} res_size {res_size} k {k})"
                );
            }
        }
    }
}

/// `vec_znx_rsh_sub`: the OEP default body against a two-step oracle built on
/// the api, `tmp = rsh(a, k)` into a fresh buffer, then `res - (a >> k)` with the
/// three-operand `vec_znx_sub`. The two spread the same value over different
/// non-canonical digits, so the comparison is on canonical forms. The arena is
/// sized by the family's `vec_znx_rsh_tmp_bytes(res_size)`.
pub fn test_vec_znx_rsh_sub_derived<BE: TestBackend + HalVecZnxImpl>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: VecZnxRshSub<BE>
        + VecZnxRsh<BE>
        + VecZnxSub<BE>
        + VecZnxRshTmpBytes
        + VecZnxNormalizeAssign<BE>
        + VecZnxNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let mut source: Source = Source::new([0u8; 32]);
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(params.n as u64);

    for a_size in [1usize, 2, 4] {
        let mut a = module_host.vec_znx_alloc(params.n, 1, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_backend = upload_vec_znx::<BE>(&a);

        for res_size in [1usize, 2, 4] {
            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                module
                    .vec_znx_rsh_tmp_bytes(res_size)
                    .max(module.vec_znx_normalize_tmp_bytes()),
            );

            for k in 0..=(res_size * base2k) {
                let mut res = module_host.vec_znx_alloc(params.n, 1, res_size);
                res.fill_uniform(base2k, &mut source);
                let orig_backend = upload_vec_znx::<BE>(&res);
                let mut have_backend = upload_vec_znx::<BE>(&res);
                let mut want_backend = upload_vec_znx::<BE>(&res);

                module.vec_znx_rsh_sub(
                    base2k,
                    k,
                    &mut vec_znx_backend_mut::<BE>(&mut have_backend),
                    0,
                    &vec_znx_backend_ref::<BE>(&a_backend),
                    0,
                    &mut scratch.borrow(),
                );

                // Oracle: shift into a fresh buffer, then combine on the api.
                let mut tmp_backend = upload_vec_znx::<BE>(&module_host.vec_znx_alloc(params.n, 1, res_size));
                module.vec_znx_rsh(
                    base2k,
                    k,
                    &mut vec_znx_backend_mut::<BE>(&mut tmp_backend),
                    0,
                    &vec_znx_backend_ref::<BE>(&a_backend),
                    0,
                    &mut scratch.borrow(),
                );
                module.vec_znx_sub(
                    &mut vec_znx_backend_mut::<BE>(&mut want_backend),
                    0,
                    &vec_znx_backend_ref::<BE>(&orig_backend),
                    0,
                    &vec_znx_backend_ref::<BE>(&tmp_backend),
                    0,
                );

                for backend in [&mut want_backend, &mut have_backend] {
                    module.vec_znx_normalize_assign(
                        base2k,
                        res_size * base2k,
                        0,
                        &mut vec_znx_backend_mut::<BE>(backend),
                        0,
                        &mut scratch.borrow(),
                    );
                }

                assert_eq!(
                    download_vec_znx::<BE>(&want_backend),
                    download_vec_znx::<BE>(&have_backend),
                    "vec_znx_rsh_sub: default body != api oracle (a_size {a_size} res_size {res_size} k {k})"
                );
            }
        }
    }
}

/// `vec_znx_lsh_assign`: the OEP default body *and* whatever `module` actually
/// dispatches to (a backend override, where there is one) against a two-step
/// oracle built on the api, `tmp = lsh(a, k)` into a fresh buffer, then
/// `vec_znx_copy` back. The default body is compared on canonical forms, the
/// dispatched op bit for bit: at equal sizes the shift truncates nothing, so an
/// override has no freedom left. `k` runs past `res_size * base2k`, where both
/// have to leave the column zero. The arena is sized by
/// `vec_znx_lsh_tmp_bytes(res_size)`.
pub fn test_vec_znx_lsh_assign_derived<BE: TestBackend + HalVecZnxImpl>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: VecZnxLshAssign<BE>
        + VecZnxLsh<BE>
        + VecZnxCopy<BE>
        + VecZnxLshTmpBytes
        + VecZnxNormalizeAssign<BE>
        + VecZnxNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let mut source: Source = Source::new([0u8; 32]);
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(params.n as u64);

    for res_size in [1usize, 2, 4] {
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .vec_znx_lsh_tmp_bytes(res_size)
                .max(module.vec_znx_normalize_tmp_bytes()),
        );

        for k in 0..=(res_size * base2k + base2k) {
            let mut res = module_host.vec_znx_alloc(params.n, 1, res_size);
            res.fill_uniform(base2k, &mut source);
            let orig_backend = upload_vec_znx::<BE>(&res);
            let mut have_backend = upload_vec_znx::<BE>(&res);
            let mut got_backend = upload_vec_znx::<BE>(&res);
            let mut want_backend = upload_vec_znx::<BE>(&res);

            crate::oep::vec_znx_lsh_assign_derived::<BE>(
                module,
                base2k,
                k,
                &mut vec_znx_backend_mut::<BE>(&mut have_backend),
                0,
                &mut scratch.borrow(),
            );

            module.vec_znx_lsh_assign(
                base2k,
                k,
                &mut vec_znx_backend_mut::<BE>(&mut got_backend),
                0,
                &mut scratch.borrow(),
            );

            // Oracle: shift into a fresh buffer, then copy back.
            let mut tmp_backend = upload_vec_znx::<BE>(&module_host.vec_znx_alloc(params.n, 1, res_size));
            module.vec_znx_lsh(
                base2k,
                k,
                &mut vec_znx_backend_mut::<BE>(&mut tmp_backend),
                0,
                &vec_znx_backend_ref::<BE>(&orig_backend),
                0,
                &mut scratch.borrow(),
            );
            module.vec_znx_copy(
                &mut vec_znx_backend_mut::<BE>(&mut want_backend),
                0,
                &vec_znx_backend_ref::<BE>(&tmp_backend),
                0,
            );

            assert_eq!(
                download_vec_znx::<BE>(&want_backend),
                download_vec_znx::<BE>(&got_backend),
                "vec_znx_lsh_assign: dispatched op != api oracle, bit for bit (res_size {res_size} k {k})"
            );

            for backend in [&mut want_backend, &mut have_backend] {
                module.vec_znx_normalize_assign(
                    base2k,
                    res_size * base2k,
                    0,
                    &mut vec_znx_backend_mut::<BE>(backend),
                    0,
                    &mut scratch.borrow(),
                );
            }

            assert_eq!(
                download_vec_znx::<BE>(&want_backend),
                download_vec_znx::<BE>(&have_backend),
                "vec_znx_lsh_assign: default body != api oracle (res_size {res_size} k {k})"
            );
        }
    }
}

/// `vec_znx_rsh_assign`: the OEP default body against a two-step oracle built on
/// the api, `tmp = rsh(a, k)` into a fresh buffer, then `vec_znx_copy` back.
/// Compared on canonical forms; the arena is sized by
/// `vec_znx_rsh_tmp_bytes(res_size)`.
pub fn test_vec_znx_rsh_assign_derived<BE: TestBackend + HalVecZnxImpl>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: VecZnxRshAssign<BE>
        + VecZnxRsh<BE>
        + VecZnxCopy<BE>
        + VecZnxRshTmpBytes
        + VecZnxNormalizeAssign<BE>
        + VecZnxNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let mut source: Source = Source::new([0u8; 32]);
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(params.n as u64);

    for res_size in [1usize, 2, 4] {
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .vec_znx_rsh_tmp_bytes(res_size)
                .max(module.vec_znx_normalize_tmp_bytes()),
        );

        for k in 0..=(res_size * base2k) {
            let mut res = module_host.vec_znx_alloc(params.n, 1, res_size);
            res.fill_uniform(base2k, &mut source);
            let orig_backend = upload_vec_znx::<BE>(&res);
            let mut have_backend = upload_vec_znx::<BE>(&res);
            let mut want_backend = upload_vec_znx::<BE>(&res);

            module.vec_znx_rsh_assign(
                base2k,
                k,
                &mut vec_znx_backend_mut::<BE>(&mut have_backend),
                0,
                &mut scratch.borrow(),
            );

            // Oracle: shift into a fresh buffer, then copy back.
            let mut tmp_backend = upload_vec_znx::<BE>(&module_host.vec_znx_alloc(params.n, 1, res_size));
            module.vec_znx_rsh(
                base2k,
                k,
                &mut vec_znx_backend_mut::<BE>(&mut tmp_backend),
                0,
                &vec_znx_backend_ref::<BE>(&orig_backend),
                0,
                &mut scratch.borrow(),
            );
            module.vec_znx_copy(
                &mut vec_znx_backend_mut::<BE>(&mut want_backend),
                0,
                &vec_znx_backend_ref::<BE>(&tmp_backend),
                0,
            );

            for backend in [&mut want_backend, &mut have_backend] {
                module.vec_znx_normalize_assign(
                    base2k,
                    res_size * base2k,
                    0,
                    &mut vec_znx_backend_mut::<BE>(backend),
                    0,
                    &mut scratch.borrow(),
                );
            }

            assert_eq!(
                download_vec_znx::<BE>(&want_backend),
                download_vec_znx::<BE>(&have_backend),
                "vec_znx_rsh_assign: default body != api oracle (res_size {res_size} k {k})"
            );
        }
    }
}

/// `vec_znx_mul_xp_minus_one`: the OEP default body versus an oracle
/// hand-built from the public api traits (`rotate` then `sub_assign`).
pub fn test_vec_znx_mul_xp_minus_one_derived<BE: TestBackend + HalVecZnxImpl>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: VecZnxMulXpMinusOne<BE> + VecZnxRotate<BE> + VecZnxSubAssign<BE>,
{
    let base2k: usize = params.base2k;
    let mut source: Source = Source::new([0u8; 32]);
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(params.n as u64);

    for size in [1usize, 2, 4] {
        let mut a = module_host.vec_znx_alloc(params.n, 1, size);
        a.fill_uniform(base2k, &mut source);
        let a_backend = upload_vec_znx::<BE>(&a);

        for p in [1i64, 5, -3, params.n as i64] {
            let mut res = module_host.vec_znx_alloc(params.n, 1, size);
            res.fill_uniform(base2k, &mut source);
            let mut have_backend = upload_vec_znx::<BE>(&res);
            let mut want_backend = upload_vec_znx::<BE>(&res);

            crate::oep::vec_znx_mul_xp_minus_one_derived::<BE>(
                module,
                p,
                &mut vec_znx_backend_mut::<BE>(&mut have_backend),
                0,
                &vec_znx_backend_ref::<BE>(&a_backend),
                0,
            );

            // Oracle: the same decomposition, through the public api traits.
            module.vec_znx_rotate(
                p,
                &mut vec_znx_backend_mut::<BE>(&mut want_backend),
                0,
                &vec_znx_backend_ref::<BE>(&a_backend),
                0,
            );
            module.vec_znx_sub_assign(
                &mut vec_znx_backend_mut::<BE>(&mut want_backend),
                0,
                &vec_znx_backend_ref::<BE>(&a_backend),
                0,
            );

            assert_eq!(
                download_vec_znx::<BE>(&want_backend),
                download_vec_znx::<BE>(&have_backend),
                "vec_znx_mul_xp_minus_one: default body != hand-built oracle (size {size} p {p})"
            );
        }
    }
}

/// `vec_znx_mul_xp_minus_one_assign`: the OEP default body, a rotate into a
/// whole `res.size()`-limb `VecZnx` temporary, then a copy back, and whatever
/// `module` actually dispatches to (a backend override, where there is one),
/// both versus the out-of-place `vec_znx_mul_xp_minus_one` on the same input,
/// an independent decomposition of the same map, so both comparisons are bit
/// for bit. The default body gets an arena sized by its own
/// `_tmp_bytes_derived`, the dispatched op one sized by the api `_tmp_bytes`,
/// which an override may have shrunk.
pub fn test_vec_znx_mul_xp_minus_one_assign_derived<BE: TestBackend + HalVecZnxImpl>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: VecZnxMulXpMinusOne<BE> + VecZnxMulXpMinusOneAssign<BE> + VecZnxMulXpMinusOneAssignTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let mut source: Source = Source::new([0u8; 32]);
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(params.n as u64);

    for size in [1usize, 2, 4] {
        // Each body gets its own `_tmp_bytes(size)` alone, so one that
        // under-reports its scratch panics here.
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(crate::oep::vec_znx_mul_xp_minus_one_assign_tmp_bytes_derived::<
            BE,
        >(module, size));
        let mut scratch_api: ScratchOwned<BE> = ScratchOwned::alloc(module.vec_znx_mul_xp_minus_one_assign_tmp_bytes(size));

        for p in [1i64, 5, -3, params.n as i64] {
            let mut a = module_host.vec_znx_alloc(params.n, 1, size);
            a.fill_uniform(base2k, &mut source);
            let a_backend = upload_vec_znx::<BE>(&a);

            let mut have_backend = upload_vec_znx::<BE>(&a);
            crate::oep::vec_znx_mul_xp_minus_one_assign_derived::<BE>(
                module,
                p,
                &mut vec_znx_backend_mut::<BE>(&mut have_backend),
                0,
                &mut scratch.borrow(),
            );

            let mut got_backend = upload_vec_znx::<BE>(&a);
            module.vec_znx_mul_xp_minus_one_assign(
                p,
                &mut vec_znx_backend_mut::<BE>(&mut got_backend),
                0,
                &mut scratch_api.borrow(),
            );

            // Oracle: the out-of-place op, which is a different decomposition
            // (whole-vector rotate then subtract) of the same map.
            let mut want_backend = upload_vec_znx::<BE>(&a);
            module.vec_znx_mul_xp_minus_one(
                p,
                &mut vec_znx_backend_mut::<BE>(&mut want_backend),
                0,
                &vec_znx_backend_ref::<BE>(&a_backend),
                0,
            );

            assert_eq!(
                download_vec_znx::<BE>(&want_backend),
                download_vec_znx::<BE>(&have_backend),
                "vec_znx_mul_xp_minus_one_assign: default body != out-of-place form (size {size} p {p})"
            );

            assert_eq!(
                download_vec_znx::<BE>(&want_backend),
                download_vec_znx::<BE>(&got_backend),
                "vec_znx_mul_xp_minus_one_assign: dispatched op != out-of-place form (size {size} p {p})"
            );
        }
    }
}

/// `vec_znx_add_scalar_assign`: the OEP default body (an `add_assign` on the
/// one-limb window) versus a whole-vector `add_assign` against a `VecZnx` that
/// carries the scalar in limb `res_limb`, an independent oracle, bit for bit.
pub fn test_vec_znx_add_scalar_assign_derived<BE: TestBackend + HalVecZnxImpl>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: VecZnxAddScalarAssign<BE> + VecZnxAddAssign<BE>,
{
    let base2k: usize = params.base2k;
    let n: usize = params.n;
    let mut source: Source = Source::new([0u8; 32]);
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(n as u64);

    let mut scalar = module_host.scalar_znx_alloc(params.n, 1);
    scalar.fill_uniform(base2k, &mut source);
    let scalar_backend = upload_scalar_znx::<BE>(&scalar);

    for res_size in [1usize, 2, 4] {
        for res_limb in 0..res_size {
            let mut res = module_host.vec_znx_alloc(params.n, 1, res_size);
            res.fill_uniform(base2k, &mut source);
            let mut have_backend = upload_vec_znx::<BE>(&res);
            let mut want_backend = upload_vec_znx::<BE>(&res);

            crate::oep::vec_znx_add_scalar_assign_derived::<BE>(
                module,
                &mut vec_znx_backend_mut::<BE>(&mut have_backend),
                0,
                res_limb,
                &scalar_znx_backend_ref::<BE>(&scalar_backend),
                0,
            );

            // Oracle: the scalar lifted into a full-width `VecZnx` whose only
            // non-zero limb is `res_limb`, added with the plain `add_assign`.
            let mut lifted = module_host.vec_znx_alloc(params.n, 1, res_size);
            lifted.at_mut(0, res_limb).copy_from_slice(scalar.at(0, 0));
            let lifted_backend = upload_vec_znx::<BE>(&lifted);
            module.vec_znx_add_assign(
                &mut vec_znx_backend_mut::<BE>(&mut want_backend),
                0,
                &vec_znx_backend_ref::<BE>(&lifted_backend),
                0,
            );

            assert_eq!(
                download_vec_znx::<BE>(&want_backend),
                download_vec_znx::<BE>(&have_backend),
                "vec_znx_add_scalar_assign: default body != hand-built oracle (res_size {res_size} res_limb {res_limb})"
            );
        }
    }
}

/// `vec_znx_big_add_small`: the scratch-free OEP default against an
/// independent oracle, `from_small(b)` then `add_assign(a)`, written out
/// through the public api traits (a different call path from the free
/// function under test, even though it is the same decomposition).
/// The same oracle also pins whatever `module` dispatches to, which is a
/// fused backend override under ruling R10.
pub fn test_vec_znx_big_add_small_derived<BE: TestBackend + HalVecZnxBigImpl>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: ModuleN
        + VecZnxBigAlloc<BE>
        + VecZnxBigFromSmall<BE>
        + VecZnxBigAddAssign<BE>
        + VecZnxBigAddSmall<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let mut source: Source = Source::new([0u8; 32]);
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(params.n as u64);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.vec_znx_big_normalize_tmp_bytes());

    // (big operand size, small operand size, res size): res longer than both,
    // small shorter than big, small longer than big, res shorter than both, res
    // past both with one operand ending early.
    // The small operand is the sparse-capable slot: the last two cases give it
    // half the module's degree, the degree the default body has to embed. Under
    // a module too small to halve below the degree floor they stay dense.
    let sparse_n: usize = if params.n >= 16 { params.n / 2 } else { params.n };
    for (a_size, b_size, res_size, small_n) in [
        (3usize, 3usize, 3usize, params.n),
        (4, 2, 3, params.n),
        (2, 4, 3, params.n),
        (3, 3, 5, params.n),
        (4, 4, 2, params.n),
        (4, 2, 5, params.n),
        (3, 3, 3, sparse_n),
        (4, 2, 5, sparse_n),
    ] {
        let mut a_small = module_host.vec_znx_alloc(params.n, 1, a_size);
        a_small.fill_uniform(base2k, &mut source);
        let mut b: VecZnxOwned<i64> = VecZnx::alloc(small_n, 1, b_size);
        b.fill_uniform(base2k, &mut source);
        let a_backend = upload_vec_znx::<BE>(&a_small);
        let b_backend = upload_vec_znx::<BE>(&b);

        // `a` as a VecZnxBig, through the basis promotion.
        let mut a_big = module.vec_znx_big_alloc(params.n, 1, a_size);
        module.vec_znx_big_from_small(&mut a_big.to_backend_mut(), 0, &vec_znx_backend_ref::<BE>(&a_backend), 0);

        let mut want_big = module.vec_znx_big_alloc(params.n, 1, res_size);
        let mut have_big = module.vec_znx_big_alloc(params.n, 1, res_size);
        let mut got_big = module.vec_znx_big_alloc(params.n, 1, res_size);

        // Both destinations start on a non-zero sentinel: the limbs the
        // operation has to zero are only asserted if they did not read back
        // zero by accident.
        let mut sentinel = module_host.vec_znx_alloc(params.n, 1, res_size);
        for limb in 0..res_size {
            sentinel.at_mut(0, limb).fill(1i64 << (base2k - 2));
        }
        let sentinel_backend = upload_vec_znx::<BE>(&sentinel);
        for dst in [&mut want_big, &mut have_big, &mut got_big] {
            module.vec_znx_big_from_small(&mut dst.to_backend_mut(), 0, &vec_znx_backend_ref::<BE>(&sentinel_backend), 0);
        }

        crate::oep::vec_znx_big_add_small_derived::<BE>(
            module,
            &mut want_big.to_backend_mut(),
            0,
            &a_big.to_backend_ref(),
            0,
            &vec_znx_backend_ref::<BE>(&b_backend),
            0,
        );

        // Oracle: `from_small(b)` then `add_assign(a)`, through the public api
        // traits rather than the free function under test.
        module.vec_znx_big_from_small(&mut have_big.to_backend_mut(), 0, &vec_znx_backend_ref::<BE>(&b_backend), 0);
        module.vec_znx_big_add_assign(&mut have_big.to_backend_mut(), 0, &a_big.to_backend_ref(), 0);

        let want_template = module_host.vec_znx_alloc(params.n, 1, res_size);
        let have_template = module_host.vec_znx_alloc(params.n, 1, res_size);
        let mut want_backend = upload_vec_znx::<BE>(&want_template);
        let mut have_backend = upload_vec_znx::<BE>(&have_template);
        module.vec_znx_big_normalize(
            &mut vec_znx_backend_mut::<BE>(&mut want_backend),
            base2k,
            res_size * base2k,
            0,
            0,
            &want_big.to_backend_ref(),
            base2k,
            0,
            &mut scratch.borrow(),
        );
        module.vec_znx_big_normalize(
            &mut vec_znx_backend_mut::<BE>(&mut have_backend),
            base2k,
            res_size * base2k,
            0,
            0,
            &have_big.to_backend_ref(),
            base2k,
            0,
            &mut scratch.borrow(),
        );

        // Whatever `module` dispatches to: the same OEP default unless the
        // backend kept a fused override (ruling R10), which must agree with it.
        module.vec_znx_big_add_small(
            &mut got_big.to_backend_mut(),
            0,
            &a_big.to_backend_ref(),
            0,
            &vec_znx_backend_ref::<BE>(&b_backend),
            0,
        );
        let got_template = module_host.vec_znx_alloc(params.n, 1, res_size);
        let mut got_backend = upload_vec_znx::<BE>(&got_template);
        module.vec_znx_big_normalize(
            &mut vec_znx_backend_mut::<BE>(&mut got_backend),
            base2k,
            res_size * base2k,
            0,
            0,
            &got_big.to_backend_ref(),
            base2k,
            0,
            &mut scratch.borrow(),
        );

        assert_eq!(
            download_vec_znx::<BE>(&want_backend),
            download_vec_znx::<BE>(&have_backend),
            "vec_znx_big_add_small: default body != independent oracle (a {a_size} b {b_size} res {res_size} small_n {small_n})"
        );

        assert_eq!(
            download_vec_znx::<BE>(&want_backend),
            download_vec_znx::<BE>(&got_backend),
            "vec_znx_big_add_small: dispatched op != independent oracle (a {a_size} b {b_size} res {res_size} small_n {small_n})"
        );
    }
}

/// `vec_znx_big_sub_small_a`: the scratch-free OEP default (`res = a - b`,
/// `a` the coefficient-domain operand) against an independent oracle,
/// `from_small(a)` then `sub_assign(b)`, through the public api traits.
/// The same oracle also pins whatever `module` dispatches to, which is a
/// fused backend override under ruling R10.
pub fn test_vec_znx_big_sub_small_a_derived<BE: TestBackend + HalVecZnxBigImpl>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: ModuleN
        + VecZnxBigAlloc<BE>
        + VecZnxBigFromSmall<BE>
        + VecZnxBigSubAssign<BE>
        + VecZnxBigSubSmallA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let mut source: Source = Source::new([0u8; 32]);
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(params.n as u64);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.vec_znx_big_normalize_tmp_bytes());

    // (small operand size, big operand size, res size): res longer than both,
    // small shorter than big, small longer than big, res shorter than both, res
    // past both with one operand ending early.
    // The small operand is the sparse-capable slot: the last two cases give it
    // half the module's degree, the degree the default body has to embed. Under
    // a module too small to halve below the degree floor they stay dense.
    let sparse_n: usize = if params.n >= 16 { params.n / 2 } else { params.n };
    for (a_size, b_size, res_size, small_n) in [
        (3usize, 3usize, 3usize, params.n),
        (2, 4, 3, params.n),
        (4, 2, 3, params.n),
        (3, 3, 5, params.n),
        (4, 4, 2, params.n),
        (4, 2, 5, params.n),
        (3, 3, 3, sparse_n),
        (4, 2, 5, sparse_n),
    ] {
        let mut a_small: VecZnxOwned<i64> = VecZnx::alloc(small_n, 1, a_size);
        a_small.fill_uniform(base2k, &mut source);
        let mut b = module_host.vec_znx_alloc(params.n, 1, b_size);
        b.fill_uniform(base2k, &mut source);
        let a_backend = upload_vec_znx::<BE>(&a_small);
        let b_backend = upload_vec_znx::<BE>(&b);

        // `b` as a VecZnxBig, through the basis promotion.
        let mut b_big = module.vec_znx_big_alloc(params.n, 1, b_size);
        module.vec_znx_big_from_small(&mut b_big.to_backend_mut(), 0, &vec_znx_backend_ref::<BE>(&b_backend), 0);

        let mut want_big = module.vec_znx_big_alloc(params.n, 1, res_size);
        let mut have_big = module.vec_znx_big_alloc(params.n, 1, res_size);
        let mut got_big = module.vec_znx_big_alloc(params.n, 1, res_size);

        // Both destinations start on a non-zero sentinel: the limbs the
        // operation has to zero are only asserted if they did not read back
        // zero by accident.
        let mut sentinel = module_host.vec_znx_alloc(params.n, 1, res_size);
        for limb in 0..res_size {
            sentinel.at_mut(0, limb).fill(1i64 << (base2k - 2));
        }
        let sentinel_backend = upload_vec_znx::<BE>(&sentinel);
        for dst in [&mut want_big, &mut have_big, &mut got_big] {
            module.vec_znx_big_from_small(&mut dst.to_backend_mut(), 0, &vec_znx_backend_ref::<BE>(&sentinel_backend), 0);
        }

        crate::oep::vec_znx_big_sub_small_a_derived::<BE>(
            module,
            &mut want_big.to_backend_mut(),
            0,
            &vec_znx_backend_ref::<BE>(&a_backend),
            0,
            &b_big.to_backend_ref(),
            0,
        );

        // Oracle: `from_small(a)` then `sub_assign(b)`, through the public
        // api traits rather than the free function under test.
        module.vec_znx_big_from_small(&mut have_big.to_backend_mut(), 0, &vec_znx_backend_ref::<BE>(&a_backend), 0);
        module.vec_znx_big_sub_assign(&mut have_big.to_backend_mut(), 0, &b_big.to_backend_ref(), 0);

        let want_template = module_host.vec_znx_alloc(params.n, 1, res_size);
        let have_template = module_host.vec_znx_alloc(params.n, 1, res_size);
        let mut want_backend = upload_vec_znx::<BE>(&want_template);
        let mut have_backend = upload_vec_znx::<BE>(&have_template);
        module.vec_znx_big_normalize(
            &mut vec_znx_backend_mut::<BE>(&mut want_backend),
            base2k,
            res_size * base2k,
            0,
            0,
            &want_big.to_backend_ref(),
            base2k,
            0,
            &mut scratch.borrow(),
        );
        module.vec_znx_big_normalize(
            &mut vec_znx_backend_mut::<BE>(&mut have_backend),
            base2k,
            res_size * base2k,
            0,
            0,
            &have_big.to_backend_ref(),
            base2k,
            0,
            &mut scratch.borrow(),
        );

        // Whatever `module` dispatches to: the same OEP default unless the
        // backend kept a fused override (ruling R10), which must agree with it.
        module.vec_znx_big_sub_small_a(
            &mut got_big.to_backend_mut(),
            0,
            &vec_znx_backend_ref::<BE>(&a_backend),
            0,
            &b_big.to_backend_ref(),
            0,
        );
        let got_template = module_host.vec_znx_alloc(params.n, 1, res_size);
        let mut got_backend = upload_vec_znx::<BE>(&got_template);
        module.vec_znx_big_normalize(
            &mut vec_znx_backend_mut::<BE>(&mut got_backend),
            base2k,
            res_size * base2k,
            0,
            0,
            &got_big.to_backend_ref(),
            base2k,
            0,
            &mut scratch.borrow(),
        );

        assert_eq!(
            download_vec_znx::<BE>(&want_backend),
            download_vec_znx::<BE>(&have_backend),
            "vec_znx_big_sub_small_a: default body != independent oracle (a {a_size} b {b_size} res {res_size} small_n {small_n})"
        );

        assert_eq!(
            download_vec_znx::<BE>(&want_backend),
            download_vec_znx::<BE>(&got_backend),
            "vec_znx_big_sub_small_a: dispatched op != independent oracle (a {a_size} b {b_size} res {res_size} small_n {small_n})"
        );
    }
}

/// `vec_znx_big_sub_small_b`: the scratch-free OEP default (`res = a - b`,
/// `b` the coefficient-domain operand) against an independent oracle,
/// `from_small(b)` then `sub_negate_assign(a)`, through the public api traits.
/// The same oracle also pins whatever `module` dispatches to, which is a
/// fused backend override under ruling R10.
pub fn test_vec_znx_big_sub_small_b_derived<BE: TestBackend + HalVecZnxBigImpl>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: ModuleN
        + VecZnxBigAlloc<BE>
        + VecZnxBigFromSmall<BE>
        + VecZnxBigSubNegateAssign<BE>
        + VecZnxBigSubSmallB<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let mut source: Source = Source::new([0u8; 32]);
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(params.n as u64);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.vec_znx_big_normalize_tmp_bytes());

    // (big operand size, small operand size, res size): res longer than both,
    // small shorter than big, small longer than big, res shorter than both, res
    // past both with one operand ending early.
    // The small operand is the sparse-capable slot: the last two cases give it
    // half the module's degree, the degree the default body has to embed. Under
    // a module too small to halve below the degree floor they stay dense.
    let sparse_n: usize = if params.n >= 16 { params.n / 2 } else { params.n };
    for (a_size, b_size, res_size, small_n) in [
        (3usize, 3usize, 3usize, params.n),
        (4, 2, 3, params.n),
        (2, 4, 3, params.n),
        (3, 3, 5, params.n),
        (4, 4, 2, params.n),
        (4, 2, 5, params.n),
        (3, 3, 3, sparse_n),
        (4, 2, 5, sparse_n),
    ] {
        let mut a_small = module_host.vec_znx_alloc(params.n, 1, a_size);
        a_small.fill_uniform(base2k, &mut source);
        let mut b: VecZnxOwned<i64> = VecZnx::alloc(small_n, 1, b_size);
        b.fill_uniform(base2k, &mut source);
        let a_backend = upload_vec_znx::<BE>(&a_small);
        let b_backend = upload_vec_znx::<BE>(&b);

        // `a` as a VecZnxBig, through the basis promotion.
        let mut a_big = module.vec_znx_big_alloc(params.n, 1, a_size);
        module.vec_znx_big_from_small(&mut a_big.to_backend_mut(), 0, &vec_znx_backend_ref::<BE>(&a_backend), 0);

        let mut want_big = module.vec_znx_big_alloc(params.n, 1, res_size);
        let mut have_big = module.vec_znx_big_alloc(params.n, 1, res_size);
        let mut got_big = module.vec_znx_big_alloc(params.n, 1, res_size);

        // Both destinations start on a non-zero sentinel: the limbs the
        // operation has to zero are only asserted if they did not read back
        // zero by accident.
        let mut sentinel = module_host.vec_znx_alloc(params.n, 1, res_size);
        for limb in 0..res_size {
            sentinel.at_mut(0, limb).fill(1i64 << (base2k - 2));
        }
        let sentinel_backend = upload_vec_znx::<BE>(&sentinel);
        for dst in [&mut want_big, &mut have_big, &mut got_big] {
            module.vec_znx_big_from_small(&mut dst.to_backend_mut(), 0, &vec_znx_backend_ref::<BE>(&sentinel_backend), 0);
        }

        crate::oep::vec_znx_big_sub_small_b_derived::<BE>(
            module,
            &mut want_big.to_backend_mut(),
            0,
            &a_big.to_backend_ref(),
            0,
            &vec_znx_backend_ref::<BE>(&b_backend),
            0,
        );

        // Oracle: `from_small(b)` then `sub_negate_assign(a)` (`res = a - res`),
        // through the public api traits rather than the free function under test.
        module.vec_znx_big_from_small(&mut have_big.to_backend_mut(), 0, &vec_znx_backend_ref::<BE>(&b_backend), 0);
        module.vec_znx_big_sub_negate_assign(&mut have_big.to_backend_mut(), 0, &a_big.to_backend_ref(), 0);

        let want_template = module_host.vec_znx_alloc(params.n, 1, res_size);
        let have_template = module_host.vec_znx_alloc(params.n, 1, res_size);
        let mut want_backend = upload_vec_znx::<BE>(&want_template);
        let mut have_backend = upload_vec_znx::<BE>(&have_template);
        module.vec_znx_big_normalize(
            &mut vec_znx_backend_mut::<BE>(&mut want_backend),
            base2k,
            res_size * base2k,
            0,
            0,
            &want_big.to_backend_ref(),
            base2k,
            0,
            &mut scratch.borrow(),
        );
        module.vec_znx_big_normalize(
            &mut vec_znx_backend_mut::<BE>(&mut have_backend),
            base2k,
            res_size * base2k,
            0,
            0,
            &have_big.to_backend_ref(),
            base2k,
            0,
            &mut scratch.borrow(),
        );

        // Whatever `module` dispatches to: the same OEP default unless the
        // backend kept a fused override (ruling R10), which must agree with it.
        module.vec_znx_big_sub_small_b(
            &mut got_big.to_backend_mut(),
            0,
            &a_big.to_backend_ref(),
            0,
            &vec_znx_backend_ref::<BE>(&b_backend),
            0,
        );
        let got_template = module_host.vec_znx_alloc(params.n, 1, res_size);
        let mut got_backend = upload_vec_znx::<BE>(&got_template);
        module.vec_znx_big_normalize(
            &mut vec_znx_backend_mut::<BE>(&mut got_backend),
            base2k,
            res_size * base2k,
            0,
            0,
            &got_big.to_backend_ref(),
            base2k,
            0,
            &mut scratch.borrow(),
        );

        assert_eq!(
            download_vec_znx::<BE>(&want_backend),
            download_vec_znx::<BE>(&have_backend),
            "vec_znx_big_sub_small_b: default body != independent oracle (a {a_size} b {b_size} res {res_size} small_n {small_n})"
        );

        assert_eq!(
            download_vec_znx::<BE>(&want_backend),
            download_vec_znx::<BE>(&got_backend),
            "vec_znx_big_sub_small_b: dispatched op != independent oracle (a {a_size} b {b_size} res {res_size} small_n {small_n})"
        );
    }
}

/// `vec_znx_idft_normalize_consume`: the OEP default body against an
/// independent oracle, `idft_apply_tmpa` into a `VecZnxBig`, the optional
/// `big_add_small_assign`, then `big_normalize`, driven through the public
/// api traits. The op clobbers its input, so each side transforms its own
/// copy of the same host operand. Both arenas are sized by their own side's
/// `_tmp_bytes` alone, so a default that under-reports its scratch panics.
pub fn test_vec_znx_idft_normalize_consume_derived<BE: TestBackend + HalVecZnxDftImpl>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: ModuleN
        + VecZnxAlloc<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let mut source: Source = Source::new([0u8; 32]);
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(params.n as u64);
    let mut oracle_scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.vec_znx_big_normalize_tmp_bytes());

    // (a_size, res_size): equal, res shorter, res longer.
    for (a_size, res_size) in [(3usize, 3usize), (4, 2), (2, 4)] {
        let mut a = module_host.vec_znx_alloc(params.n, 1, a_size);
        a.fill_uniform(base2k, &mut source);
        let mut addend = module_host.vec_znx_alloc(params.n, 1, a_size);
        addend.fill_uniform(base2k, &mut source);
        let a_backend = upload_vec_znx::<BE>(&a);
        let addend_backend = upload_vec_znx::<BE>(&addend);
        let addend_ref = vec_znx_backend_ref::<BE>(&addend_backend);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            crate::oep::vec_znx_idft_normalize_consume_tmp_bytes_derived::<BE>(module, res_size, a_size),
        );

        for with_addend in [false, true] {
            let addend_arg = with_addend.then_some((&addend_ref, 0));

            // The op consumes its input, so each side gets its own transform.
            let mut have_dft = module.vec_znx_dft_alloc(params.n, 1, a_size);
            let mut want_dft = module.vec_znx_dft_alloc(params.n, 1, a_size);
            for dft in [&mut have_dft, &mut want_dft] {
                module.vec_znx_dft_apply(1, 0, &mut dft.to_backend_mut(), 0, &vec_znx_backend_ref::<BE>(&a_backend), 0);
            }

            let res_template = module_host.vec_znx_alloc(params.n, 1, res_size);
            let mut have_backend = upload_vec_znx::<BE>(&res_template);
            let mut want_backend = upload_vec_znx::<BE>(&res_template);

            crate::oep::vec_znx_idft_normalize_consume_derived::<BE>(
                module,
                &mut vec_znx_backend_mut::<BE>(&mut have_backend),
                base2k,
                res_size * base2k,
                0,
                &mut have_dft.to_backend_mut(),
                0,
                base2k,
                addend_arg,
                &mut scratch.borrow(),
            );

            // Oracle: idft into a BIG, the optional small add, then normalize,
            // through the public api traits rather than the OEP dispatch.
            let mut big = module.vec_znx_big_alloc(params.n, 1, a_size);
            module.vec_znx_idft_apply_tmpa(&mut big.to_backend_mut(), 0, &mut want_dft.to_backend_mut(), 0);
            if with_addend {
                module.vec_znx_big_add_small_assign(&mut big.to_backend_mut(), 0, &addend_ref, 0);
            }
            module.vec_znx_big_normalize(
                &mut vec_znx_backend_mut::<BE>(&mut want_backend),
                base2k,
                res_size * base2k,
                0,
                0,
                &big.to_backend_ref(),
                base2k,
                0,
                &mut oracle_scratch.borrow(),
            );

            assert_eq!(
                download_vec_znx::<BE>(&want_backend),
                download_vec_znx::<BE>(&have_backend),
                "vec_znx_idft_normalize_consume: default body != independent oracle (a_size {a_size} res_size {res_size} \
                 addend {with_addend})"
            );
        }
    }
}

/// `vec_znx_dft_automorphism_add_with_plan`: the OEP default body against an
/// independent oracle, `automorphism_with_plan` into a `min(res, a)`-limb
/// temporary, then `vec_znx_dft_add_assign`, through the public api traits.
/// DFT-domain results are compared after `idft` and `big_normalize`. The
/// arena is sized by the op's own `_tmp_bytes` alone.
pub fn test_vec_znx_dft_automorphism_add_with_plan_derived<BE: TestBackend + HalVecZnxDftImpl>(
    params: &TestParams,
    module: &Module<BE>,
) where
    Module<BE>: ModuleN
        + VecZnxAlloc<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftAutomorphism<BE>
        + VecZnxDftAutomorphismPlan<BE, Plan = BE::AutomorphismPlan>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let mut source: Source = Source::new([1u8; 32]);
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(params.n as u64);
    let mut oracle_scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.vec_znx_big_normalize_tmp_bytes());

    let cols: usize = 2;

    for (res_size, a_size) in [(3usize, 3usize), (2, 4), (4, 2)] {
        let size: usize = res_size.min(a_size);
        let mut a = module_host.vec_znx_alloc(params.n, cols, a_size);
        let mut seed = module_host.vec_znx_alloc(params.n, cols, res_size);
        a.fill_uniform(base2k, &mut source);
        seed.fill_uniform(base2k, &mut source);
        let a_backend = upload_vec_znx::<BE>(&a);
        let seed_backend = upload_vec_znx::<BE>(&seed);

        let mut a_dft = module.vec_znx_dft_alloc(params.n, cols, a_size);
        for col in 0..cols {
            module.vec_znx_dft_apply(
                1,
                0,
                &mut a_dft.to_backend_mut(),
                col,
                &vec_znx_backend_ref::<BE>(&a_backend),
                col,
            );
        }

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            crate::oep::vec_znx_dft_automorphism_add_with_plan_tmp_bytes_derived::<BE>(module, res_size, a_size),
        );

        for p in [1i64, 5, -3] {
            let plan = module.vec_znx_dft_automorphism_plan(params.n, p);

            // Every column of both destinations carries the same seed, so a
            // default body that wrote outside `col` would diverge from the
            // oracle there too.
            for col in 0..cols {
                let mut have_dft = module.vec_znx_dft_alloc(params.n, cols, res_size);
                let mut want_dft = module.vec_znx_dft_alloc(params.n, cols, res_size);
                for dft in [&mut have_dft, &mut want_dft] {
                    for seeded in 0..cols {
                        module.vec_znx_dft_apply(
                            1,
                            0,
                            &mut dft.to_backend_mut(),
                            seeded,
                            &vec_znx_backend_ref::<BE>(&seed_backend),
                            seeded,
                        );
                    }
                }

                crate::oep::vec_znx_dft_automorphism_add_with_plan_derived::<BE>(
                    module,
                    &plan,
                    &mut have_dft.to_backend_mut(),
                    col,
                    &a_dft.to_backend_ref(),
                    col,
                    &mut scratch.borrow(),
                );

                // Oracle: automorphism into a `min(res, a)`-limb temporary, then
                // the DFT-domain accumulation, through the public api traits.
                let mut rot = module.vec_znx_dft_alloc(params.n, 1, size);
                module.vec_znx_dft_automorphism_with_plan(&plan, &mut rot.to_backend_mut(), 0, &a_dft.to_backend_ref(), col);
                module.vec_znx_dft_add_assign(&mut want_dft.to_backend_mut(), col, &rot.to_backend_ref(), 0);

                let res_template = module_host.vec_znx_alloc(params.n, cols, res_size);
                let mut have_backend = upload_vec_znx::<BE>(&res_template);
                let mut want_backend = upload_vec_znx::<BE>(&res_template);
                let mut big = module.vec_znx_big_alloc(params.n, 1, res_size);
                for (dft, out) in [(&mut have_dft, &mut have_backend), (&mut want_dft, &mut want_backend)] {
                    for normalized in 0..cols {
                        module.vec_znx_idft_apply_tmpa(&mut big.to_backend_mut(), 0, &mut dft.to_backend_mut(), normalized);
                        module.vec_znx_big_normalize(
                            &mut vec_znx_backend_mut::<BE>(out),
                            base2k,
                            res_size * base2k,
                            0,
                            normalized,
                            &big.to_backend_ref(),
                            base2k,
                            0,
                            &mut oracle_scratch.borrow(),
                        );
                    }
                }

                assert_eq!(
                    download_vec_znx::<BE>(&want_backend),
                    download_vec_znx::<BE>(&have_backend),
                    "vec_znx_dft_automorphism_add_with_plan: default body != independent oracle (p {p} col {col} \
                     res_size {res_size} a_size {a_size})"
                );
            }
        }
    }
}

/// `vec_znx_dft_automorphism`: the OEP default body against the plan form it
/// abbreviates, `vec_znx_dft_automorphism_plan` then
/// `vec_znx_dft_automorphism_with_plan` through the public api traits, and the
/// api entry itself, which routes through the OEP. DFT-domain results are
/// compared after `idft` and `big_normalize`.
pub fn test_vec_znx_dft_automorphism_derived<BE: TestBackend + HalVecZnxDftImpl>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: ModuleN
        + VecZnxAlloc<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftAutomorphism<BE>
        + VecZnxDftAutomorphismPlan<BE, Plan = BE::AutomorphismPlan>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let mut source: Source = Source::new([1u8; 32]);
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(params.n as u64);
    let mut oracle_scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.vec_znx_big_normalize_tmp_bytes());
    let cols: usize = 2;
    for (res_size, a_size) in [(3usize, 3usize), (2, 4), (4, 2)] {
        let mut a = module_host.vec_znx_alloc(params.n, cols, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_backend = upload_vec_znx::<BE>(&a);
        let mut a_dft = module.vec_znx_dft_alloc(params.n, cols, a_size);
        for col in 0..cols {
            module.vec_znx_dft_apply(
                1,
                0,
                &mut a_dft.to_backend_mut(),
                col,
                &vec_znx_backend_ref::<BE>(&a_backend),
                col,
            );
        }
        for p in [1i64, 5, -3] {
            for col in 0..cols {
                let mut have_dft = module.vec_znx_dft_alloc(params.n, cols, res_size);
                let mut api_dft = module.vec_znx_dft_alloc(params.n, cols, res_size);
                let mut want_dft = module.vec_znx_dft_alloc(params.n, cols, res_size);
                crate::oep::vec_znx_dft_automorphism_derived::<BE>(
                    module,
                    p,
                    &mut have_dft.to_backend_mut(),
                    col,
                    &a_dft.to_backend_ref(),
                    col,
                );
                module.vec_znx_dft_automorphism(p, &mut api_dft.to_backend_mut(), col, &a_dft.to_backend_ref(), col);
                let plan = module.vec_znx_dft_automorphism_plan(params.n, p);
                module.vec_znx_dft_automorphism_with_plan(
                    &plan,
                    &mut want_dft.to_backend_mut(),
                    col,
                    &a_dft.to_backend_ref(),
                    col,
                );
                let res_template = module_host.vec_znx_alloc(params.n, cols, res_size);
                let mut have_backend = upload_vec_znx::<BE>(&res_template);
                let mut api_backend = upload_vec_znx::<BE>(&res_template);
                let mut want_backend = upload_vec_znx::<BE>(&res_template);
                let mut big = module.vec_znx_big_alloc(params.n, 1, res_size);
                for (dft, out) in [
                    (&mut have_dft, &mut have_backend),
                    (&mut api_dft, &mut api_backend),
                    (&mut want_dft, &mut want_backend),
                ] {
                    module.vec_znx_idft_apply_tmpa(&mut big.to_backend_mut(), 0, &mut dft.to_backend_mut(), col);
                    module.vec_znx_big_normalize(
                        &mut vec_znx_backend_mut::<BE>(out),
                        base2k,
                        res_size * base2k,
                        0,
                        col,
                        &big.to_backend_ref(),
                        base2k,
                        0,
                        &mut oracle_scratch.borrow(),
                    );
                }
                let want = download_vec_znx::<BE>(&want_backend);
                assert_eq!(
                    want,
                    download_vec_znx::<BE>(&have_backend),
                    "vec_znx_dft_automorphism: default body != plan form (p {p} col {col} res_size {res_size} a_size {a_size})"
                );
                assert_eq!(
                    want,
                    download_vec_znx::<BE>(&api_backend),
                    "vec_znx_dft_automorphism: api entry != plan form (p {p} col {col} res_size {res_size} a_size {a_size})"
                );
            }
        }
    }
}

/// `svp_apply_dft`: the OEP default body against an independent oracle,
/// `vec_znx_dft_apply` into a fresh `VecZnxDft`, then `svp_apply_dft_to_dft`,
/// through the public api traits. DFT-domain results are compared after
/// `idft` and `big_normalize`. The arena is sized by the op's own
/// `_tmp_bytes` alone.
pub fn test_svp_apply_dft_derived<BE: TestBackend + HalSvpImpl>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: ModuleN
        + VecZnxAlloc<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + SvpPPolAlloc<BE>
        + SvpPrepare<BE>
        + SvpApplyDftToDft<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let mut source: Source = Source::new([2u8; 32]);
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(params.n as u64);
    let mut oracle_scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.vec_znx_big_normalize_tmp_bytes());

    let cols: usize = 2;

    let mut scalar = module_host.scalar_znx_alloc(params.n, cols);
    scalar.fill_uniform(base2k, &mut source);
    let scalar_backend = upload_scalar_znx::<BE>(&scalar);

    let mut svp: SvpPPolOwned<BE> = module.svp_ppol_alloc(params.n, cols, PrepareHint::Reuse);
    for col in 0..cols {
        module.svp_prepare(
            &mut svp.to_backend_mut(),
            col,
            &scalar_znx_backend_ref::<BE>(&scalar_backend),
            col,
        );
    }

    for (res_size, b_size) in [(3usize, 3usize), (2, 4), (4, 2)] {
        let mut b = module_host.vec_znx_alloc(params.n, cols, b_size);
        b.fill_uniform(base2k, &mut source);
        let b_backend = upload_vec_znx::<BE>(&b);
        let mut seed = module_host.vec_znx_alloc(params.n, cols, res_size);
        seed.fill_uniform(base2k, &mut source);
        let seed_backend = upload_vec_znx::<BE>(&seed);

        let mut scratch: ScratchOwned<BE> =
            ScratchOwned::alloc(crate::oep::svp_apply_dft_tmp_bytes_derived::<BE>(module, b_size));

        // Every column starts on the same seed on both sides, so the limbs the
        // product has to zero are asserted and a write outside `col` shows up.
        for col in 0..cols {
            let mut have_dft = module.vec_znx_dft_alloc(params.n, cols, res_size);
            let mut want_dft = module.vec_znx_dft_alloc(params.n, cols, res_size);
            for dft in [&mut have_dft, &mut want_dft] {
                for seeded in 0..cols {
                    module.vec_znx_dft_apply(
                        1,
                        0,
                        &mut dft.to_backend_mut(),
                        seeded,
                        &vec_znx_backend_ref::<BE>(&seed_backend),
                        seeded,
                    );
                }
            }

            crate::oep::svp_apply_dft_derived::<BE>(
                module,
                &mut have_dft.to_backend_mut(),
                col,
                &svp.to_backend_ref(),
                col,
                &vec_znx_backend_ref::<BE>(&b_backend),
                col,
                &mut scratch.borrow(),
            );

            // Oracle: transform b into a fresh VecZnxDft, then svp_apply_dft_to_dft,
            // through the public api traits.
            let mut b_dft = module.vec_znx_dft_alloc(params.n, 1, b_size);
            module.vec_znx_dft_apply(
                1,
                0,
                &mut b_dft.to_backend_mut(),
                0,
                &vec_znx_backend_ref::<BE>(&b_backend),
                col,
            );
            module.svp_apply_dft_to_dft(
                &mut want_dft.to_backend_mut(),
                col,
                &svp.to_backend_ref(),
                col,
                &b_dft.to_backend_ref(),
                0,
            );

            let res_template = module_host.vec_znx_alloc(params.n, cols, res_size);
            let mut have_backend = upload_vec_znx::<BE>(&res_template);
            let mut want_backend = upload_vec_znx::<BE>(&res_template);
            let mut big = module.vec_znx_big_alloc(params.n, 1, res_size);
            for (dft, out) in [(&mut have_dft, &mut have_backend), (&mut want_dft, &mut want_backend)] {
                for normalized in 0..cols {
                    module.vec_znx_idft_apply_tmpa(&mut big.to_backend_mut(), 0, &mut dft.to_backend_mut(), normalized);
                    module.vec_znx_big_normalize(
                        &mut vec_znx_backend_mut::<BE>(out),
                        base2k,
                        res_size * base2k,
                        0,
                        normalized,
                        &big.to_backend_ref(),
                        base2k,
                        0,
                        &mut oracle_scratch.borrow(),
                    );
                }
            }

            assert_eq!(
                download_vec_znx::<BE>(&want_backend),
                download_vec_znx::<BE>(&have_backend),
                "svp_apply_dft: default body != independent oracle (col {col} res_size {res_size} b_size {b_size})"
            );
        }
    }
}

/// `cnv_apply_dft_add`: the derived free function's decomposition versus an
/// explicit two-step oracle (`cnv_apply_dft` into a fresh `VecZnxDft`, then
/// `vec_znx_dft_add_assign`), and versus the backend's own
/// `module.cnv_apply_dft_add`, the inherited default on backends with no
/// fused override, the fused kernel on backends that have one. DFT-domain
/// results are compared after `vec_znx_idft_apply_tmpa` +
/// `vec_znx_big_normalize`, as the rest of this suite states DFT-domain
/// equality. The arena for the derived call is sized only by
/// `cnv_apply_dft_add_tmp_bytes_derived`, so a decomposition that
/// under-reports its own scratch panics here.
pub fn test_cnv_apply_dft_add_derived<BE: TestBackend + HalConvolutionImpl>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: ModuleN
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let cols: usize = 2;
    let a_size: usize = 15;
    let b_size: usize = 15;
    let res_size: usize = a_size + b_size;

    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(params.n as u64);
    let (a_prep, b_prep, mut scratch) =
        prepare_convolution_operands::<BE>(module, &module_host, params.n, cols, a_size, b_size, 17);

    let mut res_oracle = module.vec_znx_dft_alloc(params.n, cols, res_size);
    let mut res_derived = module.vec_znx_dft_alloc(params.n, cols, res_size);
    let mut res_module = module.vec_znx_dft_alloc(params.n, cols, res_size);
    let mut fresh = module.vec_znx_dft_alloc(params.n, 1, res_size);
    let mut big = module.vec_znx_big_alloc(params.n, 1, res_size);

    // Written at column 1 too: covers the column-interleaved `VecZnxDft` indexing.
    for res_col in 0..cols {
        for cnv_offset in [0usize, 1usize] {
            // Identical deterministic initial accumulator content for all three paths.
            for res in [&mut res_oracle, &mut res_derived, &mut res_module] {
                module.cnv_apply_dft(
                    0,
                    &mut res.to_backend_mut(),
                    res_col,
                    &a_prep.to_backend_ref(),
                    0,
                    &b_prep.to_backend_ref(),
                    0,
                    &mut scratch.borrow(),
                );
            }

            // Oracle: the contract definition, spelled out through the api traits.
            module.cnv_apply_dft(
                cnv_offset,
                &mut fresh.to_backend_mut(),
                0,
                &a_prep.to_backend_ref(),
                1,
                &b_prep.to_backend_ref(),
                0,
                &mut scratch.borrow(),
            );
            module.vec_znx_dft_add_assign(&mut res_oracle.to_backend_mut(), res_col, &fresh.to_backend_ref(), 0);

            // The derived free function, with an arena sized only by its own sibling.
            let mut derived_scratch: ScratchOwned<BE> = ScratchOwned::alloc(cnv_apply_dft_add_tmp_bytes_derived::<BE>(
                module, cnv_offset, res_size, a_size, b_size,
            ));
            cnv_apply_dft_add_derived::<BE>(
                module,
                cnv_offset,
                &mut res_derived.to_backend_mut(),
                res_col,
                &a_prep.to_backend_ref(),
                1,
                &b_prep.to_backend_ref(),
                0,
                &mut derived_scratch.borrow(),
            );

            // The backend's own dispatch through the OEP method.
            module.cnv_apply_dft_add(
                cnv_offset,
                &mut res_module.to_backend_mut(),
                res_col,
                &a_prep.to_backend_ref(),
                1,
                &b_prep.to_backend_ref(),
                0,
                &mut scratch.borrow(),
            );

            let want = normalize_dft_column::<BE>(module, base2k, &mut big, &mut res_oracle, res_col, &mut scratch);
            let have_derived = normalize_dft_column::<BE>(module, base2k, &mut big, &mut res_derived, res_col, &mut scratch);
            let have_module = normalize_dft_column::<BE>(module, base2k, &mut big, &mut res_module, res_col, &mut scratch);
            assert_eq!(
                want, have_derived,
                "cnv_apply_dft_add: derived decomposition != two-step oracle (res_col {res_col} cnv_offset {cnv_offset})"
            );
            assert_eq!(
                want, have_module,
                "cnv_apply_dft_add: module dispatch != two-step oracle (res_col {res_col} cnv_offset {cnv_offset})"
            );
        }
    }
}

/// `cnv_apply_dft_sum`: the derived free function's decomposition versus an
/// explicit oracle (the first term through `cnv_apply_dft`, every further
/// term through `cnv_apply_dft_add`), and versus the backend's own
/// `module.cnv_apply_dft_sum`, the inherited default on backends with no
/// fused override, the fused kernel on backends that have one. One-, two- and
/// three-term sums are swept, so the first-term overwrite and the folding of
/// the rest are both covered. DFT-domain results are compared after
/// `vec_znx_idft_apply_tmpa` + `vec_znx_big_normalize`, as the rest of this
/// suite states DFT-domain equality. The arena for the derived call is sized
/// only by `cnv_apply_dft_sum_tmp_bytes_derived`, so a decomposition that
/// under-reports its own scratch panics here.
pub fn test_cnv_apply_dft_sum_derived<BE: TestBackend + HalConvolutionImpl>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: ModuleN
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let cols: usize = 2;
    let a_size: usize = 15;
    let b_size: usize = 15;
    let res_size: usize = a_size + b_size;
    // Written at column 1: covers the column-interleaved `VecZnxDft` indexing.
    let res_col: usize = 1;

    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(params.n as u64);
    let (a_prep, b_prep, mut scratch) =
        prepare_convolution_operands::<BE>(module, &module_host, params.n, cols, a_size, b_size, 17);

    let mut res_oracle = module.vec_znx_dft_alloc(params.n, 2, res_size);
    let mut res_derived = module.vec_znx_dft_alloc(params.n, 2, res_size);
    let mut res_module = module.vec_znx_dft_alloc(params.n, 2, res_size);
    let mut big = module.vec_znx_big_alloc(params.n, 1, res_size);

    // Terms mixing operand columns, like one BSGS giant step; the sweep takes
    // the first one, two and three of them.
    let term_cols: [(usize, usize); 3] = [(0, 0), (1, 1), (0, 1)];

    for n_terms in 1..=term_cols.len() {
        for cnv_offset in [0usize, 1usize] {
            // Oracle: the contract definition, spelled out through the api traits.
            for (idx, &(a_col, b_col)) in term_cols[..n_terms].iter().enumerate() {
                if idx == 0 {
                    module.cnv_apply_dft(
                        cnv_offset,
                        &mut res_oracle.to_backend_mut(),
                        res_col,
                        &a_prep.to_backend_ref(),
                        a_col,
                        &b_prep.to_backend_ref(),
                        b_col,
                        &mut scratch.borrow(),
                    );
                } else {
                    module.cnv_apply_dft_add(
                        cnv_offset,
                        &mut res_oracle.to_backend_mut(),
                        res_col,
                        &a_prep.to_backend_ref(),
                        a_col,
                        &b_prep.to_backend_ref(),
                        b_col,
                        &mut scratch.borrow(),
                    );
                }
            }

            let terms: Vec<CnvDftAccTerm<'_, BE>> = term_cols[..n_terms]
                .iter()
                .map(|&(a_col, b_col)| CnvDftAccTerm {
                    a: a_prep.to_backend_ref(),
                    a_col,
                    b: b_prep.to_backend_ref(),
                    b_col,
                })
                .collect();

            // The derived free function, with an arena sized only by its own sibling.
            let mut derived_scratch: ScratchOwned<BE> = ScratchOwned::alloc(cnv_apply_dft_sum_tmp_bytes_derived::<BE>(
                module, cnv_offset, res_size, a_size, b_size,
            ));
            cnv_apply_dft_sum_derived::<BE>(
                module,
                cnv_offset,
                &mut res_derived.to_backend_mut(),
                res_col,
                &terms,
                &mut derived_scratch.borrow(),
            );

            // The backend's own dispatch through the OEP method, with an arena
            // sized only by the api `_tmp_bytes` (a fused override may need
            // more than the shared convolution arena carries).
            let mut module_scratch: ScratchOwned<BE> =
                ScratchOwned::alloc(module.cnv_apply_dft_sum_tmp_bytes(cnv_offset, res_size, a_size, b_size));
            module.cnv_apply_dft_sum(
                cnv_offset,
                &mut res_module.to_backend_mut(),
                res_col,
                &terms,
                &mut module_scratch.borrow(),
            );

            let want = normalize_dft_column::<BE>(module, base2k, &mut big, &mut res_oracle, res_col, &mut scratch);
            let have_derived = normalize_dft_column::<BE>(module, base2k, &mut big, &mut res_derived, res_col, &mut scratch);
            let have_module = normalize_dft_column::<BE>(module, base2k, &mut big, &mut res_module, res_col, &mut scratch);
            assert_eq!(
                want, have_derived,
                "cnv_apply_dft_sum: derived decomposition != per-term oracle (n_terms {n_terms} cnv_offset {cnv_offset})"
            );
            assert_eq!(
                want, have_module,
                "cnv_apply_dft_sum: module dispatch != per-term oracle (n_terms {n_terms} cnv_offset {cnv_offset})"
            );
        }
    }
}

/// `cnv_pairwise_apply_dft`: the derived free function versus an oracle that
/// writes out the expansion, one `cnv_apply_dft` when `i == j`, and the four
/// cross terms as `cnv_apply_dft` + three `cnv_apply_dft_add` otherwise, and
/// versus the backend's own `module.cnv_pairwise_apply_dft`.
pub fn test_cnv_pairwise_apply_dft_derived<BE: TestBackend + HalConvolutionImpl>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: ModuleN
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let cols: usize = 2;
    let a_size: usize = 15;
    let b_size: usize = 15;
    let res_size: usize = a_size + b_size;
    let res_col: usize = 1;

    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(params.n as u64);
    let (a_prep, b_prep, mut scratch) =
        prepare_convolution_operands::<BE>(module, &module_host, params.n, cols, a_size, b_size, 17);

    let mut res_oracle = module.vec_znx_dft_alloc(params.n, 2, res_size);
    let mut res_derived = module.vec_znx_dft_alloc(params.n, 2, res_size);
    let mut res_module = module.vec_znx_dft_alloc(params.n, 2, res_size);
    let mut big = module.vec_znx_big_alloc(params.n, 1, res_size);

    for (i, j) in [(0usize, 0usize), (0, 1), (1, 0)] {
        for cnv_offset in [0usize, 1usize] {
            // Oracle: the expansion written out through the api traits.
            module.cnv_apply_dft(
                cnv_offset,
                &mut res_oracle.to_backend_mut(),
                res_col,
                &a_prep.to_backend_ref(),
                i,
                &b_prep.to_backend_ref(),
                i,
                &mut scratch.borrow(),
            );
            if i != j {
                for (a_col, b_col) in [(i, j), (j, i), (j, j)] {
                    module.cnv_apply_dft_add(
                        cnv_offset,
                        &mut res_oracle.to_backend_mut(),
                        res_col,
                        &a_prep.to_backend_ref(),
                        a_col,
                        &b_prep.to_backend_ref(),
                        b_col,
                        &mut scratch.borrow(),
                    );
                }
            }

            let mut derived_scratch: ScratchOwned<BE> = ScratchOwned::alloc(cnv_pairwise_apply_dft_tmp_bytes_derived::<BE>(
                module, cnv_offset, res_size, a_size, b_size,
            ));
            cnv_pairwise_apply_dft_derived::<BE>(
                module,
                cnv_offset,
                &mut res_derived.to_backend_mut(),
                res_col,
                &a_prep.to_backend_ref(),
                &b_prep.to_backend_ref(),
                i,
                j,
                &mut derived_scratch.borrow(),
            );

            module.cnv_pairwise_apply_dft(
                cnv_offset,
                &mut res_module.to_backend_mut(),
                res_col,
                &a_prep.to_backend_ref(),
                &b_prep.to_backend_ref(),
                i,
                j,
                &mut scratch.borrow(),
            );

            let want = normalize_dft_column::<BE>(module, base2k, &mut big, &mut res_oracle, res_col, &mut scratch);
            let have_derived = normalize_dft_column::<BE>(module, base2k, &mut big, &mut res_derived, res_col, &mut scratch);
            let have_module = normalize_dft_column::<BE>(module, base2k, &mut big, &mut res_module, res_col, &mut scratch);
            assert_eq!(
                want, have_derived,
                "cnv_pairwise_apply_dft: derived decomposition != expanded oracle (i {i} j {j} cnv_offset {cnv_offset})"
            );
            assert_eq!(
                want, have_module,
                "cnv_pairwise_apply_dft: module dispatch != expanded oracle (i {i} j {j} cnv_offset {cnv_offset})"
            );
        }
    }
}

/// `cnv_prepare_self`: the derived free function versus separate
/// `cnv_prepare_left` / `cnv_prepare_right` calls, and versus the backend's
/// own `module.cnv_prepare_self`. The prepared bytes are opaque (spec
/// decision 4), so equality is stated on an observable: the same
/// `cnv_apply_dft` is run against each of the three prepared pairs and the
/// normalized products are compared.
pub fn test_cnv_prepare_self_derived<BE: TestBackend + HalConvolutionImpl>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: ModuleN
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + VecZnxAlloc<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let cols: usize = 2;
    let a_size: usize = 15;
    let res_size: usize = a_size + a_size;
    let res_col: usize = 1;
    let mut source: Source = Source::new([0u8; 32]);

    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(params.n as u64);
    let mut a = module_host.vec_znx_alloc(params.n, cols, a_size);
    a.fill_uniform(17, &mut source);
    let a_backend = upload_vec_znx::<BE>(&a);
    let a_ref = vec_znx_backend_ref::<BE>(&a_backend);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .cnv_prepare_left_tmp_bytes(a_size, a_size)
            .max(module.cnv_prepare_right_tmp_bytes(a_size, a_size))
            .max(module.cnv_prepare_self_tmp_bytes(a_size, a_size))
            .max(module.cnv_apply_dft_tmp_bytes(0, res_size, a_size, a_size))
            .max(module.vec_znx_big_normalize_tmp_bytes()),
    );

    // Oracle: the two prepares, driven separately.
    let mut left_oracle: CnvPVecLOwned<BE> = module.cnv_pvec_left_alloc(params.n, cols, a_size, PrepareHint::Reuse);
    let mut right_oracle: CnvPVecROwned<BE> = module.cnv_pvec_right_alloc(params.n, cols, a_size, PrepareHint::Reuse);
    module.cnv_prepare_left(&mut left_oracle.to_backend_mut(), &a_ref, &mut scratch.borrow());
    module.cnv_prepare_right(&mut right_oracle.to_backend_mut(), &a_ref, &mut scratch.borrow());

    // The derived free function, with an arena sized only by its own sibling.
    let mut left_derived: CnvPVecLOwned<BE> = module.cnv_pvec_left_alloc(params.n, cols, a_size, PrepareHint::Reuse);
    let mut right_derived: CnvPVecROwned<BE> = module.cnv_pvec_right_alloc(params.n, cols, a_size, PrepareHint::Reuse);
    let mut derived_scratch: ScratchOwned<BE> =
        ScratchOwned::alloc(cnv_prepare_self_tmp_bytes_derived::<BE>(module, a_size, a_size));
    cnv_prepare_self_derived::<BE>(
        module,
        &mut left_derived.to_backend_mut(),
        &mut right_derived.to_backend_mut(),
        &a_ref,
        &mut derived_scratch.borrow(),
    );

    // The backend's own dispatch through the OEP method.
    let mut left_module: CnvPVecLOwned<BE> = module.cnv_pvec_left_alloc(params.n, cols, a_size, PrepareHint::Reuse);
    let mut right_module: CnvPVecROwned<BE> = module.cnv_pvec_right_alloc(params.n, cols, a_size, PrepareHint::Reuse);
    module.cnv_prepare_self(
        &mut left_module.to_backend_mut(),
        &mut right_module.to_backend_mut(),
        &a_ref,
        &mut scratch.borrow(),
    );

    let mut res_dft = module.vec_znx_dft_alloc(params.n, 2, res_size);
    let mut big = module.vec_znx_big_alloc(params.n, 1, res_size);
    let mut product =
        |left: &CnvPVecLOwned<BE>, right: &CnvPVecROwned<BE>, col: usize, cnv_offset: usize, scratch: &mut ScratchOwned<BE>| {
            module.cnv_apply_dft(
                cnv_offset,
                &mut res_dft.to_backend_mut(),
                res_col,
                &left.to_backend_ref(),
                col,
                &right.to_backend_ref(),
                col,
                &mut scratch.borrow(),
            );
            normalize_dft_column::<BE>(module, base2k, &mut big, &mut res_dft, res_col, scratch)
        };

    for col in 0..cols {
        for cnv_offset in [0usize, 1usize] {
            let want = product(&left_oracle, &right_oracle, col, cnv_offset, &mut scratch);
            let have_derived = product(&left_derived, &right_derived, col, cnv_offset, &mut scratch);
            let have_module = product(&left_module, &right_module, col, cnv_offset, &mut scratch);
            assert_eq!(
                want, have_derived,
                "cnv_prepare_self: derived decomposition != separate prepares (col {col} cnv_offset {cnv_offset})"
            );
            assert_eq!(
                want, have_module,
                "cnv_prepare_self: module dispatch != separate prepares (col {col} cnv_offset {cnv_offset})"
            );
        }
    }
}

/// `cnv_by_const_apply_add`: the derived free function versus an explicit
/// two-step oracle (`cnv_by_const_apply` into a fresh `VecZnxBig`, then
/// `vec_znx_big_add_assign`), and versus the backend's own
/// `module.cnv_by_const_apply_add`. The result never enters the DFT domain,
/// so the comparison is on the `vec_znx_big_normalize` output alone.
pub fn test_cnv_by_const_apply_add_derived<BE: TestBackend + HalConvolutionImpl>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: ModuleN
        + Convolution<BE>
        + VecZnxAlloc<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxBigAddAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let a_cols: usize = 2;
    let a_size: usize = 15;
    let b_size: usize = 3;
    let res_size: usize = a_size + b_size;
    let mut source: Source = Source::new([0u8; 32]);

    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(params.n as u64);
    let mut a = module_host.vec_znx_alloc(params.n, a_cols, a_size);
    a.fill_uniform(17, &mut source);
    let mut b = module_host.vec_znx_alloc(params.n, 1, b_size);
    b.fill_uniform(base2k, &mut source);
    let a_backend = upload_vec_znx::<BE>(&a);
    let b_backend = upload_vec_znx::<BE>(&b);
    let a_ref = vec_znx_backend_ref::<BE>(&a_backend);
    let b_ref = vec_znx_backend_ref::<BE>(&b_backend);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .cnv_by_const_apply_tmp_bytes(0, res_size, a_size, b_size)
            .max(module.cnv_by_const_apply_add_tmp_bytes(0, res_size, a_size, b_size))
            .max(module.vec_znx_big_normalize_tmp_bytes()),
    );

    let mut acc_oracle = module.vec_znx_big_alloc(params.n, 1, res_size);
    let mut acc_derived = module.vec_znx_big_alloc(params.n, 1, res_size);
    let mut acc_module = module.vec_znx_big_alloc(params.n, 1, res_size);
    let mut fresh = module.vec_znx_big_alloc(params.n, 1, res_size);

    for cnv_offset in [0usize, 1usize] {
        // Identical deterministic initial accumulator content for all three paths.
        for acc in [&mut acc_oracle, &mut acc_derived, &mut acc_module] {
            module.cnv_by_const_apply(
                0,
                &mut acc.to_backend_mut(),
                0,
                &a_ref,
                0,
                &b_ref,
                0,
                0,
                &mut scratch.borrow(),
            );
        }

        module.cnv_by_const_apply(
            cnv_offset,
            &mut fresh.to_backend_mut(),
            0,
            &a_ref,
            1,
            &b_ref,
            0,
            1,
            &mut scratch.borrow(),
        );
        module.vec_znx_big_add_assign(&mut acc_oracle.to_backend_mut(), 0, &fresh.to_backend_ref(), 0);

        let mut derived_scratch: ScratchOwned<BE> = ScratchOwned::alloc(cnv_by_const_apply_add_tmp_bytes_derived::<BE>(
            module, cnv_offset, res_size, a_size, b_size,
        ));
        cnv_by_const_apply_add_derived::<BE>(
            module,
            cnv_offset,
            &mut acc_derived.to_backend_mut(),
            0,
            &a_ref,
            1,
            &b_ref,
            0,
            1,
            &mut derived_scratch.borrow(),
        );

        module.cnv_by_const_apply_add(
            cnv_offset,
            &mut acc_module.to_backend_mut(),
            0,
            &a_ref,
            1,
            &b_ref,
            0,
            1,
            &mut scratch.borrow(),
        );

        let normalize = |big: &crate::layouts::VecZnxBigOwned<BE>, scratch: &mut ScratchOwned<BE>| {
            let template = module_host.vec_znx_alloc(params.n, 1, res_size);
            let mut backend = upload_vec_znx::<BE>(&template);
            module.vec_znx_big_normalize(
                &mut vec_znx_backend_mut::<BE>(&mut backend),
                base2k,
                res_size * base2k,
                0,
                0,
                &big.to_backend_ref(),
                base2k,
                0,
                &mut scratch.borrow(),
            );
            download_vec_znx::<BE>(&backend)
        };
        let want = normalize(&acc_oracle, &mut scratch);
        let have_derived = normalize(&acc_derived, &mut scratch);
        let have_module = normalize(&acc_module, &mut scratch);
        assert_eq!(
            want, have_derived,
            "cnv_by_const_apply_add: derived decomposition != two-step oracle (cnv_offset {cnv_offset})"
        );
        assert_eq!(
            want, have_module,
            "cnv_by_const_apply_add: module dispatch != two-step oracle (cnv_offset {cnv_offset})"
        );
    }
}

/// Uniform `a`/`b` operands, prepared as left and right convolution factors,
/// plus an arena covering every `_tmp_bytes` the convolution parity tests use
/// on the module-dispatch side.
#[allow(clippy::type_complexity)]
fn prepare_convolution_operands<BE: TestBackend + HalConvolutionImpl>(
    module: &Module<BE>,
    module_host: &Module<HostBytesBackend>,
    n: usize,
    cols: usize,
    a_size: usize,
    b_size: usize,
    fill_base2k: usize,
) -> (CnvPVecLOwned<BE>, CnvPVecROwned<BE>, ScratchOwned<BE>)
where
    Module<BE>: ModuleN + Convolution<BE> + CnvPVecAlloc<BE> + VecZnxAlloc<BE> + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let res_size: usize = a_size + b_size;
    let mut source: Source = Source::new([0u8; 32]);

    let mut a = module_host.vec_znx_alloc(n, cols, a_size);
    let mut b = module_host.vec_znx_alloc(n, cols, b_size);
    a.fill_uniform(fill_base2k, &mut source);
    b.fill_uniform(fill_base2k, &mut source);
    let a_backend = upload_vec_znx::<BE>(&a);
    let b_backend = upload_vec_znx::<BE>(&b);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .cnv_apply_dft_tmp_bytes(0, res_size, a_size, b_size)
            .max(module.cnv_apply_dft_add_tmp_bytes(0, res_size, a_size, b_size))
            .max(module.cnv_pairwise_apply_dft_tmp_bytes(0, res_size, a_size, b_size))
            .max(module.cnv_prepare_left_tmp_bytes(a_size, a_size))
            .max(module.cnv_prepare_right_tmp_bytes(b_size, b_size))
            .max(module.vec_znx_big_normalize_tmp_bytes()),
    );

    let mut a_prep: CnvPVecLOwned<BE> = module.cnv_pvec_left_alloc(n, cols, a_size, PrepareHint::Reuse);
    let mut b_prep: CnvPVecROwned<BE> = module.cnv_pvec_right_alloc(n, cols, b_size, PrepareHint::Reuse);
    module.cnv_prepare_left(
        &mut a_prep.to_backend_mut(),
        &vec_znx_backend_ref::<BE>(&a_backend),
        &mut scratch.borrow(),
    );
    module.cnv_prepare_right(
        &mut b_prep.to_backend_mut(),
        &vec_znx_backend_ref::<BE>(&b_backend),
        &mut scratch.borrow(),
    );

    (a_prep, b_prep, scratch)
}

/// One DFT column, brought back to the coefficient domain and normalized at
/// `base2k`: the comparison form this suite uses for DFT-domain results.
fn normalize_dft_column<BE: TestBackend>(
    module: &Module<BE>,
    base2k: usize,
    big: &mut crate::layouts::VecZnxBigOwned<BE>,
    res: &mut crate::layouts::VecZnxDftOwned<BE>,
    res_col: usize,
    scratch: &mut ScratchOwned<BE>,
) -> crate::layouts::VecZnxOwned<i64>
where
    Module<BE>: ModuleN + VecZnxAlloc<BE> + VecZnxIdftApplyTmpA<BE> + VecZnxBigNormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
{
    let size: usize = ZnxInfos::size(big);
    let template: VecZnxOwned<i64> = VecZnx::alloc(big.n(), 1, size);
    let mut backend = upload_vec_znx::<BE>(&template);
    module.vec_znx_idft_apply_tmpa(&mut big.to_backend_mut(), 0, &mut res.to_backend_mut(), res_col);
    module.vec_znx_big_normalize(
        &mut vec_znx_backend_mut::<BE>(&mut backend),
        base2k,
        size * base2k,
        0,
        0,
        &big.to_backend_ref(),
        base2k,
        0,
        &mut scratch.borrow(),
    );
    download_vec_znx::<BE>(&backend)
}
