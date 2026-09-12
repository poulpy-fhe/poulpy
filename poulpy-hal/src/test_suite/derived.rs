//! Default-body decomposition pinning for the derived operations.
//!
//! Each test hand-builds the oracle for one derived op by driving the *same*
//! decomposition the op's `poulpy_hal::oep::*_derived` function performs, but
//! through the public api traits on `Module` rather than the OEP dispatch the
//! derived function uses internally, then compares that oracle against the
//! derived function's own output on the same inputs. `test_vmp_apply_dft` in
//! `test_suite/vmp.rs` is the correctness oracle for `vmp_apply_dft` itself
//! (checked bit-for-bit against a second backend there); this test instead
//! pins the *shape* of the default body's decomposition — the column
//! alignment and call order — independently of any backend override, since
//! after PR4 there is no override left to compare against. Coefficient- and
//! big-domain results are compared bit for bit; DFT-domain results are
//! compared after `idft` and `big_normalize` at the suite's `base2k`, which is
//! how the rest of this suite states DFT-domain equality (spec decision 4).
//!
//! Where the two sides are allowed to distribute the same value over
//! different non-canonical digits — the accumulating shifts — the comparison
//! is on canonical forms, as in PR2's `rsh_sub` proof.

use crate::{
    api::{
        MatZnxAlloc, ModuleN, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAdd, VecZnxAddAssign, VecZnxAddScalarAssign,
        VecZnxAlloc, VecZnxBigAddAssign, VecZnxBigAddSmallAssign, VecZnxBigAlloc, VecZnxBigFromSmall, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxBigSubAssign, VecZnxBigSubNegateAssign, VecZnxCopy, VecZnxDftAddAssign, VecZnxDftAlloc,
        VecZnxDftApply, VecZnxDftAutomorphism, VecZnxDftAutomorphismPlan, VecZnxDftZero, VecZnxIdftApplyTmpA, VecZnxLsh,
        VecZnxLshAdd, VecZnxLshAssign, VecZnxLshSub, VecZnxLshTmpBytes, VecZnxMulXpMinusOne, VecZnxMulXpMinusOneAssign,
        VecZnxMulXpMinusOneAssignTmpBytes, VecZnxNormalize, VecZnxNormalizeAssign, VecZnxNormalizeTmpBytes, VecZnxRotate,
        VecZnxRsh, VecZnxRshAdd, VecZnxRshAssign, VecZnxRshSub, VecZnxRshTmpBytes, VecZnxSub, VecZnxSubAssign, VmpApplyDft,
        VmpApplyDftTmpBytes, VmpApplyDftToDft, VmpPMatAlloc, VmpPrepare, VmpPrepareTmpBytes,
    },
    layouts::{
        FillUniform, HostBytesBackend, MatZnx, MatZnxInfos, MatZnxToBackendRef, Module, PrepareHint, ScratchOwned,
        VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxDftToBackendMut, VecZnxDftToBackendRef, VecZnxInfos,
        VmpPMatToBackendMut, VmpPMatToBackendRef, ZnxInfos, ZnxView, ZnxViewMut, vec_znx_backend_mut, vec_znx_backend_ref,
    },
    oep::{HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl, HalVmpImpl, vmp_apply_dft_derived},
    source::Source,
    test_suite::{
        TestBackend, TestParams, download_vec_znx, scalar_znx_backend_ref, upload_mat_znx, upload_scalar_znx, upload_vec_znx,
    },
};

/// `vmp_apply_dft`: the derived free function's decomposition versus an
/// oracle hand-built from the public api traits.
pub fn test_vmp_apply_dft_derived<BE: TestBackend + HalVmpImpl<BE>>(params: &TestParams, module: &Module<BE>)
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

    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(module.n() as u64);

    let mut a = module_host.vec_znx_alloc(cols_in, size);
    a.fill_uniform(base2k, &mut source);
    let mut mat = module_host.mat_znx_alloc(rows, cols_in, cols_out, size);
    mat.fill_uniform(base2k, &mut source);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .vmp_prepare_tmp_bytes(rows, cols_in, cols_out, size)
            .max(module.vmp_apply_dft_tmp_bytes(size, size, rows, cols_in, cols_out, size))
            .max(module.vec_znx_big_normalize_tmp_bytes()),
    );

    let mat_backend = upload_mat_znx::<BE>(&mat);
    let mut pmat = module.vmp_pmat_alloc(rows, cols_in, cols_out, size, PrepareHint::Reuse);
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

    let mut a_dft_oracle = module.vec_znx_dft_alloc(b_cols_in, a_dft_size);
    for j in 0..offset {
        module.vec_znx_dft_zero(&mut a_dft_oracle.to_backend_mut(), j);
    }
    for j in 0..cols_to_copy {
        module.vec_znx_dft_apply(1, 0, &mut a_dft_oracle.to_backend_mut(), offset + j, &a_ref, a_start_col + j);
    }

    let mut res_oracle = module.vec_znx_dft_alloc(cols_out, size);
    module.vmp_apply_dft_to_dft(
        &mut res_oracle.to_backend_mut(),
        &a_dft_oracle.to_backend_ref(),
        &pmat_ref,
        0,
        &mut scratch.borrow(),
    );

    let mut res_derived = module.vec_znx_dft_alloc(cols_out, size);
    vmp_apply_dft_derived::<BE, BE, _>(module, &mut res_derived, &a_ref, &pmat_ref, &mut scratch.borrow());

    let mut big = module.vec_znx_big_alloc(1, size);
    for col in 0..cols_out {
        let want_template = module_host.vec_znx_alloc(1, size);
        let have_template = module_host.vec_znx_alloc(1, size);
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

/// `vec_znx_lsh`: the OEP default body against the definition it is derived
/// from, written out — `normalize(res_base2k = base2k, res_k = res_size *
/// base2k, res_offset = +k)`. Both sides are canonical, so this is bit for
/// bit. The arena is sized by `vec_znx_lsh_tmp_bytes(res_size)` alone, so a
/// default that under-reports its scratch panics here.
pub fn test_vec_znx_lsh_derived<BE: TestBackend + HalVecZnxImpl<BE>>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: VecZnxLsh<BE> + VecZnxLshTmpBytes + VecZnxNormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let mut source: Source = Source::new([0u8; 32]);
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(module.n() as u64);

    for a_size in [1usize, 2, 4] {
        let mut a = module_host.vec_znx_alloc(1, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_backend = upload_vec_znx::<BE>(&a);

        for res_size in [1usize, 2, 4] {
            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.vec_znx_lsh_tmp_bytes(res_size));

            for k in 0..=(res_size * base2k) {
                let mut res = module_host.vec_znx_alloc(1, res_size);
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

                // Oracle: the spec definition of `lsh`, spelled out.
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
/// from, written out — `normalize(res_base2k = base2k, res_k = res_size *
/// base2k, res_offset = -k)`. Both sides are canonical, so this is bit for
/// bit. The arena is sized by `vec_znx_rsh_tmp_bytes(res_size)` alone, so a
/// default that under-reports its scratch panics here.
pub fn test_vec_znx_rsh_derived<BE: TestBackend + HalVecZnxImpl<BE>>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: VecZnxRsh<BE> + VecZnxRshTmpBytes + VecZnxNormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let mut source: Source = Source::new([0u8; 32]);
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(module.n() as u64);

    for a_size in [1usize, 2, 4] {
        let mut a = module_host.vec_znx_alloc(1, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_backend = upload_vec_znx::<BE>(&a);

        for res_size in [1usize, 2, 4] {
            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.vec_znx_rsh_tmp_bytes(res_size));

            for k in 0..=(res_size * base2k) {
                let mut res = module_host.vec_znx_alloc(1, res_size);
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

                // Oracle: the spec definition of `rsh`, spelled out.
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
        let mut a = module_host.vec_znx_alloc(1, 2);
        a.fill_uniform(base2k, &mut source);
        let a_backend = upload_vec_znx::<BE>(&a);
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.vec_znx_rsh_tmp_bytes(res_size));
        let mut res = module_host.vec_znx_alloc(1, res_size);
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
/// the api — `tmp = lsh(a, k)` into a fresh buffer, then `res + (a << k)` with the
/// three-operand `vec_znx_add`. The two spread the same value over different
/// non-canonical digits, so the comparison is on canonical forms. The arena is
/// sized by the family's `vec_znx_lsh_tmp_bytes(res_size)`.
pub fn test_vec_znx_lsh_add_derived<BE: TestBackend + HalVecZnxImpl<BE>>(params: &TestParams, module: &Module<BE>)
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
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(module.n() as u64);

    for a_size in [1usize, 2, 4] {
        let mut a = module_host.vec_znx_alloc(1, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_backend = upload_vec_znx::<BE>(&a);

        for res_size in [1usize, 2, 4] {
            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                module
                    .vec_znx_lsh_tmp_bytes(res_size)
                    .max(module.vec_znx_normalize_tmp_bytes()),
            );

            for k in 0..=(res_size * base2k) {
                let mut res = module_host.vec_znx_alloc(1, res_size);
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
                let mut tmp_backend = upload_vec_znx::<BE>(&module_host.vec_znx_alloc(1, res_size));
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
/// the api — `tmp = lsh(a, k)` into a fresh buffer, then `res - (a << k)` with the
/// three-operand `vec_znx_sub`. The two spread the same value over different
/// non-canonical digits, so the comparison is on canonical forms. The arena is
/// sized by the family's `vec_znx_lsh_tmp_bytes(res_size)`.
pub fn test_vec_znx_lsh_sub_derived<BE: TestBackend + HalVecZnxImpl<BE>>(params: &TestParams, module: &Module<BE>)
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
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(module.n() as u64);

    for a_size in [1usize, 2, 4] {
        let mut a = module_host.vec_znx_alloc(1, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_backend = upload_vec_znx::<BE>(&a);

        for res_size in [1usize, 2, 4] {
            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                module
                    .vec_znx_lsh_tmp_bytes(res_size)
                    .max(module.vec_znx_normalize_tmp_bytes()),
            );

            for k in 0..=(res_size * base2k) {
                let mut res = module_host.vec_znx_alloc(1, res_size);
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
                let mut tmp_backend = upload_vec_znx::<BE>(&module_host.vec_znx_alloc(1, res_size));
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
/// the api — `tmp = rsh(a, k)` into a fresh buffer, then `res + (a >> k)` with the
/// three-operand `vec_znx_add`. The two spread the same value over different
/// non-canonical digits, so the comparison is on canonical forms. The arena is
/// sized by the family's `vec_znx_rsh_tmp_bytes(res_size)`.
pub fn test_vec_znx_rsh_add_derived<BE: TestBackend + HalVecZnxImpl<BE>>(params: &TestParams, module: &Module<BE>)
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
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(module.n() as u64);

    for a_size in [1usize, 2, 4] {
        let mut a = module_host.vec_znx_alloc(1, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_backend = upload_vec_znx::<BE>(&a);

        for res_size in [1usize, 2, 4] {
            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                module
                    .vec_znx_rsh_tmp_bytes(res_size)
                    .max(module.vec_znx_normalize_tmp_bytes()),
            );

            for k in 0..=(res_size * base2k) {
                let mut res = module_host.vec_znx_alloc(1, res_size);
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
                let mut tmp_backend = upload_vec_znx::<BE>(&module_host.vec_znx_alloc(1, res_size));
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
/// the api — `tmp = rsh(a, k)` into a fresh buffer, then `res - (a >> k)` with the
/// three-operand `vec_znx_sub`. The two spread the same value over different
/// non-canonical digits, so the comparison is on canonical forms. The arena is
/// sized by the family's `vec_znx_rsh_tmp_bytes(res_size)`.
pub fn test_vec_znx_rsh_sub_derived<BE: TestBackend + HalVecZnxImpl<BE>>(params: &TestParams, module: &Module<BE>)
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
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(module.n() as u64);

    for a_size in [1usize, 2, 4] {
        let mut a = module_host.vec_znx_alloc(1, a_size);
        a.fill_uniform(base2k, &mut source);
        let a_backend = upload_vec_znx::<BE>(&a);

        for res_size in [1usize, 2, 4] {
            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                module
                    .vec_znx_rsh_tmp_bytes(res_size)
                    .max(module.vec_znx_normalize_tmp_bytes()),
            );

            for k in 0..=(res_size * base2k) {
                let mut res = module_host.vec_znx_alloc(1, res_size);
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
                let mut tmp_backend = upload_vec_znx::<BE>(&module_host.vec_znx_alloc(1, res_size));
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

/// `vec_znx_lsh_assign`: the OEP default body against a two-step oracle built on
/// the api — `tmp = lsh(a, k)` into a fresh buffer, then `vec_znx_copy` back.
/// Compared on canonical forms; the arena is sized by
/// `vec_znx_lsh_tmp_bytes(res_size)`.
pub fn test_vec_znx_lsh_assign_derived<BE: TestBackend + HalVecZnxImpl<BE>>(params: &TestParams, module: &Module<BE>)
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
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(module.n() as u64);

    for res_size in [1usize, 2, 4] {
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .vec_znx_lsh_tmp_bytes(res_size)
                .max(module.vec_znx_normalize_tmp_bytes()),
        );

        for k in 0..=(res_size * base2k) {
            let mut res = module_host.vec_znx_alloc(1, res_size);
            res.fill_uniform(base2k, &mut source);
            let orig_backend = upload_vec_znx::<BE>(&res);
            let mut have_backend = upload_vec_znx::<BE>(&res);
            let mut want_backend = upload_vec_znx::<BE>(&res);

            module.vec_znx_lsh_assign(
                base2k,
                k,
                &mut vec_znx_backend_mut::<BE>(&mut have_backend),
                0,
                &mut scratch.borrow(),
            );

            // Oracle: shift into a fresh buffer, then copy back.
            let mut tmp_backend = upload_vec_znx::<BE>(&module_host.vec_znx_alloc(1, res_size));
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

            for backend in [&mut want_backend, &mut have_backend] {
                module.vec_znx_normalize_assign(
                    base2k,
                    res_size * base2k,
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
/// the api — `tmp = rsh(a, k)` into a fresh buffer, then `vec_znx_copy` back.
/// Compared on canonical forms; the arena is sized by
/// `vec_znx_rsh_tmp_bytes(res_size)`.
pub fn test_vec_znx_rsh_assign_derived<BE: TestBackend + HalVecZnxImpl<BE>>(params: &TestParams, module: &Module<BE>)
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
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(module.n() as u64);

    for res_size in [1usize, 2, 4] {
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .vec_znx_rsh_tmp_bytes(res_size)
                .max(module.vec_znx_normalize_tmp_bytes()),
        );

        for k in 0..=(res_size * base2k) {
            let mut res = module_host.vec_znx_alloc(1, res_size);
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
            let mut tmp_backend = upload_vec_znx::<BE>(&module_host.vec_znx_alloc(1, res_size));
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
pub fn test_vec_znx_mul_xp_minus_one_derived<BE: TestBackend + HalVecZnxImpl<BE>>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: VecZnxMulXpMinusOne<BE> + VecZnxRotate<BE> + VecZnxSubAssign<BE>,
{
    let base2k: usize = params.base2k;
    let mut source: Source = Source::new([0u8; 32]);
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(module.n() as u64);

    for size in [1usize, 2, 4] {
        let mut a = module_host.vec_znx_alloc(1, size);
        a.fill_uniform(base2k, &mut source);
        let a_backend = upload_vec_znx::<BE>(&a);

        for p in [1i64, 5, -3, module.n() as i64] {
            let mut res = module_host.vec_znx_alloc(1, size);
            res.fill_uniform(base2k, &mut source);
            let mut want_backend = upload_vec_znx::<BE>(&res);
            let mut have_backend = upload_vec_znx::<BE>(&res);

            crate::oep::vec_znx_mul_xp_minus_one_derived::<BE, BE>(
                module,
                p,
                &mut vec_znx_backend_mut::<BE>(&mut want_backend),
                0,
                &vec_znx_backend_ref::<BE>(&a_backend),
                0,
            );

            // Oracle: the same decomposition, through the public api traits.
            module.vec_znx_rotate(
                p,
                &mut vec_znx_backend_mut::<BE>(&mut have_backend),
                0,
                &vec_znx_backend_ref::<BE>(&a_backend),
                0,
            );
            module.vec_znx_sub_assign(
                &mut vec_znx_backend_mut::<BE>(&mut have_backend),
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

/// `vec_znx_mul_xp_minus_one_assign`: the per-limb OEP default body versus the
/// whole-vector `vec_znx_mul_xp_minus_one` on the same input — an independent
/// decomposition of the same map, so the comparison is bit for bit.
pub fn test_vec_znx_mul_xp_minus_one_assign_derived<BE: TestBackend + HalVecZnxImpl<BE>>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: VecZnxMulXpMinusOne<BE> + VecZnxMulXpMinusOneAssign<BE> + VecZnxMulXpMinusOneAssignTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let mut source: Source = Source::new([0u8; 32]);
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(module.n() as u64);

    for size in [1usize, 2, 4] {
        // Sized by the op's own `_tmp_bytes(size)` alone, so a default body
        // that under-reports its scratch panics here.
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.vec_znx_mul_xp_minus_one_assign_tmp_bytes(size));

        for p in [1i64, 5, -3, module.n() as i64] {
            let mut a = module_host.vec_znx_alloc(1, size);
            a.fill_uniform(base2k, &mut source);
            let a_backend = upload_vec_znx::<BE>(&a);

            let mut want_backend = upload_vec_znx::<BE>(&a);
            crate::oep::vec_znx_mul_xp_minus_one_assign_derived::<BE, BE>(
                module,
                p,
                &mut vec_znx_backend_mut::<BE>(&mut want_backend),
                0,
                &mut scratch.borrow(),
            );

            // Oracle: the out-of-place op, which is a different decomposition
            // (whole-vector rotate then subtract) of the same map.
            let mut have_backend = upload_vec_znx::<BE>(&a);
            module.vec_znx_mul_xp_minus_one(
                p,
                &mut vec_znx_backend_mut::<BE>(&mut have_backend),
                0,
                &vec_znx_backend_ref::<BE>(&a_backend),
                0,
            );

            assert_eq!(
                download_vec_znx::<BE>(&want_backend),
                download_vec_znx::<BE>(&have_backend),
                "vec_znx_mul_xp_minus_one_assign: default body != out-of-place form (size {size} p {p})"
            );
        }
    }
}

/// `vec_znx_add_scalar_assign`: the OEP default body (an `add_assign` on the
/// one-limb window) versus a whole-vector `add_assign` against a `VecZnx` that
/// carries the scalar in limb `res_limb` — an independent oracle, bit for bit.
pub fn test_vec_znx_add_scalar_assign_derived<BE: TestBackend + HalVecZnxImpl<BE>>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: VecZnxAddScalarAssign<BE> + VecZnxAddAssign<BE>,
{
    let base2k: usize = params.base2k;
    let n: usize = module.n();
    let mut source: Source = Source::new([0u8; 32]);
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(n as u64);

    let mut scalar = module_host.scalar_znx_alloc(1);
    scalar.fill_uniform(base2k, &mut source);
    let scalar_backend = upload_scalar_znx::<BE>(&scalar);

    for res_size in [1usize, 2, 4] {
        for res_limb in 0..res_size {
            let mut res = module_host.vec_znx_alloc(1, res_size);
            res.fill_uniform(base2k, &mut source);
            let mut want_backend = upload_vec_znx::<BE>(&res);
            let mut have_backend = upload_vec_znx::<BE>(&res);

            crate::oep::vec_znx_add_scalar_assign_derived::<BE, BE>(
                module,
                &mut vec_znx_backend_mut::<BE>(&mut want_backend),
                0,
                res_limb,
                &scalar_znx_backend_ref::<BE>(&scalar_backend),
                0,
            );

            // Oracle: the scalar lifted into a full-width `VecZnx` whose only
            // non-zero limb is `res_limb`, added with the plain `add_assign`.
            let mut lifted = module_host.vec_znx_alloc(1, res_size);
            lifted.at_mut(0, res_limb).copy_from_slice(scalar.at(0, 0));
            let lifted_backend = upload_vec_znx::<BE>(&lifted);
            module.vec_znx_add_assign(
                &mut vec_znx_backend_mut::<BE>(&mut have_backend),
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
/// independent oracle — `from_small(b)` then `add_assign(a)`, written out
/// through the public api traits (a different call path from the free
/// function under test, even though it is the same decomposition).
pub fn test_vec_znx_big_add_small_derived<BE: TestBackend + HalVecZnxBigImpl<BE>>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: ModuleN
        + VecZnxBigAlloc<BE>
        + VecZnxBigFromSmall<BE>
        + VecZnxBigAddAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let mut source: Source = Source::new([0u8; 32]);
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(module.n() as u64);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.vec_znx_big_normalize_tmp_bytes());

    // (big operand size, small operand size, res size): res longer than both,
    // small shorter than big, small longer than big, res shorter than both.
    for (a_size, b_size, res_size) in [(3usize, 3usize, 3usize), (4, 2, 3), (2, 4, 3), (3, 3, 5), (4, 4, 2)] {
        let mut a_small = module_host.vec_znx_alloc(1, a_size);
        a_small.fill_uniform(base2k, &mut source);
        let mut b = module_host.vec_znx_alloc(1, b_size);
        b.fill_uniform(base2k, &mut source);
        let a_backend = upload_vec_znx::<BE>(&a_small);
        let b_backend = upload_vec_znx::<BE>(&b);

        // `a` as a VecZnxBig, through the basis promotion.
        let mut a_big = module.vec_znx_big_alloc(1, a_size);
        module.vec_znx_big_from_small(&mut a_big.to_backend_mut(), 0, &vec_znx_backend_ref::<BE>(&a_backend), 0);

        let mut want_big = module.vec_znx_big_alloc(1, res_size);
        let mut have_big = module.vec_znx_big_alloc(1, res_size);

        crate::oep::vec_znx_big_add_small_derived::<BE, BE>(
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

        let want_template = module_host.vec_znx_alloc(1, res_size);
        let have_template = module_host.vec_znx_alloc(1, res_size);
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

        assert_eq!(
            download_vec_znx::<BE>(&want_backend),
            download_vec_znx::<BE>(&have_backend),
            "vec_znx_big_add_small: default body != independent oracle (a {a_size} b {b_size} res {res_size})"
        );
    }
}

/// `vec_znx_big_sub_small_a`: the scratch-free OEP default (`res = a - b`,
/// `a` the coefficient-domain operand) against an independent oracle —
/// `from_small(a)` then `sub_assign(b)`, through the public api traits.
pub fn test_vec_znx_big_sub_small_a_derived<BE: TestBackend + HalVecZnxBigImpl<BE>>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: ModuleN
        + VecZnxBigAlloc<BE>
        + VecZnxBigFromSmall<BE>
        + VecZnxBigSubAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let mut source: Source = Source::new([0u8; 32]);
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(module.n() as u64);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.vec_znx_big_normalize_tmp_bytes());

    // (small operand size, big operand size, res size): res longer than both,
    // small shorter than big, small longer than big, res shorter than both.
    for (a_size, b_size, res_size) in [(3usize, 3usize, 3usize), (2, 4, 3), (4, 2, 3), (3, 3, 5), (4, 4, 2)] {
        let mut a_small = module_host.vec_znx_alloc(1, a_size);
        a_small.fill_uniform(base2k, &mut source);
        let mut b = module_host.vec_znx_alloc(1, b_size);
        b.fill_uniform(base2k, &mut source);
        let a_backend = upload_vec_znx::<BE>(&a_small);
        let b_backend = upload_vec_znx::<BE>(&b);

        // `b` as a VecZnxBig, through the basis promotion.
        let mut b_big = module.vec_znx_big_alloc(1, b_size);
        module.vec_znx_big_from_small(&mut b_big.to_backend_mut(), 0, &vec_znx_backend_ref::<BE>(&b_backend), 0);

        let mut want_big = module.vec_znx_big_alloc(1, res_size);
        let mut have_big = module.vec_znx_big_alloc(1, res_size);

        crate::oep::vec_znx_big_sub_small_a_derived::<BE, BE>(
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

        let want_template = module_host.vec_znx_alloc(1, res_size);
        let have_template = module_host.vec_znx_alloc(1, res_size);
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

        assert_eq!(
            download_vec_znx::<BE>(&want_backend),
            download_vec_znx::<BE>(&have_backend),
            "vec_znx_big_sub_small_a: default body != independent oracle (a {a_size} b {b_size} res {res_size})"
        );
    }
}

/// `vec_znx_big_sub_small_b`: the scratch-free OEP default (`res = a - b`,
/// `b` the coefficient-domain operand) against an independent oracle —
/// `from_small(b)` then `sub_negate_assign(a)`, through the public api traits.
pub fn test_vec_znx_big_sub_small_b_derived<BE: TestBackend + HalVecZnxBigImpl<BE>>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: ModuleN
        + VecZnxBigAlloc<BE>
        + VecZnxBigFromSmall<BE>
        + VecZnxBigSubNegateAssign<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let mut source: Source = Source::new([0u8; 32]);
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(module.n() as u64);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.vec_znx_big_normalize_tmp_bytes());

    // (big operand size, small operand size, res size): res longer than both,
    // small shorter than big, small longer than big, res shorter than both.
    for (a_size, b_size, res_size) in [(3usize, 3usize, 3usize), (4, 2, 3), (2, 4, 3), (3, 3, 5), (4, 4, 2)] {
        let mut a_small = module_host.vec_znx_alloc(1, a_size);
        a_small.fill_uniform(base2k, &mut source);
        let mut b = module_host.vec_znx_alloc(1, b_size);
        b.fill_uniform(base2k, &mut source);
        let a_backend = upload_vec_znx::<BE>(&a_small);
        let b_backend = upload_vec_znx::<BE>(&b);

        // `a` as a VecZnxBig, through the basis promotion.
        let mut a_big = module.vec_znx_big_alloc(1, a_size);
        module.vec_znx_big_from_small(&mut a_big.to_backend_mut(), 0, &vec_znx_backend_ref::<BE>(&a_backend), 0);

        let mut want_big = module.vec_znx_big_alloc(1, res_size);
        let mut have_big = module.vec_znx_big_alloc(1, res_size);

        crate::oep::vec_znx_big_sub_small_b_derived::<BE, BE>(
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

        let want_template = module_host.vec_znx_alloc(1, res_size);
        let have_template = module_host.vec_znx_alloc(1, res_size);
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

        assert_eq!(
            download_vec_znx::<BE>(&want_backend),
            download_vec_znx::<BE>(&have_backend),
            "vec_znx_big_sub_small_b: default body != independent oracle (a {a_size} b {b_size} res {res_size})"
        );
    }
}

/// `vec_znx_idft_normalize_consume`: the OEP default body against an
/// independent oracle — `idft_apply_tmpa` into a `VecZnxBig`, the optional
/// `big_add_small_assign`, then `big_normalize` — driven through the public
/// api traits. The op clobbers its input, so each side transforms its own
/// copy of the same host operand. Both arenas are sized by their own side's
/// `_tmp_bytes` alone, so a default that under-reports its scratch panics.
pub fn test_vec_znx_idft_normalize_consume_derived<BE: TestBackend + HalVecZnxDftImpl<BE>>(
    params: &TestParams,
    module: &Module<BE>,
) where
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
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(module.n() as u64);
    let mut oracle_scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.vec_znx_big_normalize_tmp_bytes());

    // (a_size, res_size): equal, res shorter, res longer.
    for (a_size, res_size) in [(3usize, 3usize), (4, 2), (2, 4)] {
        let mut a = module_host.vec_znx_alloc(1, a_size);
        a.fill_uniform(base2k, &mut source);
        let mut addend = module_host.vec_znx_alloc(1, a_size);
        addend.fill_uniform(base2k, &mut source);
        let a_backend = upload_vec_znx::<BE>(&a);
        let addend_backend = upload_vec_znx::<BE>(&addend);
        let addend_ref = vec_znx_backend_ref::<BE>(&addend_backend);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(crate::oep::vec_znx_idft_normalize_consume_tmp_bytes_derived::<
            BE,
            BE,
        >(module, res_size, a_size));

        for with_addend in [false, true] {
            let addend_arg = with_addend.then_some((&addend_ref, 0));

            // The op consumes its input, so each side gets its own transform.
            let mut have_dft = module.vec_znx_dft_alloc(1, a_size);
            let mut want_dft = module.vec_znx_dft_alloc(1, a_size);
            for dft in [&mut have_dft, &mut want_dft] {
                module.vec_znx_dft_apply(1, 0, &mut dft.to_backend_mut(), 0, &vec_znx_backend_ref::<BE>(&a_backend), 0);
            }

            let res_template = module_host.vec_znx_alloc(1, res_size);
            let mut have_backend = upload_vec_znx::<BE>(&res_template);
            let mut want_backend = upload_vec_znx::<BE>(&res_template);

            crate::oep::vec_znx_idft_normalize_consume_derived::<BE, BE>(
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
            let mut big = module.vec_znx_big_alloc(1, a_size);
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
/// independent oracle — `automorphism_with_plan` into a `min(res, a)`-limb
/// temporary, then `vec_znx_dft_add_assign` — through the public api traits.
/// DFT-domain results are compared after `idft` and `big_normalize`. The
/// arena is sized by the op's own `_tmp_bytes` alone.
pub fn test_vec_znx_dft_automorphism_add_with_plan_derived<BE: TestBackend + HalVecZnxDftImpl<BE>>(
    params: &TestParams,
    module: &Module<BE>,
) where
    Module<BE>: ModuleN
        + VecZnxAlloc<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftAutomorphism<BE>
        + VecZnxDftAutomorphismPlan<BE, Plan = <BE as HalVecZnxDftImpl<BE>>::AutomorphismPlan>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let mut source: Source = Source::new([1u8; 32]);
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(module.n() as u64);
    let mut oracle_scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.vec_znx_big_normalize_tmp_bytes());

    for (res_size, a_size) in [(3usize, 3usize), (2, 4), (4, 2)] {
        let size: usize = res_size.min(a_size);
        let mut a = module_host.vec_znx_alloc(1, a_size);
        let mut seed = module_host.vec_znx_alloc(1, res_size);
        a.fill_uniform(base2k, &mut source);
        seed.fill_uniform(base2k, &mut source);
        let a_backend = upload_vec_znx::<BE>(&a);
        let seed_backend = upload_vec_znx::<BE>(&seed);

        let mut a_dft = module.vec_znx_dft_alloc(1, a_size);
        module.vec_znx_dft_apply(
            1,
            0,
            &mut a_dft.to_backend_mut(),
            0,
            &vec_znx_backend_ref::<BE>(&a_backend),
            0,
        );

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            crate::oep::vec_znx_dft_automorphism_add_with_plan_tmp_bytes_derived::<BE, BE>(module, res_size, a_size),
        );

        for p in [1i64, 5, -3] {
            let plan = module.vec_znx_dft_automorphism_plan(p);

            let mut have_dft = module.vec_znx_dft_alloc(1, res_size);
            let mut want_dft = module.vec_znx_dft_alloc(1, res_size);
            for dft in [&mut have_dft, &mut want_dft] {
                module.vec_znx_dft_apply(
                    1,
                    0,
                    &mut dft.to_backend_mut(),
                    0,
                    &vec_znx_backend_ref::<BE>(&seed_backend),
                    0,
                );
            }

            crate::oep::vec_znx_dft_automorphism_add_with_plan_derived::<BE, BE>(
                module,
                &plan,
                &mut have_dft.to_backend_mut(),
                0,
                &a_dft.to_backend_ref(),
                0,
                &mut scratch.borrow(),
            );

            // Oracle: automorphism into a `min(res, a)`-limb temporary, then
            // the DFT-domain accumulation, through the public api traits.
            let mut rot = module.vec_znx_dft_alloc(1, size);
            module.vec_znx_dft_automorphism_with_plan(&plan, &mut rot.to_backend_mut(), 0, &a_dft.to_backend_ref(), 0);
            module.vec_znx_dft_add_assign(&mut want_dft.to_backend_mut(), 0, &rot.to_backend_ref(), 0);

            let res_template = module_host.vec_znx_alloc(1, res_size);
            let mut have_backend = upload_vec_znx::<BE>(&res_template);
            let mut want_backend = upload_vec_znx::<BE>(&res_template);
            let mut big = module.vec_znx_big_alloc(1, res_size);
            for (dft, out) in [(&mut have_dft, &mut have_backend), (&mut want_dft, &mut want_backend)] {
                module.vec_znx_idft_apply_tmpa(&mut big.to_backend_mut(), 0, &mut dft.to_backend_mut(), 0);
                module.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BE>(out),
                    base2k,
                    res_size * base2k,
                    0,
                    0,
                    &big.to_backend_ref(),
                    base2k,
                    0,
                    &mut oracle_scratch.borrow(),
                );
            }

            assert_eq!(
                download_vec_znx::<BE>(&want_backend),
                download_vec_znx::<BE>(&have_backend),
                "vec_znx_dft_automorphism_add_with_plan: default body != independent oracle (p {p} res_size {res_size} \
                 a_size {a_size})"
            );
        }
    }
}
