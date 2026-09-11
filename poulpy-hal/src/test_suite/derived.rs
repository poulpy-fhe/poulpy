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
        MatZnxAlloc, ModuleN, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAlloc, VecZnxBigAlloc, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxDftAlloc, VecZnxDftApply, VecZnxDftZero, VecZnxIdftApplyTmpA, VmpApplyDft,
        VmpApplyDftTmpBytes, VmpApplyDftToDft, VmpPMatAlloc, VmpPrepare, VmpPrepareTmpBytes,
    },
    layouts::{
        FillUniform, HostBytesBackend, MatZnx, MatZnxInfos, MatZnxToBackendRef, Module, PrepareHint, ScratchOwned,
        VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxDftToBackendMut, VecZnxDftToBackendRef, VecZnxInfos,
        VmpPMatToBackendMut, VmpPMatToBackendRef, ZnxInfos, vec_znx_backend_mut, vec_znx_backend_ref,
    },
    oep::{HalVmpImpl, vmp_apply_dft_derived},
    source::Source,
    test_suite::{TestBackend, TestParams, download_vec_znx, upload_mat_znx, upload_vec_znx},
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
