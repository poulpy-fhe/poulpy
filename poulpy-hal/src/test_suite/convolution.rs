use super::{download_vec_znx, upload_vec_znx, vec_znx_backend_mut, vec_znx_backend_ref};
use crate::layouts::CnvPVecLToBackendMut;
use crate::layouts::CnvPVecLToBackendRef;
use crate::layouts::CnvPVecRToBackendMut;
use crate::layouts::CnvPVecRToBackendRef;
use crate::layouts::VecZnxBigToBackendMut;
use crate::layouts::VecZnxBigToBackendRef;
use crate::layouts::VecZnxDftToBackendMut;
use crate::layouts::VecZnxDftToBackendRef;
use rand::Rng;

use crate::{
    api::{
        CnvPVecAlloc, Convolution, ModuleN, ScratchOwnedAlloc, VecZnxAddIntoBackend, VecZnxBigAlloc, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxCopyBackend, VecZnxDftAddAssign, VecZnxDftAlloc, VecZnxDftApply, VecZnxIdftApplyTmpA,
        VecZnxNormalizeAssignBackend,
    },
    layouts::{DataView, FillUniform, ScratchArena, ScratchOwned, VecZnx, VecZnxOwned, ZnxView, ZnxViewMut, ZnxZero},
    source::Source,
};

use crate::layouts::VecZnxDftOwned;
use crate::layouts::{CnvPVecLOwned, CnvPVecROwned, VecZnxBigOwned};

pub fn test_convolution_by_const<M, BE: crate::test_suite::TestBackend>(module: &M, base2k: usize)
where
    M: ModuleN
        + Convolution<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxNormalizeAssignBackend<BE>
        + VecZnxBigAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let mut source: Source = Source::new([0u8; 32]);

    let a_cols: usize = 2;
    let a_size: usize = 15;
    let b_size: usize = 15;
    let res_size: usize = a_size + b_size;

    let mut a = VecZnx::alloc(module.n(), a_cols, a_size);
    let mut b = VecZnx::alloc(module.n(), 1, b_size);

    let mut res_want = VecZnx::alloc(module.n(), 1, res_size);
    let mut res_big: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(1, res_size);

    a.fill_uniform(17, &mut source);

    let mask = (1 << base2k) - 1;
    for j in 0..1 {
        let r = source.next_u64() & mask;
        b.at_mut(0, j)[0] = ((r << (64 - 17)) as i64) >> (64 - 17);
    }

    let a_backend = upload_vec_znx::<BE>(&a);
    let b_backend = upload_vec_znx::<BE>(&b);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .cnv_by_const_apply_tmp_bytes(0, res_size, a_size, b_size)
            .max(module.vec_znx_big_normalize_tmp_bytes()),
    );

    for a_col in 0..a.cols() {
        for cnv_offset in 0..res_size {
            module.cnv_by_const_apply(
                cnv_offset,
                &mut res_big.to_backend_mut(),
                0,
                &vec_znx_backend_ref::<BE>(&a_backend),
                a_col,
                &vec_znx_backend_ref::<BE>(&b_backend),
                0,
                0,
                &mut scratch.arena(),
            );

            let res_host_template = VecZnx::alloc(module.n(), 1, res_size);
            let mut res_have_backend = upload_vec_znx::<BE>(&res_host_template);
            module.vec_znx_big_normalize(
                &mut vec_znx_backend_mut::<BE>(&mut res_have_backend),
                base2k,
                0,
                0,
                &res_big.to_backend_ref(),
                base2k,
                0,
                &mut scratch.arena(),
            );
            let res_have = download_vec_znx::<BE>(&res_have_backend);

            bivariate_convolution_naive(
                module,
                base2k,
                (cnv_offset + 1) as i64,
                &mut res_want,
                0,
                &a,
                a_col,
                &b,
                0,
                &mut scratch.arena(),
            );

            assert_eq!(res_want, res_have);
        }
    }
}

pub fn test_convolution<M, BE: crate::test_suite::TestBackend>(module: &M, base2k: usize)
where
    M: ModuleN
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxNormalizeAssignBackend<BE>
        + VecZnxBigAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let mut source: Source = Source::new([0u8; 32]);

    let a_cols: usize = 2;
    let b_cols: usize = 2;
    let a_size: usize = 15;
    let b_size: usize = 15;
    let res_size: usize = a_size + b_size;

    let mut a = VecZnx::alloc(module.n(), a_cols, a_size);
    let mut b = VecZnx::alloc(module.n(), b_cols, b_size);

    let mut res_want = VecZnx::alloc(module.n(), 1, res_size);
    // Two-column DFT destination written at column 1: covers the
    // column-interleaved `VecZnxDft` indexing of the backend kernels.
    let res_dft_col: usize = 1;
    let mut res_dft: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(2, res_size);
    let mut res_big: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(1, res_size);

    a.fill_uniform(17, &mut source);
    b.fill_uniform(17, &mut source);

    let a_backend = upload_vec_znx::<BE>(&a);
    let b_backend = upload_vec_znx::<BE>(&b);

    let mut a_prep: CnvPVecLOwned<BE> = module.cnv_pvec_left_alloc(a_cols, a_size);
    let mut b_prep: CnvPVecROwned<BE> = module.cnv_pvec_right_alloc(b_cols, b_size);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .cnv_apply_dft_tmp_bytes(0, res_size, a_size, b_size)
            .max(module.cnv_prepare_left_tmp_bytes(res_size, a_size))
            .max(module.cnv_prepare_right_tmp_bytes(res_size, b_size))
            .max(module.vec_znx_big_normalize_tmp_bytes()),
    );

    {
        let mut a_prep_backend = a_prep.to_backend_mut();
        module.cnv_prepare_left(
            &mut a_prep_backend,
            &vec_znx_backend_ref::<BE>(&a_backend),
            !0i64,
            &mut scratch.arena(),
        );
    }
    {
        let mut b_prep_backend = b_prep.to_backend_mut();
        module.cnv_prepare_right(
            &mut b_prep_backend,
            &vec_znx_backend_ref::<BE>(&b_backend),
            !0i64,
            &mut scratch.arena(),
        );
    }

    for a_col in 0..a.cols() {
        for b_col in 0..b.cols() {
            for cnv_offset in 0..res_size {
                module.cnv_apply_dft(
                    cnv_offset,
                    &mut res_dft.to_backend_mut(),
                    res_dft_col,
                    &a_prep.to_backend_ref(),
                    a_col,
                    &b_prep.to_backend_ref(),
                    b_col,
                    &mut scratch.arena(),
                );

                module.vec_znx_idft_apply_tmpa(&mut res_big.to_backend_mut(), 0, &mut res_dft.to_backend_mut(), res_dft_col);

                let res_host_template = VecZnx::alloc(module.n(), 1, res_size);
                let mut res_have_backend = upload_vec_znx::<BE>(&res_host_template);
                module.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BE>(&mut res_have_backend),
                    base2k,
                    0,
                    0,
                    &res_big.to_backend_ref(),
                    base2k,
                    0,
                    &mut scratch.arena(),
                );
                let res_have = download_vec_znx::<BE>(&res_have_backend);

                bivariate_convolution_naive(
                    module,
                    base2k,
                    (cnv_offset + 1) as i64,
                    &mut res_want,
                    0,
                    &a,
                    a_col,
                    &b,
                    b_col,
                    &mut scratch.arena(),
                );

                assert_eq!(res_want, res_have);
            }
        }
    }
}

/// `cnv_apply_dft_accumulate` matches `cnv_apply_dft` followed by a DFT add,
/// bit-for-bit on the raw prepared data.
pub fn test_convolution_accumulate<M, BE: crate::test_suite::TestBackend>(module: &M, _base2k: usize)
where
    M: ModuleN + Convolution<BE> + CnvPVecAlloc<BE> + VecZnxDftAlloc<BE> + VecZnxDftAddAssign<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let mut source: Source = Source::new([0u8; 32]);

    let cols: usize = 2;
    let a_size: usize = 15;
    let b_size: usize = 15;
    let res_size: usize = a_size + b_size;

    let mut a = VecZnx::alloc(module.n(), cols, a_size);
    let mut b = VecZnx::alloc(module.n(), cols, b_size);
    a.fill_uniform(17, &mut source);
    b.fill_uniform(17, &mut source);

    let a_backend = upload_vec_znx::<BE>(&a);
    let b_backend = upload_vec_znx::<BE>(&b);

    let mut a_prep: CnvPVecLOwned<BE> = module.cnv_pvec_left_alloc(cols, a_size);
    let mut b_prep: CnvPVecROwned<BE> = module.cnv_pvec_right_alloc(cols, b_size);

    // Two-column accumulators exercised at both columns: covers the
    // column-interleaved `VecZnxDft` indexing of the backend kernels.
    let res_cols: usize = 2;
    let mut res_acc: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(res_cols, res_size);
    let mut res_ref: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(res_cols, res_size);
    let mut tmp_dft: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(1, res_size);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .cnv_apply_dft_tmp_bytes(0, res_size, a_size, b_size)
            .max(module.cnv_prepare_left_tmp_bytes(res_size, a_size))
            .max(module.cnv_prepare_right_tmp_bytes(res_size, b_size)),
    );

    {
        let mut a_prep_backend = a_prep.to_backend_mut();
        module.cnv_prepare_left(
            &mut a_prep_backend,
            &vec_znx_backend_ref::<BE>(&a_backend),
            !0i64,
            &mut scratch.arena(),
        );
    }
    {
        let mut b_prep_backend = b_prep.to_backend_mut();
        module.cnv_prepare_right(
            &mut b_prep_backend,
            &vec_znx_backend_ref::<BE>(&b_backend),
            !0i64,
            &mut scratch.arena(),
        );
    }

    for res_col in 0..res_cols {
        // Identical deterministic initial accumulator content for both paths.
        module.cnv_apply_dft(
            0,
            &mut res_acc.to_backend_mut(),
            res_col,
            &a_prep.to_backend_ref(),
            0,
            &b_prep.to_backend_ref(),
            0,
            &mut scratch.arena(),
        );
        module.cnv_apply_dft(
            0,
            &mut res_ref.to_backend_mut(),
            res_col,
            &a_prep.to_backend_ref(),
            0,
            &b_prep.to_backend_ref(),
            0,
            &mut scratch.arena(),
        );

        for a_col in 0..cols {
            for b_col in 0..cols {
                for cnv_offset in (0..res_size).step_by(3) {
                    module.cnv_apply_dft_accumulate(
                        cnv_offset,
                        &mut res_acc.to_backend_mut(),
                        res_col,
                        &a_prep.to_backend_ref(),
                        a_col,
                        &b_prep.to_backend_ref(),
                        b_col,
                        &mut scratch.arena(),
                    );

                    module.cnv_apply_dft(
                        cnv_offset,
                        &mut tmp_dft.to_backend_mut(),
                        0,
                        &a_prep.to_backend_ref(),
                        a_col,
                        &b_prep.to_backend_ref(),
                        b_col,
                        &mut scratch.arena(),
                    );
                    module.vec_znx_dft_add_assign(&mut res_ref.to_backend_mut(), res_col, &tmp_dft.to_backend_ref(), 0);

                    assert!(
                        BE::to_host_bytes(res_acc.data()) == BE::to_host_bytes(res_ref.data()),
                        "accumulate != apply + add (res_col={res_col} a_col={a_col} b_col={b_col} cnv_offset={cnv_offset})"
                    );
                }
            }
        }
    }
}

/// `cnv_accumulate_dft` matches the per-term `cnv_apply_dft` +
/// `cnv_apply_dft_accumulate` sequence after normalization to the coefficient
/// domain (the fused path reduces once per output, so the raw q120 lazy
/// representatives may differ).
pub fn test_convolution_accumulate_fused<M, BE: crate::test_suite::TestBackend>(module: &M, base2k: usize)
where
    M: ModuleN
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxBigAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    use crate::layouts::{CnvDftAccTerm, CnvDftStore};

    let mut source: Source = Source::new([0u8; 32]);

    let cols: usize = 2;
    let a_size: usize = 15;
    let b_size: usize = 15;
    let res_size: usize = a_size + b_size;
    // Two-column destination written at column 1: covers the column-interleaved
    // `VecZnxDft` indexing of the backend kernels.
    let res_col: usize = 1;

    let mut a = VecZnx::alloc(module.n(), cols, a_size);
    let mut b = VecZnx::alloc(module.n(), cols, b_size);
    a.fill_uniform(17, &mut source);
    b.fill_uniform(17, &mut source);

    let a_backend = upload_vec_znx::<BE>(&a);
    let b_backend = upload_vec_znx::<BE>(&b);

    let mut a_prep: CnvPVecLOwned<BE> = module.cnv_pvec_left_alloc(cols, a_size);
    let mut b_prep: CnvPVecROwned<BE> = module.cnv_pvec_right_alloc(cols, b_size);

    let mut res_fused: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(2, res_size);
    let mut res_ref: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(2, res_size);
    let mut big_fused: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(1, res_size);
    let mut big_ref: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(1, res_size);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .cnv_accumulate_dft_tmp_bytes(0, res_size, a_size, b_size)
            .max(module.cnv_apply_dft_tmp_bytes(0, res_size, a_size, b_size))
            .max(module.cnv_prepare_left_tmp_bytes(res_size, a_size))
            .max(module.cnv_prepare_right_tmp_bytes(res_size, b_size))
            .max(module.vec_znx_big_normalize_tmp_bytes()),
    );

    {
        let mut a_prep_backend = a_prep.to_backend_mut();
        module.cnv_prepare_left(
            &mut a_prep_backend,
            &vec_znx_backend_ref::<BE>(&a_backend),
            !0i64,
            &mut scratch.arena(),
        );
    }
    {
        let mut b_prep_backend = b_prep.to_backend_mut();
        module.cnv_prepare_right(
            &mut b_prep_backend,
            &vec_znx_backend_ref::<BE>(&b_backend),
            !0i64,
            &mut scratch.arena(),
        );
    }

    // Three terms mixing operand columns, like one BSGS giant step.
    let term_cols: [(usize, usize); 3] = [(0, 0), (1, 1), (0, 1)];

    for cnv_offset in (0..res_size).step_by(3) {
        {
            let terms: Vec<CnvDftAccTerm<'_, BE>> = term_cols
                .iter()
                .map(|&(a_col, b_col)| CnvDftAccTerm {
                    a: a_prep.to_backend_ref(),
                    a_col,
                    b: b_prep.to_backend_ref(),
                    b_col,
                })
                .collect();
            module.cnv_accumulate_dft(
                cnv_offset,
                &mut res_fused.to_backend_mut(),
                res_col,
                &terms,
                &mut scratch.arena(),
            );
        }

        for (idx, &(a_col, b_col)) in term_cols.iter().enumerate() {
            if idx == 0 {
                module.cnv_apply_dft(
                    cnv_offset,
                    &mut res_ref.to_backend_mut(),
                    res_col,
                    &a_prep.to_backend_ref(),
                    a_col,
                    &b_prep.to_backend_ref(),
                    b_col,
                    &mut scratch.arena(),
                );
            } else {
                module.cnv_apply_dft_accumulate(
                    cnv_offset,
                    &mut res_ref.to_backend_mut(),
                    res_col,
                    &a_prep.to_backend_ref(),
                    a_col,
                    &b_prep.to_backend_ref(),
                    b_col,
                    &mut scratch.arena(),
                );
            }
        }

        // Compare in the normalized coefficient domain.
        module.vec_znx_idft_apply_tmpa(&mut big_fused.to_backend_mut(), 0, &mut res_fused.to_backend_mut(), res_col);
        module.vec_znx_idft_apply_tmpa(&mut big_ref.to_backend_mut(), 0, &mut res_ref.to_backend_mut(), res_col);

        let host_template = VecZnx::alloc(module.n(), 1, res_size);
        let mut have_backend = upload_vec_znx::<BE>(&host_template);
        let mut want_backend = upload_vec_znx::<BE>(&host_template);
        module.vec_znx_big_normalize(
            &mut vec_znx_backend_mut::<BE>(&mut have_backend),
            base2k,
            0,
            0,
            &big_fused.to_backend_ref(),
            base2k,
            0,
            &mut scratch.arena(),
        );
        module.vec_znx_big_normalize(
            &mut vec_znx_backend_mut::<BE>(&mut want_backend),
            base2k,
            0,
            0,
            &big_ref.to_backend_ref(),
            base2k,
            0,
            &mut scratch.arena(),
        );
        let have = download_vec_znx::<BE>(&have_backend);
        let want = download_vec_znx::<BE>(&want_backend);
        assert_eq!(have, want, "fused accumulate != per-term sequence (cnv_offset={cnv_offset})");

        // Multi-column broadcast: `res[c] = Sum_t a[t.a_col + c] (x) b[t.b_col]`,
        // then the same terms accumulated on top. `a_col` is the base column of
        // the broadcast, `b_col` is shared by every output column.
        let broadcast_b_cols: [usize; 2] = [0, 1];
        for store in [CnvDftStore::Overwrite, CnvDftStore::Accumulate] {
            // The comparison below consumes both destinations
            // (`vec_znx_idft_apply_tmpa` is destructive), so every pass rebuilds
            // its own `Overwrite` baseline before optionally accumulating on top.
            for pass in [CnvDftStore::Overwrite, store] {
                let terms: Vec<CnvDftAccTerm<'_, BE>> = broadcast_b_cols
                    .iter()
                    .map(|&b_col| CnvDftAccTerm {
                        a: a_prep.to_backend_ref(),
                        a_col: 0,
                        b: b_prep.to_backend_ref(),
                        b_col,
                    })
                    .collect();
                module.cnv_accumulate_dft_columns(
                    cnv_offset,
                    pass,
                    &mut res_fused.to_backend_mut(),
                    0,
                    cols,
                    &terms,
                    &mut scratch.arena(),
                );

                for col in 0..cols {
                    for (idx, &b_col) in broadcast_b_cols.iter().enumerate() {
                        if idx == 0 && pass == CnvDftStore::Overwrite {
                            module.cnv_apply_dft(
                                cnv_offset,
                                &mut res_ref.to_backend_mut(),
                                col,
                                &a_prep.to_backend_ref(),
                                col,
                                &b_prep.to_backend_ref(),
                                b_col,
                                &mut scratch.arena(),
                            );
                        } else {
                            module.cnv_apply_dft_accumulate(
                                cnv_offset,
                                &mut res_ref.to_backend_mut(),
                                col,
                                &a_prep.to_backend_ref(),
                                col,
                                &b_prep.to_backend_ref(),
                                b_col,
                                &mut scratch.arena(),
                            );
                        }
                    }
                }
                if pass == CnvDftStore::Overwrite && store == CnvDftStore::Overwrite {
                    break;
                }
            }

            for col in 0..cols {
                module.vec_znx_idft_apply_tmpa(&mut big_fused.to_backend_mut(), 0, &mut res_fused.to_backend_mut(), col);
                module.vec_znx_idft_apply_tmpa(&mut big_ref.to_backend_mut(), 0, &mut res_ref.to_backend_mut(), col);
                let mut have_backend = upload_vec_znx::<BE>(&host_template);
                let mut want_backend = upload_vec_znx::<BE>(&host_template);
                module.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BE>(&mut have_backend),
                    base2k,
                    0,
                    0,
                    &big_fused.to_backend_ref(),
                    base2k,
                    0,
                    &mut scratch.arena(),
                );
                module.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BE>(&mut want_backend),
                    base2k,
                    0,
                    0,
                    &big_ref.to_backend_ref(),
                    base2k,
                    0,
                    &mut scratch.arena(),
                );
                assert_eq!(
                    download_vec_znx::<BE>(&have_backend),
                    download_vec_znx::<BE>(&want_backend),
                    "column accumulate != per-column sequence (cnv_offset={cnv_offset}, col={col}, store={store:?})"
                );
            }
        }
    }
}

/// `cnv_accumulate_dft_columns_batch` matches the ordered per-result
/// `cnv_accumulate_dft_columns` calls.
///
/// Sweeps batch lengths 0 to 4, both stores, a zero and two non-zero convolution
/// offsets (leaving a poisoned tail past the convolution bound), heterogeneous
/// result sizes, and term sets with different lengths and order, left operands
/// absent from some lanes, duplicate left identities at different
/// multiplicities, empty sets, and a 33-term set crossing a plausible
/// fused-launch descriptor chunk.
pub fn test_convolution_accumulate_batch<M, BE: crate::test_suite::TestBackend>(module: &M, base2k: usize)
where
    M: ModuleN
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxBigAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    use crate::layouts::CnvDftStore;

    /// One term: which prepared left operand, and which column of the shared
    /// right operand.
    type TermSpec = (usize, usize);

    /// Runs one batch against the ordered ordinary calls and compares every
    /// result column in the coefficient domain.
    #[allow(clippy::too_many_arguments)]
    fn run_case<M, BE: crate::test_suite::TestBackend>(
        module: &M,
        base2k: usize,
        cols: usize,
        a_preps: &[CnvPVecLOwned<BE>],
        b_prep: &CnvPVecROwned<BE>,
        cnv_offset: usize,
        store: CnvDftStore,
        spec: &[Vec<TermSpec>],
        sizes: &[usize],
        scratch: &mut ScratchOwned<BE>,
    ) where
        M: ModuleN + Convolution<BE> + VecZnxDftAlloc<BE> + VecZnxIdftApplyTmpA<BE> + VecZnxBigNormalize<BE> + VecZnxBigAlloc<BE>,
    {
        use crate::layouts::{CnvDftAccTerm, CnvDftStore};

        assert_eq!(spec.len(), sizes.len());
        let mut have: Vec<VecZnxDftOwned<BE>> = sizes.iter().map(|&s| module.vec_znx_dft_alloc(cols, s)).collect();
        let mut want: Vec<VecZnxDftOwned<BE>> = sizes.iter().map(|&s| module.vec_znx_dft_alloc(cols, s)).collect();

        // Seed each pair identically: `Accumulate` then has a base to add onto
        // and `Overwrite` a poisoned tail past the convolution bound to clear.
        for (idx, dst) in have.iter_mut().chain(want.iter_mut()).enumerate() {
            let seed = [CnvDftAccTerm {
                a: a_preps[(idx % sizes.len()) % a_preps.len()].to_backend_ref(),
                a_col: 0,
                b: b_prep.to_backend_ref(),
                b_col: (idx % sizes.len()) % cols,
            }];
            module.cnv_accumulate_dft_columns(
                0,
                CnvDftStore::Overwrite,
                &mut dst.to_backend_mut(),
                0,
                cols,
                &seed,
                &mut scratch.arena(),
            );
        }

        let terms: Vec<Vec<CnvDftAccTerm<'_, BE>>> = spec
            .iter()
            .map(|set| {
                set.iter()
                    .map(|&(a, b_col)| CnvDftAccTerm {
                        a: a_preps[a].to_backend_ref(),
                        a_col: 0,
                        b: b_prep.to_backend_ref(),
                        b_col,
                    })
                    .collect()
            })
            .collect();
        {
            let term_sets: Vec<&[CnvDftAccTerm<'_, BE>]> = terms.iter().map(Vec::as_slice).collect();
            let mut results: Vec<_> = have.iter_mut().map(|r| r.to_backend_mut()).collect();
            module.cnv_accumulate_dft_columns_batch(cnv_offset, store, &mut results, 0, cols, &term_sets, &mut scratch.arena());
        }
        for (dst, set) in want.iter_mut().zip(&terms) {
            module.cnv_accumulate_dft_columns(
                cnv_offset,
                store,
                &mut dst.to_backend_mut(),
                0,
                cols,
                set,
                &mut scratch.arena(),
            );
        }

        for (lane, (&size, (have, want))) in sizes.iter().zip(have.iter_mut().zip(want.iter_mut())).enumerate() {
            let host_template = VecZnx::alloc(module.n(), 1, size);
            let mut big: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(1, size);
            for col in 0..cols {
                // `vec_znx_idft_apply_tmpa` consumes its source column, so each
                // column of each destination is drained exactly once.
                let mut drained = [None, None];
                for (slot, res) in drained.iter_mut().zip([&mut *have, &mut *want]) {
                    module.vec_znx_idft_apply_tmpa(&mut big.to_backend_mut(), 0, &mut res.to_backend_mut(), col);
                    let mut out = upload_vec_znx::<BE>(&host_template);
                    module.vec_znx_big_normalize(
                        &mut vec_znx_backend_mut::<BE>(&mut out),
                        base2k,
                        0,
                        0,
                        &big.to_backend_ref(),
                        base2k,
                        0,
                        &mut scratch.arena(),
                    );
                    *slot = Some(download_vec_znx::<BE>(&out));
                }
                assert_eq!(
                    drained[0],
                    drained[1],
                    "batch accumulate != ordered ordinary calls (cnv_offset={cnv_offset}, store={store:?}, \
                     batch={}, lane={lane}, col={col})",
                    sizes.len()
                );
            }
        }
    }

    let mut source: Source = Source::new([1u8; 32]);

    let cols: usize = 2;
    let a_size: usize = 15;
    let b_size: usize = 15;
    let res_size: usize = a_size + b_size;

    let mut b = VecZnx::alloc(module.n(), cols, b_size);
    b.fill_uniform(17, &mut source);
    let b_backend = upload_vec_znx::<BE>(&b);
    let mut b_prep: CnvPVecROwned<BE> = module.cnv_pvec_right_alloc(cols, b_size);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .cnv_accumulate_dft_tmp_bytes(0, res_size, a_size, b_size)
            .max(module.cnv_prepare_left_tmp_bytes(res_size, a_size))
            .max(module.cnv_prepare_right_tmp_bytes(res_size, b_size))
            .max(module.vec_znx_big_normalize_tmp_bytes()),
    );

    module.cnv_prepare_right(
        &mut b_prep.to_backend_mut(),
        &vec_znx_backend_ref::<BE>(&b_backend),
        !0i64,
        &mut scratch.arena(),
    );

    // Three distinct prepared left operands, so "absent from this lane" and
    // "duplicated in that lane" are observable.
    let a_preps: Vec<CnvPVecLOwned<BE>> = (0..3)
        .map(|_| {
            let mut a = VecZnx::alloc(module.n(), cols, a_size);
            a.fill_uniform(17, &mut source);
            let a_backend = upload_vec_znx::<BE>(&a);
            let mut prep: CnvPVecLOwned<BE> = module.cnv_pvec_left_alloc(cols, a_size);
            module.cnv_prepare_left(
                &mut prep.to_backend_mut(),
                &vec_znx_backend_ref::<BE>(&a_backend),
                !0i64,
                &mut scratch.arena(),
            );
            prep
        })
        .collect();

    let batches: Vec<Vec<Vec<TermSpec>>> = vec![
        vec![],
        vec![vec![(0, 0)]],
        vec![vec![], vec![]],
        vec![vec![(0, 0), (1, 1)], vec![]],
        // Different lengths and order; left operand 1 is absent from lane 1.
        vec![vec![(0, 0), (1, 1), (2, 0)], vec![(2, 0), (0, 1)]],
        // Duplicate left identities at multiplicity 2 / 1 / 2.
        vec![vec![(0, 0), (0, 1), (1, 0)], vec![(0, 0)], vec![(1, 1), (0, 0), (0, 1)]],
        // Four lanes, one of them past a plausible descriptor chunk.
        vec![
            (0..33).map(|i| (i % 3, i % cols)).collect(),
            vec![(2, 1), (2, 0)],
            vec![],
            vec![(1, 0), (0, 1), (1, 1)],
        ],
    ];

    for cnv_offset in [0usize, 3, 11] {
        for store in [CnvDftStore::Overwrite, CnvDftStore::Accumulate] {
            for spec in &batches {
                let sizes: Vec<usize> = vec![res_size; spec.len()];
                run_case(
                    module,
                    base2k,
                    cols,
                    &a_preps,
                    &b_prep,
                    cnv_offset,
                    store,
                    spec,
                    &sizes,
                    &mut scratch,
                );
            }

            // Heterogeneous result sizes: lane 1 truncates the convolution
            // earlier than lane 0.
            let spec = vec![vec![(0, 0), (1, 1)], vec![(1, 0), (2, 1), (0, 0)]];
            run_case(
                module,
                base2k,
                cols,
                &a_preps,
                &b_prep,
                cnv_offset,
                store,
                &spec,
                &[res_size, res_size - 4],
                &mut scratch,
            );
        }
    }
}

pub fn test_convolution_pairwise<M, BE: crate::test_suite::TestBackend>(module: &M, base2k: usize)
where
    M: ModuleN
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxNormalizeAssignBackend<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxAddIntoBackend<BE>
        + VecZnxCopyBackend<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let mut source: Source = Source::new([0u8; 32]);

    let cols: usize = 2;
    let a_size: usize = 15;
    let b_size: usize = 15;
    let res_size: usize = a_size + b_size;

    let mut a = VecZnx::alloc(module.n(), cols, a_size);
    let mut b = VecZnx::alloc(module.n(), cols, b_size);
    let mut tmp_a = VecZnx::alloc(module.n(), 1, a_size);
    let mut tmp_b = VecZnx::alloc(module.n(), 1, b_size);

    let mut res_want = VecZnx::alloc(module.n(), 1, res_size);
    // Two-column DFT destination written at column 1: covers the
    // column-interleaved `VecZnxDft` indexing of the backend kernels.
    let res_dft_col: usize = 1;
    let mut res_dft: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(2, res_size);
    let mut res_big: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(1, res_size);

    a.fill_uniform(17, &mut source);
    b.fill_uniform(17, &mut source);

    let a_backend = upload_vec_znx::<BE>(&a);
    let b_backend = upload_vec_znx::<BE>(&b);

    let mut a_prep: CnvPVecLOwned<BE> = module.cnv_pvec_left_alloc(cols, a_size);
    let mut b_prep: CnvPVecROwned<BE> = module.cnv_pvec_right_alloc(cols, b_size);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .cnv_pairwise_apply_dft_tmp_bytes(0, res_size, a_size, b_size)
            .max(module.cnv_prepare_left_tmp_bytes(res_size, a_size))
            .max(module.cnv_prepare_right_tmp_bytes(res_size, b_size))
            .max(module.vec_znx_big_normalize_tmp_bytes()),
    );

    {
        let mut a_prep_backend = a_prep.to_backend_mut();
        module.cnv_prepare_left(
            &mut a_prep_backend,
            &vec_znx_backend_ref::<BE>(&a_backend),
            !0i64,
            &mut scratch.arena(),
        );
    }
    {
        let mut b_prep_backend = b_prep.to_backend_mut();
        module.cnv_prepare_right(
            &mut b_prep_backend,
            &vec_znx_backend_ref::<BE>(&b_backend),
            !0i64,
            &mut scratch.arena(),
        );
    }

    for col_i in 0..cols {
        for col_j in 0..cols {
            for cnv_offset in 0..res_size {
                module.cnv_pairwise_apply_dft(
                    cnv_offset,
                    &mut res_dft.to_backend_mut(),
                    res_dft_col,
                    &a_prep.to_backend_ref(),
                    &b_prep.to_backend_ref(),
                    col_i,
                    col_j,
                    &mut scratch.arena(),
                );

                module.vec_znx_idft_apply_tmpa(&mut res_big.to_backend_mut(), 0, &mut res_dft.to_backend_mut(), res_dft_col);

                let res_host_template = VecZnx::alloc(module.n(), 1, res_size);
                let mut res_have_backend = upload_vec_znx::<BE>(&res_host_template);
                module.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BE>(&mut res_have_backend),
                    base2k,
                    0,
                    0,
                    &res_big.to_backend_ref(),
                    base2k,
                    0,
                    &mut scratch.arena(),
                );
                let res_have = download_vec_znx::<BE>(&res_have_backend);

                let mut tmp_a_backend = upload_vec_znx::<BE>(&tmp_a);
                let mut tmp_b_backend = upload_vec_znx::<BE>(&tmp_b);
                if col_i != col_j {
                    module.vec_znx_add_into_backend(
                        &mut vec_znx_backend_mut::<BE>(&mut tmp_a_backend),
                        0,
                        &vec_znx_backend_ref::<BE>(&a_backend),
                        col_i,
                        &vec_znx_backend_ref::<BE>(&a_backend),
                        col_j,
                    );
                    module.vec_znx_add_into_backend(
                        &mut vec_znx_backend_mut::<BE>(&mut tmp_b_backend),
                        0,
                        &vec_znx_backend_ref::<BE>(&b_backend),
                        col_i,
                        &vec_znx_backend_ref::<BE>(&b_backend),
                        col_j,
                    );
                } else {
                    module.vec_znx_copy_backend(
                        &mut vec_znx_backend_mut::<BE>(&mut tmp_a_backend),
                        0,
                        &vec_znx_backend_ref::<BE>(&a_backend),
                        col_i,
                    );
                    module.vec_znx_copy_backend(
                        &mut vec_znx_backend_mut::<BE>(&mut tmp_b_backend),
                        0,
                        &vec_znx_backend_ref::<BE>(&b_backend),
                        col_j,
                    );
                }

                tmp_a = download_vec_znx::<BE>(&tmp_a_backend);
                tmp_b = download_vec_znx::<BE>(&tmp_b_backend);

                bivariate_convolution_naive(
                    module,
                    base2k,
                    (cnv_offset + 1) as i64,
                    &mut res_want,
                    0,
                    &tmp_a,
                    0,
                    &tmp_b,
                    0,
                    &mut scratch.arena(),
                );

                assert_eq!(res_want, res_have);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn bivariate_convolution_naive<M, BE: crate::test_suite::TestBackend>(
    module: &M,
    base2k: usize,
    k: i64,
    res: &mut VecZnxOwned<BE::ZnxWord>,
    res_col: usize,
    a: &VecZnxOwned<BE::ZnxWord>,
    a_col: usize,
    b: &VecZnxOwned<BE::ZnxWord>,
    b_col: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    M: VecZnxNormalizeAssignBackend<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    for j in 0..res.size() {
        res.zero_at(res_col, j);
    }

    for a_limb in 0..a.size() {
        for b_limb in 0..b.size() {
            let res_scale_abs = k.unsigned_abs() as usize;

            let mut res_limb: usize = a_limb + b_limb + 1;

            if k <= 0 {
                res_limb += res_scale_abs;

                if res_limb < res.size() {
                    negacyclic_convolution_naive_add(res.at_mut(res_col, res_limb), a.at(a_col, a_limb), b.at(b_col, b_limb));
                }
            } else if res_limb >= res_scale_abs {
                res_limb -= res_scale_abs;

                if res_limb < res.size() {
                    negacyclic_convolution_naive_add(res.at_mut(res_col, res_limb), a.at(a_col, a_limb), b.at(b_col, b_limb));
                }
            }
        }
    }

    let mut res_backend = upload_vec_znx::<BE>(res);
    module.vec_znx_normalize_assign_backend(base2k, &mut vec_znx_backend_mut::<BE>(&mut res_backend), res_col, scratch);
    *res = download_vec_znx::<BE>(&res_backend);
}

fn bivariate_tensoring_naive<M, BE: crate::test_suite::TestBackend>(
    module: &M,
    base2k: usize,
    k: i64,
    res: &mut VecZnxOwned<BE::ZnxWord>,
    a: &VecZnxOwned<BE::ZnxWord>,
    b: &VecZnxOwned<BE::ZnxWord>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    M: VecZnxNormalizeAssignBackend<BE>,
{
    let cols = res.cols();

    assert!(res.cols() >= a.cols() + b.cols() - 1);

    res.zero();

    for a_col in 0..a.cols() {
        for a_limb in 0..a.size() {
            for b_col in 0..b.cols() {
                for b_limb in 0..b.size() {
                    let res_scale_abs = k.unsigned_abs() as usize;

                    let mut res_limb: usize = a_limb + b_limb + 1;

                    if k <= 0 {
                        res_limb += res_scale_abs;

                        if res_limb < res.size() {
                            negacyclic_convolution_naive_add(
                                res.at_mut(a_col + b_col, res_limb),
                                a.at(a_col, a_limb),
                                b.at(b_col, b_limb),
                            );
                        }
                    } else if res_limb >= res_scale_abs {
                        res_limb -= res_scale_abs;

                        if res_limb < res.size() {
                            negacyclic_convolution_naive_add(
                                res.at_mut(a_col + b_col, res_limb),
                                a.at(a_col, a_limb),
                                b.at(b_col, b_limb),
                            );
                        }
                    }
                }
            }
        }
    }

    let mut res_backend = upload_vec_znx::<BE>(res);
    for i in 0..cols {
        module.vec_znx_normalize_assign_backend(base2k, &mut vec_znx_backend_mut::<BE>(&mut res_backend), i, scratch);
    }
    *res = download_vec_znx::<BE>(&res_backend);
}

fn negacyclic_convolution_naive_add(res: &mut [i64], a: &[i64], b: &[i64]) {
    let n: usize = res.len();
    for i in 0..n {
        let ai: i64 = a[i];
        let lim: usize = n - i;
        for j in 0..lim {
            res[i + j] += ai * b[j];
        }
        for j in lim..n {
            res[i + j - n] -= ai * b[j];
        }
    }
}

fn negacyclic_convolution_naive(res: &mut [i64], a: &[i64], b: &[i64]) {
    let n: usize = res.len();
    res.fill(0);
    for i in 0..n {
        let ai: i64 = a[i];
        let lim: usize = n - i;
        for j in 0..lim {
            res[i + j] += ai * b[j];
        }
        for j in lim..n {
            res[i + j - n] -= ai * b[j];
        }
    }
}
