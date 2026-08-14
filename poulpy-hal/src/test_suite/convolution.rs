use super::{download_vec_znx, upload_vec_znx, vec_znx_backend_mut, vec_znx_backend_ref};
use crate::layouts::CnvPVecLToBackendMut;
use crate::layouts::CnvPVecLToBackendRef;
use crate::layouts::CnvPVecRToBackendMut;
use crate::layouts::CnvPVecRToBackendRef;
use crate::layouts::VecZnxBigToBackendMut;
use crate::layouts::VecZnxBigToBackendRef;
use crate::layouts::VecZnxDftToBackendMut;
use crate::layouts::VecZnxDftToBackendRef;
use crate::layouts::{
    CnvDftAccTermPvec, CnvDftAccTermTvec, CnvTVecLOwned, CnvTVecLToBackendMut, CnvTVecLToBackendRef, CnvTVecROwned,
    CnvTVecRToBackendMut, CnvTVecRToBackendRef,
};
use rand::Rng;

use crate::{
    api::{
        CnvPVecAlloc, CnvTVecAlloc, Convolution, ModuleN, ScratchOwnedAlloc, VecZnxAddIntoBackend, VecZnxBigAlloc,
        VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxCopyBackend, VecZnxDftAddAssign, VecZnxDftAlloc, VecZnxDftApply,
        VecZnxIdftApplyTmpA, VecZnxNormalizeAssignBackend,
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
            .cnv_apply_pvec_to_dft_tmp_bytes(0, res_size, a_size, b_size)
            .max(module.cnv_prepare_left_pvec_tmp_bytes(res_size, a_size))
            .max(module.cnv_prepare_right_pvec_tmp_bytes(res_size, b_size))
            .max(module.vec_znx_big_normalize_tmp_bytes()),
    );

    {
        let mut a_prep_backend = a_prep.to_backend_mut();
        module.cnv_prepare_left_pvec(
            &mut a_prep_backend,
            &vec_znx_backend_ref::<BE>(&a_backend),
            !0i64,
            &mut scratch.arena(),
        );
    }
    {
        let mut b_prep_backend = b_prep.to_backend_mut();
        module.cnv_prepare_right_pvec(
            &mut b_prep_backend,
            &vec_znx_backend_ref::<BE>(&b_backend),
            !0i64,
            &mut scratch.arena(),
        );
    }

    for a_col in 0..a.cols() {
        for b_col in 0..b.cols() {
            for cnv_offset in 0..res_size {
                module.cnv_apply_pvec_to_dft(
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

/// `cnv_apply_pvec_to_dft_accumulate` matches `cnv_apply_pvec_to_dft` followed by a DFT add,
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
            .cnv_apply_pvec_to_dft_tmp_bytes(0, res_size, a_size, b_size)
            .max(module.cnv_prepare_left_pvec_tmp_bytes(res_size, a_size))
            .max(module.cnv_prepare_right_pvec_tmp_bytes(res_size, b_size)),
    );

    {
        let mut a_prep_backend = a_prep.to_backend_mut();
        module.cnv_prepare_left_pvec(
            &mut a_prep_backend,
            &vec_znx_backend_ref::<BE>(&a_backend),
            !0i64,
            &mut scratch.arena(),
        );
    }
    {
        let mut b_prep_backend = b_prep.to_backend_mut();
        module.cnv_prepare_right_pvec(
            &mut b_prep_backend,
            &vec_znx_backend_ref::<BE>(&b_backend),
            !0i64,
            &mut scratch.arena(),
        );
    }

    for res_col in 0..res_cols {
        // Identical deterministic initial accumulator content for both paths.
        module.cnv_apply_pvec_to_dft(
            0,
            &mut res_acc.to_backend_mut(),
            res_col,
            &a_prep.to_backend_ref(),
            0,
            &b_prep.to_backend_ref(),
            0,
            &mut scratch.arena(),
        );
        module.cnv_apply_pvec_to_dft(
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
                    module.cnv_apply_pvec_to_dft_accumulate(
                        cnv_offset,
                        &mut res_acc.to_backend_mut(),
                        res_col,
                        &a_prep.to_backend_ref(),
                        a_col,
                        &b_prep.to_backend_ref(),
                        b_col,
                        &mut scratch.arena(),
                    );

                    module.cnv_apply_pvec_to_dft(
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

/// `cnv_accumulate_pvec_to_dft` matches the per-term `cnv_apply_pvec_to_dft` +
/// `cnv_apply_pvec_to_dft_accumulate` sequence after normalization to the coefficient
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
    use crate::layouts::CnvDftAccTermPvec;

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
            .cnv_accumulate_pvec_to_dft_tmp_bytes(0, res_size, a_size, b_size)
            .max(module.cnv_apply_pvec_to_dft_tmp_bytes(0, res_size, a_size, b_size))
            .max(module.cnv_prepare_left_pvec_tmp_bytes(res_size, a_size))
            .max(module.cnv_prepare_right_pvec_tmp_bytes(res_size, b_size))
            .max(module.vec_znx_big_normalize_tmp_bytes()),
    );

    {
        let mut a_prep_backend = a_prep.to_backend_mut();
        module.cnv_prepare_left_pvec(
            &mut a_prep_backend,
            &vec_znx_backend_ref::<BE>(&a_backend),
            !0i64,
            &mut scratch.arena(),
        );
    }
    {
        let mut b_prep_backend = b_prep.to_backend_mut();
        module.cnv_prepare_right_pvec(
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
            let terms: Vec<CnvDftAccTermPvec<'_, BE>> = term_cols
                .iter()
                .map(|&(a_col, b_col)| CnvDftAccTermPvec::new(a_prep.to_backend_ref(), a_col, b_prep.to_backend_ref(), b_col))
                .collect();
            module.cnv_accumulate_pvec_to_dft(
                cnv_offset,
                &mut res_fused.to_backend_mut(),
                res_col,
                &terms,
                &mut scratch.arena(),
            );
        }

        for (idx, &(a_col, b_col)) in term_cols.iter().enumerate() {
            if idx == 0 {
                module.cnv_apply_pvec_to_dft(
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
                module.cnv_apply_pvec_to_dft_accumulate(
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
            .cnv_pairwise_apply_pvec_to_dft_tmp_bytes(0, res_size, a_size, b_size)
            .max(module.cnv_prepare_left_pvec_tmp_bytes(res_size, a_size))
            .max(module.cnv_prepare_right_pvec_tmp_bytes(res_size, b_size))
            .max(module.vec_znx_big_normalize_tmp_bytes()),
    );

    {
        let mut a_prep_backend = a_prep.to_backend_mut();
        module.cnv_prepare_left_pvec(
            &mut a_prep_backend,
            &vec_znx_backend_ref::<BE>(&a_backend),
            !0i64,
            &mut scratch.arena(),
        );
    }
    {
        let mut b_prep_backend = b_prep.to_backend_mut();
        module.cnv_prepare_right_pvec(
            &mut b_prep_backend,
            &vec_znx_backend_ref::<BE>(&b_backend),
            !0i64,
            &mut scratch.arena(),
        );
    }

    for col_i in 0..cols {
        for col_j in 0..cols {
            for cnv_offset in 0..res_size {
                module.cnv_pairwise_apply_pvec_to_dft(
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

/// Pass `all_forms = false` for a backend whose `tvec` tier only implements the
/// plain apply.
///
/// The `tvec` (hot-prep) tier computes the same convolution as the `pvec`
/// (cold-prep) tier, for every DFT-domain form.
///
/// Compared after inverse DFT and normalization rather than on the prepared
/// buffers, so it holds even where the two tiers use different internal layouts.
pub fn test_convolution_tier_equivalence<M, BE: crate::test_suite::TestBackend>(module: &M, base2k: usize, all_forms: bool)
where
    M: ModuleN
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + CnvTVecAlloc<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxBigAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let mut source: Source = Source::new([1u8; 32]);

    let cols: usize = 2;
    let a_size: usize = 9;
    let b_size: usize = 9;
    let res_size: usize = a_size + b_size;

    let mut a = VecZnx::alloc(module.n(), cols, a_size);
    let mut b = VecZnx::alloc(module.n(), cols, b_size);
    a.fill_uniform(17, &mut source);
    b.fill_uniform(17, &mut source);

    let a_backend = upload_vec_znx::<BE>(&a);
    let b_backend = upload_vec_znx::<BE>(&b);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .cnv_apply_pvec_to_dft_tmp_bytes(0, res_size, a_size, b_size)
            .max(module.cnv_apply_tvec_to_dft_tmp_bytes(0, res_size, a_size, b_size))
            .max(module.cnv_accumulate_pvec_to_dft_tmp_bytes(0, res_size, a_size, b_size))
            .max(module.cnv_accumulate_tvec_to_dft_tmp_bytes(0, res_size, a_size, b_size))
            .max(module.cnv_pairwise_apply_pvec_to_dft_tmp_bytes(0, res_size, a_size, b_size))
            .max(module.cnv_pairwise_apply_tvec_to_dft_tmp_bytes(0, res_size, a_size, b_size))
            .max(module.cnv_prepare_left_pvec_tmp_bytes(res_size, a_size))
            .max(module.cnv_prepare_right_pvec_tmp_bytes(res_size, b_size))
            .max(module.cnv_prepare_left_tvec_tmp_bytes(res_size, a_size))
            .max(module.cnv_prepare_right_tvec_tmp_bytes(res_size, b_size))
            .max(module.vec_znx_big_normalize_tmp_bytes()),
    );

    let mut a_p: CnvPVecLOwned<BE> = module.cnv_pvec_left_alloc(cols, a_size);
    let mut b_p: CnvPVecROwned<BE> = module.cnv_pvec_right_alloc(cols, b_size);
    let mut a_t: CnvTVecLOwned<BE> = module.cnv_tvec_left_alloc(cols, a_size);
    let mut b_t: CnvTVecROwned<BE> = module.cnv_tvec_right_alloc(cols, b_size);

    module.cnv_prepare_left_pvec(
        &mut a_p.to_backend_mut(),
        &vec_znx_backend_ref::<BE>(&a_backend),
        !0i64,
        &mut scratch.arena(),
    );
    module.cnv_prepare_right_pvec(
        &mut b_p.to_backend_mut(),
        &vec_znx_backend_ref::<BE>(&b_backend),
        !0i64,
        &mut scratch.arena(),
    );
    module.cnv_prepare_left_tvec(
        &mut a_t.to_backend_mut(),
        &vec_znx_backend_ref::<BE>(&a_backend),
        !0i64,
        &mut scratch.arena(),
    );
    module.cnv_prepare_right_tvec(
        &mut b_t.to_backend_mut(),
        &vec_znx_backend_ref::<BE>(&b_backend),
        !0i64,
        &mut scratch.arena(),
    );

    let mut res_dft: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(1, res_size);
    let mut res_big: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(1, res_size);

    // Inverse-DFT and normalize the current `res_dft` into a host `VecZnx`.
    macro_rules! settle {
        () => {{
            module.vec_znx_idft_apply_tmpa(&mut res_big.to_backend_mut(), 0, &mut res_dft.to_backend_mut(), 0);
            let mut out = upload_vec_znx::<BE>(&VecZnx::alloc(module.n(), 1, res_size));
            module.vec_znx_big_normalize(
                &mut vec_znx_backend_mut::<BE>(&mut out),
                base2k,
                0,
                0,
                &res_big.to_backend_ref(),
                base2k,
                0,
                &mut scratch.arena(),
            );
            download_vec_znx::<BE>(&out)
        }};
    }

    for cnv_offset in [0usize, 1, res_size - 1] {
        for a_col in 0..cols {
            for b_col in 0..cols {
                // 1. apply_to_dft
                module.cnv_apply_pvec_to_dft(
                    cnv_offset,
                    &mut res_dft.to_backend_mut(),
                    0,
                    &a_p.to_backend_ref(),
                    a_col,
                    &b_p.to_backend_ref(),
                    b_col,
                    &mut scratch.arena(),
                );
                let want = settle!();
                module.cnv_apply_tvec_to_dft(
                    cnv_offset,
                    &mut res_dft.to_backend_mut(),
                    0,
                    &a_t.to_backend_ref(),
                    a_col,
                    &b_t.to_backend_ref(),
                    b_col,
                    &mut scratch.arena(),
                );
                assert_eq!(want, settle!(), "apply_to_dft: tvec != pvec");

                if !all_forms {
                    continue;
                }

                // 2. apply_to_dft_accumulate, on top of the result just verified in 1
                module.cnv_apply_pvec_to_dft(
                    cnv_offset,
                    &mut res_dft.to_backend_mut(),
                    0,
                    &a_p.to_backend_ref(),
                    a_col,
                    &b_p.to_backend_ref(),
                    b_col,
                    &mut scratch.arena(),
                );
                module.cnv_apply_pvec_to_dft_accumulate(
                    cnv_offset,
                    &mut res_dft.to_backend_mut(),
                    0,
                    &a_p.to_backend_ref(),
                    a_col,
                    &b_p.to_backend_ref(),
                    b_col,
                    &mut scratch.arena(),
                );
                let want = settle!();
                module.cnv_apply_tvec_to_dft(
                    cnv_offset,
                    &mut res_dft.to_backend_mut(),
                    0,
                    &a_t.to_backend_ref(),
                    a_col,
                    &b_t.to_backend_ref(),
                    b_col,
                    &mut scratch.arena(),
                );
                module.cnv_apply_tvec_to_dft_accumulate(
                    cnv_offset,
                    &mut res_dft.to_backend_mut(),
                    0,
                    &a_t.to_backend_ref(),
                    a_col,
                    &b_t.to_backend_ref(),
                    b_col,
                    &mut scratch.arena(),
                );
                assert_eq!(want, settle!(), "apply_to_dft_accumulate: tvec != pvec");

                // 3. pairwise_apply_to_dft
                module.cnv_pairwise_apply_pvec_to_dft(
                    cnv_offset,
                    &mut res_dft.to_backend_mut(),
                    0,
                    &a_p.to_backend_ref(),
                    &b_p.to_backend_ref(),
                    a_col,
                    b_col,
                    &mut scratch.arena(),
                );
                let want = settle!();
                module.cnv_pairwise_apply_tvec_to_dft(
                    cnv_offset,
                    &mut res_dft.to_backend_mut(),
                    0,
                    &a_t.to_backend_ref(),
                    &b_t.to_backend_ref(),
                    a_col,
                    b_col,
                    &mut scratch.arena(),
                );
                assert_eq!(want, settle!(), "pairwise_apply_to_dft: tvec != pvec");
            }
        }

        if !all_forms {
            continue;
        }

        // 4. accumulate_to_dft over a multi-term slice
        let terms_p: Vec<CnvDftAccTermPvec<'_, BE>> = (0..cols)
            .map(|c| CnvDftAccTermPvec::new(a_p.to_backend_ref(), c, b_p.to_backend_ref(), c))
            .collect();
        module.cnv_accumulate_pvec_to_dft(cnv_offset, &mut res_dft.to_backend_mut(), 0, &terms_p, &mut scratch.arena());
        let want = settle!();

        let terms_t: Vec<CnvDftAccTermTvec<'_, BE>> = (0..cols)
            .map(|c| CnvDftAccTermTvec::new(a_t.to_backend_ref(), c, b_t.to_backend_ref(), c))
            .collect();
        module.cnv_accumulate_tvec_to_dft(cnv_offset, &mut res_dft.to_backend_mut(), 0, &terms_t, &mut scratch.arena());
        assert_eq!(want, settle!(), "accumulate_to_dft: tvec != pvec");
    }
}
