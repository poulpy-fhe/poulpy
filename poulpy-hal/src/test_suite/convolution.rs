use super::{TestParams, download_vec_znx, upload_vec_znx, vec_znx_backend_mut, vec_znx_backend_ref};
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
        CnvPVecAlloc, Convolution, ModuleN, ScratchOwnedAlloc, VecZnxAdd, VecZnxBigAlloc, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxCopy, VecZnxDftAddAssign, VecZnxDftAlloc, VecZnxDftApply, VecZnxIdftApplyTmpA,
        VecZnxNormalizeAssign, VecZnxSwitchRing,
    },
    layouts::{
        CnvDftAccTerm, CnvPVecL, CnvPVecR, DataView, FillUniform, Module, PrepareHint, ScratchArena, ScratchOwned, VecZnx,
        VecZnxOwned, ZnxView, ZnxViewMut, ZnxZero,
    },
    source::Source,
};

use crate::layouts::VecZnxDftOwned;
use crate::layouts::{CnvPVecLOwned, CnvPVecROwned, VecZnxBigOwned};

pub fn test_convolution_by_const<M, BE: crate::test_suite::TestBackend>(module: &M, n: usize, base2k: usize)
where
    M: ModuleN
        + Convolution<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxNormalizeAssign<BE>
        + VecZnxBigAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let mut source: Source = Source::new([0u8; 32]);

    let a_cols: usize = 2;
    let a_size: usize = 15;
    let b_size: usize = 15;
    let res_size: usize = a_size + b_size;

    let mut a = VecZnx::alloc(n, a_cols, a_size);
    let mut b = VecZnx::alloc(n, 1, b_size);

    let mut res_want = VecZnx::alloc(n, 1, res_size);
    let mut res_big: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(n, 1, res_size);

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

            let res_host_template = VecZnx::alloc(n, 1, res_size);
            let mut res_have_backend = upload_vec_znx::<BE>(&res_host_template);
            module.vec_znx_big_normalize(
                &mut vec_znx_backend_mut::<BE>(&mut res_have_backend),
                base2k,
                res_size * base2k,
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

/// Verifies `cnv_by_const_apply_add` against `cnv_by_const_apply` +
/// `vec_znx_big_add_assign`, including the untouched-limb contract.
pub fn test_convolution_by_const_add<M, BE: crate::test_suite::TestBackend>(module: &M, n: usize, base2k: usize)
where
    M: ModuleN
        + Convolution<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + crate::api::VecZnxBigAddAssign<BE>
        + VecZnxBigAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let mut source: Source = Source::new([0u8; 32]);

    let a_cols: usize = 2;
    let a_size: usize = 15;
    let b_size: usize = 3;
    let res_size: usize = a_size + b_size;

    let mut a = VecZnx::alloc(n, a_cols, a_size);
    let mut b = VecZnx::alloc(n, 1, b_size);
    a.fill_uniform(17, &mut source);
    b.fill_uniform(base2k, &mut source);

    let a_backend = upload_vec_znx::<BE>(&a);
    let b_backend = upload_vec_znx::<BE>(&b);
    let mut acc_have: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(n, 1, res_size);
    let mut acc_want: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(n, 1, res_size);
    let mut term: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(n, 1, res_size);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .cnv_by_const_apply_tmp_bytes(0, res_size, a_size, b_size)
            .max(module.vec_znx_big_normalize_tmp_bytes()),
    );

    // `cnv_offset` is a limb count; the tail values are clamped out of range.
    for base_offset in [0, 1, 2, res_size / 2, res_size - 1] {
        for add_offset in [0, 1, 2, res_size / 2, res_size - 1, res_size, res_size + 1] {
            // Fused: apply the first term, accumulate the second in place.
            module.cnv_by_const_apply(
                base_offset,
                &mut acc_have.to_backend_mut(),
                0,
                &vec_znx_backend_ref::<BE>(&a_backend),
                0,
                &vec_znx_backend_ref::<BE>(&b_backend),
                0,
                0,
                &mut scratch.arena(),
            );
            module.cnv_by_const_apply_add(
                add_offset,
                &mut acc_have.to_backend_mut(),
                0,
                &vec_znx_backend_ref::<BE>(&a_backend),
                1,
                &vec_znx_backend_ref::<BE>(&b_backend),
                0,
                1,
                &mut scratch.arena(),
            );

            // Composition: both terms into separate BIG buffers, then added.
            module.cnv_by_const_apply(
                base_offset,
                &mut acc_want.to_backend_mut(),
                0,
                &vec_znx_backend_ref::<BE>(&a_backend),
                0,
                &vec_znx_backend_ref::<BE>(&b_backend),
                0,
                0,
                &mut scratch.arena(),
            );
            module.cnv_by_const_apply(
                add_offset,
                &mut term.to_backend_mut(),
                0,
                &vec_znx_backend_ref::<BE>(&a_backend),
                1,
                &vec_znx_backend_ref::<BE>(&b_backend),
                0,
                1,
                &mut scratch.arena(),
            );
            module.vec_znx_big_add_assign(&mut acc_want.to_backend_mut(), 0, &term.to_backend_ref(), 0);

            let normalized = |big: &VecZnxBigOwned<BE>, scratch: &mut ScratchOwned<BE>| {
                let res_host_template = VecZnx::alloc(n, 1, res_size);
                let mut res_backend = upload_vec_znx::<BE>(&res_host_template);
                module.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BE>(&mut res_backend),
                    base2k,
                    res_size * base2k,
                    0,
                    0,
                    &big.to_backend_ref(),
                    base2k,
                    0,
                    &mut scratch.arena(),
                );
                download_vec_znx::<BE>(&res_backend)
            };
            let res_have = normalized(&acc_have, &mut scratch);
            let res_want = normalized(&acc_want, &mut scratch);
            assert_eq!(
                res_want, res_have,
                "cnv_by_const_apply_add != apply + big add for offsets ({base_offset}, {add_offset})"
            );
        }
    }
}

pub fn test_convolution<M, BE: crate::test_suite::TestBackend>(module: &M, n: usize, base2k: usize)
where
    M: ModuleN
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxNormalizeAssign<BE>
        + VecZnxBigAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let mut source: Source = Source::new([0u8; 32]);

    let a_cols: usize = 2;
    let b_cols: usize = 2;
    let a_size: usize = 15;
    let b_size: usize = 15;
    let res_size: usize = a_size + b_size;

    let mut a = VecZnx::alloc(n, a_cols, a_size);
    let mut b = VecZnx::alloc(n, b_cols, b_size);

    let mut res_want = VecZnx::alloc(n, 1, res_size);
    // Two-column DFT destination written at column 1: covers the
    // column-interleaved `VecZnxDft` indexing of the backend kernels.
    let res_dft_col: usize = 1;
    let mut res_dft: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(n, 2, res_size);
    let mut res_big: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(n, 1, res_size);

    a.fill_uniform(17, &mut source);
    b.fill_uniform(17, &mut source);

    let a_backend = upload_vec_znx::<BE>(&a);
    let b_backend = upload_vec_znx::<BE>(&b);

    let mut a_prep: CnvPVecLOwned<BE> = module.cnv_pvec_left_alloc(n, a_cols, a_size, PrepareHint::Reuse);
    let mut b_prep: CnvPVecROwned<BE> = module.cnv_pvec_right_alloc(n, b_cols, b_size, PrepareHint::Reuse);

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
            &mut scratch.arena(),
        );
    }
    {
        let mut b_prep_backend = b_prep.to_backend_mut();
        module.cnv_prepare_right(
            &mut b_prep_backend,
            &vec_znx_backend_ref::<BE>(&b_backend),
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

                let res_host_template = VecZnx::alloc(n, 1, res_size);
                let mut res_have_backend = upload_vec_znx::<BE>(&res_host_template);
                module.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BE>(&mut res_have_backend),
                    base2k,
                    res_size * base2k,
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

/// `cnv_apply_dft_add` matches `cnv_apply_dft` followed by a DFT add,
/// bit-for-bit on the raw prepared data.
pub fn test_convolution_add<M, BE: crate::test_suite::TestBackend>(module: &M, n: usize, _base2k: usize)
where
    M: ModuleN + Convolution<BE> + CnvPVecAlloc<BE> + VecZnxDftAlloc<BE> + VecZnxDftAddAssign<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let mut source: Source = Source::new([0u8; 32]);

    let cols: usize = 2;
    let a_size: usize = 15;
    let b_size: usize = 15;
    let res_size: usize = a_size + b_size;

    let mut a = VecZnx::alloc(n, cols, a_size);
    let mut b = VecZnx::alloc(n, cols, b_size);
    a.fill_uniform(17, &mut source);
    b.fill_uniform(17, &mut source);

    let a_backend = upload_vec_znx::<BE>(&a);
    let b_backend = upload_vec_znx::<BE>(&b);

    let mut a_prep: CnvPVecLOwned<BE> = module.cnv_pvec_left_alloc(n, cols, a_size, PrepareHint::Reuse);
    let mut b_prep: CnvPVecROwned<BE> = module.cnv_pvec_right_alloc(n, cols, b_size, PrepareHint::Reuse);

    // Two-column accumulators exercised at both columns: covers the
    // column-interleaved `VecZnxDft` indexing of the backend kernels.
    let res_cols: usize = 2;
    let mut res_acc: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(n, res_cols, res_size);
    let mut res_ref: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(n, res_cols, res_size);
    let mut tmp_dft: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(n, 1, res_size);

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
            &mut scratch.arena(),
        );
    }
    {
        let mut b_prep_backend = b_prep.to_backend_mut();
        module.cnv_prepare_right(
            &mut b_prep_backend,
            &vec_znx_backend_ref::<BE>(&b_backend),
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
                    module.cnv_apply_dft_add(
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

/// `cnv_apply_dft_sum` matches the per-term `cnv_apply_dft` +
/// `cnv_apply_dft_add` sequence after normalization to the coefficient
/// domain (the fused path reduces once per output, so the raw q120 lazy
/// representatives may differ).
pub fn test_convolution_sum<M, BE: crate::test_suite::TestBackend>(module: &M, n: usize, base2k: usize)
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
    use crate::layouts::CnvDftAccTerm;

    let mut source: Source = Source::new([0u8; 32]);

    let cols: usize = 2;
    let a_size: usize = 15;
    let b_size: usize = 15;
    let res_size: usize = a_size + b_size;
    // Two-column destination written at column 1: covers the column-interleaved
    // `VecZnxDft` indexing of the backend kernels.
    let res_col: usize = 1;

    let mut a = VecZnx::alloc(n, cols, a_size);
    let mut b = VecZnx::alloc(n, cols, b_size);
    a.fill_uniform(17, &mut source);
    b.fill_uniform(17, &mut source);

    let a_backend = upload_vec_znx::<BE>(&a);
    let b_backend = upload_vec_znx::<BE>(&b);

    let mut a_prep: CnvPVecLOwned<BE> = module.cnv_pvec_left_alloc(n, cols, a_size, PrepareHint::Reuse);
    let mut b_prep: CnvPVecROwned<BE> = module.cnv_pvec_right_alloc(n, cols, b_size, PrepareHint::Reuse);

    let mut res_fused: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(n, 2, res_size);
    let mut res_ref: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(n, 2, res_size);
    let mut big_fused: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(n, 1, res_size);
    let mut big_ref: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(n, 1, res_size);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .cnv_apply_dft_sum_tmp_bytes(0, res_size, a_size, b_size)
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
            &mut scratch.arena(),
        );
    }
    {
        let mut b_prep_backend = b_prep.to_backend_mut();
        module.cnv_prepare_right(
            &mut b_prep_backend,
            &vec_znx_backend_ref::<BE>(&b_backend),
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
            module.cnv_apply_dft_sum(
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
                module.cnv_apply_dft_add(
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

        let host_template = VecZnx::alloc(n, 1, res_size);
        let mut have_backend = upload_vec_znx::<BE>(&host_template);
        let mut want_backend = upload_vec_znx::<BE>(&host_template);
        module.vec_znx_big_normalize(
            &mut vec_znx_backend_mut::<BE>(&mut have_backend),
            base2k,
            res_size * base2k,
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
            res_size * base2k,
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

pub fn test_convolution_pairwise<M, BE: crate::test_suite::TestBackend>(module: &M, n: usize, base2k: usize)
where
    M: ModuleN
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxNormalizeAssign<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxAdd<BE>
        + VecZnxCopy<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let mut source: Source = Source::new([0u8; 32]);

    let cols: usize = 2;
    let a_size: usize = 15;
    let b_size: usize = 15;
    let res_size: usize = a_size + b_size;

    let mut a = VecZnx::alloc(n, cols, a_size);
    let mut b = VecZnx::alloc(n, cols, b_size);
    let mut tmp_a = VecZnx::alloc(n, 1, a_size);
    let mut tmp_b = VecZnx::alloc(n, 1, b_size);

    let mut res_want = VecZnx::alloc(n, 1, res_size);
    // Two-column DFT destination written at column 1: covers the
    // column-interleaved `VecZnxDft` indexing of the backend kernels.
    let res_dft_col: usize = 1;
    let mut res_dft: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(n, 2, res_size);
    let mut res_big: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(n, 1, res_size);

    a.fill_uniform(17, &mut source);
    b.fill_uniform(17, &mut source);

    let a_backend = upload_vec_znx::<BE>(&a);
    let b_backend = upload_vec_znx::<BE>(&b);

    let mut a_prep: CnvPVecLOwned<BE> = module.cnv_pvec_left_alloc(n, cols, a_size, PrepareHint::Reuse);
    let mut b_prep: CnvPVecROwned<BE> = module.cnv_pvec_right_alloc(n, cols, b_size, PrepareHint::Reuse);

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
            &mut scratch.arena(),
        );
    }
    {
        let mut b_prep_backend = b_prep.to_backend_mut();
        module.cnv_prepare_right(
            &mut b_prep_backend,
            &vec_znx_backend_ref::<BE>(&b_backend),
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

                let res_host_template = VecZnx::alloc(n, 1, res_size);
                let mut res_have_backend = upload_vec_znx::<BE>(&res_host_template);
                module.vec_znx_big_normalize(
                    &mut vec_znx_backend_mut::<BE>(&mut res_have_backend),
                    base2k,
                    res_size * base2k,
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
                    module.vec_znx_add(
                        &mut vec_znx_backend_mut::<BE>(&mut tmp_a_backend),
                        0,
                        &vec_znx_backend_ref::<BE>(&a_backend),
                        col_i,
                        &vec_znx_backend_ref::<BE>(&a_backend),
                        col_j,
                    );
                    module.vec_znx_add(
                        &mut vec_znx_backend_mut::<BE>(&mut tmp_b_backend),
                        0,
                        &vec_znx_backend_ref::<BE>(&b_backend),
                        col_i,
                        &vec_znx_backend_ref::<BE>(&b_backend),
                        col_j,
                    );
                } else {
                    module.vec_znx_copy(
                        &mut vec_znx_backend_mut::<BE>(&mut tmp_a_backend),
                        0,
                        &vec_znx_backend_ref::<BE>(&a_backend),
                        col_i,
                    );
                    module.vec_znx_copy(
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
    M: VecZnxNormalizeAssign<BE>,
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
    module.vec_znx_normalize_assign(
        base2k,
        res_backend.size() * base2k,
        0,
        &mut vec_znx_backend_mut::<BE>(&mut res_backend),
        res_col,
        scratch,
    );
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
    M: VecZnxNormalizeAssign<BE>,
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
        module.vec_znx_normalize_assign(
            base2k,
            res_backend.size() * base2k,
            0,
            &mut vec_znx_backend_mut::<BE>(&mut res_backend),
            i,
            scratch,
        );
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

/// The prepare kernels take their column count from the operand they write
/// and read `a` by it, so `a.cols()` must equal it; `cnv_prepare_self` writes
/// `right` with `left`'s shape, so the pair must agree in columns and size.
/// Every backend must reject a mismatch rather than index past an operand.
pub fn test_convolution_prepare_shape_rejected<BE: crate::test_suite::TestBackend>(
    params: &TestParams,
    module: &crate::layouts::Module<BE>,
) where
    crate::layouts::Module<BE>: CnvPVecAlloc<BE> + Convolution<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let (cols, size) = (2usize, 2usize);
    let mut source = Source::new([5u8; 32]);
    let mut a = VecZnxOwned::<i64>::alloc(params.n, cols, size);
    a.fill_uniform(params.base2k, &mut source);
    let a_be = upload_vec_znx::<BE>(&a);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .cnv_prepare_left_tmp_bytes(size, size)
            .max(module.cnv_prepare_right_tmp_bytes(size, size))
            .max(module.cnv_prepare_self_tmp_bytes(size, size)),
    );

    let mut narrow_left = module.cnv_pvec_left_alloc(params.n, cols - 1, size, PrepareHint::Reuse);
    let left_panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        module.cnv_prepare_left(
            &mut narrow_left.to_backend_mut(),
            &vec_znx_backend_ref::<BE>(&a_be),
            &mut scratch.arena(),
        );
    }))
    .is_err();
    assert!(
        left_panicked,
        "cnv_prepare_left accepted res.cols() != a.cols() instead of panicking"
    );

    let mut narrow_right = module.cnv_pvec_right_alloc(params.n, cols - 1, size, PrepareHint::Reuse);
    let right_panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        module.cnv_prepare_right(
            &mut narrow_right.to_backend_mut(),
            &vec_znx_backend_ref::<BE>(&a_be),
            &mut scratch.arena(),
        );
    }))
    .is_err();
    assert!(
        right_panicked,
        "cnv_prepare_right accepted res.cols() != a.cols() instead of panicking"
    );

    let mut left = module.cnv_pvec_left_alloc(params.n, cols, size, PrepareHint::Reuse);
    let mut short_right = module.cnv_pvec_right_alloc(params.n, cols, size - 1, PrepareHint::Reuse);
    let size_panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        module.cnv_prepare_self(
            &mut left.to_backend_mut(),
            &mut short_right.to_backend_mut(),
            &vec_znx_backend_ref::<BE>(&a_be),
            &mut scratch.arena(),
        );
    }))
    .is_err();
    assert!(
        size_panicked,
        "cnv_prepare_self accepted right.size() != left.size() instead of panicking"
    );

    let cols_panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        module.cnv_prepare_self(
            &mut left.to_backend_mut(),
            &mut narrow_right.to_backend_mut(),
            &vec_znx_backend_ref::<BE>(&a_be),
            &mut scratch.arena(),
        );
    }))
    .is_err();
    assert!(
        cols_panicked,
        "cnv_prepare_self accepted right.cols() != left.cols() instead of panicking"
    );
}

/// `cnv_by_const_apply` and `cnv_by_const_apply_add` reject a first operand
/// whose degree is not the module degree, in release builds too: the kernels
/// would otherwise truncate the coefficient loop to the shorter operand.
pub fn test_convolution_by_const_degree_rejected<BE: crate::test_suite::TestBackend>(
    params: &TestParams,
    module: &crate::layouts::Module<BE>,
) where
    crate::layouts::Module<BE>: Convolution<BE> + VecZnxBigAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let size = 2usize;
    let mut source = Source::new([6u8; 32]);
    let mut a = VecZnxOwned::<i64>::alloc(params.n / 2, 1, size);
    a.fill_uniform(params.base2k, &mut source);
    let mut b = VecZnxOwned::<i64>::alloc(params.n, 1, size);
    b.fill_uniform(params.base2k, &mut source);
    let a_be = upload_vec_znx::<BE>(&a);
    let b_be = upload_vec_znx::<BE>(&b);
    let mut res: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(params.n, 1, size);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .cnv_by_const_apply_tmp_bytes(0, size, size, size)
            .max(module.cnv_by_const_apply_add_tmp_bytes(0, size, size, size)),
    );

    let apply_panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        module.cnv_by_const_apply(
            0,
            &mut res.to_backend_mut(),
            0,
            &vec_znx_backend_ref::<BE>(&a_be),
            0,
            &vec_znx_backend_ref::<BE>(&b_be),
            0,
            0,
            &mut scratch.arena(),
        );
    }))
    .is_err();
    assert!(
        apply_panicked,
        "cnv_by_const_apply accepted a.n() != res.n() instead of panicking"
    );

    let add_panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        module.cnv_by_const_apply_add(
            0,
            &mut res.to_backend_mut(),
            0,
            &vec_znx_backend_ref::<BE>(&a_be),
            0,
            &vec_znx_backend_ref::<BE>(&b_be),
            0,
            0,
            &mut scratch.arena(),
        );
    }))
    .is_err();
    assert!(
        add_panicked,
        "cnv_by_const_apply_add accepted a.n() != res.n() instead of panicking"
    );
}

/// `switch_ring_{a.n() -> n}(a)` on the backend, a copy when `a` is dense.
fn switch_ring_up<M, BE: crate::test_suite::TestBackend>(
    module: &M,
    n: usize,
    a: &VecZnx<BE::OwnedBuf, i64>,
) -> VecZnx<BE::OwnedBuf, i64>
where
    M: ModuleN + VecZnxSwitchRing<BE>,
{
    let mut res = upload_vec_znx::<BE>(&VecZnx::alloc(n, a.cols(), a.size()));
    for col in 0..a.cols() {
        module.vec_znx_switch_ring(
            &mut vec_znx_backend_mut::<BE>(&mut res),
            col,
            &vec_znx_backend_ref::<BE>(a),
            col,
        );
    }
    res
}

/// `idft` of column `res_col` of `res_dft`, normalized at `base2k` over `res_size` limbs.
fn idft_normalized<M, BE: crate::test_suite::TestBackend>(
    module: &M,
    base2k: usize,
    res_dft: &mut VecZnxDftOwned<BE>,
    res_col: usize,
    scratch: &mut ScratchOwned<BE>,
) -> VecZnx<Vec<u8>, i64>
where
    M: ModuleN + VecZnxIdftApplyTmpA<BE> + VecZnxBigAlloc<BE> + VecZnxBigNormalize<BE>,
{
    let n = res_dft.n();
    let res_size = res_dft.size();
    let mut res_big: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(n, 1, res_size);
    module.vec_znx_idft_apply_tmpa(&mut res_big.to_backend_mut(), 0, &mut res_dft.to_backend_mut(), res_col);
    let mut out = upload_vec_znx::<BE>(&VecZnx::alloc(n, 1, res_size));
    module.vec_znx_big_normalize(
        &mut vec_znx_backend_mut::<BE>(&mut out),
        base2k,
        res_size * base2k,
        0,
        0,
        &res_big.to_backend_ref(),
        base2k,
        0,
        &mut scratch.arena(),
    );
    download_vec_znx::<BE>(&out)
}

/// One convolution form under test: it writes `res` from the shared prepared
/// left operand and the given right operand, drawing its scratch from the owned
/// arena.
type ConvolutionForm<'a, BE> = dyn Fn(&mut VecZnxDftOwned<BE>, &CnvPVecROwned<BE>, &mut ScratchOwned<BE>) + 'a;

/// Sparse operands: a compact right operand, prepared at its own degree under the
/// degree-`n` module, gives the convolution that the dense prepare of
/// `switch_ring_{b_n->n}` of the same input gives, through `cnv_apply_dft`,
/// `cnv_apply_dft_add`, `cnv_apply_dft_sum` and `cnv_pairwise_apply_dft`, compared
/// after `idft` and normalization. The prepares are not sparsity-aware, `res` and `a` share one
/// degree: `cnv_prepare_left`, `cnv_prepare_right` and `cnv_prepare_self` with a
/// degree-`n/2` input into a degree-`n` destination panic, `cnv_prepare_self`
/// rejects a `right` whose degree differs from `left`'s, and `cnv_apply_dft`
/// rejects a prepared left operand whose degree differs from `res`'s.
pub fn test_convolution_sparse<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: ModuleN
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxSwitchRing<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let n = params.n;
    let base2k = params.base2k;
    let cols = 2usize;
    let (a_size, b_size, res_size) = (3usize, 4usize, 6usize);
    let res_col = 1usize;
    let mut source = Source::new([3u8; 32]);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .cnv_apply_dft_tmp_bytes(0, res_size, a_size, b_size)
            .max(module.cnv_apply_dft_add_tmp_bytes(0, res_size, a_size, b_size))
            .max(module.cnv_apply_dft_sum_tmp_bytes(0, res_size, a_size, b_size))
            .max(module.cnv_pairwise_apply_dft_tmp_bytes(0, res_size, a_size, b_size))
            .max(module.cnv_prepare_left_tmp_bytes(a_size, a_size))
            .max(module.cnv_prepare_right_tmp_bytes(b_size, b_size))
            .max(module.cnv_prepare_self_tmp_bytes(a_size, a_size))
            .max(module.vec_znx_big_normalize_tmp_bytes()),
    );

    // The left operand is dense at the module degree, shared by every form.
    let mut a_host = VecZnx::alloc(n, cols, a_size);
    a_host.fill_uniform(base2k, &mut source);
    let a_be = upload_vec_znx::<BE>(&a_host);
    let mut a_prep: CnvPVecLOwned<BE> = module.cnv_pvec_left_alloc(params.n, cols, a_size, PrepareHint::Reuse);
    module.cnv_prepare_left(
        &mut a_prep.to_backend_mut(),
        &vec_znx_backend_ref::<BE>(&a_be),
        &mut scratch.arena(),
    );

    // At params.n == 8 no sparse degree exists below it, so that leg of the sweep runs
    // only the rejection block; the compact prepare under the one module is checked on
    // the full-degree leg.
    for b_n in (1..=4).map(|g| n >> g).filter(|&d| d >= 8) {
        let mut b_host = VecZnx::alloc(b_n, cols, b_size);
        b_host.fill_uniform(base2k, &mut source);
        let b_be = upload_vec_znx::<BE>(&b_host);
        let b_dense = switch_ring_up(module, n, &b_be);

        let mut b_prep_dense: CnvPVecROwned<BE> = module.cnv_pvec_right_alloc(params.n, cols, b_size, PrepareHint::Reuse);
        module.cnv_prepare_right(
            &mut b_prep_dense.to_backend_mut(),
            &vec_znx_backend_ref::<BE>(&b_dense),
            &mut scratch.arena(),
        );
        // The compact operand at its own degree, prepared under the same module; the
        // apply below crosses the two degrees.
        let mut b_prep: CnvPVecROwned<BE> = module.cnv_pvec_right_alloc(b_n, cols, b_size, PrepareHint::Reuse);
        module.cnv_prepare_right(
            &mut b_prep.to_backend_mut(),
            &vec_znx_backend_ref::<BE>(&b_be),
            &mut scratch.arena(),
        );

        let label = format!("b.n()={b_n}");
        // Runs `form` on the dense right operand and on the compact one; both must agree after idft + normalize.
        let mut check = |what: &str, form: &ConvolutionForm<'_, BE>| {
            let mut want_dft: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(params.n, 2, res_size);
            let mut have_dft: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(params.n, 2, res_size);
            form(&mut want_dft, &b_prep_dense, &mut scratch);
            form(&mut have_dft, &b_prep, &mut scratch);
            let want = idft_normalized(module, base2k, &mut want_dft, res_col, &mut scratch);
            let have = idft_normalized(module, base2k, &mut have_dft, res_col, &mut scratch);
            assert_eq!(want, have, "{what} {label}");
        };

        for cnv_offset in [0usize, 1, 3] {
            check(&format!("cnv_apply_dft cnv_offset={cnv_offset}"), &|res, b, scratch| {
                module.cnv_apply_dft(
                    cnv_offset,
                    &mut res.to_backend_mut(),
                    res_col,
                    &a_prep.to_backend_ref(),
                    0,
                    &b.to_backend_ref(),
                    1,
                    &mut scratch.arena(),
                );
            });
            check(&format!("cnv_apply_dft_add cnv_offset={cnv_offset}"), &|res, b, scratch| {
                // The seed is the same dense product in both runs; only the added term is sparse.
                module.cnv_apply_dft(
                    cnv_offset,
                    &mut res.to_backend_mut(),
                    res_col,
                    &a_prep.to_backend_ref(),
                    1,
                    &b_prep_dense.to_backend_ref(),
                    0,
                    &mut scratch.arena(),
                );
                module.cnv_apply_dft_add(
                    cnv_offset,
                    &mut res.to_backend_mut(),
                    res_col,
                    &a_prep.to_backend_ref(),
                    0,
                    &b.to_backend_ref(),
                    1,
                    &mut scratch.arena(),
                );
            });
            check(&format!("cnv_apply_dft_sum cnv_offset={cnv_offset}"), &|res, b, scratch| {
                let terms = [
                    CnvDftAccTerm {
                        a: a_prep.to_backend_ref(),
                        a_col: 0,
                        b: b.to_backend_ref(),
                        b_col: 1,
                    },
                    CnvDftAccTerm {
                        a: a_prep.to_backend_ref(),
                        a_col: 1,
                        b: b.to_backend_ref(),
                        b_col: 0,
                    },
                ];
                module.cnv_apply_dft_sum(cnv_offset, &mut res.to_backend_mut(), res_col, &terms, &mut scratch.arena());
            });
            check(
                &format!("cnv_pairwise_apply_dft cnv_offset={cnv_offset}"),
                &|res, b, scratch| {
                    module.cnv_pairwise_apply_dft(
                        cnv_offset,
                        &mut res.to_backend_mut(),
                        res_col,
                        &a_prep.to_backend_ref(),
                        &b.to_backend_ref(),
                        0,
                        1,
                        &mut scratch.arena(),
                    );
                },
            );
        }
    }

    // No prepare is sparsity-aware: under the degree-N module a degree-N/2 input is
    // rejected by all three forms.
    let mut half = VecZnx::alloc(n / 2, cols, a_size);
    half.fill_uniform(base2k, &mut source);
    let half_be = upload_vec_znx::<BE>(&half);
    let mut full_left: CnvPVecLOwned<BE> = module.cnv_pvec_left_alloc(params.n, cols, a_size, PrepareHint::Reuse);
    let mut full_right: CnvPVecROwned<BE> = module.cnv_pvec_right_alloc(params.n, cols, a_size, PrepareHint::Reuse);
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        module.cnv_prepare_left(
            &mut full_left.to_backend_mut(),
            &vec_znx_backend_ref::<BE>(&half_be),
            &mut scratch.arena(),
        );
    }))
    .is_err();
    assert!(panicked, "cnv_prepare_left accepted a.n() != res.n()");
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        module.cnv_prepare_self(
            &mut full_left.to_backend_mut(),
            &mut full_right.to_backend_mut(),
            &vec_znx_backend_ref::<BE>(&half_be),
            &mut scratch.arena(),
        );
    }))
    .is_err();
    assert!(panicked, "cnv_prepare_self accepted a.n() != left.n()");
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        module.cnv_prepare_right(
            &mut full_right.to_backend_mut(),
            &vec_znx_backend_ref::<BE>(&half_be),
            &mut scratch.arena(),
        );
    }))
    .is_err();
    assert!(panicked, "cnv_prepare_right accepted a.n() != params.n");
    // A degree-n/2 prepared right operand is served by this module like any other
    // degree it covers, so it is only a stand-in for the mismatch check below.
    let mut half_right: CnvPVecROwned<BE> = CnvPVecR::alloc(n / 2, cols, a_size, PrepareHint::Reuse);

    // `cnv_prepare_self` prepares one input into both halves, so the two must share a degree:
    // the parallel path writes into `right` at offsets computed from `left`'s degree.
    let mut full = VecZnx::alloc(n, cols, a_size);
    full.fill_uniform(base2k, &mut source);
    let full_be = upload_vec_znx::<BE>(&full);
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        module.cnv_prepare_self(
            &mut full_left.to_backend_mut(),
            &mut half_right.to_backend_mut(),
            &vec_znx_backend_ref::<BE>(&full_be),
            &mut scratch.arena(),
        );
    }))
    .is_err();
    assert!(panicked, "cnv_prepare_self accepted right.n() != left.n()");

    // The left operand of an apply takes the module degree: the entry assert fires
    // before the kernel reads the zeroed compact operand.
    let half_left: CnvPVecLOwned<BE> = CnvPVecL::alloc(n / 2, cols, a_size, PrepareHint::Reuse);
    let mut res_dft: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(params.n, 2, res_size);
    let mut b_full: CnvPVecROwned<BE> = module.cnv_pvec_right_alloc(params.n, cols, a_size, PrepareHint::Reuse);
    module.cnv_prepare_right(
        &mut b_full.to_backend_mut(),
        &vec_znx_backend_ref::<BE>(&full_be),
        &mut scratch.arena(),
    );
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        module.cnv_apply_dft(
            0,
            &mut res_dft.to_backend_mut(),
            res_col,
            &half_left.to_backend_ref(),
            0,
            &b_full.to_backend_ref(),
            1,
            &mut scratch.arena(),
        );
    }))
    .is_err();
    assert!(panicked, "cnv_apply_dft accepted a.n() != res.n()");
}
