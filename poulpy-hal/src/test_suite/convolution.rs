use super::{TestParams, download_vec_znx, upload_vec_znx, vec_znx_backend_mut, vec_znx_backend_ref};
use crate::AlignedBuf;
use crate::api::VecZnxBigFromSmall;
use crate::layouts::CnvPVecLToBackendMut;
use crate::layouts::CnvPVecLToBackendRef;
use crate::layouts::CnvPVecRToBackendMut;
use crate::layouts::CnvPVecRToBackendRef;
use crate::layouts::VecZnxBigToBackendMut;
use crate::layouts::VecZnxBigToBackendRef;
use crate::layouts::VecZnxDftToBackendMut;
use crate::layouts::VecZnxDftToBackendRef;
use rand::Rng;

use crate::test_suite::poisoned_arena;
use crate::{
    api::{
        CnvPVecAlloc, Convolution, ModuleN, VecZnxAdd, VecZnxAlloc, VecZnxAutomorphism, VecZnxBigAlloc, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxCopy, VecZnxDftAddAssign, VecZnxDftAlloc, VecZnxDftApply, VecZnxFillUniformSourceAll,
        VecZnxIdftApply, VecZnxIdftApplyTmpBytes, VecZnxNormalizeAssign, VecZnxNormalizeTmpBytes, VecZnxSwitchRing,
    },
    layouts::{
        CnvDftAccTerm, CnvPVecL, CnvPVecR, DataView, Module, PrepareHint, ScratchArena, ScratchOwned, VecZnx, VecZnxOwned,
        ZnxView, ZnxViewMut, ZnxZero,
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
        + VecZnxNormalizeTmpBytes
        + VecZnxBigAlloc<BE>
        + VecZnxAlloc<BE>
        + VecZnxFillUniformSourceAll<BE>,
{
    let mut source: Source = Source::new([0u8; 32]);

    let a_cols: usize = 2;
    let a_size: usize = 15;
    let b_size: usize = 15;
    let res_size: usize = a_size + b_size;

    let mut a_backend = module.vec_znx_alloc(n, a_cols, a_size);
    let mut b = VecZnx::alloc(n, 1, b_size);

    let mut res_want = VecZnx::alloc(n, 1, res_size);
    let mut res_big: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(n, 1, res_size);

    module.vec_znx_fill_uniform_source_all(17, a_size * 17, &mut a_backend, &mut source);
    let a = download_vec_znx::<BE>(&a_backend);

    let mask = (1 << base2k) - 1;
    for j in 0..1 {
        let r = source.next_u64() & mask;
        b.at_mut(0, j)[0] = ((r << (64 - 17)) as i64) >> (64 - 17);
    }

    let b_backend = upload_vec_znx::<BE>(&b);
    let mut scratch: ScratchOwned<BE> = poisoned_arena(module.cnv_by_const_apply_tmp_bytes(0, res_size, a_size, b_size));
    let mut scratch_setup: ScratchOwned<BE> = poisoned_arena(
        module
            .vec_znx_big_normalize_tmp_bytes()
            .max(module.vec_znx_normalize_tmp_bytes()),
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
                &mut scratch_setup.arena(),
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
                &mut scratch_setup.arena(),
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
        + VecZnxBigAlloc<BE>
        + VecZnxAlloc<BE>
        + VecZnxFillUniformSourceAll<BE>,
{
    let mut source: Source = Source::new([0u8; 32]);

    let a_cols: usize = 2;
    let a_size: usize = 15;
    let b_size: usize = 3;
    let res_size: usize = a_size + b_size;

    let mut a_backend = module.vec_znx_alloc(n, a_cols, a_size);
    let mut b_backend = module.vec_znx_alloc(n, 1, b_size);
    module.vec_znx_fill_uniform_source_all(17, a_size * 17, &mut a_backend, &mut source);
    module.vec_znx_fill_uniform_source_all(base2k, b_size * base2k, &mut b_backend, &mut source);

    let mut acc_have: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(n, 1, res_size);
    let mut acc_want: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(n, 1, res_size);
    let mut term: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(n, 1, res_size);
    let mut scratch_apply: ScratchOwned<BE> = poisoned_arena(module.cnv_by_const_apply_tmp_bytes(0, res_size, a_size, b_size));
    let mut scratch_add: ScratchOwned<BE> = poisoned_arena(module.cnv_by_const_apply_add_tmp_bytes(0, res_size, a_size, b_size));
    let mut scratch_setup: ScratchOwned<BE> = poisoned_arena(module.vec_znx_big_normalize_tmp_bytes());

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
                &mut scratch_apply.arena(),
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
                &mut scratch_add.arena(),
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
                &mut scratch_apply.arena(),
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
                &mut scratch_apply.arena(),
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
            let res_have = normalized(&acc_have, &mut scratch_setup);
            let res_want = normalized(&acc_want, &mut scratch_setup);
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
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxNormalizeAssign<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxBigAlloc<BE>
        + VecZnxAlloc<BE>
        + VecZnxFillUniformSourceAll<BE>,
{
    let mut source: Source = Source::new([0u8; 32]);

    let a_cols: usize = 2;
    let b_cols: usize = 2;
    let a_size: usize = 15;
    let b_size: usize = 15;
    let res_size: usize = a_size + b_size;

    let mut a_backend = module.vec_znx_alloc(n, a_cols, a_size);
    let mut b_backend = module.vec_znx_alloc(n, b_cols, b_size);

    let mut res_want = VecZnx::alloc(n, 1, res_size);
    // Two-column DFT destination written at column 1: covers the
    // column-interleaved `VecZnxDft` indexing of the backend kernels.
    let res_dft_col: usize = 1;
    let mut res_dft: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(n, 2, res_size);
    let mut res_big: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(n, 1, res_size);

    module.vec_znx_fill_uniform_source_all(17, a_size * 17, &mut a_backend, &mut source);
    let a = download_vec_znx::<BE>(&a_backend);
    module.vec_znx_fill_uniform_source_all(17, b_size * 17, &mut b_backend, &mut source);
    let b = download_vec_znx::<BE>(&b_backend);

    let mut a_prep: CnvPVecLOwned<BE> = module.cnv_pvec_left_alloc(n, a_cols, a_size, PrepareHint::Reuse);
    let mut b_prep: CnvPVecROwned<BE> = module.cnv_pvec_right_alloc(n, b_cols, b_size, PrepareHint::Reuse);

    let mut scratch_apply: ScratchOwned<BE> = poisoned_arena(module.cnv_apply_dft_tmp_bytes(0, res_size, a_size, b_size));
    let mut scratch_setup: ScratchOwned<BE> = poisoned_arena(
        module
            .cnv_prepare_left_tmp_bytes(res_size, a_size)
            .max(module.cnv_prepare_right_tmp_bytes(res_size, b_size))
            .max(module.vec_znx_big_normalize_tmp_bytes())
            .max(module.vec_znx_idft_apply_tmp_bytes())
            .max(module.vec_znx_normalize_tmp_bytes()),
    );

    {
        let mut a_prep_backend = a_prep.to_backend_mut();
        module.cnv_prepare_left(
            &mut a_prep_backend,
            &vec_znx_backend_ref::<BE>(&a_backend),
            &mut scratch_setup.arena(),
        );
    }
    {
        let mut b_prep_backend = b_prep.to_backend_mut();
        module.cnv_prepare_right(
            &mut b_prep_backend,
            &vec_znx_backend_ref::<BE>(&b_backend),
            &mut scratch_setup.arena(),
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
                    1,
                    &mut scratch_apply.arena(),
                );

                module.vec_znx_idft_apply(
                    &mut res_big.to_backend_mut(),
                    0,
                    &res_dft.to_backend_ref(),
                    res_dft_col,
                    &mut scratch_setup.arena(),
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
                    &mut scratch_setup.arena(),
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
                    &mut scratch_setup.arena(),
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
    M: ModuleN
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxAlloc<BE>
        + VecZnxFillUniformSourceAll<BE>,
{
    let mut source: Source = Source::new([0u8; 32]);

    let cols: usize = 2;
    let a_size: usize = 15;
    let b_size: usize = 15;
    let res_size: usize = a_size + b_size;

    let mut a_backend = module.vec_znx_alloc(n, cols, a_size);
    let mut b_backend = module.vec_znx_alloc(n, cols, b_size);
    module.vec_znx_fill_uniform_source_all(17, a_size * 17, &mut a_backend, &mut source);
    module.vec_znx_fill_uniform_source_all(17, b_size * 17, &mut b_backend, &mut source);

    let mut a_prep: CnvPVecLOwned<BE> = module.cnv_pvec_left_alloc(n, cols, a_size, PrepareHint::Reuse);
    let mut b_prep: CnvPVecROwned<BE> = module.cnv_pvec_right_alloc(n, cols, b_size, PrepareHint::Reuse);

    // Two-column accumulators exercised at both columns: covers the
    // column-interleaved `VecZnxDft` indexing of the backend kernels.
    let res_cols: usize = 2;
    let mut res_acc: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(n, res_cols, res_size);
    let mut res_ref: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(n, res_cols, res_size);
    let mut tmp_dft: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(n, 1, res_size);

    let mut scratch_apply: ScratchOwned<BE> = poisoned_arena(module.cnv_apply_dft_tmp_bytes(0, res_size, a_size, b_size));
    let mut scratch_add: ScratchOwned<BE> = poisoned_arena(module.cnv_apply_dft_add_tmp_bytes(0, res_size, a_size, b_size));
    let mut scratch_setup: ScratchOwned<BE> = poisoned_arena(
        module
            .cnv_prepare_left_tmp_bytes(res_size, a_size)
            .max(module.cnv_prepare_right_tmp_bytes(res_size, b_size)),
    );

    {
        let mut a_prep_backend = a_prep.to_backend_mut();
        module.cnv_prepare_left(
            &mut a_prep_backend,
            &vec_znx_backend_ref::<BE>(&a_backend),
            &mut scratch_setup.arena(),
        );
    }
    {
        let mut b_prep_backend = b_prep.to_backend_mut();
        module.cnv_prepare_right(
            &mut b_prep_backend,
            &vec_znx_backend_ref::<BE>(&b_backend),
            &mut scratch_setup.arena(),
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
            1,
            &mut scratch_apply.arena(),
        );
        module.cnv_apply_dft(
            0,
            &mut res_ref.to_backend_mut(),
            res_col,
            &a_prep.to_backend_ref(),
            0,
            &b_prep.to_backend_ref(),
            0,
            1,
            &mut scratch_apply.arena(),
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
                        1,
                        &mut scratch_add.arena(),
                    );

                    module.cnv_apply_dft(
                        cnv_offset,
                        &mut tmp_dft.to_backend_mut(),
                        0,
                        &a_prep.to_backend_ref(),
                        a_col,
                        &b_prep.to_backend_ref(),
                        b_col,
                        1,
                        &mut scratch_apply.arena(),
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
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxBigAlloc<BE>
        + VecZnxAlloc<BE>
        + VecZnxFillUniformSourceAll<BE>,
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

    let mut a_backend = module.vec_znx_alloc(n, cols, a_size);
    let mut b_backend = module.vec_znx_alloc(n, cols, b_size);
    module.vec_znx_fill_uniform_source_all(17, a_size * 17, &mut a_backend, &mut source);
    module.vec_znx_fill_uniform_source_all(17, b_size * 17, &mut b_backend, &mut source);

    let mut a_prep: CnvPVecLOwned<BE> = module.cnv_pvec_left_alloc(n, cols, a_size, PrepareHint::Reuse);
    let mut b_prep: CnvPVecROwned<BE> = module.cnv_pvec_right_alloc(n, cols, b_size, PrepareHint::Reuse);

    let mut res_fused: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(n, 2, res_size);
    let mut res_ref: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(n, 2, res_size);
    let mut big_fused: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(n, 1, res_size);
    let mut big_ref: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(n, 1, res_size);

    let mut scratch_sum: ScratchOwned<BE> = poisoned_arena(module.cnv_apply_dft_sum_tmp_bytes(0, res_size, a_size, b_size));
    let mut scratch_apply: ScratchOwned<BE> = poisoned_arena(module.cnv_apply_dft_tmp_bytes(0, res_size, a_size, b_size));
    let mut scratch_add: ScratchOwned<BE> = poisoned_arena(module.cnv_apply_dft_add_tmp_bytes(0, res_size, a_size, b_size));
    let mut scratch_setup: ScratchOwned<BE> = poisoned_arena(
        module
            .cnv_prepare_left_tmp_bytes(res_size, a_size)
            .max(module.cnv_prepare_right_tmp_bytes(res_size, b_size))
            .max(module.vec_znx_big_normalize_tmp_bytes())
            .max(module.vec_znx_idft_apply_tmp_bytes()),
    );

    {
        let mut a_prep_backend = a_prep.to_backend_mut();
        module.cnv_prepare_left(
            &mut a_prep_backend,
            &vec_znx_backend_ref::<BE>(&a_backend),
            &mut scratch_setup.arena(),
        );
    }
    {
        let mut b_prep_backend = b_prep.to_backend_mut();
        module.cnv_prepare_right(
            &mut b_prep_backend,
            &vec_znx_backend_ref::<BE>(&b_backend),
            &mut scratch_setup.arena(),
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
                    b_galois: 1,
                })
                .collect();
            module.cnv_apply_dft_sum(
                cnv_offset,
                &mut res_fused.to_backend_mut(),
                res_col,
                &terms,
                &mut scratch_sum.arena(),
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
                    1,
                    &mut scratch_apply.arena(),
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
                    1,
                    &mut scratch_add.arena(),
                );
            }
        }

        // Compare in the normalized coefficient domain.
        module.vec_znx_idft_apply(
            &mut big_fused.to_backend_mut(),
            0,
            &res_fused.to_backend_ref(),
            res_col,
            &mut scratch_setup.arena(),
        );
        module.vec_znx_idft_apply(
            &mut big_ref.to_backend_mut(),
            0,
            &res_ref.to_backend_ref(),
            res_col,
            &mut scratch_setup.arena(),
        );

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
            &mut scratch_setup.arena(),
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
            &mut scratch_setup.arena(),
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
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxNormalizeAssign<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxBigAlloc<BE>
        + VecZnxAdd<BE>
        + VecZnxCopy<BE>
        + VecZnxAlloc<BE>
        + VecZnxFillUniformSourceAll<BE>,
{
    let mut source: Source = Source::new([0u8; 32]);

    let cols: usize = 2;
    let a_size: usize = 15;
    let b_size: usize = 15;
    let res_size: usize = a_size + b_size;

    let mut a_backend = module.vec_znx_alloc(n, cols, a_size);
    let mut b_backend = module.vec_znx_alloc(n, cols, b_size);
    let mut tmp_a = VecZnx::alloc(n, 1, a_size);
    let mut tmp_b = VecZnx::alloc(n, 1, b_size);

    let mut res_want = VecZnx::alloc(n, 1, res_size);
    // Two-column DFT destination written at column 1: covers the
    // column-interleaved `VecZnxDft` indexing of the backend kernels.
    let res_dft_col: usize = 1;
    let mut res_dft: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(n, 2, res_size);
    let mut res_big: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(n, 1, res_size);

    module.vec_znx_fill_uniform_source_all(17, a_size * 17, &mut a_backend, &mut source);
    module.vec_znx_fill_uniform_source_all(17, b_size * 17, &mut b_backend, &mut source);

    let mut a_prep: CnvPVecLOwned<BE> = module.cnv_pvec_left_alloc(n, cols, a_size, PrepareHint::Reuse);
    let mut b_prep: CnvPVecROwned<BE> = module.cnv_pvec_right_alloc(n, cols, b_size, PrepareHint::Reuse);

    let mut scratch_pairwise: ScratchOwned<BE> =
        poisoned_arena(module.cnv_pairwise_apply_dft_tmp_bytes(0, res_size, a_size, b_size));
    let mut scratch_setup: ScratchOwned<BE> = poisoned_arena(
        module
            .cnv_prepare_left_tmp_bytes(res_size, a_size)
            .max(module.cnv_prepare_right_tmp_bytes(res_size, b_size))
            .max(module.vec_znx_big_normalize_tmp_bytes())
            .max(module.vec_znx_idft_apply_tmp_bytes())
            .max(module.vec_znx_normalize_tmp_bytes()),
    );

    {
        let mut a_prep_backend = a_prep.to_backend_mut();
        module.cnv_prepare_left(
            &mut a_prep_backend,
            &vec_znx_backend_ref::<BE>(&a_backend),
            &mut scratch_setup.arena(),
        );
    }
    {
        let mut b_prep_backend = b_prep.to_backend_mut();
        module.cnv_prepare_right(
            &mut b_prep_backend,
            &vec_znx_backend_ref::<BE>(&b_backend),
            &mut scratch_setup.arena(),
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
                    &mut scratch_pairwise.arena(),
                );

                module.vec_znx_idft_apply(
                    &mut res_big.to_backend_mut(),
                    0,
                    &res_dft.to_backend_ref(),
                    res_dft_col,
                    &mut scratch_setup.arena(),
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
                    &mut scratch_setup.arena(),
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
                    &mut scratch_setup.arena(),
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
/// Every backend must reject a mismatch, writing nothing, rather than index
/// past an operand.
pub fn test_cnv_prepare_rejects_mismatched_shape<BE: crate::test_suite::TestBackend>(
    params: &TestParams,
    module: &crate::layouts::Module<BE>,
) where
    crate::layouts::Module<BE>: CnvPVecAlloc<BE> + Convolution<BE>,
{
    let (cols, size) = (2usize, 2usize);
    let mut source = Source::new([5u8; 32]);
    let mut operand = |cols: usize, size: usize| {
        let mut a = module.vec_znx_alloc(params.n, cols, size);
        module.vec_znx_fill_uniform_source_all(params.base2k, size * params.base2k, &mut a, &mut source);
        a
    };
    let a_be = operand(cols, size);
    let a_narrow = operand(cols - 1, size);
    let a_short = operand(cols, size - 1);
    let mut scratch_left: ScratchOwned<BE> = poisoned_arena(module.cnv_prepare_left_tmp_bytes(size, size));
    let mut scratch_right: ScratchOwned<BE> = poisoned_arena(module.cnv_prepare_right_tmp_bytes(size, size));
    let mut scratch_self: ScratchOwned<BE> = poisoned_arena(module.cnv_prepare_self_tmp_bytes(size, size));
    let mut scratch_setup: ScratchOwned<BE> = poisoned_arena(
        module
            .cnv_prepare_left_tmp_bytes(size, size)
            .max(module.cnv_prepare_right_tmp_bytes(size, size)),
    );

    macro_rules! rejects {
        ($op:literal, $condition:literal, [$($dst:expr),+], $call:expr) => {{
            let before = [$(BE::to_host_bytes(DataView::data(&$dst))),+];
            let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $call)).is_err();
            assert!(panicked, concat!($op, " accepted ", $condition, " instead of panicking"));
            assert!(
                before == [$(BE::to_host_bytes(DataView::data(&$dst))),+],
                concat!($op, " wrote its destination before rejecting ", $condition)
            );
        }};
    }

    let mut narrow_left = module.cnv_pvec_left_alloc(params.n, cols - 1, size, PrepareHint::Reuse);
    module.cnv_prepare_left(
        &mut narrow_left.to_backend_mut(),
        &vec_znx_backend_ref::<BE>(&a_narrow),
        &mut scratch_setup.arena(),
    );
    rejects!(
        "cnv_prepare_left",
        "res.cols() != a.cols()",
        [narrow_left],
        module.cnv_prepare_left(
            &mut narrow_left.to_backend_mut(),
            &vec_znx_backend_ref::<BE>(&a_be),
            &mut scratch_left.arena(),
        )
    );

    let mut narrow_right = module.cnv_pvec_right_alloc(params.n, cols - 1, size, PrepareHint::Reuse);
    module.cnv_prepare_right(
        &mut narrow_right.to_backend_mut(),
        &vec_znx_backend_ref::<BE>(&a_narrow),
        &mut scratch_setup.arena(),
    );
    rejects!(
        "cnv_prepare_right",
        "res.cols() != a.cols()",
        [narrow_right],
        module.cnv_prepare_right(
            &mut narrow_right.to_backend_mut(),
            &vec_znx_backend_ref::<BE>(&a_be),
            &mut scratch_right.arena(),
        )
    );

    let mut left = module.cnv_pvec_left_alloc(params.n, cols, size, PrepareHint::Reuse);
    module.cnv_prepare_left(
        &mut left.to_backend_mut(),
        &vec_znx_backend_ref::<BE>(&a_be),
        &mut scratch_setup.arena(),
    );
    let mut short_right = module.cnv_pvec_right_alloc(params.n, cols, size - 1, PrepareHint::Reuse);
    module.cnv_prepare_right(
        &mut short_right.to_backend_mut(),
        &vec_znx_backend_ref::<BE>(&a_short),
        &mut scratch_setup.arena(),
    );
    rejects!(
        "cnv_prepare_self",
        "right.size() != left.size()",
        [left, short_right],
        module.cnv_prepare_self(
            &mut left.to_backend_mut(),
            &mut short_right.to_backend_mut(),
            &vec_znx_backend_ref::<BE>(&a_be),
            &mut scratch_self.arena(),
        )
    );
    rejects!(
        "cnv_prepare_self",
        "right.cols() != left.cols()",
        [left, narrow_right],
        module.cnv_prepare_self(
            &mut left.to_backend_mut(),
            &mut narrow_right.to_backend_mut(),
            &vec_znx_backend_ref::<BE>(&a_be),
            &mut scratch_self.arena(),
        )
    );
}

/// `cnv_by_const_apply` and `cnv_by_const_apply_add` reject a first operand
/// whose degree is not the module degree, in release builds too: the kernels
/// would otherwise truncate the coefficient loop to the shorter operand. The
/// destination is left unchanged.
pub fn test_cnv_by_const_rejects_degree_mismatch<BE: crate::test_suite::TestBackend>(
    params: &TestParams,
    module: &crate::layouts::Module<BE>,
) where
    crate::layouts::Module<BE>: Convolution<BE> + VecZnxBigAlloc<BE> + VecZnxBigFromSmall<BE>,
{
    let size = 2usize;
    let mut source = Source::new([6u8; 32]);
    let mut a_be = module.vec_znx_alloc(params.n / 2, 1, size);
    module.vec_znx_fill_uniform_source_all(params.base2k, size * params.base2k, &mut a_be, &mut source);
    let mut b_be = module.vec_znx_alloc(params.n, 1, size);
    module.vec_znx_fill_uniform_source_all(params.base2k, size * params.base2k, &mut b_be, &mut source);
    let mut res: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(params.n, 1, size);
    module.vec_znx_big_from_small(&mut res.to_backend_mut(), 0, &vec_znx_backend_ref::<BE>(&b_be), 0);
    let mut scratch_apply: ScratchOwned<BE> = poisoned_arena(module.cnv_by_const_apply_tmp_bytes(0, size, size, size));
    let mut scratch_add: ScratchOwned<BE> = poisoned_arena(module.cnv_by_const_apply_add_tmp_bytes(0, size, size, size));

    macro_rules! rejects {
        ($op:literal, $call:expr) => {{
            let before = BE::to_host_bytes(DataView::data(&res));
            let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $call)).is_err();
            assert!(panicked, concat!($op, " accepted a.n() != res.n() instead of panicking"));
            assert!(
                before == BE::to_host_bytes(DataView::data(&res)),
                concat!($op, " wrote its destination before rejecting a.n() != res.n()")
            );
        }};
    }

    rejects!(
        "cnv_by_const_apply",
        module.cnv_by_const_apply(
            0,
            &mut res.to_backend_mut(),
            0,
            &vec_znx_backend_ref::<BE>(&a_be),
            0,
            &vec_znx_backend_ref::<BE>(&b_be),
            0,
            0,
            &mut scratch_apply.arena(),
        )
    );
    rejects!(
        "cnv_by_const_apply_add",
        module.cnv_by_const_apply_add(
            0,
            &mut res.to_backend_mut(),
            0,
            &vec_znx_backend_ref::<BE>(&a_be),
            0,
            &vec_znx_backend_ref::<BE>(&b_be),
            0,
            0,
            &mut scratch_add.arena(),
        )
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
    res_dft: &VecZnxDftOwned<BE>,
    res_col: usize,
) -> VecZnx<AlignedBuf, i64>
where
    M: ModuleN
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigAlloc<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
{
    let n = res_dft.n();
    let res_size = res_dft.size();
    let mut res_big: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(n, 1, res_size);
    module.vec_znx_idft_apply(
        &mut res_big.to_backend_mut(),
        0,
        &res_dft.to_backend_ref(),
        res_col,
        &mut poisoned_arena::<BE>(module.vec_znx_idft_apply_tmp_bytes()).arena(),
    );
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
        &mut poisoned_arena::<BE>(module.vec_znx_big_normalize_tmp_bytes()).arena(),
    );
    download_vec_znx::<BE>(&out)
}

/// One convolution form under test: it writes `res` from the shared prepared
/// left operand and the given right operand, on poisoned arenas of its own.
type ConvolutionForm<'a, BE> = dyn Fn(&mut VecZnxDftOwned<BE>, &CnvPVecROwned<BE>) + 'a;

/// Embedded operands: a lower-degree right operand, prepared at its own degree under the
/// degree-`n` module, gives the convolution that the dense prepare of
/// `switch_ring_{b_n->n}` of the same input gives, through `cnv_apply_dft`,
/// `cnv_apply_dft_add`, `cnv_apply_dft_sum` and `cnv_pairwise_apply_dft`, compared
/// after `idft` and normalization. The prepares are not embedding-aware, `res` and `a` share one
/// degree: `cnv_prepare_left`, `cnv_prepare_right` and `cnv_prepare_self` with a
/// degree-`n/2` input into a degree-`n` destination panic, `cnv_prepare_self`
/// rejects a `right` whose degree differs from `left`'s, and `cnv_apply_dft`
/// rejects a prepared left operand whose degree differs from `res`'s, and
/// `cnv_apply_dft` rejects a right operand below `BE::MIN_DEGREE` without writing `res`.
pub fn test_convolution_embed<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: ModuleN
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigAlloc<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxSwitchRing<BE>
        + VecZnxFillUniformSourceAll<BE>,
{
    let n = params.n;
    let base2k = params.base2k;
    let cols = 2usize;
    let (a_size, b_size, res_size) = (3usize, 4usize, 6usize);
    let res_col = 1usize;
    let mut source = Source::new([3u8; 32]);
    let apply_arena = || poisoned_arena::<BE>(module.cnv_apply_dft_tmp_bytes(0, res_size, a_size, b_size));
    let add_arena = || poisoned_arena::<BE>(module.cnv_apply_dft_add_tmp_bytes(0, res_size, a_size, b_size));
    let sum_arena = || poisoned_arena::<BE>(module.cnv_apply_dft_sum_tmp_bytes(0, res_size, a_size, b_size));
    let pairwise_arena = || poisoned_arena::<BE>(module.cnv_pairwise_apply_dft_tmp_bytes(0, res_size, a_size, b_size));
    let prepare_left_arena = || poisoned_arena::<BE>(module.cnv_prepare_left_tmp_bytes(a_size, a_size));
    let prepare_right_arena = || poisoned_arena::<BE>(module.cnv_prepare_right_tmp_bytes(b_size, b_size));
    let prepare_self_arena = || poisoned_arena::<BE>(module.cnv_prepare_self_tmp_bytes(a_size, a_size));

    // The left operand is dense at the module degree, shared by every form.
    let mut a_be = module.vec_znx_alloc(n, cols, a_size);
    module.vec_znx_fill_uniform_source_all(base2k, a_size * base2k, &mut a_be, &mut source);
    let mut a_prep: CnvPVecLOwned<BE> = module.cnv_pvec_left_alloc(params.n, cols, a_size, PrepareHint::Reuse);
    module.cnv_prepare_left(
        &mut a_prep.to_backend_mut(),
        &vec_znx_backend_ref::<BE>(&a_be),
        &mut prepare_left_arena().arena(),
    );

    // At params.n == BE::MIN_DEGREE no embedded degree exists below it, so that leg of the
    // sweep runs only the rejection block; the lower-degree prepare under the one module is
    // checked on the full-degree leg.
    for b_n in (1..=4).map(|g| n >> g).filter(|&d| d >= BE::MIN_DEGREE) {
        let mut b_be = module.vec_znx_alloc(b_n, cols, b_size);
        module.vec_znx_fill_uniform_source_all(base2k, b_size * base2k, &mut b_be, &mut source);
        let b_dense = switch_ring_up(module, n, &b_be);

        let mut b_prep_dense: CnvPVecROwned<BE> = module.cnv_pvec_right_alloc(params.n, cols, b_size, PrepareHint::Reuse);
        module.cnv_prepare_right(
            &mut b_prep_dense.to_backend_mut(),
            &vec_znx_backend_ref::<BE>(&b_dense),
            &mut prepare_right_arena().arena(),
        );
        // The operand at its own degree, prepared under the same module; the
        // apply below crosses the two degrees.
        let mut b_prep: CnvPVecROwned<BE> = module.cnv_pvec_right_alloc(b_n, cols, b_size, PrepareHint::Reuse);
        module.cnv_prepare_right(
            &mut b_prep.to_backend_mut(),
            &vec_znx_backend_ref::<BE>(&b_be),
            &mut prepare_right_arena().arena(),
        );

        let label = format!("b.n()={b_n}");
        // Runs `form` on the dense right operand and on the embedded one; both must agree after idft + normalize.
        let check = |what: &str, form: &ConvolutionForm<'_, BE>| {
            let mut want_dft: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(params.n, 2, res_size);
            let mut have_dft: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(params.n, 2, res_size);
            form(&mut want_dft, &b_prep_dense);
            form(&mut have_dft, &b_prep);
            let want = idft_normalized(module, base2k, &want_dft, res_col);
            let have = idft_normalized(module, base2k, &have_dft, res_col);
            assert_eq!(want, have, "{what} {label}");
        };

        for cnv_offset in [0usize, 1, 3] {
            check(&format!("cnv_apply_dft cnv_offset={cnv_offset}"), &|res, b| {
                module.cnv_apply_dft(
                    cnv_offset,
                    &mut res.to_backend_mut(),
                    res_col,
                    &a_prep.to_backend_ref(),
                    0,
                    &b.to_backend_ref(),
                    1,
                    1,
                    &mut apply_arena().arena(),
                );
            });
            check(&format!("cnv_apply_dft_add cnv_offset={cnv_offset}"), &|res, b| {
                // The seed is the same dense product in both runs; only the added term is embedded.
                module.cnv_apply_dft(
                    cnv_offset,
                    &mut res.to_backend_mut(),
                    res_col,
                    &a_prep.to_backend_ref(),
                    1,
                    &b_prep_dense.to_backend_ref(),
                    0,
                    1,
                    &mut apply_arena().arena(),
                );
                module.cnv_apply_dft_add(
                    cnv_offset,
                    &mut res.to_backend_mut(),
                    res_col,
                    &a_prep.to_backend_ref(),
                    0,
                    &b.to_backend_ref(),
                    1,
                    1,
                    &mut add_arena().arena(),
                );
            });
            check(&format!("cnv_apply_dft_sum cnv_offset={cnv_offset}"), &|res, b| {
                let terms = [
                    CnvDftAccTerm {
                        a: a_prep.to_backend_ref(),
                        a_col: 0,
                        b: b.to_backend_ref(),
                        b_col: 1,
                        b_galois: 1,
                    },
                    CnvDftAccTerm {
                        a: a_prep.to_backend_ref(),
                        a_col: 1,
                        b: b.to_backend_ref(),
                        b_col: 0,
                        b_galois: 1,
                    },
                ];
                module.cnv_apply_dft_sum(
                    cnv_offset,
                    &mut res.to_backend_mut(),
                    res_col,
                    &terms,
                    &mut sum_arena().arena(),
                );
            });
            check(&format!("cnv_pairwise_apply_dft cnv_offset={cnv_offset}"), &|res, b| {
                module.cnv_pairwise_apply_dft(
                    cnv_offset,
                    &mut res.to_backend_mut(),
                    res_col,
                    &a_prep.to_backend_ref(),
                    &b.to_backend_ref(),
                    0,
                    1,
                    &mut pairwise_arena().arena(),
                );
            });
        }
    }

    // A right operand below the backend's minimum degree is rejected, with the dirty destination untouched.
    let below = BE::MIN_DEGREE / 2;
    if below >= 1 {
        let mut dirty: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(params.n, 2, res_size);
        let mut seed_be = module.vec_znx_alloc(n, cols, b_size);
        module.vec_znx_fill_uniform_source_all(base2k, b_size * base2k, &mut seed_be, &mut source);
        let mut seed_right: CnvPVecROwned<BE> = module.cnv_pvec_right_alloc(params.n, cols, b_size, PrepareHint::Reuse);
        module.cnv_prepare_right(
            &mut seed_right.to_backend_mut(),
            &vec_znx_backend_ref::<BE>(&seed_be),
            &mut prepare_right_arena().arena(),
        );
        module.cnv_apply_dft(
            0,
            &mut dirty.to_backend_mut(),
            res_col,
            &a_prep.to_backend_ref(),
            0,
            &seed_right.to_backend_ref(),
            1,
            1,
            &mut apply_arena().arena(),
        );
        let narrow_right: CnvPVecROwned<BE> = CnvPVecR::alloc(below, cols, b_size, PrepareHint::Reuse);
        let before = BE::to_host_bytes(DataView::data(&dirty));
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            module.cnv_apply_dft(
                0,
                &mut dirty.to_backend_mut(),
                res_col,
                &a_prep.to_backend_ref(),
                0,
                &narrow_right.to_backend_ref(),
                1,
                1,
                &mut apply_arena().arena(),
            );
        }))
        .is_err();
        assert!(panicked, "cnv_apply_dft accepted b.n() < BE::MIN_DEGREE instead of panicking");
        assert!(
            before == BE::to_host_bytes(DataView::data(&dirty)),
            "cnv_apply_dft wrote its destination before rejecting b.n() < BE::MIN_DEGREE"
        );
    }

    // No prepare is sparsity-aware: under the degree-N module a degree-N/2 input is
    // rejected by all three forms.
    let mut half_be = module.vec_znx_alloc(n / 2, cols, a_size);
    module.vec_znx_fill_uniform_source_all(base2k, a_size * base2k, &mut half_be, &mut source);
    let mut full_left: CnvPVecLOwned<BE> = module.cnv_pvec_left_alloc(params.n, cols, a_size, PrepareHint::Reuse);
    let mut full_right: CnvPVecROwned<BE> = module.cnv_pvec_right_alloc(params.n, cols, a_size, PrepareHint::Reuse);
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        module.cnv_prepare_left(
            &mut full_left.to_backend_mut(),
            &vec_znx_backend_ref::<BE>(&half_be),
            &mut prepare_left_arena().arena(),
        );
    }))
    .is_err();
    assert!(panicked, "cnv_prepare_left accepted a.n() != res.n()");
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        module.cnv_prepare_self(
            &mut full_left.to_backend_mut(),
            &mut full_right.to_backend_mut(),
            &vec_znx_backend_ref::<BE>(&half_be),
            &mut prepare_self_arena().arena(),
        );
    }))
    .is_err();
    assert!(panicked, "cnv_prepare_self accepted a.n() != left.n()");
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        module.cnv_prepare_right(
            &mut full_right.to_backend_mut(),
            &vec_znx_backend_ref::<BE>(&half_be),
            &mut prepare_right_arena().arena(),
        );
    }))
    .is_err();
    assert!(panicked, "cnv_prepare_right accepted a.n() != params.n");
    // A degree-n/2 prepared right operand is served by this module like any other
    // degree it covers, so it is only a stand-in for the mismatch check below.
    let mut half_right: CnvPVecROwned<BE> = CnvPVecR::alloc(n / 2, cols, a_size, PrepareHint::Reuse);

    // `cnv_prepare_self` prepares one input into both halves, so the two must share a degree:
    // the parallel path writes into `right` at offsets computed from `left`'s degree.
    let mut full_be = module.vec_znx_alloc(n, cols, a_size);
    module.vec_znx_fill_uniform_source_all(base2k, a_size * base2k, &mut full_be, &mut source);
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        module.cnv_prepare_self(
            &mut full_left.to_backend_mut(),
            &mut half_right.to_backend_mut(),
            &vec_znx_backend_ref::<BE>(&full_be),
            &mut prepare_self_arena().arena(),
        );
    }))
    .is_err();
    assert!(panicked, "cnv_prepare_self accepted right.n() != left.n()");

    // The left operand of an apply takes the module degree: the entry assert fires
    // before the kernel reads the zeroed half-degree operand.
    let half_left: CnvPVecLOwned<BE> = CnvPVecL::alloc(n / 2, cols, a_size, PrepareHint::Reuse);
    let mut res_dft: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(params.n, 2, res_size);
    let mut b_full: CnvPVecROwned<BE> = module.cnv_pvec_right_alloc(params.n, cols, a_size, PrepareHint::Reuse);
    module.cnv_prepare_right(
        &mut b_full.to_backend_mut(),
        &vec_znx_backend_ref::<BE>(&full_be),
        &mut prepare_right_arena().arena(),
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
            1,
            &mut apply_arena().arena(),
        );
    }))
    .is_err();
    assert!(panicked, "cnv_apply_dft accepted a.n() != res.n()");
}

/// Galois reads: the apply of a prepared right operand `b` through `σ_p` equals, bit for bit after `idft` and
/// normalization, the identity apply of the operand prepared from `σ_p` of the coefficients of `b`, for
/// `cnv_apply_dft`, `cnv_apply_dft_add` and `cnv_apply_dft_sum` (the sum mixing two Galois elements), on dense and
/// embedded right operands; an even `b_galois` is rejected.
pub fn test_convolution_galois<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: ModuleN
        + Convolution<BE>
        + CnvPVecAlloc<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigAlloc<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxAutomorphism<BE>
        + VecZnxFillUniformSourceAll<BE>,
{
    let n = params.n;
    let base2k = params.base2k;
    let cols = 2usize;
    let (a_size, b_size, res_size) = (3usize, 4usize, 6usize);
    let res_col = 1usize;
    let mut source = Source::new([5u8; 32]);
    let apply_arena = || poisoned_arena::<BE>(module.cnv_apply_dft_tmp_bytes(0, res_size, a_size, b_size));
    let add_arena = || poisoned_arena::<BE>(module.cnv_apply_dft_add_tmp_bytes(0, res_size, a_size, b_size));
    let sum_arena = || poisoned_arena::<BE>(module.cnv_apply_dft_sum_tmp_bytes(0, res_size, a_size, b_size));
    let prepare_left_arena = || poisoned_arena::<BE>(module.cnv_prepare_left_tmp_bytes(a_size, a_size));
    let prepare_right_arena = || poisoned_arena::<BE>(module.cnv_prepare_right_tmp_bytes(b_size, b_size));
    let mut a_be = module.vec_znx_alloc(n, cols, a_size);
    module.vec_znx_fill_uniform_source_all(base2k, a_size * base2k, &mut a_be, &mut source);
    let mut a_prep: CnvPVecLOwned<BE> = module.cnv_pvec_left_alloc(n, cols, a_size, PrepareHint::Reuse);
    module.cnv_prepare_left(
        &mut a_prep.to_backend_mut(),
        &vec_znx_backend_ref::<BE>(&a_be),
        &mut prepare_left_arena().arena(),
    );

    let a_ref = &a_prep;
    let prepare_right = |b: &VecZnx<BE::OwnedBuf, i64>| {
        let mut prep: CnvPVecROwned<BE> = module.cnv_pvec_right_alloc(b.n(), cols, b_size, PrepareHint::Reuse);
        module.cnv_prepare_right(
            &mut prep.to_backend_mut(),
            &vec_znx_backend_ref::<BE>(b),
            &mut prepare_right_arena().arena(),
        );
        prep
    };
    let automorphism = |p: i64, b: &VecZnx<BE::OwnedBuf, i64>| {
        let mut res = module.vec_znx_alloc(b.n(), cols, b_size);
        for col in 0..cols {
            module.vec_znx_automorphism(
                p,
                &mut vec_znx_backend_mut::<BE>(&mut res),
                col,
                &vec_znx_backend_ref::<BE>(b),
                col,
            );
        }
        res
    };

    for b_n in (0..=2).map(|g| n >> g).filter(|&d| d >= BE::MIN_DEGREE) {
        let mut b_be = module.vec_znx_alloc(b_n, cols, b_size);
        module.vec_znx_fill_uniform_source_all(base2k, b_size * base2k, &mut b_be, &mut source);
        let b_prep = prepare_right(&b_be);
        let b_auto5 = prepare_right(&automorphism(5, &b_be));
        let auto5_ref = &b_auto5;
        let two_n = 2 * b_n as i64;
        for p in [1i64, 5, 25, 3, -1, two_n - 1, two_n + 5, 4 * two_n - 7] {
            let b_auto = prepare_right(&automorphism(p, &b_be));
            let label = format!("b.n()={b_n} p={p}");
            let check = |what: &str, want_form: &ConvolutionForm<'_, BE>, have_form: &ConvolutionForm<'_, BE>| {
                let mut want_dft: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(n, 2, res_size);
                let mut have_dft: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(n, 2, res_size);
                want_form(&mut want_dft, &b_auto);
                have_form(&mut have_dft, &b_prep);
                let want = idft_normalized(module, base2k, &want_dft, res_col);
                let have = idft_normalized(module, base2k, &have_dft, res_col);
                assert_eq!(want, have, "{what} {label}");
            };
            for cnv_offset in [0usize, 1, 3] {
                let apply = |galois: i64| {
                    move |res: &mut VecZnxDftOwned<BE>, b: &CnvPVecROwned<BE>| {
                        module.cnv_apply_dft(
                            cnv_offset,
                            &mut res.to_backend_mut(),
                            res_col,
                            &a_ref.to_backend_ref(),
                            0,
                            &b.to_backend_ref(),
                            1,
                            galois,
                            &mut apply_arena().arena(),
                        );
                    }
                };
                check(&format!("cnv_apply_dft cnv_offset={cnv_offset}"), &apply(1), &apply(p));
                let add = |galois: i64| {
                    move |res: &mut VecZnxDftOwned<BE>, b: &CnvPVecROwned<BE>| {
                        module.cnv_apply_dft(
                            cnv_offset,
                            &mut res.to_backend_mut(),
                            res_col,
                            &a_ref.to_backend_ref(),
                            1,
                            &b.to_backend_ref(),
                            0,
                            galois,
                            &mut apply_arena().arena(),
                        );
                        module.cnv_apply_dft_add(
                            cnv_offset,
                            &mut res.to_backend_mut(),
                            res_col,
                            &a_ref.to_backend_ref(),
                            0,
                            &b.to_backend_ref(),
                            1,
                            galois,
                            &mut add_arena().arena(),
                        );
                    }
                };
                check(&format!("cnv_apply_dft_add cnv_offset={cnv_offset}"), &add(1), &add(p));
                // Term 0 reads `b` through `p` (want: the pre-automorphed operand); term 1 reads `b` through `5`
                // (want: `b_auto5`).
                let sum = |first_galois: i64, second_is_b: bool, second_galois: i64| {
                    move |res: &mut VecZnxDftOwned<BE>, b: &CnvPVecROwned<BE>| {
                        let terms = [
                            CnvDftAccTerm {
                                a: a_ref.to_backend_ref(),
                                a_col: 0,
                                b: b.to_backend_ref(),
                                b_col: 1,
                                b_galois: first_galois,
                            },
                            CnvDftAccTerm {
                                a: a_ref.to_backend_ref(),
                                a_col: 1,
                                b: (if second_is_b { b } else { auto5_ref }).to_backend_ref(),
                                b_col: 0,
                                b_galois: second_galois,
                            },
                        ];
                        module.cnv_apply_dft_sum(
                            cnv_offset,
                            &mut res.to_backend_mut(),
                            res_col,
                            &terms,
                            &mut sum_arena().arena(),
                        );
                    }
                };
                check(
                    &format!("cnv_apply_dft_sum cnv_offset={cnv_offset}"),
                    &sum(1, false, 1),
                    &sum(p, true, 5),
                );
            }
        }
    }

    let mut b_be = module.vec_znx_alloc(n, cols, b_size);
    module.vec_znx_fill_uniform_source_all(base2k, b_size * base2k, &mut b_be, &mut source);
    let b_prep = prepare_right(&b_be);
    let mut res_dft: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(n, 2, res_size);
    // A product the rejected terms below would overwrite: a rejection must leave it in place.
    module.cnv_apply_dft(
        0,
        &mut res_dft.to_backend_mut(),
        res_col,
        &a_ref.to_backend_ref(),
        0,
        &b_prep.to_backend_ref(),
        1,
        1,
        &mut apply_arena().arena(),
    );
    let mut expect_rejected = |what: &str, run: &mut dyn FnMut(&mut VecZnxDftOwned<BE>)| {
        let before = idft_normalized(module, base2k, &res_dft, res_col);
        let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&mut res_dft)));
        let payload = rejected.expect_err(what);
        let message = payload
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_default();
        assert_eq!(message, "convolution: b_galois must be odd", "{what}");
        let after = idft_normalized(module, base2k, &res_dft, res_col);
        assert_eq!(before, after, "{what}: res written before the rejection");
    };
    expect_rejected("cnv_apply_dft rejects an even b_galois", &mut |res| {
        module.cnv_apply_dft(
            0,
            &mut res.to_backend_mut(),
            res_col,
            &a_ref.to_backend_ref(),
            0,
            &b_prep.to_backend_ref(),
            1,
            4,
            &mut apply_arena().arena(),
        );
    });
    expect_rejected("cnv_apply_dft_add rejects an even b_galois", &mut |res| {
        module.cnv_apply_dft_add(
            0,
            &mut res.to_backend_mut(),
            res_col,
            &a_ref.to_backend_ref(),
            0,
            &b_prep.to_backend_ref(),
            1,
            4,
            &mut add_arena().arena(),
        );
    });
    // 17 terms put the even one past a fused group of 16.
    for count in [2usize, 17] {
        expect_rejected(
            &format!("cnv_apply_dft_sum rejects an even b_galois in term {}", count - 1),
            &mut |res| {
                let terms: Vec<_> = (0..count)
                    .map(|t| CnvDftAccTerm {
                        a: a_ref.to_backend_ref(),
                        a_col: 1,
                        b: b_prep.to_backend_ref(),
                        b_col: 0,
                        b_galois: if t + 1 == count { 4 } else { 1 },
                    })
                    .collect();
                module.cnv_apply_dft_sum(0, &mut res.to_backend_mut(), res_col, &terms, &mut sum_arena().arena());
            },
        );
    }
}
