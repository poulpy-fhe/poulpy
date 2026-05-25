//! Per-giant DFT-domain inner products.
//!
//! This is the PROD block from docs/lt_bsgs.md §6.3: for one giant bucket,
//! compute `Σ_k ũ_{j,k} ⊙ rot(v,k)` and leave the result in `VecZnxBig`
//! (extended-precision, un-normalized). `cnv_apply_dft` overwrites its
//! destination, so the first term initializes the DFT accumulator and the
//! following terms are accumulated explicitly before a single IDFT per column.

use poulpy_hal::{
    api::{
        Convolution, ModuleN, ScratchArenaTakeBasic, VecZnxBigBytesOf, VecZnxDftAddAssign, VecZnxDftBytesOf, VecZnxIdftApplyTmpA,
    },
    layouts::{
        Backend, CnvPVecLToBackendRef, CnvPVecRToBackendRef, ScratchArena, VecZnxBigBackendMut, VecZnxDftToBackendMut,
        VecZnxDftToBackendRef, ZnxInfos,
    },
};

/// PROD block for one giant step: writes `prod_big[c] = Σ_k baby_k[c] ⊗ diagonal_k`
/// for every column `c`, leaving each column in `VecZnxBig` (un-normalized).
///
/// `cnv_offset_hi` is the limb-skip the convolution applies to its top limbs;
/// the sub-limb `cnv_offset_lo` companion is the caller's concern and is folded
/// into the single final normalize in lt_bsgs.md §6.4.
pub(super) fn glwe_accumulate_prepared_inner_product_big<'a, BE, M, BL, BR>(
    module: &M,
    cnv_offset_hi: usize,
    prod_big: &mut VecZnxBigBackendMut<'_, BE>,
    terms: &[(i64, &'a BL, &'a BR)],
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend + 'a,
    M: ModuleN + Convolution<BE> + VecZnxBigBytesOf + VecZnxDftAddAssign<BE> + VecZnxDftBytesOf + VecZnxIdftApplyTmpA<BE>,
    BL: CnvPVecLToBackendRef<BE> + ZnxInfos + 'a,
    BR: CnvPVecRToBackendRef<BE> + ZnxInfos + 'a,
{
    assert!(!terms.is_empty(), "linear transformation giant step has no terms");
    let cols = terms[0].1.cols();
    let res_dft_size = terms[0].1.size() + terms[0].2.size() - cnv_offset_hi;

    let scratch = scratch.borrow();
    // Hoist the per-column DFT scratch outside the column loop: the shape is
    // identical for every output column, so one allocation suffices.
    let (mut res_dft, scratch_1) = scratch.take_vec_znx_dft_scratch(module, 1, res_dft_size);
    let (mut term_dft, mut scratch_2) = scratch_1.take_vec_znx_dft_scratch(module, 1, res_dft_size);

    for col in 0..cols {
        for (term_idx, (_, baby, diagonal)) in terms.iter().enumerate() {
            assert_eq!(baby.cols(), cols);
            assert_eq!(baby.size() + diagonal.size() - cnv_offset_hi, res_dft_size);
            if term_idx == 0 {
                // `cnv_apply_dft` overwrites its destination, so the first
                // baby/diagonal product initializes the PROD accumulator.
                let mut res_dft_backend = res_dft.to_backend_mut();
                module.cnv_apply_dft(
                    cnv_offset_hi,
                    &mut res_dft_backend,
                    0,
                    &baby.to_backend_ref(),
                    col,
                    &diagonal.to_backend_ref(),
                    0,
                    &mut scratch_2.borrow(),
                );
            } else {
                {
                    let mut term_dft_backend = term_dft.to_backend_mut();
                    module.cnv_apply_dft(
                        cnv_offset_hi,
                        &mut term_dft_backend,
                        0,
                        &baby.to_backend_ref(),
                        col,
                        &diagonal.to_backend_ref(),
                        0,
                        &mut scratch_2.borrow(),
                    );
                }
                // Later terms are accumulated explicitly, matching the sum
                // `Σ_k ũ_{j,k} ⊙ rot(v,k)` in lt_bsgs.md §6.3.
                module.vec_znx_dft_add_assign(&mut res_dft, 0, &term_dft.to_backend_ref(), 0);
            }
        }

        // IDFT directly into the caller-provided BIG accumulator. The PROD
        // result rides through ROT and the final accumulator in `VecZnxBig`,
        // exactly as docs/lt_bsgs.md §6.3 prescribes.
        module.vec_znx_idft_apply_tmpa(prod_big, col, &mut res_dft.to_backend_mut(), 0);
    }
}

/// Scratch bytes required by [`glwe_accumulate_prepared_inner_product_big`].
pub(super) fn glwe_accumulate_prepared_inner_product_big_tmp_bytes<M>(
    module: &M,
    cnv_offset_hi: usize,
    baby_size: usize,
    diagonal_size: usize,
) -> usize
where
    M: ModuleN + VecZnxDftBytesOf,
{
    let res_dft_size = baby_size + diagonal_size - cnv_offset_hi;
    // Two DFT scratch buffers hoisted outside the column loop: the accumulator
    // and one temporary used to overwrite-then-add successive terms.
    2 * module.bytes_of_vec_znx_dft(1, res_dft_size)
}
