//! Theta-column masking accumulation (SHIP §3.3 masks fused with the
//! low-digit column selection of the §4.4 hybrid).

use crate::{CKKSResult as Result, ckks_ensure};
use poulpy_core::{
    layouts::{GLWEToBackendMut, GLWEToBackendRef, LWEInfos},
    msb_mask_bottom_limb,
};
use poulpy_hal::{
    api::{
        CnvPVecBytesOf, Convolution, ScratchArenaTakeBasic, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxDftBytesOf, VecZnxIdftApplyTmpA,
    },
    layouts::{
        Backend, CnvDftAccTerm, CnvPVecL, CnvPVecLToBackendRef, CnvPVecRToBackendRef, Module, ScratchArena,
        VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxDftToBackendMut,
    },
};

use crate::{
    CKKSInfos, SetCKKSInfos,
    default::mul::mul_pt_params_raw,
    layouts::{CKKSCiphertext, CKKSPlaintext, ShipPlan},
};

/// Scratch bytes for [`ship_masking_accumulate`].
pub(crate) fn ship_masking_tmp_bytes<BE>(module: &Module<BE>, plan: &ShipPlan, base2k: usize) -> usize
where
    BE: Backend,
    Module<BE>: Convolution<BE> + CnvPVecBytesOf + VecZnxDftBytesOf + VecZnxBigBytesOf + VecZnxBigNormalizeTmpBytes,
{
    let a_size = plan.raised_k(base2k).div_ceil(base2k);
    let b_size = (plan.log_delta_work() + base2k).div_ceil(base2k);
    let res_dft_size = a_size + b_size;
    let preps = 4 * plan.theta() * module.bytes_of_cnv_pvec_right(1, b_size);
    let work = module
        .cnv_prepare_right_tmp_bytes(b_size, b_size)
        .max(module.cnv_accumulate_dft_tmp_bytes(0, res_dft_size, a_size, b_size))
        .max(module.bytes_of_vec_znx_big(2, res_dft_size) + module.vec_znx_big_normalize_tmp_bytes());
    module.bytes_of_vec_znx_dft(2, res_dft_size) + preps + work
}

/// Lazy masking accumulation: `acc = sum_i masks[i] * pis[i]` over the
/// keygen-prepared mask operands, the per-pair convolutions accumulated in
/// the DFT domain with a single IDFT + normalize per column.
pub(crate) fn ship_masking_accumulate<BE>(
    module: &Module<BE>,
    acc: &mut CKKSCiphertext<BE::OwnedBuf>,
    plan: &ShipPlan,
    masks: &[CnvPVecL<BE::OwnedBuf, BE::DftWord, BE>],
    pis: &[CKKSPlaintext<BE::OwnedBuf>],
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: Convolution<BE> + CnvPVecBytesOf + VecZnxDftBytesOf + VecZnxIdftApplyTmpA<BE> + VecZnxBigNormalize<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    CKKSPlaintext<BE::OwnedBuf>: GLWEToBackendRef<BE>,
{
    const OP: &str = "ship_masking_accumulate";
    ckks_ensure!(
        !masks.is_empty() && masks.len() == pis.len(),
        "{OP}: empty or mismatched operands"
    );
    let base2k = acc.base2k().as_usize();
    let kk = plan.raised_k(base2k);
    let ld = plan.log_delta_work();
    let (res_log_budget, res_log_delta, cnv_offset) = mul_pt_params_raw(
        acc.max_k().as_usize(),
        ld,
        kk - ld,
        pis[0].log_delta(),
        pis[0].log_budget(),
        pis[0].max_k().as_usize(),
    )?;
    let a_size = kk.div_ceil(base2k);
    let b_size = pis[0].size();
    let b_mask = msb_mask_bottom_limb(base2k, pis[0].max_k().as_usize());
    let (cnv_offset_hi, cnv_offset_lo) = if cnv_offset < base2k {
        (0, -((base2k - (cnv_offset % base2k)) as i64))
    } else {
        ((cnv_offset / base2k).saturating_sub(1), (cnv_offset % base2k) as i64)
    };
    let res_dft_size = a_size + b_size - cnv_offset_hi;

    let scratch = scratch.borrow();
    let (mut sum_dft, scratch_1) = scratch.take_vec_znx_dft_scratch(module, 2, res_dft_size);

    let mut preps = Vec::with_capacity(pis.len());
    let mut rest = scratch_1;
    for (mask, pi) in masks.iter().zip(pis) {
        ckks_ensure!(
            mask.size() == a_size && pi.size() == b_size,
            "{OP}: inconsistent operand sizes"
        );
        let (mut b_prep, next) = rest.take_cnv_pvec_right_scratch(module, 1, b_size);
        rest = next
            .apply_mut(|s| module.cnv_prepare_right(&mut b_prep, GLWEToBackendRef::<BE>::to_backend_ref(pi).data(), b_mask, s));
        preps.push(b_prep);
    }

    {
        let mut sum_dft_mut = sum_dft.to_backend_mut();
        for col in 0..2 {
            let terms: Vec<CnvDftAccTerm<'_, BE>> = masks
                .iter()
                .zip(&preps)
                .map(|(mask, prep)| CnvDftAccTerm {
                    a: mask.to_backend_ref(),
                    a_col: col,
                    b: prep.to_backend_ref(),
                    b_col: 0,
                })
                .collect();
            module.cnv_accumulate_dft(cnv_offset_hi, &mut sum_dft_mut, col, &terms, &mut rest.borrow());
        }
    }

    let (mut res_big, mut scratch_2) = rest.take_vec_znx_big_scratch(module, 2, res_dft_size);
    {
        let mut res_big_mut = res_big.to_backend_mut();
        let mut sum_dft_mut = sum_dft.to_backend_mut();
        for col in 0..2 {
            module.vec_znx_idft_apply_tmpa(&mut res_big_mut, col, &mut sum_dft_mut, col);
        }
    }
    let res_big_ref = res_big.to_backend_ref();
    {
        let mut acc_mut = acc.to_backend_mut();
        for col in 0..2 {
            module.vec_znx_big_normalize(
                acc_mut.data_mut(),
                base2k,
                cnv_offset_lo,
                col,
                &res_big_ref,
                base2k,
                col,
                &mut scratch_2.borrow(),
            );
        }
    }
    acc.set_log_budget(res_log_budget);
    acc.set_log_delta(res_log_delta);
    Ok(())
}
