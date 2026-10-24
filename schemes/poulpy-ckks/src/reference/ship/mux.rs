//! Hoisted base-B mux blind rotation (SHIP §5.1, Algorithm 5).

use crate::{CKKSResult as Result, ckks_ensure};
use poulpy_core::{
    layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, digits_aligned, prepared::GGLWEPreparedToBackendRef},
    reference::keyswitching::glwe::{
        GGLWEProductReference, gadget_dft_apply, gadget_dft_apply_tmp_bytes, gadget_input_size,
        gglwe_product_accumulation_output_size,
    },
};
use poulpy_hal::{
    api::{
        ScratchArenaTakeBasic, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAddAssign,
        VecZnxDftApply, VecZnxDftAutomorphism, VecZnxDftAutomorphismPlan, VecZnxDftBytesOf, VecZnxDftZero, VecZnxGadgetDecompose,
        VecZnxGadgetDecomposeTmpBytes, VecZnxIdftApply, VecZnxIdftApplyTmpBytes,
    },
    layouts::{
        Backend, Module, ScratchArena, VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxDftToBackendMut, VecZnxDftToBackendRef,
    },
};

use std::collections::HashMap;

use crate::layouts::{CKKSCiphertextOwned, ship::keyset::HMuxRotKeyPrepared};

/// Automorphism plans of the mux output twists, keyed by Galois element.
pub(crate) type ShipMuxPlans<BE> = HashMap<i64, <Module<BE> as VecZnxDftAutomorphismPlan<BE>>::Plan>;

/// Builds the automorphism plans of every distinct mux Galois element once;
/// the same rotation amounts recur across all support slots.
pub(crate) fn ship_mux_plans<'a, BE>(
    module: &Module<BE>,
    groups: impl Iterator<Item = &'a [HMuxRotKeyPrepared<BE::OwnedBuf, BE>]>,
) -> ShipMuxPlans<BE>
where
    BE: Backend + 'a,
    Module<BE>: VecZnxDftAutomorphismPlan<BE>,
{
    let mut plans = ShipMuxPlans::<BE>::new();
    for group in groups {
        for key in group {
            if key.gal_el != 1 {
                plans
                    .entry(key.gal_el)
                    .or_insert_with(|| module.vec_znx_dft_automorphism_plan(module.n(), key.gal_el));
            }
        }
    }
    plans
}

/// Scratch bytes for [`ship_mux_rotate`].
pub(crate) fn ship_mux_rotate_tmp_bytes<BE, C, K>(module: &Module<BE>, ct: &C, key: &K, term_count: usize) -> usize
where
    BE: Backend,
    C: LWEInfos,
    K: GGLWEInfos,
    Module<BE>: VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApplyTmpBytes
        + VecZnxGadgetDecomposeTmpBytes
        + GGLWEProductReference<BE>,
{
    let a_size = ct.size();
    let output_size = gglwe_product_accumulation_output_size::<BE, _, _, _>(ct, ct, key, term_count);
    let (base2k, k_digit, dnum) = (key.base2k(), key.k_digit(), key.dnum());
    let operand_size = gadget_input_size(base2k, k_digit, dnum, ct.k(), a_size);
    let layout = gadget_dft_apply_tmp_bytes::<BE, _>(module, base2k, k_digit, dnum, ct.k(), a_size);
    let product = module.gglwe_product_dft_tmp_bytes_reference(output_size, operand_size, key);
    let mux = 2 * module.bytes_of_vec_znx_dft(module.n(), 2, output_size) + product;
    let finalize = module.bytes_of_vec_znx_big(module.n(), 2, output_size)
        + module
            .vec_znx_big_normalize_tmp_bytes()
            .max(module.vec_znx_idft_apply_tmp_bytes());
    // The operand is laid out before the accumulator is taken.
    module.bytes_of_vec_znx_dft(module.n(), 2, operand_size)
        + layout.max(module.bytes_of_vec_znx_dft(module.n(), 2, output_size) + mux.max(finalize))
}

/// Hoisted B-to-1 mux-rotate: `ct <- sum_d beta_d * Rot_{rot_d}(ct)` over the
/// keys of one digit position. The rank-2 input `(0, a_mask, a_body)` is
/// DFT'd once and shared by every key; each key contributes one VMP followed
/// by its DFT-domain output automorphism, and a single IDFT + normalize
/// closes the position.
pub(crate) fn ship_mux_rotate<BE>(
    module: &Module<BE>,
    ct: &mut CKKSCiphertextOwned<BE>,
    keys: &[HMuxRotKeyPrepared<BE::OwnedBuf, BE>],
    plans: &ShipMuxPlans<BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: VecZnxDftApply<BE>
        + VecZnxDftZero<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftAutomorphism<BE>
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigNormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxGadgetDecompose<BE>
        + GGLWEProductReference<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
{
    const OP: &str = "ship_mux_rotate";
    ckks_ensure!(!keys.is_empty(), "{OP}: empty key group");
    let a_size = ct.size();
    let key_size = keys[0].key.size();
    let output_size = gglwe_product_accumulation_output_size::<BE, _, _, _>(ct, ct, &keys[0].key, keys.len());
    let base2k = ct.base2k().as_usize();
    ckks_ensure!(
        keys[0].key.base2k().as_usize() == base2k,
        "{OP}: ciphertext/key base2k mismatch"
    );
    ckks_ensure!(
        keys.iter().all(|key| ct.k() <= key.key.gglwe_layout().max_input_k()),
        "{OP}: the input is wider than a key's max_input_k"
    );
    // One operand serves every key of the group: whole-limb digits read the
    // input limbs whatever their width, other widths must share one digit grid.
    let (k_digit, dnum) = (keys[0].key.k_digit(), keys[0].key.dnum());
    let key_base2k = keys[0].key.base2k();
    ckks_ensure!(
        keys.iter().all(|key| digits_aligned(key.key.base2k(), key.key.k_digit()))
            || keys.iter().all(|key| (key.key.k_digit(), key.key.dnum()) == (k_digit, dnum)),
        "{OP}: key digits off a limb boundary differ in width or count within the group"
    );
    let operand_size = gadget_input_size(key_base2k, k_digit, dnum, ct.k(), a_size);

    let scratch = scratch.borrow();
    let (mut a_dft, mut scratch_1) = scratch.take_vec_znx_dft_scratch(module.n(), 2, operand_size);
    {
        let a_ref = GLWEToBackendRef::<BE>::to_backend_ref(ct);
        let mut a_dft_mut = a_dft.to_backend_mut();
        // The rank-2 input is (0, a_mask, a_body): the mask column then the body.
        for (res_col, a_col) in [(0, 1), (1, 0)] {
            gadget_dft_apply(
                module,
                &mut a_dft_mut,
                res_col,
                a_ref.data(),
                a_col,
                key_base2k,
                k_digit,
                dnum,
                ct.k(),
                &mut scratch_1.borrow(),
            );
        }
    }
    let a_dft_ref = a_dft.to_backend_ref();

    let (mut sum_dft, mut scratch_2) = scratch_1.take_vec_znx_dft_scratch(module.n(), 2, output_size);
    {
        let mut sum_dft_mut = sum_dft.to_backend_mut();
        for col in 0..2 {
            module.vec_znx_dft_zero(&mut sum_dft_mut, col);
        }
        let (mut prod_dft, scratch_3) = scratch_2.borrow().take_vec_znx_dft_scratch(module.n(), 2, output_size);
        let (mut rot_dft, mut scratch_4) = scratch_3.take_vec_znx_dft_scratch(module.n(), 2, output_size);
        for key in keys {
            ckks_ensure!(key.key.size() == key_size, "{OP}: inconsistent key sizes in group");
            ckks_ensure!(key.key.k_offset().0 == 0, "{OP}: keys with a row offset are not supported");
            {
                let mut prod_dft_mut = prod_dft.to_backend_mut();
                module.gglwe_product_dft_reference(
                    &mut prod_dft_mut,
                    &a_dft_ref,
                    &key.key.to_backend_ref(),
                    keys.len(),
                    &mut scratch_4.borrow(),
                );
            }
            if key.gal_el == 1 {
                let prod_ref = prod_dft.to_backend_ref();
                for col in 0..2 {
                    module.vec_znx_dft_add_assign(&mut sum_dft_mut, col, &prod_ref, col);
                }
            } else {
                let plan = plans
                    .get(&key.gal_el)
                    .ok_or_else(|| anyhow::anyhow!("{OP}: missing automorphism plan for Galois element {}", key.gal_el))?;
                {
                    let prod_ref = prod_dft.to_backend_ref();
                    let mut rot_dft_mut = rot_dft.to_backend_mut();
                    for col in 0..2 {
                        module.vec_znx_dft_automorphism_with_plan(plan, &mut rot_dft_mut, col, &prod_ref, col);
                    }
                }
                let rot_ref = rot_dft.to_backend_ref();
                for col in 0..2 {
                    module.vec_znx_dft_add_assign(&mut sum_dft_mut, col, &rot_ref, col);
                }
            }
        }
    }

    let (mut res_big, mut scratch_3) = scratch_2.take_vec_znx_big_scratch(module.n(), 2, output_size);
    {
        let mut res_big_mut = res_big.to_backend_mut();
        let sum_dft_ref = sum_dft.to_backend_ref();
        for col in 0..2 {
            module.vec_znx_idft_apply(&mut res_big_mut, col, &sum_dft_ref, col, &mut scratch_3.borrow());
        }
    }
    let k = ct.k().as_usize();
    ct.set_canonical(true);
    {
        let res_big_ref = res_big.to_backend_ref();
        let mut ct_mut = ct.to_backend_mut();
        for col in 0..2 {
            module.vec_znx_big_normalize(
                ct_mut.data_mut(),
                base2k,
                k,
                0,
                col,
                &res_big_ref,
                base2k,
                col,
                &mut scratch_3.borrow(),
            );
        }
    }
    Ok(())
}
