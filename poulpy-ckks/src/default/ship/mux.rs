//! Hoisted base-B mux blind rotation (SHIP §5.1, Algorithm 5).

use crate::{CKKSResult as Result, ckks_ensure};
use poulpy_core::layouts::{
    GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
    prepared::{GGLWEPreparedVmpPMatRef, GLWESwitchingKeyPrepared},
};
use poulpy_hal::{
    api::{
        ScratchArenaTakeBasic, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAddAssign,
        VecZnxDftApply, VecZnxDftAutomorphism, VecZnxDftAutomorphismPlan, VecZnxDftBytesOf, VecZnxDftCopy, VecZnxDftZero,
        VecZnxIdftApplyTmpA, VmpApplyDftToDft, VmpApplyDftToDftTmpBytes,
    },
    layouts::{
        Backend, Module, ScratchArena, VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef,
        VecZnxDftToBackendMut, VecZnxDftToBackendRef,
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
                    .or_insert_with(|| module.vec_znx_dft_automorphism_plan(key.gal_el));
            }
        }
    }
    plans
}

/// `res = a x key` in the DFT domain (the VMP of the keyswitch, without the
/// input DFT or the output IDFT/normalize); replicates the gadget digit loop
/// of the default GGLWE product for `dsize > 1`.
fn ship_switching_key_product<BE>(
    module: &Module<BE>,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    a: &VecZnxDftBackendRef<'_, BE>,
    key: &GLWESwitchingKeyPrepared<BE::OwnedBuf, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>: VmpApplyDftToDft<BE> + VecZnxDftAddAssign<BE> + VecZnxDftCopy<BE> + VecZnxDftZero<BE> + VecZnxDftBytesOf,
{
    let pmat = key.vmp_pmat_backend_ref();
    let dsize = key.dsize().as_usize();
    if dsize == 1 {
        module.vmp_apply_dft_to_dft(res, a, &pmat, 0, scratch);
        return;
    }
    let dnum: usize = key.dnum().into();
    let cols = a.cols();
    let cols_out = res.cols();
    let a_size = a.size();
    for col in 0..cols_out {
        module.vec_znx_dft_zero(res, col);
    }
    for di in 0..dsize {
        let (mut ai_dft, mut scratch_1) =
            scratch
                .borrow()
                .take_vec_znx_dft_scratch(module, cols, ((a_size + di) / dsize).min(dnum));
        let res_compute_size = res.size() - ((dsize - di) as isize - 2).max(0) as usize;
        let mut res_view = res.with_size_mut(res_compute_size);
        for j in 0..cols {
            module.vec_znx_dft_copy(dsize, dsize - di - 1, &mut ai_dft.to_backend_mut(), j, a, j);
        }
        if di == 0 {
            module.vmp_apply_dft_to_dft(&mut res_view, &ai_dft.to_backend_ref(), &pmat, 0, &mut scratch_1.borrow());
        } else {
            let (mut res_tmp, mut scratch_2) = scratch_1.take_vec_znx_dft_scratch(module, cols_out, res_view.size());
            module.vmp_apply_dft_to_dft(
                &mut res_tmp.to_backend_mut(),
                &ai_dft.to_backend_ref(),
                &pmat,
                di,
                &mut scratch_2.borrow(),
            );
            for col in 0..cols_out {
                module.vec_znx_dft_add_assign(&mut res_view, col, &res_tmp.to_backend_ref(), col);
            }
        }
    }
}

/// Scratch bytes for [`ship_mux_rotate`].
pub(crate) fn ship_mux_rotate_tmp_bytes<BE, C, K>(module: &Module<BE>, ct: &C, key: &K) -> usize
where
    BE: Backend,
    C: LWEInfos,
    K: GGLWEInfos,
    Module<BE>: VecZnxDftBytesOf + VecZnxBigBytesOf + VmpApplyDftToDftTmpBytes + VecZnxBigNormalizeTmpBytes,
{
    let a_size = ct.size();
    let key_size = key.size();
    let dsize = key.dsize().as_usize();
    let dnum: usize = key.dnum().into();
    let vmp = module.vmp_apply_dft_to_dft_tmp_bytes(key_size, a_size, dnum, 2, 2, key_size);
    let product = if dsize == 1 {
        vmp
    } else {
        module.bytes_of_vec_znx_dft(2, a_size.div_ceil(dsize).min(dnum)) + module.bytes_of_vec_znx_dft(2, key_size) + vmp
    };
    let mux = 2 * module.bytes_of_vec_znx_dft(2, key_size) + product;
    let finalize = module.bytes_of_vec_znx_big(2, key_size) + module.vec_znx_big_normalize_tmp_bytes();
    module.bytes_of_vec_znx_dft(2, a_size) + module.bytes_of_vec_znx_dft(2, key_size) + mux.max(finalize)
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
        + VecZnxDftCopy<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftAutomorphism<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VmpApplyDftToDft<BE>
        + VecZnxDftBytesOf,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
{
    const OP: &str = "ship_mux_rotate";
    ckks_ensure!(!keys.is_empty(), "{OP}: empty key group");
    let a_size = ct.size();
    let key_size = keys[0].key.size();
    let base2k = ct.base2k().as_usize();
    ckks_ensure!(
        keys[0].key.base2k().as_usize() == base2k,
        "{OP}: ciphertext/key base2k mismatch"
    );

    let scratch = scratch.borrow();
    let (mut a_dft, scratch_1) = scratch.take_vec_znx_dft_scratch(module, 2, a_size);
    {
        let a_ref = GLWEToBackendRef::<BE>::to_backend_ref(ct);
        let mut a_dft_mut = a_dft.to_backend_mut();
        module.vec_znx_dft_apply(1, 0, &mut a_dft_mut, 0, a_ref.data(), 1);
        module.vec_znx_dft_apply(1, 0, &mut a_dft_mut, 1, a_ref.data(), 0);
    }
    let a_dft_ref = a_dft.to_backend_ref();

    let (mut sum_dft, mut scratch_2) = scratch_1.take_vec_znx_dft_scratch(module, 2, key_size);
    {
        let mut sum_dft_mut = sum_dft.to_backend_mut();
        for col in 0..2 {
            module.vec_znx_dft_zero(&mut sum_dft_mut, col);
        }
        let (mut prod_dft, scratch_3) = scratch_2.borrow().take_vec_znx_dft_scratch(module, 2, key_size);
        let (mut rot_dft, mut scratch_4) = scratch_3.take_vec_znx_dft_scratch(module, 2, key_size);
        for key in keys {
            ckks_ensure!(key.key.size() == key_size, "{OP}: inconsistent key sizes in group");
            {
                let mut prod_dft_mut = prod_dft.to_backend_mut();
                ship_switching_key_product(module, &mut prod_dft_mut, &a_dft_ref, &key.key, &mut scratch_4.borrow());
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

    let (mut res_big, mut scratch_3) = scratch_2.take_vec_znx_big_scratch(module, 2, key_size);
    {
        let mut res_big_mut = res_big.to_backend_mut();
        let mut sum_dft_mut = sum_dft.to_backend_mut();
        for col in 0..2 {
            module.vec_znx_idft_apply_tmpa(&mut res_big_mut, col, &mut sum_dft_mut, col);
        }
    }
    let res_big_ref = res_big.to_backend_ref();
    let mut ct_mut = ct.to_backend_mut();
    for col in 0..2 {
        module.vec_znx_big_normalize(
            ct_mut.data_mut(),
            base2k,
            0,
            col,
            &res_big_ref,
            base2k,
            col,
            &mut scratch_3.borrow(),
        );
    }
    Ok(())
}
