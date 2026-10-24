//! PaCo slot folds `Tr_{a→b}` (by addition) and `Pr_{a→b}` (by multiplication).
//!
//! The trace fold uses the fused primitive of the core trace cascade
//! (`glwe_automorphism_add_assign`, `ct += σ(ct)`) but not `glwe_trace` itself:
//! PaCo folds need an arbitrary upper bound `a` and the un-normalized sum.

use crate::{CKKSResult as Result, ckks_ensure};
use poulpy_core::{
    GLWEAutomorphism, GLWEBytesOf,
    layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey},
};
use poulpy_hal::{
    api::ModuleN,
    layouts::{CyclotomicOrder, Module, ScratchArena, galois_element},
};

use crate::{
    CKKSCompositionError, CKKSCtBounds, SetCKKSInfos,
    api::{CKKSMulOps, CKKSRotateOps},
    layouts::ScratchArenaTakeCKKS,
    layouts::slots::fold_rotations,
    oep::CKKSPaCoImpl,
};

pub(crate) fn ckks_slot_trace_assign_tmp_bytes_derived<BE, C, K>(module: &Module<BE>, ct_infos: &C, key_infos: &K) -> usize
where
    BE: CKKSPaCoImpl,
    C: CKKSCtBounds,
    K: GGLWEInfos,
{
    module.glwe_automorphism_tmp_bytes(ct_infos, ct_infos, key_infos)
}

pub(crate) fn ckks_slot_trace_assign_derived<BE, Dst, H>(
    module: &Module<BE>,
    ct: &mut Dst,
    a: usize,
    b: usize,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSPaCoImpl,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    H: GetAutomorphismKey<BE>,
{
    let order = module.cyclotomic_order();
    for rot in checked_fold_rotations(module, a, b)? {
        let key = keys
            .get_automorphism_key(galois_element(rot, order), ct.k())
            .map_err(|source| CKKSCompositionError::MissingAutomorphismKey {
                op: "paco_slot_trace",
                rotation: rot,
                k: ct.k().into(),
                source,
            })?;
        module.glwe_automorphism_add_assign(ct, &key, scratch);
    }
    Ok(())
}

pub(crate) fn ckks_slot_product_assign_tmp_bytes_derived<BE, C, K, T>(
    module: &Module<BE>,
    ct_infos: &C,
    key_infos: &K,
    tsk_infos: &T,
) -> usize
where
    BE: CKKSPaCoImpl,
    C: CKKSCtBounds,
    K: GGLWEInfos,
    T: GGLWEInfos,
{
    BE::scratch_aligned(module.glwe_bytes_of_from_infos(ct_infos))
        .checked_add(
            module
                .ckks_rotate_tmp_bytes(ct_infos, key_infos)
                .max(module.ckks_mul_tmp_bytes(ct_infos, ct_infos, ct_infos, tsk_infos)),
        )
        .expect("ckks_slot_product_assign_tmp_bytes: the byte count overflows usize")
}

pub(crate) fn ckks_slot_product_assign_derived<BE, Dst, H, TH>(
    module: &Module<BE>,
    ct: &mut Dst,
    a: usize,
    b: usize,
    keys: &H,
    tsk: &TH,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSPaCoImpl,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    H: GetAutomorphismKey<BE>,
    TH: GetTensorKey<BE>,
{
    let rotations = checked_fold_rotations(module, a, b)?;
    scratch.scope(|scratch_local| {
        let (mut tmp, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(ct);
        for k in rotations {
            module.ckks_rotate_into(&mut tmp, ct, k, keys, &mut scratch_local.borrow())?;
            module.ckks_mul_assign(ct, &tmp, tsk, &mut scratch_local.borrow())?;
        }
        Ok(())
    })
}

/// Validates the fold bounds against the module's slot count and returns the
/// rotation schedule.
fn checked_fold_rotations<M: ModuleN>(module: &M, a: usize, b: usize) -> Result<Vec<i64>> {
    ckks_ensure!(
        a.is_power_of_two() && b.is_power_of_two() && b <= a,
        "invalid PaCo fold ({a} → {b})",
    );
    ckks_ensure!(
        a <= module.n() / 2,
        "fold period a = {a} exceeds the slot count {}",
        module.n() / 2
    );
    Ok(fold_rotations(a, b))
}
