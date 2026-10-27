//! PaCo slot folds `Tr_{a→b}` (by addition) and `Pr_{a→b}` (by multiplication).
//!
//! The trace fold uses the fused primitive of the core trace cascade
//! (`glwe_automorphism_add_assign`, `ct += σ(ct)`) but not `glwe_trace` itself:
//! PaCo folds need an arbitrary upper bound `a` and the un-normalized sum.

use crate::{CKKSResult as Result, ckks_ensure};
use poulpy_core::{
    GLWEAutomorphism, GLWEBytesOf,
    layouts::{
        GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey, LWEInfos, TorusPrecision,
        prepared::GLWEAutomorphismKeyPreparedBackendRef,
    },
    oep::AutomorphismImpl,
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
    mul_ct_params_raw,
    oep::{CKKSMulImpl, CKKSRotateImpl},
};

pub(super) fn ckks_slot_trace_assign_tmp_bytes<BE, C, K>(module: &Module<BE>, ct_infos: &C, key_infos: &K) -> usize
where
    BE: AutomorphismImpl,
    C: CKKSCtBounds,
    K: GGLWEInfos,
{
    module.glwe_automorphism_tmp_bytes(ct_infos, ct_infos, key_infos)
}

pub(super) fn ckks_slot_trace_assign<BE, Dst, H>(
    module: &Module<BE>,
    ct: &mut Dst,
    a: usize,
    b: usize,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: AutomorphismImpl,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    H: GetAutomorphismKey<BE>,
{
    const OP: &str = "ckks_slot_trace_assign";
    let rotations = checked_fold_rotations(OP, module, a, b)?;
    let k = ct.k();
    let mut bytes = 0;
    for &rotation in &rotations {
        let key = rotation_key(OP, module, keys, rotation, k)?;
        bytes = bytes.max(ckks_slot_trace_assign_tmp_bytes(module, &*ct, &key));
    }
    ckks_ensure!(scratch.available() >= bytes, "{OP}: scratch below its query");
    for rotation in rotations {
        let key = rotation_key(OP, module, keys, rotation, k)?;
        module.glwe_automorphism_add_assign(ct, &key, scratch);
    }
    Ok(())
}

pub(super) fn ckks_slot_product_assign_tmp_bytes<BE, C, K, T>(
    module: &Module<BE>,
    ct_infos: &C,
    key_infos: &K,
    tsk_infos: &T,
) -> usize
where
    BE: CKKSRotateImpl + CKKSMulImpl,
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

pub(super) fn ckks_slot_product_assign<BE, Dst, H, TH>(
    module: &Module<BE>,
    ct: &mut Dst,
    a: usize,
    b: usize,
    keys: &H,
    tsk: &TH,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSRotateImpl + CKKSMulImpl,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    H: GetAutomorphismKey<BE>,
    TH: GetTensorKey<BE>,
{
    const OP: &str = "ckks_slot_product_assign";
    let rotations = checked_fold_rotations(OP, module, a, b)?;
    if rotations.is_empty() {
        return Ok(());
    }
    // Step `i` rotates and multiplies at the width the `i` products before it leave `ct` at.
    let log_delta = ct.log_delta();
    let mut k = ct.k().as_usize();
    let mut bytes = 0;
    for &rotation in &rotations {
        let width = TorusPrecision(u32::try_from(k).expect("a ciphertext width fits u32"));
        let key = rotation_key(OP, module, keys, rotation, width)?;
        let tensor_key = tsk
            .get_tensor_key(width)
            .map_err(|source| CKKSCompositionError::MissingRelinearizationKey {
                op: OP,
                k: width.into(),
                source,
            })?;
        bytes = bytes.max(ckks_slot_product_assign_tmp_bytes(module, &*ct, &key, &tensor_key));
        let (log_budget, product_log_delta, _) = mul_ct_params_raw(OP, k, log_delta, k, log_delta, k)?;
        k = log_budget + product_log_delta;
    }
    ckks_ensure!(scratch.available() >= bytes, "{OP}: scratch below its query");
    scratch.scope(|scratch_local| {
        let (mut tmp, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(ct);
        for k in rotations {
            module.ckks_rotate_into(k, &mut tmp, ct, keys, &mut scratch_local.borrow())?;
            module.ckks_mul_assign(ct, &tmp, tsk, &mut scratch_local.borrow())?;
        }
        Ok(())
    })
}

/// The automorphism key of a fold rotation at `k`.
fn rotation_key<'k, BE, M, H>(
    op: &'static str,
    module: &M,
    keys: &'k H,
    rotation: i64,
    k: TorusPrecision,
) -> Result<GLWEAutomorphismKeyPreparedBackendRef<'k, BE>>
where
    BE: poulpy_hal::layouts::Backend,
    M: CyclotomicOrder,
    H: GetAutomorphismKey<BE>,
{
    keys.get_automorphism_key(galois_element(rotation, module.cyclotomic_order()), k)
        .map_err(|source| {
            CKKSCompositionError::MissingAutomorphismKey {
                op,
                rotation,
                k: k.into(),
                source,
            }
            .into()
        })
}

/// Validates the fold bounds against the module's slot count and returns the
/// rotation schedule.
fn checked_fold_rotations<M: ModuleN>(op: &'static str, module: &M, a: usize, b: usize) -> Result<Vec<i64>> {
    if !(a.is_power_of_two() && b.is_power_of_two() && b <= a && a <= module.n() / 2) {
        return Err(CKKSCompositionError::InvalidOperand {
            op,
            reason: "the fold bounds are not powers of two with b <= a and a at most the slot count".into(),
        }
        .into());
    }
    Ok(fold_rotations(a, b))
}
