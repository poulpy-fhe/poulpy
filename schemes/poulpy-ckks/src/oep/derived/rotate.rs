//! Rotations by a slot shift compose the key lookup with the keyed rotation; the identity is a copy.
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LWEInfos};
use poulpy_hal::layouts::{Module, ScratchArena};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSResult as Result, SetCKKSInfos,
    api::{CKKSCopyOps, CKKSModuleInfos},
    oep::CKKSRotateImpl,
};

pub(crate) fn ckks_rotate_tmp_bytes_derived<BE: CKKSRotateImpl, C: CKKSCtBounds, K: GGLWEInfos>(
    module: &Module<BE>,
    ct_infos: &C,
    key_infos: &K,
) -> usize {
    BE::ckks_rotate_with_key_tmp_bytes(module, ct_infos, key_infos).max(module.ckks_copy_tmp_bytes(ct_infos, ct_infos))
}

pub(crate) fn ckks_rotate_into_derived<BE, Dst, Src, H>(
    module: &Module<BE>,
    k: i64,
    dst: &mut Dst,
    src: &Src,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSRotateImpl,
    H: GetAutomorphismKey<BE>,
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    crate::reference::mul::ensure_admissible_rotation("ckks_rotate_by_into", crate::GLWEPlaintextInfos::slot_layout(src), k)?;
    let p = module.ckks_galois_element(k);
    if p == 1 {
        module.ckks_copy(dst, src, scratch)?;
        dst.set_noise(None);
        return Ok(());
    }
    let switch_k = crate::reference::rotate::automorphism_input_k(dst, src);
    let key = keys
        .get_automorphism_key(p, switch_k)
        .map_err(|source| CKKSCompositionError::MissingAutomorphismKey {
            op: "ckks_rotate_into",
            rotation: k,
            k: switch_k.into(),
            source,
        })?;
    BE::ckks_rotate_with_key_into(module, dst, src, &key, scratch)
}

pub(crate) fn ckks_rotate_assign_derived<BE, Dst, H>(
    module: &Module<BE>,
    k: i64,
    dst: &mut Dst,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSRotateImpl,
    H: GetAutomorphismKey<BE>,
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
{
    crate::reference::mul::ensure_admissible_rotation("ckks_rotate_by_assign", crate::GLWEPlaintextInfos::slot_layout(dst), k)?;
    let p = module.ckks_galois_element(k);
    if p == 1 {
        dst.set_noise(None);
        return Ok(());
    }
    let key = keys
        .get_automorphism_key(p, dst.k())
        .map_err(|source| CKKSCompositionError::MissingAutomorphismKey {
            op: "ckks_rotate_assign",
            rotation: k,
            k: dst.k().into(),
            source,
        })?;
    BE::ckks_rotate_with_key_assign(module, dst, &key, scratch)
}
