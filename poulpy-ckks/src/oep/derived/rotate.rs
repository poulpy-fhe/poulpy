//! Rotations by a slot shift compose the key lookup with the keyed rotation; the identity is a copy.
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LWEInfos};
use poulpy_hal::layouts::{Module, ScratchArena};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSResult as Result, SetCKKSInfos,
    api::{CKKSCopyOps, CKKSModuleInfos},
    oep::CKKSRotateImpl,
};

pub(crate) fn ckks_rotate_by_tmp_bytes<BE: CKKSRotateImpl, C: CKKSCtBounds, K: GGLWEInfos>(
    module: &Module<BE>,
    ct_infos: &C,
    key_infos: &K,
) -> usize {
    BE::ckks_rotate_tmp_bytes_impl(module, ct_infos, key_infos).max(module.ckks_copy_tmp_bytes(ct_infos, ct_infos))
}

pub(crate) fn ckks_rotate_by_into<BE, Dst, Src, H>(
    module: &Module<BE>,
    dst: &mut Dst,
    src: &Src,
    k: i64,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSRotateImpl,
    H: GetAutomorphismKey<BE>,
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    let p = module.ckks_galois_element(k);
    if p == 1 {
        return module.ckks_copy(dst, src, scratch);
    }
    let key = keys
        .get_automorphism_key(p, src.k())
        .map_err(|_| CKKSCompositionError::MissingAutomorphismKey {
            op: "rotate",
            rotation: k,
            k: src.k().into(),
        })?;
    BE::ckks_rotate_into_impl(module, dst, src, &key, scratch)
}

pub(crate) fn ckks_rotate_by_assign<BE, Dst, H>(
    module: &Module<BE>,
    dst: &mut Dst,
    k: i64,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSRotateImpl,
    H: GetAutomorphismKey<BE>,
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
{
    let p = module.ckks_galois_element(k);
    if p == 1 {
        return Ok(());
    }
    let key = keys
        .get_automorphism_key(p, dst.k())
        .map_err(|_| CKKSCompositionError::MissingAutomorphismKey {
            op: "rotate_assign",
            rotation: k,
            k: dst.k().into(),
        })?;
    BE::ckks_rotate_assign_impl(module, dst, &key, scratch)
}
