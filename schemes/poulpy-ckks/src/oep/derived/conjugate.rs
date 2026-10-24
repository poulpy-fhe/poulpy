use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSResult, SetCKKSInfos, oep::CKKSConjugateImpl,
    reference::paco::ops::conj_rotate_galois_element,
};
use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey};
use poulpy_hal::layouts::{CyclotomicOrder, Module, ScratchArena};

pub(crate) fn conjugate_rotate<BE, Dst, Src, H>(
    module: &Module<BE>,
    dst: &mut Dst,
    src: &Src,
    k: i64,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<()>
where
    BE: CKKSConjugateImpl,
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    H: GetAutomorphismKey<BE>,
{
    let p = conj_rotate_galois_element(k, module.cyclotomic_order());
    // The switch reads `src` at `dst.k()` when `dst` is narrower.
    let switch_k = src.k().min(dst.k());
    let key = keys
        .get_automorphism_key(p, switch_k)
        .map_err(|source| CKKSCompositionError::MissingAutomorphismKey {
            op: "conjugate",
            rotation: k,
            k: switch_k.into(),
            source,
        })?;
    BE::ckks_conjugate_with_key_into(module, dst, src, &key, scratch)
}

pub(crate) fn conjugate_assign<BE, Dst, H>(
    module: &Module<BE>,
    dst: &mut Dst,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<()>
where
    BE: CKKSConjugateImpl,
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    H: GetAutomorphismKey<BE>,
{
    let key = keys
        .get_automorphism_key(-1, dst.k())
        .map_err(|source| CKKSCompositionError::MissingAutomorphismKey {
            op: "conjugate_assign",
            rotation: 0,
            k: dst.k().into(),
            source,
        })?;
    BE::ckks_conjugate_with_key_assign(module, dst, &key, scratch)
}
