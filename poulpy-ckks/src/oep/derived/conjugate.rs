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
    let key = keys
        .get_automorphism_key(p, src.k())
        .map_err(|_| CKKSCompositionError::MissingAutomorphismKey {
            op: "conjugate",
            rotation: k,
            k: src.k().into(),
        })?;
    BE::ckks_conjugate_into_impl(module, dst, src, &key, scratch)
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
        .map_err(|_| CKKSCompositionError::MissingAutomorphismKey {
            op: "conjugate_assign",
            rotation: 0,
            k: dst.k().into(),
        })?;
    BE::ckks_conjugate_assign_impl(module, dst, &key, scratch)
}
