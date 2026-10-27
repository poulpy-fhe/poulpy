use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSResult, SetCKKSInfos, api::CKKSConjugateOps,
    layouts::slots::conj_rotate_galois_element, oep::CKKSConjugateImpl,
};
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey};
use poulpy_hal::layouts::{CyclotomicOrder, Module, ScratchArena};

pub(crate) fn ckks_conjugate_tmp_bytes_derived<BE: CKKSConjugateImpl, C: CKKSCtBounds, K: GGLWEInfos>(
    module: &Module<BE>,
    ct_infos: &C,
    key_infos: &K,
) -> usize {
    BE::ckks_conjugate_with_key_tmp_bytes(module, ct_infos, key_infos)
}

pub(crate) fn ckks_conjugate_into_derived<BE, Dst, Src, H>(
    module: &Module<BE>,
    dst: &mut Dst,
    src: &Src,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<()>
where
    BE: CKKSConjugateImpl,
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    H: GetAutomorphismKey<BE>,
{
    module.ckks_conjugate_rotate_into(0, dst, src, keys, scratch)
}

pub(crate) fn ckks_conjugate_rotate_into_derived<BE, Dst, Src, H>(
    module: &Module<BE>,
    k: i64,
    dst: &mut Dst,
    src: &Src,
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
            op: "ckks_conjugate_into",
            rotation: k,
            k: switch_k.into(),
            source,
        })?;
    BE::ckks_conjugate_with_key_into(module, dst, src, &key, scratch)
}

pub(crate) fn ckks_conjugate_assign_derived<BE, Dst, H>(
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
            op: "ckks_conjugate_assign",
            rotation: 0,
            k: dst.k().into(),
            source,
        })?;
    BE::ckks_conjugate_with_key_assign(module, dst, &key, scratch)
}
