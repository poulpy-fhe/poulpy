use crate::CKKSAtkBounds;
use crate::CKKSResult as Result;
use poulpy_core::{
    GLWEAutomorphism, GLWEShift,
    layouts::{
        GGLWEInfos, GLWEAutomorphismKeyHelper, GLWEAutomorphismKeyLayoutHelper, GLWEToBackendMut, GLWEToBackendRef,
        WithEffectiveDsize,
    },
};
use poulpy_hal::layouts::{Backend, GaloisElement, Module, ScratchArena};

use crate::{CKKSCompositionError, CKKSCtBounds, SetCKKSInfos, oep::CKKSRotateImpl};

use crate::api::CKKSRotateOps;

impl<BE: Backend + CKKSRotateImpl<BE>> CKKSRotateOps<BE> for Module<BE>
where
    Module<BE>: GLWEAutomorphism<BE> + GLWEShift<BE> + GaloisElement,
{
    fn ckks_rotate_tmp_bytes<C, K>(&self, ct_infos: &C, key_infos: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        BE::ckks_rotate_tmp_bytes_impl(self, ct_infos, key_infos)
    }

    fn ckks_rotate_into<Dst, Src, H, K>(
        &self,
        dst: &mut Dst,
        src: &Src,
        k: i64,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        K: CKKSAtkBounds<BE>,
        H: GLWEAutomorphismKeyHelper<K> + GLWEAutomorphismKeyLayoutHelper<K>,
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        let (key, effective_dsize) = keys.get_automorphism_key_for(self.galois_element(k), src.k()).map_err(|_| {
            CKKSCompositionError::MissingAutomorphismKey {
                op: "rotate",
                rotation: k,
            }
        })?;
        BE::ckks_rotate_into_impl(self, dst, src, &key.with_dsize(effective_dsize), scratch)
    }

    fn ckks_rotate_assign<Dst, H, K>(&self, dst: &mut Dst, k: i64, keys: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        K: CKKSAtkBounds<BE>,
        H: GLWEAutomorphismKeyHelper<K> + GLWEAutomorphismKeyLayoutHelper<K>,
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        let (key, effective_dsize) = keys.get_automorphism_key_for(self.galois_element(k), dst.k()).map_err(|_| {
            CKKSCompositionError::MissingAutomorphismKey {
                op: "rotate_assign",
                rotation: k,
            }
        })?;
        BE::ckks_rotate_assign_impl(self, dst, &key.with_dsize(effective_dsize), scratch)
    }
}
