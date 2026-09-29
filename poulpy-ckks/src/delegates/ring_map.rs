use crate::CKKSResult as Result;
use poulpy_core::layouts::{GLWE, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos};
use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, Data, Module, ScratchArena, Standard},
};

use crate::{CKKSCtBounds, SetCKKSInfos, api::CKKSCIRingMapOps, layouts::CKKSCiphertext, oep::CKKSCIRingMapImpl};

impl<BE: Backend + CKKSCIRingMapImpl> CKKSCIRingMapOps<BE> for Module<BE> {
    fn ckks_ci_unfold<D, Src>(&self, dst: &mut CKKSCiphertext<D, BE::ZnxWord, Standard>, src: &Src) -> Result<()>
    where
        D: Data,
        GLWE<D, BE::ZnxWord>: GLWEToBackendMut<BE>,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        validate_ring_map(self, src, dst)?;
        ensure_holds(dst, src)?;
        crate::ckks_ensure!(dst.base2k() == src.base2k(), "unfold keeps the radix of its input");
        BE::ckks_ci_unfold_impl(self, dst, src)
    }

    fn ckks_ci_fold_tmp_bytes<R, A>(&self, res_infos: &R, a_infos: &A) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
    {
        BE::ckks_ci_fold_tmp_bytes_impl(self, res_infos, a_infos)
    }

    fn ckks_ci_fold<Dst, D>(
        &self,
        dst: &mut Dst,
        src: &CKKSCiphertext<D, BE::ZnxWord, Standard>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        D: Data,
        GLWE<D, BE::ZnxWord>: GLWEToBackendRef<BE>,
    {
        validate_ring_map(self, dst, src)?;
        ensure_holds(dst, src)?;
        BE::ckks_ci_fold_impl(self, dst, src, scratch)
    }
}

/// `ci` has the module degree `N` and `standard` has degree `2N`.
fn validate_ring_map<BE: Backend, C: GLWEInfos, S: GLWEInfos>(module: &Module<BE>, ci: &C, standard: &S) -> Result<()>
where
    Module<BE>: ModuleN,
{
    crate::ckks_ensure!(
        ci.n().as_usize() == module.n() && standard.n().as_usize() == 2 * module.n(),
        "ring map requires a conjugate-invariant degree N and a standard degree 2N"
    );
    crate::ckks_ensure!(ci.rank() == standard.rank(), "ring map ranks do not match");
    Ok(())
}

fn ensure_holds<D: LWEInfos, S: LWEInfos>(dst: &D, src: &S) -> Result<()> {
    crate::ckks_ensure!(
        src.k().as_usize() <= dst.max_size() * dst.base2k().as_usize(),
        "ring map output storage cannot hold the input width"
    );
    Ok(())
}
