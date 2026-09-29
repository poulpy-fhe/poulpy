//! Maps between conjugate-invariant and standard ciphertexts, composed from the
//! core conjugate-invariant maps.
use crate::CKKSResult as Result;
use poulpy_core::{
    GLWECIFold, GLWECIUnfold,
    layouts::{GLWE, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos},
};
use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena, Standard};

use crate::{CKKSInfos, CKKSMeta, SetCKKSInfos, SlotsKind, layouts::CKKSCiphertext};

pub fn ckks_ci_unfold_reference<BE, D, Src>(
    module: &Module<BE>,
    dst: &mut CKKSCiphertext<D, BE::ZnxWord, Standard>,
    src: &Src,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: GLWECIUnfold<BE>,
    D: Data,
    GLWE<D, BE::ZnxWord>: GLWEToBackendMut<BE>,
    Src: GLWEToBackendRef<BE> + GLWEInfos + CKKSInfos,
{
    dst.set_meta(CKKSMeta {
        slots: SlotsKind::Real,
        ..src.meta()
    });
    dst.set_k(src.k());
    module.glwe_ci_unfold(&mut dst.inner, src);
    Ok(())
}

pub fn ckks_ci_fold_tmp_bytes_reference<BE, R, A>(module: &Module<BE>, res_infos: &R, a_infos: &A) -> usize
where
    BE: Backend,
    Module<BE>: GLWECIFold<BE>,
    R: GLWEInfos,
    A: GLWEInfos,
{
    module.glwe_ci_fold_tmp_bytes(res_infos, a_infos)
}

pub fn ckks_ci_fold_reference<BE, Dst, D>(
    module: &Module<BE>,
    dst: &mut Dst,
    src: &CKKSCiphertext<D, BE::ZnxWord, Standard>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: GLWECIFold<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEInfos + SetCKKSInfos,
    D: Data,
    GLWE<D, BE::ZnxWord>: GLWEToBackendRef<BE>,
{
    dst.set_meta(CKKSMeta {
        slots: SlotsKind::Real,
        log_delta: src.log_delta() + 1,
        ..src.meta()
    });
    dst.set_k(src.k());
    module.glwe_ci_fold(dst, &src.inner, scratch);
    Ok(())
}
