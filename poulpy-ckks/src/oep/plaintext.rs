use crate::default::plaintext::CKKSPlaintextDefault;

use anyhow::Result;
use poulpy_core::{
    ScratchArenaTakeCore,
    layouts::{GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos},
};
use poulpy_hal::{
    api::{VecZnxLshBackend, VecZnxLshTmpBytes, VecZnxRshBackend, VecZnxRshTmpBytes},
    layouts::{Backend, Module, ScratchArena},
};

use crate::{CKKSInfos, SetCKKSInfos};

/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures.
pub unsafe trait CKKSPlaintextZnxImpl<BE: Backend>: Backend {
    fn ckks_extract_pt_tmp_bytes(module: &Module<BE>) -> usize;

    fn ckks_extract_pt<Dst, Src>(module: &Module<BE>, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos;
}

#[allow(private_bounds)]
unsafe impl<BE: Backend> CKKSPlaintextZnxImpl<BE> for BE
where
    BE: poulpy_hal::oep::HalVecZnxImpl<BE>,
    Module<BE>: CKKSPlaintextDefault<BE> + VecZnxLshTmpBytes + VecZnxRshTmpBytes + VecZnxLshBackend<BE> + VecZnxRshBackend<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_extract_pt_tmp_bytes(module: &Module<BE>) -> usize {
        module.ckks_extract_pt_tmp_bytes_default()
    }

    fn ckks_extract_pt<Dst, Src>(module: &Module<BE>, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos,
    {
        module.ckks_extract_pt_default(dst, src, scratch)
    }
}
