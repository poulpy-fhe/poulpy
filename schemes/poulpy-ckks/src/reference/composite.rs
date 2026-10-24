use crate::{
    CKKSCtBounds, CKKSResult as Result, SetCKKSInfos,
    api::{CKKSAddOps, CKKSCopyOps},
    ckks_bail,
};
use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

pub(crate) fn ckks_add_many_reference<BE: Backend, Dst, Src>(
    module: &Module<BE>,
    dst: &mut Dst,
    inputs: &[&Src],
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    Module<BE>: CKKSAddOps<BE> + CKKSCopyOps<BE>,
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    match inputs.len() {
        0 => ckks_bail!("ckks_add_many: inputs must contain at least one ciphertext"),
        1 => {
            module.ckks_copy(dst, inputs[0], scratch)?;
        }
        _ => {
            module.ckks_add_into(dst, inputs[0], inputs[1], scratch)?;
            for ct in &inputs[2..] {
                module.ckks_add_assign(dst, *ct, scratch)?;
            }
        }
    }
    dst.set_noise(None);
    Ok(())
}
