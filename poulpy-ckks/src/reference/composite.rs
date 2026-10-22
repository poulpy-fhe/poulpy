use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSResult as Result, SetCKKSInfos,
    api::{CKKSAddOps, CKKSCopyOps},
};
use poulpy_core::{
    GLWENormalize,
    layouts::{GLWEToBackendMut, GLWEToBackendRef, LWEInfos},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

/// Scratch of [`ckks_add_many_into_reference`] into a destination of `res_size` limbs: the additions and the
/// normalizations.
pub(crate) fn ckks_add_many_tmp_bytes_reference<BE: Backend>(module: &Module<BE>, res_size: usize) -> usize
where
    Module<BE>: CKKSAddOps<BE> + GLWENormalize<BE>,
{
    module.ckks_add_tmp_bytes(res_size).max(module.glwe_normalize_tmp_bytes())
}

pub(crate) fn ckks_add_many_into_reference<BE: Backend, Dst, Src>(
    module: &Module<BE>,
    dst: &mut Dst,
    inputs: &[&Src],
    normalize_after: usize,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    Module<BE>: CKKSAddOps<BE> + CKKSCopyOps<BE> + GLWENormalize<BE>,
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    let invalid = |reason: &str| -> Result<()> {
        Err(CKKSCompositionError::InvalidOperand {
            op: "add_many",
            reason: reason.into(),
        }
        .into())
    };
    if normalize_after == 0 {
        return invalid("normalize_after must be at least 1");
    }
    if inputs.is_empty() {
        return invalid("inputs must contain at least one ciphertext");
    }
    assert!(
        scratch.available() >= ckks_add_many_tmp_bytes_reference(module, dst.size()),
        "add_many: scratch below CKKSAddManyOps::ckks_add_many_tmp_bytes"
    );
    if let [input] = inputs {
        module.ckks_copy(dst, *input, scratch)?;
    } else {
        module.ckks_add_into(dst, inputs[0], inputs[1], scratch)?;
        for (additions, ct) in (1..).zip(&inputs[2..]) {
            if additions % normalize_after == 0 {
                module.glwe_normalize_assign(dst, scratch);
            }
            module.ckks_add_assign(dst, *ct, scratch)?;
        }
    }
    dst.set_noise(None);
    Ok(())
}
