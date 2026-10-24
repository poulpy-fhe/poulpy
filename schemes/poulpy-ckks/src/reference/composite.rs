use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSResult as Result, SetCKKSInfos,
    api::{CKKSAddOps, CKKSCopyOps},
    ckks_ensure,
};
use poulpy_core::{
    GLWENormalize,
    layouts::{GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

/// Scratch of [`ckks_add_many_reference`] from terms of layout `a` into `res`: the additions, the normalizations
/// and, for one term, the copy.
pub(crate) fn ckks_add_many_tmp_bytes_reference<BE: Backend, R, A>(module: &Module<BE>, res: &R, a: &A) -> usize
where
    Module<BE>: CKKSAddOps<BE> + CKKSCopyOps<BE> + GLWENormalize<BE>,
    R: CKKSCtBounds,
    A: CKKSCtBounds,
{
    module
        .ckks_add_tmp_bytes(res.size())
        .max(module.glwe_normalize_tmp_bytes())
        .max(module.ckks_copy_tmp_bytes(res, a))
}

pub(crate) fn ckks_add_many_reference<BE: Backend, R, A>(
    module: &Module<BE>,
    res: &mut R,
    a: &[&A],
    normalize_after: usize,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    Module<BE>: CKKSAddOps<BE> + CKKSCopyOps<BE> + GLWENormalize<BE>,
    R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE> + CKKSCtBounds,
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
    let [first, rest @ ..] = a else {
        return invalid("a must contain at least one ciphertext");
    };
    ckks_ensure!(
        scratch.available() >= ckks_add_many_tmp_bytes_reference(module, &*res, *first),
        "add_many: scratch below CKKSAddManyOps::ckks_add_many_tmp_bytes"
    );
    match rest {
        [] => module.ckks_copy(res, *first, scratch)?,
        [second, rest @ ..] => {
            module.ckks_add_into(res, *first, *second, scratch)?;
            for (additions, term) in (1..).zip(rest) {
                if additions % normalize_after == 0 {
                    module.glwe_normalize_assign(res, scratch);
                }
                module.ckks_add_assign(res, *term, scratch)?;
            }
        }
    }
    res.set_noise(None);
    Ok(())
}
