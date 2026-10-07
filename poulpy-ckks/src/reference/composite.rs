use crate::{
    CKKSCtBounds, CKKSResult as Result, SetCKKSInfos,
    api::{CKKSAddOps, CKKSCopyOps},
    ckks_bail, ckks_ensure,
};
use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef, LWEInfos};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};
/// Guards `n` lazy accumulations against worst-case `i64` overflow.
///
/// Signed limb digits lie in `[−2^(base2k−1), 2^(base2k−1))`.  In the worst
/// case (all summands aligned in sign) the digit magnitude after `n` additions
/// is `n · 2^(base2k−1)`, which overflows `i64` once `n ≥ 2^(64 − base2k)`.
/// The bound enforced here, `n ≤ 2^(63 − base2k)`, provides one extra bit of
/// headroom below that threshold.
///
/// In the typical case (sign-balanced CKKS inputs) digit growth follows an
/// Irwin–Hall distribution with std dev `O(sqrt(n) · 2^(base2k−1) / sqrt(3))`,
/// so the practical limit is much higher than this conservative bound.
pub(crate) fn ensure_accumulation_fits<C: LWEInfos + ?Sized>(op: &'static str, dst: &C, n: usize) -> Result<()> {
    let base2k: usize = dst.base2k().as_usize();
    ckks_ensure!(base2k < 64, "{op}: unsupported base2k={base2k}");
    ckks_ensure!(
        n <= (1usize << (63 - base2k)),
        "{op}: {n} terms risks i64 overflow at base2k={base2k}",
    );
    Ok(())
}

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
            ensure_accumulation_fits("ckks_add_many", dst, inputs.len())?;
            module.ckks_add_into(dst, inputs[0], inputs[1], scratch)?;
            for ct in &inputs[2..] {
                module.ckks_add_assign(dst, *ct, scratch)?;
            }
        }
    }
    dst.set_noise(None);
    Ok(())
}
