//! Reference bodies of the bootstrapping steps, one file per step. The transforms between them are in
//! [`reference::dft`](crate::reference::dft), and the circuit of EvalMod in
//! [`reference::eval_mod`](crate::reference::eval_mod).

use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSResult as Result, SetCKKSInfos, SlotsKind,
    api::{CKKSAddOps, CKKSConjugateOps, CKKSImagOps},
    layouts::BootstrappingLinearKeys,
};

mod eval_lut;
mod eval_mod;
mod eval_mod_ci;
mod mod_raise;

pub use eval_lut::CKKSEvalLutReference;
pub use eval_mod::CKKSBootstrapEvalModReference;
pub use mod_raise::{
    CKKSEncapsulatedModUpReference, CKKSModRaiseReference, ckks_encapsulated_mod_up_reference,
    ckks_encapsulated_mod_up_tmp_bytes_reference,
};

fn recombine_halves<BE, R1, R2>(
    module: &Module<BE>,
    res_re: &mut R1,
    res_im: &mut R2,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSImagOps<BE> + CKKSAddOps<BE>,
    R1: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    R2: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
{
    module.ckks_mul_i_assign(res_im, scratch)?;
    module.ckks_add_assign(res_re, &*res_im, scratch)?;
    Ok(())
}

/// `res = a + conj(a) = 2·Re(a)`: the input polynomial's coefficients.
fn ckks_bootstrap_real_part<BE, K, R, A>(
    module: &Module<BE>,
    res: &mut R,
    a: &A,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSConjugateOps<BE> + CKKSAddOps<BE>,
    K: BootstrappingLinearKeys<BE>,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    module.ckks_conjugate_into(res, a, keys, scratch)?;
    module.ckks_add_assign(res, a, scratch)?;
    res.set_slots(SlotsKind::Real);
    Ok(())
}
