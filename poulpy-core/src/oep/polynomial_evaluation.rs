use crate::layouts::GLWERelinearizationKeyHelper;
use anyhow::Result;
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    default::polynomial_evaluation::BSGSOps,
    layouts::{
        BabyStep, GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, Parity, PowerBasisHelper,
        prepared::{GGLWEPreparedToBackendRef, GLWETensorKeyPreparedToBackendRef},
    },
};

/// Backend-provided Baby-Step / Giant-Step polynomial-evaluation phases.
///
/// # Safety
/// Implementations must preserve the BSGS schedule semantics and the precision
/// metadata contract expected by the scheme-supplied operations.
pub unsafe trait PolynomialEvaluationImpl<BE: Backend>: Backend {
    #[allow(clippy::too_many_arguments)]
    fn glwe_eval_baby_step<Ops, R, P, A, G>(
        module: &Module<BE>,
        ops: &Ops,
        res: &mut R,
        parity: Parity,
        coeffs: &P,
        power_basis: &G,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Ops: BSGSOps<BE, R, P, A, R>,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
        P: GLWEToBackendRef<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE>,
        G: PowerBasisHelper<BE, A>;

    #[allow(clippy::too_many_arguments)]
    fn glwe_eval_giant_steps<Ops, R, B, V, P, A, G, H>(
        module: &Module<BE>,
        ops: &Ops,
        res: &mut R,
        baby_steps: &mut [B],
        power_basis: &G,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Ops: BSGSOps<BE, V, P, A, R>,
        R: GLWEToBackendMut<BE>,
        B: BabyStep<BE, Value = V>,
        V: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
        P: GLWEToBackendRef<BE>,
        A: GLWEToBackendRef<BE>,
        G: PowerBasisHelper<BE, A>,
        H: GLWERelinearizationKeyHelper,
        H::Key: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>;
}

/// Override surface carrying the reference BSGS phase implementations.
pub trait PolynomialEvaluationDefault<BE: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn glwe_eval_baby_step_default<Ops, R, P, A, G>(
        &self,
        ops: &Ops,
        res: &mut R,
        parity: Parity,
        coeffs: &P,
        power_basis: &G,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Ops: BSGSOps<BE, R, P, A, R>,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
        P: GLWEToBackendRef<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE>,
        G: PowerBasisHelper<BE, A>;

    fn glwe_eval_giant_steps_default<Ops, R, B, V, P, A, G, H>(
        &self,
        ops: &Ops,
        res: &mut R,
        baby_steps: &mut [B],
        power_basis: &G,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Ops: BSGSOps<BE, V, P, A, R>,
        R: GLWEToBackendMut<BE>,
        B: BabyStep<BE, Value = V>,
        V: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
        P: GLWEToBackendRef<BE>,
        A: GLWEToBackendRef<BE>,
        G: PowerBasisHelper<BE, A>,
        H: GLWERelinearizationKeyHelper,
        H::Key: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>;
}

unsafe impl<BE: Backend> PolynomialEvaluationImpl<BE> for BE
where
    Module<BE>: PolynomialEvaluationDefault<BE>,
{
    fn glwe_eval_baby_step<Ops, R, P, A, G>(
        module: &Module<BE>,
        ops: &Ops,
        res: &mut R,
        parity: Parity,
        coeffs: &P,
        power_basis: &G,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Ops: BSGSOps<BE, R, P, A, R>,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
        P: GLWEToBackendRef<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE>,
        G: PowerBasisHelper<BE, A>,
    {
        module.glwe_eval_baby_step_default::<Ops, R, P, A, G>(ops, res, parity, coeffs, power_basis, scratch)
    }

    fn glwe_eval_giant_steps<Ops, R, B, V, P, A, G, H>(
        module: &Module<BE>,
        ops: &Ops,
        res: &mut R,
        baby_steps: &mut [B],
        power_basis: &G,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Ops: BSGSOps<BE, V, P, A, R>,
        R: GLWEToBackendMut<BE>,
        B: BabyStep<BE, Value = V>,
        V: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
        P: GLWEToBackendRef<BE>,
        A: GLWEToBackendRef<BE>,
        G: PowerBasisHelper<BE, A>,
        H: GLWERelinearizationKeyHelper,
        H::Key: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>,
    {
        module.glwe_eval_giant_steps_default::<Ops, R, B, V, P, A, G, H>(ops, res, baby_steps, power_basis, tsk, scratch)
    }
}
