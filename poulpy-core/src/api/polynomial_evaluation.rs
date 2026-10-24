use crate::layouts::GetTensorKey;
use anyhow::Result;
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::layouts::{BabyStep, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, Parity, PowerBasisHelper};

/// Arithmetic policy of the polynomial evaluation.
///
/// The derived schedule owns the combinatorial structure of the evaluation. It
/// has no concept of scale: every arithmetic operation (the `ct×pt` baby-step
/// terms, the hoisted `ct×ct` giant-step multiply, the `ct+ct` add, the
/// accumulator seed and the final copy) is supplied by the scheme through this
/// trait, which owns all precision bookkeeping and normalization.
pub trait BSGSOps<BE, V, P, A, R = V>
where
    BE: Backend,
    V: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    P: GLWEToBackendRef<BE>,
    A: GLWEToBackendRef<BE>,
    R: GLWEToBackendMut<BE>,
{
    /// Backend-resident prepared right multiply operand, reusable across a
    /// giant-step level.
    type Prepared;

    /// Initializes the accumulator `res` from `seed`'s precision: sets `res`'s
    /// metadata to that of `seed` and zeroes its data.
    fn init_accumulator(&self, module: &Module<BE>, res: &mut V, seed: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>;

    /// Computes `res[res_coeff] += coeffs[idx]`, normalizing `res`.
    fn add_pt_const_assign(
        &self,
        module: &Module<BE>,
        res: &mut V,
        res_coeff: usize,
        coeffs: &P,
        idx: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;

    /// Tries to evaluate `res = Σ coeffs[idx]·term` in one pass. Returning
    /// `false` leaves `res` untouched and selects the default operation sequence.
    fn eval_baby_linear_combination(
        &self,
        _module: &Module<BE>,
        _res: &mut V,
        _terms: &[(&A, usize)],
        _coeffs: &P,
        _scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<bool> {
        Ok(false)
    }

    /// Computes `res = a · coeffs[idx]` (ct × pt), setting `res`'s precision metadata.
    fn mul_pt_const(
        &self,
        module: &Module<BE>,
        res: &mut V,
        a: &A,
        coeffs: &P,
        idx: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;

    /// Computes `res += a · coeffs[idx]` (ct × pt), keeping `res` normalized.
    fn mul_add_pt_const(
        &self,
        module: &Module<BE>,
        res: &mut V,
        a: &A,
        coeffs: &P,
        idx: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;

    /// Prepares `a` as a reusable right operand for [`Self::mul_prepared_assign`].
    fn prepare_right(&self, module: &Module<BE>, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<Self::Prepared>;

    /// Computes `dst *= prepared` (ct × ct), relinearizing with `tsk` and
    /// stamping the consumed budget on the result.
    fn mul_prepared_assign<H>(
        &self,
        module: &Module<BE>,
        dst: &mut V,
        prepared: &Self::Prepared,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        H: GetTensorKey<BE>;

    /// Computes `dst += a` with budget alignment, normalizing `dst`.
    fn add_assign(&self, module: &Module<BE>, dst: &mut V, a: &V, scratch: &mut ScratchArena<'_, BE>) -> Result<()>;

    /// Computes `res = src`, stamping `res` with `src`'s effective precision.
    fn copy(&self, module: &Module<BE>, res: &mut R, src: &V, scratch: &mut ScratchArena<'_, BE>) -> Result<()>;
}

/// Baby-Step / Giant-Step polynomial-evaluation phases.
///
/// All arithmetic is supplied by the scheme through `ops`; the engine only
/// sequences the GLWE BSGS schedule.
pub trait GLWEPolynomialEvaluation<BE: Backend> {
    /// Evaluates a single baby step into `res`.
    #[allow(clippy::too_many_arguments)]
    fn glwe_eval_baby_step<Ops, V, P, A, G>(
        &self,
        ops: &Ops,
        res: &mut V,
        parity: Parity,
        coeffs: &P,
        power_basis: &G,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Ops: BSGSOps<BE, V, P, A, V>,
        V: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
        P: GLWEToBackendRef<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE>,
        G: PowerBasisHelper<BE, A>;

    /// Folds the evaluated baby steps into `res` using the giant-step schedule.
    fn glwe_eval_giant_steps<Ops, R, B, V, P, A, G, H>(
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
        H: GetTensorKey<BE>;
}
