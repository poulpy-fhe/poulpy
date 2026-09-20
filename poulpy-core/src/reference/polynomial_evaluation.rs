//! Arithmetic policy and HAL bounds for polynomial evaluation.
//!
//! The crate-private derived schedule owns the combinatorial structure of the
//! evaluation. It has no concept of scale: every
//! arithmetic operation (the `ct×pt` baby-step terms, the hoisted `ct×ct`
//! giant-step multiply, the `ct+ct` add, the accumulator seed and the final
//! copy) is supplied by the scheme through [`BSGSOps`], which owns all precision
//! bookkeeping and normalization.

use crate::layouts::GetTensorKey;
use anyhow::Result;
use poulpy_hal::{
    api::{
        CnvPVecBytesOf, Convolution, ModuleN, VecZnxAddAssign, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxCopy, VecZnxDftBytesOf, VecZnxIdftApplyTmpA, VecZnxNegate, VecZnxNormalizeAssign, VecZnxNormalizeTmpBytes,
        VecZnxSubAssign,
    },
    layouts::{Backend, Module, ScratchArena},
};

use crate::layouts::{GLWEToBackendMut, GLWEToBackendRef};

/// HAL bounds required to run the hoisted prepared-right tensor product.
///
/// Retained as a convenience bundle for the **scheme** implementations of
/// [`BSGSOps::mul_prepared_assign`] (the engine itself no longer touches
/// these primitives).
pub trait GiantStepTensorBounds<BE: Backend>:
    Sized
    + ModuleN
    + CnvPVecBytesOf
    + VecZnxDftBytesOf
    + VecZnxBigBytesOf
    + VecZnxIdftApplyTmpA<BE>
    + VecZnxBigNormalize<BE>
    + Convolution<BE>
    + VecZnxSubAssign<BE>
    + VecZnxAddAssign<BE>
    + VecZnxBigNormalizeTmpBytes
    + VecZnxCopy<BE>
    + VecZnxNegate<BE>
    + VecZnxNormalizeAssign<BE>
    + VecZnxNormalizeTmpBytes
{
}

impl<BE: Backend, M> GiantStepTensorBounds<BE> for M where
    M: Sized
        + ModuleN
        + CnvPVecBytesOf
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + Convolution<BE>
        + VecZnxSubAssign<BE>
        + VecZnxAddAssign<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxCopy<BE>
        + VecZnxNegate<BE>
        + VecZnxNormalizeAssign<BE>
        + VecZnxNormalizeTmpBytes
{
}

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
