//! Prepared (convolution-domain) operands for the GLWE linear transformation.
//!
//! The BSGS evaluation `M·v = Σ_k baby_k ⊗ diagonal_k` is a bivariate
//! convolution: the input baby rotations are the *left* operand (`CnvPVecL`)
//! and the matrix diagonals are the *right* operand (`CnvPVecR`). The two
//! caches here hold those prepared operands. They are allocated and populated
//! by the HAL-dependent routines in [`crate::reference::linear_transformation`];
//! the unprepared transform and BSGS schedule types live in
//! [`crate::layouts::linear_transformation`](crate::layouts).

use poulpy_hal::layouts::{CnvPVecLBackendMut, CnvPVecLOwned, CnvPVecLToBackendMut};
use std::collections::BTreeMap;

use poulpy_hal::{
    api::{CnvPVecBytesOf, Convolution, ModuleN},
    layouts::{Backend, CnvPVecR, Data, ScratchArena, VecZnxDftBackendMut},
};

use crate::reference::linear_transformation::GiantStepProducts;

use crate::layouts::{
    Base2K, Degree, GLWEInfos, LWEInfos, LinearTransformation, LinearTransformationGiantStep, Rank, TorusPrecision,
};

/// A resident (prepared) linear transformation: the BSGS transform with every
/// diagonal stored as a convolution-domain [`PreparedDiagonal`].
///
/// This is the prepared analogue of the streamed
/// `LinearTransformation<CKKSPlaintext<…>>`; the two share the single
/// [`LinearTransformation`] container and differ only in the diagonal
/// representation `P`. Allocated up-front via
/// [`LinearTransformation::alloc_prepared`](LinearTransformation::alloc_prepared)
/// and populated by `glwe_prepare_linear_transformation_rhs`.
pub type LinearTransformationPrepared<BE> = LinearTransformation<PreparedDiagonal<<BE as Backend>::OwnedBuf, BE>>;

/// One prepared (convolution-domain) matrix diagonal: the right operand of the
/// BSGS convolution, paired with the plaintext layout it was encoded from.
///
/// This is the prepared analogue of an encoded plaintext diagonal. The streamed
/// transform stores each diagonal as a `CKKSPlaintext`; the resident transform
/// stores it as a `PreparedDiagonal`. Because this type implements [`LWEInfos`],
/// the shared evaluator reads `base2k` / `k` / `size` off it exactly as it
/// does off a plaintext — which is what lets both flavors share the single
/// [`crate::layouts::LinearTransformation`] container
/// (`LinearTransformation<PreparedDiagonal<…>>` vs
/// `LinearTransformation<CKKSPlaintext<…>>`).
///
/// `D` is the buffer backing the convolution operand: usually `BE::OwnedBuf`
/// (resident cache), but it may also be scratch-borrowed.
pub struct PreparedDiagonal<D: Data, BE: Backend> {
    /// Prepared convolution-domain diagonal (the BSGS right operand).
    pub(crate) cnv: CnvPVecR<D, BE::DftWord, BE>,
    /// Limb base of the diagonal it was encoded from.
    pub(crate) base2k: Base2K,
    /// Storage precision of the diagonal it was encoded from.
    pub(crate) k: TorusPrecision,
    /// Base-2 log of the plaintext scaling factor. The only scheme-provided
    /// quantity the convolution-offset computation needs; the core engine treats
    /// it as opaque and only checks that summed transforms share it. Mirrors the
    /// streamed plaintext's `log_delta`; set by the scheme layer during the
    /// populate step.
    pub(crate) log_scale: usize,
    /// Whether the scheme claims the diagonal's slots are real. Opaque to the
    /// core engine; `false` (no claim) until the scheme sets it.
    pub(crate) real_slots: bool,
}

impl<D: Data, BE: Backend> PreparedDiagonal<D, BE> {
    /// The prepared convolution-domain operand.
    pub fn cnv(&self) -> &CnvPVecR<D, BE::DftWord, BE> {
        &self.cnv
    }

    /// Mutable access to the prepared convolution-domain operand (populate step).
    pub fn cnv_mut(&mut self) -> &mut CnvPVecR<D, BE::DftWord, BE> {
        &mut self.cnv
    }

    /// Base-2 log of the plaintext scaling factor; the core engine only checks
    /// that summed transforms share it.
    pub fn log_scale(&self) -> usize {
        self.log_scale
    }

    /// Sets the plaintext scale exponent; called by the scheme layer during the
    /// populate step.
    pub fn set_log_scale(&mut self, log_scale: usize) {
        self.log_scale = log_scale;
    }

    /// Whether the scheme claims the diagonal's slots are real (opaque to the core engine).
    pub fn real_slots(&self) -> bool {
        self.real_slots
    }

    /// Sets the real-slot claim; called by the scheme layer during the populate step.
    pub fn set_real_slots(&mut self, real_slots: bool) {
        self.real_slots = real_slots;
    }
}

impl<D: Data, BE: Backend> crate::layouts::IntPolyInfos for PreparedDiagonal<D, BE> {
    /// Prepared diagonals are integer polynomials prepared across their whole
    /// stored width.
    fn encoded_k(&self) -> crate::layouts::TorusPrecision {
        self.max_k()
    }
}

impl<D: Data, BE: Backend> LWEInfos for PreparedDiagonal<D, BE> {
    fn n(&self) -> Degree {
        Degree(self.cnv.n() as u32)
    }

    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn max_size(&self) -> usize {
        self.cnv.size()
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }
}

impl<D: Data, BE: Backend> GLWEInfos for PreparedDiagonal<D, BE> {
    fn rank(&self) -> Rank {
        Rank(self.cnv.cols() as u32 - 1)
    }
}

/// Prepared left operands for the baby rotations of one input ciphertext.
///
/// The values are populated by `glwe_prepare_linear_transformation_baby_steps`; the
/// cache is sized via [`LinearTransformationBabySteps::alloc`].
pub struct LinearTransformationBabySteps<BE: Backend> {
    pub(crate) values: BTreeMap<i64, CnvPVecLOwned<BE>>,
    /// How the preparation reaches the rotations.
    pub(crate) rotations: crate::layouts::BsgsRotations,
}

impl<BE: Backend> LinearTransformationBabySteps<BE> {
    /// How the preparation reaches the rotations.
    pub fn rotations(&self) -> crate::layouts::BsgsRotations {
        self.rotations
    }

    pub fn size(&self) -> usize {
        let (_, cnv_pvec) = self.values.first_key_value().unwrap();
        cnv_pvec.size()
    }

    pub fn cols(&self) -> usize {
        let (_, cnv_pvec) = self.values.first_key_value().unwrap();
        cnv_pvec.cols()
    }

    /// The slot rotations represented by this prepared baby cache.
    pub fn baby_steps(&self) -> impl ExactSizeIterator<Item = i64> + '_ {
        self.values.keys().copied()
    }

    /// Returns true when `rot` is available in this prepared baby cache.
    pub fn contains_baby_step(&self, rot: i64) -> bool {
        self.values.contains_key(&rot)
    }

    /// Number of prepared baby rotations.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns true when no baby rotations are prepared.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns the prepared left operand `rot(v, k)` for the baby rotation `k`.
    pub fn baby_step(&self, rot: i64) -> &CnvPVecLOwned<BE> {
        self.values
            .get(&rot)
            .unwrap_or_else(|| panic!("missing prepared baby-step rotation {rot}"))
    }

    /// Borrows an allocated operand for preparation, as a view that cannot change its shape.
    pub fn baby_step_mut(&mut self, rot: i64) -> CnvPVecLBackendMut<'_, BE> {
        self.values
            .get_mut(&rot)
            .unwrap_or_else(|| panic!("missing prepared baby-step rotation {rot}"))
            .to_backend_mut()
    }

    /// Borrows all allocated operands in rotation order, as views that can change
    /// neither the cache keys nor the operand shapes.
    pub fn baby_steps_mut(&mut self) -> impl ExactSizeIterator<Item = (i64, CnvPVecLBackendMut<'_, BE>)> + '_ {
        self.values.iter_mut().map(|(&rot, operand)| (rot, operand.to_backend_mut()))
    }
}

/// Per-giant PROD, provided by the diagonal representation itself.
///
/// The shared evaluator `glwe_eval_giant_steps` is generic over the diagonal
/// type `P` stored in the [`LinearTransformation`]; the only per-flavor step is
/// how one giant step's `Σ_k ũ_{j,k} ⊙ rot(v,k)` is computed. Each concrete
/// diagonal type implements this once:
///
/// - [`PreparedDiagonal`] (resident): diagonals are already in convolution
///   domain, so the whole giant step is one fused accumulation
///   (`glwe_accumulate_prepared_baby_steps_dft`).
/// - a plaintext diagonal (streamed): each diagonal is prepared on the fly
///   through one reused scratch slot
///   (`glwe_accumulate_unprepared_baby_steps_dft`); implemented by the scheme
///   layer for its plaintext type.
///
/// Dispatching per concrete type, rather than via a blanket keyed on
/// [`GLWEToBackendRef`](crate::layouts::GLWEToBackendRef), is what keeps the
/// impls coherent: the backend type parameter prevents the compiler from
/// ruling out a downstream `GLWEToBackendRef` impl for [`PreparedDiagonal`], so
/// a blanket would clash with the resident impl.
pub trait DiagonalProd<BE: Backend>: LWEInfos + Sized {
    /// Runs the PROD inner product of one giant step into `prod_dft`.
    fn accumulate_giant_prod<M>(
        module: &M,
        cnv_offset_hi: usize,
        prod_dft: &mut VecZnxDftBackendMut<'_, BE>,
        lhs: &LinearTransformationBabySteps<BE>,
        gs: &LinearTransformationGiantStep<Self>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        M: CnvPVecBytesOf + Convolution<BE> + ModuleN;

    /// Most giant steps [`Self::accumulate_giant_prods`] takes per call.
    fn giant_step_batch<M>(_module: &M) -> usize
    where
        M: GiantStepProducts<BE>,
    {
        1
    }

    /// Runs the PROD inner products of several giant steps, `prods[g]` receiving step `steps[g]`.
    fn accumulate_giant_prods<M>(
        module: &M,
        cnv_offset_hi: usize,
        prods: &mut [VecZnxDftBackendMut<'_, BE>],
        lhs: &LinearTransformationBabySteps<BE>,
        steps: &[LinearTransformationGiantStep<Self>],
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        M: CnvPVecBytesOf + Convolution<BE> + ModuleN + GiantStepProducts<BE>,
    {
        assert_eq!(prods.len(), steps.len());
        for (prod, step) in prods.iter_mut().zip(steps) {
            Self::accumulate_giant_prod(module, cnv_offset_hi, prod, lhs, step, scratch);
        }
    }
}
