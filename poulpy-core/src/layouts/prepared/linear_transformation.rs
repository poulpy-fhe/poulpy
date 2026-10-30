//! Prepared (convolution-domain) operands for the GLWE linear transformation.
//!
//! The BSGS evaluation `M·v = Σ_k baby_k ⊗ diagonal_k` is a bivariate
//! convolution: the input baby rotations are the *left* operand (`CnvPVecL`)
//! and the matrix diagonals are the *right* operand (`CnvPVecR`). The two
//! caches here hold those prepared operands. They are allocated and populated
//! by the HAL-dependent routines in [`crate::reference::linear_transformation`];
//! the unprepared transform and BSGS schedule types live in
//! [`crate::layouts::linear_transformation`](crate::layouts).

use poulpy_hal::layouts::{CnvPVecLBackendMut, CnvPVecLOwned, CnvPVecLToBackendMut, VecZnxInfos};
use std::collections::BTreeMap;

use poulpy_hal::{
    api::{CnvPVecBytesOf, Convolution, ModuleN, ScratchArenaTakeBasic},
    layouts::{
        Backend, CnvDftAccTerm, CnvPVecLToBackendRef, CnvPVecR, CnvPVecRToBackendRef, Data, Module, PrepareHint, ScratchArena,
        VecZnxDftBackendMut,
    },
};

use crate::{
    gadget::prod_dft_size,
    layouts::{
        Base2K, Degree, DiagonalOperand, GLWEBackendRef, GLWEInfos, LWEInfos, LinearTransformation,
        LinearTransformationGiantStep, Rank, TorusPrecision,
    },
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
    /// What the scheme claims of the plaintext the diagonal encodes, set during the populate step: opaque to the core
    /// engine, which only checks that summed transforms share its scale. The default claims nothing.
    pub(crate) plaintext_meta: crate::layouts::GLWEPlaintextMeta,
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

    /// What the scheme claims of the plaintext the diagonal encodes.
    pub fn plaintext_meta(&self) -> crate::layouts::GLWEPlaintextMeta {
        self.plaintext_meta
    }

    /// Sets the scheme's claim; called by the scheme layer during the populate step.
    pub fn set_plaintext_meta(&mut self, meta: crate::layouts::GLWEPlaintextMeta) {
        self.plaintext_meta = meta;
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

/// A prepared baby rotation of a [`LinearTransformationBabySteps`] cache: owned
/// ([`CnvPVecLOwned`]) or carved from a scratch arena ([`CnvPVecLViewMut`](poulpy_hal::layouts::CnvPVecLViewMut)).
/// Blanket-implemented.
pub trait BabyStepOperand<BE: Backend>: CnvPVecLToBackendRef<BE> + CnvPVecLToBackendMut<BE> + VecZnxInfos + Send {}

impl<BE: Backend, T> BabyStepOperand<BE> for T where T: CnvPVecLToBackendRef<BE> + CnvPVecLToBackendMut<BE> + VecZnxInfos + Send {}

/// Prepared left operands for the baby rotations of one input ciphertext.
///
/// The values are populated by `glwe_prepare_linear_transformation_baby_steps`. A cache
/// a caller keeps is allocated by [`LinearTransformationBabySteps::alloc`], and a cache
/// one call uses is carved from its scratch arena by
/// [`ScratchArenaTakeCore::take_linear_transformation_baby_steps_scratch`](crate::ScratchArenaTakeCore::take_linear_transformation_baby_steps_scratch),
/// its operands of type `L` either way.
// Generic over the operand rather than its storage `D`: hal gives backend views of a `CnvPVecL` of owned storage
// and of its scratch wrapper, and a pair of impls over `BE::OwnedBuf` and `BE::BufMut<'_>` would overlap.
pub struct LinearTransformationBabySteps<BE: Backend, L = CnvPVecLOwned<BE>> {
    pub(crate) values: BTreeMap<i64, L>,
    /// How the preparation reaches the rotations.
    pub(crate) rotations: crate::layouts::BsgsRotations,
    /// Width of the source the rotations were prepared from.
    pub(crate) k: TorusPrecision,
    pub(crate) _backend: std::marker::PhantomData<fn() -> BE>,
}

impl<BE: Backend, L: BabyStepOperand<BE>> LinearTransformationBabySteps<BE, L> {
    /// How the preparation reaches the rotations.
    pub fn rotations(&self) -> crate::layouts::BsgsRotations {
        self.rotations
    }

    pub fn size(&self) -> usize {
        let (_, cnv_pvec) = self.values.first_key_value().unwrap();
        cnv_pvec.size()
    }

    /// Width of the source the rotations were prepared from, that of the source it was allocated for until prepared.
    pub fn k(&self) -> TorusPrecision {
        self.k
    }

    /// Records the width of the source the rotations were prepared from, as every preparation of the cache does.
    pub fn set_k(&mut self, k: TorusPrecision) {
        self.k = k;
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
    pub fn baby_step(&self, rot: i64) -> &L {
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

/// Per-giant PROD, read off the diagonal representation through [`Self::operand`].
///
/// The shared evaluator `glwe_eval_giant_steps` is generic over the diagonal type `P` stored in the
/// [`LinearTransformation`]; each concrete diagonal type gives its diagonal once, as a [`DiagonalOperand`]:
///
/// - prepared ([`PreparedDiagonal`], resident): the diagonals are already in convolution domain, so the whole giant
///   step is one fused accumulation (`glwe_accumulate_prepared_baby_steps_dft`);
/// - unprepared, a GLWE view (streamed): each diagonal is prepared on the fly through one reused scratch slot
///   (`glwe_accumulate_unprepared_baby_steps_dft`).
///
/// The rank of the diagonals picks the product: a transform whose diagonals are of rank 0 multiplies them as
/// plaintexts through [`Self::accumulate_giant_prod`], one whose diagonals are of the rank of its source multiplies
/// them as ciphertexts, tensored and relinearized, preparing an unprepared one in the call.
///
/// Dispatching per concrete type, rather than via a blanket keyed on
/// [`GLWEToBackendRef`](crate::layouts::GLWEToBackendRef), is what keeps the
/// impls coherent: the backend type parameter prevents the compiler from
/// ruling out a downstream `GLWEToBackendRef` impl for [`PreparedDiagonal`], so
/// a blanket would clash with the resident impl.
pub trait DiagonalProd<BE: Backend>: GLWEInfos + Sized {
    /// The diagonal as the right operand of its products: prepared, or a GLWE view the product prepares.
    fn operand(&self) -> DiagonalOperand<'_, BE>;

    /// The scale the scheme records on the diagonal, which the terms of a sum share; 0 where it records none.
    fn log_scale(&self) -> usize {
        0
    }

    /// Runs the PROD inner product of one giant step of plaintext diagonals into `prod_dft`, in the form of its first
    /// diagonal, which every diagonal of a transform shares.
    fn accumulate_giant_prod<M, L: BabyStepOperand<BE>>(
        module: &M,
        cnv_offset_hi: usize,
        prod_dft: &mut VecZnxDftBackendMut<'_, BE>,
        lhs: &LinearTransformationBabySteps<BE, L>,
        gs: &LinearTransformationGiantStep<Self>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        M: CnvPVecBytesOf + Convolution<BE> + ModuleN,
    {
        let first = gs
            .diagonals
            .first()
            .expect("linear transformation giant step has no diagonals");
        match first.plaintext.operand() {
            DiagonalOperand::Prepared(_) => {
                glwe_accumulate_prepared_baby_steps_dft(module, cnv_offset_hi, prod_dft, lhs, gs, scratch)
            }
            DiagonalOperand::Unprepared(_) => {
                glwe_accumulate_unprepared_baby_steps_dft(module, cnv_offset_hi, prod_dft, lhs, gs, scratch)
            }
        }
    }

    /// Most giant steps [`Self::accumulate_giant_prods`] takes per call, at least 1.
    fn giant_step_batch(_module: &Module<BE>) -> usize
    where
        BE: crate::oep::LinearTransformationImpl,
    {
        1
    }

    /// Runs the PROD inner products of several giant steps, `prods[g]` receiving step `steps[g]`.
    fn accumulate_giant_prods<L: BabyStepOperand<BE>>(
        module: &Module<BE>,
        cnv_offset_hi: usize,
        prods: &mut [VecZnxDftBackendMut<'_, BE>],
        lhs: &LinearTransformationBabySteps<BE, L>,
        steps: &[LinearTransformationGiantStep<Self>],
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BE: crate::oep::LinearTransformationImpl,
        Module<BE>: CnvPVecBytesOf + Convolution<BE> + ModuleN,
    {
        assert_eq!(prods.len(), steps.len());
        for (prod, step) in prods.iter_mut().zip(steps) {
            Self::accumulate_giant_prod(module, cnv_offset_hi, prod, lhs, step, scratch);
        }
    }
}

/// The prepared operand of a diagonal of a giant step whose first diagonal is prepared.
fn prepared<BE: Backend, P>(operand: DiagonalOperand<'_, BE, P>) -> &P {
    match operand {
        DiagonalOperand::Prepared(prep) => prep,
        DiagonalOperand::Unprepared(_) => panic!("a giant step mixes prepared and unprepared diagonals"),
    }
}

/// The view of a diagonal of a giant step whose first diagonal is unprepared.
fn unprepared<BE: Backend>(operand: DiagonalOperand<'_, BE>) -> GLWEBackendRef<'_, BE> {
    match operand {
        DiagonalOperand::Unprepared(view) => view,
        DiagonalOperand::Prepared(_) => panic!("a giant step mixes prepared and unprepared diagonals"),
    }
}

/// PROD block for one giant step of a resident (prepared) transform, kept in DFT
/// domain.
///
/// The diagonals are already in convolution domain, so the whole giant step is
/// handed to the backend as a single fused accumulation per output column: it
/// sums all `baby ⊛ diagonal` terms with one reduction per output limb and
/// writes `prod_dft` once. Leaves the `(r+1)` output columns in `prod_dft`.
pub(crate) fn glwe_accumulate_prepared_baby_steps_dft<BE, M, P, L: BabyStepOperand<BE>>(
    module: &M,
    cnv_offset_hi: usize,
    prod_dft: &mut VecZnxDftBackendMut<'_, BE>,
    lhs: &LinearTransformationBabySteps<BE, L>,
    gs: &LinearTransformationGiantStep<P>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: Convolution<BE>,
    P: DiagonalProd<BE>,
{
    let cols = lhs.cols();
    let first = gs
        .diagonals
        .first()
        .expect("prepared linear transformation giant step has no diagonals");
    let diagonal_size = prepared(first.plaintext.operand()).size();
    let res_dft_size = prod_dft_size(lhs.size(), diagonal_size, cnv_offset_hi);
    assert_eq!(prod_dft.cols(), cols);
    assert_eq!(prod_dft.size(), res_dft_size);

    for col in 0..cols {
        let terms: Vec<CnvDftAccTerm<'_, BE>> = gs
            .diagonals
            .iter()
            .map(|d| {
                let diagonal = prepared(d.plaintext.operand());
                let baby = lhs.baby_step(d.baby);
                assert_eq!(baby.cols(), cols);
                assert_eq!(prod_dft_size(baby.size(), diagonal.size(), cnv_offset_hi), res_dft_size);
                CnvDftAccTerm {
                    a: baby.to_backend_ref(),
                    a_col: col,
                    b: diagonal.to_backend_ref(),
                    b_col: 0,
                    b_galois: 1,
                }
            })
            .collect();
        module.cnv_apply_dft_sum(cnv_offset_hi, prod_dft, col, &terms, scratch);
    }
}

/// PROD block for one giant step from an *unprepared* matrix: identical to
/// [`glwe_accumulate_prepared_baby_steps_dft`] except each diagonal is prepared
/// (`cnv_prepare_right`) on the fly into a single reused scratch `CnvPVecR` and
/// discarded immediately, so the full prepared RHS is never materialized. Trades
/// per-eval recompute for memory when memory bandwidth is the limiting factor. The
/// baby loop is the outer one so each diagonal is prepared exactly once per eval.
fn glwe_accumulate_unprepared_baby_steps_dft<BE, M, P, L: BabyStepOperand<BE>>(
    module: &M,
    cnv_offset_hi: usize,
    prod_dft: &mut VecZnxDftBackendMut<'_, BE>,
    lhs: &LinearTransformationBabySteps<BE, L>,
    gs: &LinearTransformationGiantStep<P>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: CnvPVecBytesOf + Convolution<BE> + ModuleN,
    P: DiagonalProd<BE>,
{
    let cols = lhs.cols();
    let first = gs
        .diagonals
        .first()
        .expect("streamed linear transformation giant step has no diagonals");
    let first = unprepared(first.plaintext.operand());
    // The streamed diagonal is an integer poly encoded across its full physical
    // width, so its width and size are `max_k`/`max_size`, not the (possibly
    // smaller) effective `k`/`size`.
    let diagonal_size = first.max_size();
    // A diagonal is prepared at its own degree; the apply reads it
    // through the convolution's embedding slot.
    let diagonal_n = first.n().as_usize();
    let res_dft_size = prod_dft_size(lhs.size(), diagonal_size, cnv_offset_hi);
    assert_eq!(prod_dft.cols(), cols);
    assert_eq!(prod_dft.size(), res_dft_size);

    // One reused right-operand slot: the whole RHS streams through it.
    let (mut diagonal, mut scratch_1) =
        scratch
            .borrow()
            .take_cnv_pvec_right_scratch(diagonal_n, 1, diagonal_size, PrepareHint::OneShot);

    // Baby is the outer loop, so the first baby initializes every output column
    // (overwrite) and the rest accumulate in place; each diagonal is prepared once.
    for (term_idx, d) in gs.diagonals.iter().enumerate() {
        let baby = lhs.baby_step(d.baby);
        assert_eq!(baby.cols(), cols);
        assert_eq!(prod_dft_size(baby.size(), diagonal_size, cnv_offset_hi), res_dft_size);
        let plaintext = unprepared(d.plaintext.operand());
        assert_eq!(
            plaintext.n().as_usize(),
            diagonal_n,
            "streamed linear transformation diagonals do not share one degree"
        );

        // Stream the RHS: prepare this diagonal on the fly, then reuse the slot.
        module.cnv_prepare_right(&mut diagonal, &plaintext.data, &mut scratch_1.borrow());

        for col in 0..cols {
            if term_idx == 0 {
                module.cnv_apply_dft(
                    cnv_offset_hi,
                    prod_dft,
                    col,
                    &baby.to_backend_ref(),
                    col,
                    &diagonal.to_backend_ref(),
                    0,
                    1,
                    &mut scratch_1.borrow(),
                );
            } else {
                module.cnv_apply_dft_add(
                    cnv_offset_hi,
                    prod_dft,
                    col,
                    &baby.to_backend_ref(),
                    col,
                    &diagonal.to_backend_ref(),
                    0,
                    1,
                    &mut scratch_1.borrow(),
                );
            }
        }
    }
}
