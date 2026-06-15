//! Prepared (convolution-domain) operands for the GLWE linear transformation.
//!
//! The BSGS evaluation `M·v = Σ_k baby_k ⊗ diagonal_k` is a bivariate
//! convolution: the input baby rotations are the *left* operand (`CnvPVecL`)
//! and the matrix diagonals are the *right* operand (`CnvPVecR`). The two
//! caches here hold those prepared operands. They are allocated and populated
//! by the HAL-dependent routines in [`crate::default::linear_transformation`];
//! the unprepared transform and BSGS schedule types live in
//! [`crate::layouts::linear_transformation`](crate::layouts).

use std::collections::BTreeMap;

use poulpy_hal::layouts::{Backend, CnvPVecL, CnvPVecR, Data};

use crate::layouts::{Base2K, Degree, GLWEInfos, LWEInfos, LinearTransformation, Rank, TorusPrecision};

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
/// the shared evaluator reads `base2k` / `max_k` / `size` off it exactly as it
/// does off a plaintext — which is what lets both flavors share the single
/// [`LinearTransformation`](crate::layouts::LinearTransformation) container
/// (`LinearTransformation<PreparedDiagonal<…>>` vs
/// `LinearTransformation<CKKSPlaintext<…>>`).
///
/// `D` is the buffer backing the convolution operand: usually `BE::OwnedBuf`
/// (resident cache), but it may also be scratch-borrowed.
pub struct PreparedDiagonal<D: Data, BE: Backend> {
    /// Prepared convolution-domain diagonal (the BSGS right operand).
    pub(crate) cnv: CnvPVecR<D, BE>,
    /// Limb base of the diagonal it was encoded from.
    pub(crate) base2k: Base2K,
    /// Storage precision of the diagonal it was encoded from.
    pub(crate) max_k: TorusPrecision,
    /// Base-2 log of the plaintext scaling factor. The only scheme-provided
    /// quantity the convolution-offset computation needs; the core engine treats
    /// it as opaque and never reads it. Mirrors the streamed plaintext's
    /// `log_delta`; set by the scheme layer during the populate step.
    pub(crate) log_scale: usize,
}

impl<D: Data, BE: Backend> PreparedDiagonal<D, BE> {
    /// The prepared convolution-domain operand.
    pub fn cnv(&self) -> &CnvPVecR<D, BE> {
        &self.cnv
    }

    /// Mutable access to the prepared convolution-domain operand (populate step).
    pub fn cnv_mut(&mut self) -> &mut CnvPVecR<D, BE> {
        &mut self.cnv
    }

    /// Base-2 log of the plaintext scaling factor (opaque to the core engine).
    pub fn log_scale(&self) -> usize {
        self.log_scale
    }

    /// Sets the plaintext scale exponent; called by the scheme layer during the
    /// populate step.
    pub fn set_log_scale(&mut self, log_scale: usize) {
        self.log_scale = log_scale;
    }
}

impl<D: Data, BE: Backend> LWEInfos for PreparedDiagonal<D, BE> {
    fn n(&self) -> Degree {
        Degree(self.cnv.n() as u32)
    }

    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn size(&self) -> usize {
        self.cnv.size()
    }

    fn max_k(&self) -> TorusPrecision {
        self.max_k
    }
}

impl<D: Data, BE: Backend> GLWEInfos for PreparedDiagonal<D, BE> {
    fn rank(&self) -> Rank {
        Rank(self.cnv.cols() as u32 - 1)
    }
}

/// Prepared left operands for the baby rotations of one input ciphertext.
///
/// The values are populated by `glwe_prepare_linear_transformation_lhs`; the
/// cache is sized via [`LinearTransformationBabySteps::alloc`].
pub struct LinearTransformationBabySteps<BE: Backend> {
    pub(crate) values: BTreeMap<i64, CnvPVecL<BE::OwnedBuf, BE>>,
}

impl<BE: Backend> LinearTransformationBabySteps<BE> {
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
    pub fn baby_step(&self, rot: i64) -> &CnvPVecL<BE::OwnedBuf, BE> {
        self.values
            .get(&rot)
            .unwrap_or_else(|| panic!("missing prepared baby-step rotation {rot}"))
    }
}
