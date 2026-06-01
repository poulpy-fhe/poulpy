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

use poulpy_hal::layouts::{Backend, CnvPVecL, CnvPVecR};

use crate::layouts::{Base2K, TorusPrecision};

/// A prepared giant step (right side: the matrix diagonals).
pub struct GLWEPreparedLinearTransformationRhsGiantStep<BE: Backend> {
    /// Slot rotation amount.
    pub(crate) rot: i64,
    /// Indexes into [`GLWEPreparedLinearTransformationRhs::baby_steps`] used by this giant step.
    pub(crate) baby_step_indexes: Vec<usize>,
    /// Prepared right convolution operands keyed by real baby-step rotation.
    pub(crate) diagonals: BTreeMap<i64, CnvPVecR<BE::OwnedBuf, BE>>,
}

impl<BE: Backend> GLWEPreparedLinearTransformationRhsGiantStep<BE> {
    /// Slot rotation amount applied by this giant step.
    pub fn rot(&self) -> i64 {
        self.rot
    }

    /// Indexes into the enclosing transform's prepared baby-step rotation list.
    pub fn baby_step_indexes(&self) -> &[usize] {
        &self.baby_step_indexes
    }

    /// First baby-step index used by this giant step.
    pub fn first_baby_step_index(&self) -> usize {
        *self
            .baby_step_indexes
            .first()
            .expect("linear transformation giant step has no terms")
    }

    /// Number of prepared diagonals carried by this giant step.
    pub fn num_diagonals(&self) -> usize {
        self.diagonals.len()
    }

    /// Returns true when a prepared diagonal exists for the given baby-step rotation.
    pub fn contains_diagonal(&self, baby_rot: i64) -> bool {
        self.diagonals.contains_key(&baby_rot)
    }

    /// Prepared diagonal operand for the given real baby-step rotation.
    pub fn diagonal(&self, baby_rot: i64) -> &CnvPVecR<BE::OwnedBuf, BE> {
        self.diagonals
            .get(&baby_rot)
            .unwrap_or_else(|| panic!("missing prepared diagonal for baby-step rotation {baby_rot}"))
    }
}

/// A linear transformation with pruned BSGS metadata and prepared diagonals
/// (the right operand of the BSGS convolution).
///
/// The cache is allocated up-front by
/// [`GLWEPreparedLinearTransformationRhs::alloc`] from a
/// [`LinearTransformationLayout`](crate::layouts::LinearTransformationLayout)
/// and a plaintext-shape proxy; the diagonals are then populated by
/// `glwe_prepare_linear_transformation_rhs`. Eval pulls the plaintext limb
/// layout ([`Self::pt_base2k`] / [`Self::pt_max_k`]) from this struct, so it no
/// longer needs the raw
/// [`GLWELinearTransform`](crate::layouts::GLWELinearTransform) alongside.
///
/// The struct is scheme-agnostic. Deriving the convolution offset and the output
/// precision needs exactly one scheme-provided integer beyond the limb layout:
/// the plaintext scale exponent, exposed by [`Self::pt_log_scale`]. The core
/// engine never reads it; the scheme layer sets it during the populate step (via
/// [`Self::set_pt_log_scale`]) and consumes it when computing `cnv_offset`.
///
/// All fields are private; read them through the accessors below.
pub struct GLWEPreparedLinearTransformationRhs<BE: Backend> {
    /// Baby-step rotations actually used by at least one diagonal.
    pub(crate) baby_steps: Vec<i64>,
    /// Non-empty giant steps.
    pub(crate) giant_steps: Vec<GLWEPreparedLinearTransformationRhsGiantStep<BE>>,
    /// Limb base of the encoded diagonals; same for every diagonal.
    pub(crate) pt_base2k: Base2K,
    /// Storage precision of the encoded diagonals; same for every diagonal.
    pub(crate) pt_max_k: TorusPrecision,
    /// Base-2 log of the plaintext scaling factor. This is the only
    /// scheme-provided quantity the convolution-offset computation needs; the
    /// core engine treats it as opaque and never reads it. Set by the populate
    /// step.
    pub(crate) pt_log_scale: usize,
}

impl<BE: Backend> GLWEPreparedLinearTransformationRhs<BE> {
    /// Baby-step rotations used by at least one diagonal (index `0` first).
    pub fn baby_steps(&self) -> &[i64] {
        &self.baby_steps
    }

    /// The non-empty giant steps of this prepared transform.
    pub fn giant_steps(&self) -> &[GLWEPreparedLinearTransformationRhsGiantStep<BE>] {
        &self.giant_steps
    }

    /// Limb base of the encoded diagonals; shared by every diagonal.
    pub fn pt_base2k(&self) -> Base2K {
        self.pt_base2k
    }

    /// Storage precision of the encoded diagonals; shared by every diagonal.
    pub fn pt_max_k(&self) -> TorusPrecision {
        self.pt_max_k
    }

    /// Base-2 log of the plaintext scaling factor (opaque to the core engine).
    pub fn pt_log_scale(&self) -> usize {
        self.pt_log_scale
    }

    /// Sets the plaintext scale exponent; called by the scheme layer during the
    /// populate step.
    pub fn set_pt_log_scale(&mut self, pt_log_scale: usize) {
        self.pt_log_scale = pt_log_scale;
    }

    /// Returns the real baby-step rotation stored at `baby_step_idx`.
    pub fn baby_step_rotation(&self, baby_step_idx: usize) -> i64 {
        self.baby_steps
            .get(baby_step_idx)
            .copied()
            .unwrap_or_else(|| panic!("missing prepared baby-step index {baby_step_idx}"))
    }

    /// The non-zero baby- and giant-step rotations whose automorphism keys are required.
    pub fn required_rotations(&self) -> Vec<i64> {
        let mut rots: Vec<i64> = self.baby_steps.iter().copied().filter(|&r| r != 0).collect();
        rots.extend(self.giant_steps.iter().map(|gs| gs.rot).filter(|&r| r != 0));
        rots.sort_unstable();
        rots.dedup();
        rots
    }
}

/// Prepared left operands for the baby rotations of one input ciphertext.
///
/// The values are populated by `glwe_prepare_linear_transformation_lhs`; the
/// cache is sized via [`GLWEPreparedLinearTransformationLhs::alloc`].
pub struct GLWEPreparedLinearTransformationLhs<BE: Backend> {
    pub(crate) values: BTreeMap<i64, CnvPVecL<BE::OwnedBuf, BE>>,
}

impl<BE: Backend> GLWEPreparedLinearTransformationLhs<BE> {
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
