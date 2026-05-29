//! Public data structures for GLWE linear transformations.
//!
//! These structs mirror the notation of docs/lt_bsgs.md §3 and §5: unprepared
//! transforms store BSGS buckets of encoded diagonals, while prepared transforms
//! additionally cache each diagonal as a right convolution operand (`CnvPVecR`).

use std::collections::BTreeMap;

use poulpy_hal::layouts::{Backend, CnvPVecR};

/// One non-zero diagonal of the linear map, attached to a giant step.
///
/// `plaintext` is the pre-rotated diagonal `u~_{j,k} = rot(diag_{n1*j+k}, -n1*j)`,
/// already encoded as a plaintext polynomial. `baby` is the baby-step slot
/// rotation `k` whose prepared `rot(v, k)` it multiplies.
pub struct GLWELinearTransformDiagonal<P> {
    /// Baby-step slot rotation.
    pub baby: i64,
    /// Pre-rotated diagonal, encoded as a plaintext polynomial.
    pub plaintext: P,
}

/// A single giant step `j`: its inner sum is rotated by `rot` slots and added to
/// the output (`rot == 0` is the identity giant step).
pub struct GLWELinearTransformGiantStep<P> {
    /// Slot rotation amount `n1*j`.
    pub rot: i64,
    /// The non-zero diagonals contributing to this giant step.
    pub diagonals: Vec<GLWELinearTransformDiagonal<P>>,
}

/// A linear transformation prepared in baby-step / giant-step form.
///
/// `P` is the encoded-plaintext container.
pub struct GLWELinearTransform<P> {
    /// Distinct baby-step slot rotations `k`; `baby_steps[0] == 0` (the identity).
    pub baby_steps: Vec<i64>,
    /// The giant steps.
    pub giant_steps: Vec<GLWELinearTransformGiantStep<P>>,
}

/// Strategy used to derive a linear-transformation evaluation schedule from
/// non-zero diagonal indexes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearTransformationStrategy {
    /// Pick a BSGS giant step with the Lattigo-compatible cost rule.
    Auto,
    /// Use an explicit BSGS giant-step width.
    Bsgs { giant_step: usize },
    /// Use one giant step per diagonal and no baby rotations.
    Direct,
}

/// BSGS index schedule for a linear transformation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GLWELinearTransformIndex {
    /// Distinct baby-step rotations used by the schedule.
    pub baby_steps: Vec<i64>,
    /// Giant-step rotations used by the schedule.
    pub giant_steps: Vec<i64>,
    /// Baby-step rotations grouped by giant step.
    ///
    /// `index[g]` contains the real baby rotations `k` used with
    /// `giant_steps[g]`. The corresponding diagonal is
    /// `giant_steps[g] + k` modulo the slot count.
    pub index: Vec<Vec<i64>>,
}

/// A prepared giant step.
pub struct GLWEPreparedLinearTransformGiantStep<BE: Backend> {
    /// Slot rotation amount.
    pub rot: i64,
    /// Indexes into [`GLWEPreparedLinearTransform::baby_steps`] used by this giant step.
    pub baby_step_indexes: Vec<usize>,
    /// Prepared right convolution operands keyed by real baby-step rotation.
    pub diagonals: BTreeMap<i64, CnvPVecR<BE::OwnedBuf, BE>>,
}

/// A linear transformation with pruned BSGS metadata and prepared diagonals.
pub struct GLWEPreparedLinearTransform<BE: Backend> {
    /// Baby-step rotations actually used by at least one diagonal.
    pub baby_steps: Vec<i64>,
    /// Non-empty giant steps.
    pub giant_steps: Vec<GLWEPreparedLinearTransformGiantStep<BE>>,
}

impl<BE: Backend> Default for GLWEPreparedLinearTransform<BE> {
    fn default() -> Self {
        Self {
            baby_steps: Vec::new(),
            giant_steps: Vec::new(),
        }
    }
}

impl<P> GLWELinearTransform<P> {
    /// The non-zero baby- and giant-step rotations whose automorphism keys are required.
    pub fn required_rotations(&self) -> Vec<i64> {
        let mut rots: Vec<i64> = self
            .giant_steps
            .iter()
            .flat_map(|gs| gs.diagonals.iter())
            .map(|d| d.baby)
            .filter(|&r| r != 0)
            .collect();
        rots.extend(
            self.giant_steps
                .iter()
                .filter(|gs| !gs.diagonals.is_empty())
                .map(|gs| gs.rot)
                .filter(|&r| r != 0),
        );
        rots.sort_unstable();
        rots.dedup();
        rots
    }
}

impl<BE: Backend> GLWEPreparedLinearTransform<BE> {
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

impl<BE: Backend> GLWEPreparedLinearTransformGiantStep<BE> {
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

    /// Prepared diagonal operand for the given real baby-step rotation.
    pub fn diagonal(&self, baby_rot: i64) -> &CnvPVecR<BE::OwnedBuf, BE> {
        self.diagonals
            .get(&baby_rot)
            .unwrap_or_else(|| panic!("missing prepared diagonal for baby-step rotation {baby_rot}"))
    }
}

impl GLWELinearTransformIndex {
    /// The non-zero baby- and giant-step rotations required by this schedule.
    pub fn required_rotations(&self) -> Vec<i64> {
        let mut rots: Vec<i64> = self.baby_steps.iter().copied().filter(|&k| k != 0).collect();
        rots.extend(self.giant_steps.iter().copied().filter(|&r| r != 0));
        rots.sort_unstable();
        rots.dedup();
        rots
    }
}
