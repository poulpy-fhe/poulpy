//! GLWE-level linear transformation (matrix-vector product over the slots) in
//! baby-step / giant-step (BSGS) form.
//!
//! This module is scheme-agnostic: CKKS derives scale/capacity metadata and
//! passes only base2k alignment integers (`a_effective_k`, `cnv_offset`). The
//! files under `linear_transformation/` follow the phases in `docs/lt_bsgs.md`:
//! schedule construction (§3), setup/preparation (§5), baby hoisting (§6.2),
//! giant-step products and rotations (§6.3), and final normalization (§6.4).

mod baby_steps;
mod eval;
mod index;
mod inner_product;
mod lazy;
mod ops;
mod prepare;
mod prepared_giants;
mod types;

#[cfg(test)]
mod tests;

pub use baby_steps::{GLWEPreparedBabyRotations, GLWEPreparedBabyStepHelper};
pub use index::{bsgs_index, linear_transform_index, normalize_linear_transform_diagonal, optimal_bsgs_giant_step};
pub use ops::GLWELinearTransformOps;
pub use prepare::GLWEPrepareLinearTransformOps;
pub use types::{
    GLWELinearTransform, GLWELinearTransformDiagonal, GLWELinearTransformGiantStep, GLWELinearTransformIndex,
    GLWEPreparedLinearTransform, GLWEPreparedLinearTransformGiantStep, LinearTransformationStrategy,
};
