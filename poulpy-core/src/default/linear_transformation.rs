//! GLWE-level linear transformation (matrix-vector product over the slots) in
//! baby-step / giant-step (BSGS) form.
//!
//! This module is scheme-agnostic: CKKS derives scale/capacity metadata and
//! passes only base2k alignment integers (`a_effective_k`, `cnv_offset`). The
//! files under `linear_transformation/` follow the phases in `docs/lt_bsgs.md`:
//! schedule construction (§3), setup/preparation (§5), baby hoisting (§6.2),
//! giant-step products and rotations (§6.3), and final normalization (§6.4).
//!
//! `docs/lt_bsgs_impl.md` is a file-by-file walkthrough of this module: how data
//! flows through the phases, where each spec saving lands, the two evaluation
//! paths (lazy DFT vs base-mismatch fallback), and the streamed unprepared-RHS
//! variant.

mod baby_steps;
mod eval;
mod inner_product;
mod lazy;
mod prepare;
mod prepared_giants;

#[cfg(test)]
mod tests;

// The data types and BSGS schedule derivation live in `crate::layouts`; the
// prepared (convolution-domain) caches live in `crate::layouts::prepared`.
// This module owns the HAL-dependent allocators and the prepare/eval
// algorithms. Re-exported here so `poulpy_core::*` keeps exposing them at the
// crate root.
pub use crate::layouts::prepared::{LinearTransformationBabySteps, LinearTransformationPrepared, PreparedDiagonal};
pub use crate::layouts::{
    LinearTransformation, LinearTransformationDiagonal, LinearTransformationGiantStep, LinearTransformationLayout,
    LinearTransformationPlan, LinearTransformationStrategy, optimal_bsgs_giant_step,
};

// Reference implementations forwarded to from `crate::oep::LinearTransformationDefault`.
pub use eval::{
    glwe_eval_linear_transformation_into_default, glwe_eval_linear_transformation_tmp_bytes_default,
    glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes_default, glwe_prepare_linear_transformation_baby_steps_default,
    glwe_prepare_linear_transformation_baby_steps_tmp_bytes_default,
};
pub use prepare::{glwe_prepare_linear_transformation_rhs_default, glwe_prepare_linear_transformation_rhs_tmp_bytes_default};
pub use prepared_giants::{DiagonalProd, glwe_accumulate_streamed_baby_steps_dft};
