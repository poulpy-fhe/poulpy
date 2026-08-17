//! Criterion benchmark harnesses for coefficient-domain [`VecZnx`](poulpy_hal::layouts::VecZnx) operations.
//!
//! Each submodule mirrors the corresponding `poulpy-cpu-ref` `vec_znx` implementation
//! file; the runners are assembled into `BenchOp` tables in `hal::suites`.

mod add;
mod automorphism;
mod mul_xp_minus_one;
mod negate;
mod normalize;
mod rotate;
mod shift;
mod sub;

pub use add::{runner_vec_znx_add_assign, runner_vec_znx_add_into};
pub use automorphism::{runner_vec_znx_automorphism, runner_vec_znx_automorphism_assign};
pub use mul_xp_minus_one::{runner_vec_znx_mul_xp_minus_one, runner_vec_znx_mul_xp_minus_one_assign};
pub use negate::{runner_vec_znx_negate, runner_vec_znx_negate_assign};
pub use normalize::{runner_vec_znx_normalize, runner_vec_znx_normalize_assign};
pub use rotate::{runner_vec_znx_rotate, runner_vec_znx_rotate_assign};
pub use shift::{runner_vec_znx_lsh, runner_vec_znx_lsh_assign, runner_vec_znx_rsh, runner_vec_znx_rsh_assign};
pub use sub::{runner_vec_znx_sub, runner_vec_znx_sub_assign, runner_vec_znx_sub_negate_assign};
