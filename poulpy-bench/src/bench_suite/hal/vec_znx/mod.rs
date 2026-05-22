//! Criterion benchmark harnesses for coefficient-domain [`VecZnx`](poulpy_hal::layouts::VecZnx) operations.
//!
//! Each submodule mirrors the corresponding `poulpy-cpu-ref` `vec_znx` implementation
//! file; the helpers are invoked from `poulpy-bench` harness binaries.

mod add;
mod automorphism;
mod mul_xp_minus_one;
mod negate;
mod normalize;
mod rotate;
mod shift;
mod sub;

pub use add::{bench_vec_znx_add_assign, bench_vec_znx_add_into};
pub use automorphism::{bench_vec_znx_automorphism, bench_vec_znx_automorphism_assign};
pub use mul_xp_minus_one::{bench_vec_znx_mul_xp_minus_one, bench_vec_znx_mul_xp_minus_one_assign};
pub use negate::{bench_vec_znx_negate, bench_vec_znx_negate_assign};
pub use normalize::{bench_vec_znx_normalize, bench_vec_znx_normalize_assign};
pub use rotate::{bench_vec_znx_rotate, bench_vec_znx_rotate_assign};
pub use shift::{bench_vec_znx_lsh, bench_vec_znx_lsh_assign, bench_vec_znx_rsh, bench_vec_znx_rsh_assign};
pub use sub::{bench_vec_znx_sub, bench_vec_znx_sub_assign, bench_vec_znx_sub_negate_assign};
