//! Safe, user-facing trait definitions for polynomial arithmetic operations.
//!
//! Scheme authors program against these traits; the computation is dispatched
//! to a backend through the [`oep`](crate::oep) extension points. Each trait
//! will document one operation with a structured contract (`op / class / mutation /
//! definition / domain / requires / ensures / fallback / override / exact /
//! test`; basis operations omit `definition`, `fallback` and `override`) once
//! the contract pass lands. The shared vocabulary those contracts use is
//! defined here, once.
//!
//! # Value model
//!
//! The ring is `R_N = Z[X]/(X^N + 1)`, `N` the module degree. A column of a
//! [`VecZnx`](crate::layouts::VecZnx) with `size` limbs, read at radix
//! `base2k`, denotes
//!
//! ```text
//! [[a]]_base2k = sum_{j < size} a_j * 2^(-base2k * (j + 1)),   a_j in R_N
//! ```
//!
//! `base2k` is a call parameter, not a property of the buffer, so the value is
//! always written with its radix. On a window (see
//! [`layouts`](crate::layouts#windows)) `a_j` is an element of `Z^n` instead.
//! [`VecZnxBig`](crate::layouts::VecZnxBig) denotes the same quantity with
//! wider limbs. [`VecZnxDft`](crate::layouts::VecZnxDft) and the prepared
//! types are opaque; their contracts are stated through `idft`.
//!
//! # Canonical form
//!
//! A column is canonical at radix `base2k` and precision `k` when both hold:
//!
//! - every digit lies in the centered range `[-2^(base2k-1), 2^(base2k-1))`,
//!   the range the `znx_normalize_*` step kernels of `poulpy-cpu-ref` produce;
//! - nothing lives below precision `k`: limbs `ceil(k / base2k)..size` are
//!   zero, and the low `(-k) mod base2k` bits of limb `ceil(k / base2k) - 1`
//!   are zero.
//!
//! The rounding rule applied when precision is dropped is the one the step
//! kernels implement; the test suite's `assert_canonical` pins both
//! conditions. `normalize` is the only operation that produces canonical
//! output, and its contract restates them.
//!
//! # Limb rule
//!
//! Every operation fully defines every visible element of `res`. Inputs
//! shorter than `res` are zero-extended; a `res` shorter than the exact
//! result truncates in the `2^(-base2k)` expansion. No operation leaves
//! limbs of `res` untouched.
//!
//! # Columns
//!
//! An operation takes one column index per operand, reads that column of each
//! input object and writes that column of `res`. Columns are independent.
//!
//! # Mutation classes
//!
//! Every operation belongs to one of three classes. The classes share one
//! arithmetic definition `f`; only the storage and mutation contract differs.
//!
//! - Out-of-place: `res_after = f(a, b)`. `res` is a distinct object from every
//!   input; the signatures (`&mut res`, `&a`, `&b`) make anything else
//!   unrepresentable, so no operation has an aliasing precondition. Inputs may
//!   share an object.
//! - In-place application, the `*_assign` operations: `res_after = f(res_before, a)`.
//!   The contract refers to the destination's value before the call.
//! - Accumulation: `res_after = res_before + f(a, b)`.
//!
//! Every contract names its class. An in-place or accumulating operation may be
//! realized by a scratch-based default body (`tmp = f(...)`, then copy or add
//! into `res`); a backend may override it with a kernel that needs no temporary.
//!
//! # Exactness
//!
//! Coefficient-domain and big-domain operations are exact and bit-identical
//! across backends. DFT-domain operations carry the backend's exactness class:
//! exact for NTT backends, approximate for FFT64, whose error bound is a
//! function of `N`, `base2k` and the operand sizes. Their contracts are stated
//! on `idft(...)`: `idft(dft(a)) = a`, `idft(svp_apply(dft(a), prep(s))) =
//! a * s`, `idft(vmp_apply(dft(a), prep(M))) = a * M`, and the DFT-domain
//! `add`, `sub`, `automorphism` are the images of the ring operations.
//!
//! # Preconditions
//!
//! Shape and parameter preconditions are checked by the backend kernels with
//! `assert!` in every build; the delegate layer is pure forwarding and never
//! asserts.
//! Numeric-range preconditions on input digits are caller obligations and are
//! never scanned.
//!
//! # Scratch
//!
//! A `scratch` argument must offer at least the matching `*_tmp_bytes`; its
//! contents are unspecified afterwards.

mod convolution;
mod module;
mod reim;
mod scratch;
mod svp_ppol;
mod vec_znx;
mod vec_znx_big;
mod vec_znx_dft;
mod vmp_pmat;

pub use convolution::*;
pub use module::*;
pub use reim::*;
pub use scratch::*;
pub use svp_ppol::*;
pub use vec_znx::*;
pub use vec_znx_big::*;
pub use vec_znx_dft::*;
pub use vmp_pmat::*;
