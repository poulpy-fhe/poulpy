//! Safe, user-facing trait definitions for polynomial arithmetic operations.
//!
//! Scheme authors program against these traits; the computation is dispatched
//! to a backend through the [`oep`](crate::oep) extension points. Every trait
//! below documents its operations with one prose paragraph and one structured
//! contract block. The shared vocabulary those blocks use is defined here,
//! once.
//!
//! # Contract blocks
//!
//! A block is fenced as `text`, one line per key:
//!
//! ```text
//! op         vec_znx_add(res, res_col, a, a_col, b, b_col)
//! class      basis
//! mutation   out-of-place
//! domain     res, a, b: VecZnx or windows of one, read at one shared base2k
//! ensures    res[res_col] = a[a_col] + b[b_col] limb by limb
//! test       test_vec_znx_add_matches_reference
//! ```
//!
//! `op`, `class`, `mutation`, `domain`, `ensures` and `test` are
//! always present; `definition`, `requires`, `fallback`, `override` and
//! `sparse` appear where they have something to say.
//!
//! - `class` is `basis` (its own definition, required of every backend),
//!   `variant` (a basis definition under a different storage or mutation
//!   contract, still required), `derived` (an OEP default body that every
//!   backend inherits and may override) or `support` (allocation, byte sizes,
//!   scratch reports: nothing arithmetic).
//! - `mutation` is one of the three classes below, or `none` on a support
//!   trait, which computes nothing.
//! - `definition` is the composition a variant or a derived operation stands
//!   for, `fallback` the body a backend inherits, `override` whether a backend
//!   may replace it and which `_tmp_bytes` it must replace with it.
//! - `test` names the `pub fn` of [`test_suite`](crate::test_suite) that pins
//!   the operation. A test in this crate reads every block, checks that each
//!   trait carries one and that its class brings the lines it needs, and
//!   resolves the names on the `test` line against the suite.
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
//!   the range the backend's normalize step kernels produce;
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
//! Every operation's result is the exact integer result under the operand
//! bounds its contract states. A backend whose transform is floating point
//! rounds back to those integers within the bounds it documents, so the
//! result does not depend on the backend. The DFT-domain types are opaque;
//! their contracts are stated on `idft(...)`: `idft(dft(a)) = a`,
//! `idft(svp_apply(dft(a), prep(s))) = a * s`, `idft(vmp_apply(dft(a),
//! prep(M))) = a * M`, and the DFT-domain `add`, `sub`, `automorphism` are
//! the images of the ring operations.
//!
//! # Degree embedding
//!
//! Every operation is defined on `R_N`. A degree-`n` object, `n` a
//! power-of-two divisor of `N`, denotes its image `p(X^(N/n))` under
//! `switch_ring`; the compact storage, `N/n` smaller, is a representation
//! choice the contracts never mention. Only the operand slots a contract's
//! `sparse` line names accept a degree other than the module's; `res` and
//! every other operand share `N`. The coefficient-domain `add` and `sub`
//! families read such an operand with a stride. In the convolution only the
//! prepared right operand is sparse-capable, and the prepares are not
//! sparsity-aware: a degree-`n` right operand is prepared under a degree-`n`
//! module, like any dense prepare, and the apply forms of a degree-`N` module
//! read it through the backend's slot correspondence, while the left operand
//! and the result take the module degree. A backend may reject a
//! sparse degree below its transform block width in the apply; the
//! coefficient-domain add and sub families accept any power-of-two divisor.
//!
//! The coefficient-wise operations are outside that rule: `vec_znx_big_inner_sum`,
//! `vec_znx_big_col_weighted_sum` and `vec_znx_scalar_product` act on
//! coefficients, not on `R_N`. Their operands take any degree, each contract's
//! `domain` line gives the relations between them, and the module's degree
//! does not enter. LWE encryption drives them on buffers of the LWE dimension
//! and reduces to degree one.
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
