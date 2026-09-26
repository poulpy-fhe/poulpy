//! Safe, user-facing trait definitions for polynomial arithmetic operations.
//!
//! The structured contracts below use the mathematical vocabulary defined here.
//! A trait may have separate blocks for operations with different contracts.
//!
//! # Contract blocks
//!
//! A block is fenced as `text`, with one line per key:
//!
//! ```text
//! op         vec_znx_add(res, res_col, a, a_col, b, b_col)
//! class      basis
//! mutation   out-of-place
//! definition res[res_col,j] = a[a_col,j] + b[b_col,j]; other columns of res are unchanged
//! domain     res, a, b: VecZnx or windows of one, of the same degree
//! ensures    every limb of the selected output column is defined
//! test       test_vec_znx_add_matches_reference
//! ```
//!
//! `op`, `class`, `mutation`, `domain`, `ensures` and `test` are always
//! present, `definition` on every class but `support`; `requires`,
//! `fallback`, `override` and `sparse` appear where applicable.
//!
//! - `class` is `basis` (an independently defined operation), `variant`
//!   (an operation with a different storage or mutation contract), `derived`
//!   (an operation with an overridable default), or `support` (allocation,
//!   dimensions, scratch sizes, or plans, with no arithmetic).
//! - Every `basis`, `variant` and `derived` block has exactly one `definition`
//!   line; a `support` block has none.
//! - A basis definition is a formula specifying the entire visible result;
//!   variants and derived operations may instead compose named HAL operations
//!   with all arguments, including destination columns and scratch.
//! - A definition states unchanged coefficients, limbs and columns explicitly;
//!   it does not rely on `ensures` or surrounding prose to complete its meaning.
//! - `fallback` names the inherited implementation and `override` states
//!   whether it may be replaced, including any paired scratch-size operation.
//! - `test` names public functions of [`test_suite`](crate::test_suite) that
//!   check the operation; only support blocks may use `none`.
//!
//! # Value model
//!
//! The backend selects the polynomial ring and its coefficient basis.
//! By default, `R_n = Z[X]/(X^n + 1)` with the monomial basis. In the
//! conjugate invariant ring, `R_n` is the fixed subring of `Z[X]/(X^(2n)+1)`
//! with basis `1, X^j + X^(-j)` for `1 <= j < n`. Automorphisms `sigma_p` substitute
//! `X -> X^p` in that basis; monomial multiplication requires the standard ring.
//! Prepared objects and keys must be used with the ring configuration that
//! produced them, even when their backend types and degrees agree.
//! A module is built at a
//! degree `N` and serves every power-of-two degree `n` with
//! `MIN_DEGREE <= n <= N`, `MIN_DEGREE` the backend's floor; in a contract
//! `N` denotes the degree of the call, the degree its operands share, which
//! the kernel reads from them and which is at most the module's.
//! A selected column `a[c]` with `a.size()` limbs and radix width `b` denotes
//! the coefficient-wise dyadic value
//!
//! ```text
//! [[a[c]]]_b = sum_{0 <= j < a.size()} a[c,j] * 2^(-b * (j + 1)).
//! ```
//!
//! The radix width is a call parameter, not a property of the object.
//! [`VecZnx`](crate::layouts::VecZnx) and
//! [`VecZnxBig`](crate::layouts::VecZnxBig) share this value model.
//! On coefficient windows, polynomial notation denotes the visible vector of
//! coefficients; ring operations require the degree specified by their domain.
//! [`VecZnxDft`](crate::layouts::VecZnxDft) is opaque and is observed through
//! `idft`; prepared operands are observed as the polynomials they read as.
//!
//! # Notation
//!
//! - `a[c,j]` is limb `j` of column `c`, a polynomial of degree less than
//!   `a.n()`, and `a[c,j,i]` is its coefficient of `X^i`.
//! - `a[c]` is a whole vector column; for a scalar or prepared scalar it is
//!   the polynomial in that column, also written `a[c,0]`.
//! - `v[j]` selects a limb of a column and `p[i]` selects coefficient `i`
//!   of a polynomial `p`.
//! - `mat[r,c,d,j]` is the polynomial at row `r`, input column `c`, output
//!   column `d` and limb `j` of a matrix, with coefficient selection by a
//!   further index `i`.
//! - A prepared scalar, vector or matrix uses the same indexing as its source
//!   and reads as the logical polynomial its preparation selected, regardless
//!   of how it is stored.
//! - `idft(a)[c,j]` is the coefficient-domain polynomial denoted by transformed
//!   limb `j` of column `c`, and `idft(a)[c,j,i]` selects its coefficient.
//! - `old(a)` denotes the entire object `a` immediately before the call;
//!   unmodified input objects retain that value throughout a definition.
//! - An unbound `j` on a definition line means every integer
//!   `0 <= j < destination.size()`, where the destination is `res` or the
//!   in-place operand; a call parameter named `j` instead keeps its argument
//!   meaning, and the output limb index must then be bound explicitly.
//! - All indices are integers, every other index is bound on its definition
//!   line, and a sum over an empty index set is zero.
//! - Input limbs outside `0 <= j < a.size()` read as zero, and an input
//!   column outside `0 <= c < a.cols()` reads as zero only where a definition
//!   explicitly uses column alignment or zero extension.
//! - `X^e` in `R_n`, for any integer `e`, means
//!   `(-1)^floor(e/n) * X^(e mod n)`, with `0 <= e mod n < n`.
//! - `tau_p(q)` is substitution `q(X^p)` in `R_n`, for odd `p`.
//! - `plan.p` is the odd exponent supplied when an automorphism plan was made.
//! - `rnd(x,k) = floor(x * 2^k + 1/2) / 2^k` rounds coefficient-wise to
//!   precision `k`, with ties toward positive infinity.
//! - `canon(v,b,k,L)` is the unique column of `L` limbs canonical at radix
//!   width `b` and precision `k` whose value is congruent coefficient-wise
//!   to `rnd(v,k)` modulo 1, for `0 <= k <= L*b`.
//! - `live(k,b) = ceil(k/b)` is the number of potentially nonzero limbs at
//!   precision `k`, and `pad(k,b) = (-k) mod b` is the low-bit padding count.
//! - `draw(s,t,b)` is the low `b` bits of the `t`-th 64-bit word drawn from
//!   the stream `s`, numbered from zero, as an unsigned integer.
//! - `reseed(source)` is the pseudorandom stream seeded by the 32 bytes the
//!   call draws from `source`; the call consumes exactly those bytes of
//!   `source`.
//! - `addend_limb(None,j) = 0`, while `addend_limb(Some((a,c)),j) = a[c,j]`
//!   denotes a limb of an optional selected-column addend.
//! - `terms[t]` is term `t` of a convolution list, with fields `a`, `a_col`,
//!   `b` and `b_col` naming its prepared operands and selected columns.
//! - `wrap(x)` is the signed residue of `x` modulo `2^w` in
//!   `[-2^(w-1), 2^(w-1))`, where `w` is the destination's big coefficient
//!   width; it applies coefficient-wise to polynomials.
//! - For a raw complex transform, `m` is the table length returned by `m()`,
//!   `I` is the imaginary unit,
//!   `pi` is the circle constant, and `z(data,t) = data[t] + I*data[m+t]`.
//! - `rev_m(r)` reverses the `log2(m)` bits of `r`, with `rev_1(0) = 0`.
//! - `omega(m,r) = exp(2*pi*I*(rev_m(r) + 1/4)/m)` is evaluation root `r`
//!   of the raw complex transform.
//!
//! # Canonical form
//!
//! A column is canonical at radix width `b` and precision `k` when every
//! coefficient of every limb lies in `[-2^(b-1), 2^(b-1))`, all limbs from
//! `live(k,b)` onward are zero, and, when `k > 0`, the low `pad(k,b)` bits
//! of limb `live(k,b)-1` are zero.
//! At precision zero the whole column is zero.
//! Normalization chooses the unique canonical column congruent modulo 1 to
//! the value rounded by `rnd`; centered digits do not imply that the value
//! itself belongs to `[-1/2, 1/2)`.
//!
//! # Limb rule
//!
//! Definitions apply to every visible limb of the named output column.
//! Missing input limbs read as zero; shorter outputs truncate limbwise
//! operations, while normalization rounds to its explicit precision.
//! Any preserved coefficient or limb window is stated on the definition line.
//!
//! # Columns
//!
//! Every operand column is named explicitly, including all column indices
//! in matrix and multi-column operations; preserved columns are stated in
//! the definition.
//!
//! # Mutation classes
//!
//! - Out-of-place operations write a distinct destination from input objects.
//! - In-place operations read the pre-call value denoted by `old`.
//! - Accumulating operations add their contribution to `old(res)`.
//! - Support operations have mutation `none`.
//!
//! In a composition, reading the destination as an input to the same named
//! call refers to its value immediately before that call.
//!
//! # Exactness
//!
//! Every operation yields exactly the result its definition states, under
//! the bounds its domain states. An integer formula without `wrap` requires
//! each stored intermediate and output coefficient to fit its coefficient
//! type; normalization additionally has its stated input bounds, while the
//! dyadic values inside `canon` and `rnd` are not restricted by a machine
//! coefficient width. A formula using `wrap` specifies arithmetic modulo the
//! destination big coefficient width, which may differ between
//! implementations. A backend whose transform is floating point rounds back
//! to the integers the definition states, within the bounds it documents, so
//! a DFT-domain result does not depend on the backend; DFT definitions are
//! stated through `idft` and constrain neither representation equality nor
//! accumulation order. The raw complex transform of `NegacyclicFFT` is the one
//! floating-point operation: its definition is an ideal complex formula and
//! its result carries floating arithmetic error.
//!
//! # Degree
//!
//! A computing operation takes no degree argument: its operands share one
//! degree `N`, read from `res` (or the in-place operand), and every kernel
//! asserts at entry that the module serves it. An allocation, `bytes_of_*`,
//! scratch take or plan has no polynomial operand and takes the degree `n`
//! as its first argument. The `*_tmp_bytes` are sized at the module's degree,
//! an upper bound for a call at a smaller one.
//!
//! # Degree embedding
//!
//! Every operation is defined on `R_N`, `N` the degree of the call. A
//! degree-`n` object in a sparse-capable slot, `n` a power-of-two divisor of
//! `N`, denotes its image `p(X^(N/n))` in `R_N`, and a formula on such an
//! operand uses that image; the compact storage is a representation choice
//! the contracts never mention. Only the operand slots a contract's `sparse`
//! line names accept a degree other than `N`; `res` and every other
//! operand share `N`. Only a dense operand takes such a degree: a window in a
//! sparse-capable slot has the width of `res`. The coefficient-domain `add`
//! and `sub` families and `vec_znx_big_from_small` accept the slot, so the
//! derived small-operand forms inherit it. The shift-add and shift-sub forms
//! of `VecZnx` accept it in `a`. In the convolution only the prepared right
//! operand is sparse-capable, and the prepares are not sparsity-aware: a
//! degree-`n` right operand is prepared at degree `n`, like any dense
//! prepare, and the degree-`N` apply forms read it through the backend's slot
//! correspondence, while the left operand and the result take the degree `N`
//! of the call. A backend may reject a sparse degree below its transform
//! block width in the apply; the coefficient-domain add and sub families
//! accept any power-of-two divisor. `switch_ring` states both directions of
//! the coefficient map explicitly.
//!
//! Coefficient-wise reductions and products act on visible coefficient
//! vectors rather than `R_N`; their contracts specify all degree relations.
//!
//! # Preconditions
//!
//! The `domain` and `requires` lines give the shape and parameter
//! preconditions; the backend kernels check them with `assert!` in every
//! build, and the delegate layer is pure forwarding and never asserts.
//! Numeric input bounds and representability are caller obligations and
//! are not checked by scanning input coefficients.
//!
//! # Scratch
//!
//! A `scratch` argument must offer at least the matching `*_tmp_bytes`;
//! its contents are unspecified afterwards.

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
