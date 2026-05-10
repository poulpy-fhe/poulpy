# poulpy-ckks Polynomial Evaluation Implementation Plan

## Scope

This plan translates `docs/polynomial-evaluation-spec.md` into a concrete
implementation path for `poulpy-ckks`.

It is written for the current repository shape, including the in-progress
polynomial-evaluation files already present under:

- `poulpy-ckks/src/polynomial.rs`
- `poulpy-ckks/src/api/polynomial_evaluation.rs`
- `poulpy-ckks/src/default/polynomial_evaluation.rs`
- `poulpy-ckks/src/delegates/polynomial_evaluation.rs`
- `poulpy-ckks/src/oep/polynomial_evaluation.rs`
- `poulpy-ckks/src/test_suite/polynomial_evaluation.rs`

The target is a CKKS-native, backend-agnostic implementation that follows the
same public/delegate/OEP/default layering as the rest of `poulpy-ckks`.

## Current State

The current draft already has a good skeleton:

- `Basis`, `Polynomial`, `PowerBasis`, `split_degree`, and `optimal_split` are
  present.
- The API/OEP/delegate/default chain is wired in the same style as other CKKS
  operations.
- A pre-encoded baby-step representation exists:
  `BSGSPolynomial<EncodedPolynomialStep<CKKSPlaintext<_>>>`.
- The default evaluator evaluates baby steps with plaintext-constant
  multiplication and combines them with giant steps.
- The test suite has a legacy cubic test and a degree-7 monomial test.

The draft is intentionally incomplete:

- Chebyshev decomposition and Chebyshev power generation are not implemented.
- There is no need for a Lattigo-style recursive metadata planner in Poulpy's
  base-`2^k` torus setting.
- Power generation can stay simple because `ckks_mul_into` already performs
  the tensor/key-switching behavior required by `poulpy-ckks`.
- The high-level API requires callers to build/populate a `PowerBasis` and
  encode a BSGS polynomial manually.
- `PolynomialInfos::coeffs<C>()` and `BSGSPolynomialInfos::get_baby_step<P>()`
  rely on unchecked type casts. This matches some current test patterns but is
  not ideal for a public long-term API.

## Design Direction

For `poulpy-ckks`, the Lattigo algorithm should be adapted to Poulpy's torus
precision model rather than copied literally.

Lattigo tracks modulus-chain levels and therefore compensates coefficients for
future modulus-chain changes. Poulpy's ciphertext modulus is a power-of-two
torus width, so coefficient plaintexts do not need that recursive compensation
step.

Poulpy tracks:

- `log_delta`: semantic fixed-point precision
- `log_budget`: remaining homomorphic headroom
- `effective_k = log_delta + log_budget`
- `max_k`: storage capacity

Therefore the implementation should not add a recursive `CKKSMeta` simulator
just to mirror Lattigo. It only needs a decomposition shape:

- the baby-step base
- the baby polynomials
- the powers required in the `PowerBasis`
- the multiplicative depth/budget check

Coefficient plaintexts can be encoded directly at a caller- or API-selected
`CKKSMeta`, typically matching the intended plaintext precision. Existing CKKS
operation metadata rules then propagate `log_delta` and `log_budget`.

The adapted invariant is:

```text
Every baby-step ciphertext and every giant-step product must have metadata that
allows ckks_add_assign/ckks_add_assign_unnormalized to align operands without
unexpected precision loss beyond the destination capacity policy.
```

In practice, this means:

- Use existing CKKS operation metadata rules as the source of truth.
- Let `ckks_add_assign` align log budgets during giant-step addition.
- Let the destination/capacity policy of the normal CKKS operations determine
  the final stored precision.
- Keep coefficient encoding policy explicit and simple. Do not recursively
  compensate coefficients for future giant steps.

## Target Architecture

### Public Data Model

Keep `poulpy-ckks/src/polynomial.rs` as the home for host-side polynomial
objects:

- `Basis`
- `Polynomial`
- `PowerBasis`
- `BSGSPolynomial`
- `EncodedPolynomialStep`
- decomposition helpers such as `split_degree`, `optimal_split`, and
  `bit_length`

`BSGSPolynomial<C>` should be the object given to the evaluator. It is
constructed on the host from `P(X)` and stores encoded baby polynomials in:

```rust
Vec<C>
```

with the evaluator-side bound:

```rust
C: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos
```

In practice, `C` is a `CKKSPlaintext` or a backend/device-owned plaintext
uploaded from the host-constructed helper. The evaluator retrieves
`&baby_steps[i]` by index; it does not need any separate polynomial collection
or slot-mapping abstraction.

Add a small evaluation-shape helper, not a recursive metadata planner:

```rust
pub struct PolynomialEvaluationShape {
    pub degree: usize,
    pub log_degree: usize,
    pub log_split: usize,
    pub base: usize,
    pub required_powers: Vec<usize>,
    pub multiplicative_depth: usize,
}
```

The exact field names can change. The important boundary is that host-side
encoding and power-basis population consume the same decomposition shape, while
CKKS metadata propagation remains delegated to the existing arithmetic kernels.

### API Layer

Keep a low-level API for pre-encoded polynomials:

```rust
ckks_eval_poly_const_coeffs(res, poly, power_basis, tsk, scratch)
```

Add a convenience API after the low-level path is stable:

```rust
ckks_eval_poly(res, input, polynomial, tsk, scratch)
```

The convenience method should:

1. Build the evaluation shape.
2. Encode baby-step coefficient plaintexts.
3. Build and populate the power basis.
4. Call `ckks_eval_poly_const_coeffs`.

This keeps the fast pre-encoded path available for callers that reuse a
polynomial across many ciphertexts.

### OEP/Default Boundary

Keep the current backend-facing dispatch shape:

```text
api::PolynomialEvaluation
  -> delegates::polynomial_evaluation
  -> oep::CKKSPolynomialEvaluationImpl
  -> default::PolynomialEvaluationDefault
```

The default trait should own the generic algorithm body. Backends can override
only `CKKSPolynomialEvaluationImpl` if they need a specialized evaluator.

This mirrors the rest of `poulpy-ckks` and preserves the hidden-default
override pattern used in the broader Poulpy architecture.

## Milestone Plan

### Milestone 1: Stabilize the Monomial Pre-Encoded Path

Goal: make the current monomial BSGS path correct, documented, and covered by
tests before adding Chebyshev support.

Work items:

- Replace all mathematical `ceil(log2(degree))` wording/code comments with
  `bit_length(degree)`, matching the upstream Lattigo boundary behavior.
- Add tests for degrees `1`, `2`, `3`, `4`, `7`, `8`, and `15`.
- Add tests for constant polynomial handling, or explicitly reject degree `0`
  at the public constructor boundary with a clear error.
- Verify `giant_step_power(degree) = 1 << bit_length(degree)` for all tested
  baby-step degrees.
- Ensure `PowerBasis::populate` generates exactly the powers required by the
  decomposition, not just those required by the current degree-7 test.
- Confirm the output destination capacity behavior by testing a destination
  with both exact and larger-than-needed `max_k`.

Acceptance criteria:

- `cargo check -p poulpy-ckks --lib`
- Polynomial tests pass for `poulpy-cpu-ref`.
- The low-level pre-encoded monomial API is usable without test-only helper
  structs.

### Milestone 2: Remove Unsafe Type-Erasure From the Public Polynomial Traits

Goal: make the polynomial API safer before users start depending on it.

Current traits expose methods like:

```rust
fn coeffs<C>(&self) -> &C
fn get_baby_step<P>(&self, i: usize) -> &P
```

These require unchecked casts in implementations. Replace them with associated
types where possible:

```rust
pub trait PolynomialInfos<BE: Backend> {
    type Coeffs: GLWEToBackendRef<BE> + GLWEInfos + CKKSInfos;
    fn degree(&self) -> usize;
    fn is_odd(&self) -> bool;
    fn is_even(&self) -> bool;
    fn coeffs(&self) -> &Self::Coeffs;
}

pub trait BSGSPolynomialInfos<BE: Backend> {
    type Step: PolynomialInfos<BE>;
    fn degree(&self) -> usize;
    fn baby_steps(&self) -> usize;
    fn baby_step(&self, i: usize) -> &Self::Step;
    fn basis(&self) -> Basis;
}
```

If associated types become too restrictive for object-safety or backend
genericity, keep the generic methods crate-private and expose concrete
`BSGSPolynomial<C>` APIs publicly.

Acceptance criteria:

- No `unsafe` pointer casts are needed for normal polynomial evaluation data
  structures.
- Test-only compatibility shims are either removed or isolated in tests.

### Milestone 3: Define Direct Coefficient Metadata Policy

Goal: make coefficient encoding explicit without adding recursive coefficient
compensation.

The policy should define:

- how `Polynomial::encode_bsgs` chooses or receives coefficient `CKKSMeta`
- how much input `log_budget` is required from multiplicative depth
- how destination capacity is handled when an evaluated value is wider than
  the output buffer

Start with a deliberately simple rule:

- The caller supplies a coefficient `CKKSMeta`, or the high-level API derives
  one from the input/output precision target.
- Every encoded baby-step plaintext uses that same metadata.
- No coefficient is adjusted differently because of its future PS position.
- Budget checks use the existing CKKS multiplication rules and a conservative
  depth estimate.

This is enough because base-`2^k` arithmetic does not require Lattigo's RNS
coefficient compensation.

Acceptance criteria:

- `Polynomial::encode_bsgs` either keeps an explicit `coeff_meta` argument or
  has a high-level wrapper that derives it predictably.
- Tests fail early with a useful error when `log_budget` is insufficient.
- No recursive metadata-planning pass is needed for monomial evaluation.

### Milestone 4: Implement Monomial `PowerBasis` Policies

Goal: make power-basis generation a faithful, reusable component.

For Poulpy CKKS, `ckks_mul_into` already tensor-keyswitches back into a normal
CKKS ciphertext. Keep the algorithm structured like the spec:

- `gen_power(n)` should use `split_degree(n)`.
- Generated powers should be cached and reused.
- `populate_for_shape(shape)` should call `gen_power` for exactly the shape's
  required powers.

Do not implement Chebyshev powers in this milestone unless monomial is already
fully green.

Acceptance criteria:

- Reusing one `PowerBasis` for two polynomial evaluations avoids recomputing
  powers.
- Missing powers produce clear errors at evaluation time.
- Power generation tests compare decrypted `X^n` against cleartext powers for
  `n = 2..8`.

### Milestone 5: Complete the Default Baby/Giant Evaluator

Goal: make `default/polynomial_evaluation.rs` match the spec's evaluation
shape while respecting Poulpy metadata.

Work items:

- Make baby-step evaluation allocate its accumulator from a deterministic
  metadata/capacity rule, such as the first required power or an explicit
  accumulator template. It does not need planned per-baby output metadata.
- Keep unnormalized accumulation for `sum coeff_i * X^i`, but enforce a safe
  bound on the number of accumulated terms.
- In giant-step combination, rely on `ckks_mul_assign` and `ckks_add_assign`
  for metadata updates and alignment.
- Add explicit sanity checks after each giant step:
  - output metadata fits allocated storage
  - no unexpected budget underflow
  - required power exists

Acceptance criteria:

- Baby-step and giant-step unit tests can exercise the internal helpers via
  public evaluation cases.
- Precision behavior is stable across degree boundaries.
- Errors identify whether failure happened in power lookup, baby evaluation,
  giant multiplication, addition alignment, or destination-capacity fitting.

### Milestone 6: Add High-Level `ckks_eval_poly`

Goal: provide the ergonomic API users will actually call.

Suggested signature shape:

```rust
fn ckks_eval_poly<R, X, T>(
    &self,
    res: &mut R,
    x: &X,
    polynomial: &Polynomial,
    tsk: &T,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
```

This method should:

- reject unsupported bases with explicit errors
- build the evaluation shape from the polynomial degree
- check budget from `x.meta()` and destination fitting from `res.max_k()`
- encode coefficients on the host
- transfer or allocate plaintexts as required by the target backend
- populate a power basis
- call `ckks_eval_poly_const_coeffs`

Because coefficient encoding currently uses `Module<HostBytesBackend>`, this
milestone should also define the host-to-backend transfer story. If plaintexts
must remain host-owned for all backends, document that constraint. If backend
owned plaintexts are needed, add a conversion helper.

Acceptance criteria:

- A user can evaluate a monomial polynomial from `Polynomial { coeffs }` with
  one call.
- The pre-encoded path remains available and is used internally.

### Milestone 7: Chebyshev Support

Goal: add Chebyshev basis support only after monomial evaluation is stable.

Required pieces:

- Chebyshev `PowerBasis::gen_power`:

```text
T_(a+b) = 2*T_a*T_b - T_|a-b|
T_0 = 1
```

- Chebyshev factorization:
  - quotient coefficients above the split are doubled
  - mirrored remainder coefficients are subtracted
  - basis metadata is preserved
- Input-domain change of basis:

```text
x' = 2/(b-a) * x + (-a-b)/(b-a)
```

The change of basis can be a separate affine pre-processing call rather than
hidden inside polynomial evaluation.

Acceptance criteria:

- Cleartext Chebyshev evaluation matches encrypted evaluation for low degrees.
- Tests cover `T_2`, `T_3`, and a mixed Chebyshev polynomial requiring the
  subtraction term.

## Testing Matrix

Add tests in this order:

1. `split_degree` and `optimal_split` boundary tests.
2. `Polynomial::evaluate` cleartext monomial and Chebyshev tests.
3. `PowerBasis` decrypted power tests for `X^2..X^8`.
4. Low-level pre-encoded evaluation for degrees `1`, `2`, `3`, `4`, `7`, `8`,
   and `15`.
5. Insufficient-budget errors.
6. Destination-capacity behavior.
7. High-level `ckks_eval_poly` monomial test.
8. Chebyshev power and evaluation tests.
9. Host construction plus device upload of `BSGSPolynomial<C>`.
10. Reuse test: evaluate two polynomials from the same populated power basis.

Run commands:

```text
cargo check -p poulpy-ckks --lib
cargo test -p poulpy-ckks polynomial_evaluation
cargo test -p poulpy-cpu-ref
```

If AVX support is in scope for the branch:

```text
cargo test -p poulpy-cpu-avx
```

## Risk Register

- The current pre-encoded representation may choose a coefficient `CKKSMeta`
  that is too aggressive for deeper polynomials. The direct metadata policy
  and budget tests are the mitigation.
- Unsafe type erasure in polynomial traits can become public API debt quickly.
  Remove or isolate it before stabilizing the API.
- Chebyshev support is easy to get subtly wrong because both power generation
  and factorization differ from monomial basis.
- `ckks_add_assign_unnormalized` is valuable for baby-step accumulation, but
  it needs an accumulation bound to avoid digit overflow.
- Host plaintext encoding may not be enough for all backends. Decide the
  host/backend plaintext ownership story before adding the high-level API.

## Recommended First PR

The best first PR should be deliberately narrow:

1. Keep monomial-only support.
2. Fix terminology and boundary behavior around `bit_length`.
3. Add degree-boundary tests.
4. Add power-basis tests.
5. Make the low-level pre-encoded path pass reliably.

That PR gives us a stable spine. The direct metadata policy and safer public
API can then be introduced without debugging Chebyshev at the same time.
