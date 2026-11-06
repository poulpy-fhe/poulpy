# poulpy-ckks review rules

## 1. Layering, OEP families and macros

- **Reference family shape.** A family's reference is a trait of default-bodied methods over an empty `Module<BE>` impl.
- **Bootstrapping splits by step.** The bootstrapping family keeps one API, delegate and OEP file, and `layouts/bootstrapping/` and `reference/bootstrapping/` hold a file per step or concept, a reference step being its own trait of that shape.
- **Registration follows the ring.** A family serving both rings registers through the one both-ring umbrella, a standard-only family is listed by hand on each standard backend type, and no second umbrella is added.
- **`CKKSImpl` stays scalar-independent.** It takes no scalar-generic, fold, bootstrapping or ModUp seam, and stays the crate's only blanket `*Impl` impl.
- **Scalar-generic seams.** Anything generic over the encoding scalar `F` is its own OEP seam, never a parameter of a scalar-independent family.
- **Core through its public seams.** A CKKS reference reaches core through core's API and hooks, and a missing seam is added in core.
- **Containers hold data, modules compute.** Computations are `Module<BE>` traits, and container methods only manage data or are host-named codecs.

## 2. Naming, signatures and argument order

- **Operand names.** A new operation names its source `a`, binary operands `a` and `b`, plaintext `pt`, keys `keys` and context `ctx`.
- **Coefficient selectors.** A coefficient selector comes right after its operand.
- **Generic operands.** An operation bounds its operands by the infos and backend-view traits, never a concrete CKKS container.

## 3. Errors and validation order

- **Typed refusals.** A caller-fixable condition returns a `CKKSCompositionError`, `InvalidOperand` unless a more specific variant applies, and `InvalidPlan` only for a plan literal.
- **Adding a variant.** A new variant serves a condition callers branch on that `InvalidOperand` cannot, and carries `op` or `plan`, the explaining numbers and its source error.
- **The op label.** The `op` of an error is the invoked public method, or its stem when one body serves both forms, and a shared helper takes the caller's `op`.
- **Setup errors keep the chain.** Setup adds context to a typed error, never formatting it into a string.
- **Validate, stamp, then work.** A fallible body checks every term, lane, output and worker, then stamps, then computes.
- **Plans validate once.** A setup type with invariants validates at construction or compile, makes invalid shapes unrepresentable, and its evaluators re-check only what depends on runtime operands.
- **Plans carry metadata, not layouts.** A plan literal carries coefficient metadata, never a layout, and an encoder allocating from one checks the width first.

## 4. Scratch, allocation and prepared operands

- **Queries read allocations.** `k` is metadata below the allocation, so a query sizes a nested call on the caller's destination at its allocated width and a temporary at its carve width.
- **Staged compilation.** A context whose preparation budget depends on generated material can compile and prepare in two stages, beside a one-shot form covering both budgets.

## 5. Ring, slots and keys

- **Standard-only by need.** A family or composition needing complex slots or standard-ring automorphisms is standard-only.
- **Ring-shared code asks the ring.** Slot counts and Galois elements come from the ring's queries, never a standard-ring formula.
- **Identity rotations are keyless.** A rotation by Galois element 1, baby and giant step 0 included, is a copy or direct fold that looks up no key, and key sets omit element 1.
- **Keys at the decomposed width.** Keys are sized and pre-checked at the width the switch decomposes, the same one they are looked up at.
- **Key sets from the schedule.** A key-consuming pipeline derives its Galois elements from the schedule it evaluates, never a hand-kept list.
- **Keys checked on both views.** A pipeline validates keys before any work on both the reported infos and the backend view.

## 6. Metadata and stamping

- **Every operation stamps everything.** Every operation, overrides included, stamps `k`, base2k, rank, scale, slot kind and sparsity, and `log_budget` is always derived as `k - log_delta`, never stored.
- **No rescale operation.** None exists or is added: products rescale through `cnv_offset`, chained transforms carry none between factors, results land at `res.k()` and not at the limb-aligned allocation, and only a division by a power of two moves budget bits under `log_delta`.
- **Stamp before the rounding write.** Offsets are computed against the pre-stamp `res`, and the final metadata is stamped before any step rounds into `res`, an in-place product reading `res` stamping once it has read it.
- **One formula per accounting rule.** Offsets, budgets and `cnv_offset` each come from one shared helper, fused paths included, never re-derived at a call site.
- **A lift spends budget.** A lift at a fixed modulus changes the scale in the metadata, never through a setter that preserves the budget, which grows `k` and cancels the shift.
- **Slot kinds.** Sums, products and linear transformations stamp the operands' `join`, multiplying by `i` or adding a constant at a nonzero coefficient stamps `Complex`, `Real` or a `meet` needs an identity proving it, and an operation that can break integrality refuses `Integer` input.
- **Sparsity composition.** `log_sparsity` is the operands' minimum, an operation moving coefficient positions sets it explicitly, and merging distinct inputs clears it.
- **Encryption stamps, decryption does not.** Encryption stamps the plaintext's scale, slot kind and sparsity, and decryption writes coefficients only, into a smaller degree only when the ciphertext's sparsity allows it.
- **Copy CKKS values with `ckks_copy`.** A core copy into a CKKS destination serves only one already carrying the intended label or relabelled right after.
- **Document the metadata rule.** A stamping API documents under `# Metadata` its output formulas per variant family, with slot kind, sparsity, flag effect and preconditions, or links the plan that decides them.

## 7. Laziness and normalization

- **Lazy linear operations.** Additions, subtractions and doubling write raw limb sums and clear the flag, digit-preserving operations keep it, products, rotations, conjugation, key switches and decryption output canonical values, and a linear operation normalizes only through an aligning shift.
- **Raw limb shortcuts.** A limb-wise operation skips its normalizing shift only when the metadata proves the limbs line up.
- **Long sums.** A CKKS sum never rejects a term count for its length.

## 8. Products and DFT economy

- **Stages at their narrowest width.** A pipeline runs each stage at the width the remaining circuit needs.

## 9. Plaintexts and encoding

- **Plaintext operand checks.** A plaintext operand is checked for radix, degree embedding or coefficient range, and alignment before any write.
- **One value per coefficient.** Encoders and decoders take exactly one value per coefficient or slot, refuse any other count, and never spread values at a gap.
- **Quantization rejects before writing.** Every encoding path, overrides included, quantizes through the shared codec, never a cast, rejects a non-finite or out-of-range value before any write, and keeps the preset metadata, narrowing only the slot kind.
- **Resident encoding signatures.** Encoding API and OEP signatures take only backend-resident buffers, host slices entering only through host-named adapters.

## 10. Numerics and determinism

- **Setup math through `CKKSFloat`.** Setup values are computed in the working scalar through `CKKSFloat`, never platform math, irrational constants never pass through `f64`, and a wider intermediate narrows to `F` mantissa-exactly.
- **An optional math backing changes no bit.** `libquadmath` stays optional and allowlisted, changing no stored bit, codec or type.
- **Byte-identical encoding.** Encoding, decoding and bootstrapping setup constants are byte-identical on every backend and platform at each precision in both rings (`docs/review/backends.md` section 5.9).
- **The oracle derives roots independently.** A change to the roots of unity, the slot-transform butterflies or their rounding is matched in the oracle, neither side adjusted to the other's values.
- **PaCo and SHIP scalars.** They are never extended to `f32`, and a radix or modulus past the mantissa is refused.

## 11. Linear transformations and diagonals

- **Scale-agnostic engines live in core.** CKKS linear-transformation and polynomial-evaluation code holds only scale, budget and `cnv_offset` accounting, encoding and the Galois-to-key map.
- **A diagonal stores one period.** A diagonal stores its shortest predictable period (`docs/review/core.md` section 8.4), and encoders are sized from that period, never a ciphertext layout.
- **Uniform diagonals.** A transformation's diagonals share one scale, width, slot kind and degree, and every consumer of a caller-built transformation checks this in release.
- **Summed transforms check every term.** Every term is checked against the first, baby-step caches and keys included.
- **One DFT generator.** Every homomorphic DFT factor comes from one generator under the crate's conventions, scheme conventions being post-transforms of its factors, never a private factor stack or FFT.
- **Replays share the generator's geometry.** A value-free replay shares the generator's geometry code.
- **Schedules in evaluation order.** A DFT schedule lists factors in evaluation order in both directions, never reordered by kind.
- **Direction and format are types.** Each evaluation entry takes factors of its exact direction and format, and only the raw chain and its query stay free in the format. Evaluating a `Split` chain unsplit enters at `Standard` only through a run or a reference, and leaves half a `Standard` encode, which a later split or the composer doubles.

## 12. Bootstrapping, PaCo and SHIP

- **One API per step.** A bootstrap flavor is a plan setting, never a method: no `_real` or `_ci` operation. The plan decides how a step runs.
- **Steps defer to the plan.** A step's doc says what it does according to the plan and links it. Routes and what each parameter does live once, on the plan's types, and a new parameter documents how it behaves and interacts with the others there.
- **A composition stamps nothing.** The labels of a bootstrap reach its steps through the run, and a step that needs one more label reads it there, never from the composer.
- **Bootstrap entry checks.** A bootstrap checks before any work what its steps would refuse, every key's coverage of the run's widths included, and each step checks its own operands and keys before it writes `res`.
- **One key-width model.** Preset key layouts, entry checks and stage widths derive from one width model, which a change of a stage's working width updates.
- **Presets state what was measured.** A public preset states its measured precision rounded down, a test pins its widths, consumed bits and key shapes, and an insecure preset stays out of the public list.
- **Preset key shapes are derived.** Preset key layouts are derived from their use through one policy, each key covering the widest ciphertext it switches, and the digit cap is the only override.
- **Key widths are exact.** The width a plan gives a key equals its widest lookup in a run, which the bootstrap tests assert, so a route changing where a key switches updates the plan's offsets with it.
- **Preset modulus bounds.** Every preset declares a cited modulus bound per secret, checked against the bootstrap width and every key's modulus.
- **PaCo slot order.** The default slot order stays stable for persisted keys.
- **Key-bundle checks are structural.** PaCo and SHIP key-bundle constructors and preflight validate structure only.

## 13. Polynomial evaluation and approximation

- **Depth from core's model.** Degree and depth selection uses core's model, never a local formula.
- **`consumed_bits` is the budget prediction.** Every compiled plan or approximation exposes `consumed_bits()` from its evaluation's accounting, the evaluation checks it before any write, and a budget change updates it and its pinned tests.
- **Shared scratch bound.** Polynomial evaluation has no query of its own, so anything new inside it updates the shared all-operations bound.
- **Approximation entry points.** A host fitting or selection entry point validates every scalar argument first and documents its callbacks as pure.

## 14. Containers, secrets and serialization

- **`CKKSInfos` only on CKKS containers.** It is never implemented on a core type.
- **Setup-created secrets are encapsulated.** A secret that setup creates is drawn from a caller-passed source and tagged as encapsulated, and an ephemeral one never leaves its call.
- **Ciphertexts serialize as GLWE.** `CKKSCiphertext` serializes as its GLWE with no CKKS header, and key bundles store core key types.

## 15. Tests and registration

### 15.1. Registration

- **Parity helpers on every backend.** A parity helper joins every backend crate running CKKS parity (`docs/review/backends.md` section 13.2), at each scalar it applies to, on the conjugate-invariant ring unless a bound needs `Ring = Standard`.
- **Rank-2 coverage.** A rank-generic operation has a test and a parity helper at rank 2, reading the rank from the parameters.
- **Shared helpers stay ring-generic.** Shared test helpers bound nothing that excludes a ring.

### 15.2. Parity

- **Parity sweeps the metadata.** A driver sweeps each metadata dimension its operation reads or stamps: narrower and wider destinations, off-grid widths, sparsity, every stamped slot kind, unequal scales, smaller-degree plaintexts, rank, and, through a DFT, lazy and canonical operands.
- **Encoding parity is exact.** Encodings compare byte for byte, sign of zero included, never within a tolerance.

### 15.3. Functional tests

- **Expected values.** Exact operations compare against expected values from the quantized inputs, never an absolute tolerance, and approximating pipelines use precision statistics.
- **Floors document their measurement.** A precision or noise floor states its measurement or model and margin, and moves only with a new measurement.
- **Output metadata from the inputs.** Tests assert output metadata against values computed from the inputs before the call, and leveled pipelines their exact budget use against `consumed_bits()`.
- **Arithmetic case matrix.** A leveled operation tests aligned operands, a narrower off-grid output, the in-place form, unequal scales and budgets a limb apart, plus one case per typed refusal.
- **Key widths are pinned.** A diff touching key lookup or width selection pins the looked-up width.
- **Transforms against independent references.** A transform is tested in each direction against a reference independent of production code, not only by a round trip.

### 15.4. Fixtures

- **Determinism fixture keys.** A fixture key names only ring, precision, degree and case, never a backend, and the fixture file holds exactly the keys the drivers produce.
- **Fixture inputs are exact.** Fixture values come from platform-independent inputs, never platform math.
