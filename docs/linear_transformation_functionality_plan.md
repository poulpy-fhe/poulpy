# Linear Transformation Missing Functionality And Optimization Roadmap

## Summary

This roadmap breaks the linear transformation work into independent implementation
plans. Each plan records the current status, the desired output status and shape,
the implementation approach, and a checklist.

Prepared BSGS is the promoted evaluation path. Borrowed one-shot APIs remain as
convenience wrappers that prepare temporary borrowed transforms internally.

## 1. Prepared Linear Transformation Representation

### Current Status

Poulpy stores BSGS transforms as `baby_steps` and `giant_steps` with plaintext
diagonals. Diagonals are consumed directly by the evaluator and are not prepared
once for repeated evaluation.

### Desired Output Status And Shape

Add a core-side prepared representation that stores precomputed BSGS groups and
diagonals as right convolution operands.

The prepared form should:

- live in `poulpy-core`;
- store pruned baby steps and non-empty giant steps;
- store diagonals as prepared `CnvPVecR` operands;
- expose the same required-rotation semantics as the unprepared transform;
- keep the current `GLWELinearTransform<P>` as the source of the original
  plaintext diagonals and scale metadata.

### Implementation Plan

Add `GLWEPreparedLinearTransform<BE>` and prepared giant-step structs in
`poulpy-core`. Each prepared giant step stores a map from real baby rotation to
prepared `CnvPVecR`. Implement conversion from `GLWELinearTransform<P>` by
preparing each diagonal with `cnv_prepare_right`, pruning unused baby steps and
empty giant steps during construction. Preparation fills a caller-owned
`&mut GLWEPreparedLinearTransform<BE>` cache; the cache does not store original
plaintexts.

### Checklist

- [x] Define prepared transform structs.
- [x] Add diagonal preparation with `cnv_prepare_right`.
- [x] Add `required_rotations()` on prepared transforms.
- [x] Prune unused baby and empty giant steps.
- [x] Add tests comparing unprepared and prepared metadata.

## 2. BSGS Indexing And Strategy Selection

### Current Status

Callers must provide BSGS layout manually. Poulpy does not yet expose Lattigo-style
indexing or automatic giant-step selection.

### Desired Output Status And Shape

Add reusable BSGS indexing helpers and an explicit strategy selector.

The strategy interface should expose:

```rust
pub enum LinearTransformationStrategy {
    Auto,
    Bsgs { giant_step: usize },
    Direct,
}
```

The BSGS helpers should normalize diagonal indexes modulo the slot count and return
stable, sorted baby and giant rotation lists.

### Implementation Plan

Implement `linear_transformation_schedule(non_zero_diags, slots, giant_step)` and an optimal
giant-step helper compatible with Lattigo's rule: minimize
`(N1 + N2) + abs(N1 - N2)`. Add helpers to derive required rotations from the
index map and strategy.

### Checklist

- [x] Add strategy enum.
- [x] Add BSGS index helper.
- [x] Add optimal giant-step helper.
- [x] Add required rotation helper from indexes.
- [x] Test sparse, dense, wrapped, and single-diagonal inputs.

## 3. Hoisted Baby-Step Rotations

### Current Status

Baby rotations are full `glwe_automorphism` calls. Each rotation recomputes
decomposition and DFT work.

### Desired Output Status And Shape

Compute the input mask DFT once and reuse it for every baby-step key switch.
Materialize each needed baby rotation and prepare it as a left convolution
operand.

### Implementation Plan

Factor a core helper that computes mask columns with `vec_znx_dft_apply`. For each
non-zero baby rotation, call `gglwe_product_dft_default`, IDFT the result, add the
input body, normalize to SMALL, apply the automorphism, and prepare the result
with `cnv_prepare_left`. Treat baby rotation `0` as an identity and prepare it
without key switching.

### Checklist

- [x] Add hoisted mask DFT helper.
- [x] Add baby rotation materialization helper.
- [x] Prepare each baby rotation as `CnvPVecL`.
- [x] Skip unused babies.
- [x] Parity-test each baby rotation against `glwe_automorphism`.

## 4. DFT-Domain Inner Product Accumulation

### Current Status

Each plaintext product uses `glwe_mul_plain`, which normalizes per product.

### Desired Output Status And Shape

For each giant step, accumulate all baby/plaintext products in DFT form and IDFT
once per ciphertext column.

### Implementation Plan

Use prepared baby rotations as `CnvPVecL` and prepared diagonals as `CnvPVecR`.
For each ciphertext column, call `cnv_apply_dft` into a DFT accumulator for the
first term. For subsequent terms, convolve into a DFT temp and add with
`vec_znx_dft_add_assign`. Convert each accumulated column to BIG with
`vec_znx_idft_apply`.

### Checklist

- [x] Add per-giant DFT accumulator.
- [x] Add convolution temp for non-first terms.
- [x] IDFT once per column per giant step.
- [x] Handle one-diagonal and empty-step cases.
- [x] Parity-test inner products against baseline sums.

## 5. Lazy Giant-Step Rotation

### Current Status

Giant steps use full `glwe_automorphism_assign`, which normalizes each giant
accumulator.

### Desired Output Status And Shape

Rotate giant accumulators while keeping the result in BIG form until final output
normalization.

### Implementation Plan

Add a deferred-normalization automorphism helper in `poulpy-core`. Normalize only
mask columns needed for gadget decomposition. Apply `gglwe_product_dft_default`,
IDFT into BIG columns, add the carried BIG body, apply
`vec_znx_big_automorphism_assign`, then add the rotated BIG result into the final
BIG accumulator. For giant rotation `0`, add the BIG product directly.

### Checklist

- [x] Add deferred giant automorphism helper.
- [x] Share scratch layout with existing automorphism/key-switching code.
- [x] Skip key lookup for giant rotation `0`.
- [x] Add parity test against normalized `glwe_automorphism`.
- [x] Add headroom stress test for BIG accumulation.

## 6. Final Lazy Normalization

### Current Status

Normalization happens inside GLWE operations throughout evaluation.

### Desired Output Status And Shape

Keep final output accumulation in BIG columns and normalize once into the output
ciphertext.

### Implementation Plan

Add a final BIG accumulator for all giant-step outputs. At the end of evaluation,
apply `vec_znx_big_normalize` once per output column. Use the CKKS-derived
`cnv_offset`, `a_effective_k`, and output base2k metadata consistently when
normalizing into the destination.

### Checklist

- [x] Add final BIG accumulator.
- [x] Add final normalization pass.
- [x] Validate output metadata in CKKS wrapper.
- [x] Compare optimized result to baseline.
- [x] Add precision regression tests for FFT64 and NTT120.

## 7. Prepared CKKS API

### Current Status

CKKS exposes single-transform evaluation and assign. It computes scale metadata
but has no prepared transform API.

### Desired Output Status And Shape

Add prepared APIs while keeping borrowed one-shot APIs available.

The API should include:

- `ckks_prepare_linear_transformation`;
- `ckks_eval_prepared_linear_transformation_into`;
- `ckks_eval_prepared_linear_transformation_assign`.

Current unprepared evaluation APIs should remain available and delegate through temporary
preparation.

### Implementation Plan

Add prepared CKKS traits and default implementations. The preparation function
prepares diagonals through the core transform preparation path into a caller-owned
cache. Evaluation takes both the original transform and the prepared cache, so
CKKS scale metadata remains tied to the original diagonal plaintexts. The existing
unprepared evaluation methods prepare a temporary cache and call the prepared
evaluator.

### Checklist

- [x] Add prepared CKKS API traits.
- [x] Add default implementations.
- [x] Preserve current public methods.
- [x] Add missing-key validation on prepared transforms.
- [x] Add tests for one-shot and prepared API equivalence.

## 8. Multi-Transform Evaluation

### Current Status

Each transform evaluation recomputes baby rotations.

### Desired Output Status And Shape

Evaluate several transforms on the same input while sharing hoisted baby rotations.

### Implementation Plan

Add `ckks_eval_many_prepared_linear_transformations_into`. Build the union of baby
rotations required by all transforms, hoist the input once, materialize and prepare
the shared baby rotation cache, then evaluate each prepared transform against the
cache. Validate output slice sizing and compatibility of levels, base2k, rank, and
scale assumptions before evaluation.

### Checklist

- [x] Add many-evaluation API.
- [x] Union required baby rotations.
- [x] Share hoisted baby cache.
- [x] Validate output slice sizing.
- [x] Test two transforms against separate baseline calls.

## 9. Sequential Evaluation

### Current Status

Poulpy has no Lattigo-style sequential transform evaluator.

### Desired Output Status And Shape

Evaluate `M_n(...M_1(M_0(ct)))` with the normal per-step CKKS linear-transform
scale handling.

### Implementation Plan

Add `ckks_eval_sequential_prepared_linear_transformations_into`. Evaluate the first
transform into the output. For each following transform, evaluate the next
transform through a temporary and copy the result back. Reuse the existing
`ct x pt`/linear-transform `cnv_offset` metadata path rather than introducing a
new scaling policy.

### Checklist

- [x] Add sequential API.
- [x] Add temporary allocation strategy.
- [x] Reuse per-step CKKS `cnv_offset`/scale handling.
- [x] Preserve metadata after each step.
- [x] Test against manual step-by-step evaluation.

## 10. Direct/Naive Diagonal Path

### Current Status

Poulpy has only BSGS-shaped evaluation.

### Desired Output Status And Shape

Add a direct path for transforms with very few diagonals.

### Implementation Plan

Implement after optimized BSGS is stable. Use single hoisting for all non-zero
diagonal rotations, but avoid baby/giant grouping. Let `LinearTransformationStrategy::Auto`
choose the direct path for small diagonal counts once benchmarks identify the
threshold.

### Checklist

- [x] Add direct prepared transform path.
- [x] Add direct evaluator.
- [x] Add auto strategy threshold.
- [x] Test direct vs BSGS vs baseline.
- [x] Benchmark sparse transforms.

## 11. Benchmarks And Promotion

### Current Status

Correctness tests and focused CKKS linear-transformation benchmarks exist. The
benchmark harness covers sparse, medium, dense, interleaved, direct, one-shot,
prepared, and many-prepared shapes, and prints schedule/scratch metadata next to the
Criterion timing output.

### Desired Output Status And Shape

Prepared APIs are the optimized default path for repeated linear transforms. The
borrowed one-shot API remains available for convenience and prepares a temporary
cache from the borrowed transform internally.

### Implementation Plan

Add benchmark cases for sparse, medium, dense, and interleaved transforms.
Measure one-shot preparation, prepared BSGS, optimized BSGS, direct path, and
multi-transform sharing. Track rotation count, runtime, scratch usage, and
backend-specific precision behavior. Document the prepared API as the promoted
reusable path.

### Checklist

- [x] Add benchmark fixtures.
- [x] Measure rotation count and runtime.
- [x] Compare memory/scratch usage.
- [x] Document backend-specific precision behavior.
- [x] Promote prepared APIs as the optimized path while keeping borrowed one-shot APIs.

## Assumptions

- One-shot and prepared evaluators remain covered by parity tests.
- Optimized implementation belongs in `poulpy-core`.
- CKKS APIs remain backward compatible.
- Prepared APIs are additive.
- BIG accumulator headroom must be validated before making lazy normalization unconditional.
