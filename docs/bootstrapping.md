# Bootstrapping in Poulpy

This document describes how Poulpy refreshes the homomorphic budget of a CKKS ciphertext.
It covers the four-stage pipeline, the modulus-raise mechanism, the scale and budget accounting through each stage, the standard and EvalRound+ variants, the parameters and keys, and where to find the code.

## Overview

Given a ciphertext whose homomorphic budget has been consumed down to a small modulus, bootstrapping produces a ciphertext encrypting the same plaintext at a larger modulus, regaining budget.

The usual pipeline is

```text
ModUp ─► CoeffsToSlots ─► EvalMod ─► SlotsToCoeffs
```

ModUp is the modulus raise, provided by the bootstrapping trait (`CKKSBootstrappingOps`).
CoeffsToSlots and SlotsToCoeffs are the homomorphic DFT (`DFTOps`), a chain of linear transformations over the slots (see [linear_transformation.md](linear_transformation.md)); EvalMod is homomorphic `x mod 1`, a polynomial evaluation (`CKKSEvalModOps`, see [polynomial_evaluation.md](polynomial_evaluation.md)).

The engine follows the usual `api` / `oep` / `default` / `delegates` split.
A ready-made orchestrator, `ckks_bootstrap`, runs the whole refresh from a compiled `BootstrappingContext` and a prepared `BootstrappingKeys`.
The individual stages stay public, so a caller can assemble a custom pipeline instead.
The end-to-end test is exactly such a hand-composed reference.

## The integer wrap-around

A CKKS ciphertext at modulus `q` encrypts a message `m` at scale `Δ = 2^log_delta`, so its plaintext polynomial is `Δ·m(X)`.
Decryption recovers that value only modulo `q`: the inner product of the ciphertext with the secret key is `Δ·m(X) + q·I(X)`, where `I(X)` is an unknown integer polynomial, that needs to be removed to preserve the plaintext.

Normalizing by `q` turns `q·I + Δ·m` into `I + Δ·m/q`, and `mod 1` removes the integer `I`, leaving `Δ·m/q`; scaling back by `q` returns `Δ·m` at the wide modulus.
That homomorphic `x mod 1` is what EvalMod approximates.

The wrap-around `q·I(X)` lives in the **coefficients** of the plaintext, while EvalMod evaluates a polynomial on the **slots**.
The homomorphic DFT bridges the two: CoeffsToSlots moves the coefficients into the slots so the reduction can be applied slot-wise, and SlotsToCoeffs moves the cleaned values back.

## ModUp

ModUp raises the ciphertext modulus from `q = 2^k_small` to the wider bootstrap modulus `2^k_large`.
In the base-`2^K` representation this is a digit shift with no arithmetic: it MSB-aligns the source into the wide ciphertext (`glwe_copy`, leaving the new low-order limbs zero), then shifts the digits down to their natural integer magnitude (`glwe_rsh` by `k_large − k_small`).
The raised-from modulus `q` becomes an explicit, un-reduced multiple `I(X)·q` in the `[0, 2^k_large)` window, which EvalMod later removes.
The encoding scale `log_delta` is unchanged: the headroom now spans the full raised modulus, so `log_budget = k_large − log_delta`.

Right after ModUp the ciphertext is relabeled at the input-modulus scale, a free division by the message ratio: setting `log_delta := log_modulus_in` reinterprets `q·I + Δ·m` as `I + m·Δ/q`, separating the integer part `I` from the residue.
The message ratio is `q/Δ = 2^log_msg_ratio`, the bit gap between the payload and the integer part.

### Sparse-secret encapsulation

The size of the wrap-around `I` is governed by the Hamming weight of the secret key, so under a dense key it can be large, forcing EvalMod to cover a wide interval.
Sparse-secret encapsulation bounds it instead by the weight of a small *ephemeral* sparse secret: the pipeline key-switches the ciphertext from the dense secret to the sparse one *before* ModUp and back *after*,

```text
denseToSparse ─► ModUp ─► sparseToDense
```

so `I·q` stays small and EvalMod can use a much smaller interval `K` with negligible failure probability ([eprint 2022/024](https://eprint.iacr.org/2022/024)).
It is enabled when `ephemeral_secret_weight > 0`.
Its two key-switching keys are derived from the dense secret and so are part of the keys, not of the secret-independent `BootstrappingContext`.

## The homomorphic DFT (CoeffsToSlots / SlotsToCoeffs)

CKKS packs `m` complex slots into the `n` coefficients of a plaintext polynomial through a negacyclic DFT.
CoeffsToSlots is that map applied homomorphically (a homomorphic encoding, the IDFT) and SlotsToCoeffs is its inverse (a homomorphic decoding, the DFT).
Each is a linear map over the slots, factored — as an FFT is — into a chain of sparse factor matrices, every factor evaluated as one matrix-vector product over the slots via the baby-step/giant-step diagonal method of [linear_transformation.md](linear_transformation.md).
The factorization schedule is caller-chosen: each entry is one factor matrix and gives how many radix-2 layers it merges, so their sum is `log_slots`.

Bootstrapping uses the **split real/imaginary** format.
CoeffsToSlots returns the real and imaginary coefficient halves as two separate real-slot ciphertexts, so EvalMod can reduce each one independently, and SlotsToCoeffs recombines them.
The split forward transform needs a conjugation key — the automorphism for Galois element `−1` — in addition to the rotation keys, to separate the two halves.

Scale accounting is implicit.
Poulpy's torus plaintext-multiply already realigns its result to the input `log_delta` through its `cnv_offset`, so the rescale is folded into each linear-transform evaluation: the transform is simply one prepared linear transformation per factor, chained, with no explicit rescale between factors.
Each factor consumes its per-factor `log_delta` of budget, so the whole transform consumes `num_factors × factor_log_delta` bits.
Two constant scalings ride along the matrices for free: CoeffsToSlots is pre-scaled by `1/K` to land in EvalMod's `[−1, 1]` domain, and SlotsToCoeffs scales up by `2^log_msg_ratio` to restore the message scale.

## EvalMod

EvalMod is the homomorphic `x mod 1`, the pipeline's only non-linear stage.
No low-degree polynomial computes `mod`, so EvalMod approximates it with a **periodic** function `f` whose period matches `q`: periodicity collapses every `I·q`, so `f(I·q + Δ·m)` depends on `m` alone.
Because `f` is only locally linear in `m`, it can be post-composed with its inverse `f⁻¹` (the arcsine for the trigonometric families) to recover a value linear in `m` across the whole interval.

Four approximation families are available, selected by `EvalModType`.
`CosHK` is a discrete cosine fit (the Han & Ki method) that is best for a small interval `K`; `SinCheby` and `CosCheby` are continuous Chebyshev fits, with `CosCheby` overtaking `CosHK` as `K` grows; `ExpCmplx` is the complex exponential.

The evaluation has up to three sub-stages.

1. **Base polynomial.** A baby-step/giant-step evaluation (see [polynomial_evaluation.md](polynomial_evaluation.md)) of `f` over the reduced range `[−K/2^r, K/2^r]`, where `K = f_mod_interval` and `r = f_mod_log_interval_reduction`.
2. **Range extension.** `r` doubling steps (`cos 2θ = 2cos²θ − 1` for the cosine families, `exp 2θ = (exp θ)²` for the complex one) extend the reduced range back to `[−K, K]`, trading multiplicative depth for a cheaper low-degree base polynomial. These squarings use the tensor (relinearization) key.
3. **Inverse.** An optional `f⁻¹` post-composition that linearizes the result in `m`.

EvalMod runs at its own scale.
`ckks_eval_mod` raises the ciphertext to `f_mod_log_delta` on entry and restores the input scale on exit — a budget-neutral reinterpretation — so its `ct × ct` chain keeps more precision than the message scale alone would allow.

## Scale and budget chain

A ciphertext enters at the input modulus `2^log_modulus_in` carrying `Δ·m` with `log_msg_ratio` bits of headroom, and leaves at the bootstrap modulus `2^k_boot` carrying the same `Δ·m` with a much larger budget.
The budget is tracked exactly, stage by stage; the table below mirrors the per-stage assertions in the end-to-end test.

| After stage | `log_budget` |
| --- | --- |
| Input | `log_msg_ratio` |
| ModUp | `k_boot − log_modulus_in` |
| CoeffsToSlots | previous `−` `coeffs_to_slots.consumed_bits()` |
| EvalMod | previous `−` `eval_mod.consumed_bits()` |
| SlotsToCoeffs | previous `−` `slots_to_coeffs.consumed_bits()` |

The whole pipeline consumes

```text
consumed_bits = coeffs_to_slots.consumed_bits()
              + eval_mod.consumed_bits()
              + slots_to_coeffs.consumed_bits()
```

and the bootstrap modulus is sized from that, leaving head-room for the refreshed output:

```text
k_boot = (log_modulus_in + consumed_bits + 2·log_delta).next_multiple_of(base2k)
```

EvalMod is charged at the scale it runs (`f_mod_log_delta`), not the message scale; the surrounding set-scale round-trip is budget-neutral and does not enter the total.

## Standard and EvalRound+ pipelines

The orchestrator selects the pipeline from the compiled context.

The **standard** pipeline feeds EvalMod's clean residue straight into SlotsToCoeffs.
It is used when the context carries no bypass transform.

The **EvalRound+** pipeline ([eprint 2024/1379](https://eprint.iacr.org/2024/1379)) runs CoeffsToSlots twice — a low-precision transform that feeds EvalMod, and a high-precision "bypass" transform — and combines them as

```text
r1 = r0_hp − K·r0_lp + EvalMod(r0_lp)
```

The low-precision DFT error `e` cancels: the high-precision branch holds `Δm + I·q`, the scaled low-precision branch holds `Δm + I·q + e`, and EvalMod yields `Δm + e`, so the integer part and `e` both annihilate and leave `Δm` at the **high-precision** transform's accuracy.
Because EvalMod only has to resolve the large integer part, halving its CoeffsToSlots precision shrinks the bootstrap modulus by `num_factors × (hp_log_delta − lp_log_delta)` bits, while the high-precision transform runs inside the depth the low-precision path already occupies and so does not enlarge `k_boot`.
This pipeline is used when the context carries a bypass transform.

## Parameters and keys

A `BootstrappingPlan` is the parameter bundle: the ephemeral secret weight, the two homomorphic DFT plans (and an optional bypass), and the EvalMod plan.

```rust
let plan = BootstrappingPlan {
    ephemeral_secret_weight: 32,
    coeffs_to_slots: DFTPlan {
        kind: DFTType::Encode,
        factorization_depth: vec![2, 2, 3, 3],
        giant_steps: vec![4, 4, 4, 4],
        format: DFTOutputFormat::SplitRealAndImag,
        scaling: Some(1.0 / 16.0),          // 1 / K
        bit_reversed: false,
        meta: meta(58, 2),
    },
    coeffs_to_slots_bypass: None,           // Some(..) selects EvalRound+
    eval_mod: EvalModPlan {
        eval_mod_type: EvalModType::CosHK,
        log_msg_ratio: 11,
        f_mod_degree: 30,
        f_mod_interval: 16,                 // K
        f_mod_log_interval_reduction: 3,    // r
        f_mod_inv_degree: None,
        scaling: None,
        split_strategy: SplitStrategy::MinDepth,
        coeffs_meta: meta(48, 4),
        f_mod_log_delta: 60,
    },
    slots_to_coeffs: DFTPlan {
        kind: DFTType::Decode,
        factorization_depth: vec![3, 3, 2, 2],
        giant_steps: vec![4, 4, 4, 4],
        format: DFTOutputFormat::SplitRealAndImag,
        scaling: Some((11_f64).exp2()),     // 2^log_msg_ratio
        bit_reversed: false,
        meta: meta(39, 2),
    },
};
```

`BootstrappingContext::compile` turns the plan into the resident, secret-independent form: the prepared backend-resident DFT matrices (with the `1/K` and message-ratio scalings baked in) and the encoded, uploaded EvalMod.
It is built once and reused across bootstraps.

The keys are generated by `generate_keys`, which returns the unprepared `BootstrappingKeySet` — the serializable, GPU-resident form — and a `prepare` step preprocesses the whole set for evaluation.
Four key roles are used:

| Key | Role |
| --- | --- |
| `rotation_keys` | Automorphism keys for the DFT rotations, read off the compiled matrices |
| `conjugation_key` | The Galois `−1` automorphism, for the split real/imaginary transform |
| `tensor_key` | Relinearization key for the EvalMod range-extension squarings |
| `encapsulation_keys` | Optional `denseToSparse` / `sparseToDense` pair for sparse-secret encapsulation |

## Cost and where to look

The pipeline costs the modulus of the three sub-circuits — `consumed_bits` above — and a handful of key-switches per DFT factor plus the EvalMod multiplications; the bootstrap modulus `k_boot` follows from the sizing formula.
A small self-contained parameter set (ring degree `n = 2048`, `K = 16`, message ratio `2^11`, `log_delta = 45`, a degree-30 `CosHK` EvalMod) recovers the slots to a few bits of precision on the reference backend, which is the floor the end-to-end test asserts; wider parameters recover proportionally more.

- `poulpy-ckks/src/test_suite/bootstrapping.rs` is the reference composition, with both the standard and the EvalRound+ pipelines spelled out stage by stage and cross-checked against the `ckks_bootstrap` orchestrator.
- `poulpy-cpu-ref/examples/bootstrap_trace.rs` runs the standard pipeline for profiling.

```sh
cargo test -p poulpy-cpu-ref --features enable-ckks --release ntt4x30_f64::bootstrapping_e2e -- --nocapture
```
