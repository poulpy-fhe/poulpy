# Bootstrapping in Poulpy

This document describes how Poulpy refreshes the homomorphic budget of a CKKS ciphertext.
It covers the C2S-first and S2C-first pipelines, the modulus raise, budget accounting, optional techniques, parameters, and keys.

## Overview

Given a ciphertext whose homomorphic budget has been consumed down to a small modulus, bootstrapping produces a ciphertext encrypting the same plaintext at a larger modulus, regaining budget.

The C2S-first pipeline is

```text
ModUp ─► CoeffsToSlots ─► EvalMod ─► SlotsToCoeffs
```

The slim S2C-first pipeline moves SlotsToCoeffs below the modulus raise:

```text
SlotsToCoeffs ─► ModUp ─► CoeffsToSlots ─► EvalMod
```

ModUp is the modulus raise, provided by the bootstrapping trait (`CKKSBootstrappingOps`).
CoeffsToSlots and SlotsToCoeffs are the homomorphic DFT (`CKKSDFTOps`), a chain of linear transformations over the slots (see [linear_transformation.md](linear_transformation.md)); EvalMod is homomorphic `x mod 1`, a polynomial evaluation (`CKKSEvalModOps`, see [polynomial_evaluation.md](polynomial_evaluation.md)).

The engine follows the usual `api` / `oep` / `reference` / `delegates` split.
A ready-made orchestrator, `ckks_bootstrap`, runs the whole refresh from a compiled `BootstrappingContext` and a prepared `BootstrappingKeys`; [a fold](#batches) packs batches for it.
Backends select the pipeline through `CKKSBootstrappingImpl` (`impl_ckks_bootstrapping_reference!`), which also covers ModUp and functional bootstrapping, and `test_bootstrapping_parity` compares an override against a reference backend.
The individual stages stay public, so a caller can assemble a custom pipeline instead.
The end-to-end tests drive the orchestrator through every pipeline.

## The integer wrap-around

A CKKS ciphertext at modulus `q` encrypts a message `m` at scale `Δ = 2^log_delta`, so its plaintext polynomial is `Δ·m(X)`.
Decryption recovers that value only modulo `q`: the inner product of the ciphertext with the secret key is `Δ·m(X) + q·I(X)`, where `I(X)` is an unknown integer polynomial, that needs to be removed to preserve the plaintext.

Normalizing by `q` turns `q·I + Δ·m` into `I + Δ·m/q`, and `mod 1` removes the integer `I`, leaving `Δ·m/q`; scaling back by `q` returns `Δ·m` at the wide modulus.
That homomorphic `x mod 1` is what EvalMod approximates.

The wrap-around `q·I(X)` lives in the **coefficients** of the plaintext, while EvalMod evaluates a polynomial on the **slots**.
The homomorphic DFT bridges the two: CoeffsToSlots moves the coefficients into the slots so the reduction can be applied slot-wise, and SlotsToCoeffs moves the cleaned values back.

## ModUp

ModUp raises the ciphertext modulus from `q = 2^k_small` to the wider bootstrap modulus `2^k_large`.
In the base-`2^K` representation this is a digit shift with no arithmetic: it MSB-aligns the source into the wide ciphertext (`glwe_copy`, leaving the new low-order limbs zero), then shifts the digits down (`glwe_rsh`).
The raised-from modulus `q` becomes an explicit, un-reduced multiple `I(X)·q` in the `[0, 2^k_large)` window, which EvalMod later removes.
The shift stops `scale_up` bits short of the natural magnitude `k_large − k_small`, a lift fused into the raise: the value is multiplied by `2^scale_up` and `log_delta` grows by `scale_up`, so the message lands at the plan's working scale while the encoded value is unchanged.
The headroom now spans the full raised modulus, so `log_budget = k_large − log_delta`.

`ckks_mod_up_into` then relabels the ciphertext by the message ratio, adding `log_msg_ratio` to `log_delta`: this free division reinterprets `q·I + Δ·m` as `I + m·Δ/q`, separating the integer part `I` from the residue.
The message ratio is `q/Δ = 2^log_msg_ratio`, the bit gap between the payload and the integer part.

### Sparse-secret encapsulation

The size of the wrap-around `I` is governed by the Hamming weight of the secret key, so under a dense key it can be large, forcing EvalMod to cover a wide interval.
Sparse-secret encapsulation bounds it instead by the weight of a small *ephemeral* sparse secret: the pipeline key-switches the ciphertext from the dense secret to the sparse one *before* ModUp and back *after*,

```text
denseToSparse ─► ModUp ─► sparseToDense
```

so `I·q` stays small and EvalMod can use a much smaller interval `K` with negligible failure probability ([eprint 2022/024](https://eprint.iacr.org/2022/024)).
The raise and the switch back can be one key switch: a `sparseToDense` key whose rows sit `k_offset` bits lower (`GGLWEInfos::k_offset` in `poulpy-core`) lowers its input by that many bits as it switches it, integer part `I` included.
Its rows then cover only the bits the raised ciphertext fills at the bottom of the raised width (`BootstrappingPlan::raised_window`), not the whole width: at `N = 2^16` the presets' key drops from 7 rows of 208-bit digits (231 MiB) to one row of 104 bits (29 MiB).
A raise whose shift exceeds the key's offset first shifts its input by the difference, inside the key's window, as S2C-first does for its narrower inputs.
It is selected by the bootstrapping recipe through `BootstrappingTechniques::sparse_secret_encapsulation`, whose Hamming weight must be chosen together with EvalMod's interval.
`BootstrappingKeysLayout::encapsulation` contains only the two physical key-switch layouts; key generation rejects its presence or absence when it disagrees with the compiled recipe, and execution likewise rejects a mismatched key set.
The generated keys remain part of the key bundle, while the public Hamming-weight parameter is retained by the otherwise secret-independent `BootstrappingPlan`/`BootstrappingContext`.

## The homomorphic DFT (CoeffsToSlots / SlotsToCoeffs)

CKKS packs `m` complex slots into the `n` coefficients of a plaintext polynomial through a negacyclic DFT.
CoeffsToSlots is that map applied homomorphically (a homomorphic encoding, the IDFT) and SlotsToCoeffs is its inverse (a homomorphic decoding, the DFT).
Each is a linear map over the slots, factored — as an FFT is — into a chain of sparse factor matrices, every factor evaluated as one matrix-vector product over the slots via the baby-step/giant-step diagonal method of [linear_transformation.md](linear_transformation.md).
The factorization schedule is caller-chosen: each entry is one factor matrix and gives how many radix-2 layers it merges, so their sum is `log_slots`.

Bootstrapping uses the **split real/imaginary** format.
CoeffsToSlots returns the real and imaginary coefficient halves as two separate real-slot ciphertexts, so EvalMod can reduce each one independently, and SlotsToCoeffs recombines them.
The split forward transform also applies the automorphism for Galois element `−1` (conjugation) to separate the two halves; key generation includes it among the rotation keys.

Scale accounting is implicit.
Poulpy's torus plaintext-multiply already realigns its result to the input `log_delta` through its `cnv_offset`, so the rescale is folded into each linear-transform evaluation: the transform is simply one prepared linear transformation per factor, chained, with no explicit rescale between factors.
Each factor consumes its per-factor `log_delta` of budget, so the whole transform consumes `num_factors × factor_log_delta` bits.
Two constant scalings ride along the matrices for free. CoeffsToSlots is pre-scaled by `1/K`. C2S-first uses `2^log_msg_ratio` on its final SlotsToCoeffs; S2C-first uses `1/2` on its initial SlotsToCoeffs to cancel the real/imaginary split.

## EvalMod

EvalMod is the homomorphic `x mod 1`, the pipeline's only non-linear stage.
No low-degree polynomial computes `mod`, so EvalMod approximates it with a **periodic** function `f` whose period matches `q`: periodicity collapses every `I·q`, so `f(I·q + Δ·m)` depends on `m` alone.
Because `f` is only locally linear in `m`, it can be post-composed with its inverse `f⁻¹` (the arcsine for the trigonometric families) to recover a value linear in `m` across the whole interval.

Five approximation families are available, selected by `EvalModType`.
`CosHK` is a discrete cosine fit (the Han & Ki method) that is best for a small interval `K`; `CosHKEven` fits it in a centred variable so the base polynomial is even, and is the variant the presets use; `SinCheby` and `CosCheby` are continuous Chebyshev fits, with `CosCheby` overtaking `CosHK` as `K` grows; `ExpCmplx` is the complex exponential.

The evaluation has up to three sub-stages.

1. **Base polynomial.** A baby-step/giant-step evaluation (see [polynomial_evaluation.md](polynomial_evaluation.md)) of `f` over the reduced range `[−K/2^r, K/2^r]`, where `K = f_mod_interval` and `r = f_mod_log_interval_reduction`.
2. **Range extension.** `r` doubling steps (`cos 2θ = 2cos²θ − 1` for the cosine families, `exp 2θ = (exp θ)²` for the complex one) extend the reduced range back to `[−K, K]`, trading multiplicative depth for a cheaper low-degree base polynomial. These squarings use the tensor (relinearization) key.
3. **Inverse.** An optional `f⁻¹` post-composition that linearizes the result in `m`.

EvalMod runs at its own scale.
`ckks_eval_mod` raises the ciphertext to `f_mod_log_delta` on entry and restores the input scale on exit — a budget-neutral reinterpretation — so its `ct × ct` chain keeps more precision than the message scale alone would allow.

## Scale and budget chain

A ciphertext enters at the input modulus `2^log_modulus_in` carrying `Δ·m` with `log_msg_ratio` bits of headroom, and leaves at the output modulus `2^k_out` carrying the same `Δ·m` with a much larger budget. Both pipelines restore the input scale automatically.
For C2S-first, the budget is tracked as follows.

| After stage | `log_budget` |
| --- | --- |
| Input | `log_msg_ratio` |
| ModUp | `k_boot − f_mod_log_delta` |
| CoeffsToSlots | previous `−` `coeffs_to_slots.consumed_bits()` |
| EvalMod | previous `−` `eval_mod.consumed_bits()` |
| SlotsToCoeffs | previous `−` `slots_to_coeffs.consumed_bits()` |
| Return to input scale | unchanged; `k` decreases by `f_mod_log_delta − log_delta` |

The total circuit cost is

```text
consumed_bits = pre_mod_up_consumed_bits()
              + post_mod_up_consumed_bits()
```

S2C-first places SlotsToCoeffs at the bottom of the input modulus. CoeffsToSlots and EvalMod consume the top of the raised modulus. The plan exposes both sides directly:

```text
k_in   = plan.input_k(log_modulus_in)
k_boot = plan.bootstrap_k(k_out, log_delta)
```

`k_out` is the desired output torus width at the input scale `log_delta`, before limb rounding. C2S-first includes the extra working width needed to restore that scale at the end of the pipeline.

EvalMod is charged at the scale it runs (`f_mod_log_delta`), not the message scale; the surrounding set-scale round-trip is budget-neutral and does not enter the total.

## Functional bootstrapping

Functional bootstrapping replaces the final identity refresh with a lookup table.
`EncodedLut::general` interpolates exactly `p` entries with degree `p - 1`, for any table length `p >= 2`.
Inputs contain integer messages `m`; the result is `table[m mod p]`, including for negative messages.
Configure the plan with `plan.with_functional_bootstrap(&lut)?` before compiling the context.
The initial SlotsToCoeffs transform includes the normalization to phase `m/p`.
The input contract is:

```text
ct_in.log_budget() - slots_to_coeffs.consumed_bits() = ceil(log2(p))
```

The output scale is `ct_in.log_delta() + ceil(log2(p))`.
Use `BootstrappingPlan::functional_bootstrap_k` to size the raised ciphertext for a desired output width, then round that width to whole limbs if necessary.
LUTs are encoded on the host and uploaded once with `EncodedLut::transfer_to`.

General LUTs use trigonometric Hermite interpolation on the unit circle.
They therefore require an S2C-first recipe whose EvalMod type is `ExpCmplx` with `scaling = 2π`.
`ckks_functional_bootstrap` takes a slice of LUTs and a slice of outputs, so one LUT and many go through the same entry point: the batch shares the SlotsToCoeffs, ModUp and CoeffsToSlots stages, and equal-arity general LUTs additionally share the power basis of each transformed half (binary or mixed batches fall back to evaluating each LUT against the shared transformed input). Every LUT in a batch must have the same table length. Each imaginary half is folded into its output as it is produced, so the scratch bound does not grow with the batch size.
Real slots are selected by the input's metadata rather than by a separate entry point: with an S2C-first context, `ct.set_slots(SlotsKind::Real)` makes `ckks_functional_bootstrap`, and `ckks_bootstrap` without EvalRound+, skip the imaginary branch.

`EncodedLut::binary` is specialized for two entries.
Its cosine polynomial is controlled by `degree`, `k_interval`, and `log_interval_reduction`; it skips EvalMod and is cheaper than the general construction.
For now it still uses a `BootstrappingPlan` containing an EvalMod plan because EvalMod is not optional in the compiled context, although those tables are not evaluated by the binary path.

### EvalMod on the conjugate-invariant ring

After an unsplit CoeffsToSlots, the `N` coefficients sit in `N/2` complex slots `z_j = c_j + i·c_{j+N/2}`, and EvalMod's real polynomial runs once per half. The conjugate-invariant (CI) ring of degree `N` has `N` real slots, so one EvalMod there reduces them all (`BootstrappingPlan::with_eval_mod_ring(EvalModRing::ConjugateInvariant { .. })`, S2C-first only).
The ciphertext is embedded in the standard ring of degree `2N` as `z(X²)`, which holds each slot twice, multiplied by `u` (`1` on one copy, `−i` on the other) and traced onto the CI ring with a key switch, which leaves `c_j` and `c_{j+N/2}` in two real slots. After EvalMod on the CI backend, the result is embedded back and multiplied by `v` (`2` and `2i`), and its even coefficients, back at degree `N`, pair the halves again as `r_j + i·r_{j+N/2}`. The two products consume the scales of `u` and `v`; the trace and embedding keys are of degree `2N`.
The product by `u` can instead be folded into the last CoeffsToSlots factor (`CIConversion::Folded`). Writing `u = u_e(X²) + X·u_o(X²)`, `u·z(X²) = (u_e·z)(X²) + X·(u_o·z)(X²)`, where `u_e·z` and `u_o·z` are slot-wise products at degree `N`: the last factor runs twice from the same baby steps, scaled by the slots of `u_e`, `(1 − i)/2`, and of `u_o`, `(1 + i)/(2ω_j)` at the root `ω_j` of slot `j`. The fold costs no level, which narrows the raised width and so every key, at the price of the second pass over the factor. `v` does not fold: nothing linear follows EvalMod in the S2C-first pipeline.
`CKKSBootstrappingCIOps::ckks_bootstrap_ci_into` runs it with one standard module of degree `2N`, which serves the conversions and the stages of degree `N`, and a CI module of degree `N` whose backend shares the buffers of the bootstrap's, on a context of `BootstrappingContext::compile_ci` and keys of `generate_keys_ci` and `prepare_ci`: no conjugation key, a tensor key of a CI secret, and the conversion keys, all checked at entry, as are both arenas against `ckks_bootstrap_ci_tmp_bytes`. Backends override it through `CKKSBootstrappingCIImpl`, whose `CI` names the twin.

## Pipeline order and EvalRound+

The orchestrator selects the pipeline from the compiled context.

The **C2S-first** pipeline feeds EvalMod's clean residue into SlotsToCoeffs. Without a bypass transform this is the standard recipe.

The **S2C-first** pipeline splits the input into real and imaginary halves, applies the `1/2`-scaled SlotsToCoeffs at the bottom of the input modulus, and raises the resulting coefficient ciphertext. The standard variant then runs CoeffsToSlots and EvalMod at the top of the raised modulus; its two EvalMod outputs are recombined and relabeled at the original input scale.

The **EvalRound+** pipeline ([eprint 2024/1379](https://eprint.iacr.org/2024/1379)) runs CoeffsToSlots twice — a low-precision transform that feeds EvalMod, and a high-precision "bypass" transform — and combines them as

```text
r1 = r0_hp − K·r0_lp + EvalMod(r0_lp)
```

The low-precision DFT error `e` cancels: the high-precision branch holds `Δm + I·q`, the scaled low-precision branch holds `Δm + I·q + e`, and EvalMod yields `Δm + e`, so the integer part and `e` both annihilate and leave `Δm` at the **high-precision** transform's accuracy.
Because EvalMod only has to resolve the large integer part, halving its CoeffsToSlots precision shrinks the bootstrap modulus by `num_factors × (hp_log_delta − lp_log_delta)` bits. The bootstrap budget charges the larger of the high-precision transform and the low-precision CoeffsToSlots plus EvalMod branch; with the standard parameters, the high-precision transform fits inside the latter and does not enlarge `k_boot`.
EvalRound+ is used when either pipeline carries a bypass transform. With S2C-first, the initial SlotsToCoeffs remains below ModUp and the same LP/HP CoeffsToSlots cancellation runs above it.
With EvalMod on the conjugate-invariant ring, the cancellation runs on EvalMod's own input `x` on that ring, `EvalMod(x) − K·x`, before the conversion back, and the bypass, unsplit, is added at degree `N`: this holds whether `u` is a product or folded into CoeffsToSlots, where `x` comes from the two folded factors. The product by `v` then rounds a value that holds the integer part `I·q`, so its scale must carry about `log_msg_ratio + log2(K)` more bits than without EvalRound+.

## Parameters and keys

A `BootstrappingPlan` is the complete ModUp/EvalMod recipe: its C2S-first or S2C-first pipeline, optional techniques (sparse-secret encapsulation and EvalRound+), the two homomorphic DFT plans, and the EvalMod plan.
The constructors validate once — `DFTPlan::new` checks the factorization schedule (`(depth, giant_step)` pairs in evaluation order), while `BootstrappingPlan::new` checks the selected pipeline, stage directions, sparse weight, and EvalRound+ constraints — so a plan that exists is always shape-valid and the derived-key APIs (`galois_elements`) are infallible. `DFTPlan::with_optimal_bsgs(log_n)` can replace the per-factor widths with the structure-aware, rotation-count-balanced defaults for the target ring.

```rust
// Coefficient meta: CoeffsMeta::from_delta_budget(log_delta, log_budget).
let meta = CoeffsMeta::from_delta_budget;

let coeffs_to_slots = DFTPlan::new(
    DFTType::Encode,
    vec![(2, 1), (2, 1), (3, 1), (3, 1)],   // (depth, giant_step) per factor
    DFTOutputFormat::SplitRealAndImag,
    meta(58, 2),
)?
.with_optimal_bsgs(11)                       // ring degree 2^11
.with_scaling(1.0 / 16.0)?;                 // 1 / K

let slots_to_coeffs = DFTPlan::new(
    DFTType::Decode,
    vec![(3, 1), (3, 1), (2, 1), (2, 1)],
    DFTOutputFormat::SplitRealAndImag,
    meta(39, 2),
)?
.with_optimal_bsgs(11)
.with_scaling((11_f64).exp2())?;            // 2^log_msg_ratio

let plan = BootstrappingPlan::new(
    BootstrappingPipeline::C2SFirst,
    BootstrappingTechniques {
        sparse_secret_encapsulation: Some(SparseSecretEncapsulation {
            hamming_weight: 32,
        }),
        eval_round_plus: None,               // Some(EvalRoundPlus { ... }) selects EvalRound+
    },
    coeffs_to_slots,
    EvalModPlan {
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
    slots_to_coeffs,
)?;
```

`BootstrappingContext::compile` turns the plan into the resident, secret-independent form: the prepared backend-resident DFT matrices and the encoded, uploaded EvalMod.
It is built once and reused across bootstraps.
To size the preparation scratch exactly, compile in two stages: `BootstrappingContext::compile_unprepared` generates the matrices, `prepare_tmp_bytes` reports the backend's preparation budget for them, and `prepare` returns the context.

### Default presets

`poulpy_ckks::presets::bootstrapping` provides complete bundles for applications that do not need to assemble a plan manually.
A preset includes its circuit plan, composable input/output/bootstrap widths, secret weights, ciphertext allocation layouts, and physical evaluation-key layouts:

```rust
use poulpy_ckks::presets::bootstrapping::n16_d35_k636_p20_c2s;

let preset = n16_d35_k636_p20_c2s()?;
let plan = preset.plan();
let keys_layout = preset.keys_layout();
let input_layout = preset.input_layout();
let bootstrap_layout = preset.bootstrap_layout();
```

Presets are named by what they offer, one token per axis: `n{log_n}_d{log_delta}_k{output_k}_p{log2_precision}_{circuit}`, i.e. ring-degree exponent, input scale exponent, output width in bits, guaranteed output precision in bits, and circuit (`c2s` for C2S-first, `s2c` for S2C-first), followed by a suffix per further technique: `ffr` for the first CoeffsToSlots factor on the raise and `ci` for EvalMod on the conjugate-invariant ring.
All output layouts use the input scale. The net usable budget is `output_k - input_k`: the application must stop consuming at `input_k`.
This matters for S2C-first presets: their SlotsToCoeffs runs before ModUp on the application's width, so `input_k` includes that consumption and a larger tail of the output is reserved than for a C2S-first preset; compare presets across circuits by their usable budget, never by `k`.
The presets take inputs at scale `2^35` and use an optimized Han–Ki EvalMod:

| Constructor | Pipeline | Input `k` | Output `k` | Minimum precision | Net usable bits (levels) | Bootstrap `k` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `n15_d35_k180_p18_c2s` | C2S-first | 40 | 180 | 18 bits | 140 (4) | 780 |
| `n16_d35_k636_p20_c2s` | C2S-first | 41 | 636 | 20 bits | 595 (17) | 1381 |
| `n16_d35_k733_p20_s2c` | S2C-first | 138 | 733 | 20 bits | 595 (17) | 1353 |
| `n16_d35_k685_p19_s2c_ffr_ci` | S2C-first, EvalMod on the CI ring | 160 | 685 | 19 bits | 525 (15) | 1431 |

The logN15 preset merges each transform into two seven-layer factors: C2S uses `[(7, 2048), (7, 16)]` at matrix scale `2^50`, and S2C uses the reversed schedule at scale `2^29`. EvalMod keeps the degree-30, interval-16 Han–Ki polynomial, three range-reduction steps, and coefficient scale `2^42`, with evaluation scale `2^53`. C2S, EvalMod, and S2C consume 100, 424, and 58 bits; restoring scale `2^35` consumes another 18 bits. Thus `780 - 100 - 424 - 58 - 18 = 180`, leaving four 35-bit levels above the 40-bit input width.

Both logN16 presets use three factors per transform, C2S `[(5, 4096), (5, 256), (5, 8)]` and S2C `[(4, 4), (5, 128), (6, 4096)]`, each with the giant steps that need the fewest key switches. With three factors the precision of CoeffsToSlots depends far more on its scale than with four: the S2C-first preset runs it at matrix scale `2^50` and SlotsToCoeffs at `2^30`, the C2S-first preset at `2^51` and `2^35`. Both keep 17 usable levels; the S2C-first preset needs a raised modulus 28 bits narrower.

The logN16 C2S-first preset takes log message ratio 6, hence its 41-bit input. It internally reaches 659 bits at scale `2^58`; `ckks_bootstrap` restores scale `2^35` and returns 636 bits automatically. This leaves exactly `636 - 41 = 595` bits before the next bootstrap. No caller-side scale adjustment is needed.

The S2C-first preset uses six internal guard bits for CoeffsToSlots, log message ratio 13, and C2S matrix scale 50. The guard bits are removed at the bootstrap output; the application scale remains `2^35`. Custom S2C-first plans can select this lift with `with_c2s_guard_bits`; width accounting includes its cost.

All presets use weight 1024 for the dense secret and weight 32 for sparse-secret encapsulation. Each secret carries its modulus bound for 128 bits of security, an estimate every key must fit: 1707 and 345 bits at logN16, 851 and 162 bits at logN15. The bounds come from the 128-bit table of [sparse-key-estimate](https://github.com/jdumezy/sparse-key-estimate/blob/aa2c40d4eee4551e6fba06c55d7432f6918e1597/Precomputed-Tables/128bits_security.md), which estimates the primal hybrid attack on sparse secrets ([ePrint 2026/279](https://eprint.iacr.org/2026/279)); the module documentation of `presets::bootstrapping` gives the cross-checks against [ePrint 2022/024](https://eprint.iacr.org/2022/024) and the dual attacks.
Preset construction validates the ciphertext, gadget, auxiliary, and total moduli of every key against these bounds.
A key's digit count comes from `GGLWELayout::dnum_for_input` (`⌈k / k_digit⌉`), the same rounding the key-switch operations apply. Behind its digits a key keeps a guard `k_aux` of one digit plus `log2(N)` bits, or what the bound leaves when that is less, but never less than one digit (`k_aux >= k_digit`), and its precision exceeds the input and one digit by a 2-bit margin.
Each high-modulus key takes the whole-limb digit of fewest rows its use admits, and at that row count the narrowest. Digits off a limb boundary would cost the key switches the digit layout of their input. At the nominal radix of 52 bits, the logN16 keys used at the raised width have 7 digits of 208 bits; the rotation keys of the later CoeffsToSlots factors and the conjugation key have wider digits and fewer rows, and EvalMod relinearizes with a tensor key shaped for its widest product. LogN15 keeps 15 digits of 52 bits at the raised width. The dense-to-sparse key keeps one 52-bit digit: 52 gadget bits plus 68 auxiliary bits at logN16.
`with_base2k` re-derives a preset at another limb radix, and `with_dsizes` with one digit of whole limbs for every high-modulus key, keeping the plan and widths and re-running the same validation.
`BootstrappingPreset::max_backend_base2k` gives the largest radix at which every accumulation of the bootstrap fits the extended precision of a backend, and of its CI backend for EvalMod on that ring, with a failure probability of at most `2^-failure_bits` over the bootstrap, by the model of `Module::max_base2k`, and `with_backend_base2k` re-derives the preset at the largest such radix the modulus bounds admit.
The widest accumulations are the giant-step passes of the transforms: an output limb sums the product of every diagonal of a factor with a baby rotation, over the diagonal's limbs, and the key switch of every giant step, over the operand limbs, a few hundred products at logN16 and logN15. A first factor on the raise sums one key switch per diagonal over fewer limbs, which that count covers, a folded `u` runs the last CoeffsToSlots pass twice, and EvalRound+ adds the pass of its cancellation. An output limb of each of EvalMod's ciphertext products, which `EvalModPlan::max_ciphertext_products` bounds from the degrees of its polynomials and its range-extension steps, sums a product per limb, one of them a square, whose mean on the CI ring comes off the threshold once (`Module::max_base2k_squares`); on that ring the two conversions of degree `2N` sum a key-switch product per operand limb. The fast preset takes 52 on NTT4x30: at 53 a CI square's mean alone exceeds half the modulus.
The test and benchmark runner takes, for a failure probability of `2^-128` per bootstrap, radix 53 on NTT4x30, 56 on NTT3x42 and 17 on FFT64, and for a preset with EvalMod on the CI ring 52 on NTT4x30 and 14 on FFT64. At 56 the logN15 preset's high-modulus keys reach its dense-secret bound exactly. Below the nominal radix it keeps the largest uniform high-modulus digit up to 7 limbs that fits the bounds: FFT64 uses high-modulus digits of 2 limbs for logN15 or 7 for logN16, and a dense-to-sparse digit of 1 limb.
A plan can leave the first CoeffsToSlots factor to the modulus raise (`DFTPlan::with_first_factor_on_raise`, `BootstrappingPreset::with_first_factor_on_raise`). A raised ciphertext is small until its first key switch, so a plaintext product followed by a division keeps it at the same precision: the factor consumes no modulus, and the raised width drops by one CoeffsToSlots scale. Each diagonal's term is rotated with no key and switched to the dense secret by a switching key of its own, from the rotated raise secret, whose rows sit under the term's few limbs and end at the width the raise works at, at least 16 guard bits above the raised width and rounded up to a limb (`k_offset`); the sparse-to-dense switch is not used and its key is not generated (`EncapsulationKeysLayout::sparse_to_dense` is `None`). The switches are summed in the transform domain and rounded once. The technique follows ePrint 2025/1403.
Applied to `n16_d35_k636_p20_c2s` with the CoeffsToSlots schedule `[(3, 1), (6, 512), (6, 8)]`, it lowers the raised width by 51 bits, from 1381 to 1330, but its eight first-factor keys and the 60 rotation keys of two six-layer factors grow the key set from 8310 to 9386 MiB, and a bootstrap takes about 6% longer on the NEON Rayon backend.
`n16_d35_k685_p19_s2c_ffr_ci` combines it with EvalMod on the CI ring: four three-layer CoeffsToSlots factors at scale `2^48` after the one on the raise, `u` and `v` at `2^46` and `2^38`, and four SlotsToCoeffs factors at `2^28`. Three-layer factors take 24 key switches where three four-layer ones take 30, for one more scale of raised width (1431 bits instead of 1383) and fewer rotation keys (35 instead of 43). Against `n16_d35_k733_p20_s2c` it gives up a bit of precision and two levels for about 20% less time per bootstrap on the NEON Rayon backend and a smaller key set: on an Apple M5 Pro, `NTT4x30NeonRayon/ckks/ckks_bootstrapping/n16_d35_k685_p19_s2c_ffr_ci` takes 0.65 s and `NTT4x30NeonRayon/ckks/ckks_bootstrapping/n16_d35_k733_p20_s2c` 0.81 s.
The advertised precision is the minimum slot-wise precision measured on the reference vector at the nominal shape with `f64` DFT matrices; `test_suite::presets::bootstrapping_presets_meet_precision` (registered as an ignored test by the backend crates) and the `ckks_bootstrapping` benchmark assert the measurement against it on exact backends.

The keys are generated by `generate_keys`, which returns the unprepared `BootstrappingKeySet` — the serializable, GPU-resident form — and a `prepare` step preprocesses the whole set for evaluation.
Four key roles are used:

| Key | Role |
| --- | --- |
| `rotation_keys` | Automorphism keys for the DFT rotations, read off the compiled matrices, and the Galois `−1` automorphism of the split real/imaginary transform |
| `tensor_key` | Relinearization key for the EvalMod range-extension squarings |
| `encapsulation_keys` | Optional `denseToSparse` / `sparseToDense` pair for sparse-secret encapsulation, with no `sparseToDense` key when the first CoeffsToSlots factor is on the raise |
| `first_factor_keys` | With the first CoeffsToSlots factor on the raise, one switching key per diagonal of that factor, from the rotated raise secret to the dense one |

Keys shaped by use cover the widths of the bootstrap they were laid out for. `ckks_bootstrap` runs at the width of `ct_out` and checks every rotation key and the tensor key at the widths that width sets before it starts, so a wider output fails up front, with the key source's refusal as the error's source.

## Batches

A batch is refreshed with as few bootstraps as possible by folding it into standard ciphertexts of the bootstrap degree `N` (`CKKSFoldOps::ckks_fold`), bootstrapping each with `ckks_bootstrap`, and unfolding the results (`ckks_unfold`) into outputs labeled like the inputs.
The fold packs real inputs in pairs `x + i·y`, conjugate-invariant inputs in pairs once embedded, merges inputs of a smaller degree `n` by `N/n`, and lets sparse inputs share coefficient positions; unfolding reverses each step, and paired outputs lose one bit to the split.

The keys are passed directly:

- The ring-switch keys (`RingSwitchKeysLayout::generate`) switch between the input secret and the bootstrap secret: the fold takes the inbound one and the unfold the outbound one; inputs under the bootstrap secret at its degree pass `None`.
- `automorphisms`, keys of the input secret for the elements of `ckks_unfold_galois_elements`, split real pairs and sparse inputs; for sparse inputs they cover `log_sparsity` bits beyond the refreshed width. Inputs under the bootstrap secret use its rotation keys.

The fold takes the bootstrap degree explicitly, as the degree of the folded ciphertexts: the module only bounds it. `CKKSFoldLayoutOps` gives the allocation-only `GLWELayout` of that degree and sizes the fold from layouts (`ckks_fold_count` gives the number of folded ciphertexts), and the context's slot count must match the folded ciphertexts: merged ones need full-slot transforms (`log_slots = log2(N) − 1`). The fold sets scale, sparsity and slot kind on each folded ciphertext; pairing or merging marks it complex, and merging clears sparsity.

Conjugate-invariant inputs of degree `n` fold on the same standard module: each is embedded into the standard ring of degree `2n`, and the unfold takes the relative trace back, at one bit of budget. Their ring-switch keys and the automorphism keys that split sparse inputs are those of the embedded CI secret (`glwe_secret_ci_embed`); pairs split without keys, since that secret is invariant under conjugation.

## Cost and where to look

The total arithmetic cost is `consumed_bits`; its placement around ModUp is given by the pre- and post-ModUp costs.
A small self-contained parameter set (ring degree `n = 2048`, `K = 16`, message ratio `2^11`, `log_delta = 45`, a degree-30 `CosHK` EvalMod) recovers the slots to a few bits of precision on the reference backend, which is the floor the end-to-end test asserts; wider parameters recover proportionally more.

- `poulpy-ckks/src/test_suite/bootstrapping.rs` runs `ckks_bootstrap` end to end for the C2S-first, EvalRound+, and S2C-first pipelines.
- `poulpy-cpu-portable/examples/bootstrap_trace.rs` runs the standard pipeline for profiling.

```sh
cargo test -p poulpy-cpu-portable --features enable-ckks --release ntt4x30_f64::bootstrapping -- --nocapture
```
