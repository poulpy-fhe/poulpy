# Replicated slot layout

`poulpy-ckks` implements the slot layout of the small-batch scheme of *Fully Homomorphic Encryption for Small Batches of Complex Numbers* (Dumezy, [ePrint 2026/2513](https://eprint.iacr.org/2026/2513)) as a slot layout of CKKS.
A batch of `k = 2^log_k` complex values is replicated across the ring, one slot class out of `R = 2^log_r` carrying data and the others serving as guards.
A shaped rounding leaves an error that is a multiple of a fixed polynomial `g`, small on the data class, so the guards absorb it.
Here encoding, encryption, products with their rescale, and the key switches of automorphisms round with shaping, and sums and differences add exactly.
The operations that still round plainly refuse a replicated layout.

## Layout

`SlotLayout` is part of the plaintext metadata of every CKKS plaintext and ciphertext (`GLWEPlaintextMeta::slot_layout`).
`SlotLayout::Full` is the usual CKKS layout; `SlotLayout::Replicated(ReplicatedLayout)` describes:

- `log_k`: the batch size `k = 2^log_k`. Slot `t` carries data iff `R | t`, holding value `(t/R) mod k`, so the batch repeats every `kR` slots;
- `log_r`: the period `R = 2^log_r` (`1 <= log_r <= 5`);
- `U ∈ Z[Y]/(Y^(2R)+1)`: the shaping polynomial, with `g = U(X^(N/(2R)))`. `U(1)` odd makes it invertible modulo every power of two.

Slot class `j` sees `g` through `U(ω^(5^j))`, `ω = e^(iπ/(2R))`.
`ShapingProfile` (`poulpy_ckks::shaping`) reports these class magnitudes, the suppression on the data class, the amplification on the worst guard class and the largest exact rounding pass.
`ShapingUnit` lists cyclotomic units `U` with the smallest guard amplification for a prescribed suppression, solved exactly as integer programs: `U4_{10,15}` (`R = 2`), `U8_{20,30,40}` (`R = 4`), `U16_{20,40,60}` (`R = 8`) and `U32_{20,40,45,50,60}` (`R = 16`). For `R = 32`, `U64_{20,40,60,80}` are the best units branch and bound found in 15 minutes, each guard bound within 0.81 bits of the proven optimum; they amplify the guards less than the `R = 16` units at the same suppression (4.24 bits against 5.69 at 60 bits) and reach 80 bits, for half the batch capacity.
A unit has norm one, so the `R - 1` guard classes share the suppression: at least `T/(R - 1)` bits each for `T` bits of suppression.
At `R = 2` the one guard class grows as much as the data shrinks. Fewer guard classes also take larger coefficients for the same suppression, which leave `i16` past a target of 15 bits at `R = 2` and 45 bits at `R = 4`, and a bootstrapping at `R = 2` is only as precise as its unit's suppression.
`ShapingUnit::layout(log_k)` builds the layout of a batch.

## Shaped operations

`CKKSRoundOps::ckks_round` rounds a ciphertext to a smaller width at a fixed torus value.
On a replicated layout, the `λ` dropped bits are divided out as `h = [U^(-1)·c]_(2^λ)` and the remainder `c − g·h` is carried, so the error is a multiple of `g` up to a remainder `2^-λ` smaller.
The other operations build on it:

- encoding quantizes at `log_delta + λ` bits and rounds with shaping;
- `ckks_encrypt_sk` encrypts `ShapingProfile::max_pass_bits` wider and rounds to the ciphertext width, so the fresh error is shaped;
- `ckks_encode_slots_host_into` and `ckks_decode_slots_host_into` take the `2^log_k` values of the batch, decoding every replica and averaging them;
- products (`ckks_mul_*`, `ckks_square_*`, `ckks_mul_pt_*`) and automorphisms (`ckks_rotate_*`, `ckks_conjugate_*`) are computed `ShapingProfile::shaping_depth(log_delta)` bits under the scale, `min(log_delta + 24, max_pass_bits)` (`SHAPED_WORKING_BITS`), and rounded with shaping; a ciphertext product that takes the tensor path needs a tensor key covering that wider intermediate;
- constants are multiplied as masked plaintexts `c·π_S`, constant on the data slots and zero on the guards (`ckks_encode_masked_constant_host_into`, `MaskedCoeffs`); `ckks_{add,sub}_one_assign` add the masked one;
- polynomial evaluation and EvalMod take `MaskedCoeffs` (`encode_bsgs_replicated`, `ckks_eval_mod_replicated_compile`), refusing constants without their masked plaintexts before any work, and evaluate each baby step as one shaped linear combination, each power prepared once per polynomial.

Sums and differences of one slot layout and one scale add exactly.
Operations that would round plainly or write the guard slots return `InvalidOperand` on a replicated layout: constant additions, rotations by a slot count that is not a multiple of `R`, dot products of prepared operands, inner sums, the full-layout linear transformations and DFT, ModUp, bootstrapping, PaCo and SHIP.

## Conversions

`CKKSReplicatedOps` moves a batch between CKKS and the replicated layout with the same keys.
`ckks_to_replicated` maps `k` values of a full-layout ciphertext with period `k` onto the data slots; `ckks_from_replicated` gathers them back, averaging the replicas.
`ReplicatedConversion` holds the encoded diagonals and lists the rotations its keys cover.
`ReplicatedConversion::new` encodes each map as `k` diagonals; `ReplicatedConversion::new_factored` encodes it as the move between consecutive slots and the data slots in `placement_factors` factors of about `k^(1/placement_factors)` 0/1 mask diagonals, which needs no `R | k` for the return.
Every factor consumes its diagonal bits: the last factor of `ckks_to_replicated` is quantized with shaping, its others plainly at `inner_log_delta_diag` bits, a few bits above the working precision.
Allocate a `ReplicatedTransformWorkspace` once with `conv.alloc_to_workspace(&module, &input)` or `conv.alloc_from_workspace(&module, &input)` and pass it after `conv` to each evaluation.
The workspace owns the prepared baby rotations for that input layout; evaluations refill those buffers without reallocating them.

## Bootstrapping

`CKKSReplicatedBootstrapOps::ckks_replicated_bootstrap` refreshes a replicated ciphertext in `SlotsToCoeffs`-first order:

1. a normalized trace averages the `N/(2kR)` copies of the period;
2. `SlotsToCoeffs` gathers the data slots onto the coefficients of the subring of degree `2k`;
3. ModUp, with sparse-secret encapsulation;
4. a normalized trace onto period `k`, then `CoeffsToSlots`, with the real and imaginary parts side by side when they fit (packed);
5. EvalMod (cosine with double angles), on both parts at once when packed;
6. a shaped rounding back to the leveled scale.

`ReplicatedBootstrapPlan` holds the circuit (`ReplicatedBootstrapPlan::paper` is the configuration of the paper) and its widths (`k_in`, `k_boot`, `key_k`).
EvalMod enters its scale with a shaped rounding, which leaves noise of about `2^(G + log2(sqrt(N)))` units on the guard slots for a unit whose guards grow by `G` bits.
The cosine polynomial and the double angles square the guards, so they must stay well below one: at `2^28`, the guards of `U8_40` (`G = 14.58`) reach `2^7` after the polynomial and wrap the data after the double angles.
`ReplicatedBootstrapPlan::eval_mod_log_delta` keeps the paper's `2^28` for units whose guards grow by at most 13 bits, and runs EvalMod at `ceil(G) + 15` bits above that, `2^30` for `U8_40` and `2^31` for `U4_15`.
Its `transforms` select how `SlotsToCoeffs` and `CoeffsToSlots` are evaluated:

- `ReplicatedTransforms::Dense`, the paper's: each map is one BSGS transform of `k` (`SlotsToCoeffs`) or `2k` (packed `CoeffsToSlots`) diagonals, consuming one diagonal scale;
- `ReplicatedTransforms::Factored`: each map is the DFT on period `k` in `s2c_factors` or `c2s_factors` factors and the move between consecutive slots and the data slots in `placement_factors` factors of 0/1 masks, each factor a BSGS transform whose giant step is fitted to its offsets. Every `SlotsToCoeffs` factor consumes `s2c_log_delta_diag` bits; the last `CoeffsToSlots` factor is quantized with shaping at `c2s_log_delta_diag` bits and the others plainly at `c2s_inner_log_delta_diag` bits. Factored maps take at least `2^f` values for `f` factors, and a packed plan needs the two batches to fit the ring.

Factoring trades levels and rotation keys for fewer diagonals: at `N = 2^16` and `k = 2^10`, the dense maps multiply by about 3000 diagonals with 69 rotation keys, the factored maps of `n16_d20_k763_p19_s2c_factored` by 381 diagonals with 83 keys, for a raised modulus 256 bits wider at the same levels.
At `k = 2^10`, the factored preset bootstraps in 6.3 s on one `NTT4x30Neon` thread and 0.36 s on Metal, against 8.1 s and 0.52 s for dense maps at the same levels, within 0.2 bits of their precision. At `k = 2^8` the dense maps are faster, the diagonals no longer dominating.
`ReplicatedBootstrapping` compiles it for a module: the transforms, the masked EvalMod constants, and the Galois elements its keys cover.
`ReplicatedBootstrapPlan::secure_key_layouts` sizes the gadgets of the keys with the fewest digits whose modulus fits the caller's bounds for the dense secret and for the ephemeral sparse secret, with one automorphism key for each Galois element of the compiled bootstrapping.

### Presets

`poulpy_ckks::presets::replicated_bootstrapping` provides three parameter sets for 128-bit security, with dense secrets of weight 1024 and an ephemeral ModUp secret of weight 32.
The batch size and the shaping unit are the caller's.
The input and output widths do not depend on them, while the raised modulus and the keys follow the unit's EvalMod scale: `plan` and `keys_layout` take the layout.
A preset refuses a unit whose suppression is less than 8 bits above its precision, since the bootstrapping falls short of the suppression by up to 5 bits: the units of `R = 2` and those targeting 20 bits, which measure 16 to 19 bits on the logN14 preset.
The logN16 preset factors its maps into two DFT factors and two placement factors each, with `SlotsToCoeffs` diagonals at `2^30` and inner `CoeffsToSlots` diagonals at `2^54`, for batches of 4 to `2^10` values at `R = 16`.
The sparse-to-dense key covers the 49-bit lifted ModUp input, with a row offset ending its gadget at the raised modulus. Its full modulus includes that offset, the gadget and the auxiliary bits.

| preset | ring | levels at `2^20` | keys (digits × width + aux) | ephemeral key | sparse-to-dense key (offset + gadget + aux) | min precision, k = 16 / 64 |
|---|---|---|---|---|---|---|
| `n14_d20_k109_p21_s2c` | `2^14` | 2 | 14 × 28 + 28 | 2 × 22 + 22 | 333 + 2 × 25 + 25 | 21.79 / 21.73 bits |
| `n15_d20_k389_p20_s2c` | `2^15` | 16 | 4 × 168 + 168 | 1 × 43 + 43 | 614 + 1 × 49 + 49 | 21.16 / 20.84 bits |
| `n16_d20_k763_p19_s2c_factored` | `2^16` | 30 | 3 × 403 + 403 | 1 × 43 + 43 | 1150 + 1 × 49 + 49 | 20.52 / 20.38 bits |

The logN14 preset advertises 21 bits of precision on the data slots, the logN15 preset 20 bits and the logN16 preset 19 bits, which the precision test asserts.
The logN16 preset measures 20.35 bits at `k = 2^8` and 20.00 bits at `k = 2^10` on `NTT4x30Neon`. The test covers offset and full-width return keys on matched ciphertexts, and `U8_40` at `k = 64`.

With `U8_40`, EvalMod at `2^30` raises the modulus by 16 bits:

| preset | raised modulus | keys (digits × width + aux) | min precision, k = 64 |
|---|---|---|---|
| `n14_d20_k109_p21_s2c` | 399 bits | 34 × 12 + 12 | 21.28 bits |
| `n15_d20_k389_p20_s2c` | 679 bits | 5 × 138 + 138 | 20.94 bits |
| `n16_d20_k763_p19_s2c_factored` | 1215 bits | 3 × 408 + 408 | 20.61 bits |

The logN14 bound of 424 bits leaves only 12-bit digits, so its key switches take 34 digits instead of 14.

## Conjugate-invariant ring

The layout also runs on the conjugate-invariant ring of degree `N`, the subring of `Z[X]/(X^(2N)+1)` fixed by `X -> X^-1`, and its `N` real slots.
A batch of `k` real values takes `kR` slots, slot `t` in class `t mod R` as on the standard ring, and the period repeats `N/(kR)` times.
The shaping polynomial must be invariant too, `g = U(X^(N/R))` with `U` in `Z[Y + Y^-1]`.
Every unit of `Z[Y]/(Y^(2R)+1)` is a power of `Y` times an invariant unit with the same class magnitudes and norm, so this ring shapes with `Y^s*U` for the shift `s` that `ShapingProfile::<ConjugateInvariant>::try_new` finds (`ShapingProfile::shift`). The profile carries its ring in its type, so a profile of one ring cannot reach the shaped rounding of the other.
Every `ShapingUnit` has one, and a `U` without one is refused.
Rounding reads the invariant coefficients, which keeps `h` invariant and rounds each tie once.

Leveled operations, polynomial evaluation, EvalMod and the conversions take the same calls as on the standard ring.
Masked constants and the masked one keep their real part, as the slot encoder does.
The conjugation, complex polynomials, the packed EvalMod and diagonals with an imaginary part are standard-only.

`CKKSReplicatedCIBootstrapOps::ckks_replicated_ci_bootstrap` bootstraps through the standard ring of degree `2N` of its module, where the `ReplicatedBootstrapping` is compiled, the conjugate-invariant input and output of degree `N` living on the same module:

1. The input is embedded (`GLWECIEmbedKey`) at the shaping depth under its scale, then rounded with shaping to its width.
2. The standard bootstrapping refreshes it, one bit wider than the output.
3. The relative trace (`GLWECITraceKey`) brings it back at the shaping depth under the leveled scale, then a shaped rounding gives the output width.

The key switches of the ring maps and the halving of the trace stay under the shaping depth, so the data slots keep the precision of the standard bootstrapping.
The two keys switch between the conjugate-invariant secret and the standard secret of the bootstrapping keys (`GLWECIKeyEncryptSk`), and cover the widths of the ring maps.
`ReplicatedBootstrapping` compiles for the standard ring, and the presets are sized for it.

## Construction outline

```rust,ignore
use poulpy_ckks::{
    api::CKKSReplicatedBootstrapOps,
    layouts::ReplicatedBootstrapping,
    presets::replicated_bootstrapping::n15_d20_k389_p20_s2c,
    shaping::ShapingUnit,
};

let preset = n15_d20_k389_p20_s2c()?;
let layout = ShapingUnit::U32_45.layout(log_k)?;
let plan = preset.plan(layout)?;
// `scratch` covers `ReplicatedBootstrapping::<MyBackend, f64>::new_tmp_bytes(&module, &plan, base2k)?`.
let ctx = ReplicatedBootstrapping::<MyBackend, f64>::new(&module, plan, base2k, &mut scratch)?;
let keys = ctx
    .generate_keys(&module, &sk, &preset.keys_layout(layout, &ctx.galois_elements(&module))?, preset.sparse_secret_hamming_weight(),
                   &mut source_xs, &mut source_xe, &mut source_xa, &mut scratch)
    .prepare(&module, &mut scratch);

// `input` holds `2^log_k` values at the leveled scale and width `preset.input_k()`.
let mut output = module.ckks_ciphertext_alloc_from_infos(&preset.output_layout(layout));
let mut workspace = ctx.alloc_workspace(&module, &output);
module.ckks_replicated_bootstrap(&mut output, &input, &ctx, &mut workspace, &keys, &mut scratch)?;
```

The bootstrap workspace is reusable for the same output width and plan. Concurrent evaluations use separate workspaces and scratch arenas.

`ReplicatedBootstrappingPresetRun` in `poulpy_ckks::test_suite::replicated_presets` is a complete example, from key generation to the precision measurement.

## Validation

The CKKS test suite checks every shaped operation against the full layout on the data slots (`test_round_*`, `test_replicated_*`), every unit on a product chain, that operations which would round plainly reject a replicated input, and the bootstrapping on a small ring.
`test_replicated_parity` checks that the shaped operations give the same digits on every pair of backends, on both rings and at ranks 1 and 2, and the encryption parity suites check shaped encryption under the same draws.
`conjugate_invariant_ckks_test_suite!` runs the tests that apply to the conjugate-invariant ring the shaped rounding on the unfolded coefficients and `test_replicated_ci_bootstrap`.
The ignored test `replicated_bootstrapping_presets_meet_precision`, registered on the NTT backends, runs every preset at both batch sizes and asserts the advertised precision; the benchmark `bench_ckks_replicated_bootstrapping` of `poulpy-bench` runs the same configurations.
