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
At `R = 2` the one guard class grows as much as the data shrinks. Fewer guard classes also take larger coefficients for the same suppression, which leave `i16` past a target of 15 bits at `R = 2` and 45 bits at `R = 4`.
`ShapingUnit::layout(log_k)` builds the layout of a batch.

## Shaped operations

`CKKSRoundOps::ckks_round` rounds a ciphertext to a smaller width at a fixed torus value.
On a replicated layout, the `λ` dropped bits are divided out as `h = [U^(-1)·c]_(2^λ)` and the remainder `c − g·h` is carried, so the error is a multiple of `g` up to a remainder `2^-λ` smaller.
The other operations build on it:

- encoding quantizes at `log_delta + λ` bits and rounds with shaping;
- `ckks_encrypt_sk` encrypts `ShapingProfile::max_pass_bits` wider and rounds to the ciphertext width, so the fresh error is shaped;
- `ckks_encode_slots_host_into` and `ckks_decode_slots_host_into` take the `2^log_k` values of the batch, decoding every replica and averaging them;
- products (`ckks_mul_*`, `ckks_square_*`, `ckks_mul_pt_*`) and automorphisms (`ckks_rotate_*`, `ckks_conjugate_*`) are computed `ShapingProfile::shaping_depth(log_delta)` bits under the scale, `min(log_delta + 24, max_pass_bits)` (`SHAPED_WORKING_BITS`), and rounded with shaping; a ciphertext product that takes the tensor path needs a tensor key covering that wider intermediate;
- constants are multiplied as masked plaintexts `c·π_S`, constant on the data slots and zero on the guards (`ckks_encode_masked_constant_host_into`), the full layout being the case `π_S = 1`; `ckks_{add,sub}_one_assign` add the masked one;
- polynomial evaluation and EvalMod take each constant as its own plaintext in the layout of their input (`encode_bsgs` on the host module with `coeff_meta` of that layout, each constant of degree `2R`, `R` on the conjugate-invariant ring, and `ckks_eval_mod_compile` of a plan `with_slot_layout`), refusing constants of another layout before any work, and evaluate each baby step as one shaped linear combination, each power prepared once per polynomial.

Sums and differences of one slot layout and one scale add exactly.
Operations that would round plainly or write the guard slots return `InvalidOperand` on a replicated layout: rotations by a slot count that is not a multiple of `R`, dot products of prepared operands, inner sums, the full-layout linear transformations and DFT, ModUp, a bootstrapping plan of the full layout, PaCo and SHIP.

## Conversions

`CKKSReplicatedOps` moves a batch between CKKS and the replicated layout with the same keys.
`ckks_to_replicated` maps `k` values of a full-layout ciphertext with period `k` onto the data slots, and `ckks_from_replicated` gathers them back, averaging the replicas.
`ReplicatedConversionPrepared` holds the prepared maps and lists the Galois elements of its keys (`galois_elements`).
`ReplicatedConversionPrepared::new` encodes each map as `k` diagonals for ciphertexts of a degree `n` at most the module's, which the conversions take.
`ReplicatedConversionPrepared::new_factored` encodes it as the move between consecutive slots and the data slots in `placement_factors` factors of about `k^(1/placement_factors)` 0/1 mask diagonals, which needs no `R | k` for the return.
Every factor consumes its diagonal bits: the last factor of `ckks_to_replicated` is quantized with shaping, its others plainly at `inner_log_delta_diag` bits, a few bits above the working precision.

## Bootstrapping

A replicated ciphertext is refreshed by `ckks_bootstrap` with an S2C-first plan set on its layout, `BootstrappingPlan::new(..).with_slot_layout(SlotLayout::Replicated(layout), log_n)`, for ciphertexts of degree `2^log_n`.
The plan takes the settings of the paper as plan settings: `with_s2c_guard_bits` lifts the input of `SlotsToCoeffs`, `with_working_bits` sets the bits under the scale at which `SlotsToCoeffs` works, which the raise reads, and EvalMod rounds its parts, and `with_c2s_guard_bits` the guard bits of the raise, the working bits included. Its queries give its input and bootstrap `k` (`input_k`, `bootstrap_k`) and keys as on the full layout.
The steps read the plan from the run of the compiled context:

1. `ckks_slots_to_coeffs_assign`, at scaling 1 on a copy of the input lifted by the S2C guard and working bits, averages the `N/(2kR)` copies of the period by a normalized trace, and gathers the data slots onto the coefficients of the subring of degree `2k`.
2. `ckks_mod_raise`, with sparse-secret encapsulation, reads the working bits and lifts the message by the rest of the C2S guard bits.
3. `ckks_coeffs_to_slots_assign` takes a normalized trace onto period `k`, then writes the coefficients onto the data slots, with the real and imaginary parts side by side when EvalMod is packed.
4. `ckks_bootstrap_eval_mod` rounds the real and imaginary parts with shaping, evaluates EvalMod (cosine with double angles) on both at once when packed (`EvalModPlan::packed`), on the real part alone for a real message otherwise, recombines them, and rounds with shaping to the scale of the input, both guard bits removed.

The output, allocated at the bootstrap `k`, ends at the scale and slots of the input and at the `k` the plan leaves, as on the full layout.

EvalMod enters its scale with a shaped rounding, which leaves noise of about `2^(G + log2(sqrt(N)))` units on the guard slots for a unit whose guards grow by `G` bits.
The cosine polynomial and the double angles square the guards: at `2^28`, the guards of `U8_40` (`G = 14.58`) reach `2^7` after the polynomial and wrap the data after the double angles.
The presets run EvalMod at `2^max(28, ceil(G) + 15)`: the paper's `2^28` for units whose guards grow by at most 13 bits, `2^30` for `U8_40`.
`SlotsToCoeffs` and `CoeffsToSlots` come from the DFT generator on the subring of degree `2k`, as `DFTPlan`s on the layouts of the plan, whose evaluation starts with the normalized trace. Their settings select the maps (`DFTPlan::with_placement_factors`, `with_last_factor_log_delta`):

- dense, the paper's: each map is one factor, the DFT merged with the placement, one BSGS transform of `k` diagonals, consuming one diagonal scale.
- factored: each map is the DFT on period `k` in the factors of its schedule and the move between consecutive slots and the data slots in `placement_factors` factors of 0/1 masks, each factor a BSGS transform, the DFT factors on the giant steps of the schedule and the placement factors on a square-root one, unless `DFTPlan::with_optimal_bsgs` fits every factor's to its offsets, as the factored preset does. Every `SlotsToCoeffs` factor consumes the scale of its plan. The last `CoeffsToSlots` factor is quantized with shaping at `with_last_factor_log_delta` bits and the others plainly at the scale of its plan. Factored maps take at least `2^f` values for `f` factors.

A packed EvalMod needs its two batches to fit the ring, `4kR <= N`, which `with_slot_layout` checks with the other settings the layout takes.
Factoring trades levels and rotation keys for fewer diagonals: at `N = 2^16` and `k = 2^10`, the dense maps take 69 rotation keys and the factored maps of `n16_d20_k763_p22_s2c_factored` 83, for a raised modulus of 1207 bits against 955 at the same levels.
`BootstrappingContextPrepared::compile` compiles the plan on a module of at least its degree `2^log_n`, which the bootstrap takes its ciphertexts at: the transforms through `ckks_dft_matrix_new` and the masked EvalMod constants through `ckks_eval_mod_compile`, as on the full layout.
`BootstrappingPlan::galois_elements` lists the keys of the plan's traces, maps and packed rotation, and `key_widths` the `k` at which a bootstrapping uses each, its widest use, from the index-only replay of the transforms, which shares the generator's geometry. The entry check reads these `k`, `BootstrappingKeyPolicy::keys_layout` shapes keys from them with the digits of fewest rows under the caller's modulus bounds, and `generate_keys` generates them, from the plan or the compiled context.

### Presets

`poulpy_ckks::presets::bootstrapping` provides three replicated parameter sets for 128-bit security, with dense secrets of weight 1024 and an ephemeral ModUp secret of weight 32.
Each constructor takes the caller's layout, its batch size and shaping unit, and returns a `BootstrappingPreset`.
The input and output `k` do not depend on the layout, while the raised modulus and the keys follow the unit's EvalMod scale.
A preset refuses a unit whose suppression is less than 8 bits above its precision: the units of `R = 2` and those targeting 20 bits.
The logN16 preset factors its maps into two DFT factors and two placement factors each, with `SlotsToCoeffs` diagonals at `2^30` and inner `CoeffsToSlots` diagonals at `2^54`, for batches of `2^4` to `2^10` values at `R = 16`.
The keys come from `BootstrappingKeyPolicy` under the bounds of the Security section of the bootstrapping presets, each covering its widest use with whole-limb digits, or a digit off a limb boundary where no whole limb fits the bound. The tensor key covers the result of EvalMod's widest product, which a replicated layout computes its shaping depth under the scale.
The sparse-to-dense key covers the 57-bit lifted ModUp input, with a row offset ending its gadget at the raised modulus. Its full modulus includes that offset, the gadget and the auxiliary bits.

| preset | ring | levels at `2^20` | widest rotation key (digits x `k_digit` + aux) | tensor key | ephemeral key | sparse-to-dense key (offset + gadget + aux) | min precision, k = 16 / 64 |
|---|---|---|---|---|---|---|---|
| `n14_d20_k109_p21_s2c` | `2^14` | 2 | 14 x 28 + 32 | 7 x 52 + 60 | 3 x 17 + 26 | 333 + 2 x 29 + 33 | 23.26 / 22.93 bits |
| `n15_d20_k389_p21_s2c` | `2^15` | 16 | 7 x 104 + 119 | 7 x 104 + 119 | 1 x 52 + 67 | 567 + 1 x 104 + 119 | 23.30 / 22.99 bits |
| `n16_d20_k763_p22_s2c_factored` | `2^16` | 30 | 3 x 416 + 432 | 2 x 520 + 536 | 1 x 52 + 68 | 1103 + 1 x 104 + 120 | 23.23 / 23.33 bits |

The logN14 and logN15 presets advertise 21 bits of precision on the data slots and the logN16 preset 22 bits, their floors measured on `U8_40` and `U32_45`, which the precision test asserts; the other units a preset admits are not measured.
On the rows of `replicated_presets`, with its own keys, the logN16 preset measures 22.95 bits at `k = 2^8` and 22.30 bits at `k = 2^10`, against 23.02 and 23.00 bits with dense maps. The test covers these rows with offset and offset-free return keys, and `U8_40` at `k = 64`.
The dense rows of `replicated_presets`, `n16_d20_k673_p23_s2c` in the test suite, keep the levels of the logN16 preset and advertise their measured 23 bits; a preset refuses every re-derivation but its radix, key digits and output `k`.

With `U8_40`, EvalMod at `2^30` raises the modulus by 16 bits:

| preset | raised modulus | widest rotation key (digits x `k_digit` + aux) | tensor key | min precision, k = 64 |
|---|---|---|---|---|
| `n14_d20_k109_p21_s2c` | 407 bits | 34 x 12 + 16 | 7 x 53 + 53 | 21.39 bits |
| `n15_d20_k389_p21_s2c` | 687 bits | 7 x 104 + 119 | 7 x 104 + 119 | 21.61 bits |
| `n16_d20_k763_p22_s2c_factored` | 1223 bits | 3 x 416 + 432 | 2 x 520 + 536 | 23.00 bits |

Under the logN14 bound of 424 bits, its key switches take 34 digits of 12 bits instead of 14, and its tensor key digits of 53 bits.

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

`ckks_ci_embed` and `ckks_ci_trace` take a replicated layout: each switches at the shaping depth under the scale, then rounds with shaping.
A conjugate-invariant ciphertext of degree `N` is bootstrapped through the standard ring of degree `2N` of its module, where the context is compiled, by composing them:

1. `ckks_ci_embed` embeds the input under its `GLWECIEmbedKey`.
2. `ckks_bootstrap` refreshes it, into a standard ciphertext one bit wider than the output.
3. `ckks_ci_trace` brings it back under its `GLWECITraceKey`, the trace spending the extra bit.

The key switches of the ring maps and the halving of the trace stay under the shaping depth, and the data slots keep the precision of the standard bootstrapping.
The two keys switch between the conjugate-invariant secret and the standard secret of the bootstrapping keys (`GLWECIKeyEncryptSk`), and cover the `k` of the ring maps at the shaping depth.
The context compiles for the standard ring, and the presets are sized for it.

## Construction outline

```rust,ignore
use poulpy_ckks::{
    api::CKKSBootstrappingOps,
    layouts::BootstrappingContextPrepared,
    presets::bootstrapping::n15_d20_k389_p21_s2c,
    shaping::ShapingUnit,
};

let preset = n15_d20_k389_p21_s2c(ShapingUnit::U32_45.layout(log_k)?)?;
let (plan, keys_layout) = (preset.plan(), preset.keys_layout());
// `scratch` covers `compile_unprepared_tmp_bytes(&module, plan)?` and the preparation of the unprepared context.
let ctx = BootstrappingContextPrepared::<MyBackend, f64>::compile(&module, base2k, plan, &mut scratch)?;
// `scratch` covers `keys_layout.generate_tmp_bytes(&module)` and `keys_layout.prepare_tmp_bytes(&module)`.
let keys = ctx
    .generate_keys(&module, &sk, keys_layout, &mut source_xs, &mut source_xe, &mut source_xa, &mut scratch)?
    .prepare(&module, &mut scratch)?;

// `input` holds `2^log_k` values at the leveled scale and `k = preset.input_k()`.
let mut output = module.ckks_ciphertext_alloc_from_infos(&preset.bootstrap_layout());
// `scratch` covers `module.ckks_bootstrap_tmp_bytes(&output, &input, &ctx, keys_layout)`.
module.ckks_bootstrap(&mut output, &input, &ctx, &keys, &mut scratch)?;
```

`BootstrappingPresetRun` in `poulpy_ckks::test_suite::presets` is a complete example, from key generation to the precision measurement.

## Validation

The CKKS test suite checks every shaped operation against the full layout on the data slots (`test_round_*`, `test_replicated_*`), every unit on a product chain, that operations which would round plainly reject a replicated input, and the bootstrapping on a small ring.
`test_replicated_parity` checks that the shaped operations give the same digits on every pair of backends, on both rings and at ranks 1 and 2, and the encryption parity suites check shaped encryption under the same draws.
`conjugate_invariant_ckks_test_suite!` runs the tests that apply to the conjugate-invariant ring, the shaped rounding on the unfolded coefficients, and `test_ckks_bootstrap_replicated_ci`, which composes the ring maps with the bootstrapping.
The ignored test `test_ckks_bootstrap_replicated_presets_with_client`, registered on the NTT backends, and the benchmark `bench_ckks_bootstrapping_with_client` of `poulpy-bench`, on the backends whose radix model takes the presets' 52 bits, run the rows of `test_suite::presets::replicated_presets`, every preset at `k = 16` and `64` and the logN16 preset with factored and dense maps at `k = 2^8` and `2^10`, and check each named row against the precision it advertises; `small_s2c_preset_meets_precision` runs an insecure replicated preset at `N = 2^11` in a default run.
