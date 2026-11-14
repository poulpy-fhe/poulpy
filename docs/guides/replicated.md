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
- a general LUT takes `MaskedCoeffs` (`EncodedLut::general_replicated`), with its constants at the scale of the complex exponential it reads: on `x = I + m/p` from `2^36`, the exponential at `2^28` and the interpolation keep 26.5 bits of a table of 5 entries, against 14 to 15 on a full layout; the exponential's squarings keep the guards bounded, `|E| = 1` on every slot.

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

A batch of real slots (`SlotsKind::Real`), such as the conjugate-invariant bootstrapping feeds, leaves only multiples of the modulus on the imaginary coefficients, whose EvalMod is zero.
Unpacked (`ReplicatedBootstrapPlan::packed` off), `CoeffsToSlots` fills the `k` data slots alone and the real part takes one EvalMod, and the output keeps the real slots.
Packed, real slots take the packed route.
On one `NTT4x30Neon` thread at `k = 2^10`, a real batch bootstraps in 5.17 s unpacked against 5.55 s packed on `n16_d20_k771_p20_s2c_factored`, and in 2.70 s against 3.10 s for a complex batch on `n15_d20_k393_p20_s2c`, whose two batches do not fit the ring side by side. At `k = 2^8` on the logN15 preset the two routes take the same time.

`ReplicatedBootstrapPlan` holds the circuit (`ReplicatedBootstrapPlan::paper` is the configuration of the paper) and its widths (`k_in`, `k_boot`, `key_k`).
EvalMod enters its scale with a shaped rounding, which leaves noise of about `2^(G + log2(sqrt(N)))` units on the guard slots for a unit whose guards grow by `G` bits.
The cosine polynomial and the double angles square the guards, so they must stay well below one: at `2^28`, the guards of `U8_40` (`G = 14.58`) reach `2^7` after the polynomial and wrap the data after the double angles.
`ReplicatedBootstrapPlan::eval_mod_log_delta` keeps the paper's `2^28` for units whose guards grow by at most 13 bits, and runs EvalMod at `ceil(G) + 15` bits above that, `2^30` for `U8_40` and `2^31` for `U4_15`.
The shaped quantization of the last `CoeffsToSlots` factor also leaves an error on the guard slots at the EvalMod input, of about `2^(G - c2s_log_delta_diag)` times the input, summed over the factor's diagonals.
A guard above about 0.6 leaves the range of the cosine and wraps the whole ciphertext. With dense maps and `U8_40` at the paper's `2^34`, the worst guard reaches 0.26 at `k = 2^9` and 0.72 at `k = 2^10`, about 1.5 bits per doubling of `k`, each bit of `c2s_log_delta_diag` halving it.
The plain quantization of the `SlotsToCoeffs` diagonals leaks the input's guard noise onto the data: with `U8_40` at `k = 2^10`, dense maps measure 17.4 bits at `2^26`, 19.3 at `2^28` and 21.2 at `2^30`.
Its `transforms` select how `SlotsToCoeffs` and `CoeffsToSlots` are evaluated:

- `ReplicatedTransforms::Dense`, the paper's: each map is one BSGS transform of `k` (`SlotsToCoeffs`) or `2k` (packed `CoeffsToSlots`) diagonals, consuming one diagonal scale;
- `ReplicatedTransforms::Factored`: each map is the DFT on period `k` in `s2c_factors` or `c2s_factors` factors and the move between consecutive slots and the data slots in `placement_factors` factors of 0/1 masks, each factor a BSGS transform whose giant step is fitted to its offsets. Every `SlotsToCoeffs` factor consumes `s2c_log_delta_diag` bits; the last `CoeffsToSlots` factor is quantized with shaping at `c2s_log_delta_diag` bits and the others plainly at `c2s_inner_log_delta_diag` bits. Factored maps take at least `2^f` values for `f` factors, and a packed plan needs the two batches to fit the ring.
  With `c2s_first_factor_on_raise`, the first DFT factor of `CoeffsToSlots` runs on the modulus raise, as the standard bootstrapping's can: the raised ciphertext is small until its first key switch, so the factor consumes no modulus. Each term is rotated with no key and switched to the secret by its own first-factor key with a row offset, and these keys replace the sparse-to-dense key. The diagonals have period `k`, so the top trace onto period `k` commutes with the factor, and they are encoded at the widest scale of the fewest limbs holding `C2S_FIRST_FACTOR_MIN_LOG_DELTA` bits (`ReplicatedBootstrapPlan::c2s_first_factor_log_delta_diag`, 49 bits on one limb at radix 52), since every term of the raise carries a diagonal.

Factoring trades levels and rotation keys for fewer diagonals: at `N = 2^16` and `k = 2^10`, the dense maps multiply by about 3000 diagonals with 69 rotation keys, the factored maps of `n16_d20_k771_p20_s2c_factored` by 381 diagonals with 83 keys, for a raised modulus 256 bits wider at the same levels.
At `k = 2^10`, the factored preset bootstraps in 6.3 s on one `NTT4x30Neon` thread and 0.36 s on Metal, against 8.1 s and 0.52 s for dense maps at the same levels, within 0.2 bits of their precision. At `k = 2^8` the dense maps are faster, the diagonals no longer dominating.
The preset now leaves its first `CoeffsToSlots` factor to the raise, which narrows the raised modulus by the factor's scale and takes about 5% off a bootstrapping on both backends, the raise's 63 small switches costing most of the factor they replace, at 21.6 bits instead of 22.3.
`ReplicatedBootstrapping` compiles it for a module: the transforms, the masked EvalMod constants, and the Galois elements its keys cover.
`ReplicatedBootstrapping::key_widths` gives the width at which a bootstrapping into `k_out` bits uses each key: the keys of the bottom trace and `SlotsToCoeffs` at the input width, those of the top trace and `CoeffsToSlots` at the raised width, each giant rotation one factor narrower, and the tensor key at the widest EvalMod product.
The bootstrapping looks every key up at its width before any work.
A preset shapes each key from its width with the shared `BootstrappingKeyPolicy` of the standard presets: the whole-limb digit of fewest rows under the modulus bound of the secret, or, when no whole limb fits that bound, the digit of fewest rows of any width.

### Functional bootstrapping

`ReplicatedBootstrapPlan::with_functional_bootstrap(&lut)` turns the bootstrapping into a lookup table of `p` entries on integer messages: `SlotsToCoeffs` scales the messages by `2^r/p`, `r = ceil(log2(p))` the message ratio, so that the coefficients carry the phase `m/p`, and the plan runs unpacked.
The EvalMod plan is the complex exponential, `EvalModPlan::complex_exponential`, at the scale `eval_mod_log_delta` of the unit.
`EncodedLut::general_replicated` encodes the LUT's constants as masked plaintexts at `ReplicatedBootstrapPlan::functional_lut_log_delta(p)`, the raised scale, at which the exponential returns, and the widths count the LUT's budget there.
`CKKSReplicatedBootstrapOps::ckks_replicated_functional_bootstrap` takes one output per LUT, all of one width, and writes `table[m mod p]` at the leveled scale.
It runs the stages up to `CoeffsToSlots` once, then the exponential and each LUT's interpolation `T + conj(T)` on the real part, and for complex slots on the imaginary part too, recombined; two or more LUTs share the exponential and its powers.
A binary LUT (`EncodedLut::binary_replicated`, from the plan) skips the exponential: each part is rounded with shaping to the EvalMod scale, at which its constants are encoded, and the cosine reads it as `x/K` for the interval `K` of the EvalMod plan, then the double-angle chain adds and subtracts its constants as masked plaintexts; the plan's widths leave EvalMod out.
On a test ring at `k = 8` with `U16_40`, a general table of 5 entries keeps 26.8 bits on real and complex slots, alone or beside a second LUT, a table of values up to 16 keeps 24.7 bits, and binary tables keep 27.5 and 28.5 bits.

### Presets

`poulpy_ckks::presets::replicated_bootstrapping` provides three parameter sets for 128-bit security, with dense secrets of weight 1024 and an ephemeral ModUp secret of weight 32.
The batch size and the shaping unit are the caller's.
The input and output widths do not depend on them, while the raised modulus and the keys follow the unit's EvalMod scale: `plan` and `keys_layout` take the layout, and `keys_layout` the widths of the compiled bootstrapping.
A preset refuses a unit whose suppression is less than 8 bits above its precision, since the bootstrapping falls short of the suppression by up to 5 bits: the units of `R = 2` and those targeting 20 bits, which measure 16 to 19 bits on the logN14 preset.
The logN14 and logN15 presets take dense maps, with `SlotsToCoeffs` diagonals at `2^30` and the shaped `CoeffsToSlots` ones at `2^36` and `2^37`, which keep the guards of `U8_40` under 0.16 at `k = 2^10`.
The logN16 preset factors its maps into two DFT factors and two placement factors each, with `SlotsToCoeffs` diagonals at `2^32` and inner `CoeffsToSlots` diagonals at `2^49`, one limb each at radix 52, for batches of 4 to `2^10` values at `R = 16`.
At `2^49` instead of `2^54` a bootstrapping is about 5% faster on Metal and 2% on NEON for 0.05 bits; at `2^44` the precision falls to 17 bits.
Its first `CoeffsToSlots` factor runs on the raise, with first-factor keys in place of the sparse-to-dense key; the dense presets' sparse-to-dense key covers the 49-bit lifted ModUp input, with a row offset ending its gadget at the raised modulus.

Each preset advertises 20 bits of precision on the data slots.
The tables give the gadgets as digits × digit width + auxiliary bits, and the least precision over two key and secret seeds at the largest batch, on `NTT4x30Neon`:

| preset | ring | levels at `2^20` | raised modulus | widest rotation key | tensor key | raise key (offset + gadget) | min precision |
|---|---|---|---|---|---|---|---|
| `n14_d20_k113_p20_s2c` | `2^14` | 2 | 389 bits | 19 × 21 + 25 | 7 × 52 + 60 | 339 + 2 × 25 + 35 | 22.91 bits, `k = 2^9` |
| `n15_d20_k393_p20_s2c` | `2^15` | 16 | 670 bits | 7 × 104 + 119 | 3 × 208 + 223 | 618 + 1 × 52 + 67 | 22.87 bits, `k = 2^10` |
| `n16_d20_k771_p20_s2c_factored` | `2^16` | 30 | 1143 bits | 3 × 416 + 432 | 2 × 520 + 536 | first factor, 1 × 156 + 172 | 21.65 bits, `k = 2^10` |

With `U8_40`, EvalMod at `2^30` raises the modulus by 16 bits:

| preset | raised modulus | widest rotation key | tensor key | min precision, `k = 2^10` |
|---|---|---|---|---|
| `n14_d20_k113_p20_s2c` | 405 bits | 46 × 9 + 10 | 7 × 52 + 60 | 21.25 bits |
| `n15_d20_k393_p20_s2c` | 686 bits | 7 × 104 + 119 | 7 × 104 + 119 | 21.19 bits |
| `n16_d20_k771_p20_s2c_factored` | 1159 bits | 3 × 416 + 432 | 2 × 520 + 536 | 21.41 bits |

The logN14 bound of 424 bits leaves no whole-limb digit to its rotation keys, so they take 19 digits of 21 bits, and 46 of 9 bits with `U8_40`.
The precision test asserts the advertised bits with `U32_45` at `k = 16` and `k = 64`, and with `U8_40` at `k = 64`.

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
    presets::replicated_bootstrapping::n15_d20_k393_p20_s2c,
    shaping::ShapingUnit,
};

let preset = n15_d20_k393_p20_s2c()?;
let layout = ShapingUnit::U32_45.layout(log_k)?;
let plan = preset.plan(layout)?;
// `scratch` covers `ReplicatedBootstrapping::<MyBackend, f64>::new_tmp_bytes(&module, &plan, base2k)?`.
let ctx = ReplicatedBootstrapping::<MyBackend, f64>::new(&module, plan, base2k, &mut scratch)?;
let keys = ctx
    .generate_keys(&module, &sk, &preset.keys_layout(layout, &ctx.key_widths(&module, preset.output_k()))?, preset.sparse_secret_hamming_weight(),
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
