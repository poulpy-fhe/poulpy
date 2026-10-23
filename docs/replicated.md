# Replicated slot layout

`poulpy-ckks` implements the small-batch scheme of *Fully Homomorphic Encryption for Small Batches of Complex Numbers* (Dumezy, [ePrint 2026/2513](https://eprint.iacr.org/2026/2513)) as a slot layout of CKKS.
A batch of `k = 2^log_k` complex values is replicated across the ring, one slot class out of `R = 2^log_r` carrying data and the others serving as guards.
Every rounding, from encryption to rescaling and key switching, is shaped: its error is a multiple of a fixed polynomial `g` that is small on the data class and absorbs the error on the guards.
Precision on the data slots then no longer depends on the scale, so the scheme works at small scales (`2^20` levels) and keeps the modulus, and the keys, small.

## Layout

`SlotLayout` is part of the plaintext metadata of every CKKS plaintext and ciphertext (`GLWEPlaintextMeta::slot_layout`).
`SlotLayout::Full` is the usual CKKS layout; `SlotLayout::Replicated(ReplicatedLayout)` describes:

- `log_k`: the batch size `k = 2^log_k`. Slot `t` carries data iff `R | t`, holding value `(t/R) mod k`, so the batch repeats every `kR` slots;
- `log_r`: the period `R = 2^log_r` (`1 <= log_r <= 5`);
- `U ∈ Z[Y]/(Y^(2R)+1)`: the shaping polynomial, with `g = U(X^(N/(2R)))`. `U(1)` odd makes it invertible modulo every power of two;
- `w`: the extra working bits of a shaped rounding.

Slot class `j` sees `g` through `U(ω^(5^j))`, `ω = e^(iπ/(2R))`.
`ShapingProfile` (`poulpy_ckks::shaping`) reports these class magnitudes, the suppression on the data class, the amplification on the worst guard class and the largest exact rounding pass.
`ShapingUnit` lists cyclotomic units `U` with the smallest guard amplification for a prescribed suppression, solved exactly as integer programs: `U16_{20,40,60}` (`R = 8`) and `U32_{20,40,45,50,60}` (`R = 16`).
`ShapingUnit::layout(log_k, w)` builds the layout of a batch.

## Shaped operations

`CKKSRoundOps::ckks_round_into` rounds a ciphertext to a smaller width at a fixed torus value.
On a replicated layout, the `λ` dropped bits are divided out as `h = [U^(-1)·c]_(2^λ)` and the remainder `c − g·h` is carried, so the error is a multiple of `g` up to a remainder `2^-λ` smaller.
The other operations build on it:

- encoding quantizes at `log_delta + λ` bits and rounds with shaping; `ckks_encrypt_sk` encrypts `ShapingProfile::max_pass_bits` wider and rounds to the ciphertext width, so the fresh error is shaped;
- products (`ckks_mul_*`, `ckks_square_*`, `ckks_mul_pt_*`) and automorphisms (`ckks_rotate_*`, `ckks_conjugate_*`) are computed `ShapingProfile::shaping_depth(log_delta, w)` bits under the scale and rounded with shaping;
- constants are multiplied as masked plaintexts `c·π_S`, constant on the data slots and zero on the guards (`ckks_encode_masked_constant_into`, `MaskedCoeffs`); `ckks_{add,sub}_one_assign` add the masked one;
- polynomial evaluation and EvalMod take `MaskedCoeffs` (`encode_bsgs_replicated`, `compile_eval_mod_replicated`), and evaluate each baby step as one shaped linear combination.

Operations that would round plainly or write the guard slots return `InvalidOperand` on a replicated layout: constant additions, rotations by a slot count that is not a multiple of `R`, the full-layout linear transformations and DFT, ModUp, bootstrapping, PaCo and SHIP.

## Conversions

`CKKSReplicatedOps` moves a batch between CKKS and the replicated layout with the same keys.
`ckks_to_replicated_into` maps `k` values of a full-layout ciphertext with period `k` onto the data slots; `ckks_from_replicated_into` gathers them back, averaging the replicas.
`ReplicatedConversion` holds the encoded diagonals and lists the rotations its keys cover.

## Bootstrapping

`CKKSReplicatedBootstrapOps::ckks_replicated_bootstrap_into` refreshes a replicated ciphertext in `SlotsToCoeffs`-first order:

1. a normalized trace averages the `N/(2kR)` copies of the period;
2. `SlotsToCoeffs` gathers the data slots onto the coefficients of the subring of degree `2k`;
3. ModUp, with sparse-secret encapsulation;
4. a normalized trace onto period `k`, then `CoeffsToSlots`, with the real and imaginary parts side by side when they fit (packed);
5. EvalMod (cosine with double angles), on both parts at once when packed;
6. a shaped rounding back to the leveled scale.

`ReplicatedBootstrapPlan` holds the circuit (`ReplicatedBootstrapPlan::paper` is the configuration of the paper) and its widths (`k_in`, `k_boot`, `key_k`).
`ReplicatedBootstrapping` compiles it for a module: the transforms, the masked EvalMod constants, and the Galois elements its keys cover.
`ReplicatedBootstrapPlan::secure_key_layouts` sizes the gadgets of the keys with the fewest digits whose modulus fits the caller's bounds for the dense secret and for the ephemeral sparse secret.

### Presets

`poulpy_ckks::presets::replicated_bootstrapping` provides two parameter sets for 128-bit security, with dense secrets of weight 1024 and an ephemeral ModUp secret of weight 32.
The batch size and the shaping unit are the caller's; the moduli do not depend on them.
The sparse-to-dense key covers the 49-bit lifted ModUp input, with a row offset ending its gadget at the raised modulus. Its full modulus includes that offset, the gadget and the auxiliary bits.

| preset | ring | levels at `2^20` | keys (digits × width + aux) | ephemeral key | sparse-to-dense key (offset + gadget + aux) | min precision, k = 16 / 64 |
|---|---|---|---|---|---|---|
| `n14_d20_k109_p21_s2c` | `2^14` | 2 | 14 × 28 + 28 | 2 × 22 + 22 | 333 + 2 × 25 + 25 | 21.79 / 21.73 bits |
| `n15_d20_k389_p20_s2c` | `2^15` | 16 | 4 × 168 + 168 | 1 × 43 + 43 | 614 + 1 × 49 + 49 | 21.16 / 20.84 bits |

The logN14 preset advertises 21 bits of precision on the data slots and the logN15 preset 20 bits, which the precision test asserts. The test covers offset and full-width return keys on matched ciphertexts.

## Construction outline

```rust,ignore
use poulpy_ckks::{
    api::CKKSReplicatedBootstrapOps,
    layouts::ReplicatedBootstrapping,
    presets::replicated_bootstrapping::n15_d20_k389_p20_s2c,
    shaping::ShapingUnit,
};

let preset = n15_d20_k389_p20_s2c()?;
let layout = ShapingUnit::U32_45.layout(log_k, 24)?;
let ctx = ReplicatedBootstrapping::<MyBackend, f64>::new(&module, preset.plan(layout), base2k, &mut scratch)?;
let keys = ctx
    .generate_keys(&module, &sk, preset.keys_layout(), preset.sparse_secret_hamming_weight(),
                   &mut source_xs, &mut source_xe, &mut source_xa, &mut scratch)
    .prepare(&module, &mut scratch);

// `input` holds `2^log_k` values at the leveled scale and width `preset.input_k()`.
let mut output = module.ckks_ciphertext_alloc_from_infos(&preset.output_layout(layout));
module.ckks_replicated_bootstrap_into(&mut output, &input, &ctx, &keys, &mut scratch)?;
```

`ReplicatedBootstrappingPresetRun` in `poulpy_ckks::test_suite::replicated_presets` is a complete example, from key generation to the precision measurement.

## Validation

The CKKS test suite checks every shaped operation against the full layout on the data slots (`test_replicated_*`), that operations which would round plainly reject a replicated input, and the bootstrapping on a small ring.
`test_replicated_parity` checks that the shaped operations give the same digits on every pair of CPU backends.
The ignored test `replicated_bootstrapping_presets_meet_precision`, registered on the NTT backends, runs every preset at both batch sizes and asserts the advertised precision; the benchmark `bench_ckks_replicated_bootstrapping` of `poulpy-bench` runs the same configurations.
