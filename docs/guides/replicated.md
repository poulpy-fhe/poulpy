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
- products (`ckks_mul_*`, `ckks_square_*`, `ckks_mul_pt_*`) and automorphisms (`ckks_rotate_*`, `ckks_conjugate_*`) are computed `ShapingProfile::shaping_depth(log_delta)` bits under the scale, `min(log_delta + 24, max_pass_bits)` (`SHAPED_WORKING_BITS`), and rounded with shaping;
- constants are multiplied as masked plaintexts `c·π_S`, constant on the data slots and zero on the guards (`ckks_encode_masked_constant_host_into`, `MaskedCoeffs`); `ckks_{add,sub}_one_assign` add the masked one;
- polynomial evaluation and EvalMod take `MaskedCoeffs` (`encode_bsgs_replicated`, `ckks_eval_mod_replicated_compile`), and evaluate each baby step as one shaped linear combination.

Sums and differences of one slot layout and one scale add exactly.
Operations that would round plainly or write the guard slots return `InvalidOperand` on a replicated layout: constant additions, rotations by a slot count that is not a multiple of `R`, dot products of prepared operands, inner sums, the full-layout linear transformations and DFT, ModUp, bootstrapping, PaCo and SHIP.

## Conjugate-invariant ring

The layout also runs on the conjugate-invariant ring of degree `N`, the subring of `Z[X]/(X^(2N)+1)` fixed by `X -> X^-1`, and its `N` real slots.
A batch of `k` real values takes `kR` slots, slot `t` in class `t mod R` as on the standard ring, and the period repeats `N/(kR)` times.
The shaping polynomial must be invariant too, `g = U(X^(N/R))` with `U` in `Z[Y + Y^-1]`.
Every unit of `Z[Y]/(Y^(2R)+1)` is a power of `Y` times an invariant unit with the same class magnitudes and norm, so this ring shapes with `Y^s*U` for the shift `s` that `ShapingProfile::<ConjugateInvariant>::try_new` finds (`ShapingProfile::shift`). The profile carries its ring in its type, so a profile of one ring cannot reach the shaped rounding of the other.
Every `ShapingUnit` has one, and a `U` without one is refused.
Rounding reads the invariant coefficients, which keeps `h` invariant and rounds each tie once.

Leveled operations, polynomial evaluation and EvalMod take the same calls as on the standard ring.
Masked constants and the masked one keep their real part, as the slot encoder does.
The conjugation, complex polynomials and the packed EvalMod are standard-only.

## Construction outline

```rust,ignore
use poulpy_ckks::{GLWEPlaintextMeta, Scale, SlotLayout, SlotsKind, shaping::ShapingUnit};

let layout = ShapingUnit::U32_45.layout(log_k)?;
let mut pt = module.ckks_plaintext_alloc_from_slots(module.ckks_max_slots(), base2k, k_pt);
pt.set_meta(GLWEPlaintextMeta {
    scale: Scale::Log(20),
    slots: SlotsKind::Complex,
    log_sparsity: 0,
    slot_layout: SlotLayout::Replicated(layout),
});
// `re` and `im` hold the `2^log_k` values of the batch.
module.ckks_encode_slots_host_into(&mut pt, &re, &im, &mut scratch)?;
module.ckks_encrypt_sk(&mut ct, &pt, &sk, &mut source_xe, &mut source_xa, &mut scratch)?;
```

## Validation

The CKKS test suite checks every shaped operation against the full layout on the data slots (`test_round_*`, `test_replicated_*`), every unit on a product chain, and that operations which would round plainly reject a replicated input.
`test_replicated_parity` checks that the shaped operations give the same digits on every pair of backends, on both rings and at ranks 1 and 2, and the encryption parity suites check shaped encryption under the same draws.
`conjugate_invariant_ckks_test_suite!` runs the tests that apply to the conjugate-invariant ring and the shaped rounding on the unfolded coefficients.
