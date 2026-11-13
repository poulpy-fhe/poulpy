# Encrypted Keccak-f[1600]

Keccak-f[1600], the permutation of SHA-3, on eight encrypted states at once with `poulpy-ckks` at 128-bit security.
The design is that of T. Seure and E. Suvanto, [Faster Homomorphic Evaluation of SHA3](https://fhe.org/conferences/conference-2026/resources/posters/[21]%20-%20Faster%20Homomorphic%20Evaluation%20of%20SHA3.pdf), FHE.org 2026 poster, which builds on the free XOR of C. Niu et al., [XBOOT: Free-XOR Gates for CKKS with Applications to Transciphering](https://doi.org/10.46586/tches.v2025.i4.118-144), TCHES 2025.

```sh
cargo run --release -p poulpy-keccak --bin permutation
```

The binary encrypts eight pseudorandom states in one ciphertext, runs the 24 rounds and checks every decrypted state against the plain permutation.
The client runs on `NTT4x30Portable`, and so does the server unless a feature picks its backend:

| Feature | Server backend |
|---|---|
| `portable-rayon` | `NTT4x30PortableRayon` |
| `neon` | `NTT4x30NeonRayon` |
| `avx` | `NTT4x30AvxRayon`, built with `RUSTFLAGS="-C target-feature=+avx2,+fma"` |
| `avx512` | `NTT4x30Avx512Rayon`, built with `RUSTFLAGS="-C target-feature=+avx2,+fma,+avx512f"` |
| `cuda`, `hip` | `NTT4x30Cuda`, `NTT4x30Hip` |
| `metal`, `vulkan` | `NTT4x30Wgpu` |

For example `cargo run --release -p poulpy-keccak --bin permutation --features neon`.

## Bits as integers

A bit is a slot holding 0 or 1.
XOR is addition and AND is the product, provided something takes the result back to its parity, since `a XOR b = (a + b) mod 2`.
Additions cost nothing, so a round accumulates integers, and one binary bootstrapping per round reduces them mod 2 while refreshing the modulus.

## The layout

A state is 25 lanes of 64 bits.
It takes 2048 positions, bit `z` of lane `(x, y)` at position `x + 5y + 32z`, the last 7 positions of each block of 32 empty.
Eight states interleave in the `2^14` slots of a ring of degree `2^15`, position `p` of state `s` at slot `8p + s`.
Rotating a lane by `r` bits is then rotating all slots by `8 * 32r`, and the moves of theta, pi and chi along `x` and `y` stay inside a block.
Every map of a round has the diagonals of a single state, spread by 8, so the eight states cost what one does.

## A round

| Step | On the slots | Modulus |
|---|---|---|
| theta, rho, pi | one linear map `L` of 264 diagonals | 20 bits |
| bootstrapping | `b = L c mod 2` | 125 bits in, 203 out |
| chi | `c = b - a2 + a1 a2`, `a1` and `a2` the reads of `b` at `x + 1` and `x + 2` | 20 bits for the reads, 35 for the product |
| iota | subtract the round constant | none |

A test in `src/layout.rs` runs this circuit on plain integers and checks it against the plain permutation.

## The bootstrapping

`src/bootstrapping.rs` builds it step by step rather than from a preset.
It is S2C-first, on all `2^14` slots:

1. SlotsToCoeffs moves the slots into the coefficients.
2. The raise reinterprets the ciphertext modulo a wide modulus, under an ephemeral sparse secret that keeps its integer part `I` small.
3. CoeffsToSlots brings the phase `(m/2 + I)/K` back to the slots, with `K = 16` as in the presets.
4. A cosine of period 2 in `m`, a degree-31 polynomial followed by 3 double angles, returns `[0, 1][m mod 2]` and ignores `I`.

The scale sets the precision: each bit of it halves the error of the output bits, which at `2^35` stay within about `2^-10` of 0 or 1.

## Parameters

| | |
|---|---|
| Ring degree | `2^15` |
| Dense secret | weight 1024, moduli up to 851 bits |
| Sparse secret | weight 32, moduli up to 162 bits |
| Bootstrapping input, output, raised | 125, 203 and 624 bits |
| A round | 20 bits for `L`, 20 for the shifts of chi, 35 for its product, 3 of margin |

The bounds are the 128-bit ones `poulpy-ckks` uses for its presets.
Every key takes the fewest digits that keep its modulus under the bound of its secret, see `key_layout` in `src/params.rs`.
