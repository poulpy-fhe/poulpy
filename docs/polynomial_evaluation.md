# Polynomial Evaluation in Poulpy

This document describes how Poulpy evaluates a polynomial on an encrypted input.
It covers the global method, the two split strategies, the currently supported polynomial flavors, and a table of modulus consumption per degree.

## Overview

Given a polynomial `P` and a ciphertext encrypting a value `x`, polynomial evaluation produces a ciphertext encrypting `P(x)`.
The slots of `x` are evaluated independently, so the same `P` is applied to every slot at once.
Poulpy uses the Baby-Step Giant-Step decomposition, also known as the Paterson-Stockmeyer method, to keep both the number of multiplications and the multiplicative depth low.

The evaluation engine lives in `poulpy-core` and is scheme agnostic.
It operates on GLWE ciphertexts and the core arithmetic primitives, and it receives the per operation precision from the scheme through a small trait.
The CKKS layer in `poulpy-ckks` is a thin wrapper that owns the scale and budget accounting and supplies the encoded coefficients.

The method has three stages.
First the polynomial is decomposed into a set of low degree baby-step chunks together with a giant-step schedule.
Second a power basis of the input is built, which holds the precomputed powers of the input ciphertext.
Third each baby-step chunk is evaluated as a sum of one plaintext coefficient times one power, and the chunk results are recombined with the giant-step powers.

The baby-step chunks are evaluated with plaintext times ciphertext multiplications, which multiply a power by a scalar coefficient and accumulate.
The giant-step recombination is a binary tree of ciphertext times ciphertext multiplications, where each stage multiplies a partial result by a giant-step power and adds the next chunk.
The multiplicative depth, and therefore the modulus consumption, is set by the power basis generation and the giant-step recombination.

The power basis is the reusable left operand of the evaluation, and the encoded coefficients are the right operand.
A power basis built once can be shared across several polynomials evaluated on the same input.
Inside the giant-step recombination the shared giant-step power is prepared once as a right convolution operand and reused across the multiplications that consume it.

## Decomposition example

The decomposition splits the polynomial recursively at the smallest giant power above the midpoint of the remaining degree, until each remaining chunk has degree below the baby-step base.
The chunks that remain are the baby polynomials, and the giant powers are the splitting points.
The example below shows a degree fifteen polynomial with a baby-step base of four.

```text
P(x) = c0 + c1 x + ... + c15 x^15          degree 15, base 4

P(x) = L(x) + x^8 H(x)                      giant split at x^8
  L(x) = c0 + c1 x + ... + c7 x^7           degree 7
  H(x) = c8 + c9 x + ... + c15 x^7          degree 7

L(x) = l0(x) + x^4 l1(x)                    giant split at x^4
  l0(x) = c0 + c1 x + c2 x^2 + c3 x^3
  l1(x) = c4 + c5 x + c6 x^2 + c7 x^3

H(x) = h0(x) + x^4 h1(x)                    giant split at x^4
  h0(x) = c8  + c9 x  + c10 x^2 + c11 x^3
  h1(x) = c12 + c13 x + c14 x^2 + c15 x^3

baby polynomials : l0, l1, h0, h1          each of degree 3, which is below the base
power basis      : x, x^2, x^3 for the baby steps, x^4 and x^8 for the giant steps
baby steps       : evaluate each baby polynomial as a sum of coefficient times power
giant steps      : L = l0 + x^4 l1, then H = h0 + x^4 h1, then P = L + x^8 H
```

Each multiplication, whether in the power basis, the baby steps, or the giant steps, is followed by a rescale that consumes `log_delta` bits of modulus.
The modulus consumed by the whole evaluation is the longest chain of these rescales, which is the multiplicative depth.

## Split strategies

The decomposition is controlled by a split that fixes the baby-step base, which is the largest baby-step degree plus one.
Two strategies are available.

- `MinDepth` picks the split that minimizes the multiplicative depth. It uses a closed form choice of the split and tends to consume the least modulus.

- `MinMult` picks the split that minimizes the total number of multiplications, counted as the pair of ciphertext times ciphertext and plaintext times ciphertext multiplications. It sweeps the candidate splits and selects the best one.

The two strategies trade depth against multiplication count.
`MinMult` usually performs fewer multiplications and runs faster, while `MinDepth` usually consumes the same or less modulus.
The table below shows that the two strategies consume the same modulus for most degrees, and that `MinMult` consumes one extra multiple of `log_delta` only in a band just below each power of two.

## Supported polynomials

The following evaluation flavors are supported in the CKKS layer.

Real coefficient polynomials in the monomial basis, where the coefficients are real scalars and the result is `Sum_k c_k x^k`.

Real coefficient polynomials in the Chebyshev basis, where the result is `Sum_k c_k T_k(x)` and the power basis holds the Chebyshev powers `T_k(x)`.

Even and odd polynomials can optionally be folded before BSGS decomposition with `encode_bsgs_folded_with`.
In the monomial basis this evaluates the lower degree `Q(x squared)` or `x times Q(x squared)`; in the Chebyshev basis it evaluates `Q(T2(x))` or `x times Q(T2(x))`, where `T2(x) = 2 x squared - 1`.
The ordinary encoder already uses parity to skip zero coefficients and unnecessary baby powers, but it does not factor the polynomial or halve its encoded degree.

For a source degree `d`, let `m = floor(d / 2)` and let `D(n)` be the depth reported in the table below for degree `n` under the selected split strategy.
The folded depth is `D(m) + 1` for an even polynomial and `D(m) + 2` for an odd polynomial.
Consequently `MinDepth` folding is depth neutral for every even degree and adds one level for every odd degree; with `MinMult`, the table determines the degree bands where folding adds zero, one, or two levels.
Folding is therefore explicit rather than automatic.

Complex coefficient polynomials in both the monomial and Chebyshev bases, where each coefficient is `a_k + i b_k`.
The complex case reuses the real engine by evaluating the real part chunks and the imaginary part chunks, then combining them as `real_part + i times imaginary_part`.
The multiplication by the imaginary unit is the capacity free monomial map provided by the scheme, so the complex evaluation consumes the same modulus as the real one and runs only slightly slower.

Both a prepared entry point and a one-shot entry point are available.
The prepared entry point takes a power basis that the caller has already built, which allows reuse across several evaluations.
The one-shot entry point builds and populates the power basis internally from the input ciphertext.

Chebyshev interpolation of a real function is also provided, which produces the Chebyshev coefficients of the interpolating polynomial.

## Modulus consumption

The table reports the modulus consumed by one evaluation, expressed in multiples of `log_delta`.
Poulpy tracks a homomorphic budget that represents the available modulus in bits, and each multiplication followed by a rescale consumes `log_delta` bits of that budget.
A table entry of `n` therefore means the evaluation consumes `n` times `log_delta` bits of modulus.

For `MinDepth` the consumption equals `ceil(log2(degree + 1))` multiples of `log_delta`.
For `MinMult` the consumption equals the `MinDepth` value, with one extra multiple of `log_delta` in a band that sits just below each power of two and widens as the degree grows.

| Degree Range | MinDepth | MinMult |
| --- | --- | --- |
| 0 | 0 | 0 |
| 1 | 1 | 1 |
| 2 - 3 | 2 | 2 |
| 4 - 6 | 3 | 3 |
| 7 | 3 | 4 |
| 8 - 14 | 4 | 4 |
| 15 | 4 | 5 |
| 16 - 28 | 5 | 5 |
| 29 - 31 | 5 | 6 |
| 32 - 60 | 6 | 6 |
| 61 - 63 | 6 | 7 |
| 64 - 120 | 7 | 7 |
| 121 - 127 | 7 | 8 |
| 128 - 248 | 8 | 8 |
| 249 - 255 | 8 | 9 |
| 256 - 496 | 9 | 9 |
| 497 - 511 | 9 | 10 |
| 512 | 10 | 10 |

The consumption is the same for real and complex coefficients, since the imaginary combination is capacity free.
