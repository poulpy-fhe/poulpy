# Specification: `mod1` for poulpy

Port of `lattigo/he/hefloat/mod1_evaluator.go` to `poulpy-ckks`. The spec mirrors the lattigo design and consumes the existing `Polynomial` / `BSGSPolynomial` evaluator in `poulpy-ckks/src/polynomial.rs`.

## 1. Purpose

Homomorphically evaluate `x mod 1` on a CKKS ciphertext. Used in bootstrapping to peel off the integer part `I(X)` from the message:

```
Δ·(Q/Δ·I + m)  →  Δ·(I + (Δ/Q)·m)  →  Δ·((Δ/Q)·m)   (x mod 1)
                                    →  Δ·m            (re-scale by Q/Δ)
```

Because `Q` is not a power of two and `Δ` is, the procedure approximates division by `2^round(log Q)`; the residual factor `qDiff = Q / 2^round(log Q)` is folded into the polynomial. The caller must pre-scale the input by `1/K` where `K` is the approximation range.

## 2. Approximation strategies (`Mod1Type`)

Three variants, identical to lattigo:

| Variant         | Function approximated                                  | Notes                                                              |
| --------------- | ------------------------------------------------------ | ------------------------------------------------------------------ |
| `CosDiscrete`   | `(1/2π)^(1/2^r) · cos(2π(x-1/4)/2^r)` via Han–Ki      | requires `degree ≥ 2(K-1)`; uses `cosine::approximate_cos`         |
| `CosContinuous` | same as above via standard Chebyshev interpolation     | works on full interval `[-K, K]`                                   |
| `SinContinuous` | `(1/2π) · sin(2πx)` via Chebyshev                      | no double-angle, no `r`                                            |

When `Mod1Type` ∈ {Cos\*}, after Chebyshev evaluation the algorithm applies `r = DoubleAngle` iterations of `cos(2θ) = 2cos²(θ) − 1`, each consuming one level (mul + rescale).

Optional arcsine correction: a Taylor polynomial `Mod1InvPoly` of odd degree `Mod1InvDegree`, evaluated after the double-angle steps, recovers `m` from `sin(2π·(Δ/Q)·m)` when `m/Q` is non-negligible.

## 3. Parameters

```rust
pub enum Mod1Type { CosDiscrete, SinContinuous, CosContinuous }

pub struct Mod1ParametersLiteral {
    pub log_scale: usize,           // log2 of the scaling factor used during mod1
    pub mod1_type: Mod1Type,
    pub scaling: f64,               // optional extra output scaling (1.0 if 0)
    pub log_message_ratio: usize,   // log2(Q0 / |m|)
    pub mod1_degree: usize,         // Chebyshev degree of f
    pub mod1_interval: usize,       // K (integer half-interval)
    pub double_angle: usize,        // r; ignored for SinContinuous
    pub mod1_inv_degree: usize,     // 0 disables arcsine
}

pub struct Mod1Parameters {
    log_default_scale: usize,
    mod1_type: Mod1Type,
    log_message_ratio: usize,
    double_angle: usize,            // 0 if SinContinuous
    q_diff: f64,                    // Q0 / 2^round(log2 Q0)
    sqrt_2pi: f64,                  // (1/2π)^(1/scale_factor) or 1 if inv used
    mod1_poly: Polynomial<f64>,     // Chebyshev, parity preset
    mod1_inv_poly: Option<Polynomial<f64>>,  // Monomial, odd, optional
}
```

Constructor `Mod1Parameters::from_literal(params, lit)` performs, matching `mod1_parameters.go:117`:

1. Validate: `SinContinuous ⇒ double_angle = 0`; `CosDiscrete ⇒ mod1_degree ≥ 2(K-1)`.
2. `K = mod1_interval / 2^double_angle`, `Q0 = params.q()[0]`, `q_diff = Q0 / 2^round(log2 Q0)`.
3. If `mod1_inv_degree > 0`, build the odd arcsine Taylor series with `c₁ = 1/(2π)·q_diff·scaling`, `c_{i+2} = c_i · (i²-4i+4)/(i²-i)` for odd `i`; set `sqrt_2pi = 1`.
   Otherwise `sqrt_2pi = (1/(2π) · q_diff · scaling)^(1/2^double_angle)`.
4. Build `mod1_poly` via:
   - `SinContinuous`: `Polynomial::chebyshev_interpolate(degree, -K, K, |x| sin(2πx))`, parity `Odd`.
   - `CosContinuous`: same with `cos(2πx)`, parity `Even`.
   - `CosDiscrete`: `cosine::approximate_cos(K, degree, message_ratio, double_angle)` (new module, see §6), parity `Even`.
5. Multiply every coefficient of `mod1_poly` by `sqrt_2pi`.

## 4. Algorithm — `Mod1Evaluator::evaluate`

Pre-condition: input ciphertext is at level `level_q`, already scaled by `1/(K · q_diff)`, with scale set so that `m/Q ∈ [-K, K] / Δ` after normalization.

Mirrors `mod1_evaluator.go:50`:

```text
1.  out = ct.clone()
2.  ensure out.level ≥ level_q; drop levels if higher
3.  out.scale = ScalingFactor()                       # mark as mod-1 normalized
4.  compute target_scale through r double-angle steps (sqrt-walk over Q[i])
5.  if Mod1Type ∈ {CosDiscrete, CosContinuous}:
        offset = -0.5 / ((B-A) · 2^double_angle)
        out += offset                                  # change-of-basis shift for Chebyshev
6.  evaluate Chebyshev polynomial:
        out = poly_eval(out, mod1_poly, target_scale)
        rescale(out)
7.  repeat r = double_angle times:                    # cos(2θ) = 2cos²θ − 1
        sqrt_2pi *= sqrt_2pi
        out = mul_relin(out, out)
        out += out                                     # ×2
        out += -sqrt_2pi                               # −1·(sqrt_2pi)
        rescale(out)
8.  if has Mod1InvPoly:
        out = poly_eval(out, mod1_inv_poly, out.scale)
        rescale(out)
9.  out.scale = ct.scale                              # restore semantic scale
10. if affine b ≠ 0: out += b
```

Step 4 (target-scale walk) chains, for `i = 0..r-1`:
`target = sqrt(target · Q[ levelQ - depth(poly) - r + i + 1 ])`.
This pre-conditions the polynomial output so that after `r` rescales the ciphertext lands at `ScalingFactor()` exactly.

`EvaluateWithAffineTransformation(ct, a, b)` is the general entry; when `mod1_inv_poly` is `None`, `a` is folded into `mod1_poly` via `a^(1/2^r)` and into `sqrt_2pi`; otherwise `a` is folded into `mod1_inv_poly`.

## 5. Proposed Rust API

In `poulpy-ckks/src/mod1.rs`:

```rust
pub struct Mod1Evaluator<'a, BE: Backend, P> {
    pub poly_eval: &'a P,                 // anything PolynomialEvaluation<BE>
    pub params: Mod1Parameters,
    // pre-encoded BSGS polynomials, allocated once at construction
    pub mod1_bsgs: BSGSPolynomial<CKKSPlaintext<Vec<u8>>>,
    pub mod1_inv_bsgs: Option<BSGSPolynomial<CKKSPlaintext<Vec<u8>>>>,
}

impl<'a, BE, P> Mod1Evaluator<'a, BE, P>
where
    BE: Backend,
    P: PolynomialEvaluation<BE> + /* ckks add / mul_relin / rescale */ ,
{
    pub fn new(module, base2k, coeff_meta, params, poly_eval) -> Result<Self>;

    pub fn evaluate(
        &self,
        out: &mut CKKSCiphertext<…>,
        ct:  &CKKSCiphertext<…>,
        tsk: &GLWETensorKeyPrepared,
        scratch: &mut ScratchArena<BE>,
    ) -> Result<()>;

    pub fn evaluate_affine(
        &self, out, ct, a: f64, b: f64, tsk, scratch,
    ) -> Result<()>;
}
```

Notes:

- BSGS encoding happens once in `new`, not per call (lattigo re-encodes on every call; we should avoid that). The optional `a` for affine forces a re-encode of `mod1_poly` / `mod1_inv_poly`, so expose `evaluate_affine_owned(a, b, …)` only as a slower path.
- Reuse `Polynomial::chebyshev_interpolate` + `Polynomial::encode_bsgs` (already in `polynomial.rs`). No new evaluator core.
- Scale bookkeeping (steps 3, 4, 9) uses `SetCKKSInfos` — set `log_delta` on `out` directly; do not invent a `Scale` wrapper.
- Double-angle loop (step 7) sits on top of existing `mul_relin`, `rescale`, `add` (and a scalar-add). All already exist in `poulpy-ckks/src/api/`.

## 6. New supporting module: `poulpy-ckks/src/cosine.rs`

Port of `lattigo/he/hefloat/cosine/cosine_approx.go`. Single public entry point:

```rust
pub fn approximate_cos(k: usize, degree: usize, dev: f64, sc_num: usize) -> Vec<f64>;
```

Returns Chebyshev coefficients in `[-K, K]` for `cos(2π·x/2^sc_num)` via Han–Ki linear-system solve. Faithful port — use `rug` / `num-bigfloat` for the 256-bit big-float work, or `f64` if precision is sufficient at the supported parameter ranges (lattigo uses 256-bit `big.Float` because at very large degree / `K` the conditioning becomes nasty; quantify before downgrading).

## 7. Tests (port of `mod1_evaluator_test.go`)

Three test cases, identical literals:

1. `SinContinuous` + arcsine, `degree=127`, `K=14`, `inv=7`, `log_scale=60`.
2. `CosDiscrete`, `degree=30`, `K=12`, `r=3`, `log_scale=60`.
3. `CosContinuous`, `degree=177`, `K=325`, `r=4`, `log_message_ratio=4`, `log_scale=60`.

Test harness (`evaluateMod1` in `mod1_evaluator_test.go:117`):

- Sample `values[i] = round(U(-K,K)) · Q + U(-1,1)`, plus `values[0] = K·Q + 0.5` boundary case.
- Scale plaintext to `Δ = Q / MessageRatio`, then to `Sine / MessageRatio`, then multiply by `1/(K·q_diff)` and rescale.
- Call `evaluate`.
- Plaintext oracle: `x → sin(2π·x/(MessageRatio·q_diff)) · MessageRatio·q_diff / (2π)` (with `asin` inserted if inv enabled).
- Compare via `VerifyTestVectors` at `LogDefaultScale`.

## 8. Out of scope

- Bootstrapping driver (combines mod1 with C2S / S2C) — separate spec.
- Marshalling of `Mod1Parameters` — only needed if we serialize bootstrapping keys; defer.
- Non-real (complex) coefficient paths — lattigo carries `bignum.Complex` throughout; mod1 only ever needs the real part, so keep `Polynomial<f64>`.

---

Suggested file layout:

```
poulpy-ckks/src/
  cosine.rs          # §6
  mod1.rs            # §3–§5
  lib.rs             # +mod cosine; mod mod1;
poulpy-ckks/tests/
  mod1.rs            # §7
```

No changes needed to `polynomial.rs`, the polynomial-evaluation OEP, or any existing CKKS arithmetic API.
