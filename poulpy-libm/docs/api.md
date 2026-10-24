# API

`poulpy-libm` exposes CKKS operations on `Module<BE>`. Function names follow `libm`; operation methods add the `ckks_` prefix.

## Conventions

Let:

- `D` be the input `log_delta`;
- `P(D)` and `p` be a plan's `consumed_bits(D)` and `depth()`;
- `S(D)` and `s` be a `SignComposite`'s `consumed_bits(D)` and `depth()`.

Consumed modulus is the expected `log_budget` drop.
Depth is the longest multiplicative chain.
A plan's methods are authoritative when its degree, interval, or reduction schedule changes.
The input budget must be strictly larger than the stated consumption.

Direct unary plans expose `interval()`, `degree()`, `consumed_bits(D)`, `depth()`, and `map_plaintexts(...)`.
Composed plans expose their declared domains as fields and provide the same cost and mapping methods.
The `approximation_bits` field is fitted host-polynomial accuracy, not a guarantee on the final ciphertext.

Build plans on the host with `from_precision`, transfer their plaintexts with `map_plaintexts`, size scratch with the matching `ckks_*_tmp_bytes` method, and then evaluate.
No operation bootstraps.

## Precision tuning

`target_bits` controls the fitted polynomial error, while `max_degree` limits its cost.
`coeff_meta.log_delta` controls the scale of prepared coefficients independently from the input ciphertext scale.
Plans used in the same circuit may use different coefficient scales.
The input ciphertext scale remains a Poulpy encoding parameter and determines the cost of ciphertext multiplications.

Increasing the target may select a higher degree, and increasing the coefficient scale may increase the modulus consumed by plaintext multiplications.
Degree, depth, and modulus consumption change discretely, so a larger target does not always change the circuit.
A smaller coefficient scale only saves modulus when a plaintext multiplication lies on the critical path.
The tabulated cost formulas assume `coeff_meta.log_delta <= log_delta`; a larger coefficient scale can consume additional modulus on plaintext multiplications.
Use the constructed plan's `degree()`, `depth()`, and `consumed_bits(...)` methods as the authoritative cost values.

`reduction_steps` controls exponential range reduction.
`reciprocal_iters` controls reciprocal refinement in `softmax` and `atan2`.
Public iteration counts control reciprocal, division, square-root, and normalization circuits.
These parameters allow each stage to trade accuracy for modulus independently.

## Bootstrapping handoff

Libm plans do not bootstrap implicitly. Bootstrap first, then size and evaluate the
next libm operation from the refreshed ciphertext's actual metadata:

```text
D_out    = refreshed.log_delta()
required = libm_plan.consumed_bits(D_out)
refreshed.log_budget() > required
```

The strict inequality is the same precondition used by approximation evaluation.
Allocate the libm destination at the full circuit/bootstrap modulus; reusing a
destination whose previous result narrowed its precision metadata can add a
destination-capacity alignment cost.

When the bootstrap modulus is selected together with the following libm stage,
reserve that stage before rounding to Poulpy's limb width. For the standard
pipeline this means choosing a modulus satisfying

```text
k_boot > log_modulus_in
       + bootstrap_plan.consumed_bits()
       + libm_plan.consumed_bits(D_out)
```

Add any desired post-libm headroom before rounding up. `D_out` is recipe-dependent;
use the bootstrap output metadata rather than the pre-bootstrap ciphertext scale.
The plan's fitted `target_bits` is independent of bootstrapping, while its
coefficient scale and polynomial depth determine how much of the refreshed budget
the subsequent operation consumes.

## Arithmetic

The formulas below assume compatible operands at scale `D`.

| Function | Operation | Domain | Modulus | Depth |
|---|---|---|---:|---:|
| `fma` | `ckks_fma(a, b, c)` | Values fitting the CKKS capacity | `D` | `1` |
| `scalbn` | `ckks_scalbn(input, n)` | Public `i32` exponent | `max(-n, 0)` bits | `0` |
| `ldexp` | `ckks_ldexp(input, n)` | Alias of `scalbn` | `max(-n, 0)` bits | `0` |

Positive power-of-two scaling consumes headroom but no modulus.
Negative scaling consumes one modulus bit per divided power of two.

## Exponential and logarithmic

| Function | Operation and plan | Prepared interval | Modulus | Depth |
|---|---|---|---:|---:|
| `exp` | `ckks_exp`, `ExpPlan` | Any finite `[a, b]` | `P(D)` | `p` |
| `exp2` | `ckks_exp2`, `Exp2Plan` | Any finite `[a, b]` | `P(D)` | `p` |
| `exp10` | `ckks_exp10`, `Exp10Plan` | Any finite `[a, b]` | `P(D)` | `p` |
| `expm1` | `ckks_expm1`, `Expm1Plan` | Any finite `[a, b]` | `P(D)` | `p` |
| `log` | `ckks_log`, `LogPlan` | `0 < a < b` | `P(D)` | `p` |
| `log2` | `ckks_log2`, `Log2Plan` | `0 < a < b` | `P(D)` | `p` |
| `log10` | `ckks_log10`, `Log10Plan` | `0 < a < b` | `P(D)` | `p` |
| `log1p` | `ckks_log1p`, `Log1pPlan` | `-1 < a < b` | `P(D)` | `p` |

The exponential plans fit a reduced interval, then apply a fixed number of squarings.
Their cost is `approximation.consumed_bits(D) + reduction_steps * D`; their depth is the polynomial depth plus `reduction_steps`.
Logarithms use one fixed-domain polynomial and do not perform automatic range reduction.

## Powers and roots

| Function | Operation and plan | Domain | Modulus | Depth |
|---|---|---|---:|---:|
| `pow` | `ckks_pow`, `PowPlan` | Positive prepared base interval; prepared exponent interval | `plan.consumed_bits(Db, De)` | `plan.depth()` |
| `powi` | `ckks_powi(input, e)` | Public `u32` exponent | `M(e) * D` | `M(e)` |
| `sqrt` | `ckks_sqrt(x, in_half, r)` | Positive and conditioned near `1` | `(2r + 1)D` | `2r + 1` |
| `cbrt` | `ckks_cbrt`, `CbrtPlan` | Any finite prepared `[a, b]` | `P(D)` | `p` |
| `hypot` | `ckks_hypot`, `HypotPlan` | `x_interval` and `y_interval` | `plan.consumed_bits(D)` | `plan.depth()` |

For `pow`, with log plan `L` and exponential plan `E`, the cost is `L(Db) + max(Db, De) + E(min(Db, De))`; the depth is `l + 1 + e`.

For `powi`, `M(0) = M(1) = 0`; otherwise `M(e) = floor(log2(e)) + popcount(e) - 1`.
Negative integer exponents use a reciprocal explicitly; `ckks_powi` accepts only non-negative exponents.

`ckks_sqrt` uses `x` as the initial inverse-square-root estimate.
`in_half` must encode `x / 2` at the same scale and at least the same budget.
Self-seeding converges only for `x ∈ (0, ∛3) ≈ (0, 1.442)`; larger inputs make the Newton iterate diverge, so condition them toward `1` first (public power-of-two scaling or normalization) or drive `ckks_rsqrt` directly with a hand-picked seed.
There is no runtime guard on this bound.

## Trigonometric

| Function | Operation and plan | Prepared interval | Modulus | Depth |
|---|---|---|---:|---:|
| `sin` | `ckks_sin`, `SinPlan` | Any finite `[a, b]` | `P(D)` | `p` |
| `cos` | `ckks_cos`, `CosPlan` | Any finite `[a, b]` | `P(D)` | `p` |
| `sincos` | `ckks_sincos`, `SinPlan` + `CosPlan` | Intersection of both intervals | Per output plan | Maximum output depth |
| `tan` | `ckks_tan`, `TanPlan` | `-pi/2 < a < b < pi/2` | `P(D)` | `p` |
| `asin` | `ckks_asin`, `AsinPlan` | `-1 <= a < b <= 1` | `P(D)` | `p` |
| `acos` | `ckks_acos`, `AcosPlan` | `-1 <= a < b <= 1` | `P(D)` | `p` |
| `atan` | `ckks_atan`, `AtanPlan` | Any finite `[a, b]` | `P(D)` | `p` |
| `atan2` | `ckks_atan2(y, x)`, `Atan2Plan` | Away from both axes; see below | `plan.consumed_bits(D)` | `plan.depth()` |

`sin` and `cos` do not perform periodic reduction.
`tan` covers one continuous branch.

For `atan2`, the plan requires `plan.sign_gap() <= |x|, |y| <= plan.input_bound` and `|y / x| <= plan.ratio_bound`.
With reciprocal iterations `r`, atan cost `A(D)`, sign cost `S(D)`, and optional normalization cost `N`, the total is `N + S(D) + (r + 4)D + A(D)`.
The normalization cost is `plan.coeff_log_delta` when enabled and zero otherwise.

## Hyperbolic and error functions

| Function | Operation and plan | Prepared interval | Modulus | Depth |
|---|---|---|---:|---:|
| `sinh` | `ckks_sinh`, `SinhPlan` | Any finite `[a, b]` | `P(D)` | `p` |
| `cosh` | `ckks_cosh`, `CoshPlan` | Any finite `[a, b]` | `P(D)` | `p` |
| `tanh` | `ckks_tanh`, `TanhPlan` | Any finite `[a, b]` | `P(D)` | `p` |
| `asinh` | `ckks_asinh`, `AsinhPlan` | Any finite `[a, b]` | `P(D)` | `p` |
| `acosh` | `ckks_acosh`, `AcoshPlan` | `1 <= a < b` | `P(D)` | `p` |
| `atanh` | `ckks_atanh`, `AtanhPlan` | `-1 < a < b < 1` | `P(D)` | `p` |
| `erf` | `ckks_erf`, `ErfPlan` | Any finite `[a, b]` | `P(D)` | `p` |
| `erfc` | `ckks_erfc`, `ErfcPlan` | Any finite `[a, b]` | `P(D)` | `p` |

## Gamma and Bessel functions

| Function | Operation and plan | Prepared interval | Modulus | Depth |
|---|---|---|---:|---:|
| `tgamma` | `ckks_tgamma`, `TgammaPlan` | `0 < a < b` | `P(D)` | `p` |
| `lgamma` | `ckks_lgamma`, `LgammaPlan` | `0 < a < b` | `P(D)` | `p` |
| `j0` | `ckks_j0`, `J0Plan` | Any finite `[a, b]` | `P(D)` | `p` |
| `j1` | `ckks_j1`, `J1Plan` | Any finite `[a, b]` | `P(D)` | `p` |
| `jn` | `ckks_jn`, `JnPlan` | Any finite `[a, b]`; public `i32` order | `P(D)` | `p` |
| `y0` | `ckks_y0`, `Y0Plan` | `0 < a < b` | `P(D)` | `p` |
| `y1` | `ckks_y1`, `Y1Plan` | `0 < a < b` | `P(D)` | `p` |
| `yn` | `ckks_yn`, `YnPlan` | `0 < a < b`; public `i32` order | `P(D)` | `p` |

Gamma is deliberately restricted to a positive continuous interval.
The implementation does not cross poles or emulate `libm`'s IEEE edge behavior.

## Comparison

These operations use a smooth composite approximation of `sign`.
Every value sent to that approximation must lie in `[-1, 1]` and outside its fitted gap.
They do not reproduce NaN, signed-zero, or exact-equality semantics.

| Function | Operation | Sign input | Modulus | Depth |
|---|---|---|---:|---:|
| `fabs` | `ckks_fabs(x)` | `x` | `S(D) + D` | `s + 1` |
| `fdim` | `ckks_fdim(a, b)` | `a - b` | `S(D) + D + 1` | `s + 1` |
| `copysign` | `ckks_copysign(x, y)` | `x` and `y` | `S(D) + 2D` | `s + 2` |
| `fmax` | `ckks_fmax(a, b)` | `a - b` | `S(D) + D + 1` | `s + 1` |
| `fmin` | `ckks_fmin(a, b)` | `a - b` | `S(D) + D + 1` | `s + 1` |

`ckks_fmax_const` and `ckks_fmin_const` have the same cost and depth as their two-ciphertext forms.
The ciphertext and plaintext bound must use compatible scales.

## CKKS support operations

These operations have no direct `libm` equivalent but complete the public function layer.

### Reciprocal and normalization

| Operation | Domain | Modulus | Depth |
|---|---|---:|---:|
| `ckks_goldschmidt_division` | `0 < x < 2` | `(r + 1)D` | `r + 1` |
| `ckks_inverse_positive_domain` | Alias of Goldschmidt | `(r + 1)D` | `r + 1` |
| `ckks_inverse_negative_domain` | `-2 < x < 0` | `(r + 1)D` | `r + 1` |
| `ckks_inverse_full_domain` | `[-1, -tau] ∪ [tau, 1]` | `S(D) + (r + 3)D` | `s + r + 3` |
| `ckks_rsqrt` | Positive and conditioned near `1` | `2rD` | `2r` |
| `ckks_div(a, b)` | `0 < b < 2` | `(r + 2)D` | `r + 2` |
| `ckks_interval_normalization` | Prepared `[-max, max]` | `2nD` | `2n` |

Here `r` is the fixed iteration count, `tau` is the sign-composite gap, and `n` is `IntervalNorm::n`.
`ckks_rsqrt` refines the estimate held in `y`; it converges to `1/sqrt(x)` only when the seed satisfies the Newton basin `0 < x·y² < 3`, and there is no runtime guard.
Interval normalization writes both the compressed value and its per-slot factor; it is not applied implicitly.

### Predicates and selection

| Operation | Modulus | Depth |
|---|---:|---:|
| `ckks_sign`, `ckks_cmp` | `S(D)` | `s` |
| `ckks_step`, `ckks_gt`, `ckks_ge`, `ckks_lt`, `ckks_le` | `S(D) + 1` | `s` |
| `ckks_indicator`, `ckks_eq` | `S(D) + D + 1` | `s + 1` |
| `ckks_select` | `D` | `1` |
| `ckks_clamp` | `2(S(D) + D + 1)` | `2(s + 1)` |

Equality and ordering are smooth masks.
`gt` equals `ge`, and `lt` equals `le`, because exact branching is unavailable in approximate CKKS.

### Packed operations

For `L = log2(slots)` on dense packing:

| Operation | Domain | Modulus | Depth |
|---|---|---:|---:|
| `ckks_sum_slots` | Dense slots | `0` | `0` |
| `ckks_fmax_slots`, `ckks_fmin_slots` | Pairwise differences in the sign domain | `L(S(D) + D + 1)` | `L(s + 1)` |
| `ckks_softmax` | Dense slots in `plan.input_interval` | `plan.consumed_bits(D)` | `plan.depth()` |

`SoftmaxPlan` includes the exponential approximation, slot count, and fixed Goldschmidt schedule.
Its cost is `exp.consumed_bits(D) + (reciprocal_iters + 4)D`.

## Approximation API

`minimax`, `minimax_with`, `degree_for_precision`, and `precision_at_depth` build host polynomials.
`PolynomialApproximation::from_polynomial` prepares a Poulpy BSGS polynomial and its interval map.
`ckks_eval_approximation` evaluates that plan with cost `plan.consumed_bits(D)` and depth `plan.depth()`.

See the generated Rust documentation for complete generic bounds and argument types.
