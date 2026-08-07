//! EvalMod parameterization: the periodic-function approximation of the CKKS
//! bootstrapping `x mod 1` step, and its backend-native encoding.
//!
//! Bootstrapping's modulus-raising (ModUp) step leaves a ciphertext holding
//! `I(X)·q + Δ·m`, where `m ∈ [-1, 1]` is the payload at encoding scale `Δ`, `q`
//! (a power of two) is the raised-from modulus, and `I` is the unwanted integer
//! multiple of `q` that ModUp introduces; their separation is the *message ratio*
//! `q/Δ = 2^log_message_ratio`.
//! In CKKS-meta terms `Δ = 2^log_delta` and `q = 2^(log_delta + log_budget)`, so
//! the ratio is `2^log_budget`. Normalized by `q` the value is `I + m·Δ/q`, so
//! removing `I` is exactly `x mod 1`. No low-degree polynomial computes `mod`, so
//! the circuit approximates it with a **periodic function** `f` whose period
//! matches `q`: periodicity collapses every `I·q`, so `f(I·q + Δ·m)` depends on
//! `m` alone. Since `f` is only locally linear in `m`, it is optionally
//! post-composed with its **inverse** `f⁻¹` to recover a value linear in `m`
//! across the whole interval — e.g. for the trigonometric family
//! `f⁻¹(f(x)) = (1/2π)·arcsin(sin(2π·x))`. The implemented `f` are trigonometric
//! (`sin`, `cos`, or the complex exponential `exp(2πi·)`, each scaled by `1/(2π)`
//! times the optional `scaling`), with `f⁻¹` the arcsine; other periodic families
//! or direct approximations may be added at a later time.
//!
//! The pipeline has up to three stages, all described by [`EvalMod`]:
//!
//! 1. **Base polynomial.** A polynomial approximation of `f` over the *reduced*
//!    range `[-K/2^r, K/2^r]`, evaluated homomorphically by baby-step/giant-step
//!    ([`EvalModBsgs`]). `K = f_mod_interval`, `r = f_mod_log_interval_reduction`.
//! 2. **Range extension.** `r` applications of a doubling identity specific to `f`
//!    (for the trigonometric families `cos 2θ = 2cos²θ − 1` and `exp 2θ = (exp θ)²`)
//!    extend the reduced range back to the full `[-K, K]`. This trades
//!    multiplicative depth for a lower-degree — hence cheaper — base polynomial.
//! 3. **Inverse.** An optional post-composition with `f⁻¹`
//!    ([`EvalMod::f_mod_inv_poly`]; the arcsine for the trigonometric
//!    families) that recovers a value linear in `m` rather than only near the
//!    origin.
//!
//! [`EvalModPlan`] is the user-facing recipe; [`EvalMod`]
//! is its compiled, plaintext-encoded form built by
//! [`compile_eval_mod`]. Polynomial coefficients are generated as scalar
//! slices, then encoded directly by the destination module into its own
//! plaintext storage. The result is evaluated through the
//! [`CKKSEvalModOps`](crate::api::CKKSEvalModOps) trait (see
//! [`crate::default::eval_mod`] for the evaluation itself).

use anyhow::{Result, anyhow, ensure};
use poulpy_core::layouts::{Base2K, bsgs_consumed_bits, bsgs_eval_depth};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CoeffsMeta,
    api::{Basis, CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncodingScalar, Parity},
    cosine,
    polynomial::{BSGSPolynomial, ComplexBSGSPolynomial, ComplexPolynomial, Polynomial, SplitStrategy},
};

use super::{CKKSModuleAlloc, CKKSPlaintextOwned, CKKSScalar};

// Fallible scalar conversions: building the host-side polynomials goes through
// the generic float `F`, and a conversion that is not exactly representable in
// the target scalar is reported as an error (with the offending value's role in
// `name`) rather than silently truncated.

fn scalar_from_f64<F: CKKSScalar>(name: &'static str, v: f64) -> Result<F> {
    F::from_f64(v).ok_or_else(|| anyhow!("{name}: value {v} not representable in target scalar"))
}

fn scalar_from_u64<F: CKKSScalar>(name: &'static str, v: u64) -> Result<F> {
    F::from_u64(v).ok_or_else(|| anyhow!("{name}: value {v} not representable in target scalar"))
}

fn scalar_from_usize<F: CKKSScalar>(name: &'static str, v: usize) -> Result<F> {
    F::from_usize(v).ok_or_else(|| anyhow!("{name}: value {v} not representable in target scalar"))
}

fn scalar_from_i64<F: CKKSScalar>(name: &'static str, v: i64) -> Result<F> {
    F::from_i64(v).ok_or_else(|| anyhow!("{name}: value {v} not representable in target scalar"))
}

/// Which periodic-function approximation of `x mod 1` the circuit evaluates, and
/// in what form the result is produced.
///
/// The variants below are the trigonometric families; each realizes its function
/// scaled by `1/(2π)` (times the optional `scaling`), with that factor folded into
/// the polynomial coefficients and the range-extension constants rather than applied
/// separately, so the recovered value sits at amplitude `scaling/(2π)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvalModType {
    /// Discrete cosine approximation following Han & Ki method (*Better Bootstrapping
    /// for Approximate Homomorphic Encryption*). Targets `1/(2π)·cos(2π·(x − 1/4)/2^r)`
    /// but fits the polynomial only at the finitely many points `x` can actually
    /// take (it is `≈ I(X)·q + Δ·m(X)`, near multiples of `q`) instead of
    /// over the whole continuous interval. This is more efficient than
    /// [`CosCheby`](Self::CosCheby) for **small `K`**; the continuous fit overtakes
    /// it as `K` grows due to the requirement `f_mod_degree ≥ 2·(K − 1)`.
    /// Can be paired with `f_mod_log_interval_reduction`.
    CosHK,
    /// Continuous Chebyshev approximation of `1/(2π)·sin(2π·x)` over `[−K, K]`.
    /// Implemented as the equivalent shifted cosine
    /// `cos(2π·(x − 1/4))`, which keeps the same target while using the full
    /// Chebyshev path.
    /// Cannot be paired with `f_mod_log_interval_reduction` (must be set to 0).
    SinCheby,
    /// Direct Chebyshev approximation of `1/(2π)·cos(2π·x)` over the reduced range
    /// — the same target as [`CosHK`](Self::CosHK), but fit continuously rather
    /// than at discrete points, which is more efficient for **large `K`**. `cos` is
    /// even, so a `−1/4`-period phase shift is baked into the interpolated function
    /// to make it odd around the message (this makes the polynomial non-even, hence
    /// `Parity::Full`). Can be paired with `f_mod_log_interval_reduction`.
    CosCheby,
    /// Complex exponential `1/(2π)·exp(2πi·x) = 1/(2π)·(cos(2π·x) + i·sin(2π·x))`,
    /// a variant built as a genuine complex polynomial. Produces both real
    /// and imaginary slot components.
    ///
    /// Unlike the real variants it is generated directly — no parity forcing, phase
    /// offset, or per-step constants, since complex squaring is self-contained:
    ///
    /// 1. **Per-step scaling** `s = (scaling/(2π))^(1/2^r)` (`r = f_mod_log_interval_reduction`):
    ///    the `2^r`-th root of the target amplitude, so the `r` squarings compound
    ///    it back to exactly `scaling/(2π)` (`s^(2^r) = scaling/(2π)`).
    /// 2. **Reduced half-range** `k_eff = K/2^r` (`K = f_mod_interval`).
    /// 3. Two real degree-`f_mod_degree` **Chebyshev interpolants** over
    ///    `[−k_eff, k_eff]`: `re(x) ≈ s·cos(2π·x)` and `im(x) ≈ s·sin(2π·x)` — the
    ///    real and imaginary parts of `s·exp(2πi·x)`.
    /// 4. Packed into a [`ComplexBSGSPolynomial`] (Chebyshev basis) and BSGS-encoded.
    ///
    /// At evaluation the base polynomial yields `s·exp(2πi·t)` for `|t| ≤ k_eff`,
    /// and the `r` complex squarings (`exp(iθ)² = exp(2iθ)`) extend it to the
    /// full-range `(scaling/(2π))·exp(2πi·x)`.
    ExpCmplx,
}

/// User-facing recipe for an `x mod 1` evaluation. Compiled into the encoded
/// [`EvalMod`] by [`compile_eval_mod`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvalModPlan {
    /// Which periodic-function approximation and output form to use.
    pub eval_mod_type: EvalModType,
    /// `log2` of the *message ratio* `q/Δ`: the cleartext is `I(X)·q + Δ·m` with
    /// `m ∈ [-1, 1]`. In CKKS-meta terms the encoding scale is `Δ = 2^log_delta` and
    /// the integer part wraps at the plaintext modulus `q = 2^k =
    /// 2^(log_delta + log_budget)`, so the ratio is `q/Δ = 2^log_budget` — i.e.
    /// `log_message_ratio` is the `log_budget` of the value being reduced, the bit
    /// gap between the payload and the integer part.
    pub log_msg_ratio: usize,
    /// Degree of the base polynomial approximation.
    pub f_mod_degree: usize,
    /// `K`, the number of message intervals the reduction spans: the approximation
    /// must remain valid for integer parts `|I| ≲ K`.
    pub f_mod_interval: usize,
    /// `r`, the number of range-extension steps. The base polynomial is built on
    /// the `2^r`-times-smaller range `[-K/2^r, K/2^r]`, and `r` range-extension
    /// steps extend it back to `[-K, K]`.
    pub f_mod_log_interval_reduction: usize,
    /// Degree of the optional inverse `f⁻¹` post-composition (the arcsine for the
    /// trigonometric families; `None` disables it). Must be odd. The base `f` is
    /// only linear in `m` near the origin; composing with `f⁻¹` recovers a value
    /// linear in `m` across the whole interval.
    pub f_mod_inv_degree: Option<usize>,
    /// Optional output amplitude scaling: the recovered value is scaled by
    /// `scaling / (2π)`. `None` uses the default `1.0`.
    pub scaling: Option<f64>,
    /// Baby-step/giant-step split strategy used to encode the polynomials (depth
    /// vs. number-of-rotations trade-off).
    pub split_strategy: SplitStrategy,
    /// CKKS metadata of the coefficients
    pub coeffs_meta: CoeffsMeta,
    /// Logscale used during EvalMod
    pub f_mod_log_delta: usize,
}

impl EvalModPlan {
    /// Multiplicative levels the eval_mod pipeline consumes: BSGS depth of the
    /// base `f` polynomial + `f_mod_log_interval_reduction` range-extension steps
    /// + BSGS depth of the optional inverse `f⁻¹` post-composition.
    ///
    /// Computed analytically from the plan via
    /// [`poulpy_core::layouts::bsgs_eval_depth`] (which accounts
    /// for the [`SplitStrategy`], so this is exact for `MinMult` as well as
    /// `MinDepth`); it matches the depth of the compiled [`EvalMod::eval_depth`].
    pub fn eval_depth(&self) -> usize {
        let base = bsgs_eval_depth(self.base_degree(), self.split_strategy);
        let inv = self.f_mod_inv_degree.map_or(0, |d| bsgs_eval_depth(d, self.split_strategy));
        base + self.f_mod_log_interval_reduction + inv
    }

    /// `log_budget` bits the eval_mod pipeline consumes on an input ciphertext of
    /// scale `input_log_delta`: the base polynomial evaluation
    /// ([`poulpy_core::layouts::bsgs_consumed_bits`] with the
    /// coefficient scale [`Self::coeffs_meta`]`.meta.log_delta`), plus
    /// `f_mod_log_interval_reduction` range-extension squarings (each a `ct×ct`
    /// consuming `input_log_delta`), plus the optional arcsine inverse. Computed
    /// analytically; matches the compiled [`EvalMod::consumed_bits`].
    pub fn consumed_bits(&self) -> usize {
        let coeff = self.coeffs_meta.meta.log_delta;
        let input_log_delta = self.f_mod_log_delta;
        // `bsgs_consumed_bits`'s depth model is parity-independent (the parameter
        // is documentation-only), so no per-family parity is threaded here; the
        // compiled polynomials carry their real parity (`compile_eval_mod` pins
        // Full for the phase-shifted Sin/CosCheby, CosHK keeps auto-detection).
        let base = bsgs_consumed_bits(
            self.base_degree(),
            self.split_strategy,
            Parity::Full,
            Basis::Chebyshev,
            input_log_delta,
            coeff,
        );
        let range_ext = self.f_mod_log_interval_reduction * input_log_delta;
        let inv = self.f_mod_inv_degree.map_or(0, |d| {
            bsgs_consumed_bits(d, self.split_strategy, Parity::Odd, Basis::Monomial, input_log_delta, coeff)
        });
        base + range_ext + inv
    }

    /// Degree of the base `f` polynomial actually encoded, so its BSGS depth can
    /// be derived without building it. For `CosHK` this is the minimax degree
    /// chosen by [`cosine::approximate_cos`]; otherwise the interpolation degree
    /// `f_mod_degree`.
    fn base_degree(&self) -> usize {
        match self.eval_mod_type {
            EvalModType::CosHK => {
                // Clear preconditions instead of a usize underflow inside the
                // Han–Ki degree table / a shift overflow on the message ratio;
                // `compile_eval_mod` enforces the same bounds with typed errors.
                assert!(self.f_mod_interval > 0, "EvalModPlan: f_mod_interval must be > 0");
                assert!(self.log_msg_ratio < 64, "EvalModPlan: log_msg_ratio must be < 64");
                cosine::approximate_cos_len(self.f_mod_interval, self.f_mod_degree, (1u64 << self.log_msg_ratio) as f64)
                    .saturating_sub(1)
            }
            _ => self.f_mod_degree,
        }
    }
}

/// BSGS-encoded base polynomial driving the homomorphic evaluation. Its
/// coefficients are stored as the plaintexts `P` (host-side after
/// [`compile_eval_mod`].
pub enum EvalModBsgs<P> {
    /// Real-valued `f` (e.g. the `sin`/`cos` families).
    Real(BSGSPolynomial<P>),
    /// Complex-valued `f` (e.g. [`EvalModType::ExpCmplx`]).
    Complex(ComplexBSGSPolynomial<P>),
}

/// Host-side polynomials retained alongside their BSGS-encoded counterparts in
/// [`EvalMod`]. These are the exact functions the circuit evaluates,
/// usable for reference evaluation via [`Polynomial::evaluate`].
pub enum EvalModPoly<F> {
    /// Real-valued `f` (e.g. the `sin`/`cos` families).
    Real(Polynomial<F>),
    /// Complex-valued `f` (e.g. [`EvalModType::ExpCmplx`]).
    Complex(ComplexPolynomial<F>),
}

/// Compiled `x mod 1` parameters: the periodic-function approximation polynomials
/// of an [`EvalModPlan`], encoded into plaintexts `P` ready for
/// homomorphic evaluation, plus the host-side polynomials kept for reference.
///
/// `F` is the host floating-point scalar the polynomials were built in; `P` is
/// the plaintext storage type — a host [`CKKSPlaintext`] right after
/// [`compile_eval_mod`]. Evaluate it with
/// [`CKKSEvalModOps::ckks_eval_mod`](crate::api::CKKSEvalModOps::ckks_eval_mod).
pub struct EvalMod<F, P> {
    /// Copy of [`EvalModPlan`].
    pub plan: EvalModPlan,
    /// Encoded constants subtracted at each real range-extension step, packed one
    /// per coefficient: coefficient `i` holds the step-`i` constant `s^(2^(i+1))`
    /// (see [`Self::range_extension_scale`]) — for the trigonometric families the
    /// `cos 2θ = 2cos²θ − dac` identity, with the baked-in scaling. A single
    /// plaintext suffices since every constant shares the same metadata. `None`
    /// when no per-step constant is needed (no range extension, or the complex
    /// [`EvalModType::ExpCmplx`] path, whose squaring is exact).
    pub range_extension_consts: Option<P>,
    /// BSGS-encoded base polynomial actually evaluated on the ciphertext.
    pub f_mod_bsgs: EvalModBsgs<P>,
    /// BSGS-encoded inverse `f⁻¹` post-composition (the arcsine for the
    /// trigonometric families), present when `f_mod_inv_degree` is `Some`.
    pub f_mod_inv_bsgs: Option<BSGSPolynomial<P>>,
    /// Host-side base polynomial that `f_mod_bsgs` encodes.
    pub f_mod_poly: EvalModPoly<F>,
    /// Host-side inverse `f⁻¹` post-composition polynomial that `f_mod_inv_bsgs`
    /// encodes, when present.
    pub f_mod_inv_poly: Option<Polynomial<F>>,
}

fn encode_bsgs_backend<BE, F>(
    polynomial: &Polynomial<F>,
    module: &Module<BE>,
    base2k: Base2K,
    coeff_meta: CoeffsMeta,
    strategy: SplitStrategy,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<BSGSPolynomial<CKKSPlaintextOwned<BE>>>
where
    BE: Backend,
    Module<BE>: CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
    F: CKKSEncodingScalar,
{
    let mut step_idx = 0usize;
    polynomial.decompose_bsgs_with(strategy, |baby_coeffs| {
        let mut pt = module.ckks_pt_coeffs_alloc(baby_coeffs.len(), base2k, coeff_meta.k);
        pt.set_meta_checked(coeff_meta.meta)?;
        module
            .ckks_encode_coeffs_host_into(&mut pt, baby_coeffs, scratch)
            .map_err(|error| anyhow!("encode_bsgs: step {step_idx}: {error}"))?;
        step_idx += 1;
        Ok(pt)
    })
}

fn encode_complex_bsgs_backend<BE, F>(
    polynomial: &ComplexPolynomial<F>,
    module: &Module<BE>,
    base2k: Base2K,
    coeff_meta: CoeffsMeta,
    strategy: SplitStrategy,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<ComplexBSGSPolynomial<CKKSPlaintextOwned<BE>>>
where
    BE: Backend,
    Module<BE>: CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
    F: CKKSEncodingScalar,
{
    let (re, im) = polynomial.split_with_shared_parity();
    Ok(ComplexBSGSPolynomial {
        re: encode_bsgs_backend(&re, module, base2k, coeff_meta, strategy, scratch)?,
        im: encode_bsgs_backend(&im, module, base2k, coeff_meta, strategy, scratch)?,
    })
}

/// Compiles an [`EvalModPlan`] directly into destination-backend plaintexts.
///
/// Polynomial construction remains ordinary scalar setup work. Each BSGS baby
/// polynomial and range-extension constant is staged through `scratch` and
/// quantized by `module`'s backend-specific CKKS coefficient codec.
///
/// # Errors
///
/// Returns an error if the plan is inconsistent, a generated coefficient is
/// not representable, or `scratch` is too small for the largest coefficient
/// block.
pub fn compile_eval_mod<BE, F>(
    base2k: Base2K,
    lit: EvalModPlan,
    module: &Module<BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<EvalMod<F, CKKSPlaintextOwned<BE>>>
where
    BE: Backend,
    Module<BE>: CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
    F: CKKSEncodingScalar,
{
    if lit.eval_mod_type == EvalModType::ExpCmplx {
        return compile_eval_mod_exp(base2k, lit, module, scratch);
    }
    let coeff_meta = lit.coeffs_meta;

    ensure!(lit.f_mod_degree > 0, "f_mod_degree must be > 0");
    ensure!(lit.f_mod_interval > 0, "f_mod_interval must be > 0");
    ensure!(lit.log_msg_ratio < 64, "log_msg_ratio must be < 64");
    ensure!(
        !(lit.eval_mod_type == EvalModType::SinCheby && lit.f_mod_log_interval_reduction != 0),
        "SinCheby requires f_mod_log_interval_reduction = 0"
    );
    ensure!(
        !(lit.eval_mod_type == EvalModType::CosHK && lit.f_mod_degree < 2 * (lit.f_mod_interval - 1)),
        "CosHK requires f_mod_degree >= 2*(K-1)"
    );
    ensure!(
        lit.f_mod_log_interval_reduction < 31,
        "f_mod_log_interval_reduction must be < 31"
    );

    let f_mod_log_interval_reduction = match lit.eval_mod_type {
        EvalModType::SinCheby => 0,
        _ => lit.f_mod_log_interval_reduction,
    };

    let scaling_f64 = lit.scaling.unwrap_or(1.0);
    let scaling: F = scalar_from_f64("scaling", scaling_f64)?;
    let sc_fac: F = scalar_from_u64("2^f_mod_log_interval_reduction", 1u64 << f_mod_log_interval_reduction)?;
    let k_eff = scalar_from_usize::<F>("f_mod_interval", lit.f_mod_interval)? / sc_fac;

    let two = F::one() + F::one();
    let two_pi = two * F::PI();
    let inv_two_pi = F::one() / two_pi;

    let mut f_mod_inv_poly_opt: Option<Polynomial<F>> = None;
    let s: F = if let Some(n) = lit.f_mod_inv_degree {
        ensure!(!n.is_multiple_of(2), "f_mod_inv_degree must be odd");
        let mut coeffs = vec![F::zero(); n + 1];
        coeffs[1] = inv_two_pi * scaling;
        let mut i = 1usize;
        while i + 2 <= n {
            let next = i + 2;
            let num: F = scalar_from_i64("arcsine num", (next as i64 - 2) * (next as i64 - 2))?;
            let den: F = scalar_from_i64("arcsine den", next as i64 * (next as i64 - 1))?;
            coeffs[next] = coeffs[i] * num / den;
            i = next;
        }
        f_mod_inv_poly_opt = Some(Polynomial::new_with_parity(Basis::Monomial, coeffs, Parity::Odd));
        F::one()
    } else {
        (inv_two_pi * scaling).powf(F::one() / sc_fac)
    };

    let mut f_mod_poly: Polynomial<F> = match lit.eval_mod_type {
        EvalModType::SinCheby => {
            // Use the equivalent shifted cosine rather than an odd-only sine
            // polynomial: it evaluates the same periodic target and exercises
            // the full Chebyshev BSGS path used by the other real variants.
            let off = scalar_from_f64::<F>("-0.25", -0.25)?;
            Polynomial::chebyshev_interpolate(lit.f_mod_degree, -k_eff, k_eff, |x| (two_pi * (x + off)).cos())?
        }
        EvalModType::CosCheby => {
            // Bake the −1/4-period phase shift into the interpolated function so
            // no separate offset is added at evaluation time. The polynomial is
            // applied to the reduced argument and the `2^r` range-extension
            // squarings then scale the argument by `2^r`, so the baked shift is
            // `−1/4 / 2^r` (it reaches the full `−1/4` after the squarings). The
            // shifted cosine is no longer even in `x`, hence `Parity::Full` below.
            let off = scalar_from_f64::<F>("-0.25", -0.25)? / scalar_from_u64::<F>("2^r", 1u64 << f_mod_log_interval_reduction)?;
            Polynomial::chebyshev_interpolate(lit.f_mod_degree, -k_eff, k_eff, |x| (two_pi * (x + off)).cos())?
        }
        EvalModType::CosHK => {
            let coeffs = cosine::approximate_cos::<F>(
                lit.f_mod_interval,
                lit.f_mod_degree,
                (1u64 << lit.log_msg_ratio) as f64,
                f_mod_log_interval_reduction,
            );
            // cos(2π·(x-1/4)/2^r) is not even in x; Parity::Full preserves
            // the odd-degree Chebyshev coefficients in BSGS evaluation.
            Polynomial::new_with_parity(Basis::Chebyshev, coeffs, Parity::Full)
        }
        EvalModType::ExpCmplx => unreachable!(),
    };
    match lit.eval_mod_type {
        EvalModType::SinCheby => f_mod_poly.parity = Parity::Full,
        // The phase-shifted cosine is not even, so keep all coefficients.
        EvalModType::CosCheby => f_mod_poly.parity = Parity::Full,
        EvalModType::CosHK => {}
        EvalModType::ExpCmplx => unreachable!(),
    }

    for c in f_mod_poly.coeffs.iter_mut() {
        *c = *c * s;
    }

    let f_mod_bsgs = encode_bsgs_backend(&f_mod_poly, module, base2k, coeff_meta, lit.split_strategy, scratch)?;
    let f_mod_inv_bsgs = f_mod_inv_poly_opt
        .as_ref()
        .map(|p| encode_bsgs_backend(p, module, base2k, coeff_meta, lit.split_strategy, scratch))
        .transpose()?;

    // Pack the per-step constants `s^(2^(i+1))` one per coefficient into a
    // single plaintext (they all share `coeff_meta`); the evaluator reads
    // coefficient `i` at step `i`.
    let range_extension_consts = if f_mod_log_interval_reduction > 0 {
        let vals: Vec<F> = (0..f_mod_log_interval_reduction).map(|i| s.powi(1i32 << (i + 1))).collect();
        let mut pt = module.ckks_pt_coeffs_alloc(f_mod_log_interval_reduction, base2k, coeff_meta.k);
        pt.set_meta_checked(coeff_meta.meta)?;
        module
            .ckks_encode_coeffs_host_into(&mut pt, &vals, scratch)
            .map_err(|e| anyhow!("range_extension_consts: {e}"))?;
        Some(pt)
    } else {
        None
    };

    Ok(EvalMod {
        plan: lit,
        range_extension_consts,
        f_mod_bsgs: EvalModBsgs::Real(f_mod_bsgs),
        f_mod_inv_bsgs,
        f_mod_poly: EvalModPoly::Real(f_mod_poly),
        f_mod_inv_poly: f_mod_inv_poly_opt,
    })
}

/// [`EvalModType::ExpCmplx`] specialization of [`compile_eval_mod`]: interpolates
/// the real and imaginary parts of `s·exp(2πi·x)` on the reduced range as a
/// single [`ComplexPolynomial`] and BSGS-encodes it. The `r` range-extension
/// steps are plain complex squarings (`exp 2θ = (exp θ)²`), so no offset or
/// per-step constant is needed.
fn compile_eval_mod_exp<BE, F>(
    base2k: Base2K,
    lit: EvalModPlan,
    module: &Module<BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<EvalMod<F, CKKSPlaintextOwned<BE>>>
where
    BE: Backend,
    Module<BE>: CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
    F: CKKSEncodingScalar,
{
    let coeff_meta = lit.coeffs_meta;
    ensure!(lit.f_mod_degree > 0, "f_mod_degree must be > 0");
    ensure!(lit.f_mod_interval > 0, "f_mod_interval must be > 0");
    ensure!(lit.log_msg_ratio < 64, "log_msg_ratio must be < 64");
    ensure!(
        lit.f_mod_log_interval_reduction < 31,
        "f_mod_log_interval_reduction must be < 31"
    );

    let scaling_f64 = lit.scaling.unwrap_or(1.0);
    let scaling: F = scalar_from_f64("scaling", scaling_f64)?;
    let sc_fac: F = scalar_from_u64("2^f_mod_log_interval_reduction", 1u64 << lit.f_mod_log_interval_reduction)?;
    let k_eff = scalar_from_usize::<F>("f_mod_interval", lit.f_mod_interval)? / sc_fac;

    let two = F::one() + F::one();
    let two_pi = two * F::PI();
    let s: F = (F::one() / two_pi * scaling).powf(F::one() / sc_fac);

    let re = Polynomial::chebyshev_interpolate(lit.f_mod_degree, -k_eff, k_eff, |x| s * (two_pi * x).cos())?;
    let im = Polynomial::chebyshev_interpolate(lit.f_mod_degree, -k_eff, k_eff, |x| s * (two_pi * x).sin())?;
    let exp_poly = ComplexPolynomial::new(Basis::Chebyshev, re.coeffs, im.coeffs);
    let exp_bsgs = encode_complex_bsgs_backend(&exp_poly, module, base2k, coeff_meta, lit.split_strategy, scratch)?;

    Ok(EvalMod {
        plan: lit,
        range_extension_consts: None,
        f_mod_bsgs: EvalModBsgs::Complex(exp_bsgs),
        f_mod_inv_bsgs: None,
        f_mod_poly: EvalModPoly::Complex(exp_poly),
        f_mod_inv_poly: None,
    })
}

impl<F, P> EvalMod<F, P> {
    /// Number of CKKS multiplicative levels the eval_mod pipeline will consume:
    /// BSGS eval depth of the base `f` polynomial + `f_mod_log_interval_reduction`
    /// range-extension steps + BSGS eval depth of the optional inverse `f⁻¹`
    /// post-composition.
    ///
    /// The BSGS depths come from [`BSGSPolynomial::eval_depth`], which accounts for
    /// the chosen [`SplitStrategy`] — so this is exact for `MinMult` as well as
    /// `MinDepth` (a `MinMult` split can cost one extra level).
    pub fn eval_depth(&self) -> usize {
        let base = match &self.f_mod_bsgs {
            EvalModBsgs::Real(p) => p.eval_depth(),
            EvalModBsgs::Complex(p) => p.re.eval_depth(),
        };
        let inv = self.f_mod_inv_bsgs.as_ref().map_or(0, |p| p.eval_depth());
        base + self.plan.f_mod_log_interval_reduction + inv
    }

    /// `log_budget` bits consumed evaluating the pipeline on an input ciphertext
    /// of scale `input_log_delta`: base polynomial (heaviest BSGS chain) + the
    /// `f_mod_log_interval_reduction` range-extension squarings (`ct×ct`,
    /// `input_log_delta` each) + the optional arcsine inverse. Matches the actual
    /// runtime consumption and [`EvalModPlan::consumed_bits`].
    pub fn consumed_bits(&self) -> usize {
        let coeff = self.plan.coeffs_meta.meta.log_delta;
        let log_delta = self.plan.f_mod_log_delta;
        let base = match &self.f_mod_bsgs {
            EvalModBsgs::Real(p) => p.consumed_bits(log_delta, coeff),
            EvalModBsgs::Complex(p) => p.re.consumed_bits(log_delta, coeff),
        };
        let range_ext = self.plan.f_mod_log_interval_reduction * log_delta;
        let inv = self.f_mod_inv_bsgs.as_ref().map_or(0, |p| p.consumed_bits(log_delta, coeff));
        base + range_ext + inv
    }

    /// Per-step range-extension scaling `s`: the base polynomial bakes `s` into its
    /// coefficients, and each range-extension step folds in `s^(2^(i+1))`. When an
    /// inverse `f⁻¹` post-composition is used the base is unscaled (`s = 1`).
    pub fn range_extension_scale(&self) -> f64 {
        if self.f_mod_inv_poly.is_some() {
            return 1.0;
        }
        (std::f64::consts::TAU.recip() * self.plan.scaling.unwrap_or(1.0))
            .powf(1.0 / (1u64 << self.plan.f_mod_log_interval_reduction) as f64)
    }

    /// Maps every encoded plaintext field from storage `P` to storage `Q`,
    /// leaving the scalar polynomial descriptions untouched.
    ///
    /// Normal setup should use [`compile_eval_mod`], which already encodes on
    /// the destination backend. This utility remains useful for explicit
    /// storage conversions and serialization adapters.
    pub fn map_plaintexts<Q>(self, mut f: impl FnMut(&P) -> Q) -> EvalMod<F, Q> {
        let Self {
            plan,
            range_extension_consts,
            f_mod_bsgs,
            f_mod_inv_bsgs,
            f_mod_poly,
            f_mod_inv_poly,
        } = self;
        EvalMod {
            plan,
            range_extension_consts: range_extension_consts.as_ref().map(&mut f),
            f_mod_bsgs: match f_mod_bsgs {
                EvalModBsgs::Real(p) => EvalModBsgs::Real(p.map_baby_steps_ref(&mut f)),
                EvalModBsgs::Complex(p) => EvalModBsgs::Complex(p.map_baby_steps_ref(&mut f)),
            },
            f_mod_inv_bsgs: f_mod_inv_bsgs.as_ref().map(|p| p.map_baby_steps_ref(&mut f)),
            f_mod_poly,
            f_mod_inv_poly,
        }
    }
}
