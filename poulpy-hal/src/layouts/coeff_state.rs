//! Sealed coefficient-state algebra for the normalization/canonicality typestate.
//!
//! This module introduces the target state model of
//! `docs/spec/normalization_typestate.md` (§3) without changing any kernel or layout:
//!
//! - [`CoefficientState`] with the legal forms [`Unwritten`], [`Raw`], and
//!   [`Coeff<N, C>`](Coeff);
//! - the orthogonal axes [`Normalization`] (the existing [`Normalized`]/[`Unnormalized`]
//!   markers) and [`Canonicality`] ([`Canonical`]/[`NonCanonical`]);
//! - the sealed proof-weakening relations [`CanonicalityFitsIn`] and the product
//!   relation [`CoeffFitsIn`] (spec §3.5);
//! - the private representation context ([`CoeffContext`]) and conservative carry
//!   certificate ([`CarryCert`]) of spec §3.4, together with the reference canonical
//!   projection `P_p` of spec §3.3.
//!
//! During the migration (spec §10, PR 1) the existing roots keep their
//! [`crate::layouts::NormalizationState`] parameter; its conservative image in this
//! algebra is exposed as `NormalizationState::AsCoeff`. Later PRs move roots and
//! operation bounds onto
//! [`CoefficientState`] directly.
//!
//! All markers are conservative claims: [`NonCanonical`] means "canonicality is not
//! proven", not "padding is known dirty", and [`Unnormalized`] admits values whose bytes
//! happen to be normalized.
//!
//! # Exit-gate guarantees (spec PR 2, §11.1)
//!
//! A normalized root exposes no safe mutable words:
//!
//! ```compile_fail,E0599
//! use poulpy_hal::layouts::{VecZnx, ZnxViewMut};
//! let mut v: VecZnx<Vec<u8>, i64> = VecZnx::from_data(vec![0u8; 64], 8, 1, 1);
//! v.at_mut(0, 0)[0] = 1;
//! ```
//!
//! nor safe mutable storage:
//!
//! ```compile_fail,E0599
//! use poulpy_hal::layouts::{DataViewMut, VecZnx};
//! let mut v: VecZnx<Vec<u8>, i64> = VecZnx::from_data(vec![0u8; 64], 8, 1, 1);
//! let _ = v.data_mut();
//! ```
//!
//! A mutable *borrowed view* cannot be relabelled (only owned roots and the
//! authoritative arena view wrappers transition, so a relabelled borrow can never leave
//! a stale owner label behind):
//!
//! ```compile_fail,E0599
//! use poulpy_hal::layouts::VecZnx;
//! let mut buf = vec![0u8; 64];
//! let view: VecZnx<&mut [u8], i64> = VecZnx::from_data(buf.as_mut_slice(), 8, 1, 1);
//! let _ = view.into_unnormalized();
//! ```
//!
//! and no weakening relation manufactures a normalization proof:
//!
//! ```compile_fail,E0277
//! use poulpy_hal::layouts::{CoeffNormalized, CoeffUnnormalized, VecZnx};
//! let u: VecZnx<Vec<u8>, i64, CoeffUnnormalized> = VecZnx::from_data_unnormalized(vec![0u8; 64], 8, 1, 1);
//! let _ = u.into_state::<CoeffNormalized>();
//! ```

use std::{fmt, marker::PhantomData};

use crate::layouts::{FitsIn, Normalization, Normalized, Unnormalized};

mod sealed {
    pub trait Sealed {}
}

/// Canonicality axis marker bound. Sealed: [`Canonical`] and [`NonCanonical`] are the
/// only implementors. See spec §3.3 for the padding invariant the axis describes.
pub trait Canonicality:
    sealed::Sealed + Copy + Clone + Default + fmt::Debug + PartialEq + Eq + std::hash::Hash + Send + Sync + 'static
{
}

/// Marker: the low `p` padding bits of every bottom-live-limb word are proven zero for
/// the value's representation context.
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq, Hash)]
pub struct Canonical;

/// Marker: canonicality is not proven (the padding bits may or may not be zero).
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq, Hash)]
pub struct NonCanonical;

impl sealed::Sealed for Canonical {}
impl sealed::Sealed for NonCanonical {}
impl Canonicality for Canonical {}
impl Canonicality for NonCanonical {}

/// Coefficient-domain state bound. Sealed: the legal forms are [`Unwritten`], [`Raw`],
/// and [`Coeff<N, C>`](Coeff) (spec §3.1).
pub trait CoefficientState:
    sealed::Sealed + Copy + Clone + Default + fmt::Debug + PartialEq + Eq + std::hash::Hash + Send + Sync + 'static
{
}

/// Scratch/output storage whose complete logical contents have not been initialized.
/// Not readable as coefficients.
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq, Hash)]
pub struct Unwritten;

/// Initialized storage with no arithmetic or canonicality claim (raw ingestion,
/// deserialization, metadata edits). Readable as words/bytes only.
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq, Hash)]
pub struct Raw;

/// An initialized limb representation: `N` claims the normalization invariant, `C` the
/// canonical-padding invariant, each for the root's immutable representation context.
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq, Hash)]
pub struct Coeff<N: Normalization, C: Canonicality>(PhantomData<(N, C)>);

impl sealed::Sealed for Unwritten {}
impl sealed::Sealed for Raw {}
impl<N: Normalization, C: Canonicality> sealed::Sealed for Coeff<N, C> {}
impl CoefficientState for Unwritten {}
impl CoefficientState for Raw {}
impl<N: Normalization, C: Canonicality> CoefficientState for Coeff<N, C> {}

/// Conservative compatibility image of today's `Normalized` root state (spec PR 1:
/// "default conservatively where a temporary compatibility default is unavoidable").
pub type CoeffNormalized = Coeff<Normalized, NonCanonical>;

/// Conservative compatibility image of today's `Unnormalized` root state.
pub type CoeffUnnormalized = Coeff<Unnormalized, NonCanonical>;

/// Sealed proof-weakening relation on the canonicality axis (spec §3.5):
/// `Canonical` fits `Canonical` and `NonCanonical`; `NonCanonical` fits only itself.
pub trait CanonicalityFitsIn<C: Canonicality>: Canonicality {}

impl CanonicalityFitsIn<Canonical> for Canonical {}
impl<C: Canonicality> CanonicalityFitsIn<NonCanonical> for C {}

/// Sealed product proof-weakening relation on [`Coeff`] states (spec §3.5): a stronger
/// input may be written to a weaker destination without relabelling a borrow. Used by
/// copy and state-preserving operations once roots carry [`CoefficientState`].
/// [`Unwritten`] and [`Raw`] deliberately participate in no weakening: their transitions
/// are consuming root operations (spec §5.6), never implicit compatibility.
pub trait CoeffFitsIn<S: CoefficientState>: CoefficientState {}

impl<N1, N2, C1, C2> CoeffFitsIn<Coeff<N2, C2>> for Coeff<N1, C1>
where
    N1: Normalization + FitsIn<N2>,
    N2: Normalization,
    C1: CanonicalityFitsIn<C2>,
    C2: Canonicality,
{
}

/// The [`CoefficientState`]s that are readable as coefficients: exactly the
/// [`Coeff<N, C>`](Coeff) forms, with their axes exposed as associated types
/// (spec §3.1's "readable as coefficients" rows). [`Unwritten`] and [`Raw`] do not
/// implement it, which is how operation bounds exclude them from arithmetic inputs.
///
/// The associated types let a bound pin one axis while leaving the other free with an
/// equality constraint, which type inference resolves eagerly: a DFT entry point writes
/// `A::State: ArithmeticState<N = Normalized>` (any canonicality), a carry-producing
/// destination writes `R::State: ArithmeticState<N = Unnormalized>`, and a
/// padding-sensitive consumer writes `A::State: ArithmeticState<C = Canonical>`.
/// Every arithmetic state fits the weakest one, `Coeff<Unnormalized, NonCanonical>`;
/// carrying that as a supertrait lets it elaborate through opaque `State` associated
/// types, mirroring the old blanket "everything fits an unnormalized destination" rule.
pub trait ArithmeticState: CoefficientState + CoeffFitsIn<CoeffUnnormalized> {
    /// The normalization axis of this state.
    type N: Normalization;
    /// The canonicality axis of this state.
    type C: Canonicality;
}

impl<N: Normalization, C: Canonicality> ArithmeticState for Coeff<N, C> {
    type N = N;
    type C = C;
}

/// Immutable representation context interpreting both state axes for one arithmetic
/// root (spec §3.4). Attached to the nominal roots in PR 2; until then it is
/// constructed transiently where the invariants must be checked.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct CoeffContext {
    pub(crate) n: usize,
    pub(crate) cols: usize,
    pub(crate) live_limbs: usize,
    pub(crate) capacity_limbs: usize,
    pub(crate) base2k: usize,
    pub(crate) represented_k: usize,
}

#[cfg_attr(not(test), allow(dead_code))]
impl CoeffContext {
    /// Construction enforces the invariants of spec §3.3/§3.4: `base2k > 0`,
    /// `represented_k > 0` (so the bottom live limb always exists), the live width is
    /// exactly `ceil(represented_k / base2k)`, and live limbs fit in capacity.
    pub(crate) fn new(n: usize, cols: usize, capacity_limbs: usize, base2k: usize, represented_k: usize) -> Self {
        assert!(n > 0 && cols > 0, "CoeffContext: empty layout");
        assert!(base2k > 0, "CoeffContext: base2k must be positive");
        assert!(represented_k > 0, "CoeffContext: represented_k must be positive");
        let live_limbs = represented_k.div_ceil(base2k);
        assert!(
            live_limbs <= capacity_limbs,
            "CoeffContext: live limbs ({live_limbs}) exceed capacity ({capacity_limbs})"
        );
        Self {
            n,
            cols,
            live_limbs,
            capacity_limbs,
            base2k,
            represented_k,
        }
    }

    /// Number of proven-zero padding bits `p` in the bottom live limb for a canonical
    /// value: `0` when `represented_k` is a multiple of `base2k`, else
    /// `base2k - represented_k % base2k` (spec §3.3).
    pub(crate) fn padding_bits(&self) -> u32 {
        let r = self.represented_k % self.base2k;
        if r == 0 { 0 } else { (self.base2k - r) as u32 }
    }
}

/// Reference canonical projection `P_p` on one backend word (spec §3.3):
/// clear the low `p` bits through the unsigned bit representation, without a numeric
/// cast. Backend kernels (PR 4) must be bit-identical to this definition.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn canonical_projection_i64(d: i64, p: u32) -> i64 {
    debug_assert!(p < 64);
    (d as u64 & (!0u64 << p)) as i64
}

/// Conservative digit-bound certificate carried privately by an unnormalized root
/// (spec §3.4). It decides whether another carry-producing operation is sound; it is
/// never consulted, required, or rejected by normalization.
///
/// Reservation rules (spec §6.1, invariant 14): a carry-producing mutable child starts
/// from a copy of the root's certificate; a successful operation replaces the root's
/// certificate with the child's updated one; parallel children each update their own
/// copy and the root takes [`CarryCert::join`] of all of them before it is used again.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct CarryCert {
    pub(crate) min_digit: i128,
    pub(crate) max_digit: i128,
}

#[cfg_attr(not(test), allow(dead_code))]
impl CarryCert {
    /// Certificate of a freshly normalized value: digits in `[-2^(b-1), 2^(b-1))`.
    pub(crate) fn fresh_normalized(base2k: usize) -> Self {
        Self {
            min_digit: -(1i128 << (base2k - 1)),
            max_digit: (1i128 << (base2k - 1)) - 1,
        }
    }

    /// Certificate of a structural binding of raw words: the full word range, i.e. zero
    /// additional headroom (spec §3.2). Normalization is still always available.
    pub(crate) fn raw_binding(word_bits: u32) -> Self {
        Self {
            min_digit: -(1i128 << (word_bits - 1)),
            max_digit: (1i128 << (word_bits - 1)) - 1,
        }
    }

    /// Bound after adding a digit certified by `other` (saturating, conservative).
    pub(crate) fn after_add(&self, other: &Self) -> Self {
        Self {
            min_digit: self.min_digit.saturating_add(other.min_digit),
            max_digit: self.max_digit.saturating_add(other.max_digit),
        }
    }

    /// Bound after negation.
    pub(crate) fn after_neg(&self) -> Self {
        Self {
            min_digit: self.max_digit.saturating_neg(),
            max_digit: self.min_digit.saturating_neg(),
        }
    }

    /// Bound after the canonical projection `P_p`: words move toward negative infinity
    /// by at most `2^p - 1` (spec §3.3).
    pub(crate) fn after_canonical_projection(&self, p: u32) -> Self {
        Self {
            min_digit: self.min_digit.saturating_sub((1i128 << p) - 1),
            max_digit: self.max_digit,
        }
    }

    /// Conservative parallel join: the smallest interval covering both children.
    pub(crate) fn join(&self, other: &Self) -> Self {
        Self {
            min_digit: self.min_digit.min(other.min_digit),
            max_digit: self.max_digit.max(other.max_digit),
        }
    }

    /// Whether every certified digit still fits in a signed word of `word_bits` bits,
    /// i.e. whether one more operation certified by this bound is sound.
    pub(crate) fn fits_in_word(&self, word_bits: u32) -> bool {
        self.min_digit >= -(1i128 << (word_bits - 1)) && self.max_digit < (1i128 << (word_bits - 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layouts::NormalizationState;

    fn assert_fits<A: CoeffFitsIn<B>, B: CoefficientState>() {}

    #[test]
    fn coeff_fits_in_is_the_product_of_both_axes() {
        assert_fits::<Coeff<Normalized, Canonical>, Coeff<Normalized, Canonical>>();
        assert_fits::<Coeff<Normalized, Canonical>, Coeff<Normalized, NonCanonical>>();
        assert_fits::<Coeff<Normalized, Canonical>, Coeff<Unnormalized, Canonical>>();
        assert_fits::<Coeff<Normalized, Canonical>, Coeff<Unnormalized, NonCanonical>>();
        assert_fits::<Coeff<Normalized, NonCanonical>, Coeff<Unnormalized, NonCanonical>>();
        assert_fits::<Coeff<Unnormalized, Canonical>, Coeff<Unnormalized, NonCanonical>>();
        assert_fits::<Coeff<Unnormalized, NonCanonical>, Coeff<Unnormalized, NonCanonical>>();
        // The illegal directions (Unnormalized -> Normalized, NonCanonical -> Canonical,
        // and anything involving Unwritten/Raw) are locked by compile-fail tests in the
        // verification plan (spec §11.1) once roots carry CoefficientState.
    }

    #[test]
    fn arithmetic_state_projects_both_axes() {
        fn normalized_input<S: ArithmeticState<N = Normalized>>() {}
        fn unnormalized_destination<S: ArithmeticState<N = Unnormalized>>() {}
        fn canonical_consumer<S: ArithmeticState<C = Canonical>>() {}
        normalized_input::<Coeff<Normalized, Canonical>>();
        normalized_input::<Coeff<Normalized, NonCanonical>>();
        unnormalized_destination::<Coeff<Unnormalized, Canonical>>();
        unnormalized_destination::<Coeff<Unnormalized, NonCanonical>>();
        canonical_consumer::<Coeff<Normalized, Canonical>>();
        canonical_consumer::<Coeff<Unnormalized, Canonical>>();
        // Unwritten and Raw implement CoefficientState but not ArithmeticState, so they
        // are rejected by every such bound (compile-fail coverage lands with §11.1).
    }

    #[test]
    fn as_coeff_maps_legacy_states_conservatively() {
        fn same<A: 'static, B: 'static>() -> bool {
            std::any::TypeId::of::<A>() == std::any::TypeId::of::<B>()
        }
        assert!(same::<<Normalized as NormalizationState>::AsCoeff, CoeffNormalized>());
        assert!(same::<<Unnormalized as NormalizationState>::AsCoeff, CoeffUnnormalized>());
    }

    #[test]
    fn canonical_projection_reference_properties() {
        let words: Vec<i64> = vec![0, 1, -1, 5, -5, 255, -255, i64::MAX, i64::MIN, 0x7fff_ffff, -0x8000_0000];
        for p in 0..16u32 {
            let grid = 1i64 << p;
            for &d in &words {
                let r = canonical_projection_i64(d, p);
                // idempotent
                assert_eq!(canonical_projection_i64(r, p), r);
                // grid-aligned
                assert!(r.rem_euclid(grid) == 0);
                // rounds toward negative infinity by less than 2^p
                assert!(r <= d && d - r < grid, "d={d} p={p} r={r}");
            }
            // p == 0 is the identity
            if p == 0 {
                for &d in &words {
                    assert_eq!(canonical_projection_i64(d, 0), d);
                }
            }
        }
    }

    #[test]
    fn canonical_projection_preserves_normalized_interval() {
        // The endpoint -2^(b-1) is 2^p-grid-aligned for every p <= b-1, so projecting a
        // normalized digit stays inside [-2^(b-1), 2^(b-1)).
        for b in 2..=32u32 {
            let lo = -(1i64 << (b - 1));
            let hi = (1i64 << (b - 1)) - 1;
            for p in 0..b {
                for d in [lo, lo + 1, -1, 0, 1, hi - 1, hi] {
                    let r = canonical_projection_i64(d, p);
                    assert!(r >= lo && r <= hi, "b={b} p={p} d={d} r={r}");
                }
            }
        }
    }

    #[test]
    fn coeff_context_invariants() {
        let ctx = CoeffContext::new(64, 2, 5, 17, 70);
        assert_eq!(ctx.live_limbs, 5); // ceil(70 / 17)
        assert_eq!(ctx.padding_bits(), 15); // 17 - 70 % 17
        let aligned = CoeffContext::new(64, 2, 4, 17, 68);
        assert_eq!(aligned.padding_bits(), 0);
    }

    #[test]
    #[should_panic(expected = "represented_k must be positive")]
    fn coeff_context_rejects_zero_precision() {
        let _ = CoeffContext::new(64, 1, 1, 17, 0);
    }

    #[test]
    fn carry_cert_is_conservative() {
        let fresh = CarryCert::fresh_normalized(17);
        assert_eq!(fresh.min_digit, -(1 << 16));
        assert_eq!(fresh.max_digit, (1 << 16) - 1);
        assert!(fresh.fits_in_word(64));

        // Repeated doublings widen the bound until the word is exhausted; the exhaustion
        // decision is conservative and never blocks normalization (which does not read
        // this certificate at all). Fresh 17-bit digits leave 63 - 16 = 47 doublings.
        let mut acc = fresh;
        let mut steps = 0u32;
        while acc.after_add(&acc).fits_in_word(64) {
            acc = acc.after_add(&acc);
            steps += 1;
            assert!(steps < 64);
        }
        assert_eq!(steps, 47);
        assert!(acc.fits_in_word(64));
        assert!(!acc.after_add(&acc).fits_in_word(64));

        // Raw binding starts with zero headroom.
        let raw = CarryCert::raw_binding(64);
        assert!(raw.fits_in_word(64));
        assert!(!raw.after_add(&fresh).fits_in_word(64));

        // Join covers both children; projection widens only downward.
        let a = CarryCert {
            min_digit: -5,
            max_digit: 3,
        };
        let b = CarryCert {
            min_digit: -2,
            max_digit: 9,
        };
        assert_eq!(
            a.join(&b),
            CarryCert {
                min_digit: -5,
                max_digit: 9
            }
        );
        assert_eq!(
            a.after_neg(),
            CarryCert {
                min_digit: -3,
                max_digit: 5
            }
        );
        assert_eq!(
            a.after_canonical_projection(3),
            CarryCert {
                min_digit: -12,
                max_digit: 3
            }
        );
    }
}
