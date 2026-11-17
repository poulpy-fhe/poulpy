//! Opt-in security validation against tables of secure moduli, by default those of
//! [rlwe-params](https://github.com/jdumezy/rlwe-params).
//!
//! Nothing in the library calls this module on its own: the distribution of a
//! secret, and the moduli of the objects under it, are the caller's to choose. A
//! caller that wants a level validates its layouts before generating keys from
//! them.
//!
//! # Validating
//!
//! A [`SecurityTable`] gives, for a target in bits and a [`SecretSpec`], the
//! largest `log2(q)` of any object under that secret. [`RlweParams`] is the
//! vendored rlwe-params table, and any other table implements the trait.
//!
//! A layout that defines moduli implements [`ValidateSecurity`]: it lists each
//! [`Modulus`] it holds with the secret its samples are encrypted under, given the
//! secrets it is used with, and [`validate`] checks every entry of the list against
//! the table. [`ValidateSecurity::validate_security_128`] validates at 128 bits,
//! [`ValidateSecurity::validate_security_custom`] at any target the table has.
//!
//! ```
//! use poulpy_core::{
//!     Distribution,
//!     layouts::{Base2K, Degree, GLWELayout, Rank, TorusPrecision},
//!     security::{RlweParams, SecretSpec, ValidateSecurity},
//! };
//!
//! let ct = GLWELayout {
//!     n: Degree(1 << 15),
//!     base2k: Base2K(52),
//!     k: TorusPrecision(832),
//!     rank: Rank(1),
//! };
//! let secret = SecretSpec::new(1 << 15, Distribution::TernaryFixed(1024));
//! assert!(ct.validate_security_128(&secret, &RlweParams).is_ok());
//! ```
//!
//! # What is checked
//!
//! An object encrypted under a secret holds RLWE samples modulo `2^k`, `k` its
//! [`LWEInfos::k`]: its masks and errors are drawn at that precision, and the
//! bits below stay zero. For a key, `k` is its whole modulus, the gadget, the
//! auxiliary guard `k_aux` and the row offset `k_offset` included (see
//! [`key_k`](crate::layouts::key_k) and
//! [`GGLWEInfos::k_offset`](crate::layouts::GGLWEInfos::k_offset)), so [`Modulus::of`]
//! takes that one number. A secret is as exposed as the widest object under it,
//! so every object under it is listed.
//!
//! The bound is the one of the ring the secret lives in, which is not always the
//! ring of the object: a secret of a subring, embedded in a larger ring, is
//! bounded at the subring's degree. [`SecretSpec::new`] takes that degree, and
//! [`SecretSpec::of`] reads it from a secret.
//!
//! # The rlwe-params table
//!
//! [`RlweParams`] gives the largest `log2(q)` at which every attack of the
//! rlwe-params estimators costs at least the target. The table is vendored in
//! `third_party/rlwe-params` and committed as Rust in `rlwe_params_128.rs`, which a
//! test rebuilds from it, refusing a table of another error or sample count. It
//! covers:
//!
//! - a target of 128 bits only,
//! - a discrete Gaussian error of deviation 3.2
//!   ([`Noise::ENCRYPTION`](crate::Noise::ENCRYPTION)) and unlimited samples,
//! - ring-LWE at degrees `2^12` to `2^17`,
//! - the secrets of the table below, each component of a secret of rank `r > 1`
//!   bounded on its own at degree `N`, which is stricter than Module-LWE needs.
//!
//! It does not cover plain LWE secrets, rings below `2^12`, nor structured secrets
//! such as [`BinaryBlock`](Distribution::BinaryBlock), and the LWE layouts do not
//! implement [`ValidateSecurity`].
//!
//! | [`Distribution`] | Cell of the table |
//! | --- | --- |
//! | `TernaryFixed(h)` | sparse ternary, at the largest tabulated weight at most `h` |
//! | `BinaryFixed(h)` | sparse binary, at the largest tabulated weight at most `min(h, N - h)` |
//! | `TernaryProb(2/3)` | uniform ternary |
//! | `BinaryProb(1/2)` | uniform binary |
//! | other `TernaryProb(p)` | sparse ternary, at the largest tabulated weight at most `pN - t` |
//! | other `BinaryProb(p)` | sparse binary, at the largest tabulated weight at most `min(pN, (1 - p)N) - t'` |
//!
//! Any other distribution, a degree the table does not cover, and a cell the
//! estimators failed on are [unsupported](CoreError::SecurityUnsupported). Rounding
//! a weight down is conservative, since the bounds grow with the weight (the tests
//! check it on the vendored table). A binary secret of weight `h` is as weak as
//! one of weight `N - h`: with `1` the all-ones polynomial, `a * s = a * 1 - a * (1 - s)`
//! and `a * 1` is public, so a fixed-weight binary secret takes the smaller of the
//! two. A ternary secret has no such complement. The table's sparse ternary secrets have
//! exactly `h / 2` coefficients equal to 1, while
//! [`TernaryFixed`](Distribution::TernaryFixed) draws each sign independently.
//! Fixing the signs only lowers the entropy of the secret, so the bound holds.
//!
//! A secret of `TernaryProb(p)` or `BinaryProb(p)` draws each coefficient
//! independently, so given its weight `w` it is a fixed-weight secret of weight
//! `w`. By Hoeffding's inequality its weight is below `pN - t` with probability at
//! most `exp(-2t^2 / N)`, and `t = sqrt(N * target * ln(2) / 2)` makes that
//! `2^-target`. A binary secret bounds both tails, each at `2^-(target + 1)`, which
//! gives `t'`.

use crate::{
    CoreError, Distribution, GetDistribution, Result,
    layouts::{
        GGLWELayout, GGLWEToGGSWKeyLayout, GGSWLayout, GLWEAutomorphismKeyLayout, GLWELayout, GLWEPublicKeyLayout,
        GLWESwitchingKeyLayout, GLWETensorKeyLayout, LWEInfos, LWEToGLWEKeyLayout,
    },
};

/// A cell of a table: the largest `log2(q)` for a secret at degree `2^log_n`,
/// `None` where the table has no result.
#[derive(Clone, Copy, Debug)]
struct Cell {
    log_n: u32,
    secret: TableSecret,
    max_log_q: Option<usize>,
}

/// A secret of the tables, under their conventions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TableSecret {
    /// Uniform on `{-1, 0, 1}`.
    Ternary,
    /// Uniform on `{0, 1}`.
    Binary,
    /// `h` coefficients in `{-1, 1}`, the rest 0.
    SparseTernary(usize),
    /// `h` coefficients equal to 1, the rest 0.
    SparseBinary(usize),
}

#[cfg(test)]
mod generate;
mod rlwe_params_128;

use rlwe_params_128::{CELLS_128, TABLE_VERSION};

/// Bounds on the modulus of the objects under a secret.
pub trait SecurityTable {
    /// Largest modulus, in bits, of any object under `secret` for `target_bits`
    /// bits of security.
    ///
    /// # Errors
    ///
    /// [`CoreError::SecurityUnsupported`] when the table has no bound for the
    /// target or the secret.
    fn max_log_q(&self, target_bits: usize, secret: SecretSpec) -> Result<usize>;
}

/// The vendored rlwe-params table, see [the module](self#the-rlwe-params-table).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct RlweParams;

impl RlweParams {
    /// The rlwe-params release the table was copied from.
    pub const VERSION: &'static str = TABLE_VERSION;
}

impl SecurityTable for RlweParams {
    fn max_log_q(&self, target_bits: usize, secret: SecretSpec) -> Result<usize> {
        let unsupported = |reason| CoreError::SecurityUnsupported {
            target_bits,
            secret,
            reason,
        };
        let cells = match target_bits {
            128 => CELLS_128,
            _ => return Err(unsupported("the table has no bounds for this target")),
        };
        if !secret.n.is_power_of_two() {
            return Err(unsupported("the degree is not a power of two"));
        }
        let n = secret.n;
        let log_n = n.trailing_zeros();
        let cells = cells.iter().filter(|cell| cell.log_n == log_n);
        let sparse_ternary = |secret| match secret {
            TableSecret::SparseTernary(w) => Some(w),
            _ => None,
        };
        let sparse_binary = |secret| match secret {
            TableSecret::SparseBinary(w) => Some(w),
            _ => None,
        };
        let bound = match secret.dist {
            Distribution::TernaryFixed(h) | Distribution::BinaryFixed(h) if h > n => {
                return Err(unsupported("the weight exceeds the degree"));
            }
            Distribution::TernaryProb(p) | Distribution::BinaryProb(p) if !(0.0..=1.0).contains(&p) => {
                return Err(unsupported("the probability is not in [0, 1]"));
            }
            Distribution::TernaryFixed(h) => rounded_down(cells, h, sparse_ternary),
            Distribution::BinaryFixed(h) => rounded_down(cells, h.min(n - h), sparse_binary),
            Distribution::TernaryProb(p) if p == 2.0 / 3.0 => exact(cells, TableSecret::Ternary),
            Distribution::BinaryProb(0.5) => exact(cells, TableSecret::Binary),
            Distribution::TernaryProb(p) => rounded_down(cells, least_weight(n, p, target_bits), sparse_ternary),
            Distribution::BinaryProb(p) => {
                let least = least_weight(n, p, target_bits + 1).min(least_weight(n, 1.0 - p, target_bits + 1));
                rounded_down(cells, least, sparse_binary)
            }
            _ => return Err(unsupported("the table has no secret of this distribution")),
        };
        bound.ok_or_else(|| unsupported("the table has no bound for this secret at this degree"))
    }
}

/// A weight that `n` coefficients, each non-zero with probability `p`, reach
/// except with probability at most `2^-bits`, by Hoeffding's inequality.
fn least_weight(n: usize, p: f64, bits: usize) -> usize {
    let n = n as f64;
    let t = (n * bits as f64 * std::f64::consts::LN_2 / 2.0).sqrt();
    (p * n - t).floor().max(0.0) as usize
}

/// The bound of the largest tabulated weight at most `h` with a result, among
/// the cells whose secret `weight` maps to a weight.
fn rounded_down<'a>(
    cells: impl Iterator<Item = &'a Cell>,
    h: usize,
    weight: impl Fn(TableSecret) -> Option<usize>,
) -> Option<usize> {
    cells
        .filter_map(|cell| Some((weight(cell.secret)?, cell.max_log_q?)))
        .filter(|&(w, _)| w <= h)
        .max_by_key(|&(w, _)| w)
        .map(|(_, max_log_q)| max_log_q)
}

fn exact<'a>(mut cells: impl Iterator<Item = &'a Cell>, secret: TableSecret) -> Option<usize> {
    cells.find(|cell| cell.secret == secret).and_then(|cell| cell.max_log_q)
}

/// A secret as the tables see it: the degree of the ring it lives in and its
/// distribution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SecretSpec {
    /// Degree of the ring of the secret, which for a secret embedded from a
    /// subring is the subring's.
    pub n: usize,
    /// Distribution of each coefficient, per component.
    pub dist: Distribution,
}

impl SecretSpec {
    /// A secret of `dist` in a ring of degree `n`.
    pub const fn new(n: usize, dist: Distribution) -> Self {
        Self { n, dist }
    }

    /// The degree and distribution recorded on `sk`. An embedded secret records
    /// the degree of the larger ring: build its spec with [`Self::new`].
    pub fn of<S: LWEInfos + GetDistribution + ?Sized>(sk: &S) -> Self {
        Self::new(sk.n().as_usize(), *sk.dist())
    }
}

/// A modulus of `log_q` bits under `secret`, named by `what`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Modulus {
    /// The object, named in errors.
    pub what: &'static str,
    /// The secret its samples are encrypted under.
    pub secret: SecretSpec,
    /// Its modulus in bits.
    pub log_q: usize,
}

impl Modulus {
    /// The modulus of `obj` under `secret`: its [`LWEInfos::k`], auxiliary guard
    /// and row offset included.
    pub fn of<I: LWEInfos + ?Sized>(what: &'static str, secret: SecretSpec, obj: &I) -> Self {
        Self {
            what,
            secret,
            log_q: obj.k().as_usize(),
        }
    }
}

/// Checks every modulus of `moduli` against the bound `table` gives its secret
/// for `target_bits` bits of security, in order.
///
/// # Errors
///
/// The first [`CoreError::SecurityUnsupported`] of the table, or
/// [`CoreError::SecurityBoundExceeded`] for the first modulus above its bound.
pub fn validate<T: SecurityTable + ?Sized>(target_bits: usize, table: &T, moduli: &[Modulus]) -> Result<()> {
    for modulus in moduli {
        let max_log_q = table.max_log_q(target_bits, modulus.secret)?;
        if modulus.log_q > max_log_q {
            return Err(CoreError::SecurityBoundExceeded {
                target_bits,
                secret: modulus.secret,
                what: modulus.what.to_string(),
                log_q: modulus.log_q,
                max_log_q,
            });
        }
    }
    Ok(())
}

/// A layout whose moduli a [`SecurityTable`] validates.
pub trait ValidateSecurity {
    /// The secrets the layout is used with.
    type Secrets: ?Sized;

    /// Every modulus of the layout, under the secret of `secrets` its samples are
    /// encrypted with.
    fn moduli(&self, secrets: &Self::Secrets) -> Vec<Modulus>;

    /// [`Self::validate_security_custom`] at 128 bits.
    ///
    /// # Errors
    ///
    /// As [`validate`].
    fn validate_security_128<T: SecurityTable + ?Sized>(&self, secrets: &Self::Secrets, table: &T) -> Result<()> {
        self.validate_security_custom(128, secrets, table)
    }

    /// [`validate`] on [`Self::moduli`] for `target_bits` bits of security.
    ///
    /// # Errors
    ///
    /// As [`validate`].
    fn validate_security_custom<T: SecurityTable + ?Sized>(
        &self,
        target_bits: usize,
        secrets: &Self::Secrets,
        table: &T,
    ) -> Result<()> {
        validate(target_bits, table, &self.moduli(secrets))
    }
}

macro_rules! impl_validate_security {
    ($($layout:ty => $what:literal),* $(,)?) => {
        $(
            impl ValidateSecurity for $layout {
                type Secrets = SecretSpec;

                fn moduli(&self, secret: &SecretSpec) -> Vec<Modulus> {
                    vec![Modulus::of($what, *secret, self)]
                }
            }
        )*
    };
}

impl_validate_security!(
    GLWELayout => "GLWE ciphertext",
    GLWEPublicKeyLayout => "public key",
    GGLWELayout => "GGLWE",
    GGSWLayout => "GGSW",
    GLWESwitchingKeyLayout => "switching key",
    GLWEAutomorphismKeyLayout => "automorphism key",
    GLWETensorKeyLayout => "tensor key",
    GGLWEToGGSWKeyLayout => "GGLWE-to-GGSW key",
    LWEToGLWEKeyLayout => "LWE-to-GLWE key",
);

/// A security level, in bits, against the [`RlweParams`] table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Security {
    /// 128 bits.
    Bits128,
}

impl Security {
    /// The level in bits.
    pub const fn bits(self) -> usize {
        match self {
            Self::Bits128 => 128,
        }
    }

    /// The rlwe-params release the bounds come from.
    pub const fn table_version(self) -> &'static str {
        RlweParams::VERSION
    }

    /// [`RlweParams`] at this level.
    ///
    /// # Errors
    ///
    /// As [`SecurityTable::max_log_q`].
    pub fn max_log_q(self, secret: SecretSpec) -> Result<usize> {
        RlweParams.max_log_q(self.bits(), secret)
    }

    /// Checks that a modulus of `log_q` bits under `secret` meets this level,
    /// `what` naming the object in the error.
    ///
    /// # Errors
    ///
    /// [`CoreError::SecurityBoundExceeded`] when `log_q` is above
    /// [`Self::max_log_q`], or the error of [`Self::max_log_q`].
    pub fn check_log_q(self, secret: SecretSpec, what: &str, log_q: usize) -> Result<()> {
        let max_log_q = self.max_log_q(secret)?;
        if log_q > max_log_q {
            return Err(CoreError::SecurityBoundExceeded {
                target_bits: self.bits(),
                secret,
                what: what.to_string(),
                log_q,
                max_log_q,
            });
        }
        Ok(())
    }

    /// Checks that `obj`, encrypted under `secret`, meets this level: its
    /// modulus [`LWEInfos::k`], auxiliary guard and row offset included, is at
    /// most [`Self::max_log_q`]. `what` names the object in the error.
    ///
    /// # Errors
    ///
    /// As [`Self::check_log_q`].
    pub fn check<I: LWEInfos + ?Sized>(self, secret: SecretSpec, what: &str, obj: &I) -> Result<()> {
        self.check_log_q(secret, what, obj.k().as_usize())
    }
}

impl std::fmt::Display for Security {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}-bit security", self.bits())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Noise,
        layouts::{Base2K, Degree, Dnum, Rank, TorusPrecision},
    };

    fn bound(log_n: u32, dist: Distribution) -> Result<usize> {
        RlweParams.max_log_q(128, SecretSpec::new(1 << log_n, dist))
    }

    /// The reason of a [`CoreError::SecurityUnsupported`], panicking on anything else.
    fn reason(result: Result<usize>) -> &'static str {
        match result {
            Err(CoreError::SecurityUnsupported {
                target_bits: 128,
                reason,
                ..
            }) => reason,
            other => panic!("expected an unsupported secret, got {other:?}"),
        }
    }

    #[test]
    fn the_tables_assume_the_encryption_noise() {
        assert_eq!(Noise::ENCRYPTION, Noise::Gaussian { sigma: 3.2 });
    }

    #[test]
    fn bounds_grow_with_weight_and_degree() {
        // Rounding a weight down to a tabulated one relies on the first.
        for a in CELLS_128 {
            for b in CELLS_128 {
                let (Some(qa), Some(qb)) = (a.max_log_q, b.max_log_q) else {
                    continue;
                };
                let le = match (a.secret, b.secret) {
                    (TableSecret::SparseTernary(wa), TableSecret::SparseTernary(wb))
                    | (TableSecret::SparseBinary(wa), TableSecret::SparseBinary(wb)) => wa <= wb && a.log_n <= b.log_n,
                    (sa, sb) => sa == sb && a.log_n <= b.log_n,
                };
                assert!(!le || qa <= qb, "{a:?} has a larger bound than {b:?}");
            }
        }
    }

    #[test]
    fn bounds_of_the_bootstrapping_secrets() {
        assert_eq!(bound(16, Distribution::TernaryFixed(1024)), Ok(1707));
        assert_eq!(bound(16, Distribution::TernaryFixed(32)), Ok(342));
        assert_eq!(bound(15, Distribution::TernaryFixed(1024)), Ok(851));
        assert_eq!(bound(15, Distribution::TernaryFixed(32)), Ok(162));
    }

    #[test]
    fn weights_round_down_to_a_tabulated_one() {
        assert_eq!(
            bound(16, Distribution::TernaryFixed(1000)),
            bound(16, Distribution::TernaryFixed(512))
        );
        assert_eq!(
            bound(16, Distribution::TernaryFixed(40_000)),
            bound(16, Distribution::TernaryFixed(2048))
        );
        assert_eq!(
            reason(bound(16, Distribution::TernaryFixed(31))),
            "the table has no bound for this secret at this degree"
        );
        // The estimators failed on weight 32 at degree 2^12, so no weight below 64 has a bound there.
        assert_eq!(
            reason(bound(12, Distribution::BinaryFixed(63))),
            "the table has no bound for this secret at this degree"
        );
        assert_eq!(bound(12, Distribution::BinaryFixed(64)), Ok(53));
        // A weight above the degree is no secret of that ring.
        assert_eq!(
            reason(bound(12, Distribution::TernaryFixed(4097))),
            "the weight exceeds the degree"
        );
    }

    #[test]
    fn a_dense_binary_secret_takes_the_bound_of_its_complement() {
        // `N - 64` ones are 64 zeros: the weight-64 bound, not the weight-2048 one.
        assert_eq!(bound(12, Distribution::BinaryFixed(4096 - 64)), Ok(53));
        assert_eq!(
            bound(16, Distribution::BinaryFixed(65536 - 32)),
            bound(16, Distribution::BinaryFixed(32))
        );
        // The complement of weight 32 at degree 2^12 has no bound, and the all-ones secret none at all.
        assert_eq!(
            reason(bound(12, Distribution::BinaryFixed(4096 - 32))),
            "the table has no bound for this secret at this degree"
        );
        assert_eq!(
            reason(bound(12, Distribution::BinaryFixed(4096))),
            "the table has no bound for this secret at this degree"
        );
        // A ternary secret has no complement: its weight rounds down as it is.
        assert_eq!(
            bound(12, Distribution::TernaryFixed(4096 - 64)),
            bound(12, Distribution::TernaryFixed(2048))
        );
    }

    #[test]
    fn uniform_secrets_take_their_own_cells() {
        assert_eq!(bound(12, Distribution::TernaryProb(2.0 / 3.0)), Ok(106));
        assert_eq!(bound(12, Distribution::BinaryProb(0.5)), Ok(105));
    }

    #[test]
    fn probabilistic_secrets_take_the_weight_they_reach_but_with_negligible_probability() {
        // At 2^12, t = sqrt(4096 * 128 * ln 2 / 2) is about 426, so a weight of 2048 on average is at least 1621.
        assert_eq!(least_weight(1 << 12, 0.5, 128), 1621);
        assert_eq!(
            bound(12, Distribution::TernaryProb(0.5)),
            bound(12, Distribution::TernaryFixed(1024))
        );
        assert_eq!(
            bound(15, Distribution::TernaryProb(0.5)),
            bound(15, Distribution::TernaryFixed(2048))
        );
        // Denser than uniform, the weight still rounds down to a tabulated one.
        assert_eq!(
            bound(15, Distribution::TernaryProb(0.9)),
            bound(15, Distribution::TernaryFixed(2048))
        );
        // A binary secret takes the lighter of its weight and that of its complement.
        assert_eq!(least_weight(1 << 12, 0.25, 129), 596);
        assert_eq!(
            bound(12, Distribution::BinaryProb(0.25)),
            bound(12, Distribution::BinaryFixed(512))
        );
        assert_eq!(
            bound(12, Distribution::BinaryProb(0.75)),
            bound(12, Distribution::BinaryFixed(512))
        );
        // Too sparse to reach any tabulated weight.
        for dist in [Distribution::TernaryProb(0.01), Distribution::BinaryProb(0.99)] {
            assert_eq!(
                reason(bound(12, dist)),
                "the table has no bound for this secret at this degree"
            );
        }
        for p in [-0.1, 1.5, f64::NAN] {
            assert_eq!(
                reason(bound(12, Distribution::TernaryProb(p))),
                "the probability is not in [0, 1]"
            );
        }
    }

    #[test]
    fn other_secrets_degrees_and_targets_are_unsupported() {
        for dist in [
            Distribution::BinaryBlock(8),
            Distribution::ENCAPSULATED("test"),
            Distribution::ZERO,
            Distribution::NONE,
        ] {
            assert_eq!(reason(bound(16, dist)), "the table has no secret of this distribution");
        }
        assert_eq!(
            reason(bound(11, Distribution::TernaryFixed(1024))),
            "the table has no bound for this secret at this degree"
        );
        assert_eq!(
            reason(bound(18, Distribution::TernaryFixed(1024))),
            "the table has no bound for this secret at this degree"
        );
        assert_eq!(
            reason(RlweParams.max_log_q(128, SecretSpec::new(630, Distribution::BinaryProb(0.5)))),
            "the degree is not a power of two"
        );
        assert!(matches!(
            RlweParams.max_log_q(192, SecretSpec::new(1 << 16, Distribution::TernaryFixed(1024))),
            Err(CoreError::SecurityUnsupported {
                target_bits: 192,
                reason: "the table has no bounds for this target",
                ..
            })
        ));
    }

    fn key(k_aux: u32, k_offset: u32) -> GGLWELayout {
        GGLWELayout {
            n: Degree(1 << 12),
            base2k: Base2K(20),
            dnum: Dnum(4),
            k_aux: TorusPrecision(k_aux),
            rank_in: Rank(1),
            rank_out: Rank(1),
            k_digit: TorusPrecision(20),
            k_offset: TorusPrecision(k_offset),
            stride: 1,
        }
    }

    #[test]
    fn a_key_is_validated_on_its_whole_modulus() {
        // Bound of 105 bits: 4 digits of 20 bits and a guard of 25 reach it.
        let secret = SecretSpec::new(1 << 12, Distribution::TernaryFixed(1024));
        assert_eq!(key(25, 0).validate_security_128(&secret, &RlweParams), Ok(()));
        for wider in [key(26, 0), key(25, 1)] {
            assert_eq!(
                wider.validate_security_128(&secret, &RlweParams),
                Err(CoreError::SecurityBoundExceeded {
                    target_bits: 128,
                    secret,
                    what: "GGLWE".into(),
                    log_q: 106,
                    max_log_q: 105,
                })
            );
        }
        let ct = GLWELayout {
            n: Degree(1 << 12),
            base2k: Base2K(20),
            k: TorusPrecision(106),
            rank: Rank(1),
        };
        assert_eq!(
            ct.validate_security_128(&secret, &RlweParams),
            Err(CoreError::SecurityBoundExceeded {
                target_bits: 128,
                secret,
                what: "GLWE ciphertext".into(),
                log_q: 106,
                max_log_q: 105,
            })
        );
    }

    /// A table bounding every secret at `bits - 20` bits, for any target.
    struct Linear;

    impl SecurityTable for Linear {
        fn max_log_q(&self, target_bits: usize, _: SecretSpec) -> Result<usize> {
            Ok(target_bits - 20)
        }
    }

    #[test]
    fn validate_takes_any_table_and_target_and_stops_at_the_first_failure() {
        let secret = SecretSpec::new(1 << 12, Distribution::NONE);
        assert_eq!(key(25, 0).validate_security_custom(125, &secret, &Linear), Ok(()));
        let moduli = [
            Modulus::of("first", secret, &key(25, 0)),
            Modulus::of("second", secret, &key(26, 0)),
            Modulus::of("third", secret, &key(27, 0)),
        ];
        assert_eq!(
            validate(125, &Linear, &moduli),
            Err(CoreError::SecurityBoundExceeded {
                target_bits: 125,
                secret,
                what: "second".into(),
                log_q: 106,
                max_log_q: 105,
            })
        );
        assert_eq!(validate(127, &Linear, &moduli), Ok(()));
        // Through a trait object, as a caller choosing its table at run time would.
        let table: &dyn SecurityTable = &Linear;
        assert_eq!(validate(127, table, &moduli), Ok(()));
    }
}
