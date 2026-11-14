//! Opt-in security levels, checked against the tables of
//! [rlwe-params](https://github.com/jdumezy/rlwe-params).
//!
//! A table gives, for a ring of degree `N = 2^log_n` and a secret distribution,
//! the largest `log2(q)` at which every attack of its estimators costs at least
//! the target, for a discrete Gaussian error of deviation 3.2
//! ([`Noise::ENCRYPTION`](crate::Noise::ENCRYPTION)) and unlimited samples. The
//! tables are vendored in `third_party/rlwe-params` and read at build time,
//! which refuses a table of another error or sample count. Nothing in the
//! library calls this module on its own: a caller picks a [`Security`] level
//! and checks the objects it builds with it.
//!
//! # What is checked
//!
//! An object encrypted under a secret holds RLWE samples modulo `2^k`, `k` its
//! [`LWEInfos::k`]: its masks and errors are drawn at that precision, and the
//! bits below stay zero. For a key, `k` is its whole modulus, the gadget, the
//! auxiliary guard `k_aux` and the row offset `k_offset` included (see
//! [`key_k`](crate::layouts::key_k) and
//! [`GGLWEInfos::k_offset`](crate::layouts::GGLWEInfos::k_offset)), so [`Security::check`]
//! compares that one number with the bound. A secret is as exposed as the widest
//! object under it, so every object under it is checked.
//!
//! The bound is the one of the ring the secret lives in, which is not always the
//! ring of the object: a secret of a subring, embedded in a larger ring, is
//! bounded at the subring's degree. [`SecretSpec::new`] takes that degree, and
//! [`SecretSpec::of`] reads it from a secret.
//!
//! # Secrets
//!
//! | [`Distribution`] | Cell of the table |
//! | --- | --- |
//! | `TernaryFixed(h)` | sparse ternary, at the largest tabulated weight at most `h` |
//! | `BinaryFixed(h)` | sparse binary, at the largest tabulated weight at most `min(h, N - h)` |
//! | `TernaryProb(2/3)` | uniform ternary |
//! | `BinaryProb(1/2)` | uniform binary |
//!
//! Any other distribution, a degree the table does not cover, and a cell the
//! estimators failed on are [unsupported](CoreError::SecurityUnsupported). Rounding
//! a weight down is conservative, since the bounds grow with the weight (the tests
//! check it on the vendored tables). A binary secret of weight `h` is as weak as
//! one of weight `N - h`: with `1` the all-ones polynomial, `a * s = a * 1 - a * (1 - s)`
//! and `a * 1` is public, so a fixed-weight binary secret takes the smaller of the
//! two. A ternary secret has no such complement. The tables' sparse ternary secrets have
//! exactly `h / 2` coefficients equal to 1, while
//! [`TernaryFixed`](Distribution::TernaryFixed) draws each sign independently.
//! Fixing the signs only lowers the entropy of the secret, so the bound holds. A
//! secret of rank `r > 1` is bounded per component at degree `N`, which is
//! conservative for Module-LWE.

use crate::{CoreError, Distribution, GetDistribution, Result, layouts::LWEInfos};

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

include!(concat!(env!("OUT_DIR"), "/rlwe_params.rs"));

/// A security level, in bits, against the attacks of the rlwe-params estimators.
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
        TABLE_VERSION
    }

    fn cells(self) -> &'static [Cell] {
        match self {
            Self::Bits128 => CELLS_128,
        }
    }

    /// Largest modulus, in bits, of any object under `secret` at this level.
    ///
    /// # Errors
    ///
    /// [`CoreError::SecurityUnsupported`] when the table has no bound for the
    /// secret (see [Secrets](self#secrets)).
    pub fn max_log_q(self, secret: SecretSpec) -> Result<usize> {
        let unsupported = |reason| CoreError::SecurityUnsupported {
            security: self,
            secret,
            reason,
        };
        if !secret.n.is_power_of_two() {
            return Err(unsupported("the degree is not a power of two"));
        }
        let log_n = secret.n.trailing_zeros();
        let cells = self.cells().iter().filter(|cell| cell.log_n == log_n);
        let bound = match secret.dist {
            Distribution::TernaryFixed(h) if h <= secret.n => rounded_down(cells, h, |secret| match secret {
                TableSecret::SparseTernary(w) => Some(w),
                _ => None,
            }),
            Distribution::BinaryFixed(h) if h <= secret.n => rounded_down(cells, h.min(secret.n - h), |secret| match secret {
                TableSecret::SparseBinary(w) => Some(w),
                _ => None,
            }),
            Distribution::TernaryProb(p) if p == 2.0 / 3.0 => exact(cells, TableSecret::Ternary),
            Distribution::BinaryProb(0.5) => exact(cells, TableSecret::Binary),
            _ => return Err(unsupported("the table has no secret of this distribution")),
        };
        bound.ok_or_else(|| unsupported("the table has no bound for this secret at this degree"))
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
                security: self,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Noise,
        layouts::{Base2K, Degree, Dnum, GGLWELayout, GLWELayout, Rank, TorusPrecision},
    };

    const S: Security = Security::Bits128;

    fn bound(log_n: u32, dist: Distribution) -> Result<usize> {
        S.max_log_q(SecretSpec::new(1 << log_n, dist))
    }

    #[test]
    fn the_tables_assume_the_encryption_noise() {
        assert_eq!(Noise::ENCRYPTION, Noise::Gaussian { sigma: 3.2 });
    }

    #[test]
    fn bounds_grow_with_weight_and_degree() {
        // Rounding a weight down to a tabulated one relies on the first.
        let cells = S.cells();
        for a in cells {
            for b in cells {
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
        assert!(bound(16, Distribution::TernaryFixed(31)).is_err());
        // The estimators failed on weight 32 at degree 2^12, so no weight below 64 has a bound there.
        assert!(bound(12, Distribution::BinaryFixed(63)).is_err());
        assert_eq!(bound(12, Distribution::BinaryFixed(64)), Ok(53));
        // A weight above the degree is no secret of that ring.
        assert!(bound(12, Distribution::TernaryFixed(4097)).is_err());
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
        assert!(bound(12, Distribution::BinaryFixed(4096 - 32)).is_err());
        assert!(bound(12, Distribution::BinaryFixed(4096)).is_err());
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
        assert!(bound(12, Distribution::TernaryProb(0.5)).is_err());
    }

    #[test]
    fn other_secrets_and_degrees_are_unsupported() {
        for dist in [
            Distribution::BinaryBlock(8),
            Distribution::ENCAPSULATED("test"),
            Distribution::ZERO,
            Distribution::NONE,
        ] {
            assert!(bound(16, dist).is_err());
        }
        assert!(bound(11, Distribution::TernaryFixed(1024)).is_err());
        assert!(bound(18, Distribution::TernaryFixed(1024)).is_err());
        assert!(S.max_log_q(SecretSpec::new(630, Distribution::BinaryProb(0.5))).is_err());
    }

    #[test]
    fn a_key_is_checked_on_its_whole_modulus() {
        // Bound of 105 bits: 4 digits of 20 bits and a guard of 25 reach it.
        let secret = SecretSpec::new(1 << 12, Distribution::TernaryFixed(1024));
        let key = GGLWELayout {
            n: Degree(1 << 12),
            base2k: Base2K(20),
            dnum: Dnum(4),
            k_aux: TorusPrecision(25),
            rank_in: Rank(1),
            rank_out: Rank(1),
            k_digit: TorusPrecision(20),
            k_offset: TorusPrecision(0),
            stride: 1,
        };
        assert_eq!(S.check(secret, "key", &key), Ok(()));
        let wider_guard = GGLWELayout {
            k_aux: TorusPrecision(26),
            ..key
        };
        let offset = GGLWELayout {
            k_offset: TorusPrecision(1),
            ..key
        };
        for wider in [wider_guard, offset] {
            assert_eq!(
                S.check(secret, "key", &wider),
                Err(CoreError::SecurityBoundExceeded {
                    security: S,
                    secret,
                    what: "key".into(),
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
        assert!(S.check(secret, "ciphertext", &ct).is_err());
    }
}
