//! DFT-domain scalar constants for the 3-prime IFMA representation.

use super::primes::{PrimeSetNtt126Ifma, Primes42};

/// Lazy-reduction bound for DFT-domain arithmetic.
///
/// In the IFMA-native model, all NTT-domain values are in `[0, 2q)`.
/// `Q_SHIFTED_NTT126IFMA[k] = 2 * Q[k]`.  Used for conditional subtract
/// after add/sub operations to keep values in range.
/// Lane 3 is zero (padding).
pub const Q_SHIFTED_NTT126IFMA: [u64; 4] = {
    let q = <Primes42 as PrimeSetNtt126Ifma>::Q;
    [2 * q[0], 2 * q[1], 2 * q[2], 0]
};
