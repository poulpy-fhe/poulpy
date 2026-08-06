//! DFT-domain scalar constants and type markers for the 3-prime IFMA representation.

use super::primes::Primes42;
use bytemuck::{Pod, Zeroable};
use rand_distr::num_traits::Zero;
use std::{fmt, ops::Add};

/// 24-byte prepared-domain scalar marker for the 3-prime IFMA backend.
///
/// `VecZnxDft<NTT3x42Ifma>` stores each limb as three prime planes of `n`
/// consecutive `u64` values. This type exists for HAL sizing only; interpreting
/// a limb as `n` consecutive `Q126Scalar` values is not semantically correct.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Q126Scalar(pub [u64; 3]);

// SAFETY: Q126Scalar is #[repr(C)] with one [u64; 3] field. All bit patterns
// are valid and there is no padding.
unsafe impl Zeroable for Q126Scalar {}
unsafe impl Pod for Q126Scalar {}

/// `Q126Scalar` is an identity + sizing contract over planar storage, not an
/// element view: buffers tagged with it store three prime planes of `n`
/// consecutive `u64` values per limb. Cross-backend interchange requires both
/// matching word types and the relevant layout-compatibility marker.
impl poulpy_hal::layouts::DftWord for Q126Scalar {}

impl fmt::Display for Q126Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:#x}, {:#x}, {:#x}]", self.0[0], self.0[1], self.0[2])
    }
}

impl Add for Q126Scalar {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self([
            self.0[0].wrapping_add(rhs.0[0]),
            self.0[1].wrapping_add(rhs.0[1]),
            self.0[2].wrapping_add(rhs.0[2]),
        ])
    }
}

impl Zero for Q126Scalar {
    fn zero() -> Self {
        Self([0; 3])
    }

    fn is_zero(&self) -> bool {
        self.0 == [0; 3]
    }
}

/// Lazy-reduction bound for DFT-domain arithmetic.
///
/// In the IFMA-native model, all NTT-domain values are in `[0, 2q)`.
/// `Q_SHIFTED_NTT3X42IFMA[k] = 2 * Q[k]`.  Used for conditional subtract
/// after add/sub operations to keep values in range.
#[allow(dead_code)]
pub const Q_SHIFTED_NTT3X42IFMA: [u64; 4] = {
    let q = <Primes42 as poulpy_cpu_ref::reference::ntt4x30::primes::PrimeSet>::Q;
    [2 * q[0], 2 * q[1], 2 * q[2], 0]
};
