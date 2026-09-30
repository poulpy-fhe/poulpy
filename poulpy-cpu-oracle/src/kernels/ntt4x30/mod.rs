pub mod arithmetic;
pub mod ntt;
pub mod primes;
pub mod types;
pub mod vec_znx_big;
pub mod vec_znx_dft;

pub use vec_znx_big::*;

pub trait NttDFTExecute<Table> {
    /// Apply the NTT (or iNTT) described by `table` to `data` in place.
    fn ntt_dft_execute(table: &Table, data: &mut [u64]);
}

pub trait NttFromZnx64 {
    /// Encode the `a.len()` coefficients of `a` into `res` (q120b layout).
    ///
    /// `res` must have length `4 * a.len()`.
    fn ntt_from_znx64(res: &mut [u64], a: &[i64]);
}

pub trait NttToZnx128 {
    /// Decode `a` (q120b layout, length `4 * n`) into `res` (`n` × `i128`).
    fn ntt_to_znx128(res: &mut [i128], divisor_is_n: usize, a: &[u64]);
}

pub trait NttAdd {
    /// `res[i] = a[i] + b[i]` for each CRT component.
    ///
    /// All three slices must have the same length (a multiple of 4).
    fn ntt_add(res: &mut [u64], a: &[u64], b: &[u64]);
}

pub trait NttAddAssign {
    /// `res[i] += a[i]` for each CRT component.
    fn ntt_add_assign(res: &mut [u64], a: &[u64]);
}

pub trait NttZero {
    /// Set all elements of `res` to zero.
    fn ntt_zero(res: &mut [u64]);
}

pub trait NttCopy {
    /// Copy all elements from `a` into `res`.
    fn ntt_copy(res: &mut [u64], a: &[u64]);
}

pub trait NttSub {
    /// `res[i] = a[i] - b[i]` modulo each prime for each CRT component.
    fn ntt_sub(res: &mut [u64], a: &[u64], b: &[u64]);
}

pub trait NttSubAssign {
    /// `res[i] -= a[i]` modulo each prime for each CRT component.
    fn ntt_sub_assign(res: &mut [u64], a: &[u64]);
}

pub trait NttSubNegateAssign {
    /// `res[i] = a[i] - res[i]` modulo each prime.
    fn ntt_sub_negate_assign(res: &mut [u64], a: &[u64]);
}

pub trait NttNegate {
    /// `res[i] = -a[i]` modulo each prime.
    fn ntt_negate(res: &mut [u64], a: &[u64]);
}

pub trait NttNegateAssign {
    /// `res[i] = -res[i]` modulo each prime.
    fn ntt_negate_assign(res: &mut [u64]);
}
