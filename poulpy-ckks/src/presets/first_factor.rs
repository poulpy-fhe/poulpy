//! Widths of the first `CoeffsToSlots` factor applied on the modulus raise, shared by the
//! presets, the key layouts and the reference raise.

use crate::CoeffsMeta;

/// Fewest bits the raise keeps below the raised precision while it applies
/// the factor: the bits a division by a diagonal's scale leaves below the
/// raised ciphertext's last one, rounded once with the sum of the terms.
pub const FIRST_FACTOR_GUARD_BITS: usize = 16;

/// Limbs a raise to `k_large` bits works at, and the bits its working width
/// keeps below them. The first-factor keys' rows end at that width.
pub fn first_factor_work(base2k: usize, k_large: usize) -> (usize, usize) {
    let work = (k_large + FIRST_FACTOR_GUARD_BITS).div_ceil(base2k);
    (work, work * base2k - k_large)
}

/// Bits of the small ciphertext a first-factor key switches, for a raise into
/// `k_large` bits of a ciphertext that fills `raised` bits once lifted: those,
/// the guard of the working width below them, and above them the growth of a
/// product by a diagonal of `diagonal_growth` bits over `2^log_n` terms.
pub fn first_factor_small_bits(base2k: usize, log_n: usize, k_large: usize, raised: usize, diagonal_growth: usize) -> usize {
    let (_, guard) = first_factor_work(base2k, k_large);
    raised + guard + diagonal_growth + log_n + 2
}

/// Bits a product by a diagonal of a transform of `meta`, encoded at radix
/// `base2k`, adds once divided by its scale: the diagonals are allocated to
/// whole limbs.
pub fn first_factor_diagonal_growth(base2k: usize, meta: &CoeffsMeta) -> usize {
    meta.k.as_usize().div_ceil(base2k) * base2k - meta.log_delta()
}
