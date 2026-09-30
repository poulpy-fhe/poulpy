//! Standard-ring NTT4x30 arithmetic: the negacyclic NTT.

use poulpy_hal::layouts::Standard;

use crate::reference::ntt4x30::{
    conjugate_invariant::BasisChange,
    ntt::{NttTable, NttTableInv, intt_core, ntt_core},
    primes::PrimeSetCrt4,
    vec_znx_dft::{NttAutomorphismPlan, NttPlan, NttPlanNew},
};

impl<P: PrimeSetCrt4> NttTable<P, Standard> {
    pub fn new(n: usize) -> Self {
        Self::build(n, [BasisChange::EMPTY; 4])
    }
}

impl<P: PrimeSetCrt4> NttTableInv<P, Standard> {
    pub fn new(n: usize) -> Self {
        Self::build(n, [BasisChange::EMPTY; 4])
    }
}

impl<P: PrimeSetCrt4> NttPlanNew for NttPlan<P, Standard> {
    fn new(n: usize) -> Self {
        Self {
            ntt: NttTable::<P, Standard>::new(n),
            intt: NttTableInv::<P, Standard>::new(n),
        }
    }
}

/// Forward Q120 NTT on a polynomial of `n` coefficients (reference implementation).
///
/// `data` must be a flat `u64` slice of length `4 * n` in q120b layout.
/// After the call, each group of 4 consecutive u64 values holds the NTT
/// evaluation at the corresponding point, in the same q120b layout.
///
/// # Panics
/// Panics if `data.len() < 4 * table.n`.
pub fn ntt_ref<P: PrimeSetCrt4>(table: &NttTable<P, Standard>, data: &mut [u64]) {
    ntt_core(table, data);
}

/// Inverse Q120 NTT on a polynomial of `n` coefficients (reference implementation).
///
/// `data` must be a flat `u64` slice of length `4 * n` in q120b layout
/// (the output of [`ntt_ref`]).  After the call, each group of 4 u64
/// values holds the recovered coefficient (in q120b), scaled by 1 (the
/// `n^{-1}` factor is baked into the last-pass twiddle table).
///
/// # Panics
/// Panics if `data.len() < 4 * table.n`.
pub fn intt_ref<P: PrimeSetCrt4>(table: &NttTableInv<P, Standard>, data: &mut [u64]) {
    intt_core(table, data);
}

/// Builds the [`NttAutomorphismPlan`] for ring dimension `n` and odd `p`.
///
/// The DIF NTT places output slot `i` at the evaluation point
/// `omega^{2 * bitrev(i) + 1}` mod `2n`, where `bitrev` is the bit-reversal
/// of `i` over `log2(n)` bits and `omega` is a primitive `2n`-th root.
/// The set `{1, 3, …, 2n - 1}` is closed under multiplication by any odd
/// `p`, so the action is a pure permutation — no closure trick or
/// conjugation flag is needed.
pub fn build_ntt4x30_automorphism_plan(n: usize, p: i64) -> NttAutomorphismPlan {
    assert!(n.is_power_of_two(), "n must be a power of two, got {n}");
    assert!(p & 1 == 1, "p must be odd for an R/(X^N+1) automorphism, got {p}");

    let mask = (2 * n - 1) as i64;
    let p_mod_2n = p & mask;
    let log_n = n.trailing_zeros();
    let ir = |i: u32| -> u32 { i.reverse_bits() >> (32 - log_n) };

    let mut perm: Vec<u32> = vec![0u32; n];
    for (i, mi) in perm.iter_mut().enumerate().take(n) {
        let e_out: i64 = 2 * ir(i as u32) as i64 + 1;
        let e_src: i64 = (p_mod_2n * e_out) & mask;
        let src: u32 = ((e_src - 1) >> 1) as u32;
        *mi = ir(src);
    }
    NttAutomorphismPlan { p, perm }
}
