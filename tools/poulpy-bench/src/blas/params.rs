use std::fmt::Display;

use poulpy_hal::layouts::{Ring, Standard};

/// One point of the BLAS benchmark sweep: `d x d` blocks on a ring of degree `n`, whose `n / 2` slots hold
/// `n / (2 d^2)` matrices per ciphertext on the standard ring, and whose `n` slots hold `n / d^2` on the
/// conjugate-invariant one.
///
/// `k` is the torus width of a fresh block, `log_delta` the scale of the encrypted entries, and `p` the scale of the
/// context's maps and of the public matrices (`p_B`), each of which consumes its scale. At the defaults a ciphertext
/// by ciphertext product spends `2p + log_delta = 120` of the `k = 208` bits and leaves a result of 88, the public
/// products and the transpose spend `p` once (`Diagonal`, transpose) or twice (`Jkls`), so every operation fits.
/// Rotation and tensor keys cover a fresh block with `ceil(k / (dsize base2k))` digits and a guard of
/// `dsize base2k + log2 n` bits.
#[derive(Clone, Copy)]
pub struct BlasBenchParams {
    pub n: usize,
    pub log_d: usize,
    pub base2k: usize,
    pub k: usize,
    pub log_delta: usize,
    pub p: usize,
    pub dsize: usize,
}

impl BlasBenchParams {
    /// The block dimension `d`.
    pub fn d(&self) -> usize {
        1 << self.log_d
    }

    /// Complex slots of a standard-ring block.
    pub fn slots(&self) -> usize {
        self.slots_on::<Standard>()
    }

    /// Matrices a standard-ring ciphertext holds, `slots / d^2`: one means every matrix slot repeats the same matrix.
    pub fn batch(&self) -> usize {
        self.batch_on::<Standard>()
    }

    /// Slots of a block on ring `R`: `n / 2` complex ones on the standard ring, `n` real ones on the
    /// conjugate-invariant one.
    pub fn slots_on<R: Ring>(&self) -> usize {
        self.n * R::CYCLOTOMIC_ORDER_FACTOR as usize / 4
    }

    /// Matrices a ciphertext on ring `R` holds, `slots_on / d^2`.
    pub fn batch_on<R: Ring>(&self) -> usize {
        self.slots_on::<R>() >> (2 * self.log_d)
    }
}

impl Display for BlasBenchParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "(n={},d={},base2k={},k={},log_delta={},p={},dsize={})",
            self.n,
            self.d(),
            self.base2k,
            self.k,
            self.log_delta,
            self.p,
            self.dsize
        )
    }
}

/// The full sweep, `(log_n, log_d)` in ring-degree order. The standard tier keeps `d = n / 256`, one block per
/// ring degree holding 4, 2 and 1 matrices per ciphertext; the full tier adds `n = 2^15, d = 64` and `n = 2^16`.
pub fn default_bench_params_blas() -> Vec<BlasBenchParams> {
    [(13, 5), (14, 6), (15, 6), (15, 7), (16, 6), (16, 7)]
        .into_iter()
        .map(|(log_n, log_d)| BlasBenchParams {
            n: 1 << log_n,
            log_d,
            base2k: 52,
            k: 208,
            log_delta: 40,
            p: 40,
            dsize: 1,
        })
        .collect()
}
