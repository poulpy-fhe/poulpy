use std::fmt::Display;

/// One point of the libm benchmark sweep: one ciphertext of degree `n` at radix `base2k` and scale `2^log_delta`.
///
/// A runner sizes the ciphertext width from its plan: `consumed_bits(log_delta)` plus `2 log_delta` of result headroom
/// and two limbs of guard, rounded up to a limb. Tensor keys cover that width with digits of `dsize` limbs.
#[derive(Clone, Copy)]
pub struct LibmBenchParams {
    pub n: usize,
    pub base2k: usize,
    pub log_delta: usize,
    pub dsize: usize,
}

impl Display for LibmBenchParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "(n={},base2k={},log_delta={},dsize={})",
            self.n, self.base2k, self.log_delta, self.dsize
        )
    }
}

/// The full sweep, one point per ring degree from `2^13` to `2^16`.
pub fn default_bench_params_libm() -> Vec<LibmBenchParams> {
    (13..=16)
        .map(|log_n| LibmBenchParams {
            n: 1 << log_n,
            base2k: 52,
            log_delta: 40,
            dsize: 1,
        })
        .collect()
}
