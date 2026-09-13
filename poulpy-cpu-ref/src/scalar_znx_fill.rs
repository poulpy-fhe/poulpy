//! Host kernels for the secret distributions of [`poulpy_core`].
//!
//! These are the primitives [`impl_sampling_host!`](crate::impl_sampling_host)
//! dispatches to for each variant of
//! [`Distribution`](poulpy_core::Distribution). They run on the CPU backend's
//! own buffer, which is host memory; a backend whose buffers are not host
//! memory implements
//! [`SamplingImpl`](poulpy_core::oep::SamplingImpl) some other way.

use poulpy_hal::{
    layouts::{HostDataMut, ScalarZnx, ZnxViewMut, ZnxWord},
    source::Source,
};
use rand::{Rng, seq::SliceRandom};
use rand_distr::{Distribution as _, weighted::WeightedIndex};

/// Sampling of the secret-key distributions onto a host `ScalarZnx` column.
///
/// One method per sampleable variant of
/// [`Distribution`](poulpy_core::Distribution).
pub trait ScalarZnxFill {
    /// Fills column `col` with ternary values `{-1, 0, 1}` where each
    /// non-zero entry appears with total probability `prob` (split equally
    /// between `-1` and `+1`).
    fn fill_ternary_prob(&mut self, col: usize, prob: f64, source: &mut Source);

    /// Fills column `col` with exactly `hw` non-zero ternary values `{-1, +1}`
    /// at uniformly random positions; the remaining `N - hw` coefficients are zero.
    ///
    /// # Panics
    ///
    /// Panics if `hw > N`.
    fn fill_ternary_hw(&mut self, col: usize, hw: usize, source: &mut Source);

    /// Fills column `col` with binary values `{0, 1}` where each entry is `1`
    /// with probability `prob`.
    fn fill_binary_prob(&mut self, col: usize, prob: f64, source: &mut Source);

    /// Fills column `col` with exactly `hw` ones at uniformly random positions;
    /// the remaining `N - hw` coefficients are zero.
    ///
    /// # Panics
    ///
    /// Panics if `hw > N`.
    fn fill_binary_hw(&mut self, col: usize, hw: usize, source: &mut Source);

    /// Fills column `col` with a block-sparse binary pattern: the polynomial is
    /// partitioned into blocks of `block_size` coefficients, and each block
    /// independently receives at most one `1` at a uniformly random position
    /// (or no `1` at all with probability `1 / (block_size + 1)`).
    ///
    /// # Panics
    ///
    /// Panics if `N` is not a multiple of `block_size`.
    fn fill_binary_block(&mut self, col: usize, block_size: usize, source: &mut Source);
}

impl<D: HostDataMut, W: ZnxWord> ScalarZnxFill for ScalarZnx<D, W> {
    fn fill_ternary_prob(&mut self, col: usize, prob: f64, source: &mut Source) {
        let choices: [W; 3] = [W::from_i64(-1), W::zero(), W::from_i64(1)];
        let weights: [f64; 3] = [prob / 2.0, 1.0 - prob, prob / 2.0];
        let dist: WeightedIndex<f64> = WeightedIndex::new(weights).unwrap();
        self.at_mut(col, 0)
            .iter_mut()
            .for_each(|x: &mut W| *x = choices[dist.sample(source)]);
    }

    fn fill_ternary_hw(&mut self, col: usize, hw: usize, source: &mut Source) {
        assert!(hw <= self.n());
        // Zero-initialize before setting non-zero entries, since shuffle will
        // mix positions and we need indices hw..n to be zero.
        self.at_mut(col, 0).fill(W::zero());
        self.at_mut(col, 0)[..hw]
            .iter_mut()
            .for_each(|x: &mut W| *x = W::from_i64((((source.next_u32() & 1) as i64) << 1) - 1));
        self.at_mut(col, 0).shuffle(source);
    }

    fn fill_binary_prob(&mut self, col: usize, prob: f64, source: &mut Source) {
        let choices: [W; 2] = [W::zero(), W::from_i64(1)];
        let weights: [f64; 2] = [1.0 - prob, prob];
        let dist: WeightedIndex<f64> = WeightedIndex::new(weights).unwrap();
        self.at_mut(col, 0)
            .iter_mut()
            .for_each(|x: &mut W| *x = choices[dist.sample(source)]);
    }

    fn fill_binary_hw(&mut self, col: usize, hw: usize, source: &mut Source) {
        assert!(hw <= self.n());
        // Zero-initialize before setting non-zero entries, since shuffle will
        // mix positions and we need indices hw..n to be zero.
        self.at_mut(col, 0).fill(W::zero());
        self.at_mut(col, 0)[..hw].fill(W::from_i64(1));
        self.at_mut(col, 0).shuffle(source);
    }

    fn fill_binary_block(&mut self, col: usize, block_size: usize, source: &mut Source) {
        assert!(self.n().is_multiple_of(block_size));
        // Zero-initialize: each block gets at most one non-zero entry.
        self.at_mut(col, 0).fill(W::zero());
        let max_idx: u64 = (block_size + 1) as u64;
        let mask_idx: u64 = (1 << ((u64::BITS - max_idx.leading_zeros()) as u64)) - 1;
        for block in self.at_mut(col, 0).chunks_mut(block_size) {
            let idx: usize = source.next_u64n(max_idx, mask_idx) as usize;
            if idx != block_size {
                block[idx] = W::from_i64(1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use poulpy_hal::layouts::{Backend, HostBytesBackend, ZnxView};

    fn fresh(n: usize, cols: usize) -> ScalarZnx<Vec<u8>, i64> {
        ScalarZnx::from_data(
            HostBytesBackend::alloc_zeroed_bytes(ScalarZnx::<Vec<u8>, i64>::bytes_of(n, cols)),
            n,
            cols,
        )
    }

    #[test]
    fn host_fills_respect_value_sets_weights_and_columns() {
        let (n, cols, col) = (1usize << 10, 2usize, 1usize);
        let mut source = Source::new([3u8; 32]);

        let mut s = fresh(n, cols);
        s.fill_binary_hw(col, 37, &mut source);
        assert!(s.at(col, 0).iter().all(|&x| x == 0 || x == 1));
        assert_eq!(s.at(col, 0).iter().filter(|&&x| x == 1).count(), 37);
        assert!(s.at(0, 0).iter().all(|&x| x == 0), "other column untouched");

        let mut s = fresh(n, cols);
        s.fill_ternary_hw(col, 41, &mut source);
        assert!(s.at(col, 0).iter().all(|&x| (-1..=1).contains(&x)));
        assert_eq!(s.at(col, 0).iter().filter(|&&x| x != 0).count(), 41);

        let mut s = fresh(n, cols);
        s.fill_ternary_prob(col, 0.5, &mut source);
        assert!(s.at(col, 0).iter().all(|&x| (-1..=1).contains(&x)));
        assert!(s.at(col, 0).iter().any(|&x| x != 0));

        let mut s = fresh(n, cols);
        s.fill_binary_prob(col, 0.5, &mut source);
        assert!(s.at(col, 0).iter().all(|&x| x == 0 || x == 1));

        let mut s = fresh(n, cols);
        s.fill_binary_block(col, 8, &mut source);
        assert!(s.at(col, 0).iter().all(|&x| x == 0 || x == 1));
        assert!(
            s.at(col, 0)
                .chunks(8)
                .all(|block| block.iter().filter(|&&x| x == 1).count() <= 1)
        );
    }

    #[test]
    fn host_fills_are_deterministic_in_the_source() {
        let mut a = fresh(256, 1);
        let mut b = fresh(256, 1);
        a.fill_ternary_hw(0, 17, &mut Source::new([9u8; 32]));
        b.fill_ternary_hw(0, 17, &mut Source::new([9u8; 32]));
        assert_eq!(a.at(0, 0), b.at(0, 0));
    }
}
