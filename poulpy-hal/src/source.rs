use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng};
use rand_core::{Infallible, Rng, TryRng};

/// 2^53, the number of distinct values representable in the 53-bit significand
/// of an IEEE-754 `f64`. Used as the denominator when converting a random
/// `u64` to a uniformly distributed floating-point value in `[0, 1)`.
const MAXF64: f64 = 9007199254740992.0;

/// Deterministic pseudorandom number generator based on ChaCha8.
///
/// Wraps [`ChaCha8Rng`] to provide reproducible random sampling for
/// lattice-based cryptographic operations. Given the same 32-byte seed,
/// the output sequence is identical across platforms.
///
/// **Not suitable for cryptographic key generation.** This type is intended
/// for deterministic test vectors, noise sampling, and reproducible
/// benchmarks, not for generating secrets.
pub struct Source {
    source: ChaCha8Rng,
}

impl Source {
    /// Creates a new `Source` from a 32-byte seed.
    ///
    /// The same seed always produces the same pseudorandom sequence.
    pub fn new(seed: [u8; 32]) -> Source {
        Source {
            source: ChaCha8Rng::from_seed(seed),
        }
    }

    /// Derives an independent child `Source` for sub-stream splitting.
    ///
    /// Draws a fresh 32-byte seed from `self` and returns both the seed and
    /// a new `Source` seeded with it. The parent and child streams are
    /// statistically independent.
    pub fn branch(&mut self) -> ([u8; 32], Self) {
        let seed: [u8; 32] = self.new_seed();
        (seed, Source::new(seed))
    }

    /// Draws 32 random bytes suitable for use as a derived seed.
    pub fn new_seed(&mut self) -> [u8; 32] {
        let mut seed: [u8; 32] = [0u8; 32];
        self.fill_bytes(&mut seed);
        seed
    }

    /// Returns a uniformly distributed `u64` in `[0, max)` using rejection
    /// sampling with bitmask `mask`.
    ///
    /// `mask` should be `max.next_power_of_two() - 1` (or wider). Each
    /// iteration draws one `u64` and masks it; values `>= max` are rejected.
    /// Expected iterations: at most 2 when `mask` is tight.
    #[inline(always)]
    pub fn next_u64n(&mut self, max: u64, mask: u64) -> u64 {
        let mut x: u64 = self.next_u64() & mask;
        while x >= max {
            x = self.next_u64() & mask;
        }
        x
    }

    /// Returns a uniformly distributed f64 in [min, max).
    ///
    /// # Panics
    /// Panics if `min > max`.
    #[inline(always)]
    pub fn next_f64(&mut self, min: f64, max: f64) -> f64 {
        assert!(min <= max, "next_f64: min ({min}) > max ({max})");
        min + ((self.next_u64() << 11 >> 11) as f64) / MAXF64 * (max - min)
    }

    /// Returns a uniformly distributed `i32` (bit-reinterpretation of a random `u32`).
    #[inline(always)]
    pub fn next_i32(&mut self) -> i32 {
        self.next_u32() as i32
    }

    /// Returns a uniformly distributed `i64` (bit-reinterpretation of a random `u64`).
    #[inline(always)]
    pub fn next_i64(&mut self) -> i64 {
        self.next_u64() as i64
    }

    /// Returns a uniformly distributed `i128` (bit-reinterpretation of a random `u128`).
    #[inline(always)]
    pub fn next_i128(&mut self) -> i128 {
        self.next_u128() as i128
    }

    /// Returns a uniformly distributed `u128` by concatenating two `u64` draws.
    #[inline(always)]
    pub fn next_u128(&mut self) -> u128 {
        (self.next_u64() as u128) << 64 | (self.next_u64() as u128)
    }
}

/// Implements [`TryRng`] by delegating to the inner [`ChaCha8Rng`].
/// The blanket `impl<R: TryRng<Error = Infallible>> Rng for R` in `rand_core`
/// then provides [`Rng`] automatically.
impl TryRng for Source {
    type Error = Infallible;

    #[inline(always)]
    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        self.source.try_next_u32()
    }

    #[inline(always)]
    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        self.source.try_next_u64()
    }

    #[inline(always)]
    fn try_fill_bytes(&mut self, bytes: &mut [u8]) -> Result<(), Self::Error> {
        self.source.try_fill_bytes(bytes)
    }
}
