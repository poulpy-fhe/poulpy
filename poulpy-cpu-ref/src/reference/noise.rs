//! Integer noise samplers shared by every host backend.
//!
//! Small Gaussians use a full cumulative-table scan. Large Gaussians use the
//! exact rational rejection sampler of Canonne, Kamath and Steinke (2020),
//! Algorithm 3. Only that rejection path has sample-dependent running time.
use dashu_int::{IBig, Sign, UBig, ops::BitTest};
use poulpy_core::Noise;
use poulpy_hal::{layouts::ZnxViewMut, source::Source};
use rand_core::Rng;

// floor(2^128 P(|z| <= i)), for the exact binary64 value 3.2, truncated at
// six sigma: B = floor(6 * 3.2) = 19.
// The integer interval constructor below independently verifies these values.
const ENCRYPTION_CDT: [u128; 19] = [
    0x1fea56814b22fdba95eb5e667e6e5a60,
    0x5cb44ba97e1c7ec189f6ed45c750fb3d,
    0x9135c4d2ab6095d14f19cef5351bb08e,
    0xba57841ec437e11bdc02a6ee70688e9e,
    0xd790d29879901e3ca85bb6f44ba681d9,
    0xea65b1358d74fa5cefbfd8a861090c33,
    0xf5672d96e886ce60171cdc74e9e08229,
    0xfb3c9d22cc87d5961a1933fea5a5cd5c,
    0xfe0a92b033d640642b58e139859d2cff,
    0xff439d380073529141f946ad5f6a635f,
    0xffbf67e425d189c54415a63a75a24ade,
    0xffebcdf3a4629f94471ba8011b9486c9,
    0xfffa3f2f78880d440020335435a4ef82,
    0xfffe81f40111a23abc4a9117693b3646,
    0xffffa5d06bc38afb5ea8a82477ac592f,
    0xffffeca4870dfe3c4afb8aae1df170e6,
    0xfffffc3b68ed3b4a782ac81cafdf35ae,
    0xffffff58143b961199ccc63c7e7eb149,
    0xffffffe850ce70a637c977f072a256b1,
];

/// A positive finite binary64 value as an exact dyadic rational.
fn dyadic(sigma: f64) -> (UBig, UBig) {
    let bits = sigma.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let mantissa = bits & ((1u64 << 52) - 1);
    let (mantissa, exponent) = if exponent == 0 {
        (mantissa, -1074)
    } else {
        (mantissa | (1u64 << 52), exponent - 1023 - 52)
    };
    let trailing = mantissa.trailing_zeros() as i32;
    let numerator = UBig::from(mantissa >> trailing);
    let exponent = exponent + trailing;
    if exponent >= 0 {
        (numerator << exponent as usize, UBig::ONE)
    } else {
        (numerator, UBig::ONE << (-exponent) as usize)
    }
}

fn ceil_div(a: UBig, b: &UBig) -> UBig {
    (a + b - 1u8) / b
}

/// Certified fixed-point enclosure of exp(-a/b), scaled by 2^precision.
/// Alternating Taylor bounds are propagated through range reduction and
/// squaring, rounding outward at every integer operation.
fn exp_interval(a: &UBig, b: &UBig, precision: usize) -> (UBig, UBig) {
    let unit = UBig::ONE << precision;
    if a == &UBig::ZERO {
        return (unit.clone(), unit);
    }
    // exp(-precision) < 2^-precision, since e > 2.
    if a > &(b * precision) {
        return (UBig::ZERO, UBig::ONE);
    }
    let mut denominator = b.clone();
    let mut squares = 0;
    while a > &denominator {
        denominator <<= 1;
        squares += 1;
    }
    let mut term_lo = unit.clone();
    let mut term_hi = unit.clone();
    let mut lo = IBig::from(unit.clone());
    let mut hi = lo.clone();
    for j in 1usize.. {
        let divisor = &denominator * j;
        term_lo = &term_lo * a / &divisor;
        term_hi = ceil_div(&term_hi * a, &divisor);
        if j & 1 == 1 {
            lo -= &term_hi;
            hi -= &term_lo;
        } else {
            lo += &term_lo;
            hi += &term_hi;
        }
        if term_hi <= UBig::ONE {
            // The next Taylor remainder is bounded by the present term.
            if j & 1 == 1 {
                hi += &term_hi;
            } else {
                lo -= &term_hi;
            }
            break;
        }
    }
    let mut lo = UBig::try_from(lo).unwrap_or(UBig::ZERO);
    let mut hi = UBig::try_from(hi).unwrap();
    for _ in 0..squares {
        lo = (&lo * &lo) >> precision;
        hi = ceil_div(&hi * &hi, &unit);
    }
    (lo, hi)
}

fn cumulative_table(sigma: f64, bound: usize) -> Vec<u128> {
    let (num, den) = dyadic(sigma);
    let num2 = &num * &num * 2u8;
    let den2 = &den * &den;
    let mut precision = 256;
    loop {
        let unit = UBig::ONE << precision;
        let mut weights = vec![(unit.clone(), unit)];
        for i in 1..=bound {
            let (lo, hi) = exp_interval(&(&den2 * i * i), &num2, precision);
            weights.push((lo << 1, hi << 1));
        }
        let total_lo: UBig = weights.iter().map(|(lo, _)| lo).sum();
        let total_hi: UBig = weights.iter().map(|(_, hi)| hi).sum();
        let mut cum_lo = UBig::ZERO;
        let mut cum_hi = UBig::ZERO;
        let mut table = Vec::with_capacity(bound);
        for (lo, hi) in &weights[..bound] {
            cum_lo += lo;
            cum_hi += hi;
            let scaled_lo = &cum_lo << 128;
            let scaled_hi = &cum_hi << 128;
            let floor = &scaled_lo / &total_hi;
            let value = u128::try_from(&floor).unwrap_or(u128::MAX);
            // Certify error <= one unit of 128-bit probability. This also
            // handles tails smaller than 2^-128 without unbounded precision.
            if scaled_hi > (&floor + 1u8) * &total_lo {
                break;
            }
            table.push(value);
        }
        if table.len() == bound {
            return table;
        }
        precision *= 2;
    }
}

/// A uniform integer below `limit`; rejection is exact for arbitrary widths.
fn uniform_below(limit: &UBig, source: &mut Source) -> UBig {
    assert!(limit > &UBig::ZERO);
    if limit == &UBig::ONE {
        return UBig::ZERO;
    }
    let bits = (limit - 1u8).bit_len();
    if bits <= 64 {
        let mask = u64::MAX >> (64 - bits);
        if let Ok(limit) = u64::try_from(limit) {
            return UBig::from(source.next_u64n(limit, mask));
        }
        return UBig::from(source.next_u64());
    }
    let mut bytes = vec![0u8; bits.div_ceil(8)];
    loop {
        source.fill_bytes(&mut bytes);
        *bytes.last_mut().unwrap() &= u8::MAX >> ((8 - bits % 8) % 8);
        let value = UBig::from_le_bytes(&bytes);
        if &value < limit {
            return value;
        }
    }
}

fn bernoulli(num: &UBig, den: &UBig, source: &mut Source) -> bool {
    uniform_below(den, source) < *num
}

/// Exact Bernoulli(exp(-num/den)) for a ratio in [0, 1].
fn bernoulli_exp_unit(num: &UBig, den: &UBig, source: &mut Source) -> bool {
    let mut j = 1usize;
    while bernoulli(num, &(den * j), source) {
        j = j.checked_add(1).expect("Bernoulli series index overflow");
    }
    j & 1 == 1
}

fn bernoulli_exp(num: &UBig, den: &UBig, source: &mut Source) -> bool {
    let mut whole = num / den;
    while whole != UBig::ZERO {
        if !bernoulli_exp_unit(&UBig::ONE, &UBig::ONE, source) {
            return false;
        }
        whole -= 1u8;
    }
    bernoulli_exp_unit(&(num % den), den, source)
}

struct RejectionGaussian {
    variance_num: UBig,
    variance_den: UBig,
    t: UBig,
    bound: UBig,
    acceptance_den: UBig,
}
impl RejectionGaussian {
    fn new(sigma: f64, cutoff: usize) -> Self {
        let (num, den) = dyadic(sigma);
        let t = &num / &den + 1u8;
        let bound = &num * cutoff / &den;
        let variance_num = &num * &num;
        let variance_den = &den * &den;
        let acceptance_den = &variance_num * &variance_den * &t * &t * 2u8;
        Self {
            variance_num,
            variance_den,
            t,
            bound,
            acceptance_den,
        }
    }
    fn sample(&self, source: &mut Source) -> IBig {
        loop {
            let u = loop {
                let u = uniform_below(&self.t, source);
                if bernoulli_exp_unit(&u, &self.t, source) {
                    break u;
                }
            };
            let mut v = UBig::ZERO;
            while bernoulli_exp_unit(&UBig::ONE, &UBig::ONE, source) {
                v += 1u8;
            }
            let y = u + &self.t * v;
            let negative = source.next_u32() & 1 != 0;
            if (negative && y == UBig::ZERO) || y > self.bound {
                continue;
            }
            let lhs = &y * &self.t * &self.variance_den;
            let difference = if lhs >= self.variance_num {
                lhs - &self.variance_num
            } else {
                &self.variance_num - lhs
            };
            if bernoulli_exp(&(&difference * &difference), &self.acceptance_den, source) {
                return IBig::from_parts(if negative { Sign::Negative } else { Sign::Positive }, y);
            }
        }
    }
}

/// Scans every threshold, independently of the sampled value.
#[inline(always)]
fn sample_table(table: &[u128], source: &mut Source, sign: u64) -> i64 {
    let u = source.next_u128();
    let mut magnitude = 0i64;
    for threshold in table {
        magnitude += (u >= *threshold) as i64;
    }
    let mask = -(sign as i64);
    (magnitude ^ mask) - mask
}

// Batch entropy and keep the high and low words separate so the compiler can
// vectorize the full 128-bit comparison. No table lookup depends on a draw.
const TABLE_BATCH: usize = 64;

#[inline(always)]
fn table_entropy(source: &mut Source, high: &mut [i64; TABLE_BATCH], low: &mut [i64; TABLE_BATCH]) -> u64 {
    let mut bytes = [0u8; 8 * TABLE_BATCH];
    source.fill_bytes(&mut bytes);
    for (word, bytes) in high.iter_mut().zip(bytes.chunks_exact(8)) {
        *word = i64::from_le_bytes(bytes.try_into().unwrap()) ^ i64::MIN;
    }
    source.fill_bytes(&mut bytes);
    for (word, bytes) in low.iter_mut().zip(bytes.chunks_exact(8)) {
        *word = i64::from_le_bytes(bytes.try_into().unwrap()) ^ i64::MIN;
    }
    source.next_u64()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64", test))]
#[inline(always)]
fn scan_table_batch(table: &[u128], high: &[i64; TABLE_BATCH], low: &[i64; TABLE_BATCH]) -> [i64; TABLE_BATCH] {
    let mut magnitude = [0i64; TABLE_BATCH];
    for &threshold in table {
        // Every CDT threshold is at least 1/(2B+1) >= 1/129. Its high
        // word is nonzero, so subtracting one cannot underflow.
        let threshold_high = (((threshold >> 64) as u64 - 1) as i64) ^ i64::MIN;
        let threshold_low = (threshold as i64) ^ i64::MIN;
        for i in 0..TABLE_BATCH {
            // u >= T iff u.high > T.high - 1 + (u.low < T.low).
            // Biasing both unsigned words preserves order in signed SIMD
            // lanes. The adjusted high threshold cannot overflow i64.
            let borrow = -((low[i] < threshold_low) as i64);
            magnitude[i] += (high[i] > threshold_high.wrapping_sub(borrow)) as i64;
        }
    }
    magnitude
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64", test))]
#[inline(always)]
fn add_table_vectorizable<W: NoiseWord>(res: &mut [W], shift: usize, table: &[u128], source: &mut Source) {
    for chunk in res.chunks_mut(TABLE_BATCH) {
        let mut high = [0i64; TABLE_BATCH];
        let mut low = [0i64; TABLE_BATCH];
        let signs = table_entropy(source, &mut high, &mut low);
        let magnitude = scan_table_batch(table, &high, &low);
        for (i, dst) in chunk.iter_mut().enumerate() {
            let sign = -(((signs >> i) & 1) as i64);
            dst.add_noise_digit(((magnitude[i] ^ sign) - sign) << shift);
        }
    }
}

// This is ordinary Rust, specialized for the ISA after runtime detection. The
// portable path below consumes precisely the same entropy and uses the same
// table, so CPU feature selection cannot change a backend's seeded output.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn add_table_avx2<W: NoiseWord>(res: &mut [W], shift: usize, table: &[u128], source: &mut Source) {
    add_table_vectorizable(res, shift, table, source);
}

#[cfg(any(not(target_arch = "aarch64"), test))]
fn add_table_scalar<W: NoiseWord>(res: &mut [W], shift: usize, table: &[u128], source: &mut Source) {
    for chunk in res.chunks_mut(TABLE_BATCH) {
        let mut high = [0i64; TABLE_BATCH];
        let mut low = [0i64; TABLE_BATCH];
        let signs = table_entropy(source, &mut high, &mut low);
        for (i, dst) in chunk.iter_mut().enumerate() {
            let u = ((high[i] ^ i64::MIN) as u64 as u128) << 64 | (low[i] ^ i64::MIN) as u64 as u128;
            let magnitude = table.iter().fold(0i64, |count, threshold| count + (u >= *threshold) as i64);
            let sign = -(((signs >> i) & 1) as i64);
            dst.add_noise_digit(((magnitude ^ sign) - sign) << shift);
        }
    }
}

fn add_table<W: NoiseWord>(res: &mut [W], shift: usize, table: &[u128], source: &mut Source) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx2") {
        // Safety: the complete function is guarded by runtime ISA detection.
        unsafe { add_table_avx2(res, shift, table, source) };
        return;
    }
    // NEON is baseline on aarch64; ordinary Rust vectorizes without dispatch.
    #[cfg(target_arch = "aarch64")]
    {
        add_table_vectorizable(res, shift, table, source);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        add_table_scalar(res, shift, table, source);
    }
}

/// The generic destination interface only converts the final bounded digit.
#[doc(hidden)]
pub trait NoiseWord: Copy {
    fn add_noise_digit(&mut self, digit: i64);
}
impl NoiseWord for i64 {
    #[inline(always)]
    fn add_noise_digit(&mut self, digit: i64) {
        *self = self.wrapping_add(digit);
    }
}
impl NoiseWord for i128 {
    #[inline(always)]
    fn add_noise_digit(&mut self, digit: i64) {
        *self = self.wrapping_add(digit as i128);
    }
}

// Small table draws and their padding fit i128 even at radix 2^63.
// Fixed word carries keep this path independent of the sampled value.
fn place_small<R: ZnxViewMut>(res: &mut R, col: usize, index: usize, base2k: usize, size: usize, shift: usize, sample: i64)
where
    R::Scalar: NoiseWord,
{
    let mut value = (sample as i128) << shift;
    let half = 1i128 << (base2k - 1);
    for limb in (0..size).rev() {
        let carry = (value + half) >> base2k;
        let digit = value - (carry << base2k);
        res.at_mut(col, limb)[index].add_noise_digit(digit as i64);
        value = carry;
    }
}

fn place_integer<R: ZnxViewMut>(res: &mut R, col: usize, index: usize, base2k: usize, size: usize, shift: usize, sample: IBig)
where
    R::Scalar: NoiseWord,
{
    let mut value = sample << shift;
    let half = IBig::ONE << (base2k - 1);
    for limb in (0..size).rev() {
        let carry = (&value + &half) >> base2k;
        let digit = &value - (&carry << base2k);
        res.at_mut(col, limb)[index].add_noise_digit(i64::try_from(digit).unwrap());
        value = carry;
    }
}

/// Uniform bits are consumed in public, fixed-size chunks. Signed extension
/// and balanced carries use word arithmetic only, including above 128 bits.
fn add_uniform<R: ZnxViewMut>(res: &mut R, col: usize, base2k: usize, size: usize, shift: usize, bits: usize, source: &mut Source)
where
    R::Scalar: NoiseWord,
{
    if bits <= base2k - shift {
        let mask = (1u64 << bits) - 1;
        let sign_bit = 1u64 << (bits - 1);
        for dst in res.at_mut(col, size - 1) {
            let word = source.next_u64() & mask;
            let signed = (word ^ sign_bit).wrapping_sub(sign_bit) as i64;
            dst.add_noise_digit(signed << shift);
        }
        return;
    }
    let mask = (1u64 << base2k) - 1;
    let half = 1u64 << (base2k - 1);
    for index in 0..res.n() {
        let mut remaining = bits;
        let mut padding = shift;
        let mut sign = 0u64;
        let mut carry = 0u64;
        for limb in (0..size).rev() {
            let width = remaining.min(base2k - padding);
            let word = source.next_u64() & ((1u64 << width) - 1);
            remaining -= width;
            if width != 0 {
                sign = word >> (width - 1);
            }
            let extension = 0u64.wrapping_sub(sign) << (width + padding);
            let raw = ((word << padding) | extension) & mask;
            let balanced = raw + carry + half;
            let digit = (balanced & mask) as i64 - half as i64;
            carry = balanced >> base2k;
            res.at_mut(col, limb)[index].add_noise_digit(digit);
            padding = 0;
        }
    }
}

/// Adds one full-width integer sample per coefficient at precision `k`.
pub fn add_noise<R: ZnxViewMut>(base2k: usize, k: usize, res: &mut R, col: usize, noise: Noise, source: &mut Source)
where
    R::Scalar: NoiseWord,
{
    noise.validate();
    assert!((1..=63).contains(&base2k), "noise radix must be in 1..=63");
    assert!(k > 0, "noise precision must be positive");
    let size = k.div_ceil(base2k);
    assert!(size <= res.size(), "noise precision exceeds destination allocation");
    assert!(col < res.cols(), "noise column exceeds destination allocation");
    let shift = (base2k - k % base2k) % base2k;
    match noise {
        Noise::Uniform { bits } => add_uniform(res, col, base2k, size, shift, bits, source),
        Noise::Gaussian { sigma, cutoff } => {
            let (num, den) = dyadic(sigma);
            let bound = num * cutoff / den;
            if bound <= UBig::from(64u8) {
                let dynamic;
                let table = if noise == Noise::ENCRYPTION {
                    &ENCRYPTION_CDT[..]
                } else {
                    dynamic = cumulative_table(sigma, usize::try_from(&bound).unwrap());
                    &dynamic
                };
                let mut signs = 0u64;
                if (table.len() as u128) << shift < 1u128 << (base2k - 1) {
                    add_table(res.at_mut(col, size - 1), shift, table, source);
                } else {
                    for index in 0..res.n() {
                        if index % 64 == 0 {
                            signs = source.next_u64();
                        }
                        let sample = sample_table(table, source, signs & 1);
                        signs >>= 1;
                        place_small(res, col, index, base2k, size, shift, sample);
                    }
                }
            } else {
                let gaussian = RejectionGaussian::new(sigma, cutoff);
                for index in 0..res.n() {
                    place_integer(res, col, index, base2k, size, shift, gaussian.sample(source));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FFT64Ref, NTT4x30Ref};
    use poulpy_core::{VecZnxAddNoise, VecZnxBigAddNoise};
    use poulpy_hal::{api::*, layouts::*};

    fn assert_pmf(sigma: f64, cutoff: usize, mut sample: impl FnMut() -> i64) {
        let bound = (sigma * cutoff as f64).floor() as i64;
        let draws = 200_000usize;
        let mut counts = vec![0usize; (2 * bound + 1) as usize];
        for _ in 0..draws {
            let z = sample();
            assert!(z.abs() <= bound);
            counts[(z + bound) as usize] += 1;
        }
        let weights: Vec<_> = (-bound..=bound)
            .map(|z| (-(z as f64).powi(2) / (2.0 * sigma * sigma)).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        for (i, (count, weight)) in counts.iter().zip(weights).enumerate() {
            let p = weight / total;
            let expected = draws as f64 * p;
            let tolerance = 7.0 * (draws as f64 * p * (1.0 - p)).sqrt() + 8.0;
            assert!(
                (*count as f64 - expected).abs() < tolerance,
                "sigma={sigma}, z={}, count={count}, expected={expected}, tolerance={tolerance}",
                i as i64 - bound
            );
        }
    }

    #[test]
    fn encryption_table_matches_integer_construction() {
        assert_eq!(cumulative_table(3.2, 19), ENCRYPTION_CDT);
    }

    #[test]
    fn table_probability_mass_function() {
        let mut source = Source::new([31; 32]);
        assert_pmf(3.2, 6, || {
            let sign = source.next_u64() & 1;
            sample_table(&ENCRYPTION_CDT, &mut source, sign)
        });
        let table = cumulative_table(0.75, 4);
        assert_pmf(0.75, 6, || {
            let sign = source.next_u64() & 1;
            sample_table(&table, &mut source, sign)
        });
    }

    #[test]
    fn rejection_probability_mass_functions() {
        for sigma in [4.0, 3.2] {
            let gaussian = RejectionGaussian::new(sigma, 6);
            let mut source = Source::new([43; 32]);
            assert_pmf(sigma, 6, || i64::try_from(gaussian.sample(&mut source)).unwrap());
        }
    }

    #[test]
    fn exact_dyadic_and_extreme_parameters() {
        assert_eq!(dyadic(3.2), (UBig::from(3602879701896397u64), UBig::ONE << 50));
        assert_eq!(dyadic(f64::from_bits(1)), (UBig::ONE, UBig::ONE << 1074));
        assert_eq!(dyadic(2f64.powi(128)), (UBig::ONE << 128, UBig::ONE));
        assert_eq!(cumulative_table(0.01, 1), vec![u128::MAX]);
        assert!(cumulative_table(f64::from_bits(1), 0).is_empty());
    }

    #[test]
    fn batched_scan_matches_all_threshold_boundaries() {
        for table in [ENCRYPTION_CDT.to_vec(), cumulative_table(0.01, 1), cumulative_table(12.5, 64)] {
            let mut points = vec![0u128, u128::MAX];
            for &threshold in &table {
                points.push(threshold - 1);
                points.push(threshold);
                if threshold < u128::MAX {
                    points.push(threshold + 1);
                }
            }
            for points in points.chunks(TABLE_BATCH) {
                let mut high = [0i64; TABLE_BATCH];
                let mut low = [0i64; TABLE_BATCH];
                for (i, &u) in points.iter().enumerate() {
                    high[i] = ((u >> 64) as i64) ^ i64::MIN;
                    low[i] = (u as i64) ^ i64::MIN;
                }
                let samples = scan_table_batch(&table, &high, &low);
                for (i, u) in points.iter().enumerate() {
                    assert_eq!(samples[i], table.iter().filter(|threshold| u >= *threshold).count() as i64);
                }
            }
        }
    }

    #[test]
    fn scalar_and_vector_tables_have_identical_seeded_output() {
        for len in [1, 7, 64, 193] {
            let mut scalar = vec![7i64; len];
            let mut vector = scalar.clone();
            let mut source_scalar = Source::new([51; 32]);
            let mut source_vector = Source::new([51; 32]);
            add_table_scalar(&mut scalar, 3, &ENCRYPTION_CDT, &mut source_scalar);
            add_table(&mut vector, 3, &ENCRYPTION_CDT, &mut source_vector);
            assert_eq!(scalar, vector);
            assert_eq!(source_scalar.new_seed(), source_vector.new_seed());
        }
    }

    fn reconstruct<R: ZnxView<Scalar = i64>>(res: &R, col: usize, index: usize, base2k: usize, k: usize) -> IBig {
        let size = k.div_ceil(base2k);
        let mut value = IBig::ZERO;
        for limb in 0..size {
            value = (value << base2k) + res.at(col, limb)[index];
        }
        let padding = (base2k - k % base2k) % base2k;
        assert_eq!(&value & ((IBig::ONE << padding) - 1u8), IBig::ZERO);
        value >> padding
    }

    #[test]
    fn gaussian_reconstructs_beyond_128_bits_and_preserves_columns() {
        let module = Module::<FFT64Ref>::new(64);
        let (base2k, k, size) = (17, 211, 14);
        let noise = Noise::Gaussian {
            sigma: 2f64.powi(150),
            cutoff: 6,
        };
        let mut res = module.vec_znx_alloc(64, 2, size);
        res.at_mut(0, 0).fill(99);
        let mut parent = Source::new([97; 32]);
        let mut expected_parent = Source::new([97; 32]);
        let mut expected_source = Source::new(expected_parent.new_seed());
        let gaussian = RejectionGaussian::new(2f64.powi(150), 6);
        module.vec_znx_add_noise(
            base2k,
            k,
            &mut poulpy_hal::test_suite::vec_znx_backend_mut::<FFT64Ref>(&mut res),
            1,
            noise,
            &mut parent,
        );
        let mut exceeds_128_bits = false;
        for index in 0..64 {
            let actual = reconstruct(&res, 1, index, base2k, k);
            assert_eq!(actual, gaussian.sample(&mut expected_source));
            exceeds_128_bits |= actual.into_parts().1.bit_len() > 128;
        }
        assert!(exceeds_128_bits, "large Gaussian samples were truncated to 128 bits");
        assert!(res.at(0, 0).iter().all(|x| *x == 99));
        for limb in 1..size {
            assert!(res.at(0, limb).iter().all(|x| *x == 0));
        }
        assert!(res.at(1, size - 1).iter().all(|x| *x == 0));
        assert_eq!(parent.new_seed(), expected_parent.new_seed());
    }

    #[test]
    fn uniform_reconstructs_full_width_and_balanced_digits() {
        let module = Module::<FFT64Ref>::new(64);
        for (base2k, k, bits) in [(17usize, 211usize, 173usize), (3, 19, 1), (7, 37, 23), (63, 257, 193)] {
            let size = k.div_ceil(base2k);
            let mut res = module.vec_znx_alloc(64, 2, size + 1);
            let mut source = Source::new([109; 32]);
            let mut expected = Source::new([109; 32]);
            add_noise(base2k, k, &mut res, 1, Noise::Uniform { bits }, &mut source);
            let mut positive = false;
            let mut negative = false;
            for index in 0..64 {
                let padding = (base2k - k % base2k) % base2k;
                let mut remaining = bits;
                let mut raw = UBig::ZERO;
                let mut offset = 0;
                for limb in (0..size).rev() {
                    let width = remaining.min(base2k - if limb == size - 1 { padding } else { 0 });
                    let word = if bits <= base2k - padding && limb != size - 1 {
                        0
                    } else {
                        expected.next_u64() & ((1u64 << width) - 1)
                    };
                    raw += UBig::from(word) << offset;
                    offset += width;
                    remaining -= width;
                }
                let sign = &raw >> (bits - 1);
                let want = IBig::from(raw) - (IBig::from(sign) << bits);
                let actual = reconstruct(&res, 1, index, base2k, k);
                assert_eq!(actual, want);
                positive |= actual > IBig::ZERO;
                negative |= actual < IBig::ZERO;
                for limb in 0..size {
                    let digit = res.at(1, limb)[index];
                    assert!(digit >= -(1i64 << (base2k - 1)) && digit < 1i64 << (base2k - 1));
                    assert_eq!(res.at(0, limb)[index], 0);
                }
            }
            assert!(negative);
            if bits > 1 {
                assert!(positive);
            }
            assert!(res.at(1, size).iter().all(|x| *x == 0));
            assert_eq!(source.new_seed(), expected.new_seed());
        }
    }

    #[test]
    fn narrow_radix_gaussian_carries_and_wide_backend_match() {
        let fft = Module::<FFT64Ref>::new(64);
        let ntt = Module::<NTT4x30Ref>::new(64);
        for (base2k, k) in [(3usize, 20usize), (17, 43)] {
            for noise in [
                Noise::ENCRYPTION,
                Noise::Uniform { bits: 17 },
                Noise::Gaussian { sigma: 15.25, cutoff: 6 },
            ] {
                let size = k.div_ceil(base2k);
                let mut small = fft.vec_znx_alloc(64, 1, size);
                let mut wide = ntt.vec_znx_big_alloc(64, 1, size);
                fft.vec_znx_add_noise(
                    base2k,
                    k,
                    &mut poulpy_hal::test_suite::vec_znx_backend_mut::<FFT64Ref>(&mut small),
                    0,
                    noise,
                    &mut Source::new([17; 32]),
                );
                ntt.vec_znx_big_add_noise(base2k, k, &mut wide.to_backend_mut(), 0, noise, &mut Source::new([17; 32]));
                for limb in 0..size {
                    for index in 0..64 {
                        assert_eq!(small.at(0, limb)[index] as i128, wide.at(0, limb)[index]);
                    }
                }
            }
        }
    }
    #[test]
    fn rejects_before_sampling_or_mutation() {
        use poulpy_hal::layouts::{VecZnx, vec_znx_backend_mut};
        use std::panic::{AssertUnwindSafe, catch_unwind};

        let noise = Noise::Uniform { bits: 80 };
        for (base2k, k, col, message) in [
            (17, 100, 2, "noise column exceeds destination allocation"),
            (0, 100, 1, "noise radix must be in 1..=63"),
            (64, 100, 1, "noise radix must be in 1..=63"),
            (17, 120, 1, "noise precision exceeds destination allocation"),
        ] {
            let mut result = VecZnx::from_data(
                <crate::FFT64Ref as Backend>::alloc_zeroed_bytes(VecZnx::<poulpy_hal::AlignedBuf, i64>::bytes_of(8, 2, 6)),
                8,
                2,
                6,
            );
            for limb in 0..6 {
                result.at_mut(1, limb).fill(7);
            }
            let before: Vec<_> = (0..6).flat_map(|limb| result.at(1, limb).iter().copied()).collect();
            let mut source = Source::new([99; 32]);
            let mut pristine = Source::new([99; 32]);
            let error = catch_unwind(AssertUnwindSafe(|| {
                add_noise(
                    base2k,
                    k,
                    &mut vec_znx_backend_mut::<crate::FFT64Ref>(&mut result),
                    col,
                    noise,
                    &mut source,
                );
            }))
            .unwrap_err();
            let actual = error
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| error.downcast_ref::<String>().map(String::as_str));
            assert_eq!(actual, Some(message));
            assert_eq!(source.new_seed(), pristine.new_seed());
            let after: Vec<_> = (0..6).flat_map(|limb| result.at(1, limb).iter().copied()).collect();
            assert_eq!(after, before);
            assert!((0..6).all(|limb| result.at(0, limb).iter().all(|&x| x == 0)));
        }
    }
}
