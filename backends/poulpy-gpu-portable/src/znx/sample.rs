//! Sampling on the device, from a seed.
//!
//! Every coefficient (every block for the block-sparse secret, the whole
//! column for the fixed-weight ones) draws from its own ChaCha8 keystream:
//! the seed is the key and the coefficient's index the stream. A draw thus
//! depends on the seed, the degree, the law and the placement only, never on
//! the column, the other columns, the allocated limbs or the destination's
//! content, as `SamplingImpl` requires. The laws are those of the portable
//! samplers: the cumulative table of a Gaussian of bound up to 64, the exact
//! rejection sampler of Canonne, Kamath and Steinke (2020) above, on
//! fixed-width integers, and uniform noise and secrets drawn exactly.

use cubecl::prelude::*;

use super::Column;
use crate::runtime::{Binding, GpuRuntime, Label, launch};

/// Words of a thread's generator: the 16 words of its current block, the
/// index of its next word and its block counter.
const RNG_WORDS: usize = 18;
const NEXT: usize = 16;
const COUNTER: usize = 17;

#[cube]
fn rotl(v: u32, #[comptime] by: u32) -> u32 {
    (v << by) | (v >> comptime!(32 - by))
}

#[cube]
fn quarter(x: &mut Array<u32>, #[comptime] a: usize, #[comptime] b: usize, #[comptime] c: usize, #[comptime] d: usize) {
    x[a] += x[b];
    x[d] = rotl(x[d] ^ x[a], 16u32);
    x[c] += x[d];
    x[b] = rotl(x[b] ^ x[c], 12u32);
    x[a] += x[b];
    x[d] = rotl(x[d] ^ x[a], 8u32);
    x[c] += x[d];
    x[b] = rotl(x[b] ^ x[c], 7u32);
}

/// The next ChaCha8 block of the keystream `stream` under `key`, at the
/// generator's counter: words 0 to 3 the constants, 4 to 11 the key, 12 and 13
/// the block counter and 14 and 15 the stream, as `rand_chacha` lays them out.
#[cube]
fn refill(rng: &mut Array<u32>, key: &Array<u32>, stream: u32) {
    let mut init = Array::<u32>::new(16usize);
    init[0] = 0x6170_7865u32;
    init[1] = 0x3320_646eu32;
    init[2] = 0x7962_2d32u32;
    init[3] = 0x6b20_6574u32;
    #[unroll]
    for i in 0..8usize {
        init[4 + i] = key[i];
    }
    init[12] = rng[COUNTER];
    init[13] = 0u32;
    init[14] = stream;
    init[15] = 0u32;
    let mut x = Array::<u32>::new(16usize);
    #[unroll]
    for i in 0..16usize {
        x[i] = init[i];
    }
    #[unroll]
    for _ in 0..4usize {
        quarter(&mut x, 0usize, 4usize, 8usize, 12usize);
        quarter(&mut x, 1usize, 5usize, 9usize, 13usize);
        quarter(&mut x, 2usize, 6usize, 10usize, 14usize);
        quarter(&mut x, 3usize, 7usize, 11usize, 15usize);
        quarter(&mut x, 0usize, 5usize, 10usize, 15usize);
        quarter(&mut x, 1usize, 6usize, 11usize, 12usize);
        quarter(&mut x, 2usize, 7usize, 8usize, 13usize);
        quarter(&mut x, 3usize, 4usize, 9usize, 14usize);
    }
    #[unroll]
    for i in 0..16usize {
        rng[i] = x[i] + init[i];
    }
    rng[NEXT] = 0u32;
    rng[COUNTER] += 1u32;
}

#[cube]
fn new_rng() -> Array<u32> {
    let mut rng = Array::<u32>::new(RNG_WORDS);
    rng[NEXT] = 16u32;
    rng[COUNTER] = 0u32;
    rng
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn new_key(k0: u32, k1: u32, k2: u32, k3: u32, k4: u32, k5: u32, k6: u32, k7: u32) -> Array<u32> {
    let mut key = Array::<u32>::new(8usize);
    key[0] = k0;
    key[1] = k1;
    key[2] = k2;
    key[3] = k3;
    key[4] = k4;
    key[5] = k5;
    key[6] = k6;
    key[7] = k7;
    key
}

#[cube]
fn next_u32(rng: &mut Array<u32>, key: &Array<u32>, stream: u32) -> u32 {
    if rng[NEXT] == 16u32 {
        refill(rng, key, stream);
    }
    let value = rng[rng[NEXT] as usize];
    rng[NEXT] += 1u32;
    value
}

#[cube]
fn next_u64(rng: &mut Array<u32>, key: &Array<u32>, stream: u32) -> u64 {
    let lo = u64::cast_from(next_u32(rng, key, stream));
    let hi = u64::cast_from(next_u32(rng, key, stream));
    lo | (hi << 32u64)
}

/// A uniform integer below `limit > 0`, by rejection on the bits of `limit - 1`.
#[cube]
fn below_u32(limit: u32, rng: &mut Array<u32>, key: &Array<u32>, stream: u32) -> u32 {
    let mut value = 0u32;
    if limit > 1u32 {
        let mask = u32::MAX >> (limit - 1u32).leading_zeros();
        value = next_u32(rng, key, stream) & mask;
        while value >= limit {
            value = next_u32(rng, key, stream) & mask;
        }
    }
    value
}

/// Adds the signed digit `digit` to word `index` of `res`, an `i64` or the
/// low word of an `i128`, wrapping.
#[cube]
fn add_digit(res: &mut [u64], index: u32, words: u32, digit: u64) {
    let lo = res[index as usize];
    let sum = lo + digit;
    res[index as usize] = sum;
    if words == 2u32 {
        // The high word takes the sign extension of the digit and the carry.
        let carry = u64::cast_from(sum < lo);
        let extension = 0u64 - (digit >> 63u64);
        res[(index + 1u32) as usize] += extension + carry;
    }
}

// Fixed-width unsigned integers, little-endian `u32` limbs, `w` of them. The
// host sizes `w` so that no operation below overflows.

#[cube]
fn mw_set(dst: &mut Array<u32>, value: u32, #[comptime] w: usize) {
    dst[0] = value;
    #[unroll]
    for i in 1..w {
        dst[i] = 0u32;
    }
}

#[cube]
fn mw_copy(dst: &mut Array<u32>, src: &Array<u32>, #[comptime] w: usize) {
    #[unroll]
    for i in 0..w {
        dst[i] = src[i];
    }
}

/// `a < b`.
#[cube]
fn mw_lt(a: &Array<u32>, b: &Array<u32>, #[comptime] w: usize) -> bool {
    let mut lt = false;
    let mut decided = false;
    #[unroll]
    for k in 0..w {
        let i = w - 1 - k;
        if !decided && a[i] != b[i] {
            lt = a[i] < b[i];
            decided = true;
        }
    }
    lt
}

#[cube]
fn mw_is(a: &Array<u32>, value: u32, #[comptime] w: usize) -> bool {
    let mut equal = a[0] == value;
    #[unroll]
    for i in 1..w {
        equal = equal && a[i] == 0u32;
    }
    equal
}

/// `dst = a + b`.
#[cube]
fn mw_add(dst: &mut Array<u32>, a: &Array<u32>, b: &Array<u32>, #[comptime] w: usize) {
    let mut carry = 0u64;
    #[unroll]
    for i in 0..w {
        let sum = u64::cast_from(a[i]) + u64::cast_from(b[i]) + carry;
        dst[i] = u32::cast_from(sum);
        carry = sum >> 32u64;
    }
}

/// `dst = a - b`, `b <= a`.
#[cube]
fn mw_sub(dst: &mut Array<u32>, a: &Array<u32>, b: &Array<u32>, #[comptime] w: usize) {
    let mut borrow = 0u32;
    #[unroll]
    for i in 0..w {
        let x = a[i];
        let y = b[i];
        let d = x - y - borrow;
        borrow = u32::cast_from((x < y) || (x == y && borrow == 1u32));
        dst[i] = d;
    }
}

/// `dst = a * s`.
#[cube]
fn mw_mul_u32(dst: &mut Array<u32>, a: &Array<u32>, s: u32, #[comptime] w: usize) {
    let mut carry = 0u64;
    #[unroll]
    for i in 0..w {
        let product = u64::cast_from(a[i]) * u64::cast_from(s) + carry;
        dst[i] = u32::cast_from(product);
        carry = product >> 32u64;
    }
}

/// `dst = a * b`, the low `w` limbs. Not unrolled: a large sigma takes
/// many limbs, and the kernel stays linear in them.
#[cube]
fn mw_mul(dst: &mut Array<u32>, a: &Array<u32>, b: &Array<u32>, #[comptime] w: usize) {
    for i in 0..w {
        dst[i] = 0u32;
    }
    for i in 0..w {
        let mut carry = 0u64;
        for j in 0..(w - i) {
            let acc = u64::cast_from(dst[i + j]) + u64::cast_from(a[i]) * u64::cast_from(b[j]) + carry;
            dst[i + j] = u32::cast_from(acc);
            carry = acc >> 32u64;
        }
    }
}

/// A uniform integer below `limit > 0` into `dst`, by rejection on the bits of `limit - 1`.
#[cube]
fn mw_below(dst: &mut Array<u32>, limit: &Array<u32>, rng: &mut Array<u32>, key: &Array<u32>, stream: u32, #[comptime] w: usize) {
    let mut candidate = Array::<u32>::new(w);
    mw_set(&mut candidate, 0u32, w);
    if !mw_is(limit, 1u32, w) {
        // The bits of `limit - 1`.
        let mut one = Array::<u32>::new(w);
        mw_set(&mut one, 1u32, w);
        let mut top = Array::<u32>::new(w);
        mw_sub(&mut top, limit, &one, w);
        let mut bits = 0u32;
        #[unroll]
        for i in 0..w {
            if top[i] != 0u32 {
                bits = (i as u32) * 32u32 + 32u32 - top[i].leading_zeros();
            }
        }
        let words = bits.div_ceil(32u32);
        let mask = u32::MAX >> ((32u32 - bits % 32u32) % 32u32);
        let mut accepted = false;
        while !accepted {
            #[unroll]
            for i in 0..w {
                let mut word = 0u32;
                if (i as u32) < words {
                    word = next_u32(rng, key, stream);
                    if (i as u32) + 1u32 == words {
                        word &= mask;
                    }
                }
                candidate[i] = word;
            }
            accepted = mw_lt(&candidate, limit, w);
        }
    }
    mw_copy(dst, &candidate, w);
}

/// Bernoulli(exp(-num / den)) for `num <= den`: the series test of
/// Canonne, Kamath and Steinke, exactly.
#[cube]
fn bernoulli_exp_unit(
    num: &Array<u32>,
    den: &Array<u32>,
    rng: &mut Array<u32>,
    key: &Array<u32>,
    stream: u32,
    #[comptime] w: usize,
) -> bool {
    let mut limit = Array::<u32>::new(w);
    let mut draw = Array::<u32>::new(w);
    let mut j = 1u32;
    let mut done = false;
    let mut result = false;
    while !done {
        mw_mul_u32(&mut limit, den, j, w);
        mw_below(&mut draw, &limit, rng, key, stream, w);
        if mw_lt(&draw, num, w) {
            j += 1u32;
        } else {
            done = true;
            result = (j & 1u32) == 1u32;
        }
    }
    result
}

/// Bernoulli(exp(-1)): the series test on `1 / 1`, in words.
#[cube]
fn bernoulli_exp_one(rng: &mut Array<u32>, key: &Array<u32>, stream: u32) -> bool {
    let mut j = 1u32;
    while below_u32(j, rng, key, stream) == 0u32 {
        j += 1u32;
    }
    (j & 1u32) == 1u32
}

/// Writes the signed integer `value` (two's complement on `w` limbs), shifted
/// left by `shift`, as balanced radix-`2^base2k` digits added to limbs `first`
/// to `size - 1` of coefficient `i` of `res`, the last limb holding the least
/// significant digit.
#[cube]
#[allow(clippy::too_many_arguments)]
fn add_wide_digits(
    res: &mut [u64],
    value: &mut Array<u32>,
    i: u32,
    res_base: u32,
    res_stride: u32,
    words: u32,
    first: u32,
    size: u32,
    base2k: u32,
    shift: u32,
    #[comptime] w: usize,
) {
    // value <<= shift, shift < 64.
    let limb_shift = shift / 32u32;
    let bit_shift = shift % 32u32;
    #[unroll]
    for k in 0..w {
        let j = (w - 1 - k) as u32;
        let mut word = 0u32;
        if j >= limb_shift {
            word = value[(j - limb_shift) as usize] << bit_shift;
            if bit_shift != 0u32 && j > limb_shift {
                word |= value[(j - limb_shift - 1u32) as usize] >> (32u32 - bit_shift);
            }
        }
        value[j as usize] = word;
    }
    let mask = (1u64 << u64::cast_from(base2k)) - 1u64;
    let half = 1u64 << u64::cast_from(base2k - 1u32);
    let digit_limbs = base2k / 32u32;
    let digit_bits = base2k % 32u32;
    let mut limb = size;
    while limb > first {
        limb -= 1u32;
        let low = (u64::cast_from(value[0]) | (u64::cast_from(value[1]) << 32u64)) & mask;
        let up = low >= half;
        let mut digit = low;
        if up {
            digit = low - (mask + 1u64);
        }
        // value = floor(value / 2^base2k) + up, arithmetically.
        let fill = 0u32 - (value[w - 1] >> 31u32);
        #[unroll]
        for j in 0..w {
            let at = j as u32 + digit_limbs;
            let mut a = fill;
            if at < w as u32 {
                a = value[at as usize];
            }
            let mut b = fill;
            if at + 1u32 < w as u32 {
                b = value[(at + 1u32) as usize];
            }
            let mut word = a;
            if digit_bits != 0u32 {
                word = (a >> digit_bits) | (b << (32u32 - digit_bits));
            }
            value[j] = word;
        }
        let mut carry = u32::cast_from(up);
        #[unroll]
        for j in 0..w {
            let sum = value[j] + carry;
            carry = u32::cast_from(carry == 1u32 && sum == 0u32);
            value[j] = sum;
        }
        add_digit(res, res_base + limb * res_stride + words * i, words, digit);
    }
}

/// `Noise::Uniform { bits }`, as `add_uniform` of the portable sampler lays
/// it out: the bits in fixed chunks per digit, sign-extended, balanced.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn uniform_kernel(
    res: &mut [u64],
    n: u32,
    res_base: u32,
    res_stride: u32,
    words: u32,
    size: u32,
    base2k: u32,
    shift: u32,
    bits: u32,
    k0: u32,
    k1: u32,
    k2: u32,
    k3: u32,
    k4: u32,
    k5: u32,
    k6: u32,
    k7: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        let key = new_key(k0, k1, k2, k3, k4, k5, k6, k7);
        let mut rng = new_rng();
        if bits <= base2k - shift {
            let mask = (1u64 << u64::cast_from(bits)) - 1u64;
            let sign_bit = 1u64 << u64::cast_from(bits - 1u32);
            let word = next_u64(&mut rng, &key, i) & mask;
            let signed = (word ^ sign_bit) - sign_bit;
            add_digit(
                res,
                res_base + (size - 1u32) * res_stride + words * i,
                words,
                signed << u64::cast_from(shift),
            );
        } else {
            let mask = (1u64 << u64::cast_from(base2k)) - 1u64;
            let half = 1u64 << u64::cast_from(base2k - 1u32);
            let mut remaining = bits;
            let mut padding = shift;
            let mut sign = 0u64;
            let mut carry = 0u64;
            let mut limb = size;
            while limb > 0u32 {
                limb -= 1u32;
                let mut width = base2k - padding;
                if remaining < width {
                    width = remaining;
                }
                let word = next_u64(&mut rng, &key, i) & ((1u64 << u64::cast_from(width)) - 1u64);
                remaining -= width;
                if width != 0u32 {
                    sign = word >> u64::cast_from(width - 1u32);
                }
                let extension = (0u64 - sign) << u64::cast_from(width + padding);
                let raw = ((word << u64::cast_from(padding)) | extension) & mask;
                let balanced = raw + carry + half;
                let digit = (balanced & mask) - half;
                carry = balanced >> u64::cast_from(base2k);
                padding = 0u32;
                add_digit(res, res_base + limb * res_stride + words * i, words, digit);
            }
        }
    }
}

/// A Gaussian of bound at most 64 by a scan of its cumulative table: `table`
/// holds `entries` 128-bit thresholds as four `u32` words each, most
/// significant first.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn table_kernel(
    res: &mut [u64],
    table: &[u32],
    n: u32,
    res_base: u32,
    res_stride: u32,
    words: u32,
    size: u32,
    first: u32,
    base2k: u32,
    shift: u32,
    entries: u32,
    k0: u32,
    k1: u32,
    k2: u32,
    k3: u32,
    k4: u32,
    k5: u32,
    k6: u32,
    k7: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        let key = new_key(k0, k1, k2, k3, k4, k5, k6, k7);
        let mut rng = new_rng();
        let negative = (next_u32(&mut rng, &key, i) & 1u32) == 1u32;
        let u0 = next_u32(&mut rng, &key, i);
        let u1 = next_u32(&mut rng, &key, i);
        let u2 = next_u32(&mut rng, &key, i);
        let u3 = next_u32(&mut rng, &key, i);
        // Every threshold is scanned, whatever the draw.
        let mut magnitude = 0u32;
        for e in 0..entries {
            let t0 = table[(4u32 * e) as usize];
            let t1 = table[(4u32 * e + 1u32) as usize];
            let t2 = table[(4u32 * e + 2u32) as usize];
            let t3 = table[(4u32 * e + 3u32) as usize];
            let mut ge = u3 >= t3;
            ge = (u2 > t2) || (u2 == t2 && ge);
            ge = (u1 > t1) || (u1 == t1 && ge);
            ge = (u0 > t0) || (u0 == t0 && ge);
            magnitude += u32::cast_from(ge);
        }
        let mut value = Array::<u32>::new(4usize);
        mw_set(&mut value, magnitude, 4usize);
        if negative {
            // Two's complement negation.
            let mut carry = 1u32;
            #[unroll]
            for j in 0..4 {
                let sum = !value[j] + carry;
                carry = u32::cast_from(carry == 1u32 && sum == 0u32);
                value[j] = sum;
            }
        }
        add_wide_digits(
            res, &mut value, i, res_base, res_stride, words, first, size, base2k, shift, 4usize,
        );
    }
}

/// A Gaussian of bound above 64, exactly, by the rejection sampler of
/// Canonne, Kamath and Steinke (2020), Algorithm 3, on `w`-limb integers:
/// `consts` holds `t`, the bound, the numerator and denominator of the
/// variance and the denominator of the acceptance test, `w` limbs each.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn rejection_kernel(
    res: &mut [u64],
    consts: &[u32],
    n: u32,
    res_base: u32,
    res_stride: u32,
    words: u32,
    size: u32,
    base2k: u32,
    shift: u32,
    // `floor(bound / t)`: a larger `v` gives `y > bound`.
    v_max: u32,
    k0: u32,
    k1: u32,
    k2: u32,
    k3: u32,
    k4: u32,
    k5: u32,
    k6: u32,
    k7: u32,
    #[comptime] w: usize,
) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        let key = new_key(k0, k1, k2, k3, k4, k5, k6, k7);
        let mut rng = new_rng();
        let mut t = Array::<u32>::new(w);
        let mut bound = Array::<u32>::new(w);
        let mut variance_num = Array::<u32>::new(w);
        let mut variance_den = Array::<u32>::new(w);
        let mut acceptance_den = Array::<u32>::new(w);
        #[unroll]
        for j in 0..w {
            t[j] = consts[j];
            bound[j] = consts[w + j];
            variance_num[j] = consts[2 * w + j];
            variance_den[j] = consts[3 * w + j];
            acceptance_den[j] = consts[4 * w + j];
        }
        let mut u = Array::<u32>::new(w);
        let mut y = Array::<u32>::new(w);
        let mut a = Array::<u32>::new(w);
        let mut b = Array::<u32>::new(w);
        let mut negative = false;
        let mut accepted = false;
        while !accepted {
            let mut drawn = false;
            while !drawn {
                mw_below(&mut u, &t, &mut rng, &key, i, w);
                drawn = bernoulli_exp_unit(&u, &t, &mut rng, &key, i, w);
            }
            let mut v = 0u32;
            while bernoulli_exp_one(&mut rng, &key, i) {
                v += 1u32;
            }
            negative = (next_u32(&mut rng, &key, i) & 1u32) == 1u32;
            let mut ok = v <= v_max;
            if ok {
                mw_mul_u32(&mut a, &t, v, w);
                mw_add(&mut y, &u, &a, w);
                ok = !(negative && mw_is(&y, 0u32, w)) && !mw_lt(&bound, &y, w);
            }
            if ok {
                // (y t vd - vn)^2 against the acceptance denominator.
                mw_mul(&mut a, &y, &t, w);
                mw_mul(&mut b, &a, &variance_den, w);
                if mw_lt(&b, &variance_num, w) {
                    mw_sub(&mut a, &variance_num, &b, w);
                } else {
                    mw_sub(&mut a, &b, &variance_num, w);
                }
                mw_mul(&mut b, &a, &a, w);
                // Bernoulli(exp(-b / den)): exp(-1) once per whole unit, then the remainder.
                let mut whole = 0u32;
                while !mw_lt(&b, &acceptance_den, w) {
                    mw_sub(&mut a, &b, &acceptance_den, w);
                    mw_copy(&mut b, &a, w);
                    whole += 1u32;
                }
                let mut unit = 0u32;
                while ok && unit < whole {
                    ok = bernoulli_exp_one(&mut rng, &key, i);
                    unit += 1u32;
                }
                if ok {
                    ok = bernoulli_exp_unit(&b, &acceptance_den, &mut rng, &key, i, w);
                }
            }
            accepted = ok;
        }
        if negative {
            let mut carry = 1u32;
            #[unroll]
            for j in 0..w {
                let sum = !y[j] + carry;
                carry = u32::cast_from(carry == 1u32 && sum == 0u32);
                y[j] = sum;
            }
        }
        add_wide_digits(res, &mut y, i, res_base, res_stride, words, 0u32, size, base2k, shift, w);
    }
}

/// A ternary or binary secret, each coefficient nonzero with probability
/// `threshold / 2^64` (`always`: one), the sign of a ternary one uniform.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn probability_kernel(
    res: &mut [u64],
    n: u32,
    res_base: u32,
    threshold: u64,
    always: u32,
    k0: u32,
    k1: u32,
    k2: u32,
    k3: u32,
    k4: u32,
    k5: u32,
    k6: u32,
    k7: u32,
    #[comptime] ternary: bool,
) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        let key = new_key(k0, k1, k2, k3, k4, k5, k6, k7);
        let mut rng = new_rng();
        let draw = next_u64(&mut rng, &key, i);
        let mut value = 0u64;
        if always == 1u32 || draw < threshold {
            value = 1u64;
            if ternary && (next_u32(&mut rng, &key, i) & 1u32) == 1u32 {
                value = u64::MAX;
            }
        }
        res[(res_base + i) as usize] = value;
    }
}

/// Exactly `weight` nonzero coefficients at uniform positions, by Floyd's
/// algorithm on one thread: the column, zeroed beforehand, marks the chosen
/// positions.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn fixed_weight_kernel(
    res: &mut [u64],
    n: u32,
    res_base: u32,
    weight: u32,
    k0: u32,
    k1: u32,
    k2: u32,
    k3: u32,
    k4: u32,
    k5: u32,
    k6: u32,
    k7: u32,
    #[comptime] ternary: bool,
) {
    if ABSOLUTE_POS == 0 {
        let key = new_key(k0, k1, k2, k3, k4, k5, k6, k7);
        let mut rng = new_rng();
        for j in (n - weight)..n {
            let t = below_u32(j + 1u32, &mut rng, &key, 0u32);
            let mut position = t;
            if res[(res_base + t) as usize] != 0u64 {
                position = j;
            }
            let mut value = 1u64;
            if ternary && (next_u32(&mut rng, &key, 0u32) & 1u32) == 1u32 {
                value = u64::MAX;
            }
            res[(res_base + position) as usize] = value;
        }
    }
}

/// One `1` at a uniform position of each block of `block` coefficients, or
/// none with probability `1 / (block + 1)`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn block_kernel(
    res: &mut [u64],
    blocks: u32,
    res_base: u32,
    block: u32,
    k0: u32,
    k1: u32,
    k2: u32,
    k3: u32,
    k4: u32,
    k5: u32,
    k6: u32,
    k7: u32,
) {
    let b = ABSOLUTE_POS as u32;
    if b < blocks {
        let key = new_key(k0, k1, k2, k3, k4, k5, k6, k7);
        let mut rng = new_rng();
        let index = below_u32(block + 1u32, &mut rng, &key, b);
        for j in 0..block {
            res[(res_base + b * block + j) as usize] = u64::cast_from(j == index);
        }
    }
}

/// The key words of a seed.
fn key(seed: &[u8; 32]) -> [u32; 8] {
    std::array::from_fn(|i| u32::from_le_bytes(seed[4 * i..4 * i + 4].try_into().expect("four bytes")))
}

/// Where noise lands: the first `size` limbs of a column, at radix `2^base2k`,
/// the sample shifted left by `shift` bits of precision padding.
pub(crate) struct Placement {
    pub(crate) size: usize,
    pub(crate) base2k: usize,
    pub(crate) shift: usize,
}

/// Adds uniform noise of `bits` bits to the first `place.size` limbs of `res`.
pub(crate) fn add_uniform<RT: GpuRuntime>(res: &Column, place: &Placement, bits: usize, seed: &[u8; 32]) {
    let k = key(seed);
    let units = res.n;
    launch!(
        RT, Label::SampleUniform, units * place.size * 16 * res.words as usize, uniform_kernel, (units, RT::GROUP, RT::GROUP);
        [res.binding => u64];
        [
            res.n as u32,
            res.base,
            res.stride,
            res.words,
            place.size as u32,
            place.base2k as u32,
            place.shift as u32,
            u32::try_from(bits).expect("uniform noise width overflows u32"),
            k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7],
        ];
        []
    );
}

/// Adds a Gaussian through its cumulative `table` (`entries` thresholds) to
/// limbs `first..place.size` of `res`.
pub(crate) fn add_table<RT: GpuRuntime>(
    res: &Column,
    place: &Placement,
    first: usize,
    table: &Binding,
    entries: usize,
    seed: &[u8; 32],
) {
    let k = key(seed);
    let units = res.n;
    launch!(
        RT, Label::SampleTable, units * ((place.size - first) * 16 * res.words as usize + entries * 16), table_kernel, (units, RT::GROUP, RT::GROUP);
        [res.binding => u64, table => u32];
        [
            res.n as u32,
            res.base,
            res.stride,
            res.words,
            place.size as u32,
            first as u32,
            place.base2k as u32,
            place.shift as u32,
            entries as u32,
            k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7],
        ];
        []
    );
}

/// Adds an exact Gaussian by rejection to the first `place.size` limbs of
/// `res`; `consts` holds its constants on `limbs` limbs each.
pub(crate) fn add_rejection<RT: GpuRuntime>(
    res: &Column,
    place: &Placement,
    consts: &Binding,
    limbs: usize,
    v_max: u32,
    seed: &[u8; 32],
) {
    let k = key(seed);
    let units = res.n;
    launch!(
        RT, Label::SampleRejection, units * (place.size * 16 * res.words as usize + limbs * 20), rejection_kernel, (units, RT::GROUP, RT::GROUP);
        [res.binding => u64, consts => u32];
        [
            res.n as u32,
            res.base,
            res.stride,
            res.words,
            place.size as u32,
            place.base2k as u32,
            place.shift as u32,
            v_max,
            k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7],
        ];
        [limbs]
    );
}

/// Overwrites the one-limb `i64` column `res` with a secret nonzero with
/// probability `threshold / 2^64` per coefficient (`None`: always).
pub(crate) fn fill_probability<RT: GpuRuntime>(res: &Column, threshold: Option<u64>, ternary: bool, seed: &[u8; 32]) {
    let k = key(seed);
    let units = res.n;
    launch!(
        RT, Label::SampleSecret, units * 8, probability_kernel, (units, RT::GROUP, RT::GROUP);
        [res.binding => u64];
        [
            res.n as u32,
            res.base,
            threshold.unwrap_or(0),
            threshold.is_none() as u32,
            k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7],
        ];
        [ternary]
    );
}

/// Sets `weight` coefficients of the zeroed one-limb column `res` to `1`
/// (ternary: `+-1`), at uniform positions.
pub(crate) fn fill_fixed_weight<RT: GpuRuntime>(res: &Column, weight: usize, ternary: bool, seed: &[u8; 32]) {
    let k = key(seed);
    launch!(
        RT, Label::SampleSecret, weight * 16, fixed_weight_kernel, (1, RT::GROUP, RT::GROUP);
        [res.binding => u64];
        [
            res.n as u32,
            res.base,
            weight as u32,
            k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7],
        ];
        [ternary]
    );
}

/// Overwrites the one-limb column `res` with at most one `1` per block of `block` coefficients.
pub(crate) fn fill_block<RT: GpuRuntime>(res: &Column, block: usize, seed: &[u8; 32]) {
    let k = key(seed);
    let blocks = res.n / block;
    launch!(
        RT, Label::SampleSecret, res.n * 8, block_kernel, (blocks, RT::GROUP, RT::GROUP);
        [res.binding => u64];
        [
            blocks as u32,
            res.base,
            block as u32,
            k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7],
        ];
        []
    );
}

/// The first `words` words of keystream `stream` under `seed`, for the test
/// that the generator is ChaCha8.
#[cfg(test)]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn keystream_kernel(
    out: &mut [u32],
    words: u32,
    stream: u32,
    k0: u32,
    k1: u32,
    k2: u32,
    k3: u32,
    k4: u32,
    k5: u32,
    k6: u32,
    k7: u32,
) {
    if ABSOLUTE_POS == 0 {
        let key = new_key(k0, k1, k2, k3, k4, k5, k6, k7);
        let mut rng = new_rng();
        for j in 0..words {
            out[j as usize] = next_u32(&mut rng, &key, stream);
        }
    }
}

#[cfg(test)]
pub(crate) fn keystream<RT: GpuRuntime>(seed: &[u8; 32], stream: u32, words: usize) -> Vec<u32> {
    let out = crate::runtime::GpuBuf::<RT>::zeroed(words * 4);
    let binding = out.view().binding().expect("buffer holds storage");
    let k = key(seed);
    launch!(
        RT, Label::SampleSecret, words * 4, keystream_kernel, (1, RT::GROUP, RT::GROUP);
        [binding => u32];
        [words as u32, stream, k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]];
        []
    );
    bytemuck::cast_slice(&out.to_host()[..words * 4]).to_vec()
}
