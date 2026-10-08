//! Normalization: `res = canon(a * 2^offset, res_base2k, res_k, res.size())`.
//!
//! Each thread owns one coefficient and follows the exact definition: the
//! limbs of `a` are read from the least significant one as a stream of bits
//! (carrying across unnormalized limbs), the bits below the output precision
//! are dropped with one rounding bit and a sticky bit, rounding ties to even,
//! and the output digits are emitted from the least significant one, each
//! centered by the carry of the previous.

use cubecl::prelude::*;

use super::{Column, DIM};
use crate::runtime::{GpuRuntime, cubes, launched};

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn normalize_kernel(
    res: &mut [i64],
    a: &[i64],
    n: u32,
    res_base: u32,
    res_stride: u32,
    res_size: u32,
    a_base: u32,
    a_stride: u32,
    a_size: u32,
    a_base2k: u32,
    res_base2k: u32,
    // Limbs that carry precision, and the unused low bits of the last of them.
    active_size: u32,
    precision_padding: u32,
    // Bits of `a` below the output precision when positive; zero bits to
    // insert below `a` when negative.
    drop: i64,
) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        let ab = a_base2k as u64;
        let ab_signed = a_base2k as i64;
        let mask = (1u64 << ab) - 1u64;

        // Bit stream over the limbs of `a`, least significant first.
        let mut left = a_size;
        let mut carry = 0i64;
        let mut pending = 0u64;
        let mut available = 0u32;

        let mut rounding = 0u64;
        // Any one among the bits below the rounding bit.
        let mut sticky = 0u64;
        let mut padding = 0i64;
        if drop < 0i64 {
            padding = 0i64 - drop;
        }

        // Field 0 discards `drop - 1` bits, field 1 reads the rounding bit, and
        // each later field reads one output digit, least significant first.
        let fields = active_size + 2u32;
        for field in 0..fields {
            let mut count = 0u32;
            let mut width = 0u32;
            let mut zeroes = 0u32;
            let discard = field == 0u32;
            if field == 0u32 {
                if drop > 1i64 {
                    count = (drop - 1i64) as u32;
                }
            } else if field == 1u32 {
                if drop > 0i64 {
                    count = 1u32;
                }
            } else {
                width = res_base2k;
                if field == 2u32 && precision_padding != 0u32 {
                    width = res_base2k - precision_padding;
                }
                if padding > 0i64 {
                    if padding >= width as i64 {
                        zeroes = width;
                    } else {
                        zeroes = padding as u32;
                    }
                    padding -= zeroes as i64;
                }
                count = width - zeroes;
            }

            let mut result = 0u64;
            let mut written = 0u32;
            while count != 0u32 {
                if available == 0u32 {
                    if left != 0u32 {
                        left -= 1u32;
                        let x = a[(a_base + left * a_stride + i) as usize];
                        let low = ((x as u64) & mask) + ((carry as u64) & mask);
                        carry = (x >> ab_signed) + (carry >> ab_signed) + ((low >> ab) as i64);
                        pending = low & mask;
                        available = a_base2k;
                    } else if discard && (carry == 0i64 || carry == -1i64) {
                        // Only the sign extension is left to discard: ones below a negative value.
                        if carry == -1i64 {
                            sticky = 1u64;
                        }
                        count = 0u32;
                    } else {
                        pending = (carry as u64) & mask;
                        carry >>= ab_signed;
                        available = a_base2k;
                    }
                }
                if count != 0u32 {
                    let mut take = count;
                    if available < take {
                        take = available;
                    }
                    let bits = pending & ((1u64 << (take as u64)) - 1u64);
                    if discard {
                        sticky |= bits;
                    } else {
                        result |= bits << (written as u64);
                        written += take;
                    }
                    pending >>= take as u64;
                    available -= take;
                    count -= take;
                }
            }

            if field == 1u32 {
                rounding = result;
            } else if field >= 2u32 {
                let w = width as u64;
                let mut value = result << (zeroes as u64);
                if field == 2u32 {
                    // Ties to even: up past a half, and at a half onto an odd digit.
                    let odd = value & 1u64;
                    let mut up = 0u64;
                    if sticky != 0u64 || odd != 0u64 {
                        up = rounding;
                    }
                    value += up;
                } else {
                    value += rounding;
                }
                rounding = (value + (1u64 << (w - 1u64))) >> w;
                let mut digit = (value as i64) - ((rounding << w) as i64);
                if field == 2u32 && precision_padding != 0u32 {
                    digit <<= precision_padding as i64;
                }
                let limb = active_size + 1u32 - field;
                res[(res_base + limb * res_stride + i) as usize] = digit;
            }
        }

        for limb in active_size..res_size {
            res[(res_base + limb * res_stride + i) as usize] = 0i64;
        }
    }
}

/// Normalizes every limb of `res` from `a`. `res` may be `a` itself when
/// `offset <= 0` and both radices are equal: a digit is only written once the
/// limbs it depends on have been read.
pub(crate) fn normalize<RT: GpuRuntime>(res: &Column, res_base2k: usize, res_k: usize, offset: i64, a: &Column, a_base2k: usize) {
    assert_eq!(res.n, a.n, "normalization between different degrees");
    assert!(
        (1..=62).contains(&res_base2k) && (1..=62).contains(&a_base2k),
        "radix widths must be in 1..=62"
    );
    assert!(res_k <= res.size * res_base2k, "res_k exceeds the destination precision");
    if res.size == 0 || res.n == 0 {
        return;
    }
    let (active_size, precision_padding) = if res_k == res.size * res_base2k {
        (res.size, 0)
    } else {
        let active_size = res_k.div_ceil(res_base2k);
        (active_size, active_size * res_base2k - res_k)
    };
    let drop = a.size as i64 * a_base2k as i64 - res_k as i64 - offset;
    let client = RT::client();
    unsafe {
        normalize_kernel::launch_unchecked(
            client,
            cubes(client, res.n, DIM),
            CubeDim::new_1d(DIM),
            res.binding.arg::<i64>(),
            a.binding.arg::<i64>(),
            res.n as u32,
            res.base,
            res.stride,
            res.size as u32,
            a.base,
            a.stride,
            a.size as u32,
            a_base2k as u32,
            res_base2k as u32,
            active_size as u32,
            precision_padding as u32,
            drop,
        );
    }
    launched(res.n * 8 * (res.size + a.size));
}
