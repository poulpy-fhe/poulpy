//! Gadget digit layout of a column, see
//! [`VecZnxGadgetDecompose`](poulpy_hal::api::VecZnxGadgetDecompose).
//!
//! Each thread owns one coefficient and follows the portable kernel: the input
//! is rounded at the gadget precision, ties to even, then the limbs of the
//! digit grid are emitted from the bottom up through a carry. A digit's lower
//! limbs are centered at `base2k` bits; its top limb is centered at its own
//! width and moved up by `2^w` when it sits at `-2^(w - 1)` over negative
//! lower limbs, which centers the digit at `k_digit` bits.

use cubecl::prelude::*;

use super::Column;
use crate::runtime::{GpuRuntime, Label, launch};

/// `x >> s` on the two's complement `x`, for `s` in `0..64`, with the sign
/// extension written out: the runtimes do not agree on signed 64-bit shifts.
#[cube]
fn asr(x: u64, s: u64) -> u64 {
    let mut y = x;
    if s != 0u64 {
        y = (x >> s) | ((0u64 - (x >> 63u64)) << (64u64 - s));
    }
    y
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn gadget_decompose_kernel(
    res: &mut [u64],
    a: &[u64],
    n: u32,
    res_base: u32,
    res_stride: u32,
    a_base: u32,
    a_stride: u32,
    a_size: u32,
    base2k: u32,
    // Limbs per digit, and the width of the top one.
    r: u32,
    rem: u32,
    dnum: u32,
    gadget_k: u32,
    // Limb holding bit `gadget_k`, and its bits below it.
    j0: u32,
    low: u32,
) {
    // Every value is two's complement in unsigned registers.
    let i = ABSOLUTE_POS as u32;
    if i < n {
        let b = base2k as u64;

        // Rounding at `gadget_k`: half up, the low bit cleared when every dropped bit,
        // after the half, is zero.
        let mut acc = 0u64;
        if j0 < a_size {
            let mut carry = 0u64;
            let mut sticky = 0u64;
            if j0 + 1u32 == a_size && low == 0u32 {
                sticky = 1u64;
            }
            let limb_mask = (1u64 << b) - 1u64;
            let mut j = a_size;
            while j > j0 + 1u32 {
                j -= 1u32;
                let mut half = 0u64;
                if j == j0 + 1u32 && low == 0u32 {
                    half = 1u64 << (b - 1u64);
                }
                let s = a[(a_base + j * a_stride + i) as usize] + carry + half;
                carry = asr(s, b);
                sticky |= s & limb_mask;
            }
            let l = low as u64;
            let mut half = 0u64;
            if low > 0u32 {
                half = 1u64 << (l - 1u64);
            }
            let low_mask = (1u64 << l) - 1u64;
            let s = a[(a_base + j0 * a_stride + i) as usize] + carry + half;
            acc = asr(s, l);
            if (sticky | (s & low_mask)) == 0u64 {
                acc &= !1u64;
            }
        }

        // Cells from the bottom up, `acc` in units of the current one.
        let mut next = j0;
        if a_size < next {
            next = a_size;
        }
        let mut pos = gadget_k;
        for qq in 0..dnum {
            let q = dnum - 1u32 - qq;
            let mut neg = 0u64;
            for t in 0..r {
                let mut w = base2k;
                if t == r - 1u32 {
                    w = rem;
                }
                let top = t == r - 1u32 && r > 1u32;
                let wu = w as u64;
                let half = 1u64 << (wu - 1u64);
                let mask = (1u64 << wu) - 1u64;
                let mut y = acc;
                let mut high = 0u64;
                if next > 0u32 && next * base2k > pos - w {
                    let s = (pos - next * base2k) as u64;
                    let split = wu - s;
                    next -= 1u32;
                    let v = a[(a_base + next * a_stride + i) as usize];
                    y = acc + ((v & ((1u64 << split) - 1u64)) << s);
                    high = asr(v, split);
                }
                let centered = ((y + half) & mask) - half;
                let mut d = centered;
                if top && centered == 0u64 - half && neg != 0u64 {
                    d = centered + (half << 1u64);
                }
                acc = asr(y - d, wu) + high;
                if !top && d != 0u64 {
                    // The sign of the last nonzero lower limb.
                    neg = d >> 63u64;
                }
                res[(res_base + (q * r + r - 1u32 - t) * res_stride + i) as usize] = d;
                pos -= w;
            }
        }
    }
}

/// Lays `a` out on the digit grid of `k_digit`-bit digits in base `2^base2k`,
/// into the `size` limbs of `res`, a nonzero multiple of `ceil(k_digit / base2k)`.
pub(crate) fn gadget_decompose<RT: GpuRuntime>(res: &Column, a: &Column, base2k: usize, k_digit: usize) {
    assert!((1..=62).contains(&base2k), "base2k outside 1..=62");
    assert!(k_digit > 0, "k_digit must be positive");
    assert!(res.n == a.n, "gadget layout between different degrees");
    let r = k_digit.div_ceil(base2k);
    let rem = k_digit - (r - 1) * base2k;
    assert!(
        res.size > 0 && res.size.is_multiple_of(r),
        "res.size() is not a nonzero multiple of the digit's sub-limbs"
    );
    if res.n == 0 {
        return;
    }
    let dnum = res.size / r;
    let gadget_k = dnum * k_digit;
    let j0 = gadget_k.div_ceil(base2k) - 1;
    let low = (j0 + 1) * base2k - gadget_k;
    launch!(
        RT, Label::GadgetDecompose, res.n * 8 * (res.size + a.size), gadget_decompose_kernel, (res.n, RT::GROUP, RT::GROUP);
        [
            res.binding => u64,
            a.binding => u64,
        ];
        [
            res.n as u32,
            res.base,
            res.stride,
            a.base,
            a.stride,
            a.size as u32,
            base2k as u32,
            r as u32,
            rem as u32,
            dnum as u32,
            gadget_k as u32,
            j0 as u32,
            low as u32,
        ];
        []
    );
}
