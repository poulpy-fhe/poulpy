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

use super::{Column, DIM};
use crate::runtime::{GpuRuntime, cubes, launched};

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn gadget_decompose_kernel(
    res: &mut [i64],
    a: &[i64],
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
    let i = ABSOLUTE_POS as u32;
    if i < n {
        let b = base2k as i64;

        // Rounding at `gadget_k`: half up, the low bit cleared when every dropped bit,
        // after the half, is zero.
        let mut acc = 0i64;
        if j0 < a_size {
            let mut carry = 0i64;
            let mut sticky = 0i64;
            if j0 + 1u32 == a_size && low == 0u32 {
                sticky = 1i64;
            }
            let limb_mask = (1i64 << b) - 1i64;
            let mut j = a_size;
            while j > j0 + 1u32 {
                j -= 1u32;
                let mut half = 0i64;
                if j == j0 + 1u32 && low == 0u32 {
                    half = 1i64 << (b - 1i64);
                }
                let s = a[(a_base + j * a_stride + i) as usize] + carry + half;
                carry = s >> b;
                sticky |= s & limb_mask;
            }
            let l = low as i64;
            let mut half = 0i64;
            if low > 0u32 {
                half = 1i64 << (l - 1i64);
            }
            let low_mask = (1i64 << l) - 1i64;
            let s = a[(a_base + j0 * a_stride + i) as usize] + carry + half;
            acc = s >> l;
            if (sticky | (s & low_mask)) == 0i64 {
                acc &= -2i64;
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
            let mut neg = 0i64;
            for t in 0..r {
                let mut w = base2k;
                if t == r - 1u32 {
                    w = rem;
                }
                let top = t == r - 1u32 && r > 1u32;
                let wi = w as i64;
                let half = 1i64 << (wi - 1i64);
                let mask = (1i64 << wi) - 1i64;
                let mut y = acc;
                let mut high = 0i64;
                if next > 0u32 && next * base2k > pos - w {
                    let s = (pos - next * base2k) as i64;
                    let split = wi - s;
                    next -= 1u32;
                    let v = a[(a_base + next * a_stride + i) as usize];
                    y = acc + ((v & ((1i64 << split) - 1i64)) << s);
                    high = v >> split;
                }
                let centered = ((y + half) & mask) - half;
                let mut d = centered;
                if top && centered == 0i64 - half && neg != 0i64 {
                    d = centered + (half << 1i64);
                }
                acc = ((y - d) >> wi) + high;
                if !top && d != 0i64 {
                    // The sign of the last nonzero lower limb.
                    neg = d >> 63i64;
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
    assert!((1..=62).contains(&base2k), "base2k={base2k} outside 1..=62");
    assert!(k_digit > 0, "k_digit must be positive");
    assert_eq!(res.n, a.n, "gadget layout between different degrees");
    let r = k_digit.div_ceil(base2k);
    let rem = k_digit - (r - 1) * base2k;
    assert!(
        res.size > 0 && res.size.is_multiple_of(r),
        "res.size()={} is not a nonzero multiple of {r} sub-limbs",
        res.size
    );
    if res.n == 0 {
        return;
    }
    let dnum = res.size / r;
    let gadget_k = dnum * k_digit;
    let j0 = gadget_k.div_ceil(base2k) - 1;
    let low = (j0 + 1) * base2k - gadget_k;
    let client = RT::client();
    unsafe {
        gadget_decompose_kernel::launch_unchecked(
            client,
            cubes(client, res.n, DIM),
            CubeDim::new_1d(DIM),
            res.binding.arg::<i64>(),
            a.binding.arg::<i64>(),
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
        );
    }
    launched(res.n * 8 * (res.size + a.size));
}
