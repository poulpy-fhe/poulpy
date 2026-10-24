//! Gadget digit layout of a coefficient column, see
//! [`VecZnxGadgetDecompose`](poulpy_hal::api::VecZnxGadgetDecompose).
//!
//! The column is read limb by limb from the least significant one and the
//! limbs of the digit grid are emitted from the bottom up, carrying through
//! a per-coefficient accumulator. Coefficients go by chunks so that the carry
//! buffers stay in L1, and every inner loop is a plain map over a chunk.
//!
//! A digit's lower limbs are centered at `base2k` bits. Its top limb is
//! first centered at its own width `w` and then moved up by `2^w` when it
//! sits at `-2^(w - 1)` and the lower limbs are negative: that is exactly
//! when the digit, centered at `k_digit` bits, takes its top value. The sign
//! of the lower limbs is the sign of the last nonzero one emitted. Every
//! inner loop is branch free.

use poulpy_hal::layouts::{
    Backend, HostDataMut, HostDataRef, VecZnxBackendMut, VecZnxBackendRef, VecZnxShape, ZnxView, ZnxViewMut,
};

use crate::kernels::znx_normalize::{Limbs, LimbsMut};

/// Coefficients per pass: three `i64` buffers of this length live on the stack.
const CHUNK: usize = 512;

pub fn vec_znx_gadget_decompose_portable<'r, 'a, BE>(
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_col: usize,
    base2k: usize,
    k_digit: usize,
) where
    BE: Backend<ZnxWord = i64>,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    assert!(res_col < res.cols(), "operand column out of range");
    if res.size() > 0 {
        let _ = res.at_mut(res_col, res.size() - 1);
    }
    let (n, shape) = (res.n(), res.shape());
    let ptr: *mut i64 = res.base_mut_ptr();
    // SAFETY: `ptr` and `shape` come from `res`, bounds-checked above, and `&mut res` is exclusive.
    unsafe { vec_znx_gadget_decompose_range_raw_portable::<BE>(ptr, shape, res_col, a, a_col, base2k, k_digit, 0, n) }
}

/// [`vec_znx_gadget_decompose_portable`] restricted to the coefficients
/// `coeff_start..coeff_start + coeff_len` of `res[res_col]`: every coefficient
/// decomposes on its own, so disjoint ranges split freely.
///
/// # Safety
///
/// `res_ptr` is the base of the dense buffer `res_shape` describes, and the
/// selected range of column `res_col` is exclusively writable for this call and
/// does not overlap `a`. Concurrent calls may share the destination allocation
/// only with disjoint coefficient ranges.
#[doc(hidden)]
pub unsafe fn vec_znx_gadget_decompose_range_raw_portable<'a, BE>(
    res_ptr: *mut i64,
    res_shape: VecZnxShape,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_col: usize,
    base2k: usize,
    k_digit: usize,
    coeff_start: usize,
    coeff_len: usize,
) where
    BE: Backend<ZnxWord = i64>,
    BE::BufRef<'a>: HostDataRef,
{
    assert!((1..=62).contains(&base2k), "base2k={base2k} outside 1..=62");
    assert!(k_digit > 0, "k_digit must be positive");
    assert_eq!(res_shape.n(), a.n());
    assert!(res_col < res_shape.cols(), "operand column out of range");
    assert!(coeff_start + coeff_len <= res_shape.n(), "coefficient range out of bounds");
    let r: usize = k_digit.div_ceil(base2k);
    let rem: usize = k_digit - (r - 1) * base2k;
    assert!(
        res_shape.size() > 0 && res_shape.size().is_multiple_of(r),
        "res.size()={} is not a nonzero multiple of {r} sub-limbs",
        res_shape.size()
    );
    let res = LimbsMut::new(res_ptr, res_shape, res_col, coeff_start);
    let dnum: usize = res_shape.size() / r;
    let n: usize = coeff_len;
    let a_size: usize = a.size();
    let gadget_k: usize = dnum * k_digit;
    // Limb holding bit `gadget_k` (bit 1 is the most significant), and its bits below it.
    let j0: usize = gadget_k.div_ceil(base2k) - 1;
    let low: usize = (j0 + 1) * base2k - gadget_k;
    let nothing_dropped: i64 = (j0 + 1 == a_size && low == 0) as i64;

    assert!(a_col < a.cols(), "operand column out of range");
    // Bounds-checks the last limb of the source column, the farthest one the kernel reads.
    if a_size > 0 {
        let _ = a.at_ptr(a_col, a_size - 1);
    }
    let a_limbs = Limbs::new(a.base_ptr(), a.shape(), a_col, coeff_start);

    let mut acc = [0i64; CHUNK];
    let mut carry = [0i64; CHUNK];
    let mut neg = [0i64; CHUNK];
    let mut start = 0;
    while start < n {
        let len = CHUNK.min(n - start);
        let (acc, carry, neg) = (&mut acc[..len], &mut carry[..len], &mut neg[..len]);
        // Both limb views start at `coeff_start`. SAFETY: every limb of `a` is
        // bounds-checked above; the caller hands the range of each limb of `res`
        // over for the call, distinct from `a`.
        let a_limb = |j: usize| unsafe { std::slice::from_raw_parts(a_limbs.ptr.add(j * a_limbs.stride + start), len) };
        let res_limb = |j: usize| unsafe { std::slice::from_raw_parts_mut(res.ptr.add(j * res.stride + start), len) };

        // Ties to even at `gadget_k`: half up, low bit cleared when `acc`, sticky over the dropped remainders, is zero.
        if j0 < a_size {
            carry.fill(0);
            acc.fill(nothing_dropped);
            let limb_mask: i64 = (1i64 << base2k) - 1;
            for j in (j0 + 1..a_size).rev() {
                let half: i64 = if j == j0 + 1 && low == 0 { 1 << (base2k - 1) } else { 0 };
                for ((c, x), &v) in carry.iter_mut().zip(acc.iter_mut()).zip(a_limb(j)) {
                    let s = v + *c + half;
                    *c = s >> base2k;
                    *x |= s & limb_mask;
                }
            }
            let half: i64 = if low > 0 { 1 << (low - 1) } else { 0 };
            let low_mask: i64 = (1i64 << low) - 1;
            for ((x, &c), &v) in acc.iter_mut().zip(carry.iter()).zip(a_limb(j0)) {
                let s = v + c + half;
                let tie: i64 = ((*x | (s & low_mask)) == 0) as i64;
                *x = (s >> low) & !tie;
            }
        } else {
            acc.fill(0);
        }

        // Cells from the bottom up, `acc` in units of the current one. At most
        // one input limb has its least significant bit in a cell: it is split
        // there, its low part joining the cell and its high part the carry, so
        // no shift overflows.
        let mut next: usize = j0.min(a_size);
        let mut pos: usize = gadget_k;
        for q in (0..dnum).rev() {
            if r > 1 {
                neg.fill(0);
            }
            for t in 0..r {
                let w: usize = if t == r - 1 { rem } else { base2k };
                let top: bool = t == r - 1 && r > 1;
                let half: i64 = 1 << (w - 1);
                let mask: i64 = (1i64 << w) - 1;
                let limb = if next > 0 && next * base2k > pos - w {
                    let s: usize = pos - next * base2k;
                    next -= 1;
                    Some((s as u32, (w - s) as u32, a_limb(next)))
                } else {
                    None
                };
                let out = res_limb(q * r + r - 1 - t);
                match (limb, top) {
                    (Some((s, split, v)), false) if r == 1 => {
                        let lo_mask: i64 = (1i64 << split) - 1;
                        for ((x, o), &v) in acc.iter_mut().zip(out.iter_mut()).zip(v) {
                            let y = *x + ((v & lo_mask) << s);
                            let d = ((y + half) & mask) - half;
                            *x = ((y - d) >> w) + (v >> split);
                            *o = d;
                        }
                    }
                    (Some((s, split, v)), false) => {
                        let lo_mask: i64 = (1i64 << split) - 1;
                        for (((x, o), n), &v) in acc.iter_mut().zip(out.iter_mut()).zip(neg.iter_mut()).zip(v) {
                            let y = *x + ((v & lo_mask) << s);
                            let d = ((y + half) & mask) - half;
                            *x = ((y - d) >> w) + (v >> split);
                            *o = d;
                            *n = track_sign(*n, d);
                        }
                    }
                    (None, false) if r == 1 => {
                        // One limb per digit: no top limb to correct, no sign to keep.
                        for (x, o) in acc.iter_mut().zip(out.iter_mut()) {
                            let d = ((*x + half) & mask) - half;
                            *x = (*x - d) >> w;
                            *o = d;
                        }
                    }
                    (None, false) => {
                        for ((x, o), n) in acc.iter_mut().zip(out.iter_mut()).zip(neg.iter_mut()) {
                            let d = ((*x + half) & mask) - half;
                            *x = (*x - d) >> w;
                            *o = d;
                            *n = track_sign(*n, d);
                        }
                    }
                    (Some((s, split, v)), true) => {
                        let lo_mask: i64 = (1i64 << split) - 1;
                        for (((x, o), &n), &v) in acc.iter_mut().zip(out.iter_mut()).zip(neg.iter()).zip(v) {
                            let y = *x + ((v & lo_mask) << s);
                            let d = top_limb(y, half, mask, n);
                            *x = ((y - d) >> w) + (v >> split);
                            *o = d;
                        }
                    }
                    (None, true) => {
                        for ((x, o), &n) in acc.iter_mut().zip(out.iter_mut()).zip(neg.iter()) {
                            let d = top_limb(*x, half, mask, n);
                            *x = (*x - d) >> w;
                            *o = d;
                        }
                    }
                }
                pos -= w;
            }
        }
        start += len;
    }
}

/// All ones if the lower limbs emitted so far are negative: the sign of the
/// last nonzero one, kept when `d` is zero.
#[inline(always)]
fn track_sign(neg: i64, d: i64) -> i64 {
    let nonzero: i64 = -((d != 0) as i64);
    (neg & !nonzero) | ((d >> 63) & nonzero)
}

/// Top limb of a digit: centered at its width, then moved up by `2^w` when it
/// sits at `-2^(w - 1)` over negative lower limbs, so the digit is centered.
#[inline(always)]
fn top_limb(x: i64, half: i64, mask: i64, neg: i64) -> i64 {
    let centered: i64 = ((x + half) & mask) - half;
    let at_bottom: i64 = -((centered == -half) as i64);
    centered + (at_bottom & neg & (half << 1))
}
