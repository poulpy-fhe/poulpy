//! Q120 forward / inverse NTT — NEON-accelerated kernels.

use core::arch::aarch64::{int64x2_t, vdupq_n_s64, vmlal_u32, vmovn_u64, vmull_u32, vshlq_u64, vshrn_n_u64};
use poulpy_cpu_ref::reference::ntt4x30::{
    ntt::{NttReducMeta, NttStepMeta, NttTable, NttTableInv},
    primes::PrimeSetCrt4,
};

use super::q120::{Q120, add_q120, and_q120, load_const, load_q120, mla_epu32_q120, store_q120, sub_q120};

const CHANGE_MODE_N: usize = 1024;

/// `(inp & mask) * (po & 0xFFFFFFFF) + (inp >> h) * (po >> 32)` per lane.
/// Uses `vshrn_n_u64::<32>`
/// to fold `po >> 32` + truncate-to-u32 into one instruction per half.
#[inline(always)]
unsafe fn split_precompmul_q120(inp: Q120, po: Q120, h: int64x2_t, mask: Q120) -> Q120 {
    unsafe {
        let inp_low = and_q120(inp, mask);
        let inp_lo32_lo = vmovn_u64(inp_low.lo);
        let inp_lo32_hi = vmovn_u64(inp_low.hi);
        let po_lo32_lo = vmovn_u64(po.lo);
        let po_lo32_hi = vmovn_u64(po.hi);
        let t1_lo = vmull_u32(inp_lo32_lo, po_lo32_lo);
        let t1_hi = vmull_u32(inp_lo32_hi, po_lo32_hi);

        let inp_hi_lo32 = vmovn_u64(vshlq_u64(inp.lo, h));
        let inp_hi_hi32 = vmovn_u64(vshlq_u64(inp.hi, h));
        let po_hi32_lo = vshrn_n_u64::<32>(po.lo);
        let po_hi32_hi = vshrn_n_u64::<32>(po.hi);
        Q120 {
            lo: vmlal_u32(t1_lo, inp_hi_lo32, po_hi32_lo),
            hi: vmlal_u32(t1_hi, inp_hi_hi32, po_hi32_hi),
        }
    }
}

/// `(x & mask) + (x >> h) * cst` per lane.
#[inline(always)]
unsafe fn modq_red_q120(x: Q120, h: int64x2_t, mask: Q120, cst: Q120) -> Q120 {
    unsafe {
        let xh = Q120 {
            lo: vshlq_u64(x.lo, h),
            hi: vshlq_u64(x.hi, h),
        };
        let xl = and_q120(x, mask);
        mla_epu32_q120(xl, xh, cst)
    }
}

#[inline(always)]
unsafe fn broadcast_mask(v: u64) -> Q120 {
    use core::arch::aarch64::vdupq_n_u64;
    unsafe {
        let m = vdupq_n_u64(v);
        Q120 { lo: m, hi: m }
    }
}

/// Level-0 forward pass: `a[i] *= ω^i`.
#[inline(always)]
unsafe fn ntt_iter_first(begin: *mut u64, end: *const u64, meta: &NttStepMeta, mut po: *const u64) {
    unsafe {
        let h = vdupq_n_s64(-(meta.half_bs as i64));
        let mask = broadcast_mask(meta.mask);
        let mut data = begin;
        while (data as usize) < (end as usize) {
            let x = load_q120(data);
            let p = load_q120(po);
            store_q120(data, split_precompmul_q120(x, p, h, mask));
            data = data.add(4);
            po = po.add(4);
        }
    }
}

/// Level-0 forward pass with prior lazy reduce.
#[inline(always)]
unsafe fn ntt_iter_first_red(begin: *mut u64, end: *const u64, meta: &NttStepMeta, mut po: *const u64, reduc: &NttReducMeta) {
    unsafe {
        let h = vdupq_n_s64(-(meta.half_bs as i64));
        let mask = broadcast_mask(meta.mask);
        let rh = vdupq_n_s64(-(reduc.h as i64));
        let rmask = broadcast_mask(reduc.mask);
        let rcst = load_const(&reduc.modulo_red_cst);
        let mut data = begin;
        while (data as usize) < (end as usize) {
            let x = modq_red_q120(load_q120(data), rh, rmask, rcst);
            let p = load_q120(po);
            store_q120(data, split_precompmul_q120(x, p, h, mask));
            data = data.add(4);
            po = po.add(4);
        }
    }
}

/// Forward Cooley–Tukey butterfly level (no reduce).
/// Inner loop is 2× unrolled.
#[inline(always)]
unsafe fn ntt_iter(nn: usize, begin: *mut u64, end: *const u64, meta: &NttStepMeta, po_base: *const u64) {
    unsafe {
        let halfnn = nn / 2;
        let q2bs = load_const(&meta.q2bs);
        let mask = broadcast_mask(meta.mask);
        let h = vdupq_n_s64(-(meta.half_bs as i64));

        let mut data = begin;
        while (data as usize) < (end as usize) {
            let mut p1 = data;
            let mut p2 = data.add(4 * halfnn);

            // i = 0
            let a = load_q120(p1);
            let b = load_q120(p2);
            store_q120(p1, add_q120(a, b));
            store_q120(p2, sub_q120(add_q120(a, q2bs), b));
            p1 = p1.add(4);
            p2 = p2.add(4);

            let mut po = po_base;
            let mut remaining = halfnn - 1;
            while remaining >= 2 {
                let a0 = load_q120(p1);
                let b0 = load_q120(p2);
                let a1 = load_q120(p1.add(4));
                let b1 = load_q120(p2.add(4));
                let pw0 = load_q120(po);
                let pw1 = load_q120(po.add(4));

                store_q120(p1, add_q120(a0, b0));
                store_q120(p1.add(4), add_q120(a1, b1));
                let d0 = sub_q120(add_q120(a0, q2bs), b0);
                let d1 = sub_q120(add_q120(a1, q2bs), b1);
                store_q120(p2, split_precompmul_q120(d0, pw0, h, mask));
                store_q120(p2.add(4), split_precompmul_q120(d1, pw1, h, mask));

                p1 = p1.add(8);
                p2 = p2.add(8);
                po = po.add(8);
                remaining -= 2;
            }
            if remaining == 1 {
                let a = load_q120(p1);
                let b = load_q120(p2);
                store_q120(p1, add_q120(a, b));
                let b1 = sub_q120(add_q120(a, q2bs), b);
                let p = load_q120(po);
                store_q120(p2, split_precompmul_q120(b1, p, h, mask));
            }
            data = data.add(4 * nn);
        }
    }
}

/// Forward butterfly level with prior lazy reduce.
/// Inner loop is 2× unrolled.
#[inline(always)]
unsafe fn ntt_iter_red(
    nn: usize,
    begin: *mut u64,
    end: *const u64,
    meta: &NttStepMeta,
    po_base: *const u64,
    reduc: &NttReducMeta,
) {
    unsafe {
        let halfnn = nn / 2;
        let q2bs = load_const(&meta.q2bs);
        let mask = broadcast_mask(meta.mask);
        let h = vdupq_n_s64(-(meta.half_bs as i64));
        let rh = vdupq_n_s64(-(reduc.h as i64));
        let rmask = broadcast_mask(reduc.mask);
        let rcst = load_const(&reduc.modulo_red_cst);

        let mut data = begin;
        while (data as usize) < (end as usize) {
            let mut p1 = data;
            let mut p2 = data.add(4 * halfnn);

            // i = 0
            let a = modq_red_q120(load_q120(p1), rh, rmask, rcst);
            let b = modq_red_q120(load_q120(p2), rh, rmask, rcst);
            store_q120(p1, add_q120(a, b));
            store_q120(p2, sub_q120(add_q120(a, q2bs), b));
            p1 = p1.add(4);
            p2 = p2.add(4);

            let mut po = po_base;
            let mut remaining = halfnn - 1;
            while remaining >= 2 {
                let a0 = modq_red_q120(load_q120(p1), rh, rmask, rcst);
                let b0 = modq_red_q120(load_q120(p2), rh, rmask, rcst);
                let a1 = modq_red_q120(load_q120(p1.add(4)), rh, rmask, rcst);
                let b1 = modq_red_q120(load_q120(p2.add(4)), rh, rmask, rcst);
                let pw0 = load_q120(po);
                let pw1 = load_q120(po.add(4));

                store_q120(p1, add_q120(a0, b0));
                store_q120(p1.add(4), add_q120(a1, b1));
                let d0 = sub_q120(add_q120(a0, q2bs), b0);
                let d1 = sub_q120(add_q120(a1, q2bs), b1);
                store_q120(p2, split_precompmul_q120(d0, pw0, h, mask));
                store_q120(p2.add(4), split_precompmul_q120(d1, pw1, h, mask));

                p1 = p1.add(8);
                p2 = p2.add(8);
                po = po.add(8);
                remaining -= 2;
            }
            if remaining == 1 {
                let a = modq_red_q120(load_q120(p1), rh, rmask, rcst);
                let b = modq_red_q120(load_q120(p2), rh, rmask, rcst);
                store_q120(p1, add_q120(a, b));
                let b1 = sub_q120(add_q120(a, q2bs), b);
                let p = load_q120(po);
                store_q120(p2, split_precompmul_q120(b1, p, h, mask));
            }
            data = data.add(4 * nn);
        }
    }
}

/// Inverse Gentleman–Sande butterfly level (no reduce).
/// Inner loop is 2× unrolled.
#[inline(always)]
unsafe fn intt_iter(nn: usize, begin: *mut u64, end: *const u64, meta: &NttStepMeta, po_base: *const u64) {
    unsafe {
        let halfnn = nn / 2;
        let q2bs = load_const(&meta.q2bs);
        let mask = broadcast_mask(meta.mask);
        let h = vdupq_n_s64(-(meta.half_bs as i64));

        let mut data = begin;
        while (data as usize) < (end as usize) {
            let mut p1 = data;
            let mut p2 = data.add(4 * halfnn);

            // i = 0
            let a = load_q120(p1);
            let b = load_q120(p2);
            store_q120(p1, add_q120(a, b));
            store_q120(p2, sub_q120(add_q120(a, q2bs), b));
            p1 = p1.add(4);
            p2 = p2.add(4);

            let mut po = po_base;
            let mut remaining = halfnn - 1;
            while remaining >= 2 {
                let a0 = load_q120(p1);
                let b0 = load_q120(p2);
                let a1 = load_q120(p1.add(4));
                let b1 = load_q120(p2.add(4));
                let pw0 = load_q120(po);
                let pw1 = load_q120(po.add(4));

                let bo0 = split_precompmul_q120(b0, pw0, h, mask);
                let bo1 = split_precompmul_q120(b1, pw1, h, mask);
                store_q120(p1, add_q120(a0, bo0));
                store_q120(p1.add(4), add_q120(a1, bo1));
                store_q120(p2, sub_q120(add_q120(a0, q2bs), bo0));
                store_q120(p2.add(4), sub_q120(add_q120(a1, q2bs), bo1));

                p1 = p1.add(8);
                p2 = p2.add(8);
                po = po.add(8);
                remaining -= 2;
            }
            if remaining == 1 {
                let a = load_q120(p1);
                let b = load_q120(p2);
                let p = load_q120(po);
                let bo = split_precompmul_q120(b, p, h, mask);
                store_q120(p1, add_q120(a, bo));
                store_q120(p2, sub_q120(add_q120(a, q2bs), bo));
            }
            data = data.add(4 * nn);
        }
    }
}

/// Inverse butterfly level with prior lazy reduce.
/// Inner loop is 2× unrolled.
#[inline(always)]
unsafe fn intt_iter_red(
    nn: usize,
    begin: *mut u64,
    end: *const u64,
    meta: &NttStepMeta,
    po_base: *const u64,
    reduc: &NttReducMeta,
) {
    unsafe {
        let halfnn = nn / 2;
        let q2bs = load_const(&meta.q2bs);
        let mask = broadcast_mask(meta.mask);
        let h = vdupq_n_s64(-(meta.half_bs as i64));
        let rh = vdupq_n_s64(-(reduc.h as i64));
        let rmask = broadcast_mask(reduc.mask);
        let rcst = load_const(&reduc.modulo_red_cst);

        let mut data = begin;
        while (data as usize) < (end as usize) {
            let mut p1 = data;
            let mut p2 = data.add(4 * halfnn);

            // i = 0
            let a = modq_red_q120(load_q120(p1), rh, rmask, rcst);
            let b = modq_red_q120(load_q120(p2), rh, rmask, rcst);
            store_q120(p1, add_q120(a, b));
            store_q120(p2, sub_q120(add_q120(a, q2bs), b));
            p1 = p1.add(4);
            p2 = p2.add(4);

            let mut po = po_base;
            let mut remaining = halfnn - 1;
            while remaining >= 2 {
                let a0 = modq_red_q120(load_q120(p1), rh, rmask, rcst);
                let b0 = modq_red_q120(load_q120(p2), rh, rmask, rcst);
                let a1 = modq_red_q120(load_q120(p1.add(4)), rh, rmask, rcst);
                let b1 = modq_red_q120(load_q120(p2.add(4)), rh, rmask, rcst);
                let pw0 = load_q120(po);
                let pw1 = load_q120(po.add(4));

                let bo0 = split_precompmul_q120(b0, pw0, h, mask);
                let bo1 = split_precompmul_q120(b1, pw1, h, mask);
                store_q120(p1, add_q120(a0, bo0));
                store_q120(p1.add(4), add_q120(a1, bo1));
                store_q120(p2, sub_q120(add_q120(a0, q2bs), bo0));
                store_q120(p2.add(4), sub_q120(add_q120(a1, q2bs), bo1));

                p1 = p1.add(8);
                p2 = p2.add(8);
                po = po.add(8);
                remaining -= 2;
            }
            if remaining == 1 {
                let a = modq_red_q120(load_q120(p1), rh, rmask, rcst);
                let b = modq_red_q120(load_q120(p2), rh, rmask, rcst);
                let p = load_q120(po);
                let bo = split_precompmul_q120(b, p, h, mask);
                store_q120(p1, add_q120(a, bo));
                store_q120(p2, sub_q120(add_q120(a, q2bs), bo));
            }
            data = data.add(4 * nn);
        }
    }
}

/// Forward Q120 NTT — NEON.
pub(crate) fn ntt_neon<P: PrimeSetCrt4>(table: &NttTable<P>, data: &mut [u64]) {
    let n = table.n;
    if n == 1 {
        return;
    }
    debug_assert!(data.len() >= 4 * n);
    unsafe {
        let begin = data.as_mut_ptr();
        let end = begin.add(4 * n) as *const u64;
        let po_base = table.powomega.as_ptr();

        let mut meta_idx = 0usize;
        // po_off: current offset into powomega in u64 units (4 u64 per coefficient).
        let mut po_off = 0usize;

        // Level 0
        ntt_iter_first(begin, end, &table.level_metadata[meta_idx], po_base.add(po_off));
        po_off += 4 * n;
        meta_idx += 1;

        let split_nn = CHANGE_MODE_N.min(n);

        // By-level phase
        let mut nn = n;
        while nn > split_nn {
            let halfnn = nn / 2;
            let meta = &table.level_metadata[meta_idx];
            if meta.reduce {
                ntt_iter_red(nn, begin, end, meta, po_base.add(po_off), &table.reduc_metadata);
            } else {
                ntt_iter(nn, begin, end, meta, po_base.add(po_off));
            }
            po_off += 4 * halfnn.saturating_sub(1);
            meta_idx += 1;
            nn /= 2;
        }

        // By-block phase
        if split_nn >= 2 {
            let meta_idx_saved = meta_idx;
            let po_off_saved = po_off;
            let mut it = begin;
            while (it as usize) < (end as usize) {
                let begin1 = it;
                let end1 = it.add(4 * split_nn) as *const u64;
                meta_idx = meta_idx_saved;
                po_off = po_off_saved;
                let mut nn = split_nn;
                while nn >= 2 {
                    let halfnn = nn / 2;
                    let meta = &table.level_metadata[meta_idx];
                    if meta.reduce {
                        ntt_iter_red(nn, begin1, end1, meta, po_base.add(po_off), &table.reduc_metadata);
                    } else {
                        ntt_iter(nn, begin1, end1, meta, po_base.add(po_off));
                    }
                    po_off += 4 * halfnn.saturating_sub(1);
                    meta_idx += 1;
                    nn /= 2;
                }
                it = it.add(4 * split_nn);
            }
        }
    }
}

/// Inverse Q120 NTT — NEON.
pub(crate) fn intt_neon<P: PrimeSetCrt4>(table: &NttTableInv<P>, data: &mut [u64]) {
    let n = table.n;
    if n == 1 {
        return;
    }
    debug_assert!(data.len() >= 4 * n);
    unsafe {
        let begin = data.as_mut_ptr();
        let end = begin.add(4 * n) as *const u64;
        let po_base = table.powomega.as_ptr();

        let mut meta_idx = 0usize;
        let mut po_off = 0usize;

        let split_nn = CHANGE_MODE_N.min(n);

        // By-block phase
        if split_nn >= 2 {
            let meta_idx_saved = meta_idx;
            let po_off_saved = po_off;
            let mut it = begin;
            while (it as usize) < (end as usize) {
                let begin1 = it;
                let end1 = it.add(4 * split_nn) as *const u64;
                meta_idx = meta_idx_saved;
                po_off = po_off_saved;
                let mut nn = 2usize;
                while nn <= split_nn {
                    let halfnn = nn / 2;
                    let meta = &table.level_metadata[meta_idx];
                    if meta.reduce {
                        intt_iter_red(nn, begin1, end1, meta, po_base.add(po_off), &table.reduc_metadata);
                    } else {
                        intt_iter(nn, begin1, end1, meta, po_base.add(po_off));
                    }
                    po_off += 4 * halfnn.saturating_sub(1);
                    meta_idx += 1;
                    nn *= 2;
                }
                it = it.add(4 * split_nn);
            }
        }

        // By-level phase
        let mut nn = 2 * split_nn;
        while nn <= n {
            let halfnn = nn / 2;
            let meta = &table.level_metadata[meta_idx];
            if meta.reduce {
                intt_iter_red(nn, begin, end, meta, po_base.add(po_off), &table.reduc_metadata);
            } else {
                intt_iter(nn, begin, end, meta, po_base.add(po_off));
            }
            po_off += 4 * halfnn.saturating_sub(1);
            meta_idx += 1;
            nn *= 2;
        }

        // Last pass: a[i] *= ω^{-i} * n^{-1}
        let meta = &table.level_metadata[meta_idx];
        if meta.reduce {
            ntt_iter_first_red(begin, end, meta, po_base.add(po_off), &table.reduc_metadata);
        } else {
            ntt_iter_first(begin, end, meta, po_base.add(po_off));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use poulpy_cpu_ref::reference::ntt4x30::{
        arithmetic::{b_from_znx64_ref, b_to_znx128_ref},
        ntt::{NttTable, NttTableInv, ntt_ref},
        primes::{PrimeSet, Primes30},
    };

    /// NEON NTT then NEON iNTT round-trips to the original (mod each Q[k]).
    #[test]
    fn ntt_intt_identity_neon() {
        for log_n in 1..=8usize {
            let n = 1 << log_n;
            let fwd = NttTable::<Primes30>::new(n);
            let inv = NttTableInv::<Primes30>::new(n);
            let coeffs: Vec<i64> = (0..n as i64).map(|i| (i * 7 + 3) % 201 - 100).collect();
            let mut data = vec![0u64; 4 * n];
            b_from_znx64_ref::<Primes30>(n, &mut data, &coeffs);
            let data_orig = data.clone();
            ntt_neon::<Primes30>(&fwd, &mut data);
            intt_neon::<Primes30>(&inv, &mut data);
            for i in 0..n {
                for k in 0..4 {
                    let q = Primes30::Q[k] as u64;
                    assert_eq!(
                        data_orig[4 * i + k] % q,
                        data[4 * i + k] % q,
                        "n={n} i={i} k={k}: NEON NTT/iNTT round-trip mismatch"
                    );
                }
            }
        }
    }

    #[test]
    fn ntt_neon_vs_ref() {
        for log_n in 1..=8usize {
            let n = 1 << log_n;
            let fwd = NttTable::<Primes30>::new(n);
            let coeffs: Vec<i64> = (0..n as i64).map(|i| (i * 13 + 5) % 201 - 100).collect();
            let mut data_neon = vec![0u64; 4 * n];
            let mut data_ref = vec![0u64; 4 * n];
            b_from_znx64_ref::<Primes30>(n, &mut data_neon, &coeffs);
            b_from_znx64_ref::<Primes30>(n, &mut data_ref, &coeffs);
            ntt_neon::<Primes30>(&fwd, &mut data_neon);
            ntt_ref::<Primes30>(&fwd, &mut data_ref);
            assert_eq!(data_neon, data_ref, "n={n}: NTT NEON vs ref mismatch");
        }
    }

    #[test]
    fn ntt_convolution_neon() {
        let n = 8usize;
        let fwd = NttTable::<Primes30>::new(n);
        let inv = NttTableInv::<Primes30>::new(n);
        let a: Vec<i64> = vec![1, 2, 0, 0, 0, 0, 0, 0];
        let b: Vec<i64> = vec![3, 4, 0, 0, 0, 0, 0, 0];
        let mut da = vec![0u64; 4 * n];
        let mut db = vec![0u64; 4 * n];
        b_from_znx64_ref::<Primes30>(n, &mut da, &a);
        b_from_znx64_ref::<Primes30>(n, &mut db, &b);
        ntt_neon::<Primes30>(&fwd, &mut da);
        ntt_neon::<Primes30>(&fwd, &mut db);
        let mut dc = vec![0u64; 4 * n];
        for i in 0..n {
            for k in 0..4 {
                let q = Primes30::Q[k] as u64;
                dc[4 * i + k] = (da[4 * i + k] % q * (db[4 * i + k] % q)) % q;
            }
        }
        intt_neon::<Primes30>(&inv, &mut dc);
        let mut result = vec![0i128; n];
        b_to_znx128_ref::<Primes30>(n, &mut result, &dc);
        let expected: Vec<i128> = vec![3, 10, 8, 0, 0, 0, 0, 0];
        assert_eq!(result, expected, "NEON NTT convolution mismatch");
    }
}
