//! Q120 forward / inverse NTT — NEON-accelerated kernels.
//!
//! Direct port of `poulpy-cpu-avx/src/ntt120/ntt.rs`. Each q120b coefficient
//! is two NEON registers (`Q120 { lo, hi }`); the AVX `__m256i` per-element
//! pointer arithmetic becomes `*const u64` advancing by 4 per coefficient.
//!
//! Algorithm correctness follows the AVX backend line-for-line — see
//! `poulpy-cpu-avx/src/ntt120/ntt.rs`. Variable right shifts are done via
//! `vshlq_u64(value, vdupq_n_s64(-count))` (NEON's variable-shift intrinsic
//! treats negative counts as right shifts).
//!
//! **Status**: ports the full forward / inverse NTT but is untested on
//! aarch64 hardware. Verification path: `cargo test -p poulpy-cpu-arm
//! --features enable-neon --target aarch64-unknown-linux-musl` once
//! `qemu-aarch64-static` is installed (see `.cargo/config.toml`).

use core::arch::aarch64::{int64x2_t, vdupq_n_s64, vshlq_u64};
use poulpy_cpu_ref::reference::ntt120::{
    ntt::{NttReducMeta, NttStepMeta, NttTable, NttTableInv},
    primes::PrimeSet,
};

use super::q120::{Q120, add_q120, and_q120, load_const, load_q120, mul_epu32_q120, shr_q120, store_q120, sub_q120};

const CHANGE_MODE_N: usize = 1024;

// ─────────────────────────────────────────────────────────────────────────────
// Inline NEON arithmetic helpers
// ─────────────────────────────────────────────────────────────────────────────

/// `(inp & mask) * (po & 0xFFFFFFFF) + (inp >> h) * (po >> 32)` per lane.
/// Mirrors `split_precompmul_si256` at `ntt.rs:81`.
#[inline(always)]
unsafe fn split_precompmul_q120(inp: Q120, po: Q120, h: int64x2_t, mask: Q120) -> Q120 {
    unsafe {
        let inp_low = and_q120(inp, mask);
        let t1 = mul_epu32_q120(inp_low, po);
        let inp_high = Q120 {
            lo: vshlq_u64(inp.lo, h),
            hi: vshlq_u64(inp.hi, h),
        };
        let po_high = shr_q120::<32>(po);
        let t2 = mul_epu32_q120(inp_high, po_high);
        add_q120(t1, t2)
    }
}

/// `(x & mask) + (x >> h) * cst` per lane.
/// Mirrors `modq_red_si256` at `ntt.rs:99`.
#[inline(always)]
unsafe fn modq_red_q120(x: Q120, h: int64x2_t, mask: Q120, cst: Q120) -> Q120 {
    unsafe {
        let xh = Q120 {
            lo: vshlq_u64(x.lo, h),
            hi: vshlq_u64(x.hi, h),
        };
        let xl = and_q120(x, mask);
        let xh_scaled = mul_epu32_q120(xh, cst);
        add_q120(xl, xh_scaled)
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

// ─────────────────────────────────────────────────────────────────────────────
// NTT iteration kernels (private)
// ─────────────────────────────────────────────────────────────────────────────

/// Level-0 forward pass: `a[i] *= ω^i`. Mirrors `ntt_iter_first` at `ntt.rs:118`.
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

/// Level-0 forward pass with prior lazy reduce. Mirrors `ntt_iter_first_red` at `ntt.rs:142`.
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

/// Forward Cooley–Tukey butterfly level (no reduce). Mirrors `ntt_iter` at `ntt.rs:174`.
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
            for _ in 1..halfnn {
                let a = load_q120(p1);
                let b = load_q120(p2);
                store_q120(p1, add_q120(a, b));
                let b1 = sub_q120(add_q120(a, q2bs), b);
                let p = load_q120(po);
                store_q120(p2, split_precompmul_q120(b1, p, h, mask));
                p1 = p1.add(4);
                p2 = p2.add(4);
                po = po.add(4);
            }
            data = data.add(4 * nn);
        }
    }
}

/// Forward butterfly level with prior lazy reduce. Mirrors `ntt_iter_red` at `ntt.rs:219`.
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
            for _ in 1..halfnn {
                let a = modq_red_q120(load_q120(p1), rh, rmask, rcst);
                let b = modq_red_q120(load_q120(p2), rh, rmask, rcst);
                store_q120(p1, add_q120(a, b));
                let b1 = sub_q120(add_q120(a, q2bs), b);
                let p = load_q120(po);
                store_q120(p2, split_precompmul_q120(b1, p, h, mask));
                p1 = p1.add(4);
                p2 = p2.add(4);
                po = po.add(4);
            }
            data = data.add(4 * nn);
        }
    }
}

/// Inverse Gentleman–Sande butterfly level (no reduce). Mirrors `intt_iter` at `ntt.rs:276`.
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
            for _ in 1..halfnn {
                let a = load_q120(p1);
                let b = load_q120(p2);
                let p = load_q120(po);
                let bo = split_precompmul_q120(b, p, h, mask);
                store_q120(p1, add_q120(a, bo));
                store_q120(p2, sub_q120(add_q120(a, q2bs), bo));
                p1 = p1.add(4);
                p2 = p2.add(4);
                po = po.add(4);
            }
            data = data.add(4 * nn);
        }
    }
}

/// Inverse butterfly level with prior lazy reduce. Mirrors `intt_iter_red` at `ntt.rs:321`.
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
            for _ in 1..halfnn {
                let a = modq_red_q120(load_q120(p1), rh, rmask, rcst);
                let b = modq_red_q120(load_q120(p2), rh, rmask, rcst);
                let p = load_q120(po);
                let bo = split_precompmul_q120(b, p, h, mask);
                store_q120(p1, add_q120(a, bo));
                store_q120(p2, sub_q120(add_q120(a, q2bs), bo));
                p1 = p1.add(4);
                p2 = p2.add(4);
                po = po.add(4);
            }
            data = data.add(4 * nn);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry points
// ─────────────────────────────────────────────────────────────────────────────

/// Forward Q120 NTT — NEON. Mirrors `ntt_avx2` at `ntt.rs:389`.
pub(crate) fn ntt_neon<P: PrimeSet>(table: &NttTable<P>, data: &mut [u64]) {
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

/// Inverse Q120 NTT — NEON. Mirrors `intt_avx2` at `ntt.rs:478`.
pub(crate) fn intt_neon<P: PrimeSet>(table: &NttTableInv<P>, data: &mut [u64]) {
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
    use poulpy_cpu_ref::reference::ntt120::{
        arithmetic::{b_from_znx64_ref, b_to_znx128_ref},
        ntt::{NttTable, NttTableInv, ntt_ref},
        primes::Primes30,
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
