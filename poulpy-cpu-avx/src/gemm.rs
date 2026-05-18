//! AVX2 dot-product kernels for the packed coefficient-matrix product.
//!
//! These are the backend-specialized `dot` passed to
//! [`poulpy_cpu_ref::hal_defaults::vec_znx_matmul::matmul_gemm`]:
//! - `gemm_dot_i32`: FFT64 backends (`i64` accumulator), native signed
//!   `32x32 -> 64` multiply (`mul_epi32`).
//! - `gemm_dot_split`: NTT backends (`i128` accumulator), `i64` digits split
//!   into two signed `i32` halves, exact `i128` reassembly.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// FFT64 kernel: `sum_i u[i] * a[i]` with `i32` inputs, `i64` accumulator.
#[inline]
pub fn gemm_dot_i32(u: &[i32], a: &[i32], _s: u32) -> i64 {
    // SAFETY: this crate is only built with AVX2 enabled (see build check).
    unsafe { dot_i32_avx2(u, a) }
}

/// NTT kernel: `sum_i u[i] * a[i]` with split `[lo, hi]` `i32` digits
/// (`v = hi*2^s + lo`), exact `i128` accumulator.
#[inline]
pub fn gemm_dot_split(u: &[[i32; 2]], a: &[[i32; 2]], s: u32) -> i128 {
    // SAFETY: `[i32; 2]` is contiguous; reinterpret as interleaved `i32`.
    let uf: &[i32] = unsafe { core::slice::from_raw_parts(u.as_ptr() as *const i32, u.len() * 2) };
    let af: &[i32] = unsafe { core::slice::from_raw_parts(a.as_ptr() as *const i32, a.len() * 2) };
    // SAFETY: AVX2 guaranteed by the crate build check.
    unsafe { dot_split_avx2(uf, af, s) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i32_avx2(u: &[i32], a: &[i32]) -> i64 {
    unsafe {
        let n = u.len();
        let up = u.as_ptr();
        let ap = a.as_ptr();
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();

        #[inline(always)]
        unsafe fn madd8(uv: __m256i, av: __m256i, acc: &mut __m256i) {
            unsafe {
                let ev = _mm256_mul_epi32(uv, av);
                let od = _mm256_mul_epi32(_mm256_srli_epi64(uv, 32), _mm256_srli_epi64(av, 32));
                *acc = _mm256_add_epi64(*acc, _mm256_add_epi64(ev, od));
            }
        }

        let mut i = 0;
        while i + 16 <= n {
            madd8(
                _mm256_loadu_si256(up.add(i) as *const __m256i),
                _mm256_loadu_si256(ap.add(i) as *const __m256i),
                &mut acc0,
            );
            madd8(
                _mm256_loadu_si256(up.add(i + 8) as *const __m256i),
                _mm256_loadu_si256(ap.add(i + 8) as *const __m256i),
                &mut acc1,
            );
            i += 16;
        }
        while i + 8 <= n {
            madd8(
                _mm256_loadu_si256(up.add(i) as *const __m256i),
                _mm256_loadu_si256(ap.add(i) as *const __m256i),
                &mut acc0,
            );
            i += 8;
        }

        let acc = _mm256_add_epi64(acc0, acc1);
        let mut t = [0i64; 4];
        _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, acc);
        let mut s = t[0].wrapping_add(t[1]).wrapping_add(t[2]).wrapping_add(t[3]);
        while i < n {
            s = s.wrapping_add((*up.add(i) as i64).wrapping_mul(*ap.add(i) as i64));
            i += 1;
        }
        s
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_split_avx2(uf: &[i32], af: &[i32], s: u32) -> i128 {
    unsafe {
        const CHUNK: usize = 2048;
        let n = uf.len() / 2;
        let up = uf.as_ptr();
        let ap = af.as_ptr();
        let mut acc0: i128 = 0;
        let mut acc1: i128 = 0;
        let mut acc2: i128 = 0;

        #[inline(always)]
        unsafe fn hsum(v: __m256i) -> i128 {
            let mut t = [0i64; 4];
            unsafe { _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, v) };
            t[0] as i128 + t[1] as i128 + t[2] as i128 + t[3] as i128
        }

        let mut done = 0;
        while done < n {
            let take = CHUNK.min(n - done);
            let mut l0 = _mm256_setzero_si256();
            let mut l1 = _mm256_setzero_si256();
            let mut l2 = _mm256_setzero_si256();

            let mut i = 0;
            while i + 4 <= take {
                let off = (done + i) * 2;
                let uv = _mm256_loadu_si256(up.add(off) as *const __m256i);
                let av = _mm256_loadu_si256(ap.add(off) as *const __m256i);
                let av_sw = _mm256_shuffle_epi32(av, 0b10_11_00_01);
                let uh = _mm256_srli_epi64(uv, 32);
                let ah = _mm256_srli_epi64(av, 32);
                let ah_sw = _mm256_srli_epi64(av_sw, 32);

                l0 = _mm256_add_epi64(l0, _mm256_mul_epi32(uv, av));
                l1 = _mm256_add_epi64(
                    l1,
                    _mm256_add_epi64(_mm256_mul_epi32(uv, av_sw), _mm256_mul_epi32(uh, ah_sw)),
                );
                l2 = _mm256_add_epi64(l2, _mm256_mul_epi32(uh, ah));
                i += 4;
            }

            acc0 += hsum(l0);
            acc1 += hsum(l1);
            acc2 += hsum(l2);

            while i < take {
                let j = (done + i) * 2;
                let ulo = *up.add(j) as i64;
                let uhi = *up.add(j + 1) as i64;
                let alo = *ap.add(j) as i64;
                let ahi = *ap.add(j + 1) as i64;
                acc0 += (ulo * alo) as i128;
                acc1 += (ulo * ahi + uhi * alo) as i128;
                acc2 += (uhi * ahi) as i128;
                i += 1;
            }
            done += take;
        }

        acc0 + (acc1 << s) + (acc2 << (2 * s))
    }
}
