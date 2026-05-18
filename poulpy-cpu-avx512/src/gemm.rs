//! AVX-512 dot-product kernels for the packed coefficient-matrix product.
//!
//! Mirrors `poulpy_cpu_avx::gemm` with 512-bit registers. These are the
//! backend `dot` passed to
//! [`poulpy_cpu_ref::hal_defaults::vec_znx_matmul::matmul_gemm`]:
//! - `gemm_dot_i32`: FFT64 backends (`i64` accumulator).
//! - `gemm_dot_split`: NTT backends (`i128` accumulator), `i64` digits split
//!   into two signed `i32` halves (`v = hi*2^s + lo`).

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// FFT64 kernel: `sum_i u[i] * a[i]` with `i32` inputs, `i64` accumulator.
#[inline]
pub fn gemm_dot_i32(u: &[i32], a: &[i32], _s: u32) -> i64 {
    // SAFETY: this crate is only built with AVX-512F enabled (see build check).
    unsafe { dot_i32_avx512(u, a) }
}

/// NTT kernel: `sum_i u[i] * a[i]` with split `[lo, hi]` `i32` digits.
#[inline]
pub fn gemm_dot_split(u: &[[i32; 2]], a: &[[i32; 2]], s: u32) -> i128 {
    // SAFETY: `[i32; 2]` is contiguous; reinterpret as interleaved `i32`.
    let uf: &[i32] = unsafe { core::slice::from_raw_parts(u.as_ptr() as *const i32, u.len() * 2) };
    let af: &[i32] = unsafe { core::slice::from_raw_parts(a.as_ptr() as *const i32, a.len() * 2) };
    // SAFETY: AVX-512F guaranteed by the crate build check.
    unsafe { dot_split_avx512(uf, af, s) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_i32_avx512(u: &[i32], a: &[i32]) -> i64 {
    unsafe {
        let n = u.len();
        let up = u.as_ptr();
        let ap = a.as_ptr();
        let mut acc0 = _mm512_setzero_si512();
        let mut acc1 = _mm512_setzero_si512();

        #[inline(always)]
        unsafe fn madd16(uv: __m512i, av: __m512i, acc: &mut __m512i) {
            unsafe {
                let ev = _mm512_mul_epi32(uv, av);
                let od = _mm512_mul_epi32(_mm512_srli_epi64(uv, 32), _mm512_srli_epi64(av, 32));
                *acc = _mm512_add_epi64(*acc, _mm512_add_epi64(ev, od));
            }
        }

        let mut i = 0;
        while i + 32 <= n {
            madd16(
                _mm512_loadu_si512(up.add(i) as *const __m512i),
                _mm512_loadu_si512(ap.add(i) as *const __m512i),
                &mut acc0,
            );
            madd16(
                _mm512_loadu_si512(up.add(i + 16) as *const __m512i),
                _mm512_loadu_si512(ap.add(i + 16) as *const __m512i),
                &mut acc1,
            );
            i += 32;
        }
        while i + 16 <= n {
            madd16(
                _mm512_loadu_si512(up.add(i) as *const __m512i),
                _mm512_loadu_si512(ap.add(i) as *const __m512i),
                &mut acc0,
            );
            i += 16;
        }

        let mut s = _mm512_reduce_add_epi64(_mm512_add_epi64(acc0, acc1));
        while i < n {
            s = s.wrapping_add((*up.add(i) as i64).wrapping_mul(*ap.add(i) as i64));
            i += 1;
        }
        s
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_split_avx512(uf: &[i32], af: &[i32], s: u32) -> i128 {
    unsafe {
        const CHUNK: usize = 2048;
        let n = uf.len() / 2;
        let up = uf.as_ptr();
        let ap = af.as_ptr();
        let mut acc0: i128 = 0;
        let mut acc1: i128 = 0;
        let mut acc2: i128 = 0;

        let mut done = 0;
        while done < n {
            let take = CHUNK.min(n - done);
            let mut l0 = _mm512_setzero_si512();
            let mut l1 = _mm512_setzero_si512();
            let mut l2 = _mm512_setzero_si512();

            let mut i = 0;
            // 8 digits per iteration (16 i32 = one __m512i of 8x[lo,hi] lanes).
            while i + 8 <= take {
                let off = (done + i) * 2;
                let uv = _mm512_loadu_si512(up.add(off) as *const __m512i);
                let av = _mm512_loadu_si512(ap.add(off) as *const __m512i);
                let uh = _mm512_srli_epi64(uv, 32);
                let ah = _mm512_srli_epi64(av, 32);

                l0 = _mm512_add_epi64(l0, _mm512_mul_epi32(uv, av));
                l1 = _mm512_add_epi64(
                    l1,
                    _mm512_add_epi64(_mm512_mul_epi32(uv, ah), _mm512_mul_epi32(uh, av)),
                );
                l2 = _mm512_add_epi64(l2, _mm512_mul_epi32(uh, ah));
                i += 8;
            }

            acc0 += _mm512_reduce_add_epi64(l0) as i128;
            acc1 += _mm512_reduce_add_epi64(l1) as i128;
            acc2 += _mm512_reduce_add_epi64(l2) as i128;

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
