//! AVX2 [`GemmKernel`] implementations for the packed coefficient-matrix
//! product. `U`/`A` digits are decomposed into `W`-wide balanced pieces by the
//! shared [`matmul_gemm`] orchestration; this file provides the SIMD per-piece
//! `W x W` dot:
//! - `W = 16` (`U` bound <= 16 bits): `_mm256_madd_epi16`.
//! - `W = 32`: `_mm256_mul_epi32`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use poulpy_cpu_ref::hal_defaults::vec_znx_matmul::GemmKernel;

#[inline]
fn prep_i16(x: i64, n: usize, out: &mut [i16]) {
    let mut r: i64 = x;
    for k in 0..n {
        if k + 1 == n {
            out[k] = r as i16;
        } else {
            let hi: i64 = (r + (1i64 << 15)) >> 16;
            out[k] = (r - (hi << 16)) as i16;
            r = hi;
        }
    }
}

#[inline]
fn prep_i32(x: i64, n: usize, out: &mut [i32]) {
    let mut r: i64 = x;
    for k in 0..n {
        if k + 1 == n {
            out[k] = r as i32;
        } else {
            let hi: i64 = (r + (1i64 << 31)) >> 32;
            out[k] = (r - (hi << 32)) as i32;
            r = hi;
        }
    }
}

/// AVX2 `i16 x i16 -> i64` dot of one piece plane. `madd_epi16` yields `i32`
/// pair-sums; they are sign-widened to `i64` so accumulation cannot overflow.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn madd16(u: &[i16], a: &[i16]) -> i64 {
    unsafe {
        let n = u.len();
        let up = u.as_ptr();
        let ap = a.as_ptr();
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut i = 0;
        while i + 16 <= n {
            let uv = _mm256_loadu_si256(up.add(i) as *const __m256i);
            let av = _mm256_loadu_si256(ap.add(i) as *const __m256i);
            let m = _mm256_madd_epi16(uv, av); // 8 x i32
            acc0 = _mm256_add_epi64(acc0, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(m)));
            acc1 = _mm256_add_epi64(acc1, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(m, 1)));
            i += 16;
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

/// AVX2 `i32 x i32` dot of one piece plane, accumulating exactly in `i128`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mul32_i128(u: &[i32], a: &[i32]) -> i128 {
    unsafe {
        let n = u.len();
        let up = u.as_ptr();
        let ap = a.as_ptr();
        let mut acc: i128 = 0;
        let mut i = 0;
        let mut t = [0i64; 4];
        while i + 8 <= n {
            let uv = _mm256_loadu_si256(up.add(i) as *const __m256i);
            let av = _mm256_loadu_si256(ap.add(i) as *const __m256i);
            // even 32-bit lanes
            let ev = _mm256_mul_epi32(uv, av);
            let od = _mm256_mul_epi32(_mm256_srli_epi64(uv, 32), _mm256_srli_epi64(av, 32));
            let s = _mm256_add_epi64(ev, od);
            _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, s);
            acc += t[0] as i128 + t[1] as i128 + t[2] as i128 + t[3] as i128;
            i += 8;
        }
        while i < n {
            acc += (*up.add(i) as i128) * (*ap.add(i) as i128);
            i += 1;
        }
        acc
    }
}

/// AVX2 `i32 x i32 -> i64` dot of one piece plane (FFT64: digits stay small).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mul32_i64(u: &[i32], a: &[i32]) -> i64 {
    unsafe {
        let n = u.len();
        let up = u.as_ptr();
        let ap = a.as_ptr();
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut i = 0;
        while i + 8 <= n {
            let uv = _mm256_loadu_si256(up.add(i) as *const __m256i);
            let av = _mm256_loadu_si256(ap.add(i) as *const __m256i);
            acc0 = _mm256_add_epi64(acc0, _mm256_mul_epi32(uv, av));
            acc1 = _mm256_add_epi64(
                acc1,
                _mm256_mul_epi32(_mm256_srli_epi64(uv, 32), _mm256_srli_epi64(av, 32)),
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

/// `UP = 1` kernel body: a single U-piece, no outer `i` loop.
macro_rules! avx_kernel_up1 {
    ($name:ident, $elt:ty, $acc:ty, $w:expr, $prep:ident, $inner:ident, $widen:tt) => {
        pub struct $name;
        impl GemmKernel for $name {
            type Acc = $acc;
            type Elt = $elt;
            const W: u32 = $w;
            const UP: usize = 1;
            #[inline]
            fn prep(x: i64, n: usize, out: &mut [$elt]) {
                $prep(x, n, out)
            }
            #[inline]
            fn dot(u: &[$elt], a: &[$elt], ap: usize, rows_in: usize) -> $acc {
                let mut acc: $acc = 0;
                for j in 0..ap {
                    let p = unsafe { $inner(&u[..rows_in], &a[j * rows_in..j * rows_in + rows_in]) };
                    acc = acc.wrapping_add(avx_kernel_up1!(@widen $widen, p) << ($w as u32 * j as u32));
                }
                acc
            }
        }
    };
    (@widen i64, $p:ident) => { $p };
    (@widen i128, $p:ident) => { ($p as i128) };
    (@widen ident, $p:ident) => { $p };
}

/// `UP = 2` kernel body: two unrolled U-pieces. NTT only (i128 acc).
macro_rules! avx_kernel_up2_i128 {
    ($name:ident, $elt:ty, $w:expr, $prep:ident, $inner:ident, $widen:tt) => {
        pub struct $name;
        impl GemmKernel for $name {
            type Acc = i128;
            type Elt = $elt;
            const W: u32 = $w;
            const UP: usize = 2;
            #[inline]
            fn prep(x: i64, n: usize, out: &mut [$elt]) {
                $prep(x, n, out)
            }
            #[inline]
            fn dot(u: &[$elt], a: &[$elt], ap: usize, rows_in: usize) -> i128 {
                let u0 = &u[..rows_in];
                let u1 = &u[rows_in..2 * rows_in];
                let mut acc: i128 = 0;
                for j in 0..ap {
                    let aj = &a[j * rows_in..j * rows_in + rows_in];
                    let p0 = unsafe { $inner(u0, aj) };
                    let p1 = unsafe { $inner(u1, aj) };
                    let w0 = avx_kernel_up2_i128!(@widen $widen, p0);
                    let w1 = avx_kernel_up2_i128!(@widen $widen, p1);
                    acc += w0 << ($w as u32 * j as u32);
                    acc += w1 << ($w as u32 * (1 + j) as u32);
                }
                acc
            }
        }
    };
    (@widen i64, $p:ident) => { ($p as i128) };
    (@widen i128, $p:ident) => { $p };
}

avx_kernel_up1!(AvxK16I64, i16, i64, 16, prep_i16, madd16, i64);
avx_kernel_up1!(AvxK16I128, i16, i128, 16, prep_i16, madd16, i128);
avx_kernel_up1!(AvxK32I64, i32, i64, 32, prep_i32, mul32_i64, i64);
avx_kernel_up1!(AvxK32I128S, i32, i128, 32, prep_i32, mul32_i128, i128);
avx_kernel_up2_i128!(AvxK32I128D, i32, 32, prep_i32, mul32_i128, i128);
