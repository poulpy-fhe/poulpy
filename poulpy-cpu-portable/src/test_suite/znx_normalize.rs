//! Normalization kernel tests every backend runs against the exact integer oracle.

use dashu_int::{IBig, ops::BitTest};

use crate::kernels::znx_normalize::{Limbs, LimbsMut, NormalizeWord, ZnxNormalize};

/// `value / 2^drop` rounded to the nearest integer, ties to the even one.
pub fn round_ties_even(value: IBig, drop: usize) -> IBig {
    let floor = &value >> drop;
    let rem = &value - (&floor << drop);
    let half = IBig::ONE << (drop - 1);
    if rem > half || (rem == half && floor.bit(0)) {
        floor + IBig::ONE
    } else {
        floor
    }
}

/// `canon(a * 2^offset, kr, k, size)` on the exact integer of the limbs `a` at radix `ka`.
pub fn oracle(a: &[i128], ka: usize, kr: usize, k: usize, size: usize, offset: i64) -> Vec<i64> {
    let mut out = vec![0; size];
    let shift = k as i128 + offset as i128 - (a.len() * ka) as i128;
    // Past either bound every bit is integer or below half a unit: the result is zero.
    if k == 0 || shift >= k as i128 || -shift > (a.len() * ka) as i128 + 200 {
        return out;
    }
    let mut value = a.iter().fold(IBig::ZERO, |sum, &limb| (sum << ka) + IBig::from(limb));
    if shift >= 0 {
        value <<= shift as usize;
    } else {
        value = round_ties_even(value, (-shift) as usize);
    }
    let active = k.div_ceil(kr);
    value <<= active * kr - k;
    for digit in out[..active].iter_mut().rev() {
        let carry = (&value + (IBig::ONE << (kr - 1))) >> kr;
        *digit = i64::try_from(&value - (&carry << kr)).unwrap();
        value = carry;
    }
    out
}

/// Ties, near-ties, carries, radix pairs, offsets and in place, on both methods, over
/// 19 coefficients so every lane width leaves a tail.
pub fn test_znx_normalize<BE: ZnxNormalize>() {
    const N: usize = 19;
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state as i64
    };
    for (ka, kr) in [(3, 3), (17, 17), (50, 50), (62, 62), (17, 19), (51, 50), (62, 1), (1, 62)] {
        for a_size in [1usize, 2, 3] {
            let limbs: Vec<Vec<i64>> = (0..a_size)
                .map(|j| {
                    (0..N)
                        .map(|i| match i % 5 {
                            0 if j + 1 == a_size => 1i64 << (ka - 1),
                            1 if j + 1 == a_size => -(1i64 << (ka - 1)),
                            2 if j + 1 == a_size => (1i64 << (ka - 1)) + 1,
                            3 => (i as i64 - 9) << 58,
                            _ => next() >> 1,
                        })
                        .collect()
                })
                .collect();
            for size in [1usize, 2, 3] {
                let mut ks = vec![0, 1, kr - 1, kr, kr + 1, size * kr / 2, size * kr - 1, size * kr];
                ks.retain(|&k| k <= size * kr);
                ks.sort_unstable();
                ks.dedup();
                for k in ks {
                    for offset in [
                        -(ka as i64) - 1,
                        -(ka as i64),
                        -2,
                        -1,
                        0,
                        1,
                        ka as i64,
                        -300,
                        300,
                        i64::MIN,
                        i64::MAX,
                    ] {
                        check::<BE>(&limbs, ka, kr, k, size, offset);
                    }
                }
            }
        }
    }
    check_in_place::<BE>();
}

fn check<BE: ZnxNormalize>(a: &[Vec<i64>], ka: usize, kr: usize, k: usize, size: usize, offset: i64) {
    let n = a.first().map_or(1, Vec::len);
    let flat: Vec<i64> = a.iter().flatten().copied().collect();
    let wide: Vec<BE::BigWord> = flat.iter().map(|&x| BE::BigWord::ZERO.add_i64(x)).collect();
    let mut dst = vec![0x5a5a_i64; size * n];
    let mut dst_big = vec![0x5a5a_i64; size * n];
    unsafe {
        BE::znx_normalize(
            LimbsMut {
                ptr: dst.as_mut_ptr(),
                stride: n,
                size,
            },
            kr,
            k,
            offset,
            Limbs {
                ptr: flat.as_ptr(),
                stride: n,
                size: a.len(),
            },
            ka,
            n,
        );
        BE::znx_normalize_big(
            LimbsMut {
                ptr: dst_big.as_mut_ptr(),
                stride: n,
                size,
            },
            kr,
            k,
            offset,
            Limbs {
                ptr: wide.as_ptr(),
                stride: n,
                size: a.len(),
            },
            ka,
            n,
        );
    }
    for i in 0..n {
        let column: Vec<i128> = a.iter().map(|l| l[i] as i128).collect();
        let want = oracle(&column, ka, kr, k, size, offset);
        let have: Vec<i64> = (0..size).map(|j| dst[j * n + i]).collect();
        let have_big: Vec<i64> = (0..size).map(|j| dst_big[j * n + i]).collect();
        assert_eq!(
            have, want,
            "znx_normalize a={column:?} ka={ka} kr={kr} k={k} size={size} offset={offset}"
        );
        assert_eq!(
            have_big, want,
            "znx_normalize_big a={column:?} ka={ka} kr={kr} k={k} size={size} offset={offset}"
        );
    }
}

fn check_in_place<BE: ZnxNormalize>() {
    const N: usize = 19;
    for (b, offset) in [(17usize, 0i64), (17, 5), (17, -40), (50, -1), (62, -62)] {
        let size = 3;
        let mut buf: Vec<i64> = (0..size * N)
            .map(|x| (x as i64).wrapping_mul(0x31e5_cd28_3792_b601) >> 3)
            .collect();
        let a: Vec<Vec<i128>> = (0..size).map(|j| (0..N).map(|i| buf[j * N + i] as i128).collect()).collect();
        let ptr = buf.as_mut_ptr();
        unsafe {
            BE::znx_normalize(
                LimbsMut { ptr, stride: N, size },
                b,
                size * b,
                offset,
                Limbs {
                    ptr: ptr.cast_const(),
                    stride: N,
                    size,
                },
                b,
                N,
            )
        };
        for i in 0..N {
            let column: Vec<i128> = a.iter().map(|l| l[i]).collect();
            let have: Vec<i64> = (0..size).map(|j| buf[j * N + i]).collect();
            assert_eq!(
                have,
                oracle(&column, b, b, size * b, size, offset),
                "in place b={b} offset={offset} i={i}"
            );
        }
    }
}
