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

/// Ties, near-ties, carries, radix pairs, offsets, wide big words and in place, on both methods,
/// over 19 coefficients so every lane width leaves a tail.
pub fn test_znx_normalize<BE: ZnxNormalize>() {
    const N: usize = 19;
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state as i64
    };
    for (ka, kr) in [
        (1, 1),
        (3, 3),
        (17, 17),
        (50, 50),
        (62, 62),
        (17, 19),
        (19, 17),
        (51, 50),
        (62, 1),
        (1, 62),
    ] {
        for a_size in [1usize, 2, 3] {
            let limbs: Vec<Vec<i128>> = (0..a_size)
                .map(|j| {
                    (0..N)
                        .map(|i| match i % 7 {
                            0 if j + 1 == a_size => 1i64 << (ka - 1),
                            1 if j + 1 == a_size => -(1i64 << (ka - 1)),
                            2 if j + 1 == a_size => (1i64 << (ka - 1)) + 1,
                            3 => (i as i64 - 9) << 58,
                            // The input bound on every limb: the largest carries the kernel must hold.
                            4 => 1i64 << 62,
                            5 => -(1i64 << 62),
                            _ => next() >> 1,
                        } as i128)
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
    check_ties::<BE>();
    check_wide::<BE>();
    check_in_place::<BE>();
    check_many_limbs::<BE>();
}

/// Below, at and above the tie `q + 1/2` for every dropped width from one bit to past one limb.
fn check_ties<BE: ZnxNormalize>() {
    const N: i128 = 19;
    for b in [3usize, 4, 17, 50, 62] {
        for a_size in 1..=3usize {
            for drop in 1..(a_size * b).min(b + 4) {
                let k = a_size * b - drop;
                for q in [-3i128, -2, -1, 0, 1, 2, 3, 6, 7] {
                    let values: Vec<i128> = (0..N).map(|t| (q << drop) + (1i128 << (drop - 1)) + t % 3 - 1).collect();
                    let limbs: Vec<Vec<i128>> = (0..a_size)
                        .map(|j| {
                            let shift = (a_size - 1 - j) * b;
                            let mask = if j == 0 { -1 } else { (1i128 << b) - 1 };
                            values.iter().map(|&v| (v >> shift) & mask).collect()
                        })
                        .collect();
                    for kr in [b, (b + 2).min(63), b.max(2) - 1] {
                        for size in [k.div_ceil(kr), k.div_ceil(kr) + 1] {
                            check::<BE>(&limbs, b, kr, k, size, 0);
                        }
                    }
                }
            }
        }
    }
}

/// Non-canonical big words up to `2^126`, which only an i128 big word holds.
fn check_wide<BE: ZnxNormalize>() {
    const N: i128 = 19;
    let limbs: Vec<Vec<i128>> = vec![
        (0..N)
            .map(|t| match t % 3 {
                0 => 1i128 << 126,
                1 => -(1i128 << 126),
                _ => (1i128 << 120) - 7 * t,
            })
            .collect(),
        (0..N).map(|t| -(1i128 << 100) + t).collect(),
    ];
    for (ka, kr) in [(50, 50), (62, 62), (52, 50), (17, 19), (62, 1)] {
        for k in [1, kr, 2 * kr - 1, 2 * kr] {
            for offset in [-(ka as i64) - 1, -(ka as i64), -1, 0, 3, ka as i64] {
                check::<BE>(&limbs, ka, kr, k, 2, offset);
            }
        }
    }
}

/// Runs both methods on every source the backend words hold: `|a| <= 2^62` on i64 words,
/// `|a| <= 2^126` on i128 words.
fn check<BE: ZnxNormalize>(a: &[Vec<i128>], ka: usize, kr: usize, k: usize, size: usize, offset: i64) {
    let n = a.first().map_or(1, Vec::len);
    let flat: Vec<i128> = a.iter().flatten().copied().collect();
    let narrow = flat.iter().all(|x| x.unsigned_abs() <= 1 << 62);
    let column = |i: usize| a.iter().map(|l| l[i]).collect::<Vec<_>>();
    let compare = |method: &str, dst: &[i64]| {
        for i in 0..n {
            let have: Vec<i64> = (0..size).map(|j| dst[j * n + i]).collect();
            assert_eq!(
                have,
                oracle(&column(i), ka, kr, k, size, offset),
                "{method} a={:?} ka={ka} kr={kr} k={k} size={size} offset={offset}",
                column(i)
            );
        }
    };
    let res = |dst: &mut Vec<i64>| LimbsMut {
        ptr: dst.as_mut_ptr(),
        stride: n,
        size,
    };
    if narrow {
        let words: Vec<i64> = flat.iter().map(|&x| x as i64).collect();
        let mut dst = vec![0x5a5a_i64; size * n];
        unsafe {
            BE::znx_normalize(
                res(&mut dst),
                kr,
                k,
                offset,
                Limbs {
                    ptr: words.as_ptr(),
                    stride: n,
                    size: a.len(),
                },
                ka,
                n,
            )
        };
        compare("znx_normalize", &dst);
    }
    if narrow || size_of::<BE::BigWord>() == size_of::<i128>() {
        let words: Vec<BE::BigWord> = flat
            .iter()
            .map(|&x| {
                if narrow {
                    BE::BigWord::ZERO.add_i64(x as i64)
                } else {
                    bytemuck::cast(x)
                }
            })
            .collect();
        let mut dst = vec![0x5a5a_i64; size * n];
        unsafe {
            BE::znx_normalize_big(
                res(&mut dst),
                kr,
                k,
                offset,
                Limbs {
                    ptr: words.as_ptr(),
                    stride: n,
                    size: a.len(),
                },
                ka,
                n,
            )
        };
        compare("znx_normalize_big", &dst);
    }
}

fn check_in_place<BE: ZnxNormalize>() {
    const N: usize = 19;
    let size = 3;
    for (b, k, offset) in [
        (17usize, 51, 0i64),
        (17, 51, 5),
        (17, 51, -40),
        (50, 150, -1),
        (62, 186, -62),
        (17, 40, 0),
        (50, 101, 0),
        (62, 61, 0),
    ] {
        let mut buf: Vec<i64> = (0..size * N)
            .map(|x| (x as i64).wrapping_mul(0x31e5_cd28_3792_b601) >> 3)
            .collect();
        let a: Vec<Vec<i128>> = (0..size).map(|j| (0..N).map(|i| buf[j * N + i] as i128).collect()).collect();
        let ptr = buf.as_mut_ptr();
        unsafe {
            BE::znx_normalize(
                LimbsMut { ptr, stride: N, size },
                b,
                k,
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
                oracle(&column, b, b, k, size, offset),
                "in place b={b} k={k} offset={offset} i={i}"
            );
        }
    }
}

/// Six limbs, so each block spans more than 8 cache lines and the portable driver walks tiles:
/// two tiles of 32 blocks at lane width 8, leftover blocks and a 3-lane tail, other widths more tiles.
fn check_many_limbs<BE: ZnxNormalize>() {
    const N: usize = 2 * 32 * 8 + 3 * 8 + 3;
    const SIZE: usize = 6;
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state as i64
    };
    let narrow: Vec<Vec<i128>> = (0..SIZE)
        .map(|_| {
            (0..N)
                .map(|i| match i % 4 {
                    0 => -(1i128 << 62),
                    1 => 1 << 62,
                    _ => (next() >> 1) as i128,
                })
                .collect()
        })
        .collect();
    let wide: Vec<Vec<i128>> = narrow.iter().map(|l| l.iter().map(|&x| (x << 63) + x).collect()).collect();
    for b in [17usize, 50, 62] {
        for k in [SIZE * b, 2 * b + 1] {
            for offset in [0, b as i64, -(b as i64), -1] {
                check::<BE>(&narrow, b, b, k, SIZE, offset);
                check::<BE>(&wide, b, b, k, SIZE, offset);
            }
        }
        check::<BE>(&narrow, b, b, 4 * b, 4, b as i64);
        check::<BE>(&narrow, b, b, 7 * b, 7, -(b as i64));
    }
    // In place needs `offset < b`.
    for (b, k, offset) in [(17usize, 6 * 17, 0i64), (50, 6 * 50 - 5, 49), (62, 3 * 62 + 1, -62)] {
        let mut buf: Vec<i64> = narrow.iter().flatten().map(|&x| x as i64).collect();
        let ptr = buf.as_mut_ptr();
        unsafe {
            BE::znx_normalize(
                LimbsMut {
                    ptr,
                    stride: N,
                    size: SIZE,
                },
                b,
                k,
                offset,
                Limbs {
                    ptr: ptr.cast_const(),
                    stride: N,
                    size: SIZE,
                },
                b,
                N,
            )
        };
        for i in 0..N {
            let column: Vec<i128> = narrow.iter().map(|l| l[i]).collect();
            let have: Vec<i64> = (0..SIZE).map(|j| buf[j * N + i]).collect();
            assert_eq!(
                have,
                oracle(&column, b, b, k, SIZE, offset),
                "many limbs in place b={b} k={k} offset={offset} i={i}"
            );
        }
    }
}
