use dashu_float::{Context, FBig, round::mode::HalfEven};
use itertools::izip;

use crate::layouts::{DataMut, DataRef, VecZnx, VecZnxToMut, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut};

impl<D: DataMut> VecZnx<D> {
    /// Encodes an `i64` slice into the limb-decomposed (base-2^k) representation.
    ///
    /// The input `data` (length `N`) is placed at the appropriate limb position
    /// determined by `k` and `base2k`, then normalized across all limbs.
    ///
    /// # Panics (debug)
    ///
    /// - `k.div_ceil(base2k) > self.size()`
    /// - `col >= self.cols()`
    /// - `data.len() != N`
    pub fn encode_vec_i64(&mut self, base2k: usize, col: usize, k: usize, data: &[i64]) {
        let size: usize = k.div_ceil(base2k);

        #[cfg(debug_assertions)]
        {
            let a: VecZnx<&mut [u8]> = self.to_mut();
            assert!(
                size <= a.size(),
                "invalid argument k.div_ceil(base2k)={} > a.size()={}",
                size,
                a.size()
            );
            assert!(col < a.cols());
            assert!(data.len() == a.n())
        }

        let mut a: VecZnx<&mut [u8]> = self.to_mut();
        let a_size: usize = a.size();

        // Zeroes coefficients of the i-th column
        for i in 0..a_size {
            znx_zero_ref(a.at_mut(col, i));
        }

        // Copies the data on the correct limb
        a.at_mut(col, size - 1).copy_from_slice(data);

        let mut carry: Vec<i64> = vec![0i64; a.n()];
        let k_rem: usize = (base2k - (k % base2k)) % base2k;

        // Normalizes and shift if necessary.
        for j in (0..size).rev() {
            if j == size - 1 {
                znx_normalize_first_step_assign(base2k, k_rem, a.at_mut(col, j), &mut carry);
            } else if j == 0 {
                znx_normalize_final_step_assign(base2k, k_rem, a.at_mut(col, j), &mut carry);
            } else {
                znx_normalize_middle_step_assign(base2k, k_rem, a.at_mut(col, j), &mut carry);
            }
        }
    }

    /// Encodes an `i128` slice into the limb-decomposed (base-2^k) representation.
    ///
    /// Analogous to [`encode_vec_i64`](VecZnx::encode_vec_i64) but accepts wider
    /// input values.
    pub fn encode_vec_i128(&mut self, base2k: usize, col: usize, k: usize, data: &[i128]) {
        let size: usize = k.div_ceil(base2k);

        #[cfg(debug_assertions)]
        {
            let a: VecZnx<&mut [u8]> = self.to_mut();
            assert!(
                size <= a.size(),
                "invalid argument k.div_ceil(base2k)={} > a.size()={}",
                size,
                a.size()
            );
            assert!(col < a.cols());
            assert!(data.len() == a.n())
        }

        let mut a: VecZnx<&mut [u8]> = self.to_mut();
        let a_size: usize = a.size();

        {
            let mut carry_i128: Vec<i128> = vec![0i128; a.n()];
            carry_i128.copy_from_slice(data);

            for j in (0..size).rev() {
                for (x, a) in izip!(a.at_mut(col, j).iter_mut(), carry_i128.iter_mut()) {
                    let digit: i128 = get_digit_i128(base2k, *a);
                    let carry: i128 = get_carry_i128(base2k, *a, digit);
                    *x = digit as i64;
                    *a = carry;
                }
            }
        }

        for j in size..a_size {
            znx_zero_ref(a.at_mut(col, j));
        }

        let mut carry: Vec<i64> = vec![0i64; a.n()];
        let k_rem: usize = (base2k - (k % base2k)) % base2k;

        for j in (0..size).rev() {
            if j == size - 1 {
                znx_normalize_first_step_assign(base2k, k_rem, a.at_mut(col, j), &mut carry);
            } else if j == 0 {
                znx_normalize_final_step_assign(base2k, k_rem, a.at_mut(col, j), &mut carry);
            } else {
                znx_normalize_middle_step_assign(base2k, k_rem, a.at_mut(col, j), &mut carry);
            }
        }
    }

    /// Encodes a single coefficient at index `idx` into the limb-decomposed
    /// representation, zeroing all other coefficients of column `col`.
    pub fn encode_coeff_i64(&mut self, base2k: usize, col: usize, k: usize, idx: usize, data: i64) {
        let size: usize = k.div_ceil(base2k);

        #[cfg(debug_assertions)]
        {
            let a: VecZnx<&mut [u8]> = self.to_mut();
            assert!(idx < a.n());
            assert!(
                size <= a.size(),
                "invalid argument k.div_ceil(base2k)={} > a.size()={}",
                size,
                a.size()
            );
            assert!(col < a.cols());
        }

        let mut a: VecZnx<&mut [u8]> = self.to_mut();
        let a_size = a.size();

        for j in 0..a_size {
            a.at_mut(col, j)[idx] = 0
        }

        a.at_mut(col, size - 1)[idx] = data;

        let mut carry: Vec<i64> = vec![0i64; 1];
        let k_rem: usize = (base2k - (k % base2k)) % base2k;

        for j in (0..size).rev() {
            let slice = &mut a.at_mut(col, j)[idx..idx + 1];

            if j == size - 1 {
                znx_normalize_first_step_assign(base2k, k_rem, slice, &mut carry);
            } else if j == 0 {
                znx_normalize_final_step_assign(base2k, k_rem, slice, &mut carry);
            } else {
                znx_normalize_middle_step_assign(base2k, k_rem, slice, &mut carry);
            }
        }
    }
}

impl<D: DataRef> VecZnx<D> {
    /// Decodes column `col` from the limb-decomposed representation back into
    /// an `i64` slice, reconstructing values up to `k` bits of precision.
    pub fn decode_vec_i64(&self, base2k: usize, col: usize, k: usize, data: &mut [i64]) {
        let size: usize = k.div_ceil(base2k);
        #[cfg(debug_assertions)]
        {
            let a: VecZnx<&[u8]> = self.to_ref();
            assert!(
                data.len() >= a.n(),
                "invalid data: data.len()={} < a.n()={}",
                data.len(),
                a.n()
            );
            assert!(col < a.cols());
        }

        let a: VecZnx<&[u8]> = self.to_ref();
        data.copy_from_slice(a.at(col, 0));
        let rem: usize = base2k - (k % base2k);
        if k < base2k {
            let scale = 1 << rem as i64;
            data.iter_mut().for_each(|x| *x = div_round_i64(*x, scale));
        } else {
            (1..size).for_each(|i| {
                if i == size - 1 && rem != base2k {
                    let k_rem: usize = (base2k - rem) % base2k;
                    let scale: i64 = 1 << rem as i64;
                    izip!(a.at(col, i).iter(), data.iter_mut()).for_each(|(x, y)| {
                        *y = (*y << k_rem) + div_round_i64(*x, scale);
                    });
                } else {
                    izip!(a.at(col, i).iter(), data.iter_mut()).for_each(|(x, y)| {
                        *y = (*y << base2k) + x;
                    });
                }
            })
        }
    }

    pub fn decode_vec_i128(&self, base2k: usize, col: usize, k: usize, data: &mut [i128]) {
        let size: usize = k.div_ceil(base2k);
        #[cfg(debug_assertions)]
        {
            let a: VecZnx<&[u8]> = self.to_ref();
            assert!(
                data.len() >= a.n(),
                "invalid data: data.len()={} < a.n()={}",
                data.len(),
                a.n()
            );
            assert!(col < a.cols());
        }

        let a: VecZnx<&[u8]> = self.to_ref();
        data.iter_mut()
            .zip(a.at(col, 0).iter())
            .for_each(|(bi, ai)| *bi = *ai as i128);

        let rem: usize = base2k - (k % base2k);
        if k < base2k {
            let scale = 1 << rem as i128;
            data.iter_mut().for_each(|x| *x = div_round_i128(*x, scale));
        } else {
            (1..size).for_each(|i| {
                if i == size - 1 && rem != base2k {
                    let k_rem: usize = (base2k - rem) % base2k;
                    let scale: i128 = 1 << rem as i128;
                    izip!(a.at(col, i).iter(), data.iter_mut()).for_each(|(x, y)| {
                        *y = (*y << k_rem) + div_round_i128(*x as i128, scale);
                    });
                } else {
                    izip!(a.at(col, i).iter(), data.iter_mut()).for_each(|(x, y)| {
                        *y = (*y << base2k) + *x as i128;
                    });
                }
            })
        }
    }

    /// Decodes a single coefficient at index `idx` from the limb-decomposed
    /// representation back into an `i64`.
    pub fn decode_coeff_i64(&self, base2k: usize, col: usize, k: usize, idx: usize) -> i64 {
        #[cfg(debug_assertions)]
        {
            let a: VecZnx<&[u8]> = self.to_ref();
            assert!(idx < a.n());
            assert!(col < a.cols())
        }

        let a: VecZnx<&[u8]> = self.to_ref();
        let size: usize = k.div_ceil(base2k);
        let mut res: i64 = 0;
        let rem: usize = base2k - (k % base2k);
        (0..size).for_each(|j| {
            let x: i64 = a.at(col, j)[idx];
            if j == size - 1 && rem != base2k {
                let k_rem: usize = (base2k - rem) % base2k;
                let scale: i64 = 1 << rem as i64;
                res = (res << k_rem) + div_round_i64(x, scale);
            } else {
                res = (res << base2k) + x;
            }
        });
        res
    }

    /// Decodes column `col` into arbitrary-precision [`FBig`] values by
    /// evaluating `sum_j coeff[j] * 2^{-base2k * j}` using all limbs (Horner's method).
    pub fn decode_vec_float(&self, base2k: usize, col: usize, data: &mut [FBig<HalfEven>]) {
        #[cfg(debug_assertions)]
        {
            let a: VecZnx<&[u8]> = self.to_ref();
            assert!(
                data.len() >= a.n(),
                "invalid data: data.len()={} < a.n()={}",
                data.len(),
                a.n()
            );
            assert!(col < a.cols());
        }

        let a: VecZnx<&[u8]> = self.to_ref();
        let size: usize = a.size();
        // Extra 256 guard bits absorb cancellation in downstream reduce(x * 2^offset)
        // operations (offset up to 128 bits) without affecting the public f64 API.
        let prec = size * base2k + 256;
        let ctx = Context::<HalfEven>::new(prec);

        // 2^{base2k}
        let scale: FBig<HalfEven> = FBig::from(1u64 << base2k.min(63));

        // y[i] = sum x[j][i] * 2^{-base2k*j}  (Horner: inner-first)
        (0..size).for_each(|i| {
            if i == 0 {
                izip!(a.at(col, size - i - 1).iter(), data.iter_mut()).for_each(|(x, y)| {
                    *y = ctx.div(FBig::<HalfEven>::from(*x).repr(), scale.repr()).value();
                });
            } else {
                izip!(a.at(col, size - i - 1).iter(), data.iter_mut()).for_each(|(x, y)| {
                    *y = ctx.div((y.clone() + FBig::<HalfEven>::from(*x)).repr(), scale.repr()).value();
                });
            }
        });
    }
}

/// Integer division with rounding to nearest (ties away from zero).
///
/// # Panics
///
/// Panics if `b == 0`.
#[inline]
pub fn div_round_i64(a: i64, b: i64) -> i64 {
    assert!(b != 0, "division by zero");
    let div = a / b;
    let rem = a % b;
    if (2 * rem.abs()) >= b.abs() {
        div + a.signum() * b.signum()
    } else {
        div
    }
}

#[inline]
pub fn div_round_i128(a: i128, b: i128) -> i128 {
    assert!(b != 0, "division by zero");
    let div = a / b;
    let rem = a % b;
    if (2 * rem.abs()) >= b.abs() {
        div + a.signum() * b.signum()
    } else {
        div
    }
}

fn znx_zero_ref(res: &mut [i64]) {
    res.fill(0);
}

#[inline(always)]
fn get_digit_i64(base2k: usize, x: i64) -> i64 {
    (x << (u64::BITS - base2k as u32)) >> (u64::BITS - base2k as u32)
}

#[inline(always)]
fn get_carry_i64(base2k: usize, x: i64, digit: i64) -> i64 {
    (x.wrapping_sub(digit)) >> base2k
}

#[inline(always)]
fn get_digit_i128(base2k: usize, x: i128) -> i128 {
    (x << (u128::BITS - base2k as u32)) >> (u128::BITS - base2k as u32)
}

#[inline(always)]
fn get_carry_i128(base2k: usize, x: i128, digit: i128) -> i128 {
    (x.wrapping_sub(digit)) >> base2k
}

#[inline(always)]
fn znx_normalize_first_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert!(x.len() <= carry.len());
        assert!(lsh < base2k);
    }

    if lsh == 0 {
        x.iter_mut().zip(carry.iter_mut()).for_each(|(x, c)| {
            let digit: i64 = get_digit_i64(base2k, *x);
            *c = get_carry_i64(base2k, *x, digit);
            *x = digit;
        });
    } else {
        let base2k_lsh: usize = base2k - lsh;
        x.iter_mut().zip(carry.iter_mut()).for_each(|(x, c)| {
            let digit: i64 = get_digit_i64(base2k_lsh, *x);
            *c = get_carry_i64(base2k_lsh, *x, digit);
            *x = digit << lsh;
        });
    }
}

#[inline(always)]
fn znx_normalize_middle_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert!(x.len() <= carry.len());
        assert!(lsh < base2k);
    }

    if lsh == 0 {
        x.iter_mut().zip(carry.iter_mut()).for_each(|(x, c)| {
            let digit: i64 = get_digit_i64(base2k, *x);
            let carry: i64 = get_carry_i64(base2k, *x, digit);
            let digit_plus_c: i64 = digit + *c;
            *x = get_digit_i64(base2k, digit_plus_c);
            *c = carry + get_carry_i64(base2k, digit_plus_c, *x);
        });
    } else {
        let base2k_lsh: usize = base2k - lsh;
        x.iter_mut().zip(carry.iter_mut()).for_each(|(x, c)| {
            let digit: i64 = get_digit_i64(base2k_lsh, *x);
            let carry: i64 = get_carry_i64(base2k_lsh, *x, digit);
            let digit_plus_c: i64 = (digit << lsh) + *c;
            *x = get_digit_i64(base2k, digit_plus_c);
            *c = carry + get_carry_i64(base2k, digit_plus_c, *x);
        });
    }
}

#[inline(always)]
fn znx_normalize_final_step_assign(base2k: usize, lsh: usize, x: &mut [i64], carry: &mut [i64]) {
    #[cfg(debug_assertions)]
    {
        assert!(x.len() <= carry.len());
        assert!(lsh < base2k);
    }

    if lsh == 0 {
        x.iter_mut().zip(carry.iter_mut()).for_each(|(x, c)| {
            *x = get_digit_i64(base2k, get_digit_i64(base2k, *x) + *c);
        });
    } else {
        let base2k_lsh: usize = base2k - lsh;
        x.iter_mut().zip(carry.iter_mut()).for_each(|(x, c)| {
            *x = get_digit_i64(base2k, (get_digit_i64(base2k_lsh, *x) << lsh) + *c);
        });
    }
}
