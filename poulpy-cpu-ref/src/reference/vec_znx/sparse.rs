//! Mixed-degree coefficient-domain add and sub.
//!
//! A degree-`n` operand, `n` a power-of-two divisor of the destination degree
//! `N`, stands for its ring embedding: coefficient `k` lands on coefficient
//! `k * N / n` of the destination and every other coefficient of the embedding
//! is zero (spec 4.5, #266). These kernels read the `n` stored coefficients
//! with a stride and never build the embedding. They are the fallback the
//! dense reference kernels take when an operand's degree differs from the
//! destination's; a call whose operands all share the destination degree
//! never reaches them.

use crate::layouts::{ZnxView, ZnxViewMut};

/// The word types the strided kernels run on: the coefficient word `i64` and
/// the NTT big word `i128`.
pub trait SparseWord: Copy {
    const ZERO: Self;
    fn wadd(self, other: Self) -> Self;
    fn wsub(self, other: Self) -> Self;
    fn wneg(self) -> Self;
}

macro_rules! impl_sparse_word {
    ($($t:ty),*) => {$(
        impl SparseWord for $t {
            const ZERO: Self = 0;
            #[inline(always)]
            fn wadd(self, other: Self) -> Self {
                self.wrapping_add(other)
            }
            #[inline(always)]
            fn wsub(self, other: Self) -> Self {
                self.wrapping_sub(other)
            }
            #[inline(always)]
            fn wneg(self) -> Self {
                self.wrapping_neg()
            }
        }
    )*};
}
impl_sparse_word!(i64, i128);

/// The embedding gap `res_n / n` of a degree-`n` operand into degree `res_n`.
///
/// Panics unless `n` is a power of two that divides `res_n`; a larger degree
/// is not an embedding and is rejected here, once, at kernel entry.
pub fn embedding_gap(res_n: usize, n: usize) -> usize {
    assert!(
        n.is_power_of_two() && n <= res_n && res_n.is_multiple_of(n),
        "operand of degree {n} does not embed into degree {res_n}"
    );
    res_n / n
}

/// `res[k * gap] += a[k]` for every coefficient `k` of `a`, `gap = res.len() / a.len()`.
#[inline(always)]
pub fn znx_add_strided<T: SparseWord, U: Copy + Into<T>>(res: &mut [T], a: &[U]) {
    debug_assert!(
        res.len().is_multiple_of(a.len()),
        "res.len():{} not a multiple of a.len():{}",
        res.len(),
        a.len()
    );
    let gap = res.len() / a.len();
    res.iter_mut()
        .step_by(gap)
        .zip(a.iter())
        .for_each(|(r, &x)| *r = r.wadd(x.into()));
}

/// `res[k * gap] -= a[k]` for every coefficient `k` of `a`, `gap = res.len() / a.len()`.
#[inline(always)]
pub fn znx_sub_strided<T: SparseWord, U: Copy + Into<T>>(res: &mut [T], a: &[U]) {
    debug_assert!(
        res.len().is_multiple_of(a.len()),
        "res.len():{} not a multiple of a.len():{}",
        res.len(),
        a.len()
    );
    let gap = res.len() / a.len();
    res.iter_mut()
        .step_by(gap)
        .zip(a.iter())
        .for_each(|(r, &x)| *r = r.wsub(x.into()));
}

/// `res[k * gap] = a[k]` for every coefficient `k` of `a`, `gap = res.len() / a.len()`;
/// every coefficient of `res` off the stride is zero.
#[inline(always)]
pub fn znx_set_strided<T: SparseWord, U: Copy + Into<T>>(res: &mut [T], a: &[U]) {
    debug_assert!(
        res.len().is_multiple_of(a.len()),
        "res.len():{} not a multiple of a.len():{}",
        res.len(),
        a.len()
    );
    let gap = res.len() / a.len();
    res.fill(T::ZERO);
    res.iter_mut().step_by(gap).zip(a.iter()).for_each(|(r, &x)| *r = x.into());
}

/// `res[res_col] = a[a_col]`, `a` embedded into `res`'s degree and each
/// coefficient widened; limbs of `res` past `a.size()` are zero.
pub fn vec_znx_from_small_mixed<R, A>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    R: ZnxViewMut,
    A: ZnxView,
    R::Scalar: SparseWord,
    A::Scalar: Into<R::Scalar>,
{
    embedding_gap(res.n(), a.n());
    let a_size = a.size();
    for j in 0..res.size() {
        let r = res.at_mut(res_col, j);
        if j < a_size {
            znx_set_strided(r, a.at(a_col, j));
        } else {
            r.fill(R::Scalar::ZERO);
        }
    }
}

/// `res[res_col] = a[a_col] + b[b_col]`, each operand embedded into `res`'s degree;
/// limbs past an operand's size read as zero, every limb of `res` is written.
pub fn vec_znx_add_mixed<R, A, B>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    R: ZnxViewMut,
    A: ZnxView,
    B: ZnxView,
    R::Scalar: SparseWord,
    A::Scalar: Into<R::Scalar>,
    B::Scalar: Into<R::Scalar>,
{
    let n = res.n();
    embedding_gap(n, a.n());
    embedding_gap(n, b.n());
    let (a_size, b_size) = (a.size(), b.size());
    for j in 0..res.size() {
        let r = res.at_mut(res_col, j);
        r.fill(R::Scalar::ZERO);
        if j < a_size {
            znx_add_strided(r, a.at(a_col, j));
        }
        if j < b_size {
            znx_add_strided(r, b.at(b_col, j));
        }
    }
}

/// `res[res_col] = a[a_col] - b[b_col]`, as [`vec_znx_add_mixed`].
pub fn vec_znx_sub_mixed<R, A, B>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    R: ZnxViewMut,
    A: ZnxView,
    B: ZnxView,
    R::Scalar: SparseWord,
    A::Scalar: Into<R::Scalar>,
    B::Scalar: Into<R::Scalar>,
{
    let n = res.n();
    embedding_gap(n, a.n());
    embedding_gap(n, b.n());
    let (a_size, b_size) = (a.size(), b.size());
    for j in 0..res.size() {
        let r = res.at_mut(res_col, j);
        r.fill(R::Scalar::ZERO);
        if j < a_size {
            znx_add_strided(r, a.at(a_col, j));
        }
        if j < b_size {
            znx_sub_strided(r, b.at(b_col, j));
        }
    }
}

/// `res[res_col] += a[a_col]` over the first `min(res.size(), a.size())` limbs.
pub fn vec_znx_add_assign_mixed<R, A>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    R: ZnxViewMut,
    A: ZnxView,
    R::Scalar: SparseWord,
    A::Scalar: Into<R::Scalar>,
{
    embedding_gap(res.n(), a.n());
    for j in 0..res.size().min(a.size()) {
        znx_add_strided(res.at_mut(res_col, j), a.at(a_col, j));
    }
}

/// `res[res_col] -= a[a_col]` over the first `min(res.size(), a.size())` limbs.
pub fn vec_znx_sub_assign_mixed<R, A>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    R: ZnxViewMut,
    A: ZnxView,
    R::Scalar: SparseWord,
    A::Scalar: Into<R::Scalar>,
{
    embedding_gap(res.n(), a.n());
    for j in 0..res.size().min(a.size()) {
        znx_sub_strided(res.at_mut(res_col, j), a.at(a_col, j));
    }
}

/// `res[res_col] = a[a_col] - res[res_col]`: every limb of `res` is negated, the
/// first `min(res.size(), a.size())` then gain `a`.
pub fn vec_znx_sub_negate_assign_mixed<R, A>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    R: ZnxViewMut,
    A: ZnxView,
    R::Scalar: SparseWord,
    A::Scalar: Into<R::Scalar>,
{
    embedding_gap(res.n(), a.n());
    let a_size = a.size();
    for j in 0..res.size() {
        let r = res.at_mut(res_col, j);
        r.iter_mut().for_each(|x| *x = x.wneg());
        if j < a_size {
            znx_add_strided(r, a.at(a_col, j));
        }
    }
}
