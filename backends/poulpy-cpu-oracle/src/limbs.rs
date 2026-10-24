//! Limb-wise combinators shared by the coefficient, big and transformed
//! vectors. Input limbs past an operand's size read as absent, and output
//! limbs no input reaches are zero.

use poulpy_hal::layouts::{ZnxView, ZnxViewMut};

/// `res[j] = both(a[j], b[j])` where both limbs exist, `only_a(a[j])` or
/// `only_b(b[j])` where one does, zero elsewhere.
pub fn zip<R, A, B>(
    res: &mut R,
    res_col: usize,
    a: &A,
    a_col: usize,
    b: &B,
    b_col: usize,
    both: impl Fn(A::Scalar, B::Scalar) -> R::Scalar,
    only_a: impl Fn(A::Scalar) -> R::Scalar,
    only_b: impl Fn(B::Scalar) -> R::Scalar,
) where
    R: ZnxViewMut,
    A: ZnxView,
    B: ZnxView,
    R::Scalar: Copy + Default,
    A::Scalar: Copy,
    B::Scalar: Copy,
{
    assert_eq!(a.n(), res.n());
    assert_eq!(b.n(), res.n());
    for j in 0..res.size() {
        let out = res.at_mut(res_col, j);
        match (j < a.size(), j < b.size()) {
            (true, true) => {
                let (x, y) = (a.at(a_col, j), b.at(b_col, j));
                (0..out.len()).for_each(|i| out[i] = both(x[i], y[i]));
            }
            (true, false) => {
                let x = a.at(a_col, j);
                (0..out.len()).for_each(|i| out[i] = only_a(x[i]));
            }
            (false, true) => {
                let y = b.at(b_col, j);
                (0..out.len()).for_each(|i| out[i] = only_b(y[i]));
            }
            (false, false) => out.fill(R::Scalar::default()),
        }
    }
}

/// `res[j] = f(a[j])` where `a[j]` exists, zero elsewhere.
pub fn map<R, A>(res: &mut R, res_col: usize, a: &A, a_col: usize, f: impl Fn(A::Scalar) -> R::Scalar)
where
    R: ZnxViewMut,
    A: ZnxView,
    R::Scalar: Copy + Default,
    A::Scalar: Copy,
{
    assert_eq!(a.n(), res.n());
    for j in 0..res.size() {
        let out = res.at_mut(res_col, j);
        if j < a.size() {
            let x = a.at(a_col, j);
            (0..out.len()).for_each(|i| out[i] = f(x[i]));
        } else {
            out.fill(R::Scalar::default());
        }
    }
}

/// `res[j] = f(res[j], a[j])` where `a[j]` exists, `rest(res[j])` elsewhere.
pub fn update<R, A>(
    res: &mut R,
    res_col: usize,
    a: &A,
    a_col: usize,
    f: impl Fn(R::Scalar, A::Scalar) -> R::Scalar,
    rest: impl Fn(R::Scalar) -> R::Scalar,
) where
    R: ZnxViewMut,
    A: ZnxView,
    R::Scalar: Copy,
    A::Scalar: Copy,
{
    assert_eq!(a.n(), res.n());
    for j in 0..res.size() {
        let out = res.at_mut(res_col, j);
        if j < a.size() {
            let x = a.at(a_col, j);
            (0..out.len()).for_each(|i| out[i] = f(out[i], x[i]));
        } else {
            (0..out.len()).for_each(|i| out[i] = rest(out[i]));
        }
    }
}

/// `res[j] = f(res[j])` on every limb.
pub fn apply<R>(res: &mut R, res_col: usize, f: impl Fn(R::Scalar) -> R::Scalar)
where
    R: ZnxViewMut,
    R::Scalar: Copy,
{
    for j in 0..res.size() {
        res.at_mut(res_col, j).iter_mut().for_each(|x| *x = f(*x));
    }
}

/// `res[j] = f(a[j])` limb by limb on whole polynomials, zero past `a`.
pub fn map_poly<R, A>(res: &mut R, res_col: usize, a: &A, a_col: usize, f: impl Fn(&mut [R::Scalar], &[A::Scalar]))
where
    R: ZnxViewMut,
    A: ZnxView,
    R::Scalar: Copy + Default,
{
    for j in 0..res.size() {
        if j < a.size() {
            f(res.at_mut(res_col, j), a.at(a_col, j));
        } else {
            res.at_mut(res_col, j).fill(R::Scalar::default());
        }
    }
}

/// `res[j] = f(res[j])` on whole polynomials, through a copy.
pub fn apply_poly<R>(res: &mut R, res_col: usize, f: impl Fn(&mut [R::Scalar], &[R::Scalar]))
where
    R: ZnxViewMut,
    R::Scalar: Copy,
{
    for j in 0..res.size() {
        let input = res.at(res_col, j).to_vec();
        f(res.at_mut(res_col, j), &input);
    }
}
