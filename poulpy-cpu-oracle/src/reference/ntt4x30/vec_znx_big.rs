use crate::layouts::*;
#[cfg(feature = "enable-core")]
use crate::source::Source;
#[cfg(feature = "enable-core")]
use rand_distr::{Distribution, Normal};

pub trait I128BigOps {
    fn i128_add(res: &mut [i128], a: &[i128], b: &[i128]) {
        res.iter_mut()
            .zip(a.iter())
            .zip(b.iter())
            .for_each(|((r, &ai), &bi)| *r = ai.wrapping_add(bi));
    }

    fn i128_add_assign(res: &mut [i128], a: &[i128]) {
        res.iter_mut().zip(a.iter()).for_each(|(r, &ai)| *r = r.wrapping_add(ai));
    }

    fn i128_add_small_assign(res: &mut [i128], a: &[i64]) {
        res.iter_mut()
            .zip(a.iter())
            .for_each(|(r, &ai)| *r = r.wrapping_add(ai as i128));
    }

    fn i128_sub(res: &mut [i128], a: &[i128], b: &[i128]) {
        res.iter_mut()
            .zip(a.iter())
            .zip(b.iter())
            .for_each(|((r, &ai), &bi)| *r = ai.wrapping_sub(bi));
    }

    fn i128_sub_assign(res: &mut [i128], a: &[i128]) {
        res.iter_mut().zip(a.iter()).for_each(|(r, &ai)| *r = r.wrapping_sub(ai));
    }

    fn i128_sub_negate_assign(res: &mut [i128], a: &[i128]) {
        res.iter_mut().zip(a.iter()).for_each(|(r, &ai)| *r = ai.wrapping_sub(*r));
    }

    fn i128_sub_small_assign(res: &mut [i128], a: &[i64]) {
        res.iter_mut()
            .zip(a.iter())
            .for_each(|(r, &ai)| *r = r.wrapping_sub(ai as i128));
    }

    fn i128_sub_small_negate_assign(res: &mut [i128], a: &[i64]) {
        res.iter_mut()
            .zip(a.iter())
            .for_each(|(r, &ai)| *r = (ai as i128).wrapping_sub(*r));
    }

    fn i128_negate(res: &mut [i128], a: &[i128]) {
        res.iter_mut().zip(a.iter()).for_each(|(r, &ai)| *r = ai.wrapping_neg());
    }

    fn i128_negate_assign(res: &mut [i128]) {
        res.iter_mut().for_each(|r| *r = r.wrapping_neg());
    }

    fn i128_from_small(res: &mut [i128], a: &[i64]) {
        res.iter_mut().zip(a.iter()).for_each(|(r, &ai)| *r = ai as i128);
    }
}

pub fn ntt4x30_vec_znx_big_automorphism_assign_tmp_bytes(n: usize) -> usize {
    n * size_of::<i128>()
}

pub fn ntt4x30_vec_znx_big_add<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
    B: VecZnxBigToBackendRef<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();
    let b = b.to_backend_ref();

    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();
    let sum_size = a_size.min(b_size).min(res_size);

    for j in 0..sum_size {
        BE::i128_add(res.at_mut(res_col, j), a.at(a_col, j), b.at(b_col, j));
    }

    if a_size <= b_size {
        let b_cpy = b_size.min(res_size);
        for j in sum_size..b_cpy {
            let bj = b.at(b_col, j);
            res.at_mut(res_col, j).copy_from_slice(bj);
        }
        for j in b_cpy..res_size {
            res.at_mut(res_col, j).fill(0);
        }
    } else {
        let a_cpy = a_size.min(res_size);
        for j in sum_size..a_cpy {
            let aj = a.at(a_col, j);
            res.at_mut(res_col, j).copy_from_slice(aj);
        }
        for j in a_cpy..res_size {
            res.at_mut(res_col, j).fill(0);
        }
    }
}

pub fn ntt4x30_vec_znx_big_add_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();
    let sum_size = res.size().min(a.size());
    for j in 0..sum_size {
        BE::i128_add_assign(res.at_mut(res_col, j), a.at(a_col, j));
    }
}

pub fn ntt4x30_vec_znx_big_add_small_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxToBackendRef,
    for<'x> BE::BufMut<'x>: HostDataMut,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();
    let sum_size = res.size().min(a.size());
    for j in 0..sum_size {
        BE::i128_add_small_assign(res.at_mut(res_col, j), a.at(a_col, j));
    }
}

pub fn ntt4x30_vec_znx_big_sub<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
    B: VecZnxBigToBackendRef<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();
    let b = b.to_backend_ref();

    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();
    let sum_size = a_size.min(b_size).min(res_size);

    for j in 0..sum_size {
        BE::i128_sub(res.at_mut(res_col, j), a.at(a_col, j), b.at(b_col, j));
    }

    if a_size >= b_size {
        let a_cpy = a_size.min(res_size);
        for j in sum_size..a_cpy {
            res.at_mut(res_col, j).copy_from_slice(a.at(a_col, j));
        }
        for j in a_cpy..res_size {
            res.at_mut(res_col, j).fill(0);
        }
    } else {
        let b_cpy = b_size.min(res_size);
        for j in sum_size..b_cpy {
            BE::i128_negate(res.at_mut(res_col, j), b.at(b_col, j));
        }
        for j in b_cpy..res_size {
            res.at_mut(res_col, j).fill(0);
        }
    }
}

pub fn ntt4x30_vec_znx_big_sub_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();
    let sum_size = res.size().min(a.size());
    for j in 0..sum_size {
        BE::i128_sub_assign(res.at_mut(res_col, j), a.at(a_col, j));
    }
}

pub fn ntt4x30_vec_znx_big_sub_negate_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();
    let res_size = res.size();
    let sum_size = res_size.min(a.size());

    for j in 0..sum_size {
        BE::i128_sub_negate_assign(res.at_mut(res_col, j), a.at(a_col, j));
    }
    for j in a.size()..res_size {
        BE::i128_negate_assign(res.at_mut(res_col, j));
    }
}

pub fn ntt4x30_vec_znx_big_sub_small_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxToBackendRef,
    for<'x> BE::BufMut<'x>: HostDataMut,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();
    let sum_size = res.size().min(a.size());
    for j in 0..sum_size {
        BE::i128_sub_small_assign(res.at_mut(res_col, j), a.at(a_col, j));
    }
}

pub fn ntt4x30_vec_znx_big_sub_small_negate_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxToBackendRef,
    for<'x> BE::BufMut<'x>: HostDataMut,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();
    let res_size = res.size();
    let sum_size = res_size.min(a.size());

    for j in 0..sum_size {
        BE::i128_sub_small_negate_assign(res.at_mut(res_col, j), a.at(a_col, j));
    }
    for j in a.size()..res_size {
        BE::i128_negate_assign(res.at_mut(res_col, j));
    }
}

pub fn ntt4x30_vec_znx_big_negate<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();

    let res_size = res.size();
    let cpy_size = a.size().min(res_size);

    for j in 0..cpy_size {
        BE::i128_negate(res.at_mut(res_col, j), a.at(a_col, j));
    }
    for j in cpy_size..res_size {
        res.at_mut(res_col, j).fill(0);
    }
}

pub fn ntt4x30_vec_znx_big_negate_assign<R, BE>(res: &mut R, res_col: usize)
where
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
    R: VecZnxBigToBackendMut<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut,
{
    let mut res = res.to_backend_mut();
    for j in 0..res.size() {
        BE::i128_negate_assign(res.at_mut(res_col, j));
    }
}

pub fn ntt4x30_vec_znx_big_from_small<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i128, ZnxWord = i64> + I128BigOps,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxToBackendRef,
    for<'x> BE::BufMut<'x>: HostDataMut,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();
    let res_size = res.size();
    let min_size = res_size.min(a.size());

    for j in 0..min_size {
        BE::i128_from_small(res.at_mut(res_col, j), a.at(a_col, j));
    }
    for j in min_size..res_size {
        res.at_mut(res_col, j).fill(0);
    }
}

pub fn ntt4x30_vec_znx_big_automorphism<R, A, BE>(p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i128, ZnxWord = i64>,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();
    poulpy_hal::layouts::assert_dense(&res, "ntt4x30_vec_znx_big_automorphism");
    poulpy_hal::layouts::assert_dense(&a, "ntt4x30_vec_znx_big_automorphism");

    let n = res.n();
    let size = res.size().min(a.size());
    let mask = 2 * n - 1;
    let p_2n = (p & mask as i64) as usize;

    for limb in 0..size {
        let rj = res.at_mut(res_col, limb);
        let aj = a.at(a_col, limb);
        rj[0] = aj[0];
        let mut k: usize = 0;
        for &ai in &aj[1..] {
            k = (k + p_2n) & mask;
            if k < n {
                rj[k] = ai;
            } else {
                rj[k - n] = ai.wrapping_neg();
            }
        }
    }

    for limb in size..res.size() {
        res.at_mut(res_col, limb).iter_mut().for_each(|r| *r = 0);
    }
}

pub fn ntt4x30_vec_znx_big_automorphism_assign<R, BE>(p: i64, res: &mut R, res_col: usize, tmp: &mut [i128])
where
    BE: Backend<BigWord = i128, ZnxWord = i64>,
    R: VecZnxBigToBackendMut<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut,
{
    let mut res = res.to_backend_mut();
    poulpy_hal::layouts::assert_dense(&res, "ntt4x30_vec_znx_big_automorphism_assign");
    let n = res.n();
    let size = res.size();
    let mask = 2 * n - 1;
    let p_2n = (p & mask as i64) as usize;

    for limb in 0..size {
        let rj = res.at_mut(res_col, limb);
        tmp[..n].copy_from_slice(rj);
        rj[0] = tmp[0];
        let mut k: usize = 0;
        for &ti in &tmp[1..n] {
            k = (k + p_2n) & mask;
            if k < n {
                rj[k] = ti;
            } else {
                rj[k - n] = ti.wrapping_neg();
            }
        }
    }
}

#[cfg(feature = "enable-core")]
pub fn ntt4x30_vec_znx_big_add_normal_ref<R, BE>(
    base2k: usize,
    res: &mut R,
    res_col: usize,
    k: usize,
    sigma: f64,
    bound: f64,
    source: &mut Source,
) where
    BE: Backend<BigWord = i128, ZnxWord = i64>,
    R: VecZnxBigToBackendMut<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut,
{
    let mut res = res.to_backend_mut();
    assert!(
        (bound.log2().ceil() as i64) < 64,
        "invalid bound: ceil(log2(bound))={} > 63",
        bound.log2().ceil() as i64
    );

    let limb: usize = k.div_ceil(base2k) - 1;
    let shift: u32 = ((limb + 1) * base2k - k) as u32;
    let normal: Normal<f64> = Normal::new(0.0, sigma).unwrap();
    let rj: &mut [i128] = res.at_mut(res_col, limb);

    rj.iter_mut().for_each(|r| {
        let mut s: f64 = normal.sample(source);
        while s.abs() > bound {
            s = normal.sample(source);
        }
        *r = r.wrapping_add((s.round() as i64 as i128) << shift);
    });
}
