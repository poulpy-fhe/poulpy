use crate::layouts::VecZnxBigBackendMut;
use crate::layouts::VecZnxBigBackendRef;

use crate::{
    kernels::{
        noise::add_noise_portable,
        vec_znx::{
            vec_znx_add_assign_portable, vec_znx_add_portable, vec_znx_automorphism_assign_portable,
            vec_znx_automorphism_portable, vec_znx_negate_assign_portable, vec_znx_negate_portable, vec_znx_sub_assign_portable,
            vec_znx_sub_negate_assign_portable, vec_znx_sub_portable,
        },
        znx::{
            ZnxAdd, ZnxAddAssign, ZnxAutomorphism, ZnxCopy, ZnxNegate, ZnxNegateAssign, ZnxSub, ZnxSubAssign, ZnxSubNegateAssign,
            ZnxZero,
        },
    },
    layouts::{
        Backend, HostDataMut, HostDataRef, Ring, VecZnx, VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxToBackendRef,
    },
    source::Source,
};
use poulpy_core::Noise;

fn big_as_vec_znx_mut<'a, BE, Rg: Ring>(v: VecZnxBigBackendMut<'a, BE, Rg>) -> VecZnx<BE::BufMut<'a>, BE::ZnxWord, Rg>
where
    BE: Backend<ZnxWord = i64>,
{
    let shape = v.shape();
    VecZnx::from_shape(v.data, shape)
}

fn big_as_vec_znx_ref<'a, BE, Rg: Ring>(v: VecZnxBigBackendRef<'a, BE, Rg>) -> VecZnx<BE::BufRef<'a>, BE::ZnxWord, Rg>
where
    BE: Backend<ZnxWord = i64>,
{
    let shape = v.shape();
    VecZnx::from_shape(v.data, shape)
}

pub fn vec_znx_big_add_portable<R, A, B, BE, Rg: Ring>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxAdd + ZnxCopy + ZnxZero,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    R: VecZnxBigToBackendMut<BE, Rg>,
    A: VecZnxBigToBackendRef<BE, Rg>,
    B: VecZnxBigToBackendRef<BE, Rg>,
{
    let mut res_vznx = big_as_vec_znx_mut::<BE, Rg>(res.to_backend_mut());
    let a_vznx = big_as_vec_znx_ref::<BE, Rg>(a.to_backend_ref());
    let b_vznx = big_as_vec_znx_ref::<BE, Rg>(b.to_backend_ref());
    vec_znx_add_portable::<BE, Rg>(&mut res_vznx, res_col, &a_vznx, a_col, &b_vznx, b_col);
}

pub fn vec_znx_big_add_assign_portable<R, A, BE, Rg: Ring>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxAddAssign,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    R: VecZnxBigToBackendMut<BE, Rg>,
    A: VecZnxBigToBackendRef<BE, Rg>,
{
    let mut res_vznx = big_as_vec_znx_mut::<BE, Rg>(res.to_backend_mut());
    let a_vznx = big_as_vec_znx_ref::<BE, Rg>(a.to_backend_ref());
    vec_znx_add_assign_portable::<BE, Rg>(&mut res_vznx, res_col, &a_vznx, a_col);
}

pub fn vec_znx_big_add_small_portable<R, A, B, BE, Rg: Ring>(
    res: &mut R,
    res_col: usize,
    a: &A,
    a_col: usize,
    b: &B,
    b_col: usize,
) where
    BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxAdd + ZnxCopy + ZnxZero,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    R: VecZnxBigToBackendMut<BE, Rg>,
    A: VecZnxBigToBackendRef<BE, Rg>,
    B: VecZnxToBackendRef<BE, Rg>,
{
    let mut res_vznx = big_as_vec_znx_mut::<BE, Rg>(res.to_backend_mut());
    let a_vznx = big_as_vec_znx_ref::<BE, Rg>(a.to_backend_ref());
    let b_ref = b.to_backend_ref();
    vec_znx_add_portable::<BE, Rg>(&mut res_vznx, res_col, &a_vznx, a_col, &b_ref, b_col);
}

pub fn vec_znx_big_add_small_assign_portable<R, A, BE, Rg: Ring>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxAddAssign,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    R: VecZnxBigToBackendMut<BE, Rg>,
    A: VecZnxToBackendRef<BE, Rg>,
{
    let mut res_vznx = big_as_vec_znx_mut::<BE, Rg>(res.to_backend_mut());
    let a_ref = a.to_backend_ref();
    vec_znx_add_assign_portable::<BE, Rg>(&mut res_vznx, res_col, &a_ref, a_col);
}

pub fn vec_znx_big_automorphism_assign_tmp_bytes_portable(n: usize) -> usize {
    n * size_of::<i64>()
}

pub fn vec_znx_big_automorphism_portable<R, A, BE, Rg: Ring>(p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxAutomorphism<Rg> + ZnxZero,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    R: VecZnxBigToBackendMut<BE, Rg>,
    A: VecZnxBigToBackendRef<BE, Rg>,
{
    let mut res_vznx = big_as_vec_znx_mut::<BE, Rg>(res.to_backend_mut());
    let a_vznx = big_as_vec_znx_ref::<BE, Rg>(a.to_backend_ref());
    vec_znx_automorphism_portable::<BE, Rg>(p, &mut res_vznx, res_col, &a_vznx, a_col);
}

pub fn vec_znx_big_automorphism_assign_portable<R, BE, Rg: Ring>(p: i64, res: &mut R, res_col: usize, tmp: &mut [i64])
where
    BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxAutomorphism<Rg> + ZnxCopy,
    for<'a> BE::BufMut<'a>: HostDataMut,
    R: VecZnxBigToBackendMut<BE, Rg>,
{
    let mut res_vznx = big_as_vec_znx_mut::<BE, Rg>(res.to_backend_mut());
    vec_znx_automorphism_assign_portable::<BE, Rg>(p, &mut res_vznx, res_col, tmp);
}

pub fn vec_znx_big_negate_portable<R, A, BE, Rg: Ring>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxNegate + ZnxZero,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    R: VecZnxBigToBackendMut<BE, Rg>,
    A: VecZnxBigToBackendRef<BE, Rg>,
{
    let mut res_vznx = big_as_vec_znx_mut::<BE, Rg>(res.to_backend_mut());
    let a_vznx = big_as_vec_znx_ref::<BE, Rg>(a.to_backend_ref());
    vec_znx_negate_portable::<BE, Rg>(&mut res_vznx, res_col, &a_vznx, a_col);
}

pub fn vec_znx_big_negate_assign_portable<R, BE, Rg: Ring>(res: &mut R, res_col: usize)
where
    BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxNegateAssign,
    for<'a> BE::BufMut<'a>: HostDataMut,
    R: VecZnxBigToBackendMut<BE, Rg>,
{
    let mut res_vznx = big_as_vec_znx_mut::<BE, Rg>(res.to_backend_mut());
    vec_znx_negate_assign_portable::<BE, Rg>(&mut res_vznx, res_col);
}

pub fn vec_znx_big_add_noise_portable<R, BE, Rg: Ring>(
    base2k: usize,
    k: usize,
    res: &mut R,
    res_col: usize,
    noise: Noise,
    source: &mut Source,
) where
    BE: Backend<BigWord = i64, ZnxWord = i64>,
    for<'a> BE::BufMut<'a>: HostDataMut,
    R: VecZnxBigToBackendMut<BE, Rg>,
{
    add_noise_portable(base2k, k, &mut res.to_backend_mut(), res_col, noise, source);
}

/// R <- A - B
pub fn vec_znx_big_sub_portable<R, A, B, BE, Rg: Ring>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxSub + ZnxNegate + ZnxZero + ZnxCopy,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    R: VecZnxBigToBackendMut<BE, Rg>,
    A: VecZnxBigToBackendRef<BE, Rg>,
    B: VecZnxBigToBackendRef<BE, Rg>,
{
    let mut res_vznx = big_as_vec_znx_mut::<BE, Rg>(res.to_backend_mut());
    let a_vznx = big_as_vec_znx_ref::<BE, Rg>(a.to_backend_ref());
    let b_vznx = big_as_vec_znx_ref::<BE, Rg>(b.to_backend_ref());
    vec_znx_sub_portable::<BE, Rg>(&mut res_vznx, res_col, &a_vznx, a_col, &b_vznx, b_col);
}

/// R <- A - B
pub fn vec_znx_big_sub_assign_portable<R, A, BE, Rg: Ring>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxSubAssign,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    R: VecZnxBigToBackendMut<BE, Rg>,
    A: VecZnxBigToBackendRef<BE, Rg>,
{
    let mut res_vznx = big_as_vec_znx_mut::<BE, Rg>(res.to_backend_mut());
    let a_vznx = big_as_vec_znx_ref::<BE, Rg>(a.to_backend_ref());
    vec_znx_sub_assign_portable::<BE, Rg>(&mut res_vznx, res_col, &a_vznx, a_col);
}

/// R <- B - A
pub fn vec_znx_big_sub_negate_assign_portable<R, A, BE, Rg: Ring>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxSubNegateAssign + ZnxNegateAssign,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    R: VecZnxBigToBackendMut<BE, Rg>,
    A: VecZnxBigToBackendRef<BE, Rg>,
{
    let mut res_vznx = big_as_vec_znx_mut::<BE, Rg>(res.to_backend_mut());
    let a_vznx = big_as_vec_znx_ref::<BE, Rg>(a.to_backend_ref());
    vec_znx_sub_negate_assign_portable::<BE, Rg>(&mut res_vznx, res_col, &a_vznx, a_col);
}

/// R <- A - B
pub fn vec_znx_big_sub_small_a_portable<R, A, B, BE, Rg: Ring>(
    res: &mut R,
    res_col: usize,
    a: &A,
    a_col: usize,
    b: &B,
    b_col: usize,
) where
    BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxSub + ZnxNegate + ZnxZero + ZnxCopy,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    R: VecZnxBigToBackendMut<BE, Rg>,
    A: VecZnxToBackendRef<BE, Rg>,
    B: VecZnxBigToBackendRef<BE, Rg>,
{
    let mut res_vznx = big_as_vec_znx_mut::<BE, Rg>(res.to_backend_mut());
    let b_vznx = big_as_vec_znx_ref::<BE, Rg>(b.to_backend_ref());
    let a_ref = a.to_backend_ref();
    vec_znx_sub_portable::<BE, Rg>(&mut res_vznx, res_col, &a_ref, a_col, &b_vznx, b_col);
}

/// R <- A - B
pub fn vec_znx_big_sub_small_b_portable<R, A, B, BE, Rg: Ring>(
    res: &mut R,
    res_col: usize,
    a: &A,
    a_col: usize,
    b: &B,
    b_col: usize,
) where
    BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxSub + ZnxNegate + ZnxZero + ZnxCopy,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    R: VecZnxBigToBackendMut<BE, Rg>,
    A: VecZnxBigToBackendRef<BE, Rg>,
    B: VecZnxToBackendRef<BE, Rg>,
{
    let mut res_vznx = big_as_vec_znx_mut::<BE, Rg>(res.to_backend_mut());
    let a_vznx = big_as_vec_znx_ref::<BE, Rg>(a.to_backend_ref());
    let b_ref = b.to_backend_ref();
    vec_znx_sub_portable::<BE, Rg>(&mut res_vznx, res_col, &a_vznx, a_col, &b_ref, b_col);
}

///  R <- R - A
pub fn vec_znx_big_sub_small_a_assign_portable<R, A, BE, Rg: Ring>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxSubAssign,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    R: VecZnxBigToBackendMut<BE, Rg>,
    A: VecZnxToBackendRef<BE, Rg>,
{
    let mut res_vznx = big_as_vec_znx_mut::<BE, Rg>(res.to_backend_mut());
    let a_ref = a.to_backend_ref();
    vec_znx_sub_assign_portable::<BE, Rg>(&mut res_vznx, res_col, &a_ref, a_col);
}

/// R <- A - R
pub fn vec_znx_big_sub_small_b_assign_portable<R, A, BE, Rg: Ring>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxSubNegateAssign + ZnxNegateAssign,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    R: VecZnxBigToBackendMut<BE, Rg>,
    A: VecZnxToBackendRef<BE, Rg>,
{
    let mut res_vznx = big_as_vec_znx_mut::<BE, Rg>(res.to_backend_mut());
    let a_ref = a.to_backend_ref();
    vec_znx_sub_negate_assign_portable::<BE, Rg>(&mut res_vznx, res_col, &a_ref, a_col);
}
