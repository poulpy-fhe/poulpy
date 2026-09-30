use crate::layouts::VecZnxBigBackendMut;
use crate::layouts::VecZnxBigBackendRef;

use crate::{
    layouts::{
        Backend, HostDataMut, HostDataRef, VecZnx, VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxToBackendMut,
        VecZnxToBackendRef, ZnxViewMut,
    },
    reference::{
        vec_znx::{
            vec_znx_add, vec_znx_add_assign, vec_znx_automorphism, vec_znx_automorphism_assign, vec_znx_negate,
            vec_znx_negate_assign, vec_znx_normalize, vec_znx_normalize_tmp_bytes, vec_znx_sub, vec_znx_sub_assign,
            vec_znx_sub_negate_assign,
        },
        znx::{
            I64NormalizeOps, ZnxAdd, ZnxAddAssign, ZnxAutomorphism, ZnxCopy, ZnxMulPowerOfTwoAssign, ZnxNegate, ZnxNegateAssign,
            ZnxNormalizeDigit, ZnxNormalizeFinalStep, ZnxNormalizeFinalStepAssign, ZnxNormalizeFirstStep,
            ZnxNormalizeFirstStepCarryOnly, ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepAssign,
            ZnxNormalizeMiddleStepCarryOnly, ZnxSub, ZnxSubAssign, ZnxSubNegateAssign, ZnxZero, znx_add_normal_f64_ref,
        },
    },
    source::Source,
};

fn big_as_vec_znx_mut<'a, BE>(v: VecZnxBigBackendMut<'a, BE>) -> VecZnx<BE::BufMut<'a>, BE::ZnxWord>
where
    BE: Backend<ZnxWord = i64>,
{
    let shape = v.shape();
    VecZnx::from_shape(v.data, shape)
}

fn big_as_vec_znx_ref<'a, BE>(v: VecZnxBigBackendRef<'a, BE>) -> VecZnx<BE::BufRef<'a>, BE::ZnxWord>
where
    BE: Backend<ZnxWord = i64>,
{
    let shape = v.shape();
    VecZnx::from_shape(v.data, shape)
}

pub fn vec_znx_big_add<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxAdd + ZnxCopy + ZnxZero,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
    B: VecZnxBigToBackendRef<BE>,
{
    let mut res_vznx = big_as_vec_znx_mut::<BE>(res.to_backend_mut());
    let a_vznx = big_as_vec_znx_ref::<BE>(a.to_backend_ref());
    let b_vznx = big_as_vec_znx_ref::<BE>(b.to_backend_ref());
    vec_znx_add::<BE>(&mut res_vznx, res_col, &a_vznx, a_col, &b_vznx, b_col);
}

pub fn vec_znx_big_add_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxAddAssign,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
{
    let mut res_vznx = big_as_vec_znx_mut::<BE>(res.to_backend_mut());
    let a_vznx = big_as_vec_znx_ref::<BE>(a.to_backend_ref());
    vec_znx_add_assign::<BE>(&mut res_vznx, res_col, &a_vznx, a_col);
}

pub fn vec_znx_big_add_small<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxAdd + ZnxCopy + ZnxZero,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
    B: VecZnxToBackendRef<BE>,
{
    let mut res_vznx = big_as_vec_znx_mut::<BE>(res.to_backend_mut());
    let a_vznx = big_as_vec_znx_ref::<BE>(a.to_backend_ref());
    let b_ref = b.to_backend_ref();
    vec_znx_add::<BE>(&mut res_vznx, res_col, &a_vznx, a_col, &b_ref, b_col);
}

pub fn vec_znx_big_add_small_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxAddAssign,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxToBackendRef<BE>,
{
    let mut res_vznx = big_as_vec_znx_mut::<BE>(res.to_backend_mut());
    let a_ref = a.to_backend_ref();
    vec_znx_add_assign::<BE>(&mut res_vznx, res_col, &a_ref, a_col);
}

pub fn vec_znx_big_automorphism_assign_tmp_bytes(n: usize) -> usize {
    n * size_of::<i64>()
}

pub fn vec_znx_big_automorphism<R, A, BE>(p: i64, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxAutomorphism + ZnxZero,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
{
    let mut res_vznx = big_as_vec_znx_mut::<BE>(res.to_backend_mut());
    let a_vznx = big_as_vec_znx_ref::<BE>(a.to_backend_ref());
    vec_znx_automorphism::<BE>(p, &mut res_vznx, res_col, &a_vznx, a_col);
}

pub fn vec_znx_big_automorphism_assign<R, BE>(p: i64, res: &mut R, res_col: usize, tmp: &mut [i64])
where
    BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxAutomorphism + ZnxCopy,
    for<'a> BE::BufMut<'a>: HostDataMut,
    R: VecZnxBigToBackendMut<BE>,
{
    let mut res_vznx = big_as_vec_znx_mut::<BE>(res.to_backend_mut());
    vec_znx_automorphism_assign::<BE>(p, &mut res_vznx, res_col, tmp);
}

pub fn vec_znx_big_negate<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxNegate + ZnxZero,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
{
    let mut res_vznx = big_as_vec_znx_mut::<BE>(res.to_backend_mut());
    let a_vznx = big_as_vec_znx_ref::<BE>(a.to_backend_ref());
    vec_znx_negate::<BE>(&mut res_vznx, res_col, &a_vznx, a_col);
}

pub fn vec_znx_big_negate_assign<R, BE>(res: &mut R, res_col: usize)
where
    BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxNegateAssign,
    for<'a> BE::BufMut<'a>: HostDataMut,
    R: VecZnxBigToBackendMut<BE>,
{
    let mut res_vznx = big_as_vec_znx_mut::<BE>(res.to_backend_mut());
    vec_znx_negate_assign::<BE>(&mut res_vznx, res_col);
}

pub fn vec_znx_big_normalize_tmp_bytes(n: usize) -> usize {
    vec_znx_normalize_tmp_bytes(n)
}

#[allow(clippy::too_many_arguments)]
pub fn vec_znx_big_normalize<R, A, BE>(
    res: &mut R,
    res_base2k: usize,
    res_k: usize,
    res_offset: i64,
    res_col: usize,
    a: &A,
    a_base2k: usize,
    a_col: usize,
    carry: &mut [i64],
) where
    R: VecZnxToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
    BE: Backend<BigWord = i64, ZnxWord = i64>
        + ZnxZero
        + ZnxCopy
        + ZnxAddAssign
        + ZnxMulPowerOfTwoAssign
        + ZnxNormalizeFirstStepCarryOnly
        + ZnxNormalizeMiddleStepCarryOnly
        + ZnxNormalizeMiddleStep
        + ZnxNormalizeFinalStep
        + ZnxNormalizeFirstStep
        + I64NormalizeOps
        + ZnxNormalizeDigit
        + ZnxNormalizeMiddleStepAssign
        + ZnxNormalizeFinalStepAssign,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
{
    let a_vznx = big_as_vec_znx_ref::<BE>(a.to_backend_ref());
    let mut res_ref = res.to_backend_mut();
    vec_znx_normalize::<BE>(
        &mut res_ref,
        res_base2k,
        res_k,
        res_offset,
        res_col,
        &a_vznx,
        a_base2k,
        a_col,
        carry,
    );
}

pub fn vec_znx_big_add_normal_ref<R, B>(
    base2k: usize,
    res: &mut R,
    res_col: usize,
    k: usize,
    sigma: f64,
    bound: f64,
    source: &mut Source,
) where
    B: Backend<BigWord = i64, ZnxWord = i64>,
    for<'a> B::BufMut<'a>: HostDataMut,
    R: VecZnxBigToBackendMut<B>,
{
    let mut res = res.to_backend_mut();
    assert!(
        (bound.log2().ceil() as i64) < 64,
        "invalid bound: ceil(log2(bound))={} > 63",
        (bound.log2().ceil() as i64)
    );

    let limb: usize = k.div_ceil(base2k) - 1;
    let shift: u32 = ((limb + 1) * base2k - k) as u32;
    znx_add_normal_f64_ref(res.at_mut(res_col, limb), sigma, bound, shift, source)
}

/// R <- A - B
pub fn vec_znx_big_sub<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxSub + ZnxNegate + ZnxZero + ZnxCopy,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
    B: VecZnxBigToBackendRef<BE>,
{
    let mut res_vznx = big_as_vec_znx_mut::<BE>(res.to_backend_mut());
    let a_vznx = big_as_vec_znx_ref::<BE>(a.to_backend_ref());
    let b_vznx = big_as_vec_znx_ref::<BE>(b.to_backend_ref());
    vec_znx_sub::<BE>(&mut res_vznx, res_col, &a_vznx, a_col, &b_vznx, b_col);
}

/// R <- A - B
pub fn vec_znx_big_sub_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxSubAssign,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
{
    let mut res_vznx = big_as_vec_znx_mut::<BE>(res.to_backend_mut());
    let a_vznx = big_as_vec_znx_ref::<BE>(a.to_backend_ref());
    vec_znx_sub_assign::<BE>(&mut res_vznx, res_col, &a_vznx, a_col);
}

/// R <- B - A
pub fn vec_znx_big_sub_negate_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxSubNegateAssign + ZnxNegateAssign,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
{
    let mut res_vznx = big_as_vec_znx_mut::<BE>(res.to_backend_mut());
    let a_vznx = big_as_vec_znx_ref::<BE>(a.to_backend_ref());
    vec_znx_sub_negate_assign::<BE>(&mut res_vznx, res_col, &a_vznx, a_col);
}

/// R <- A - B
pub fn vec_znx_big_sub_small_a<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxSub + ZnxNegate + ZnxZero + ZnxCopy,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxToBackendRef<BE>,
    B: VecZnxBigToBackendRef<BE>,
{
    let mut res_vznx = big_as_vec_znx_mut::<BE>(res.to_backend_mut());
    let b_vznx = big_as_vec_znx_ref::<BE>(b.to_backend_ref());
    let a_ref = a.to_backend_ref();
    vec_znx_sub::<BE>(&mut res_vznx, res_col, &a_ref, a_col, &b_vznx, b_col);
}

/// R <- A - B
pub fn vec_znx_big_sub_small_b<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxSub + ZnxNegate + ZnxZero + ZnxCopy,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxBigToBackendRef<BE>,
    B: VecZnxToBackendRef<BE>,
{
    let mut res_vznx = big_as_vec_znx_mut::<BE>(res.to_backend_mut());
    let a_vznx = big_as_vec_znx_ref::<BE>(a.to_backend_ref());
    let b_ref = b.to_backend_ref();
    vec_znx_sub::<BE>(&mut res_vznx, res_col, &a_vznx, a_col, &b_ref, b_col);
}

///  R <- R - A
pub fn vec_znx_big_sub_small_a_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxSubAssign,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxToBackendRef<BE>,
{
    let mut res_vznx = big_as_vec_znx_mut::<BE>(res.to_backend_mut());
    let a_ref = a.to_backend_ref();
    vec_znx_sub_assign::<BE>(&mut res_vznx, res_col, &a_ref, a_col);
}

/// R <- A - R
pub fn vec_znx_big_sub_small_b_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<BigWord = i64, ZnxWord = i64> + ZnxSubNegateAssign + ZnxNegateAssign,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    R: VecZnxBigToBackendMut<BE>,
    A: VecZnxToBackendRef<BE>,
{
    let mut res_vznx = big_as_vec_znx_mut::<BE>(res.to_backend_mut());
    let a_ref = a.to_backend_ref();
    vec_znx_sub_negate_assign::<BE>(&mut res_vznx, res_col, &a_ref, a_col);
}
