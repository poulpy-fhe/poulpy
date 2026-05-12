use crate::{
    layouts::{Backend, HostDataMut, HostDataRef, VecZnxBackendMut, VecZnxBackendRef, ZnxView, ZnxViewMut},
    reference::znx::{ZnxCopy, ZnxZero},
};

pub fn vec_znx_copy<'r, 'a, BE>(res: &mut VecZnxBackendMut<'r, BE>, res_col: usize, a: &VecZnxBackendRef<'a, BE>, a_col: usize)
where
    BE: Backend + ZnxCopy + ZnxZero,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.n(), a.n())
    }

    let res_size = res.size();
    let a_size = a.size();

    let min_size = res_size.min(a_size);

    for j in 0..min_size {
        BE::znx_copy(res.at_mut(res_col, j), a.at(a_col, j));
    }

    for j in min_size..res_size {
        BE::znx_zero(res.at_mut(res_col, j));
    }
}

pub fn vec_znx_extract_coeff<'r, 'a, BE>(
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_col: usize,
    a_coeff: usize,
) where
    BE: Backend,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(
            res.n(),
            1,
            "vec_znx_extract_coeff expects a 1-coeff destination, got {}",
            res.n()
        );
        assert!(a_coeff < a.n(), "a_coeff: {a_coeff} >= a.n(): {}", a.n());
    }

    let min_size = res.size().min(a.size());

    for limb in 0..min_size {
        let dst = res.at_mut(res_col, limb);
        dst.fill(0);
        dst[0] = a.at(a_col, limb)[a_coeff];
    }

    for limb in min_size..res.size() {
        res.at_mut(res_col, limb).fill(0);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn vec_znx_copy_range<'r, 'a, BE>(
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    res_limb: usize,
    res_offset: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_col: usize,
    a_limb: usize,
    a_offset: usize,
    len: usize,
) where
    BE: Backend + ZnxCopy,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    assert!(res_offset + len <= res.n());
    assert!(a_offset + len <= a.n());

    BE::znx_copy(
        &mut res.at_mut(res_col, res_limb)[res_offset..res_offset + len],
        &a.at(a_col, a_limb)[a_offset..a_offset + len],
    );
}
