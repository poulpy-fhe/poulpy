use crate::layouts::{Backend, HostDataMut, HostDataRef, VecZnxBackendMut, VecZnxBackendRef, ZnxView, ZnxViewMut};
use crate::reference::znx::{ZnxCopy, ZnxZero};

pub fn vec_znx_add_const_into<'r, 'a, BE>(
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_col: usize,
    cnst: &VecZnxBackendRef<'a, BE>,
    cnst_col: usize,
    cnst_coeff: usize,
    res_limb: usize,
    res_coeff: usize,
) where
    BE: Backend + ZnxCopy + ZnxZero,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    #[cfg(debug_assertions)]
    {
        assert!(res_coeff < res.n(), "res_coeff: {res_coeff} >= res.n(): {}", res.n());
        assert!(res_limb <= res.size(), "res_limb: {res_limb} > res.size(): {}", res.size());
    }

    let min_size = res.size().min(a.size());
    for limb in 0..min_size {
        BE::znx_copy(res.at_mut(res_col, limb), a.at(a_col, limb));
    }
    for limb in min_size..res.size() {
        BE::znx_zero(res.at_mut(res_col, limb));
    }
    vec_znx_add_const_assign::<BE>(res, res_col, cnst, cnst_col, cnst_coeff, res_limb, res_coeff);
}

pub fn vec_znx_add_const_assign<'r, 'a, BE>(
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    cnst: &VecZnxBackendRef<'a, BE>,
    cnst_col: usize,
    cnst_coeff: usize,
    res_limb: usize,
    res_coeff: usize,
) where
    BE: Backend,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    #[cfg(debug_assertions)]
    {
        assert!(res_coeff < res.n(), "res_coeff: {res_coeff} >= res.n(): {}", res.n());
        assert!(res_limb <= res.size(), "res_limb: {res_limb} > res.size(): {}", res.size());
        assert!(cnst_coeff < cnst.n(), "cnst_coeff: {cnst_coeff} >= cnst.n(): {}", cnst.n());
    }

    let digit_count = cnst.size().min(res.size().saturating_sub(res_limb));
    for idx in 0..digit_count {
        res.at_mut(res_col, res_limb + idx)[res_coeff] += cnst.at(cnst_col, idx)[cnst_coeff];
    }
}
