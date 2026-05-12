use crate::{
    layouts::{Backend, HostDataMut, HostDataRef, VecZnxBackendMut, VecZnxBackendRef, ZnxView, ZnxViewMut},
    reference::{
        vec_znx::vec_znx_copy,
        znx::{ZnxCopy, ZnxSwitchRing, ZnxZero},
    },
};

/// Maps between negacyclic rings by changing the polynomial degree.
/// Up:  Z\[X\]/(X^N+1) -> Z\[X\]/(X^{2^d N}+1) via X -> X^{2^d}
/// Down: Z\[X\]/(X^N+1) -> Z\[X\]/(X^{N/2^d}+1) by folding indices.
pub fn vec_znx_switch_ring<'r, 'a, BE>(
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_col: usize,
) where
    BE: Backend + ZnxCopy + ZnxSwitchRing + ZnxZero,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    let (n_in, n_out) = (a.n(), res.n());

    if n_in == n_out {
        vec_znx_copy::<BE>(res, res_col, a, a_col);
        return;
    }

    let min_size: usize = a.size().min(res.size());

    for j in 0..min_size {
        BE::znx_switch_ring(res.at_mut(res_col, j), a.at(a_col, j));
    }

    for j in min_size..res.size() {
        BE::znx_zero(res.at_mut(res_col, j));
    }
}
