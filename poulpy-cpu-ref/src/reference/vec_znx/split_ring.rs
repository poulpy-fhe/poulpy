use std::mem::size_of;

use crate::{
    layouts::{Backend, HostDataMut, HostDataRef, VecZnxBackendMut, VecZnxBackendRef, ZnxView, ZnxViewMut},
    reference::znx::{ZnxRotate, ZnxSwitchRing, ZnxZero},
};

pub fn vec_znx_split_ring_tmp_bytes(n: usize) -> usize {
    n * size_of::<i64>()
}

pub fn vec_znx_split_ring<'r, 'a, BE>(
    res: &mut [VecZnxBackendMut<'r, BE>],
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_col: usize,
    tmp: &mut [i64],
) where
    BE: Backend + ZnxSwitchRing + ZnxRotate + ZnxZero,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    let a_size = a.size();
    let (_n_in, _n_out) = (a.n(), res[0].n());

    #[cfg(debug_assertions)]
    {
        assert_eq!(tmp.len(), a.n());
        assert!(_n_out < _n_in, "invalid a: output ring degree should be smaller");

        res[1..]
            .iter_mut()
            .for_each(|bi| assert_eq!(bi.n(), _n_out, "invalid input a: all VecZnx must have the same degree"));

        assert!(_n_in.is_multiple_of(_n_out));
        assert_eq!(res.len(), _n_in / _n_out);
    }

    res.iter_mut().enumerate().for_each(|(i, bi)| {
        let min_size = bi.size().min(a_size);

        if i == 0 {
            for j in 0..min_size {
                BE::znx_switch_ring(bi.at_mut(res_col, j), a.at(a_col, j));
            }
        } else {
            for j in 0..min_size {
                BE::znx_rotate(-(i as i64), tmp, a.at(a_col, j));
                BE::znx_switch_ring(bi.at_mut(res_col, j), tmp);
            }
        }

        for j in min_size..bi.size() {
            BE::znx_zero(bi.at_mut(res_col, j));
        }
    })
}
