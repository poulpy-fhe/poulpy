use std::mem::size_of;

use crate::{
    layouts::{Backend, HostDataMut, HostDataRef, VecZnxBackendMut, VecZnxBackendRef},
    reference::{
        vec_znx::{vec_znx_rotate_assign, vec_znx_switch_ring},
        znx::{ZnxCopy, ZnxRotate, ZnxSwitchRing, ZnxZero},
    },
};

pub fn vec_znx_merge_rings_tmp_bytes(n: usize) -> usize {
    n * size_of::<i64>()
}

pub fn vec_znx_merge_rings<'r, 'a, BE>(
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    a: &[VecZnxBackendRef<'a, BE>],
    a_col: usize,
    tmp: &mut [i64],
) where
    BE: Backend + ZnxCopy + ZnxSwitchRing + ZnxRotate + ZnxZero,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    let (_n_out, _n_in) = (res.n(), a[0].n());

    #[cfg(debug_assertions)]
    {
        assert_eq!(tmp.len(), res.n());
        debug_assert!(_n_out > _n_in, "invalid a: output ring degree should be greater");

        a[1..]
            .iter()
            .for_each(|ai| debug_assert_eq!(ai.n(), _n_in, "invalid input a: all VecZnx must have the same degree"));

        assert!(_n_out.is_multiple_of(_n_in));
        assert_eq!(a.len(), _n_out / _n_in);
    }

    a.iter().for_each(|ai| {
        vec_znx_switch_ring::<BE>(res, res_col, ai, a_col);
        vec_znx_rotate_assign::<BE>(-1, res, res_col, tmp);
    });

    vec_znx_rotate_assign::<BE>(a.len() as i64, res, res_col, tmp);
}
