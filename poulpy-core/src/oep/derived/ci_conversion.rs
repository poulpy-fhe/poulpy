//! Degree switch expressed by composing HAL operations.
use poulpy_hal::{api::VecZnxSwitchRing, layouts::Backend};

use crate::layouts::{GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos};

/// Writes each column of `a` into `res` at the degree of `res`: `X -> X^(N_res / N_a)`
/// when it is larger, the coefficients at multiples of the ratio when it is smaller.
/// `res` keeps the canonical flag of `a`.
pub(crate) fn glwe_switch_degree_derived<BE, M, R, A>(module: &M, res: &mut R, a: &A)
where
    BE: Backend,
    M: VecZnxSwitchRing<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
{
    res.set_canonical(a.is_canonical());
    let mut res = res.to_backend_mut();
    let a = a.to_backend_ref();
    assert_eq!(res.rank(), a.rank());
    assert_eq!(res.base2k(), a.base2k());
    for col in 0..res.rank().as_usize() + 1 {
        module.vec_znx_switch_ring(&mut res.data, col, &a.data, col);
    }
}
