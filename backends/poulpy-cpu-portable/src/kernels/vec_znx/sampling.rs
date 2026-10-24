use crate::{
    kernels::{noise::add_noise_portable, znx::znx_fill_uniform_portable},
    layouts::{Backend, HostDataMut, VecZnxBackendMut, ZnxViewMut},
    source::Source,
};
use poulpy_core::Noise;

pub fn vec_znx_fill_uniform_portable<'r, BE>(
    base2k: usize,
    k: usize,
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    source: &mut Source,
) where
    BE: Backend<ZnxWord = i64>,
    BE::BufMut<'r>: HostDataMut,
{
    assert!(k != 0, "uniform sampling precision must be non-zero");
    let size = k.div_ceil(base2k);
    assert!(size <= res.size(), "k ({k}) exceeds the allocation ({} limbs)", res.size());

    for j in 0..size {
        znx_fill_uniform_portable(base2k, res.at_mut(res_col, j), source)
    }

    let rem = k % base2k;
    if rem != 0 {
        let mask = (!0i64) << (base2k - rem);
        res.at_mut(res_col, size - 1).iter_mut().for_each(|value| *value &= mask);
    }

    for j in size..res.size() {
        res.at_mut(res_col, j).fill(0);
    }
}

pub fn vec_znx_add_noise_portable<'r, BE>(
    base2k: usize,
    k: usize,
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    noise: Noise,
    source: &mut Source,
) where
    BE: Backend<ZnxWord = i64>,
    BE::BufMut<'r>: HostDataMut,
{
    add_noise_portable(base2k, k, res, res_col, noise, source);
}
