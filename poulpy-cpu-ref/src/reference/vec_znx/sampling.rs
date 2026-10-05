use crate::{
    layouts::{Backend, HostDataMut, VecZnxBackendMut, ZnxViewMut},
    reference::znx::znx_fill_uniform_ref,
    source::Source,
};

pub fn vec_znx_fill_uniform_ref<'r, BE>(
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
        znx_fill_uniform_ref(base2k, res.at_mut(res_col, j), source)
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

pub fn vec_znx_add_noise_ref<'r, BE>(
    base2k: usize,
    k: usize,
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    noise: poulpy_core::Noise,
    source: &mut Source,
) where
    BE: Backend<ZnxWord = i64>,
    BE::BufMut<'r>: HostDataMut,
{
    crate::reference::noise::add_noise(base2k, k, res, res_col, noise, source);
}
