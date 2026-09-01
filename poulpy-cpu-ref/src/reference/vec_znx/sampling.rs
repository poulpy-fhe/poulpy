use crate::{
    layouts::{Backend, HostDataMut, NoiseInfos, NormalizationState, Unnormalized, VecZnxBackendMut, ZnxViewMut},
    reference::znx::{znx_add_normal_f64_ref, znx_fill_normal_f64_ref, znx_fill_uniform_ref},
    source::Source,
};

pub fn vec_znx_fill_uniform_ref<'r, BE>(
    base2k: usize,
    k: usize,
    res: &mut VecZnxBackendMut<'r, BE, impl NormalizationState>,
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

pub fn vec_znx_fill_normal_ref<'r, BE>(
    base2k: usize,
    res: &mut VecZnxBackendMut<'r, BE, Unnormalized>,
    res_col: usize,
    noise_infos: NoiseInfos,
    source: &mut Source,
) where
    BE: Backend<ZnxWord = i64>,
    BE::BufMut<'r>: HostDataMut,
{
    assert!(
        (noise_infos.bound.log2().ceil() as i64) < 64,
        "invalid bound: ceil(log2(bound))={} > 63",
        (noise_infos.bound.log2().ceil() as i64)
    );

    let (limb, shift) = noise_infos.target_limb_and_shift(base2k);
    znx_fill_normal_f64_ref(res.at_mut(res_col, limb), noise_infos.sigma, noise_infos.bound, shift, source)
}

pub fn vec_znx_add_normal_ref<'r, BE>(
    base2k: usize,
    res: &mut VecZnxBackendMut<'r, BE, Unnormalized>,
    res_col: usize,
    noise_infos: NoiseInfos,
    source: &mut Source,
) where
    BE: Backend<ZnxWord = i64>,
    BE::BufMut<'r>: HostDataMut,
{
    assert!(
        (noise_infos.bound.log2().ceil() as i64) < 64,
        "invalid bound: ceil(log2(bound))={} > 63",
        (noise_infos.bound.log2().ceil() as i64)
    );

    let (limb, shift) = noise_infos.target_limb_and_shift(base2k);
    znx_add_normal_f64_ref(res.at_mut(res_col, limb), noise_infos.sigma, noise_infos.bound, shift, source)
}
