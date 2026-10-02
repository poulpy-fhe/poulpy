use crate::blind_rotation::host_znx::{znx_rotate, znx_switch_ring};
use crate::blind_rotation::{DivRound, LookupTable};
use poulpy_core::layouts::LWEInfos;
use poulpy_hal::AlignedBuf;
use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalizeAssign, VecZnxNormalizeTmpBytes, VecZnxRotateAssign,
        VecZnxRotateAssignTmpBytes,
    },
    layouts::{Backend, Module, ScratchOwned, VecZnx, VecZnxToBackendMut, ZnxViewMut, vec_znx_host_backend_mut},
};

/// Encodes host function samples using the canonical LUT layout and HAL operations.
///
/// The destination must have compatible dimensions and sufficient precision for
/// the message-bit count `k`. Rotation of the encoded samples uses the canonical
/// LUT rotation composition.
pub fn lookup_table_set_ref<BE>(module: &Module<BE>, res: &mut LookupTable<BE::OwnedBuf, BE::ZnxWord>, f: &[i64], k: usize)
where
    BE: Backend<ZnxWord = i64>,
    Module<BE>: VecZnxNormalizeAssign<BE> + VecZnxNormalizeTmpBytes + VecZnxRotateAssign<BE> + VecZnxRotateAssignTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    assert!(f.len() <= module.n());

    let base2k: usize = res.base2k.into();

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.vec_znx_normalize_tmp_bytes().max(res.domain_size() << 3));

    // Get the number minimum limb to store the message modulus
    let limbs: usize = k.div_ceil(base2k);

    #[cfg(debug_assertions)]
    {
        assert!(
            (max_bit_size(f) + (k % base2k) as u32) < i64::BITS,
            "overflow: max(|f|) << (k%base2k) > i64::BITS"
        );
        assert!(limbs <= res.data[0].size());
    }

    // Scaling factor
    let mut scale = 1;
    if !k.is_multiple_of(base2k) {
        scale <<= base2k - (k % base2k);
    }

    // #elements in lookup table
    let f_len: usize = f.len();

    // If LUT size > TakeScalarZnx
    let domain_size: usize = res.domain_size();

    let size: usize = res.k.as_usize().div_ceil(base2k);

    // Equivalent to AUTO([f(0), -f(n-1), -f(n-2), ..., -f(1)], -1)
    // but staged purely in host vectors so we do not need to allocate a
    // second module with ring degree `domain_size`.
    let mut lut_full_limbs: Vec<Vec<i64>> = (0..size).map(|_| vec![0i64; domain_size]).collect();

    let lut_at: &mut [i64] = &mut lut_full_limbs[limbs - 1];

    let step: usize = domain_size.div_round(f_len);

    for (i, fi) in f.iter().enumerate() {
        let start: usize = i * step;
        let end: usize = start + step;
        lut_at[start..end].fill(fi * scale);
    }

    let drift: usize = step >> 1;

    // Rotates half the step to the left
    if res.extension_factor() > 1 {
        let mut tmp: Vec<i64> = vec![0i64; domain_size];

        for i in 0..res.extension_factor() {
            let mut host = VecZnx::<AlignedBuf, i64>::from_bytes(
                res.data[i].n().as_usize(),
                1,
                res.data[i].size(),
                poulpy_hal::alloc_aligned::<u8>(VecZnx::<AlignedBuf, i64>::bytes_of(
                    res.data[i].n().as_usize(),
                    1,
                    res.data[i].size(),
                )),
            );
            {
                let mut res_at = vec_znx_host_backend_mut(&mut host);
                for (limb, limb_data) in lut_full_limbs.iter().enumerate().take(res_at.size()) {
                    znx_switch_ring(res_at.at_mut(0, limb), limb_data);
                }
            }
            BE::copy_from_host(res.data[i].data_mut().data_mut(), host.data());
            if i + 1 < res.extension_factor() {
                for limb_data in &mut lut_full_limbs {
                    znx_rotate(-1, &mut tmp, limb_data);
                    limb_data.copy_from_slice(&tmp);
                }
            }
        }
    } else {
        let mut host = VecZnx::<AlignedBuf, i64>::from_bytes(
            res.data[0].n().as_usize(),
            1,
            res.data[0].size(),
            poulpy_hal::alloc_aligned::<u8>(VecZnx::<AlignedBuf, i64>::bytes_of(
                res.data[0].n().as_usize(),
                1,
                res.data[0].size(),
            )),
        );
        {
            let mut res_at = vec_znx_host_backend_mut(&mut host);
            for (limb, limb_data) in lut_full_limbs.iter().enumerate().take(res_at.size()) {
                res_at.at_mut(0, limb).copy_from_slice(limb_data);
            }
        }
        BE::copy_from_host(res.data[0].data_mut().data_mut(), host.data());
    }

    for a in res.data.iter_mut() {
        let mut a_data = <VecZnx<BE::OwnedBuf, BE::ZnxWord> as VecZnxToBackendMut<BE>>::to_backend_mut(a.data_mut());
        module.vec_znx_normalize_assign(res.base2k.into(), res.k.as_usize(), 0, &mut a_data, 0, &mut scratch.borrow());
    }

    lookup_table_rotate_ref(module, -(drift as i64), res);

    res.drift = drift
}

/// Rotates an encoded LUT using HAL polynomial rotation and its scratch query.
pub fn lookup_table_rotate_ref<BE>(module: &Module<BE>, k: i64, res: &mut LookupTable<BE::OwnedBuf, BE::ZnxWord>)
where
    BE: Backend<ZnxWord = i64>,
    Module<BE>: VecZnxRotateAssign<BE> + VecZnxRotateAssignTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let extension_factor: usize = res.extension_factor();
    let two_n: usize = 2 * res.data[0].n().as_usize();
    let two_n_ext: usize = two_n * extension_factor;

    let mut scratch: ScratchOwned<_> = ScratchOwned::alloc(module.vec_znx_rotate_assign_tmp_bytes());

    let k_pos: usize = ((k + two_n_ext as i64) % two_n_ext as i64) as usize;

    let k_hi: usize = k_pos / extension_factor;
    let k_lo: usize = k_pos % extension_factor;

    (0..extension_factor - k_lo).for_each(|i| {
        let mut data: poulpy_hal::layouts::VecZnxBackendMut<'_, BE> =
            <poulpy_hal::layouts::VecZnx<BE::OwnedBuf, BE::ZnxWord> as VecZnxToBackendMut<BE>>::to_backend_mut(
                res.data[i].data_mut(),
            );
        module.vec_znx_rotate_assign(k_hi as i64, &mut data, 0, &mut scratch.borrow());
    });

    (extension_factor - k_lo..extension_factor).for_each(|i| {
        let mut data: poulpy_hal::layouts::VecZnxBackendMut<'_, BE> =
            <poulpy_hal::layouts::VecZnx<BE::OwnedBuf, BE::ZnxWord> as VecZnxToBackendMut<BE>>::to_backend_mut(
                res.data[i].data_mut(),
            );
        module.vec_znx_rotate_assign(k_hi as i64 + 1, &mut data, 0, &mut scratch.borrow());
    });

    res.data.rotate_right(k_lo);
}

#[allow(dead_code)]
fn max_bit_size(vec: &[i64]) -> u32 {
    vec.iter()
        .map(|&v| if v == 0 { 0 } else { v.unsigned_abs().ilog2() + 1 })
        .max()
        .unwrap_or(0)
}
