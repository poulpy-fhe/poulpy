use std::vec;

use poulpy_core::api::TransferInto;
use poulpy_core::layouts::ModuleCoreAlloc;
use poulpy_hal::api::ModuleN;

use crate::blind_rotation::{DivRound, LookUpTableLayout, LookUpTableRotationDirection, LookupTable, LookupTableFactory};

pub fn test_lut_standard<M>(module: &M)
where
    M: LookupTableFactory<Vec<u8>, i64> + ModuleN + ModuleCoreAlloc<OwnedBuf = Vec<u8>, ZnxWord = i64>,
{
    let base2k: usize = 20;
    let k_lut: usize = 40;
    let message_modulus: usize = 16;
    let extension_factor: usize = 1;

    let log_scale: usize = base2k + 1;

    let mut f: Vec<i64> = vec![0i64; message_modulus];
    f.iter_mut().enumerate().for_each(|(i, x)| *x = (i as i64) - 8);

    let lut_infos: LookUpTableLayout = LookUpTableLayout {
        n: module.n().into(),
        extension_factor,
        k: k_lut.into(),
        base2k: base2k.into(),
    };

    let mut lut: LookupTable<Vec<u8>, i64> = LookupTable::alloc(module, &lut_infos);
    lut.set(module, &f, log_scale);

    let half_step: i64 = lut.domain_size().div_round(message_modulus << 1) as i64;
    lut.rotate(module, half_step);

    let step: usize = lut.domain_size().div_round(message_modulus);

    let mut lut_dec: Vec<i64> = vec![0i64; module.n()];
    lut.data[0].data().decode_vec_i64(base2k, 0, log_scale, &mut lut_dec);

    (0..lut.domain_size()).step_by(step).for_each(|i| {
        (0..step).for_each(|_| {
            assert_eq!(f[i / step] % message_modulus as i64, lut_dec[i]);
        });
    });
}

pub fn test_lut_extended<M>(module: &M)
where
    M: LookupTableFactory<Vec<u8>, i64> + ModuleN + ModuleCoreAlloc<OwnedBuf = Vec<u8>, ZnxWord = i64>,
{
    let base2k: usize = 20;
    let k_lut: usize = 40;
    let message_modulus: usize = 16;
    let extension_factor: usize = 4;

    let log_scale: usize = base2k + 1;

    let mut f: Vec<i64> = vec![0i64; message_modulus];
    f.iter_mut().enumerate().for_each(|(i, x)| *x = (i as i64) - 8);

    let lut_infos: LookUpTableLayout = LookUpTableLayout {
        n: module.n().into(),
        extension_factor,
        k: k_lut.into(),
        base2k: base2k.into(),
    };

    let mut lut: LookupTable<Vec<u8>, i64> = LookupTable::alloc(module, &lut_infos);
    lut.set(module, &f, log_scale);

    let half_step: i64 = lut.domain_size().div_round(message_modulus << 1) as i64;
    lut.rotate(module, half_step);

    let step: usize = module.n().div_round(message_modulus);

    let mut lut_dec: Vec<i64> = vec![0i64; module.n()];

    (0..extension_factor).for_each(|ext| {
        lut.data[ext].data().decode_vec_i64(base2k, 0, log_scale, &mut lut_dec);
        (0..module.n()).step_by(step).for_each(|i| {
            (0..step).for_each(|_| {
                assert_eq!(f[i / step] % message_modulus as i64, lut_dec[i]);
            });
        });
    });
}

/// The upload carries `drift`, which `set` makes non-zero and `alloc` leaves at 0.
pub fn test_lut_transfer_into_carries_scalars<M>(module: &M)
where
    M: LookupTableFactory<Vec<u8>, i64> + ModuleN + ModuleCoreAlloc<OwnedBuf = Vec<u8>, ZnxWord = i64>,
{
    let infos = LookUpTableLayout {
        n: module.n().into(),
        extension_factor: 2,
        k: 40usize.into(),
        base2k: 20usize.into(),
    };

    let mut src: LookupTable<Vec<u8>, i64> = LookupTable::alloc(module, &infos);
    let f: Vec<i64> = (0..8).map(|i| i - 4).collect();
    src.set(module, &f, 21);
    src.set_rotation_direction(LookUpTableRotationDirection::Right);
    assert_ne!(src.drift, 0);

    let mut dst: LookupTable<Vec<u8>, i64> = LookupTable::alloc(module, &infos);
    src.transfer_into(&mut dst);

    assert_eq!(dst.drift, src.drift);
    assert!(matches!(dst.rot_dir, LookUpTableRotationDirection::Right));
    for (a, b) in src.data.iter().zip(dst.data.iter()) {
        assert_eq!(a, b);
    }
}
