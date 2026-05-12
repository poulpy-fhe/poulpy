use std::vec;

use poulpy_core::layouts::ModuleCoreAlloc;
use poulpy_hal::api::ModuleN;

use crate::blind_rotation::{DivRound, LookUpTableLayout, LookupTable, LookupTableFactory};

pub fn test_lut_standard<M>(module: &M)
where
    M: LookupTableFactory + ModuleN + ModuleCoreAlloc<OwnedBuf = Vec<u8>>,
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

    let mut lut: LookupTable = LookupTable::alloc(module, &lut_infos);
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
    M: LookupTableFactory + ModuleN + ModuleCoreAlloc<OwnedBuf = Vec<u8>>,
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

    let mut lut: LookupTable = LookupTable::alloc(module, &lut_infos);
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
