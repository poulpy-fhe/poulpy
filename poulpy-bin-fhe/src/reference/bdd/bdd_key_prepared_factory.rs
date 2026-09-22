use crate::{bdd_arithmetic::*, blind_rotation::BlindRotationAlgo, circuit_bootstrapping::*};
use poulpy_core::layouts::*;
use poulpy_hal::layouts::*;
#[allow(clippy::too_many_arguments)]
/// Independently callable canonical implementation of [`BDDKeyPreparedFactory::alloc_bdd_key_from_infos`].
pub fn alloc_bdd_key_from_infos_reference<BRA: BlindRotationAlgo, BE: Backend<ZnxWord = i64> + 'static, A>(
    module: &Module<BE>,
    infos: &A,
) -> BDDKeyPrepared<BE::OwnedBuf, BRA, BE>
where
    A: BDDKeyInfos,
    Module<BE>: Sized + CircuitBootstrappingKeyPreparedFactory<BRA, BE> + GLWEToLWEKeyPreparedFactory<BE>,
{
    let ks_glwe = if let Some(ks_glwe_infos) = &infos.ks_glwe_infos() {
        Some(module.glwe_switching_key_prepared_alloc_from_infos(ks_glwe_infos))
    } else {
        None
    };

    BDDKeyPrepared {
        cbt: CircuitBootstrappingKeyPrepared::alloc_from_infos(module, &infos.cbt_infos()),
        ks_glwe,
        ks_lwe: module.glwe_to_lwe_key_prepared_alloc_from_infos(&infos.ks_lwe_infos()),
    }
}
#[allow(clippy::too_many_arguments)]
/// Independently callable canonical implementation of [`BDDKeyPreparedFactory::prepare_bdd_key_tmp_bytes`].
pub fn prepare_bdd_key_tmp_bytes_reference<BRA: BlindRotationAlgo, BE: Backend<ZnxWord = i64> + 'static, A>(
    module: &Module<BE>,
    infos: &A,
) -> usize
where
    A: BDDKeyInfos,
    Module<BE>: Sized + CircuitBootstrappingKeyPreparedFactory<BRA, BE> + GLWEToLWEKeyPreparedFactory<BE>,
{
    module
        .circuit_bootstrapping_key_prepare_tmp_bytes(&infos.cbt_infos())
        .max(module.glwe_to_lwe_key_prepare_tmp_bytes(&infos.ks_lwe_infos()))
        .max(
            infos
                .ks_glwe_infos()
                .map_or(0, |key| module.glwe_switching_key_prepare_tmp_bytes(&key)),
        )
}
#[allow(clippy::too_many_arguments)]
/// Independently callable canonical implementation of [`BDDKeyPreparedFactory::prepare_bdd_key`].
pub fn prepare_bdd_key_reference<BRA: BlindRotationAlgo, BE: Backend<ZnxWord = i64> + 'static>(
    module: &Module<BE>,
    res: &mut BDDKeyPrepared<BE::OwnedBuf, BRA, BE>,
    other: &BDDKey<BE::OwnedBuf, BRA, BE::ZnxWord>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    Module<BE>: Sized + CircuitBootstrappingKeyPreparedFactory<BRA, BE> + GLWEToLWEKeyPreparedFactory<BE>,
{
    res.cbt.prepare(module, &other.cbt, scratch);

    if let Some(key_prep) = &mut res.ks_glwe {
        if let Some(other) = &other.ks_glwe {
            module.glwe_switching_key_prepare(key_prep, other, scratch);
        } else {
            panic!("incompatible keys: res has Some(ks_glwe) but other has none")
        }
    }

    module.glwe_to_lwe_key_prepare(&mut res.ks_lwe, &other.ks_lwe, scratch);
}
