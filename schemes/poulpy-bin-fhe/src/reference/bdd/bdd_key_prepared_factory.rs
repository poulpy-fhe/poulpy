use crate::{bdd_arithmetic::*, blind_rotation::BlindRotationAlgo, circuit_bootstrapping::*};
use poulpy_core::api::{GLWESwitchingKeyPreparedFactory, GLWEToLWEKeyPreparedFactory};
use poulpy_hal::layouts::*;
/// Independently callable canonical implementation of [`BDDKeyPreparedFactory::bdd_key_prepare_tmp_bytes`].
pub fn bdd_key_prepare_tmp_bytes_reference<BRA: BlindRotationAlgo, BE: Backend<ZnxWord = i64>, A>(
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
/// Independently callable canonical implementation of [`BDDKeyPreparedFactory::bdd_key_prepare`].
pub fn bdd_key_prepare_reference<BRA: BlindRotationAlgo, BE: Backend<ZnxWord = i64>>(
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

/// Reference bodies of the [`BDDKeyPreparedFactoryImpl`](crate::oep::BDDKeyPreparedFactoryImpl) methods, which
/// [`impl_bdd_key_prepared_factory_reference_full!`](crate::impl_bdd_key_prepared_factory_reference_full) forwards to.
pub trait BDDKeyPreparedFactoryReference<BRA: BlindRotationAlgo, BE: Backend> {
    fn bdd_key_prepare_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: BDDKeyInfos;

    fn bdd_key_prepare_reference(
        &self,
        res: &mut BDDKeyPrepared<BE::OwnedBuf, BRA, BE>,
        other: &BDDKey<BE::OwnedBuf, BRA, BE::ZnxWord>,
        scratch: &mut ScratchArena<'_, BE>,
    );
}

impl<BRA: BlindRotationAlgo, BE: Backend<ZnxWord = i64>> BDDKeyPreparedFactoryReference<BRA, BE> for Module<BE>
where
    Module<BE>: CircuitBootstrappingKeyPreparedFactory<BRA, BE> + GLWEToLWEKeyPreparedFactory<BE>,
{
    fn bdd_key_prepare_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: BDDKeyInfos,
    {
        bdd_key_prepare_tmp_bytes_reference::<BRA, BE, A>(self, infos)
    }

    fn bdd_key_prepare_reference(
        &self,
        res: &mut BDDKeyPrepared<BE::OwnedBuf, BRA, BE>,
        other: &BDDKey<BE::OwnedBuf, BRA, BE::ZnxWord>,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        bdd_key_prepare_reference(self, res, other, scratch)
    }
}
