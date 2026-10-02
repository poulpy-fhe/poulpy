use crate::{bdd_arithmetic::*, blind_rotation::BlindRotationAlgo};
use poulpy_hal::layouts::*;
impl<BRA: BlindRotationAlgo, BE: Backend> BDDKeyPreparedFactory<BRA, BE> for Module<BE>
where
    BE: crate::oep::BDDKeyPreparedFactoryImpl<BRA>,
{
    #[allow(clippy::too_many_arguments)]
    fn alloc_bdd_key_from_infos<A>(&self, infos: &A) -> BDDKeyPrepared<BE::OwnedBuf, BRA, BE>
    where
        A: BDDKeyInfos,
    {
        BE::alloc_bdd_key_from_infos::<A>(self, infos)
    }
    #[allow(clippy::too_many_arguments)]
    fn prepare_bdd_key_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: BDDKeyInfos,
    {
        BE::prepare_bdd_key_tmp_bytes::<A>(self, infos)
    }
    #[allow(clippy::too_many_arguments)]
    fn prepare_bdd_key(
        &self,
        res: &mut BDDKeyPrepared<BE::OwnedBuf, BRA, BE>,
        other: &BDDKey<BE::OwnedBuf, BRA, BE::ZnxWord>,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        BE::prepare_bdd_key(self, res, other, scratch)
    }
}
