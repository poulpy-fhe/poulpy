use crate::{bdd_arithmetic::*, blind_rotation::BlindRotationAlgo};
use poulpy_hal::layouts::*;
impl<BRA: BlindRotationAlgo, BE: Backend> BDDKeyPreparedFactory<BRA, BE> for Module<BE>
where
    BE: crate::oep::BDDKeyPreparedFactoryImpl<BRA>,
{
    fn bdd_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: BDDKeyInfos,
    {
        BE::bdd_key_prepare_tmp_bytes::<A>(self, infos)
    }
    fn bdd_key_prepare(
        &self,
        res: &mut BDDKeyPrepared<BE::OwnedBuf, BRA, BE>,
        other: &BDDKey<BE::OwnedBuf, BRA, BE::ZnxWord>,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        BE::bdd_key_prepare(self, res, other, scratch)
    }
}
