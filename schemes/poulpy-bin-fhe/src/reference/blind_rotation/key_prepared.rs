#![allow(clippy::too_many_arguments)]
use poulpy_hal::{AlignedBuf, alloc_aligned};
use poulpy_hal::{
    api::{SvpPPolAlloc, SvpPPolPrepare},
    layouts::{Backend, Module, PrepareHint, ScalarZnx, ScratchArena, SvpPPolOwned},
};

use poulpy_core::{Distribution, api::GGSWPreparedFactory, layouts::LWEInfos};

use crate::blind_rotation::{
    BlindRotationAlgo, BlindRotationKey, BlindRotationKeyInfos, BlindRotationKeyPrepared, CGGI, utils::set_xai_plus_y,
};

/// Independently callable canonical implementation of
/// [`BlindRotationKeyPreparedFactory::blind_rotation_key_prepare_tmp_bytes`](crate::api::BlindRotationKeyPreparedFactory::blind_rotation_key_prepare_tmp_bytes), selected by
/// `impl_blind_rotation_key_prepared_factory_reference_full!`.
pub fn blind_rotation_key_prepare_tmp_bytes_reference<BE, A>(module: &Module<BE>, infos: &A) -> usize
where
    A: BlindRotationKeyInfos,
    BE: Backend<ZnxWord = i64>,
    Module<BE>: GGSWPreparedFactory<BE> + SvpPPolAlloc<BE> + SvpPPolPrepare<BE>,
{
    module.ggsw_prepare_tmp_bytes(infos)
}

/// Independently callable canonical implementation of
/// [`BlindRotationKeyPreparedFactory::blind_rotation_key_prepare`](crate::api::BlindRotationKeyPreparedFactory::blind_rotation_key_prepare), selected by
/// `impl_blind_rotation_key_prepared_factory_reference_full!`.
pub fn blind_rotation_key_prepare_reference<BE>(
    module: &Module<BE>,
    res: &mut BlindRotationKeyPrepared<BE::OwnedBuf, CGGI, BE>,
    other: &BlindRotationKey<BE::OwnedBuf, CGGI, BE::ZnxWord>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend<ZnxWord = i64>,
    Module<BE>: GGSWPreparedFactory<BE> + SvpPPolAlloc<BE> + SvpPPolPrepare<BE>,
{
    assert_eq!(res.data.len(), other.keys.len(), "blind-rotation key dimensions differ");

    let n: usize = other.n().as_usize();

    for (a, b) in res.data.iter_mut().zip(other.keys.iter()) {
        module.ggsw_prepare(a, b, &mut scratch.borrow());
    }

    res.dist = other.dist;
    res.x_pow_a = None;

    if let Distribution::BinaryBlock(_) = other.dist {
        let mut x_pow_a: Vec<SvpPPolOwned<BE>> = Vec::with_capacity(n << 1);
        let mut buf: ScalarZnx<AlignedBuf, i64> =
            ScalarZnx::from_data(alloc_aligned::<u8>(ScalarZnx::<AlignedBuf, i64>::bytes_of(n, 1)), n, 1);
        (0..n << 1).for_each(|i| {
            let mut res: SvpPPolOwned<BE> = module.svp_ppol_alloc(module.n(), 1, PrepareHint::Reuse);
            set_xai_plus_y(module, i, 0, &mut res, &mut buf);
            x_pow_a.push(res);
        });
        res.x_pow_a = Some(x_pow_a);
    }
}

/// Reference bodies of the [`BlindRotationKeyPreparedFactoryImpl`](crate::oep::BlindRotationKeyPreparedFactoryImpl)
/// methods for [`CGGI`], which
/// [`impl_blind_rotation_key_prepared_factory_reference_full!`](crate::impl_blind_rotation_key_prepared_factory_reference_full)
/// forwards to.
pub trait BlindRotationKeyPreparedFactoryReference<BRA: BlindRotationAlgo, BE: Backend> {
    fn blind_rotation_key_prepare_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: BlindRotationKeyInfos;

    fn blind_rotation_key_prepare_reference(
        &self,
        res: &mut BlindRotationKeyPrepared<BE::OwnedBuf, BRA, BE>,
        other: &BlindRotationKey<BE::OwnedBuf, BRA, BE::ZnxWord>,
        scratch: &mut ScratchArena<'_, BE>,
    );
}

impl<BE: Backend<ZnxWord = i64>> BlindRotationKeyPreparedFactoryReference<CGGI, BE> for Module<BE>
where
    Module<BE>: GGSWPreparedFactory<BE> + SvpPPolAlloc<BE> + SvpPPolPrepare<BE>,
{
    fn blind_rotation_key_prepare_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: BlindRotationKeyInfos,
    {
        blind_rotation_key_prepare_tmp_bytes_reference(self, infos)
    }

    fn blind_rotation_key_prepare_reference(
        &self,
        res: &mut BlindRotationKeyPrepared<BE::OwnedBuf, CGGI, BE>,
        other: &BlindRotationKey<BE::OwnedBuf, CGGI, BE::ZnxWord>,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        blind_rotation_key_prepare_reference(self, res, other, scratch)
    }
}
