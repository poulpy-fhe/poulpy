#![allow(clippy::too_many_arguments)]
use poulpy_hal::AlignedBuf;
use poulpy_hal::{
    api::{SvpPPolAlloc, SvpPrepare},
    layouts::{Backend, HostBytesBackend, Module, PrepareHint, ScalarZnx, ScratchArena, SvpPPolOwned},
};

use std::marker::PhantomData;

use poulpy_core::{
    Distribution,
    layouts::{GGSWPreparedFactory, LWEInfos},
};

use crate::blind_rotation::{BlindRotationKey, BlindRotationKeyInfos, BlindRotationKeyPrepared, CGGI, utils::set_xai_plus_y};

/// Canonical lower-layer composition for `blind_rotation_key_prepared_alloc`.
pub fn blind_rotation_key_prepared_alloc_ref<BE, A>(
    module: &Module<BE>,
    infos: &A,
) -> BlindRotationKeyPrepared<BE::OwnedBuf, CGGI, BE>
where
    A: BlindRotationKeyInfos,
    BE: Backend<ZnxWord = i64>,
    Module<BE>: GGSWPreparedFactory<BE> + SvpPPolAlloc<BE> + SvpPrepare<BE>,
{
    BlindRotationKeyPrepared {
        data: (0..infos.n_lwe().as_usize())
            .map(|_| module.ggsw_prepared_alloc_from_infos(infos))
            .collect(),
        dist: Distribution::NONE,
        x_pow_a: None,
        _phantom: PhantomData,
    }
}

/// Canonical lower-layer composition for `blind_rotation_key_prepare_tmp_bytes`.
pub fn blind_rotation_key_prepare_tmp_bytes_ref<BE, A>(module: &Module<BE>, infos: &A) -> usize
where
    A: BlindRotationKeyInfos,
    BE: Backend<ZnxWord = i64>,
    Module<BE>: GGSWPreparedFactory<BE> + SvpPPolAlloc<BE> + SvpPrepare<BE>,
{
    module.ggsw_prepare_tmp_bytes(infos)
}

/// Canonical lower-layer composition for `prepare_blind_rotation_key`.
pub fn prepare_blind_rotation_key_ref<BE>(
    module: &Module<BE>,
    res: &mut BlindRotationKeyPrepared<BE::OwnedBuf, CGGI, BE>,
    other: &BlindRotationKey<BE::OwnedBuf, CGGI, BE::ZnxWord>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend<ZnxWord = i64>,
    Module<BE>: GGSWPreparedFactory<BE> + SvpPPolAlloc<BE> + SvpPrepare<BE>,
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
        let mut buf: ScalarZnx<AlignedBuf, i64> = ScalarZnx::from_data(
            HostBytesBackend::alloc_zeroed_bytes(ScalarZnx::<AlignedBuf, i64>::bytes_of(n, 1)),
            n,
            1,
        );
        (0..n << 1).for_each(|i| {
            let mut res: SvpPPolOwned<BE> = module.svp_ppol_alloc(module.n(), 1, PrepareHint::Reuse);
            set_xai_plus_y(module, i, 0, &mut res, &mut buf);
            x_pow_a.push(res);
        });
        res.x_pow_a = Some(x_pow_a);
    }
}
