use std::hint::black_box;

use criterion::{Bencher, measurement::Measurement};
use poulpy_core::{
    GLWETensoring,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GLWELayout, GLWETensorKeyLayout, GLWETensorKeyPreparedFactory, ModuleCoreAlloc, Rank,
        TorusPrecision,
    },
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostDataMut, Module, ScratchOwned},
};

use crate::core::params::{CoreParams, key_dnum_k_aux};

fn glwe_layout(cp: &CoreParams) -> GLWELayout {
    GLWELayout {
        n: Degree(cp.n),
        base2k: Base2K(cp.base2k),
        k: TorusPrecision(cp.k),
        rank: Rank(cp.rank),
    }
}

fn tensor_key_layout(cp: &CoreParams) -> GLWETensorKeyLayout {
    let (dnum, k_aux) = key_dnum_k_aux(cp.k, cp.base2k, cp.dsize);
    GLWETensorKeyLayout {
        n: Degree(cp.n),
        base2k: Base2K(cp.base2k),
        k_aux: TorusPrecision(k_aux),
        rank: Rank(cp.rank),
        dnum: Dnum(dnum),
        dsize: Dsize(cp.dsize),
    }
}

/// Relinearization (the keyswitch phase of `ckks_mul`). The tensor key is left
/// zeroed: the op is data-independent, so this times the real kernel path.
pub fn runner_glwe_tensor_relinearize<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CoreParams,
) where
    Module<BE>: ModuleNew<BE> + GLWETensoring<BE> + GLWETensorKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut + AsRef<[u8]> + AsMut<[u8]> + Sync,
{
    let glwe_infos = glwe_layout(cp);
    let tsk_infos = tensor_key_layout(cp);

    let module = Module::<BE>::new(cp.n as u64);

    let mut res = module.glwe_alloc_from_infos(&glwe_infos);
    let tensor = module.glwe_tensor_alloc_from_infos(&glwe_infos);
    let tsk = module.alloc_tensor_key_prepared_from_infos(&tsk_infos);
    let mut scratch = ScratchOwned::<BE>::alloc(module.glwe_tensor_relinearize_tmp_bytes(&res, &tensor, &tsk));

    bencher.iter(|| {
        module.glwe_tensor_relinearize(&mut res, &tensor, &tsk, &mut scratch.borrow());
        black_box(());
    });
}

pub fn runner_glwe_tensor_apply<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CoreParams,
) where
    Module<BE>: ModuleNew<BE> + GLWETensoring<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut + AsRef<[u8]> + AsMut<[u8]> + Sync,
{
    let glwe_infos = glwe_layout(cp);
    let module = Module::<BE>::new(cp.n as u64);

    let a = module.glwe_alloc_from_infos(&glwe_infos);
    let b = module.glwe_alloc_from_infos(&glwe_infos);
    let mut tensor = module.glwe_tensor_alloc_from_infos(&glwe_infos);
    let mut scratch = ScratchOwned::<BE>::alloc(module.glwe_tensor_apply_tmp_bytes(&tensor, &a, &b));

    bencher.iter(|| {
        module.glwe_tensor_apply(0, &mut tensor, &a, &b, &mut scratch.borrow());
        black_box(());
    });
}

pub fn runner_glwe_tensor_square_apply<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CoreParams,
) where
    Module<BE>: ModuleNew<BE> + GLWETensoring<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut + AsRef<[u8]> + AsMut<[u8]> + Sync,
{
    let glwe_infos = glwe_layout(cp);
    let module = Module::<BE>::new(cp.n as u64);

    let a = module.glwe_alloc_from_infos(&glwe_infos);
    let mut tensor = module.glwe_tensor_alloc_from_infos(&glwe_infos);
    let mut scratch = ScratchOwned::<BE>::alloc(module.glwe_tensor_square_apply_tmp_bytes(&tensor, &a));

    bencher.iter(|| {
        module.glwe_tensor_square_apply(0, &mut tensor, &a, &mut scratch.borrow());
        black_box(());
    });
}
