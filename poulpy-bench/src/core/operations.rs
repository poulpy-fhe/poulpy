use poulpy_core::{
    GLWEAdd, GLWEMulPlain, GLWENormalize, GLWESub,
    layouts::{Base2K, Degree, GLWE, GLWELayout, GLWEPlaintext, ModuleCoreAlloc, Rank, TorusPrecision},
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostDataMut, Module, ScratchOwned},
};

use std::hint::black_box;

use criterion::{Bencher, measurement::Measurement};

use crate::params::CoreParams;

fn glwe_layout(cp: &CoreParams) -> GLWELayout {
    GLWELayout {
        n: Degree(cp.n),
        base2k: Base2K(cp.base2k),
        k: TorusPrecision(cp.k),
        rank: Rank(cp.rank),
    }
}

pub fn runner_glwe_add_into<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, cp: &CoreParams)
where
    Module<BE>: ModuleNew<BE> + GLWEAdd<BE>,
{
    let infos = glwe_layout(cp);
    let module: Module<BE> = Module::<BE>::new(cp.n as u64);

    let mut res: GLWE<Vec<u8>, i64> = module.glwe_alloc_from_infos(&infos);
    let a: GLWE<Vec<u8>, i64> = module.glwe_alloc_from_infos(&infos);
    let b: GLWE<Vec<u8>, i64> = module.glwe_alloc_from_infos(&infos);

    bencher.iter(|| {
        module.glwe_add_into(&mut res, &a, &b);
        black_box(());
    });
}

pub fn runner_glwe_add_assign<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CoreParams,
) where
    Module<BE>: ModuleNew<BE> + GLWEAdd<BE>,
{
    let infos = glwe_layout(cp);
    let module: Module<BE> = Module::<BE>::new(cp.n as u64);

    let mut res: GLWE<Vec<u8>, i64> = module.glwe_alloc_from_infos(&infos);
    let b: GLWE<Vec<u8>, i64> = module.glwe_alloc_from_infos(&infos);

    bencher.iter(|| {
        module.glwe_add_assign(&mut res, &b);
        black_box(());
    });
}

pub fn runner_glwe_sub<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, cp: &CoreParams)
where
    Module<BE>: ModuleNew<BE> + GLWESub<BE>,
{
    let infos = glwe_layout(cp);
    let module: Module<BE> = Module::<BE>::new(cp.n as u64);

    let mut res: GLWE<Vec<u8>, i64> = module.glwe_alloc_from_infos(&infos);
    let a: GLWE<Vec<u8>, i64> = module.glwe_alloc_from_infos(&infos);
    let b: GLWE<Vec<u8>, i64> = module.glwe_alloc_from_infos(&infos);

    bencher.iter(|| {
        module.glwe_sub(&mut res, &a, &b);
        black_box(());
    });
}

pub fn runner_glwe_sub_assign<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CoreParams,
) where
    Module<BE>: ModuleNew<BE> + GLWESub<BE>,
{
    let infos = glwe_layout(cp);
    let module: Module<BE> = Module::<BE>::new(cp.n as u64);

    let mut res: GLWE<Vec<u8>, i64> = module.glwe_alloc_from_infos(&infos);
    let b: GLWE<Vec<u8>, i64> = module.glwe_alloc_from_infos(&infos);

    bencher.iter(|| {
        module.glwe_sub_assign(&mut res, &b);
        black_box(());
    });
}

pub fn runner_glwe_normalize<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CoreParams,
) where
    Module<BE>: ModuleNew<BE> + GLWENormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> BE::BufMut<'a>: AsRef<[u8]> + AsMut<[u8]> + Sync,
    for<'a> BE::BufRef<'a>: AsRef<[u8]> + Send,
{
    let infos = glwe_layout(cp);
    let module: Module<BE> = Module::<BE>::new(cp.n as u64);

    let mut res: GLWE<Vec<u8>, i64> = module.glwe_alloc_from_infos(&infos);
    let a: GLWE<Vec<u8>, i64> = module.glwe_alloc_from_infos(&infos);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_normalize_tmp_bytes());

    bencher.iter(|| {
        module.glwe_normalize(&mut res, &a, &mut scratch.borrow());
        black_box(());
    });
}

pub fn runner_glwe_normalize_assign<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CoreParams,
) where
    Module<BE>: ModuleNew<BE> + GLWENormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> BE::BufMut<'a>: AsRef<[u8]> + AsMut<[u8]> + Sync,
{
    let infos = glwe_layout(cp);
    let module: Module<BE> = Module::<BE>::new(cp.n as u64);

    let mut res: GLWE<Vec<u8>, i64> = module.glwe_alloc_from_infos(&infos);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_normalize_tmp_bytes());

    bencher.iter(|| {
        module.glwe_normalize_assign(&mut res, &mut scratch.borrow());
        black_box(());
    });
}

pub fn runner_glwe_mul_plain<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CoreParams,
) where
    Module<BE>: ModuleNew<BE> + GLWEMulPlain<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut + AsRef<[u8]> + AsMut<[u8]> + Sync,
{
    let infos = glwe_layout(cp);
    let module: Module<BE> = Module::<BE>::new(cp.n as u64);

    let mut ct_out: GLWE<Vec<u8>, i64> = module.glwe_alloc_from_infos(&infos);
    let ct_in: GLWE<Vec<u8>, i64> = module.glwe_alloc_from_infos(&infos);
    let pt: GLWEPlaintext<Vec<u8>, i64> = module.glwe_plaintext_alloc_from_infos(&infos);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_mul_plain_tmp_bytes(&ct_out, &ct_in, &pt)*2); // TODO: have to double it for some reason.

    bencher.iter(|| {
        module.glwe_mul_plain(0, &mut ct_out, &ct_in, &pt, &mut scratch.borrow());
        black_box(());
    });
}

pub fn runner_glwe_mul_plain_assign<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CoreParams,
) where
    Module<BE>: ModuleNew<BE> + GLWEMulPlain<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'x> BE::BufMut<'x>: HostDataMut + AsRef<[u8]> + AsMut<[u8]> + Sync,
{
    let infos = glwe_layout(cp);
    let module: Module<BE> = Module::<BE>::new(cp.n as u64);

    let mut ct: GLWE<Vec<u8>, i64> = module.glwe_alloc_from_infos(&infos);
    let pt: GLWEPlaintext<Vec<u8>, i64> = module.glwe_plaintext_alloc_from_infos(&infos);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_mul_plain_tmp_bytes(&infos, &ct, &pt)*2); // TODO: have to double it for some reason.

    bencher.iter(|| {
        module.glwe_mul_plain_assign(0, &mut ct, &pt, &mut scratch.borrow());
        black_box(());
    });
}
