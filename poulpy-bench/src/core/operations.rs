use poulpy_core::api::TransferInto;
use poulpy_core::{
    GLWEAdd, GLWEMulPlain, GLWENormalize, GLWESub,
    layouts::{Base2K, Degree, GLWE, GLWELayout, GLWEPlaintext, ModuleCoreAlloc, Rank, TorusPrecision},
};
use poulpy_hal::layouts::CopyFromHost;
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, ScratchOwned},
    source::Source,
};

use std::hint::black_box;

use criterion::{Bencher, measurement::Measurement};

use crate::core::fill::{host_glwe, host_glwe_plaintext, staging};
use crate::core::params::CoreParams;

fn glwe_layout(cp: &CoreParams) -> GLWELayout {
    GLWELayout {
        n: Degree(cp.n),
        base2k: Base2K(cp.base2k),
        k: TorusPrecision(cp.k),
        rank: Rank(cp.rank),
    }
}

pub fn runner_glwe_add_into<BE: Backend<ZnxWord = i64, OwnedBuf: CopyFromHost>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CoreParams,
) where
    Module<BE>: ModuleNew<BE> + GLWEAdd<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>,
{
    let infos = glwe_layout(cp);
    let module: Module<BE> = Module::<BE>::new(cp.n as u64);
    let mut source: Source = Source::new([0u8; 32]);
    let host = staging(cp.n as usize);

    let mut res: GLWE<BE::OwnedBuf, i64> = module.glwe_alloc_from_infos(&infos);
    let mut a: GLWE<BE::OwnedBuf, i64> = module.glwe_alloc_from_infos(&infos);
    let mut b: GLWE<BE::OwnedBuf, i64> = module.glwe_alloc_from_infos(&infos);
    host_glwe(&host, &infos, &mut source).transfer_into(&mut res);
    host_glwe(&host, &infos, &mut source).transfer_into(&mut a);
    host_glwe(&host, &infos, &mut source).transfer_into(&mut b);

    bencher.iter(|| {
        module.glwe_add_into(&mut res, &a, &b);
        black_box(());
    });
}

pub fn runner_glwe_add_assign<BE: Backend<ZnxWord = i64, OwnedBuf: CopyFromHost>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CoreParams,
) where
    Module<BE>: ModuleNew<BE> + GLWEAdd<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>,
{
    let infos = glwe_layout(cp);
    let module: Module<BE> = Module::<BE>::new(cp.n as u64);
    let mut source: Source = Source::new([0u8; 32]);
    let host = staging(cp.n as usize);

    let mut res: GLWE<BE::OwnedBuf, i64> = module.glwe_alloc_from_infos(&infos);
    let mut b: GLWE<BE::OwnedBuf, i64> = module.glwe_alloc_from_infos(&infos);
    host_glwe(&host, &infos, &mut source).transfer_into(&mut res);
    host_glwe(&host, &infos, &mut source).transfer_into(&mut b);

    bencher.iter(|| {
        module.glwe_add_assign(&mut res, &b);
        black_box(());
    });
}

pub fn runner_glwe_sub<BE: Backend<ZnxWord = i64, OwnedBuf: CopyFromHost>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CoreParams,
) where
    Module<BE>: ModuleNew<BE> + GLWESub<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>,
{
    let infos = glwe_layout(cp);
    let module: Module<BE> = Module::<BE>::new(cp.n as u64);
    let mut source: Source = Source::new([0u8; 32]);
    let host = staging(cp.n as usize);

    let mut res: GLWE<BE::OwnedBuf, i64> = module.glwe_alloc_from_infos(&infos);
    let mut a: GLWE<BE::OwnedBuf, i64> = module.glwe_alloc_from_infos(&infos);
    let mut b: GLWE<BE::OwnedBuf, i64> = module.glwe_alloc_from_infos(&infos);
    host_glwe(&host, &infos, &mut source).transfer_into(&mut res);
    host_glwe(&host, &infos, &mut source).transfer_into(&mut a);
    host_glwe(&host, &infos, &mut source).transfer_into(&mut b);

    bencher.iter(|| {
        module.glwe_sub(&mut res, &a, &b);
        black_box(());
    });
}

pub fn runner_glwe_sub_assign<BE: Backend<ZnxWord = i64, OwnedBuf: CopyFromHost>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CoreParams,
) where
    Module<BE>: ModuleNew<BE> + GLWESub<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>,
{
    let infos = glwe_layout(cp);
    let module: Module<BE> = Module::<BE>::new(cp.n as u64);
    let mut source: Source = Source::new([0u8; 32]);
    let host = staging(cp.n as usize);

    let mut res: GLWE<BE::OwnedBuf, i64> = module.glwe_alloc_from_infos(&infos);
    let mut b: GLWE<BE::OwnedBuf, i64> = module.glwe_alloc_from_infos(&infos);
    host_glwe(&host, &infos, &mut source).transfer_into(&mut res);
    host_glwe(&host, &infos, &mut source).transfer_into(&mut b);

    bencher.iter(|| {
        module.glwe_sub_assign(&mut res, &b);
        black_box(());
    });
}

pub fn runner_glwe_normalize<BE: Backend<ZnxWord = i64, OwnedBuf: CopyFromHost>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CoreParams,
) where
    Module<BE>: ModuleNew<BE> + GLWENormalize<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let infos = glwe_layout(cp);
    let module: Module<BE> = Module::<BE>::new(cp.n as u64);
    let mut source: Source = Source::new([0u8; 32]);
    let host = staging(cp.n as usize);

    let mut res: GLWE<BE::OwnedBuf, i64> = module.glwe_alloc_from_infos(&infos);
    let mut a: GLWE<BE::OwnedBuf, i64> = module.glwe_alloc_from_infos(&infos);
    host_glwe(&host, &infos, &mut source).transfer_into(&mut res);
    host_glwe(&host, &infos, &mut source).transfer_into(&mut a);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_normalize_tmp_bytes());

    bencher.iter(|| {
        module.glwe_normalize(&mut res, &a, &mut scratch.borrow());
        black_box(());
    });
}

pub fn runner_glwe_normalize_assign<BE: Backend<ZnxWord = i64, OwnedBuf: CopyFromHost>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CoreParams,
) where
    Module<BE>: ModuleNew<BE> + GLWENormalize<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let infos = glwe_layout(cp);
    let module: Module<BE> = Module::<BE>::new(cp.n as u64);
    let mut source: Source = Source::new([0u8; 32]);
    let host = staging(cp.n as usize);

    let mut res: GLWE<BE::OwnedBuf, i64> = module.glwe_alloc_from_infos(&infos);
    host_glwe(&host, &infos, &mut source).transfer_into(&mut res);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_normalize_tmp_bytes());

    bencher.iter(|| {
        module.glwe_normalize_assign(&mut res, &mut scratch.borrow());
        black_box(());
    });
}

pub fn runner_glwe_mul_plain<BE: Backend<ZnxWord = i64, OwnedBuf: CopyFromHost>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CoreParams,
) where
    Module<BE>: ModuleNew<BE> + GLWEMulPlain<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let infos = glwe_layout(cp);
    let module: Module<BE> = Module::<BE>::new(cp.n as u64);
    let mut source: Source = Source::new([0u8; 32]);
    let host = staging(cp.n as usize);

    let mut ct_out: GLWE<BE::OwnedBuf, i64> = module.glwe_alloc_from_infos(&infos);
    let mut ct_in: GLWE<BE::OwnedBuf, i64> = module.glwe_alloc_from_infos(&infos);
    let mut pt: GLWEPlaintext<BE::OwnedBuf, i64> = module.glwe_plaintext_alloc_from_infos(&infos);
    host_glwe(&host, &infos, &mut source).transfer_into(&mut ct_out);
    host_glwe(&host, &infos, &mut source).transfer_into(&mut ct_in);
    host_glwe_plaintext(&host, &infos, &mut source).transfer_into(&mut pt);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_mul_plain_tmp_bytes(&ct_out, &ct_in, &pt));

    bencher.iter(|| {
        module.glwe_mul_plain(0, &mut ct_out, &ct_in, &pt, &mut scratch.borrow());
        black_box(());
    });
}

pub fn runner_glwe_mul_plain_assign<BE: Backend<ZnxWord = i64, OwnedBuf: CopyFromHost>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CoreParams,
) where
    Module<BE>: ModuleNew<BE> + GLWEMulPlain<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let infos = glwe_layout(cp);
    let module: Module<BE> = Module::<BE>::new(cp.n as u64);
    let mut source: Source = Source::new([0u8; 32]);
    let host = staging(cp.n as usize);

    let mut ct: GLWE<BE::OwnedBuf, i64> = module.glwe_alloc_from_infos(&infos);
    let mut pt: GLWEPlaintext<BE::OwnedBuf, i64> = module.glwe_plaintext_alloc_from_infos(&infos);
    host_glwe(&host, &infos, &mut source).transfer_into(&mut ct);
    host_glwe_plaintext(&host, &infos, &mut source).transfer_into(&mut pt);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_mul_plain_tmp_bytes(&infos, &ct, &pt));

    bencher.iter(|| {
        module.glwe_mul_plain_assign(0, &mut ct, &pt, &mut scratch.borrow());
        black_box(());
    });
}
