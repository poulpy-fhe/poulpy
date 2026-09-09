use std::hint::black_box;

use criterion::{Bencher, measurement::Measurement};

use poulpy_hal::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalize, VecZnxNormalizeAssignBackend, VecZnxNormalizeTmpBytes,
    },
    layouts::{Backend, Module, ScratchOwned},
    source::Source,
};

use crate::hal::helpers::{
    random_host_vec_znx, random_normalization_vec_znx, upload_host_vec_znx, vec_znx_backend_mut, vec_znx_backend_ref,
};
use crate::hal::params::{HalSweepParms, NormalizeSweepParams};

pub fn runner_vec_znx_normalize<B: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &HalSweepParms)
where
    Module<B>: VecZnxNormalize<B> + ModuleNew<B> + VecZnxNormalizeTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    runner_vec_znx_normalize_sweep::<B, M>(
        bencher,
        &NormalizeSweepParams {
            shape: sweep.clone(),
            a_base2k: 50,
            res_base2k: 50,
            res_offset: 0,
            drop_bits: 0,
        },
    );
}

pub fn runner_vec_znx_normalize_sweep<B: Backend<ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    params: &NormalizeSweepParams,
) where
    Module<B>: VecZnxNormalize<B> + ModuleNew<B> + VecZnxNormalizeTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let sweep = &params.shape;
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source: Source = Source::new([0u8; 32]);

    let a = random_normalization_vec_znx(module.n(), sweep.cols, sweep.size, &mut source);
    let a = upload_host_vec_znx::<B>(&a);
    let res = random_host_vec_znx(module.n(), sweep.cols, sweep.size, &mut source);
    let mut res = upload_host_vec_znx::<B>(&res);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_normalize_tmp_bytes());

    bencher.iter(|| {
        let a = vec_znx_backend_ref::<B>(&a);
        let mut res = vec_znx_backend_mut::<B>(&mut res);
        for i in 0..sweep.cols {
            module.vec_znx_normalize(
                &mut res,
                params.res_base2k,
                sweep.size * params.res_base2k - params.drop_bits,
                params.res_offset,
                i,
                &a,
                params.a_base2k,
                i,
                &mut scratch.borrow(),
            );
        }
        black_box(());
    });
}

pub fn runner_vec_znx_normalize_assign<B: Backend<ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    sweep: &HalSweepParms,
) where
    Module<B>: VecZnxNormalizeAssignBackend<B> + ModuleNew<B> + VecZnxNormalizeTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let base2k: usize = 50;

    let mut source: Source = Source::new([0u8; 32]);

    let a = random_normalization_vec_znx(module.n(), sweep.cols, sweep.size, &mut source);
    let mut a = upload_host_vec_znx::<B>(&a);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_normalize_tmp_bytes());

    bencher.iter(|| {
        let mut a = vec_znx_backend_mut::<B>(&mut a);
        for i in 0..sweep.cols {
            module.vec_znx_normalize_assign_backend(base2k, sweep.size * base2k, &mut a, i, &mut scratch.borrow());
        }
        black_box(());
    });
}
