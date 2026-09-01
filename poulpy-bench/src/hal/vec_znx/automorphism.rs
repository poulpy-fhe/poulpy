use std::hint::black_box;

use criterion::{Bencher, measurement::Measurement};

use poulpy_hal::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAutomorphismAssignBackend, VecZnxAutomorphismAssignTmpBytes,
        VecZnxAutomorphismBackend,
    },
    layouts::{Backend, Module, ScratchOwned},
    source::Source,
};

use crate::hal::helpers::{random_host_vec_znx, upload_host_vec_znx, vec_znx_backend_mut, vec_znx_backend_ref};
use crate::hal::params::HalSweepParms;

pub fn runner_vec_znx_automorphism<B: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &HalSweepParms)
where
    Module<B>: VecZnxAutomorphismBackend<B> + ModuleNew<B>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source: Source = Source::new([0u8; 32]);

    let a = random_host_vec_znx(module.n(), sweep.cols, sweep.size, &mut source);
    let a = upload_host_vec_znx::<B>(&a);
    let res = random_host_vec_znx(module.n(), sweep.cols, sweep.size, &mut source);
    let mut res = upload_host_vec_znx::<B>(&res);

    bencher.iter(|| {
        let a = vec_znx_backend_ref::<B, _>(&a);
        let mut res = vec_znx_backend_mut::<B, _>(&mut res);
        for i in 0..sweep.cols {
            module.vec_znx_automorphism_backend(-7, &mut res, i, &a, i);
        }
        black_box(());
    });
}

pub fn runner_vec_znx_automorphism_assign<B: Backend<ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    sweep: &HalSweepParms,
) where
    Module<B>: VecZnxAutomorphismAssignBackend<B> + ModuleNew<B> + VecZnxAutomorphismAssignTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source: Source = Source::new([0u8; 32]);

    let mut scratch = ScratchOwned::alloc(module.vec_znx_automorphism_assign_tmp_bytes());

    let res = random_host_vec_znx(module.n(), sweep.cols, sweep.size, &mut source);
    let mut res = upload_host_vec_znx::<B>(&res);

    bencher.iter(|| {
        let mut res = vec_znx_backend_mut::<B, _>(&mut res);
        for i in 0..sweep.cols {
            module.vec_znx_automorphism_assign_backend(-7, &mut res, i, &mut scratch.borrow());
        }
        black_box(());
    });
}
