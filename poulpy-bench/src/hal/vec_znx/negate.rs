use std::hint::black_box;

use criterion::{Bencher, measurement::Measurement};

use poulpy_hal::{
    api::{ModuleNew, VecZnxAlloc, VecZnxNegateAssignBackend, VecZnxNegateBackend},
    layouts::{Backend, Module},
};

use crate::hal::helpers::{random_host_vec_znx, upload_host_vec_znx, vec_znx_backend_mut, vec_znx_backend_ref};
use crate::params::HalSweepParms;

pub fn runner_vec_znx_negate<B: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &HalSweepParms)
where
    Module<B>: VecZnxNegateBackend<B> + ModuleNew<B> + VecZnxAlloc<B>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source = poulpy_hal::source::Source::new([0u8; 32]);

    let a = random_host_vec_znx(module.n(), sweep.cols, sweep.size, &mut source);
    let a = upload_host_vec_znx::<B>(&a);
    let mut b = module.vec_znx_alloc(sweep.cols, sweep.size);

    bencher.iter(|| {
        let a = vec_znx_backend_ref::<B>(&a);
        let mut b = vec_znx_backend_mut::<B>(&mut b);
        for i in 0..sweep.cols {
            module.vec_znx_negate_backend(&mut b, i, &a, i);
        }
        black_box(());
    });
}

pub fn runner_vec_znx_negate_assign<B: Backend<ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    sweep: &HalSweepParms,
) where
    Module<B>: VecZnxNegateAssignBackend<B> + ModuleNew<B> + VecZnxAlloc<B>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source = poulpy_hal::source::Source::new([0u8; 32]);

    let a = random_host_vec_znx(module.n(), sweep.cols, sweep.size, &mut source);
    let mut a = upload_host_vec_znx::<B>(&a);

    bencher.iter(|| {
        let mut a = vec_znx_backend_mut::<B>(&mut a);
        for i in 0..sweep.cols {
            module.vec_znx_negate_assign_backend(&mut a, i);
        }
        black_box(());
    });
}
