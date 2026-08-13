use std::hint::black_box;

use criterion::{Bencher, measurement::Measurement};

use poulpy_hal::{
    api::{ModuleNew, VecZnxAddAssignBackend, VecZnxAddIntoBackend, VecZnxAlloc},
    layouts::{Backend, Module},
};

use crate::hal::helpers::{random_host_vec_znx, upload_host_vec_znx, vec_znx_backend_mut, vec_znx_backend_ref};
use crate::params::HalSweepParms;

pub fn runner_vec_znx_add_into<B: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &HalSweepParms)
where
    Module<B>: VecZnxAddIntoBackend<B> + ModuleNew<B> + VecZnxAlloc<B>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source = poulpy_hal::source::Source::new([0u8; 32]);

    let a = random_host_vec_znx(module.n(), sweep.cols, sweep.size, &mut source);
    let b = random_host_vec_znx(module.n(), sweep.cols, sweep.size, &mut source);
    let a = upload_host_vec_znx::<B>(&a);
    let b = upload_host_vec_znx::<B>(&b);
    let mut c = module.vec_znx_alloc(sweep.cols, sweep.size);

    bencher.iter(|| {
        let a = vec_znx_backend_ref::<B>(&a);
        let b = vec_znx_backend_ref::<B>(&b);
        let mut c = vec_znx_backend_mut::<B>(&mut c);
        for i in 0..sweep.cols {
            module.vec_znx_add_into_backend(&mut c, i, &a, i, &b, i);
        }
        black_box(());
    });
}

pub fn runner_vec_znx_add_assign<B: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &HalSweepParms)
where
    Module<B>: VecZnxAddAssignBackend<B> + ModuleNew<B> + VecZnxAlloc<B>,
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
            module.vec_znx_add_assign_backend(&mut b, i, &a, i);
        }
        black_box(());
    });
}
