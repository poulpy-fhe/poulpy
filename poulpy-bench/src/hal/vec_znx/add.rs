use std::hint::black_box;

use criterion::{Bencher, measurement::Measurement};

use poulpy_hal::{
    api::{ModuleNew, ScalarZnxAlloc, VecZnxAdd, VecZnxAddAssign, VecZnxAddScalarAssign, VecZnxAlloc},
    layouts::{Backend, Module},
};

use crate::hal::helpers::{
    random_host_scalar_znx, random_host_vec_znx, scalar_znx_backend_ref, upload_host_scalar_znx, upload_host_vec_znx,
    vec_znx_backend_mut, vec_znx_backend_ref,
};
use crate::hal::params::HalSweepParms;

pub fn runner_vec_znx_add<B: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &HalSweepParms)
where
    Module<B>: VecZnxAdd<B> + ModuleNew<B> + VecZnxAlloc<B>,
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
            module.vec_znx_add(&mut c, i, &a, i, &b, i);
        }
        black_box(());
    });
}

pub fn runner_vec_znx_add_assign<B: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &HalSweepParms)
where
    Module<B>: VecZnxAddAssign<B> + ModuleNew<B> + VecZnxAlloc<B>,
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
            module.vec_znx_add_assign(&mut b, i, &a, i);
        }
        black_box(());
    });
}

pub fn runner_vec_znx_add_scalar_assign<B: Backend<ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    sweep: &HalSweepParms,
) where
    Module<B>: VecZnxAddScalarAssign<B> + ModuleNew<B> + VecZnxAlloc<B> + ScalarZnxAlloc<B>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source = poulpy_hal::source::Source::new([0u8; 32]);

    let a = random_host_scalar_znx(module.n(), sweep.cols, &mut source);
    let a = upload_host_scalar_znx::<B>(&a);
    let res = random_host_vec_znx(module.n(), sweep.cols, sweep.size, &mut source);
    let mut res = upload_host_vec_znx::<B>(&res);

    bencher.iter(|| {
        let a = scalar_znx_backend_ref::<B>(&a);
        let mut res = vec_znx_backend_mut::<B>(&mut res);
        for i in 0..sweep.cols {
            module.vec_znx_add_scalar_assign(&mut res, i, sweep.size - 1, &a, i);
        }
        black_box(());
    });
}
