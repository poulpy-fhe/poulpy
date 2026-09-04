use std::hint::black_box;

use criterion::{Bencher, measurement::Measurement};

use poulpy_hal::{
    api::{ModuleNew, VecZnxSubAssignBackend, VecZnxSubBackend, VecZnxSubNegateAssignBackend},
    layouts::{Backend, Module},
    source::Source,
};

use crate::hal::helpers::{random_host_vec_znx, upload_host_vec_znx, vec_znx_backend_mut, vec_znx_backend_ref};
use crate::hal::params::HalSweepParms;
use poulpy_hal::layouts::BorrowedCarryView;

pub fn runner_vec_znx_sub<B: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &HalSweepParms)
where
    Module<B>: VecZnxSubBackend<B> + ModuleNew<B>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source: Source = Source::new([0u8; 32]);

    let a = random_host_vec_znx(module.n(), sweep.cols, sweep.size, &mut source);
    let a = upload_host_vec_znx::<B>(&a);
    let b = random_host_vec_znx(module.n(), sweep.cols, sweep.size, &mut source);
    let b = upload_host_vec_znx::<B>(&b);
    let c = random_host_vec_znx(module.n(), sweep.cols, sweep.size, &mut source);
    let mut c = upload_host_vec_znx::<B>(&c);

    bencher.iter(|| {
        let a = vec_znx_backend_ref::<B, _>(&a);
        let b = vec_znx_backend_ref::<B, _>(&b);
        let mut c = vec_znx_backend_mut::<B, _>(&mut c).borrowed_carry_view();
        for i in 0..sweep.cols {
            module.vec_znx_sub_backend(&mut c, i, &a, i, &b, i);
        }
        black_box(());
    });
}

pub fn runner_vec_znx_sub_assign<B: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &HalSweepParms)
where
    Module<B>: VecZnxSubAssignBackend<B> + ModuleNew<B>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source: Source = Source::new([0u8; 32]);

    let a = random_host_vec_znx(module.n(), sweep.cols, sweep.size, &mut source);
    let a = upload_host_vec_znx::<B>(&a);
    let b = random_host_vec_znx(module.n(), sweep.cols, sweep.size, &mut source);
    let mut b = upload_host_vec_znx::<B>(&b);

    bencher.iter(|| {
        let a = vec_znx_backend_ref::<B, _>(&a);
        let mut b = vec_znx_backend_mut::<B, _>(&mut b).borrowed_carry_view();
        for i in 0..sweep.cols {
            module.vec_znx_sub_assign_backend(&mut b, i, &a, i);
        }
        black_box(());
    });
}

pub fn runner_vec_znx_sub_negate_assign<B: Backend<ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    sweep: &HalSweepParms,
) where
    Module<B>: VecZnxSubNegateAssignBackend<B> + ModuleNew<B>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source: Source = Source::new([0u8; 32]);

    let a = random_host_vec_znx(module.n(), sweep.cols, sweep.size, &mut source);
    let a = upload_host_vec_znx::<B>(&a);
    let b = random_host_vec_znx(module.n(), sweep.cols, sweep.size, &mut source);
    let mut b = upload_host_vec_znx::<B>(&b);

    bencher.iter(|| {
        let a = vec_znx_backend_ref::<B, _>(&a);
        let mut b = vec_znx_backend_mut::<B, _>(&mut b).borrowed_carry_view();
        for i in 0..sweep.cols {
            module.vec_znx_sub_negate_assign_backend(&mut b, i, &a, i);
        }
        black_box(());
    });
}
