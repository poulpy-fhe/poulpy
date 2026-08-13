use std::hint::black_box;

use criterion::{Bencher, measurement::Measurement};
use rand::Rng;

use poulpy_hal::{
    api::{ModuleNew, VecZnxSubAssignBackend, VecZnxSubBackend, VecZnxSubNegateAssignBackend},
    layouts::{Backend, DataViewMut, Module, VecZnx, VecZnxToBackendMut, VecZnxToBackendRef},
    source::Source,
};

use crate::params::HalSweepParms;

pub fn runner_vec_znx_sub<B: Backend, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &HalSweepParms)
where
    Module<B>: VecZnxSubBackend<B> + ModuleNew<B>,
    B::OwnedBuf: AsMut<[u8]>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source: Source = Source::new([0u8; 32]);

    let mut a = module.vec_znx_alloc(sweep.cols, sweep.size);
    let mut b = module.vec_znx_alloc(sweep.cols, sweep.size);
    let mut c = module.vec_znx_alloc(sweep.cols, sweep.size);
    source.fill_bytes(a.data_mut().as_mut());
    source.fill_bytes(b.data_mut().as_mut());
    source.fill_bytes(c.data_mut().as_mut());

    bencher.iter(|| {
        let a = <VecZnx<B::OwnedBuf, B::ZnxWord> as VecZnxToBackendRef<B>>::to_backend_ref(&a);
        let b = <VecZnx<B::OwnedBuf, B::ZnxWord> as VecZnxToBackendRef<B>>::to_backend_ref(&b);
        let mut c = <VecZnx<B::OwnedBuf, B::ZnxWord> as VecZnxToBackendMut<B>>::to_backend_mut(&mut c);
        for i in 0..sweep.cols {
            module.vec_znx_sub_backend(&mut c, i, &a, i, &b, i);
        }
        black_box(());
    });
}

pub fn runner_vec_znx_sub_assign<B: Backend, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &HalSweepParms)
where
    Module<B>: VecZnxSubAssignBackend<B> + ModuleNew<B>,
    B::OwnedBuf: AsMut<[u8]>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source: Source = Source::new([0u8; 32]);

    let mut a = module.vec_znx_alloc(sweep.cols, sweep.size);
    let mut b = module.vec_znx_alloc(sweep.cols, sweep.size);
    source.fill_bytes(a.data_mut().as_mut());
    source.fill_bytes(b.data_mut().as_mut());

    bencher.iter(|| {
        let a = <VecZnx<B::OwnedBuf, B::ZnxWord> as VecZnxToBackendRef<B>>::to_backend_ref(&a);
        let mut b = <VecZnx<B::OwnedBuf, B::ZnxWord> as VecZnxToBackendMut<B>>::to_backend_mut(&mut b);
        for i in 0..sweep.cols {
            module.vec_znx_sub_assign_backend(&mut b, i, &a, i);
        }
        black_box(());
    });
}

pub fn runner_vec_znx_sub_negate_assign<B: Backend, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &HalSweepParms)
where
    Module<B>: VecZnxSubNegateAssignBackend<B> + ModuleNew<B>,
    B::OwnedBuf: AsMut<[u8]>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source: Source = Source::new([0u8; 32]);

    let mut a = module.vec_znx_alloc(sweep.cols, sweep.size);
    let mut b = module.vec_znx_alloc(sweep.cols, sweep.size);
    source.fill_bytes(a.data_mut().as_mut());
    source.fill_bytes(b.data_mut().as_mut());

    bencher.iter(|| {
        let a = <VecZnx<B::OwnedBuf, B::ZnxWord> as VecZnxToBackendRef<B>>::to_backend_ref(&a);
        let mut b = <VecZnx<B::OwnedBuf, B::ZnxWord> as VecZnxToBackendMut<B>>::to_backend_mut(&mut b);
        for i in 0..sweep.cols {
            module.vec_znx_sub_negate_assign_backend(&mut b, i, &a, i);
        }
        black_box(());
    });
}
