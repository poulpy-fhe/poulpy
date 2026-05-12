use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};
use rand::Rng;

use poulpy_hal::{
    api::{ModuleNew, VecZnxSubAssignBackend, VecZnxSubBackend, VecZnxSubNegateAssignBackend},
    layouts::{Backend, DataViewMut, Module, VecZnx, VecZnxToBackendMut, VecZnxToBackendRef},
    source::Source,
};

pub fn bench_vec_znx_sub<B>(c: &mut Criterion, label: &str)
where
    B: Backend,
    Module<B>: VecZnxSubBackend<B> + ModuleNew<B>,
    B::OwnedBuf: AsMut<[u8]>,
{
    let group_name: String = format!("vec_znx_sub::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxSubBackend<B> + ModuleNew<B>,
        B::OwnedBuf: AsMut<[u8]>,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut a = module.vec_znx_alloc(cols, size);
        let mut b = module.vec_znx_alloc(cols, size);
        let mut c = module.vec_znx_alloc(cols, size);
        source.fill_bytes(a.data_mut().as_mut());
        source.fill_bytes(b.data_mut().as_mut());
        source.fill_bytes(c.data_mut().as_mut());

        move || {
            let a = <VecZnx<B::OwnedBuf> as VecZnxToBackendRef<B>>::to_backend_ref(&a);
            let b = <VecZnx<B::OwnedBuf> as VecZnxToBackendRef<B>>::to_backend_ref(&b);
            let mut c = <VecZnx<B::OwnedBuf> as VecZnxToBackendMut<B>>::to_backend_mut(&mut c);
            for i in 0..cols {
                module.vec_znx_sub_backend(&mut c, i, &a, i, &b, i);
            }
            black_box(());
        }
    }

    for params in [[10, 2, 2], [11, 2, 4], [12, 2, 8], [13, 2, 16], [14, 2, 32]] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << params[0], params[1], params[2],));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_sub_assign<B>(c: &mut Criterion, label: &str)
where
    B: Backend,
    Module<B>: VecZnxSubAssignBackend<B> + ModuleNew<B>,
    B::OwnedBuf: AsMut<[u8]>,
{
    let group_name: String = format!("vec_znx_sub_assign::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxSubAssignBackend<B> + ModuleNew<B>,
        B::OwnedBuf: AsMut<[u8]>,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut a = module.vec_znx_alloc(cols, size);
        let mut b = module.vec_znx_alloc(cols, size);
        source.fill_bytes(a.data_mut().as_mut());
        source.fill_bytes(b.data_mut().as_mut());

        move || {
            let a = <VecZnx<B::OwnedBuf> as VecZnxToBackendRef<B>>::to_backend_ref(&a);
            let mut b = <VecZnx<B::OwnedBuf> as VecZnxToBackendMut<B>>::to_backend_mut(&mut b);
            for i in 0..cols {
                module.vec_znx_sub_assign_backend(&mut b, i, &a, i);
            }
            black_box(());
        }
    }

    for params in [[10, 2, 2], [11, 2, 4], [12, 2, 8], [13, 2, 16], [14, 2, 32]] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << params[0], params[1], params[2]));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_sub_negate_assign<B>(c: &mut Criterion, label: &str)
where
    B: Backend,
    Module<B>: VecZnxSubNegateAssignBackend<B> + ModuleNew<B>,
    B::OwnedBuf: AsMut<[u8]>,
{
    let group_name: String = format!("vec_znx_sub_negate_assign::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxSubNegateAssignBackend<B> + ModuleNew<B>,
        B::OwnedBuf: AsMut<[u8]>,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut a = module.vec_znx_alloc(cols, size);
        let mut b = module.vec_znx_alloc(cols, size);
        source.fill_bytes(a.data_mut().as_mut());
        source.fill_bytes(b.data_mut().as_mut());

        move || {
            let a = <VecZnx<B::OwnedBuf> as VecZnxToBackendRef<B>>::to_backend_ref(&a);
            let mut b = <VecZnx<B::OwnedBuf> as VecZnxToBackendMut<B>>::to_backend_mut(&mut b);
            for i in 0..cols {
                module.vec_znx_sub_negate_assign_backend(&mut b, i, &a, i);
            }
            black_box(());
        }
    }

    for params in [[10, 2, 2], [11, 2, 4], [12, 2, 8], [13, 2, 16], [14, 2, 32]] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << params[0], params[1], params[2]));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}
