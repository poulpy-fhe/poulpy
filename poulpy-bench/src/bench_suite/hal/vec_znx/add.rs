use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};

use poulpy_hal::{
    api::{ModuleNew, VecZnxAddAssignBackend, VecZnxAddIntoBackend, VecZnxAlloc},
    layouts::{Backend, Module},
};

pub fn bench_vec_znx_add_into<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxAddIntoBackend<B> + ModuleNew<B> + VecZnxAlloc<B>,
{
    let group_name: String = format!("vec_znx_add_into::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxAddIntoBackend<B> + ModuleNew<B> + VecZnxAlloc<B>,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source = poulpy_hal::source::Source::new([0u8; 32]);

        let a = crate::random_host_vec_znx(module.n(), cols, size, &mut source);
        let b = crate::random_host_vec_znx(module.n(), cols, size, &mut source);
        let a = crate::upload_host_vec_znx::<B>(&a);
        let b = crate::upload_host_vec_znx::<B>(&b);
        let mut c = module.vec_znx_alloc(cols, size);

        move || {
            let a = crate::vec_znx_backend_ref::<B>(&a);
            let b = crate::vec_znx_backend_ref::<B>(&b);
            let mut c = crate::vec_znx_backend_mut::<B>(&mut c);
            for i in 0..cols {
                module.vec_znx_add_into_backend(&mut c, i, &a, i, &b, i);
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

pub fn bench_vec_znx_add_assign<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxAddAssignBackend<B> + ModuleNew<B> + VecZnxAlloc<B>,
{
    let group_name: String = format!("vec_znx_add_assign::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxAddAssignBackend<B> + ModuleNew<B> + VecZnxAlloc<B>,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source = poulpy_hal::source::Source::new([0u8; 32]);

        let a = crate::random_host_vec_znx(module.n(), cols, size, &mut source);
        let a = crate::upload_host_vec_znx::<B>(&a);
        let mut b = module.vec_znx_alloc(cols, size);

        move || {
            let a = crate::vec_znx_backend_ref::<B>(&a);
            let mut b = crate::vec_znx_backend_mut::<B>(&mut b);
            for i in 0..cols {
                module.vec_znx_add_assign_backend(&mut b, i, &a, i);
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
