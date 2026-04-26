use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};

use poulpy_hal::{
    api::{ModuleNew, VecZnxNegate, VecZnxNegateAssign},
    layouts::{Backend, FillUniform, Module, VecZnx},
    source::Source,
};

pub fn bench_vec_znx_negate<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxNegate + ModuleNew<B>,
{
    let group_name: String = format!("vec_znx_negate::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxNegate + ModuleNew<B>,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);
        let mut b: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);

        // Fill a with random i64
        a.fill_uniform(50, &mut source);
        b.fill_uniform(50, &mut source);

        move || {
            for i in 0..cols {
                module.vec_znx_negate(&mut b, i, &a, i);
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

pub fn bench_vec_znx_negate_assign<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxNegateAssign + ModuleNew<B>,
{
    let group_name: String = format!("vec_znx_negate_assign::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxNegateAssign + ModuleNew<B>,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);

        // Fill a with random i64
        a.fill_uniform(50, &mut source);
        move || {
            for i in 0..cols {
                module.vec_znx_negate_assign(&mut a, i);
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
