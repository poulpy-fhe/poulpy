use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};

use poulpy_cpu_ref::reference::vec_znx::{vec_znx_lsh_tmp_bytes, vec_znx_rsh_tmp_bytes};
use poulpy_hal::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAlloc, VecZnxLshAssignBackend, VecZnxLshBackend,
        VecZnxRshAssignBackend, VecZnxRshBackend,
    },
    layouts::{Backend, Module, ScratchOwned},
};

pub fn bench_vec_znx_lsh_assign<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: ModuleNew<B> + VecZnxLshAssignBackend<B> + VecZnxAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vec_znx_lsh_assign::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxLshAssignBackend<B> + ModuleNew<B> + VecZnxAlloc<B>,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let base2k: usize = 50;

        let mut source = poulpy_hal::source::Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(vec_znx_lsh_tmp_bytes(n));

        let b = crate::random_host_vec_znx(module.n(), cols, size, &mut source);
        let mut b = crate::upload_host_vec_znx::<B>(&b);

        move || {
            let mut b = crate::vec_znx_backend_mut::<B>(&mut b);
            for i in 0..cols {
                module.vec_znx_lsh_assign_backend(base2k, base2k - 1, &mut b, i, &mut scratch.borrow());
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

pub fn bench_vec_znx_lsh<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxLshBackend<B> + ModuleNew<B> + VecZnxAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vec_znx_lsh::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxLshBackend<B> + ModuleNew<B> + VecZnxAlloc<B>,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let base2k: usize = 50;

        let mut source = poulpy_hal::source::Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(vec_znx_lsh_tmp_bytes(n));

        let a = crate::random_host_vec_znx(module.n(), cols, size, &mut source);
        let a = crate::upload_host_vec_znx::<B>(&a);
        let mut res = module.vec_znx_alloc(cols, size);

        move || {
            let a = crate::vec_znx_backend_ref::<B>(&a);
            let mut res = crate::vec_znx_backend_mut::<B>(&mut res);
            for i in 0..cols {
                module.vec_znx_lsh_backend(base2k, base2k - 1, &mut res, i, &a, i, &mut scratch.borrow());
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

pub fn bench_vec_znx_rsh_assign<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxRshAssignBackend<B> + ModuleNew<B> + VecZnxAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vec_znx_rsh_assign::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxRshAssignBackend<B> + ModuleNew<B> + VecZnxAlloc<B>,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let base2k: usize = 50;

        let mut source = poulpy_hal::source::Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(vec_znx_rsh_tmp_bytes(n));

        let b = crate::random_host_vec_znx(module.n(), cols, size, &mut source);
        let mut b = crate::upload_host_vec_znx::<B>(&b);

        move || {
            let mut b = crate::vec_znx_backend_mut::<B>(&mut b);
            for i in 0..cols {
                module.vec_znx_rsh_assign_backend(base2k, base2k - 1, &mut b, i, &mut scratch.borrow());
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

pub fn bench_vec_znx_rsh<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxRshBackend<B> + ModuleNew<B> + VecZnxAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vec_znx_rsh::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxRshBackend<B> + ModuleNew<B> + VecZnxAlloc<B>,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let base2k: usize = 50;

        let mut source = poulpy_hal::source::Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(vec_znx_rsh_tmp_bytes(n));

        let a = crate::random_host_vec_znx(module.n(), cols, size, &mut source);
        let a = crate::upload_host_vec_znx::<B>(&a);
        let mut res = module.vec_znx_alloc(cols, size);

        move || {
            let a = crate::vec_znx_backend_ref::<B>(&a);
            let mut res = crate::vec_znx_backend_mut::<B>(&mut res);
            for i in 0..cols {
                module.vec_znx_rsh_backend(base2k, base2k - 1, &mut res, i, &a, i, &mut scratch.borrow());
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
