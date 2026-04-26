use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};
use rand::Rng;

use poulpy_hal::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxBigAlloc, VecZnxDftAddAssign, VecZnxDftAddInto, VecZnxDftAlloc,
        VecZnxDftApply, VecZnxDftSub, VecZnxDftSubAssign, VecZnxDftSubNegateAssign, VecZnxIdftApply, VecZnxIdftApplyTmpA,
        VecZnxIdftApplyTmpBytes,
    },
    layouts::{Backend, DataViewMut, DeviceBuf, Module, ScratchOwned, VecZnx, VecZnxBig, VecZnxDft},
    source::Source,
};

pub fn bench_vec_znx_dft_add_into<B: Backend>(params: &crate::params::HalSweepParams, c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxDftAddInto<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
{
    let group_name: String = format!("vec_znx_dft_add_into::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(sweep: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxDftAddInto<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
    {
        let n: usize = 1 << sweep[0];
        let cols: usize = sweep[1];
        let size: usize = sweep[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxDft<DeviceBuf<B>, B> = module.vec_znx_dft_alloc(cols, size);
        let mut b: VecZnxDft<DeviceBuf<B>, B> = module.vec_znx_dft_alloc(cols, size);
        let mut c: VecZnxDft<DeviceBuf<B>, B> = module.vec_znx_dft_alloc(cols, size);

        source.fill_bytes(a.data_mut().as_mut());
        source.fill_bytes(b.data_mut().as_mut());
        source.fill_bytes(c.data_mut().as_mut());

        move || {
            for i in 0..cols {
                module.vec_znx_dft_add_into(&mut c, i, &a, i, &b, i);
            }
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << sweep[0], sweep[1], sweep[2],));
        let mut runner = runner::<B>(*sweep);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_dft_add_assign<B: Backend>(params: &crate::params::HalSweepParams, c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxDftAddAssign<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
{
    let group_name: String = format!("vec_znx_dft_add_assign::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(sweep: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxDftAddAssign<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
    {
        let n: usize = 1 << sweep[0];
        let cols: usize = sweep[1];
        let size: usize = sweep[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxDft<DeviceBuf<B>, B> = module.vec_znx_dft_alloc(cols, size);
        let mut c: VecZnxDft<DeviceBuf<B>, B> = module.vec_znx_dft_alloc(cols, size);

        // Fill a with random i64
        source.fill_bytes(a.data_mut().as_mut());
        source.fill_bytes(c.data_mut().as_mut());

        move || {
            for i in 0..cols {
                module.vec_znx_dft_add_assign(&mut c, i, &a, i);
            }
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << sweep[0], sweep[1], sweep[2],));
        let mut runner = runner::<B>(*sweep);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_dft_apply<B: Backend>(params: &crate::params::HalSweepParams, c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxDftApply<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
{
    let group_name: String = format!("vec_znx_dft_apply::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(sweep: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxDftApply<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
    {
        let n: usize = 1 << sweep[0];
        let cols: usize = sweep[1];
        let size: usize = sweep[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut res: VecZnxDft<DeviceBuf<B>, B> = module.vec_znx_dft_alloc(cols, size);
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);
        source.fill_bytes(res.data_mut().as_mut());
        source.fill_bytes(a.data_mut().as_mut());

        move || {
            for i in 0..cols {
                module.vec_znx_dft_apply(1, 0, &mut res, i, &a, i);
            }
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << sweep[0], sweep[1], sweep[2],));
        let mut runner = runner::<B>(*sweep);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_idft_apply<B: Backend>(params: &crate::params::HalSweepParams, c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxIdftApply<B> + ModuleNew<B> + VecZnxIdftApplyTmpBytes + VecZnxDftAlloc<B> + VecZnxBigAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vec_znx_idft_apply::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(sweep: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxIdftApply<B> + ModuleNew<B> + VecZnxIdftApplyTmpBytes + VecZnxDftAlloc<B> + VecZnxBigAlloc<B>,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let n: usize = 1 << sweep[0];
        let cols: usize = sweep[1];
        let size: usize = sweep[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut res: VecZnxBig<DeviceBuf<B>, B> = module.vec_znx_big_alloc(cols, size);
        let mut a: VecZnxDft<DeviceBuf<B>, B> = module.vec_znx_dft_alloc(cols, size);
        source.fill_bytes(res.data_mut().as_mut());
        source.fill_bytes(a.data_mut().as_mut());

        let mut scratch = ScratchOwned::alloc(module.vec_znx_idft_apply_tmp_bytes());

        move || {
            for i in 0..cols {
                module.vec_znx_idft_apply(&mut res, i, &a, i, scratch.borrow());
            }
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << sweep[0], sweep[1], sweep[2],));
        let mut runner = runner::<B>(*sweep);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_idft_apply_tmpa<B: Backend>(params: &crate::params::HalSweepParams, c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxIdftApplyTmpA<B> + ModuleNew<B> + VecZnxDftAlloc<B> + VecZnxBigAlloc<B>,
{
    let group_name: String = format!("vec_znx_idft_apply_tmpa::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(sweep: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxIdftApplyTmpA<B> + ModuleNew<B> + VecZnxDftAlloc<B> + VecZnxBigAlloc<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << sweep[0]);

        let cols: usize = sweep[1];
        let size: usize = sweep[2];

        let mut source: Source = Source::new([0u8; 32]);

        let mut res: VecZnxBig<DeviceBuf<B>, B> = module.vec_znx_big_alloc(cols, size);
        let mut a: VecZnxDft<DeviceBuf<B>, B> = module.vec_znx_dft_alloc(cols, size);
        source.fill_bytes(res.data_mut().as_mut());
        source.fill_bytes(a.data_mut().as_mut());

        move || {
            for i in 0..cols {
                module.vec_znx_idft_apply_tmpa(&mut res, i, &mut a, i);
            }
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << sweep[0], sweep[1], sweep[2],));
        let mut runner = runner::<B>(*sweep);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_dft_sub<B: Backend>(params: &crate::params::HalSweepParams, c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxDftSub<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
{
    let group_name: String = format!("vec_znx_dft_sub::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(sweep: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxDftSub<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
    {
        let n: usize = 1 << sweep[0];
        let cols: usize = sweep[1];
        let size: usize = sweep[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxDft<DeviceBuf<B>, B> = module.vec_znx_dft_alloc(cols, size);
        let mut b: VecZnxDft<DeviceBuf<B>, B> = module.vec_znx_dft_alloc(cols, size);
        let mut c: VecZnxDft<DeviceBuf<B>, B> = module.vec_znx_dft_alloc(cols, size);

        source.fill_bytes(a.data_mut().as_mut());
        source.fill_bytes(b.data_mut().as_mut());
        source.fill_bytes(c.data_mut().as_mut());

        move || {
            for i in 0..cols {
                module.vec_znx_dft_sub(&mut c, i, &a, i, &b, i);
            }
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << sweep[0], sweep[1], sweep[2],));
        let mut runner = runner::<B>(*sweep);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_dft_sub_assign<B: Backend>(params: &crate::params::HalSweepParams, c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxDftSubAssign<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
{
    let group_name: String = format!("vec_znx_dft_sub_assign::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(sweep: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxDftSubAssign<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
    {
        let n: usize = 1 << sweep[0];
        let cols: usize = sweep[1];
        let size: usize = sweep[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxDft<DeviceBuf<B>, B> = module.vec_znx_dft_alloc(cols, size);
        let mut c: VecZnxDft<DeviceBuf<B>, B> = module.vec_znx_dft_alloc(cols, size);

        // Fill a with random i64
        source.fill_bytes(a.data_mut().as_mut());
        source.fill_bytes(c.data_mut().as_mut());

        move || {
            for i in 0..cols {
                module.vec_znx_dft_sub_assign(&mut c, i, &a, i);
            }
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << sweep[0], sweep[1], sweep[2],));
        let mut runner = runner::<B>(*sweep);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_dft_sub_negate_assign<B: Backend>(params: &crate::params::HalSweepParams, c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxDftSubNegateAssign<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
{
    let group_name: String = format!("vec_znx_dft_sub_negate_assign::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(sweep: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxDftSubNegateAssign<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
    {
        let n: usize = 1 << sweep[0];
        let cols: usize = sweep[1];
        let size: usize = sweep[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnxDft<DeviceBuf<B>, B> = module.vec_znx_dft_alloc(cols, size);
        let mut c: VecZnxDft<DeviceBuf<B>, B> = module.vec_znx_dft_alloc(cols, size);

        // Fill a with random i64
        source.fill_bytes(a.data_mut().as_mut());
        source.fill_bytes(c.data_mut().as_mut());

        move || {
            for i in 0..cols {
                module.vec_znx_dft_sub_negate_assign(&mut c, i, &a, i);
            }
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << sweep[0], sweep[1], sweep[2],));
        let mut runner = runner::<B>(*sweep);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}
