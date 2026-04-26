use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};
use rand::Rng;

use poulpy_hal::{
    api::{ModuleNew, SvpApplyDft, SvpApplyDftToDft, SvpApplyDftToDftAssign, SvpPPolAlloc, SvpPrepare, VecZnxDftAlloc},
    layouts::{Backend, DataViewMut, DeviceBuf, FillUniform, Module, ScalarZnx, SvpPPol, VecZnx, VecZnxDft},
    source::Source,
};

pub fn bench_svp_prepare<B>(params: &crate::params::SvpPrepareParams, c: &mut Criterion, label: &str)
where
    Module<B>: SvpPrepare<B> + SvpPPolAlloc<B> + ModuleNew<B>,
    B: Backend,
{
    let group_name: String = format!("svp_prepare::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(log_n: usize) -> impl FnMut()
    where
        Module<B>: SvpPrepare<B> + SvpPPolAlloc<B> + ModuleNew<B>,
        B: Backend,
    {
        let module: Module<B> = Module::<B>::new(1 << log_n);

        let cols: usize = 2;

        let mut svp: SvpPPol<DeviceBuf<B>, B> = module.svp_ppol_alloc(cols);
        let mut a: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(module.n(), cols);
        let mut source = Source::new([0u8; 32]);
        a.fill_uniform(50, &mut source);

        move || {
            module.svp_prepare(&mut svp, 0, &a, 0);
            black_box(());
        }
    }

    for &log_n in &params.log_n {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}", 1 << log_n));
        let mut runner = runner::<B>(log_n);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_svp_apply_dft<B>(params: &crate::params::HalSweepParams, c: &mut Criterion, label: &str)
where
    Module<B>: SvpApplyDft<B> + SvpPPolAlloc<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
    B: Backend,
{
    let group_name: String = format!("svp_apply_dft::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(sweep: [usize; 3]) -> impl FnMut()
    where
        Module<B>: SvpApplyDft<B> + SvpPPolAlloc<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
        B: Backend,
    {
        let n: usize = 1 << sweep[0];
        let cols: usize = sweep[1];
        let size: usize = sweep[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut svp: SvpPPol<DeviceBuf<B>, B> = module.svp_ppol_alloc(cols);
        let mut res: VecZnxDft<DeviceBuf<B>, B> = module.vec_znx_dft_alloc(cols, size);
        let mut a: VecZnx<Vec<u8>> = VecZnx::alloc(n, cols, size);

        let mut source = Source::new([0u8; 32]);

        source.fill_bytes(svp.data_mut().as_mut());
        source.fill_bytes(res.data_mut().as_mut());
        source.fill_bytes(a.data_mut().as_mut());

        move || {
            for j in 0..cols {
                module.svp_apply_dft(&mut res, j, &svp, j, &a, j);
            }
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << sweep[0], sweep[1], sweep[2]));
        let mut runner = runner::<B>(*sweep);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_svp_apply_dft_to_dft<B>(params: &crate::params::HalSweepParams, c: &mut Criterion, label: &str)
where
    Module<B>: SvpApplyDftToDft<B> + SvpPPolAlloc<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
    B: Backend,
{
    let group_name: String = format!("svp_apply_dft_to_dft::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(sweep: [usize; 3]) -> impl FnMut()
    where
        Module<B>: SvpApplyDftToDft<B> + SvpPPolAlloc<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
        B: Backend,
    {
        let n: usize = 1 << sweep[0];
        let cols: usize = sweep[1];
        let size: usize = sweep[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut svp: SvpPPol<DeviceBuf<B>, B> = module.svp_ppol_alloc(cols);
        let mut res: VecZnxDft<DeviceBuf<B>, B> = module.vec_znx_dft_alloc(cols, size);
        let mut a: VecZnxDft<DeviceBuf<B>, B> = module.vec_znx_dft_alloc(cols, size);

        let mut source = Source::new([0u8; 32]);

        source.fill_bytes(svp.data_mut().as_mut());
        source.fill_bytes(res.data_mut().as_mut());
        source.fill_bytes(a.data_mut().as_mut());

        move || {
            for j in 0..cols {
                module.svp_apply_dft_to_dft(&mut res, j, &svp, j, &a, j);
            }
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << sweep[0], sweep[1], sweep[2]));
        let mut runner = runner::<B>(*sweep);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_svp_apply_dft_to_dft_assign<B>(params: &crate::params::HalSweepParams, c: &mut Criterion, label: &str)
where
    Module<B>: SvpApplyDftToDftAssign<B> + SvpPPolAlloc<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
    B: Backend,
{
    let group_name: String = format!("svp_apply_dft_to_dft_assign::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(sweep: [usize; 3]) -> impl FnMut()
    where
        Module<B>: SvpApplyDftToDftAssign<B> + SvpPPolAlloc<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
        B: Backend,
    {
        let n: usize = 1 << sweep[0];
        let cols: usize = sweep[1];
        let size: usize = sweep[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut svp: SvpPPol<DeviceBuf<B>, B> = module.svp_ppol_alloc(cols);
        let mut res: VecZnxDft<DeviceBuf<B>, B> = module.vec_znx_dft_alloc(cols, size);

        let mut source = Source::new([0u8; 32]);

        source.fill_bytes(svp.data_mut().as_mut());
        source.fill_bytes(res.data_mut().as_mut());

        move || {
            for j in 0..cols {
                module.svp_apply_dft_to_dft_assign(&mut res, j, &svp, j);
            }
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << sweep[0], sweep[1], sweep[2]));
        let mut runner = runner::<B>(*sweep);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}
