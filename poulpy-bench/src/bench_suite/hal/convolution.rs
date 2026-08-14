use poulpy_hal::layouts::CnvPVecLOwned;
use poulpy_hal::layouts::CnvPVecLToBackendMut;
use poulpy_hal::layouts::CnvPVecLToBackendRef;
use poulpy_hal::layouts::CnvPVecROwned;
use poulpy_hal::layouts::CnvPVecRToBackendMut;
use poulpy_hal::layouts::CnvPVecRToBackendRef;
use poulpy_hal::layouts::VecZnxBigOwned;
use poulpy_hal::layouts::VecZnxBigToBackendMut;
use poulpy_hal::layouts::VecZnxDftToBackendMut;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};

use poulpy_hal::{
    api::{CnvPVecAlloc, Convolution, ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxBigAlloc, VecZnxDftAlloc},
    layouts::{Backend, Module, ScratchOwned},
    source::Source,
};

pub fn bench_cnv_prepare_left_pvec<BE>(params: &crate::params::CnvSweepParams, c: &mut Criterion, label: &str)
where
    BE: Backend<ZnxWord = i64> + 'static,
    Module<BE>: ModuleNew<BE> + Convolution<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let group_name: String = format!("cnv_prepare_left_pvec::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<BE>(n: usize, size: usize) -> impl FnMut()
    where
        BE: Backend<ZnxWord = i64> + 'static,
        Module<BE>: ModuleNew<BE> + Convolution<BE> + CnvPVecAlloc<BE>,
        ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    {
        let mut source: Source = Source::new([0u8; 32]);

        let c_size: usize = size + size - 1;

        let module: Module<BE> = Module::<BE>::new(n as u64);

        let mut a_prep: CnvPVecLOwned<BE> = module.cnv_pvec_left_alloc(1, size);

        let a = crate::random_host_vec_znx(module.n(), 1, size, &mut source);
        let a = crate::upload_host_vec_znx::<BE>(&a);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.cnv_prepare_left_pvec_tmp_bytes(c_size, size));

        move || {
            let mut a_prep_backend = a_prep.to_backend_mut();
            let a_backend = crate::vec_znx_backend_ref::<BE>(&a);
            module.cnv_prepare_left_pvec(&mut a_prep_backend, &a_backend, !0i64, &mut scratch.borrow());
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let log_n: usize = sweep[0];
        let size: usize = sweep[1];
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x{}", 1 << log_n, size));
        let mut runner = runner(1 << log_n, size);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_cnv_prepare_right_pvec<BE>(params: &crate::params::CnvSweepParams, c: &mut Criterion, label: &str)
where
    BE: Backend<ZnxWord = i64> + 'static,
    Module<BE>: ModuleNew<BE> + Convolution<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let group_name: String = format!("cnv_prepare_right_pvec::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<BE>(n: usize, size: usize) -> impl FnMut()
    where
        BE: Backend<ZnxWord = i64> + 'static,
        Module<BE>: ModuleNew<BE> + Convolution<BE> + CnvPVecAlloc<BE>,
        ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    {
        let mut source: Source = Source::new([0u8; 32]);

        let c_size: usize = size + size - 1;

        let module: Module<BE> = Module::<BE>::new(n as u64);

        let mut a_prep: CnvPVecROwned<BE> = module.cnv_pvec_right_alloc(1, size);

        let a = crate::random_host_vec_znx(module.n(), 1, size, &mut source);
        let a = crate::upload_host_vec_znx::<BE>(&a);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.cnv_prepare_right_pvec_tmp_bytes(c_size, size));

        move || {
            let mut a_prep_backend = a_prep.to_backend_mut();
            let a_backend = crate::vec_znx_backend_ref::<BE>(&a);
            module.cnv_prepare_right_pvec(&mut a_prep_backend, &a_backend, !0i64, &mut scratch.borrow());
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let log_n: usize = sweep[0];
        let size: usize = sweep[1];
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x{}", 1 << log_n, size));
        let mut runner = runner(1 << log_n, size);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_cnv_apply_pvec_to_dft<BE>(params: &crate::params::CnvSweepParams, c: &mut Criterion, label: &str)
where
    BE: Backend<ZnxWord = i64> + 'static,
    Module<BE>: ModuleNew<BE> + Convolution<BE> + VecZnxDftAlloc<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let group_name: String = format!("cnv_apply_pvec_to_dft::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<BE>(n: usize, size: usize) -> impl FnMut()
    where
        BE: Backend<ZnxWord = i64> + 'static,
        Module<BE>: ModuleNew<BE> + Convolution<BE> + VecZnxDftAlloc<BE> + CnvPVecAlloc<BE>,
        ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    {
        let mut source: Source = Source::new([0u8; 32]);

        let c_size: usize = size + size - 1;

        let module: Module<BE> = Module::<BE>::new(n as u64);

        let a_prep: CnvPVecLOwned<BE> = crate::random_backend_cnv_pvec_left::<BE>(module.n(), 1, size, &mut source);
        let b_prep: CnvPVecROwned<BE> = crate::random_backend_cnv_pvec_right::<BE>(module.n(), 1, size, &mut source);
        let mut c_dft = module.vec_znx_dft_alloc(1, c_size);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .cnv_apply_pvec_to_dft_tmp_bytes(0, c_size, size, size)
                .max(module.cnv_prepare_left_pvec_tmp_bytes(c_size, size))
                .max(module.cnv_prepare_right_pvec_tmp_bytes(c_size, size)),
        );
        move || {
            let mut c_dft_backend = c_dft.to_backend_mut();
            module.cnv_apply_pvec_to_dft(
                0,
                &mut c_dft_backend,
                0,
                &a_prep.to_backend_ref(),
                0,
                &b_prep.to_backend_ref(),
                0,
                &mut scratch.borrow(),
            );
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let log_n: usize = sweep[0];
        let size: usize = sweep[1];
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x{}", 1 << log_n, size));
        let mut runner = runner(1 << log_n, size);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_cnv_apply_pvec_to_dft_accumulate<BE>(params: &crate::params::CnvSweepParams, c: &mut Criterion, label: &str)
where
    BE: Backend<ZnxWord = i64> + 'static,
    Module<BE>: ModuleNew<BE> + Convolution<BE> + VecZnxDftAlloc<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let group_name: String = format!("cnv_apply_pvec_to_dft_accumulate::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<BE>(n: usize, size: usize) -> impl FnMut()
    where
        BE: Backend<ZnxWord = i64> + 'static,
        Module<BE>: ModuleNew<BE> + Convolution<BE> + VecZnxDftAlloc<BE> + CnvPVecAlloc<BE>,
        ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    {
        let mut source: Source = Source::new([0u8; 32]);

        let c_size: usize = size + size - 1;

        let module: Module<BE> = Module::<BE>::new(n as u64);

        let a_prep: CnvPVecLOwned<BE> = crate::random_backend_cnv_pvec_left::<BE>(module.n(), 1, size, &mut source);
        let b_prep: CnvPVecROwned<BE> = crate::random_backend_cnv_pvec_right::<BE>(module.n(), 1, size, &mut source);
        let mut c_dft = module.vec_znx_dft_alloc(1, c_size);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .cnv_apply_pvec_to_dft_tmp_bytes(0, c_size, size, size)
                .max(module.cnv_prepare_left_pvec_tmp_bytes(c_size, size))
                .max(module.cnv_prepare_right_pvec_tmp_bytes(c_size, size)),
        );
        {
            let mut c_dft_backend = c_dft.to_backend_mut();
            module.cnv_apply_pvec_to_dft(
                0,
                &mut c_dft_backend,
                0,
                &a_prep.to_backend_ref(),
                0,
                &b_prep.to_backend_ref(),
                0,
                &mut scratch.borrow(),
            );
        }
        move || {
            let mut c_dft_backend = c_dft.to_backend_mut();
            module.cnv_apply_pvec_to_dft_accumulate(
                0,
                &mut c_dft_backend,
                0,
                &a_prep.to_backend_ref(),
                0,
                &b_prep.to_backend_ref(),
                0,
                &mut scratch.borrow(),
            );
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let log_n: usize = sweep[0];
        let size: usize = sweep[1];
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x{}", 1 << log_n, size));
        let mut runner = runner(1 << log_n, size);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_cnv_pairwise_apply_pvec_to_dft<BE>(params: &crate::params::CnvSweepParams, c: &mut Criterion, label: &str)
where
    BE: Backend<ZnxWord = i64> + 'static,
    Module<BE>: ModuleNew<BE> + Convolution<BE> + VecZnxDftAlloc<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let group_name: String = format!("cnv_pairwise_apply_pvec_to_dft::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<BE>(n: usize, size: usize) -> impl FnMut()
    where
        BE: Backend<ZnxWord = i64> + 'static,
        Module<BE>: ModuleNew<BE> + Convolution<BE> + VecZnxDftAlloc<BE> + CnvPVecAlloc<BE>,
        ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    {
        let mut source: Source = Source::new([0u8; 32]);

        let module: Module<BE> = Module::<BE>::new(n as u64);

        let cols = 2;
        let c_size: usize = size + size - 1;

        let a_prep: CnvPVecLOwned<BE> = crate::random_backend_cnv_pvec_left::<BE>(module.n(), cols, size, &mut source);
        let b_prep: CnvPVecROwned<BE> = crate::random_backend_cnv_pvec_right::<BE>(module.n(), cols, size, &mut source);
        let mut c_dft = module.vec_znx_dft_alloc(1, c_size);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .cnv_pairwise_apply_pvec_to_dft_tmp_bytes(0, c_size, size, size)
                .max(module.cnv_prepare_left_pvec_tmp_bytes(c_size, size))
                .max(module.cnv_prepare_right_pvec_tmp_bytes(c_size, size)),
        );
        move || {
            let mut c_dft_backend = c_dft.to_backend_mut();
            module.cnv_pairwise_apply_pvec_to_dft(
                0,
                &mut c_dft_backend,
                0,
                &a_prep.to_backend_ref(),
                &b_prep.to_backend_ref(),
                0,
                1,
                &mut scratch.borrow(),
            );
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let log_n: usize = sweep[0];
        let size: usize = sweep[1];
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x{}", 1 << log_n, size));
        let mut runner = runner(1 << log_n, size);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_cnv_by_const_apply<BE>(params: &crate::params::CnvSweepParams, c: &mut Criterion, label: &str)
where
    BE: Backend<ZnxWord = i64> + 'static,
    Module<BE>: ModuleNew<BE> + Convolution<BE> + VecZnxBigAlloc<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let group_name: String = format!("cnv_by_const::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<BE>(n: usize, size: usize) -> impl FnMut()
    where
        BE: Backend<ZnxWord = i64> + 'static,
        Module<BE>: ModuleNew<BE> + Convolution<BE> + VecZnxBigAlloc<BE> + CnvPVecAlloc<BE>,
        ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    {
        let mut source: Source = Source::new([0u8; 32]);

        let module: Module<BE> = Module::<BE>::new(n as u64);

        let cols = 2;
        let c_size: usize = size + size - 1;

        let a = crate::random_host_vec_znx(module.n(), cols, size, &mut source);
        let a = crate::upload_host_vec_znx::<BE>(&a);
        let mut c_big: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(1, c_size);

        let b = crate::random_host_vec_znx(module.n(), 1, size, &mut source);
        let b = crate::upload_host_vec_znx::<BE>(&b);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.cnv_by_const_apply_tmp_bytes(0, c_size, size, size));
        move || {
            let mut c_big_backend = c_big.to_backend_mut();
            let a_backend = crate::vec_znx_backend_ref::<BE>(&a);
            let b_backend = crate::vec_znx_backend_ref::<BE>(&b);
            module.cnv_by_const_apply(
                0,
                &mut c_big_backend,
                0,
                &a_backend,
                0,
                &b_backend,
                0,
                0,
                &mut scratch.borrow(),
            );
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let log_n: usize = sweep[0];
        let size: usize = sweep[1];
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x{}", 1 << log_n, size));
        let mut runner = runner(1 << log_n, size);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}
