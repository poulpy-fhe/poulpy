use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};

use poulpy_hal::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxDftAlloc, VmpApplyDft, VmpApplyDftTmpBytes, VmpApplyDftToDft,
        VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPrepare, VmpPrepareTmpBytes,
    },
    layouts::{
        Backend, Module, ScratchOwned, VecZnxDft, VecZnxDftToBackendMut, VmpPMat, VmpPMatToBackendMut, VmpPMatToBackendRef,
    },
    source::Source,
};

pub fn bench_vmp_prepare<B>(params: &crate::params::VmpSweepParams, c: &mut Criterion, label: &str)
where
    Module<B>: ModuleNew<B> + VmpPMatAlloc<B> + VmpPrepare<B> + VmpPrepareTmpBytes,
    B: Backend,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vmp_prepare::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(sweep: [usize; 5]) -> impl FnMut()
    where
        Module<B>: ModuleNew<B> + VmpPMatAlloc<B> + VmpPrepare<B> + VmpPrepareTmpBytes,
        B: Backend,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << sweep[0]);

        let rows: usize = sweep[1];
        let cols_in: usize = sweep[2];
        let cols_out: usize = sweep[3];
        let size: usize = sweep[4];

        let mut source: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vmp_prepare_tmp_bytes(rows, cols_in, cols_out, size));

        let mat = crate::random_host_mat_znx(module.n(), rows, cols_in, cols_out, size, &mut source);
        let mat = crate::upload_host_mat_znx::<B>(&mat);
        let mut pmat: VmpPMat<B::OwnedBuf, B> = module.vmp_pmat_alloc(rows, cols_in, cols_out, size);

        move || {
            let mut pmat_backend = pmat.to_backend_mut();
            let mat_backend = crate::mat_znx_backend_ref::<B>(&mat);
            module.vmp_prepare(&mut pmat_backend, &mat_backend, &mut scratch.borrow());
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let id = BenchmarkId::from_parameter(format!(
            "{}x({}x{})x({}x{})",
            1 << sweep[0],
            sweep[2],
            sweep[1],
            sweep[3],
            sweep[4]
        ));
        let mut runner = runner::<B>(*sweep);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vmp_apply_dft<B: Backend>(params: &crate::params::VmpSweepParams, c: &mut Criterion, label: &str)
where
    Module<B>: ModuleNew<B> + VmpApplyDftTmpBytes + VmpApplyDft<B> + VmpPMatAlloc<B> + VecZnxDftAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vmp_apply_dft::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(sweep: [usize; 5]) -> impl FnMut()
    where
        Module<B>: ModuleNew<B> + VmpApplyDftTmpBytes + VmpApplyDft<B> + VmpPMatAlloc<B> + VecZnxDftAlloc<B>,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << sweep[0]);

        let rows: usize = sweep[1];
        let cols_in: usize = sweep[2];
        let cols_out: usize = sweep[3];
        let size: usize = sweep[4];

        let mut source: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<B> =
            ScratchOwned::alloc(module.vmp_apply_dft_tmp_bytes(size, size, rows, cols_in, cols_out, size));

        let mut res: VecZnxDft<B::OwnedBuf, B> = module.vec_znx_dft_alloc(cols_out, size);
        let a = crate::random_host_vec_znx(module.n(), cols_in, size, &mut source);
        let a = crate::upload_host_vec_znx::<B>(&a);
        let pmat: VmpPMat<B::OwnedBuf, B> =
            crate::random_backend_vmp_pmat::<B>(module.n(), rows, cols_in, cols_out, size, &mut source);

        move || {
            let pmat = pmat.to_backend_ref();
            let a = crate::vec_znx_backend_ref::<B>(&a);
            module.vmp_apply_dft(&mut res, &a, &pmat, &mut scratch.borrow());
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let id = BenchmarkId::from_parameter(format!(
            "{}x({}x{})x({}x{})",
            1 << sweep[0],
            sweep[2],
            sweep[1],
            sweep[3],
            sweep[4]
        ));
        let mut runner = runner::<B>(*sweep);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vmp_apply_dft_to_dft<B: Backend>(params: &crate::params::VmpSweepParams, c: &mut Criterion, label: &str)
where
    Module<B>: ModuleNew<B> + VecZnxDftAlloc<B> + VmpPMatAlloc<B> + VmpApplyDftToDft<B> + VmpApplyDftToDftTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vmp_apply_dft_to_dft::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(sweep: [usize; 5]) -> impl FnMut()
    where
        Module<B>: ModuleNew<B> + VecZnxDftAlloc<B> + VmpPMatAlloc<B> + VmpApplyDftToDft<B> + VmpApplyDftToDftTmpBytes,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << sweep[0]);

        let rows: usize = sweep[1];
        let cols_in: usize = sweep[2];
        let cols_out: usize = sweep[3];
        let size: usize = sweep[4];

        let mut source: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<B> =
            ScratchOwned::alloc(module.vmp_apply_dft_to_dft_tmp_bytes(size, size, rows, cols_in, cols_out, size));

        let mut res: VecZnxDft<B::OwnedBuf, B> = module.vec_znx_dft_alloc(cols_out, size);
        let a: VecZnxDft<B::OwnedBuf, B> = crate::random_backend_vec_znx_dft::<B>(module.n(), cols_in, size, &mut source);
        let pmat: VmpPMat<B::OwnedBuf, B> =
            crate::random_backend_vmp_pmat::<B>(module.n(), rows, cols_in, cols_out, size, &mut source);

        move || {
            let pmat = pmat.to_backend_ref();
            let a = crate::vec_znx_dft_backend_ref::<B>(&a);
            module.vmp_apply_dft_to_dft(&mut res.to_backend_mut(), &a, &pmat, 0, &mut scratch.borrow());
            black_box(());
        }
    }

    for sweep in &params.sweeps {
        let id = BenchmarkId::from_parameter(format!(
            "{}x({}x{})x({}x{})",
            1 << sweep[0], // n
            sweep[2],      // cols_in
            sweep[1],      // size_in (=rows)
            sweep[3],      // cols_out
            sweep[4]       // size_out
        ));
        let mut runner = runner::<B>(*sweep);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}
