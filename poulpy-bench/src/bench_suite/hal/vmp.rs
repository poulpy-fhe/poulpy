use poulpy_hal::layouts::VecZnxDftOwned;
use poulpy_hal::layouts::VecZnxDftToBackendMut;
use poulpy_hal::layouts::VmpPMatOwned;
use poulpy_hal::layouts::VmpPMatToBackendMut;
use poulpy_hal::layouts::VmpPMatToBackendRef;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};

use poulpy_hal::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxDftAlloc, VmpApplyPMatDftToDft, VmpApplyPMatDftToDftTmpBytes,
        VmpApplyPMatSmallToDft, VmpApplyPMatSmallToDftTmpBytes, VmpPMatAlloc, VmpPreparePMat, VmpPreparePMatTmpBytes,
    },
    layouts::{Backend, Module, ScratchOwned},
    source::Source,
};

pub fn bench_vmp_prepare_pmat<B>(params: &crate::params::VmpSweepParams, c: &mut Criterion, label: &str)
where
    Module<B>: ModuleNew<B> + VmpPMatAlloc<B> + VmpPreparePMat<B> + VmpPreparePMatTmpBytes,
    B: Backend<ZnxWord = i64>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vmp_prepare_pmat::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(sweep: [usize; 5]) -> impl FnMut()
    where
        Module<B>: ModuleNew<B> + VmpPMatAlloc<B> + VmpPreparePMat<B> + VmpPreparePMatTmpBytes,
        B: Backend<ZnxWord = i64>,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << sweep[0]);

        let rows: usize = sweep[1];
        let cols_in: usize = sweep[2];
        let cols_out: usize = sweep[3];
        let size: usize = sweep[4];

        let mut source: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vmp_prepare_pmat_tmp_bytes(rows, cols_in, cols_out, size));

        let mat = crate::random_host_mat_znx(module.n(), rows, cols_in, cols_out, size, &mut source);
        let mat = crate::upload_host_mat_znx::<B>(&mat);
        let mut pmat: VmpPMatOwned<B> = module.vmp_pmat_alloc(rows, cols_in, cols_out, size);

        move || {
            let mut pmat_backend = pmat.to_backend_mut();
            let mat_backend = crate::mat_znx_backend_ref::<B>(&mat);
            module.vmp_prepare_pmat(&mut pmat_backend, &mat_backend, &mut scratch.borrow());
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

pub fn bench_vmp_apply_pmat_small_to_dft<B: Backend<ZnxWord = i64>>(
    params: &crate::params::VmpSweepParams,
    c: &mut Criterion,
    label: &str,
) where
    Module<B>: ModuleNew<B> + VmpApplyPMatSmallToDftTmpBytes + VmpApplyPMatSmallToDft<B> + VmpPMatAlloc<B> + VecZnxDftAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vmp_apply_pmat_small_to_dft::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend<ZnxWord = i64>>(sweep: [usize; 5]) -> impl FnMut()
    where
        Module<B>:
            ModuleNew<B> + VmpApplyPMatSmallToDftTmpBytes + VmpApplyPMatSmallToDft<B> + VmpPMatAlloc<B> + VecZnxDftAlloc<B>,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << sweep[0]);

        let rows: usize = sweep[1];
        let cols_in: usize = sweep[2];
        let cols_out: usize = sweep[3];
        let size: usize = sweep[4];

        let mut source: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<B> =
            ScratchOwned::alloc(module.vmp_apply_pmat_small_to_dft_tmp_bytes(size, rows, cols_in, cols_out, size, size));

        let mut res: VecZnxDftOwned<B> = module.vec_znx_dft_alloc(cols_out, size);
        let a = crate::random_host_vec_znx(module.n(), cols_in, size, &mut source);
        let a = crate::upload_host_vec_znx::<B>(&a);
        let pmat: VmpPMatOwned<B> = crate::random_backend_vmp_pmat::<B>(module.n(), rows, cols_in, cols_out, size, &mut source);

        move || {
            let pmat = pmat.to_backend_ref();
            let a = crate::vec_znx_backend_ref::<B>(&a);
            module.vmp_apply_pmat_small_to_dft(&mut res.to_backend_mut(), &pmat, &a, &mut scratch.borrow());
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

pub fn bench_vmp_apply_pmat_dft_to_dft<B: Backend<ZnxWord = i64>>(
    params: &crate::params::VmpSweepParams,
    c: &mut Criterion,
    label: &str,
) where
    Module<B>: ModuleNew<B> + VecZnxDftAlloc<B> + VmpPMatAlloc<B> + VmpApplyPMatDftToDft<B> + VmpApplyPMatDftToDftTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vmp_apply_pmat_dft_to_dft::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend<ZnxWord = i64>>(sweep: [usize; 5]) -> impl FnMut()
    where
        Module<B>: ModuleNew<B> + VecZnxDftAlloc<B> + VmpPMatAlloc<B> + VmpApplyPMatDftToDft<B> + VmpApplyPMatDftToDftTmpBytes,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let module: Module<B> = Module::<B>::new(1 << sweep[0]);

        let rows: usize = sweep[1];
        let cols_in: usize = sweep[2];
        let cols_out: usize = sweep[3];
        let size: usize = sweep[4];

        let mut source: Source = Source::new([0u8; 32]);

        let mut scratch: ScratchOwned<B> =
            ScratchOwned::alloc(module.vmp_apply_pmat_dft_to_dft_tmp_bytes(size, rows, cols_in, cols_out, size, size));

        let mut res: VecZnxDftOwned<B> = module.vec_znx_dft_alloc(cols_out, size);
        let a: VecZnxDftOwned<B> = crate::random_backend_vec_znx_dft::<B>(module.n(), cols_in, size, &mut source);
        let pmat: VmpPMatOwned<B> = crate::random_backend_vmp_pmat::<B>(module.n(), rows, cols_in, cols_out, size, &mut source);

        move || {
            let pmat = pmat.to_backend_ref();
            let a = crate::vec_znx_dft_backend_ref::<B>(&a);
            module.vmp_apply_pmat_dft_to_dft(&mut res.to_backend_mut(), &pmat, &a, 0, &mut scratch.borrow());
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
