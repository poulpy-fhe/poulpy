use std::hint::black_box;

use criterion::{Bencher, measurement::Measurement};

use poulpy_hal::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxDftAlloc, VmpApplyDft, VmpApplyDftTmpBytes, VmpApplyDftToDft,
        VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPrepare, VmpPrepareTmpBytes,
    },
    layouts::{
        Backend, Module, ScratchOwned, VecZnxDftOwned, VecZnxDftToBackendMut, VmpPMatOwned, VmpPMatToBackendMut,
        VmpPMatToBackendRef,
    },
    source::Source,
};

use crate::hal::helpers::{
    mat_znx_backend_ref, random_backend_vec_znx_dft, random_backend_vmp_pmat, random_host_mat_znx, random_host_vec_znx,
    upload_host_mat_znx, upload_host_vec_znx, vec_znx_backend_ref, vec_znx_dft_backend_ref,
};
use crate::params::VmpSweepParms;

pub fn runner_vmp_prepare<B, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &VmpSweepParms)
where
    Module<B>: ModuleNew<B> + VmpPMatAlloc<B> + VmpPrepare<B> + VmpPrepareTmpBytes,
    B: Backend<ZnxWord = i64>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> =
        ScratchOwned::alloc(module.vmp_prepare_tmp_bytes(sweep.rows, sweep.cols_in, sweep.cols_out, sweep.size));

    let mat = random_host_mat_znx(module.n(), sweep.rows, sweep.cols_in, sweep.cols_out, sweep.size, &mut source);
    let mat = upload_host_mat_znx::<B>(&mat);
    let mut pmat: VmpPMatOwned<B> = module.vmp_pmat_alloc(sweep.rows, sweep.cols_in, sweep.cols_out, sweep.size);

    bencher.iter(|| {
        let mut pmat_backend = pmat.to_backend_mut();
        let mat_backend = mat_znx_backend_ref::<B>(&mat);
        module.vmp_prepare(&mut pmat_backend, &mat_backend, &mut scratch.borrow());
        black_box(());
    });
}

pub fn runner_vmp_apply_dft<B: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &VmpSweepParms)
where
    Module<B>: ModuleNew<B> + VmpApplyDftTmpBytes + VmpApplyDft<B> + VmpPMatAlloc<B> + VecZnxDftAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vmp_apply_dft_tmp_bytes(
        sweep.size,
        sweep.size,
        sweep.rows,
        sweep.cols_in,
        sweep.cols_out,
        sweep.size,
    ));

    let mut res: VecZnxDftOwned<B> = module.vec_znx_dft_alloc(sweep.cols_out, sweep.size);
    let a = random_host_vec_znx(module.n(), sweep.cols_in, sweep.size, &mut source);
    let a = upload_host_vec_znx::<B>(&a);
    let pmat: VmpPMatOwned<B> =
        random_backend_vmp_pmat::<B>(module.n(), sweep.rows, sweep.cols_in, sweep.cols_out, sweep.size, &mut source);

    bencher.iter(|| {
        let pmat = pmat.to_backend_ref();
        let a = vec_znx_backend_ref::<B>(&a);
        module.vmp_apply_dft(&mut res, &a, &pmat, &mut scratch.borrow());
        black_box(());
    });
}

pub fn runner_vmp_apply_dft_to_dft<B: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &VmpSweepParms)
where
    Module<B>: ModuleNew<B> + VecZnxDftAlloc<B> + VmpPMatAlloc<B> + VmpApplyDftToDft<B> + VmpApplyDftToDftTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source: Source = Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vmp_apply_dft_to_dft_tmp_bytes(
        sweep.size,
        sweep.size,
        sweep.rows,
        sweep.cols_in,
        sweep.cols_out,
        sweep.size,
    ));

    let mut res: VecZnxDftOwned<B> = module.vec_znx_dft_alloc(sweep.cols_out, sweep.size);
    let a: VecZnxDftOwned<B> = random_backend_vec_znx_dft::<B>(module.n(), sweep.cols_in, sweep.size, &mut source);
    let pmat: VmpPMatOwned<B> =
        random_backend_vmp_pmat::<B>(module.n(), sweep.rows, sweep.cols_in, sweep.cols_out, sweep.size, &mut source);

    bencher.iter(|| {
        let pmat = pmat.to_backend_ref();
        let a = vec_znx_dft_backend_ref::<B>(&a);
        module.vmp_apply_dft_to_dft(&mut res.to_backend_mut(), &a, &pmat, 0, &mut scratch.borrow());
        black_box(());
    });
}
