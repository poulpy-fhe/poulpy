use std::hint::black_box;

use criterion::{Bencher, measurement::Measurement};

use poulpy_hal::{
    api::{ModuleNew, SvpApplyDft, SvpApplyDftToDft, SvpApplyDftToDftAssign, SvpPPolAlloc, SvpPrepare, VecZnxDftAlloc},
    layouts::{
        Backend, Module, SvpPPol, SvpPPolOwned, SvpPPolToBackendMut, SvpPPolToBackendRef, VecZnxDft, VecZnxDftOwned, VecZnxDftToBackendMut, VecZnxDftToBackendRef
    },
    source::Source,
};

use crate::hal::helpers::{
    random_backend_svp_ppol, random_backend_vec_znx_dft, random_host_scalar_znx, random_host_vec_znx, scalar_znx_backend_ref,
    upload_host_scalar_znx, upload_host_vec_znx, vec_znx_backend_ref,
};
use crate::params::HalSweepParms;

pub fn runner_svp_prepare<B, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &HalSweepParms)
where
    Module<B>: SvpPrepare<B> + SvpPPolAlloc<B> + ModuleNew<B>,
    B: Backend<ZnxWord = i64>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source = Source::new([0u8; 32]);

    let mut svp: SvpPPolOwned<B> = module.svp_ppol_alloc(sweep.cols);
    let a = random_host_scalar_znx(module.n(), sweep.cols, &mut source);
    let a = upload_host_scalar_znx::<B>(&a);

    bencher.iter(|| {
        let a_backend = scalar_znx_backend_ref::<B>(&a);
        module.svp_prepare(&mut svp.to_backend_mut(), 0, &a_backend, 0);
        black_box(());
    });
}

pub fn runner_svp_apply_dft<B, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &HalSweepParms)
where
    Module<B>: SvpApplyDft<B> + SvpPPolAlloc<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
    B: Backend<ZnxWord = i64>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);
    let mut source = Source::new([0u8; 32]);

    let svp: SvpPPolOwned<B> = random_backend_svp_ppol::<B>(module.n(), sweep.cols, &mut source);
    let mut res: VecZnxDftOwned<B> = module.vec_znx_dft_alloc(sweep.cols, sweep.size);
    let a = random_host_vec_znx(module.n(), sweep.cols, sweep.size, &mut source);
    let a = upload_host_vec_znx::<B>(&a);

    bencher.iter(|| {
        let svp = svp.to_backend_ref();
        let a = vec_znx_backend_ref::<B>(&a);
        let mut res = res.to_backend_mut();
        for j in 0..sweep.cols {
            module.svp_apply_dft(&mut res, j, &svp, j, &a, j);
        }
        black_box(());
    });
}

pub fn runner_svp_apply_dft_to_dft<B, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &HalSweepParms)
where
    Module<B>: SvpApplyDftToDft<B> + SvpPPolAlloc<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
    B: Backend<ZnxWord = i64>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);
    let mut source = Source::new([0u8; 32]);

    let svp: SvpPPolOwned<B> = random_backend_svp_ppol::<B>(module.n(), sweep.cols, &mut source);
    let mut res: VecZnxDftOwned<B> = module.vec_znx_dft_alloc(sweep.cols, sweep.size);
    let a: VecZnxDftOwned<B> = random_backend_vec_znx_dft::<B>(module.n(), sweep.cols, sweep.size, &mut source);

    bencher.iter(|| {
        let svp = svp.to_backend_ref();
        for j in 0..sweep.cols {
            module.svp_apply_dft_to_dft(&mut res.to_backend_mut(), j, &svp, j, &a.to_backend_ref(), j);
        }
        black_box(());
    });
}

pub fn runner_svp_apply_dft_to_dft_assign<B, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &HalSweepParms)
where
    Module<B>: SvpApplyDftToDftAssign<B> + SvpPPolAlloc<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
    B: Backend<ZnxWord = i64>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);
    let mut source = Source::new([0u8; 32]);

    let svp: SvpPPolOwned<B> = random_backend_svp_ppol::<B>(module.n(), sweep.cols, &mut source);
    let mut res: VecZnxDftOwned<B> = module.vec_znx_dft_alloc(sweep.cols, sweep.size);

    bencher.iter(|| {
        let svp = svp.to_backend_ref();
        for j in 0..sweep.cols {
            module.svp_apply_dft_to_dft_assign(&mut res.to_backend_mut(), j, &svp, j);
        }
        black_box(());
    });
}
