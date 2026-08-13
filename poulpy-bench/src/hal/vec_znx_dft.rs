use std::hint::black_box;

use criterion::{Bencher, measurement::Measurement};

use poulpy_hal::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxBigAlloc, VecZnxDftAddAssign, VecZnxDftAddInto, VecZnxDftAlloc,
        VecZnxDftApply, VecZnxDftSub, VecZnxDftSubAssign, VecZnxDftSubNegateAssign, VecZnxIdftApply, VecZnxIdftApplyTmpA,
        VecZnxIdftApplyTmpBytes,
    },
    layouts::{Backend, Module, ScratchOwned, VecZnxBig, VecZnxBigOwned, VecZnxDft, VecZnxDftOwned},
    source::Source,
};

use crate::hal::helpers::{
    random_backend_vec_znx_dft, random_host_vec_znx, upload_host_vec_znx, vec_znx_backend_ref, vec_znx_big_backend_mut,
    vec_znx_dft_backend_mut, vec_znx_dft_backend_ref,
};
use crate::params::HalSweepParms;

pub fn runner_vec_znx_dft_add_into<B: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &HalSweepParms)
where
    Module<B>: VecZnxDftAddInto<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source: Source = Source::new([0u8; 32]);

    let a: VecZnxDftOwned<B> = random_backend_vec_znx_dft::<B>(module.n(), sweep.cols, sweep.size, &mut source);
    let rhs: VecZnxDftOwned<B> = random_backend_vec_znx_dft::<B>(module.n(), sweep.cols, sweep.size, &mut source);
    let mut c: VecZnxDftOwned<B> = module.vec_znx_dft_alloc(sweep.cols, sweep.size);

    bencher.iter(|| {
        let a = vec_znx_dft_backend_ref::<B>(&a);
        let rhs = vec_znx_dft_backend_ref::<B>(&rhs);
        let mut c = vec_znx_dft_backend_mut::<B>(&mut c);
        for i in 0..sweep.cols {
            module.vec_znx_dft_add_into(&mut c, i, &a, i, &rhs, i);
        }
        black_box(());
    });
}

pub fn runner_vec_znx_dft_add_assign<B: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &HalSweepParms)
where
    Module<B>: VecZnxDftAddAssign<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source: Source = Source::new([0u8; 32]);

    let a: VecZnxDftOwned<B> = random_backend_vec_znx_dft::<B>(module.n(), sweep.cols, sweep.size, &mut source);
    let mut c: VecZnxDftOwned<B> = module.vec_znx_dft_alloc(sweep.cols, sweep.size);

    bencher.iter(|| {
        let a = vec_znx_dft_backend_ref::<B>(&a);
        let mut c = vec_znx_dft_backend_mut::<B>(&mut c);
        for i in 0..sweep.cols {
            module.vec_znx_dft_add_assign(&mut c, i, &a, i);
        }
        black_box(());
    });
}

pub fn runner_vec_znx_dft_apply<B: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &HalSweepParms)
where
    Module<B>: VecZnxDftApply<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source: Source = Source::new([0u8; 32]);

    let mut res: VecZnxDftOwned<B> = module.vec_znx_dft_alloc(sweep.cols, sweep.size);
    let a = random_host_vec_znx(module.n(), sweep.cols, sweep.size, &mut source);
    let a = upload_host_vec_znx::<B>(&a);

    bencher.iter(|| {
        let a = vec_znx_backend_ref::<B>(&a);
        let mut res = vec_znx_dft_backend_mut(&mut res);
        for i in 0..sweep.cols {
            module.vec_znx_dft_apply(1, 0, &mut res, i, &a, i);
        }
        black_box(());
    });
}

pub fn runner_vec_znx_idft_apply<B: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &HalSweepParms)
where
    Module<B>: VecZnxIdftApply<B> + ModuleNew<B> + VecZnxIdftApplyTmpBytes + VecZnxDftAlloc<B> + VecZnxBigAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source: Source = Source::new([0u8; 32]);

    let mut res: VecZnxBigOwned<B> = module.vec_znx_big_alloc(sweep.cols, sweep.size);
    let a: VecZnxDftOwned<B> = random_backend_vec_znx_dft::<B>(module.n(), sweep.cols, sweep.size, &mut source);

    let mut scratch = ScratchOwned::alloc(module.vec_znx_idft_apply_tmp_bytes());

    bencher.iter(|| {
        let a = vec_znx_dft_backend_ref::<B>(&a);
        let mut res = vec_znx_big_backend_mut(&mut res);
        for i in 0..sweep.cols {
            module.vec_znx_idft_apply(&mut res, i, &a, i, &mut scratch.borrow());
        }
        black_box(());
    });
}

pub fn runner_vec_znx_idft_apply_tmpa<B: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &HalSweepParms)
where
    Module<B>: VecZnxIdftApplyTmpA<B> + ModuleNew<B> + VecZnxDftAlloc<B> + VecZnxBigAlloc<B>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source: Source = Source::new([0u8; 32]);

    let mut res: VecZnxBigOwned<B> = module.vec_znx_big_alloc(sweep.cols, sweep.size);
    let mut a: VecZnxDftOwned<B> = random_backend_vec_znx_dft::<B>(module.n(), sweep.cols, sweep.size, &mut source);

    bencher.iter(|| {
        let mut res = vec_znx_big_backend_mut(&mut res);
        let mut a = vec_znx_dft_backend_mut(&mut a);
        for i in 0..sweep.cols {
            module.vec_znx_idft_apply_tmpa(&mut res, i, &mut a, i);
        }
        black_box(());
    });
}

pub fn runner_vec_znx_dft_sub<B: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &HalSweepParms)
where
    Module<B>: VecZnxDftSub<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source: Source = Source::new([0u8; 32]);

    let a: VecZnxDftOwned<B> = random_backend_vec_znx_dft::<B>(module.n(), sweep.cols, sweep.size, &mut source);
    let rhs: VecZnxDftOwned<B> = random_backend_vec_znx_dft::<B>(module.n(), sweep.cols, sweep.size, &mut source);
    let mut c: VecZnxDftOwned<B> = module.vec_znx_dft_alloc(sweep.cols, sweep.size);

    bencher.iter(|| {
        let a = vec_znx_dft_backend_ref::<B>(&a);
        let rhs = vec_znx_dft_backend_ref::<B>(&rhs);
        let mut c = vec_znx_dft_backend_mut::<B>(&mut c);
        for i in 0..sweep.cols {
            module.vec_znx_dft_sub(&mut c, i, &a, i, &rhs, i);
        }
        black_box(());
    });
}

pub fn runner_vec_znx_dft_sub_assign<B: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &HalSweepParms)
where
    Module<B>: VecZnxDftSubAssign<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source: Source = Source::new([0u8; 32]);

    let a: VecZnxDftOwned<B> = random_backend_vec_znx_dft::<B>(module.n(), sweep.cols, sweep.size, &mut source);
    let mut c: VecZnxDftOwned<B> = module.vec_znx_dft_alloc(sweep.cols, sweep.size);

    bencher.iter(|| {
        let a = vec_znx_dft_backend_ref::<B>(&a);
        let mut c = vec_znx_dft_backend_mut::<B>(&mut c);
        for i in 0..sweep.cols {
            module.vec_znx_dft_sub_assign(&mut c, i, &a, i);
        }
        black_box(());
    });
}

pub fn runner_vec_znx_dft_sub_negate_assign<B: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &HalSweepParms)
where
    Module<B>: VecZnxDftSubNegateAssign<B> + ModuleNew<B> + VecZnxDftAlloc<B>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source: Source = Source::new([0u8; 32]);

    let a: VecZnxDftOwned<B> = random_backend_vec_znx_dft::<B>(module.n(), sweep.cols, sweep.size, &mut source);
    let mut c: VecZnxDftOwned<B> = module.vec_znx_dft_alloc(sweep.cols, sweep.size);

    bencher.iter(|| {
        let a = vec_znx_dft_backend_ref::<B>(&a);
        let mut c = vec_znx_dft_backend_mut::<B>(&mut c);
        for i in 0..sweep.cols {
            module.vec_znx_dft_sub_negate_assign(&mut c, i, &a, i);
        }
        black_box(());
    });
}
