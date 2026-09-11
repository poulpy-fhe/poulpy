use std::hint::black_box;

use criterion::{Bencher, measurement::Measurement};

// use poulpy_cpu_ref::reference::vec_znx::{vec_znx_lsh_tmp_bytes, vec_znx_rsh_tmp_bytes};
use poulpy_hal::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAlloc, VecZnxLsh, VecZnxLshAssign, VecZnxLshTmpBytes, VecZnxRsh,
        VecZnxRshAssign, VecZnxRshTmpBytes,
    },
    layouts::{Backend, Module, ScratchOwned},
};

use crate::hal::helpers::{random_host_vec_znx, upload_host_vec_znx, vec_znx_backend_mut, vec_znx_backend_ref};
use crate::hal::params::HalSweepParms;

pub fn runner_vec_znx_lsh_assign<B: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &HalSweepParms)
where
    Module<B>: VecZnxLshAssign<B> + VecZnxLshTmpBytes + ModuleNew<B> + VecZnxAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let base2k: usize = 50;

    let mut source = poulpy_hal::source::Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_lsh_tmp_bytes());

    let b = random_host_vec_znx(module.n(), sweep.cols, sweep.size, &mut source);
    let mut b = upload_host_vec_znx::<B>(&b);

    bencher.iter(|| {
        let mut b = vec_znx_backend_mut::<B>(&mut b);
        for i in 0..sweep.cols {
            module.vec_znx_lsh_assign(base2k, base2k - 1, &mut b, i, &mut scratch.borrow());
        }
        black_box(());
    });
}

pub fn runner_vec_znx_lsh<B: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &HalSweepParms)
where
    Module<B>: VecZnxLsh<B> + ModuleNew<B> + VecZnxAlloc<B> + VecZnxLshTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let base2k: usize = 50;

    let mut source = poulpy_hal::source::Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_lsh_tmp_bytes());

    let a = random_host_vec_znx(module.n(), sweep.cols, sweep.size, &mut source);
    let a = upload_host_vec_znx::<B>(&a);
    let mut res = module.vec_znx_alloc(sweep.cols, sweep.size);

    bencher.iter(|| {
        let a = vec_znx_backend_ref::<B>(&a);
        let mut res = vec_znx_backend_mut::<B>(&mut res);
        for i in 0..sweep.cols {
            module.vec_znx_lsh(base2k, base2k - 1, &mut res, i, &a, i, &mut scratch.borrow());
        }
        black_box(());
    });
}

pub fn runner_vec_znx_rsh_assign<B: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &HalSweepParms)
where
    Module<B>: VecZnxRshAssign<B> + ModuleNew<B> + VecZnxAlloc<B> + VecZnxRshTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let base2k: usize = 50;

    let mut source = poulpy_hal::source::Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_rsh_tmp_bytes());

    let b = random_host_vec_znx(module.n(), sweep.cols, sweep.size, &mut source);
    let mut b = upload_host_vec_znx::<B>(&b);

    bencher.iter(|| {
        let mut b = vec_znx_backend_mut::<B>(&mut b);
        for i in 0..sweep.cols {
            module.vec_znx_rsh_assign(base2k, base2k - 1, &mut b, i, &mut scratch.borrow());
        }
        black_box(());
    });
}

pub fn runner_vec_znx_rsh<B: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &HalSweepParms)
where
    Module<B>: VecZnxRsh<B> + ModuleNew<B> + VecZnxAlloc<B> + VecZnxRshTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let base2k: usize = 50;

    let mut source = poulpy_hal::source::Source::new([0u8; 32]);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_rsh_tmp_bytes());

    let a = random_host_vec_znx(module.n(), sweep.cols, sweep.size, &mut source);
    let a = upload_host_vec_znx::<B>(&a);
    let mut res = module.vec_znx_alloc(sweep.cols, sweep.size);

    bencher.iter(|| {
        let a = vec_znx_backend_ref::<B>(&a);
        let mut res = vec_znx_backend_mut::<B>(&mut res);
        for i in 0..sweep.cols {
            module.vec_znx_rsh(base2k, base2k - 1, &mut res, i, &a, i, &mut scratch.borrow());
        }
        black_box(());
    });
}
