use std::hint::black_box;

use criterion::{Bencher, measurement::Measurement};

use poulpy_hal::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddAssignBackend, VecZnxAlloc, VecZnxBigAddAssign,
        VecZnxBigAddInto, VecZnxBigAddSmallAssign, VecZnxBigAddSmallIntoBackend, VecZnxBigAlloc, VecZnxBigAutomorphism,
        VecZnxBigAutomorphismAssign, VecZnxBigAutomorphismAssignTmpBytes, VecZnxBigNegate, VecZnxBigNegateAssign,
        VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxBigSub, VecZnxBigSubAssign, VecZnxBigSubNegateAssign,
        VecZnxBigSubSmallABackend, VecZnxBigSubSmallBBackend, VecZnxSubAssignBackend,
    },
    layouts::{Backend, Module, ScratchOwned, VecZnxBigOwned, VecZnxBigToBackendMut, VecZnxBigToBackendRef},
    source::Source,
};

use crate::hal::helpers::{
    random_backend_vec_znx_big, random_host_vec_znx, upload_host_vec_znx, vec_znx_backend_mut, vec_znx_backend_ref,
};
use crate::params::HalSweepParms;

pub fn runner_vec_znx_big_add_into<B: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &HalSweepParms)
where
    Module<B>: VecZnxBigAddInto<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source: Source = Source::new([0u8; 32]);

    let a: VecZnxBigOwned<B> = random_backend_vec_znx_big::<B>(module.n(), sweep.cols, sweep.size, &mut source);
    let b: VecZnxBigOwned<B> = random_backend_vec_znx_big::<B>(module.n(), sweep.cols, sweep.size, &mut source);
    let mut c: VecZnxBigOwned<B> = random_backend_vec_znx_big::<B>(module.n(), sweep.cols, sweep.size, &mut source);

    bencher.iter(|| {
        let a = a.to_backend_ref();
        let b = b.to_backend_ref();
        let mut c = c.to_backend_mut();
        for i in 0..sweep.cols {
            module.vec_znx_big_add_into(&mut c, i, &a, i, &b, i);
        }
        black_box(());
    });
}

pub fn runner_vec_znx_big_add_assign<B: Backend<ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    sweep: &HalSweepParms,
) where
    Module<B>: VecZnxBigAddAssign<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source: Source = Source::new([0u8; 32]);

    let a: VecZnxBigOwned<B> = random_backend_vec_znx_big::<B>(module.n(), sweep.cols, sweep.size, &mut source);
    let mut c: VecZnxBigOwned<B> = random_backend_vec_znx_big::<B>(module.n(), sweep.cols, sweep.size, &mut source);

    bencher.iter(|| {
        let a = a.to_backend_ref();
        let mut c = c.to_backend_mut();
        for i in 0..sweep.cols {
            module.vec_znx_big_add_assign(&mut c, i, &a, i);
        }
        black_box(());
    });
}

pub fn runner_vec_znx_big_add_small_into<B: Backend<ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    sweep: &HalSweepParms,
) where
    Module<B>: VecZnxBigAddSmallIntoBackend<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source: Source = Source::new([0u8; 32]);

    let a: VecZnxBigOwned<B> = random_backend_vec_znx_big::<B>(module.n(), sweep.cols, sweep.size, &mut source);
    let b = random_host_vec_znx(module.n(), sweep.cols, sweep.size, &mut source);
    let b = upload_host_vec_znx::<B>(&b);
    let mut c: VecZnxBigOwned<B> = random_backend_vec_znx_big::<B>(module.n(), sweep.cols, sweep.size, &mut source);

    bencher.iter(|| {
        let a = a.to_backend_ref();
        let b = vec_znx_backend_ref::<B>(&b);
        let mut c = c.to_backend_mut();
        for i in 0..sweep.cols {
            module.vec_znx_big_add_small_into_backend(&mut c, i, &a, i, &b, i);
        }
        black_box(());
    });
}

pub fn runner_vec_znx_big_add_small_assign<B: Backend<ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    sweep: &HalSweepParms,
) where
    Module<B>: VecZnxBigAddSmallAssign<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source: Source = Source::new([0u8; 32]);

    let a = random_host_vec_znx(module.n(), sweep.cols, sweep.size, &mut source);
    let a = upload_host_vec_znx::<B>(&a);
    let mut c: VecZnxBigOwned<B> = random_backend_vec_znx_big::<B>(module.n(), sweep.cols, sweep.size, &mut source);

    bencher.iter(|| {
        let a = vec_znx_backend_ref::<B>(&a);
        let mut c = c.to_backend_mut();
        for i in 0..sweep.cols {
            module.vec_znx_big_add_small_assign(&mut c, i, &a, i);
        }
        black_box(());
    });
}

pub fn runner_vec_znx_big_automorphism<B: Backend<ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    sweep: &HalSweepParms,
) where
    Module<B>: VecZnxBigAutomorphism<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source: Source = Source::new([0u8; 32]);

    let a: VecZnxBigOwned<B> = random_backend_vec_znx_big::<B>(module.n(), sweep.cols, sweep.size, &mut source);
    let mut res: VecZnxBigOwned<B> = random_backend_vec_znx_big::<B>(module.n(), sweep.cols, sweep.size, &mut source);

    bencher.iter(|| {
        let a = a.to_backend_ref();
        let mut res = res.to_backend_mut();
        for i in 0..sweep.cols {
            module.vec_znx_big_automorphism(-7, &mut res, i, &a, i);
        }
        black_box(());
    });
}

pub fn runner_vec_znx_big_automorphism_assign<B: Backend<ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    sweep: &HalSweepParms,
) where
    Module<B>: VecZnxBigAutomorphismAssign<B> + ModuleNew<B> + VecZnxBigAutomorphismAssignTmpBytes + VecZnxBigAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source: Source = Source::new([0u8; 32]);

    let mut res: VecZnxBigOwned<B> = random_backend_vec_znx_big::<B>(module.n(), sweep.cols, sweep.size, &mut source);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_big_automorphism_assign_tmp_bytes());

    bencher.iter(|| {
        let mut res = res.to_backend_mut();
        for i in 0..sweep.cols {
            module.vec_znx_big_automorphism_assign(-7, &mut res, i, &mut scratch.borrow());
        }
        black_box(());
    });
}

pub fn runner_vec_znx_big_negate<B: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &HalSweepParms)
where
    Module<B>: VecZnxBigNegate<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source: Source = Source::new([0u8; 32]);

    let a: VecZnxBigOwned<B> = random_backend_vec_znx_big::<B>(module.n(), sweep.cols, sweep.size, &mut source);
    let mut b: VecZnxBigOwned<B> = random_backend_vec_znx_big::<B>(module.n(), sweep.cols, sweep.size, &mut source);

    bencher.iter(|| {
        let a = a.to_backend_ref();
        let mut b = b.to_backend_mut();
        for i in 0..sweep.cols {
            module.vec_znx_big_negate(&mut b, i, &a, i);
        }
        black_box(());
    });
}

pub fn runner_vec_znx_big_negate_assign<B: Backend<ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    sweep: &HalSweepParms,
) where
    Module<B>: VecZnxBigNegateAssign<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source: Source = Source::new([0u8; 32]);

    let mut a: VecZnxBigOwned<B> = random_backend_vec_znx_big::<B>(module.n(), sweep.cols, sweep.size, &mut source);

    bencher.iter(|| {
        let mut a = a.to_backend_mut();
        for i in 0..sweep.cols {
            module.vec_znx_big_negate_assign(&mut a, i);
        }
        black_box(());
    });
}

pub fn runner_vec_znx_big_normalize<B: Backend<ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    sweep: &HalSweepParms,
) where
    Module<B>: VecZnxBigNormalize<B> + ModuleNew<B> + VecZnxBigNormalizeTmpBytes + VecZnxBigAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let base2k: usize = 50;

    let mut source: Source = Source::new([0u8; 32]);

    let a: VecZnxBigOwned<B> = random_backend_vec_znx_big::<B>(module.n(), sweep.cols, sweep.size, &mut source);
    let res = random_host_vec_znx(module.n(), sweep.cols, sweep.size, &mut source);
    let mut res = upload_host_vec_znx::<B>(&res);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_big_normalize_tmp_bytes());

    bencher.iter(|| {
        let a = a.to_backend_ref();
        let mut res = vec_znx_backend_mut::<B>(&mut res);
        for i in 0..sweep.cols {
            module.vec_znx_big_normalize(&mut res, base2k, 0, i, &a, base2k, i, &mut scratch.borrow());
        }
        black_box(());
    });
}

pub fn runner_vec_znx_big_normalize_add_assign<B: Backend<ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    sweep: &HalSweepParms,
) where
    Module<B>: VecZnxAddAssignBackend<B>
        + VecZnxAlloc<B>
        + VecZnxBigNormalize<B>
        + ModuleNew<B>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxBigAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);
    let base2k: usize = 50;
    let mut source: Source = Source::new([0u8; 32]);

    let a: VecZnxBigOwned<B> = random_backend_vec_znx_big::<B>(module.n(), sweep.cols, sweep.size, &mut source);
    let res = random_host_vec_znx(module.n(), sweep.cols, sweep.size, &mut source);
    let mut res = upload_host_vec_znx::<B>(&res);
    let tmp = random_host_vec_znx(module.n(), 1, sweep.size, &mut source);
    let mut tmp = upload_host_vec_znx::<B>(&tmp);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_big_normalize_tmp_bytes());

    bencher.iter(|| {
        for i in 0..sweep.cols {
            let a = a.to_backend_ref();
            {
                let mut tmp_ref = vec_znx_backend_mut::<B>(&mut tmp);
                module.vec_znx_big_normalize(&mut tmp_ref, base2k, 0, 0, &a, base2k, i, &mut scratch.borrow());
            }

            let tmp_ref = vec_znx_backend_ref::<B>(&tmp);
            let mut res_ref = vec_znx_backend_mut::<B>(&mut res);
            module.vec_znx_add_assign_backend(&mut res_ref, i, &tmp_ref, 0);
        }
        black_box(());
    });
}

pub fn runner_vec_znx_big_normalize_sub_assign<B: Backend<ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    sweep: &HalSweepParms,
) where
    Module<B>: VecZnxAlloc<B>
        + VecZnxBigNormalize<B>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxBigAlloc<B>
        + VecZnxSubAssignBackend<B>
        + ModuleNew<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);
    let base2k: usize = 50;
    let mut source: Source = Source::new([0u8; 32]);

    let a: VecZnxBigOwned<B> = random_backend_vec_znx_big::<B>(module.n(), sweep.cols, sweep.size, &mut source);
    let res = random_host_vec_znx(module.n(), sweep.cols, sweep.size, &mut source);
    let mut res = upload_host_vec_znx::<B>(&res);
    let tmp = random_host_vec_znx(module.n(), 1, sweep.size, &mut source);
    let mut tmp = upload_host_vec_znx::<B>(&tmp);

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_big_normalize_tmp_bytes());

    bencher.iter(|| {
        for i in 0..sweep.cols {
            let a = a.to_backend_ref();
            {
                let mut tmp_ref = vec_znx_backend_mut::<B>(&mut tmp);
                module.vec_znx_big_normalize(&mut tmp_ref, base2k, 0, 0, &a, base2k, i, &mut scratch.borrow());
            }

            let tmp_ref = vec_znx_backend_ref::<B>(&tmp);
            let mut res_ref = vec_znx_backend_mut::<B>(&mut res);
            module.vec_znx_sub_assign_backend(&mut res_ref, i, &tmp_ref, 0);
        }
        black_box(());
    });
}

pub fn runner_vec_znx_big_sub<B: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &HalSweepParms)
where
    Module<B>: VecZnxBigSub<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source: Source = Source::new([0u8; 32]);

    let a: VecZnxBigOwned<B> = random_backend_vec_znx_big::<B>(module.n(), sweep.cols, sweep.size, &mut source);
    let b: VecZnxBigOwned<B> = random_backend_vec_znx_big::<B>(module.n(), sweep.cols, sweep.size, &mut source);
    let mut c: VecZnxBigOwned<B> = random_backend_vec_znx_big::<B>(module.n(), sweep.cols, sweep.size, &mut source);

    bencher.iter(|| {
        let a = a.to_backend_ref();
        let b = b.to_backend_ref();
        let mut c = c.to_backend_mut();
        for i in 0..sweep.cols {
            module.vec_znx_big_sub(&mut c, i, &a, i, &b, i);
        }
        black_box(());
    });
}

pub fn runner_vec_znx_big_sub_assign<B: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &HalSweepParms)
where
    Module<B>: VecZnxBigSubAssign<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source: Source = Source::new([0u8; 32]);

    let a: VecZnxBigOwned<B> = random_backend_vec_znx_big::<B>(module.n(), sweep.cols, sweep.size, &mut source);
    let mut c: VecZnxBigOwned<B> = random_backend_vec_znx_big::<B>(module.n(), sweep.cols, sweep.size, &mut source);

    bencher.iter(|| {
        let a = a.to_backend_ref();
        let mut c = c.to_backend_mut();
        for i in 0..sweep.cols {
            module.vec_znx_big_sub_assign(&mut c, i, &a, i);
        }
        black_box(());
    });
}

pub fn runner_vec_znx_big_sub_negate_assign<B: Backend<ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    sweep: &HalSweepParms,
) where
    Module<B>: VecZnxBigSubNegateAssign<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source: Source = Source::new([0u8; 32]);

    let a: VecZnxBigOwned<B> = random_backend_vec_znx_big::<B>(module.n(), sweep.cols, sweep.size, &mut source);
    let mut c: VecZnxBigOwned<B> = random_backend_vec_znx_big::<B>(module.n(), sweep.cols, sweep.size, &mut source);

    bencher.iter(|| {
        let a = a.to_backend_ref();
        let mut c = c.to_backend_mut();
        for i in 0..sweep.cols {
            module.vec_znx_big_sub_negate_assign(&mut c, i, &a, i);
        }
        black_box(());
    });
}

pub fn runner_vec_znx_big_sub_small_a<B: Backend<ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    sweep: &HalSweepParms,
) where
    Module<B>: VecZnxBigSubSmallABackend<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source: Source = Source::new([0u8; 32]);

    let a = random_host_vec_znx(module.n(), sweep.cols, sweep.size, &mut source);
    let a = upload_host_vec_znx::<B>(&a);
    let b: VecZnxBigOwned<B> = random_backend_vec_znx_big::<B>(module.n(), sweep.cols, sweep.size, &mut source);
    let mut c: VecZnxBigOwned<B> = random_backend_vec_znx_big::<B>(module.n(), sweep.cols, sweep.size, &mut source);

    bencher.iter(|| {
        let a = vec_znx_backend_ref::<B>(&a);
        let b = b.to_backend_ref();
        let mut c = c.to_backend_mut();
        for i in 0..sweep.cols {
            module.vec_znx_big_sub_small_a_backend(&mut c, i, &a, i, &b, i);
        }
        black_box(());
    });
}

pub fn runner_vec_znx_big_sub_small_b<B: Backend<ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    sweep: &HalSweepParms,
) where
    Module<B>: VecZnxBigSubSmallBBackend<B> + ModuleNew<B> + VecZnxBigAlloc<B>,
    B::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source: Source = Source::new([0u8; 32]);

    let a: VecZnxBigOwned<B> = random_backend_vec_znx_big::<B>(module.n(), sweep.cols, sweep.size, &mut source);
    let b = random_host_vec_znx(module.n(), sweep.cols, sweep.size, &mut source);
    let b = upload_host_vec_znx::<B>(&b);
    let mut c: VecZnxBigOwned<B> = random_backend_vec_znx_big::<B>(module.n(), sweep.cols, sweep.size, &mut source);

    bencher.iter(|| {
        let a = a.to_backend_ref();
        let b = vec_znx_backend_ref::<B>(&b);
        let mut c = c.to_backend_mut();
        for i in 0..sweep.cols {
            module.vec_znx_big_sub_small_b_backend(&mut c, i, &a, i, &b, i);
        }
        black_box(());
    });
}
