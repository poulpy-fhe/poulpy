use std::hint::black_box;

use criterion::{Bencher, measurement::Measurement};
use rand::Rng;

use poulpy_hal::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalize, VecZnxNormalizeAssignBackend, VecZnxNormalizeTmpBytes,
    },
    layouts::{Backend, DataViewMut, Module, ScratchOwned, VecZnx, VecZnxToBackendMut, VecZnxToBackendRef},
    source::Source,
};

use crate::params::HalSweepParms;

pub fn runner_vec_znx_normalize<B: Backend, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &HalSweepParms)
where
    Module<B>: VecZnxNormalize<B> + ModuleNew<B> + VecZnxNormalizeTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    B::OwnedBuf: AsMut<[u8]>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let base2k: usize = 50;

    let mut source: Source = Source::new([0u8; 32]);

    let mut a = module.vec_znx_alloc(sweep.cols, sweep.size);
    let mut res = module.vec_znx_alloc(sweep.cols, sweep.size);
    source.fill_bytes(a.data_mut().as_mut());
    source.fill_bytes(res.data_mut().as_mut());

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_normalize_tmp_bytes());
    let res_offset: i64 = 0;

    bencher.iter(|| {
        let a = <VecZnx<B::OwnedBuf> as VecZnxToBackendRef<B>>::to_backend_ref(&a);
        let mut res = <VecZnx<B::OwnedBuf> as VecZnxToBackendMut<B>>::to_backend_mut(&mut res);
        for i in 0..sweep.cols {
            module.vec_znx_normalize(&mut res, base2k, res_offset, i, &a, base2k, i, &mut scratch.borrow());
        }
        black_box(());
    });
}

pub fn runner_vec_znx_normalize_assign<B: Backend, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &HalSweepParms)
where
    Module<B>: VecZnxNormalizeAssignBackend<B> + ModuleNew<B> + VecZnxNormalizeTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    B::OwnedBuf: AsMut<[u8]>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let base2k: usize = 50;

    let mut source: Source = Source::new([0u8; 32]);

    let mut a = module.vec_znx_alloc(sweep.cols, sweep.size);
    source.fill_bytes(a.data_mut().as_mut());

    let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_normalize_tmp_bytes());

    bencher.iter(|| {
        let mut a = <VecZnx<B::OwnedBuf> as VecZnxToBackendMut<B>>::to_backend_mut(&mut a);
        for i in 0..sweep.cols {
            module.vec_znx_normalize_assign_backend(base2k, &mut a, i, &mut scratch.borrow());
        }
        black_box(());
    });
}
