use std::hint::black_box;

use criterion::{Bencher, measurement::Measurement};
use rand::Rng;

use poulpy_hal::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAutomorphismAssignBackend, VecZnxAutomorphismAssignTmpBytes,
        VecZnxAutomorphismBackend,
    },
    layouts::{Backend, DataViewMut, Module, ScratchOwned, VecZnx, VecZnxToBackendMut, VecZnxToBackendRef},
    source::Source,
};

use crate::params::HalSweepParms;

pub fn runner_vec_znx_automorphism<B: Backend, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &HalSweepParms)
where
    Module<B>: VecZnxAutomorphismBackend<B> + ModuleNew<B>,
    B::OwnedBuf: AsMut<[u8]>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source: Source = Source::new([0u8; 32]);

    let mut a = module.vec_znx_alloc(sweep.cols, sweep.size);
    let mut res = module.vec_znx_alloc(sweep.cols, sweep.size);
    source.fill_bytes(a.data_mut().as_mut());
    source.fill_bytes(res.data_mut().as_mut());

    bencher.iter(|| {
        let a = <VecZnx<B::OwnedBuf, B::ZnxWord> as VecZnxToBackendRef<B>>::to_backend_ref(&a);
        let mut res = <VecZnx<B::OwnedBuf, B::ZnxWord> as VecZnxToBackendMut<B>>::to_backend_mut(&mut res);
        for i in 0..sweep.cols {
            module.vec_znx_automorphism_backend(-7, &mut res, i, &a, i);
        }
        black_box(());
    });
}

pub fn runner_vec_znx_automorphism_assign<B: Backend, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &HalSweepParms)
where
    Module<B>: VecZnxAutomorphismAssignBackend<B> + ModuleNew<B> + VecZnxAutomorphismAssignTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    B::OwnedBuf: AsMut<[u8]>,
{
    let module: Module<B> = Module::<B>::new(sweep.n as u64);

    let mut source: Source = Source::new([0u8; 32]);

    let mut scratch = ScratchOwned::alloc(module.vec_znx_automorphism_assign_tmp_bytes());

    let mut res = module.vec_znx_alloc(sweep.cols, sweep.size);
    source.fill_bytes(res.data_mut().as_mut());

    bencher.iter(|| {
        let mut res = <VecZnx<B::OwnedBuf, B::ZnxWord> as VecZnxToBackendMut<B>>::to_backend_mut(&mut res);
        for i in 0..sweep.cols {
            module.vec_znx_automorphism_assign_backend(-7, &mut res, i, &mut scratch.borrow());
        }
        black_box(());
    });
}
