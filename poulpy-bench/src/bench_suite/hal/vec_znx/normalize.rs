use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};
use rand::Rng;

use poulpy_hal::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalize, VecZnxNormalizeAssignBackend, VecZnxNormalizeTmpBytes,
    },
    layouts::{Backend, DataViewMut, Module, ScratchOwned, VecZnx, VecZnxToBackendMut, VecZnxToBackendRef},
    source::Source,
};

pub fn bench_vec_znx_normalize<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxNormalize<B> + ModuleNew<B> + VecZnxNormalizeTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    B::OwnedBuf: AsMut<[u8]>,
{
    let group_name: String = format!("vec_znx_normalize::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxNormalize<B> + ModuleNew<B> + VecZnxNormalizeTmpBytes,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
        B::OwnedBuf: AsMut<[u8]>,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let base2k: usize = 50;

        let mut source: Source = Source::new([0u8; 32]);

        let mut a = module.vec_znx_alloc(cols, size);
        let mut res = module.vec_znx_alloc(cols, size);
        source.fill_bytes(a.data_mut().as_mut());
        source.fill_bytes(res.data_mut().as_mut());

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_normalize_tmp_bytes());
        let res_offset: i64 = 0;
        move || {
            let a = <VecZnx<B::OwnedBuf> as VecZnxToBackendRef<B>>::to_backend_ref(&a);
            let mut res = <VecZnx<B::OwnedBuf> as VecZnxToBackendMut<B>>::to_backend_mut(&mut res);
            for i in 0..cols {
                module.vec_znx_normalize(&mut res, base2k, res_offset, i, &a, base2k, i, &mut scratch.borrow());
            }
            black_box(());
        }
    }

    for params in [[10, 2, 2], [11, 2, 4], [12, 2, 8], [13, 2, 16], [14, 2, 32]] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << params[0], params[1], params[2],));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_normalize_assign<B: Backend>(c: &mut Criterion, label: &str)
where
    Module<B>: VecZnxNormalizeAssignBackend<B> + ModuleNew<B> + VecZnxNormalizeTmpBytes,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    B::OwnedBuf: AsMut<[u8]>,
{
    let group_name: String = format!("vec_znx_normalize_assign::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxNormalizeAssignBackend<B> + ModuleNew<B> + VecZnxNormalizeTmpBytes,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
        B::OwnedBuf: AsMut<[u8]>,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let base2k: usize = 50;

        let mut source: Source = Source::new([0u8; 32]);

        let mut a = module.vec_znx_alloc(cols, size);
        source.fill_bytes(a.data_mut().as_mut());

        let mut scratch: ScratchOwned<B> = ScratchOwned::alloc(module.vec_znx_normalize_tmp_bytes());

        move || {
            let mut a = <VecZnx<B::OwnedBuf> as VecZnxToBackendMut<B>>::to_backend_mut(&mut a);
            for i in 0..cols {
                module.vec_znx_normalize_assign_backend(base2k, &mut a, i, &mut scratch.borrow());
            }
            black_box(());
        }
    }

    for params in [[10, 2, 2], [11, 2, 4], [12, 2, 8], [13, 2, 16], [14, 2, 32]] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << params[0], params[1], params[2],));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}
