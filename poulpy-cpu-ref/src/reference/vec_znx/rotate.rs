use std::{hint::black_box, mem::size_of};

use criterion::{BenchmarkId, Criterion};

use crate::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAlloc, VecZnxRotateAssignBackend, VecZnxRotateAssignTmpBytes,
        VecZnxRotateBackend,
    },
    layouts::{
        Backend, FillUniform, HostDataMut, HostDataRef, Module, ScratchOwned, VecZnx, VecZnxBackendMut, VecZnxBackendRef,
        VecZnxToBackendMut, VecZnxToBackendRef, ZnxView, ZnxViewMut,
    },
    reference::znx::{ZnxCopy, ZnxRotate, ZnxZero},
    source::Source,
};

pub fn vec_znx_rotate_assign_tmp_bytes(n: usize) -> usize {
    n * size_of::<i64>()
}

pub fn vec_znx_rotate<'r, 'a, BE>(
    p: i64,
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_col: usize,
) where
    BE: Backend + ZnxRotate + ZnxZero,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.n(), a.n())
    }

    let res_size: usize = res.size();
    let a_size: usize = a.size();

    let min_size: usize = res_size.min(a_size);

    for j in 0..min_size {
        BE::znx_rotate(p, res.at_mut(res_col, j), a.at(a_col, j))
    }

    for j in min_size..res_size {
        BE::znx_zero(res.at_mut(res_col, j));
    }
}

pub fn vec_znx_rotate_assign<'r, BE>(p: i64, res: &mut VecZnxBackendMut<'r, BE>, res_col: usize, tmp: &mut [i64])
where
    BE: Backend + ZnxRotate + ZnxCopy,
    BE::BufMut<'r>: HostDataMut,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.n(), tmp.len());
    }
    for j in 0..res.size() {
        BE::znx_rotate(p, tmp, res.at(res_col, j));
        BE::znx_copy(res.at_mut(res_col, j), tmp);
    }
}

pub fn bench_vec_znx_rotate<B>(c: &mut Criterion, label: &str)
where
    B: Backend<OwnedBuf = Vec<u8>> + 'static,
    Module<B>: VecZnxRotateBackend<B> + ModuleNew<B> + VecZnxAlloc<B>,
    for<'x> B: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
{
    let group_name: String = format!("vec_znx_rotate::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(params: [usize; 3]) -> impl FnMut()
    where
        B: Backend<OwnedBuf = Vec<u8>> + 'static,
        Module<B>: VecZnxRotateBackend<B> + ModuleNew<B> + VecZnxAlloc<B>,
        for<'x> B: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnx<Vec<u8>> = module.vec_znx_alloc(cols, size);
        let mut res: VecZnx<Vec<u8>> = module.vec_znx_alloc(cols, size);

        // Fill a with random i64
        a.fill_uniform(50, &mut source);
        res.fill_uniform(50, &mut source);

        move || {
            for i in 0..cols {
                module.vec_znx_rotate_backend(
                    -7,
                    &mut <VecZnx<Vec<u8>> as VecZnxToBackendMut<B>>::to_backend_mut(&mut res),
                    i,
                    &<VecZnx<Vec<u8>> as VecZnxToBackendRef<B>>::to_backend_ref(&a),
                    i,
                );
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

pub fn bench_vec_znx_rotate_inplace<B>(c: &mut Criterion, label: &str)
where
    B: Backend<OwnedBuf = Vec<u8>> + 'static,
    Module<B>: VecZnxRotateAssignBackend<B> + VecZnxRotateAssignTmpBytes + ModuleNew<B> + VecZnxAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    for<'x> B: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
{
    let group_name: String = format!("vec_znx_rotate_inplace::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(params: [usize; 3]) -> impl FnMut()
    where
        B: Backend<OwnedBuf = Vec<u8>> + 'static,
        Module<B>: VecZnxRotateAssignBackend<B> + ModuleNew<B> + VecZnxRotateAssignTmpBytes + VecZnxAlloc<B>,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
        for<'x> B: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut res: VecZnx<Vec<u8>> = module.vec_znx_alloc(cols, size);

        let mut scratch = ScratchOwned::alloc(module.vec_znx_rotate_assign_tmp_bytes());

        // Fill a with random i64
        res.fill_uniform(50, &mut source);

        move || {
            for i in 0..cols {
                module.vec_znx_rotate_assign_backend(
                    -7,
                    &mut <VecZnx<Vec<u8>> as VecZnxToBackendMut<B>>::to_backend_mut(&mut res),
                    i,
                    &mut scratch.borrow(),
                );
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
