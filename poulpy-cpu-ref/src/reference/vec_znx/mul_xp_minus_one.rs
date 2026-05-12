use std::{hint::black_box, mem::size_of};

use criterion::{BenchmarkId, Criterion};

use crate::{
    api::{
        ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAlloc, VecZnxMulXpMinusOneAssignBackend,
        VecZnxMulXpMinusOneAssignTmpBytes, VecZnxMulXpMinusOneBackend,
    },
    layouts::{
        Backend, FillUniform, HostDataMut, HostDataRef, Module, ScratchOwned, VecZnx, VecZnxBackendMut, VecZnxBackendRef,
        VecZnxToBackendMut, VecZnxToBackendRef, ZnxView, ZnxViewMut,
    },
    reference::{
        vec_znx::{vec_znx_rotate, vec_znx_sub_assign},
        znx::{ZnxNegate, ZnxRotate, ZnxSubAssign, ZnxSubNegateAssign, ZnxZero},
    },
    source::Source,
};

pub fn vec_znx_mul_xp_minus_one_assign_tmp_bytes(n: usize) -> usize {
    n * size_of::<i64>()
}

pub fn vec_znx_mul_xp_minus_one<'r, 'a, BE>(
    p: i64,
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_col: usize,
) where
    BE: Backend + ZnxRotate + ZnxZero + ZnxSubAssign,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    vec_znx_rotate::<BE>(p, res, res_col, a, a_col);
    vec_znx_sub_assign::<BE>(res, res_col, a, a_col);
}

pub fn vec_znx_mul_xp_minus_one_assign<'r, BE>(p: i64, res: &mut VecZnxBackendMut<'r, BE>, res_col: usize, tmp: &mut [i64])
where
    BE: Backend + ZnxRotate + ZnxNegate + ZnxSubNegateAssign,
    BE::BufMut<'r>: HostDataMut,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.n(), tmp.len());
    }
    for j in 0..res.size() {
        BE::znx_rotate(p, tmp, res.at(res_col, j));
        BE::znx_sub_negate_assign(res.at_mut(res_col, j), tmp);
    }
}

pub fn bench_vec_znx_mul_xp_minus_one<B>(c: &mut Criterion, label: &str)
where
    B: Backend<OwnedBuf = Vec<u8>>,
    Module<B>: VecZnxMulXpMinusOneBackend<B> + ModuleNew<B> + VecZnxAlloc<B>,
{
    let group_name: String = format!("vec_znx_mul_xp_minus_one::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(params: [usize; 3]) -> impl FnMut()
    where
        B: Backend<OwnedBuf = Vec<u8>>,
        Module<B>: VecZnxMulXpMinusOneBackend<B> + ModuleNew<B> + VecZnxAlloc<B>,
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
            let a_backend = <VecZnx<Vec<u8>> as VecZnxToBackendRef<B>>::to_backend_ref(&a);
            let mut res_backend = <VecZnx<Vec<u8>> as VecZnxToBackendMut<B>>::to_backend_mut(&mut res);
            for i in 0..cols {
                module.vec_znx_mul_xp_minus_one_backend(-7, &mut res_backend, i, &a_backend, i);
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

pub fn bench_vec_znx_mul_xp_minus_one_inplace<B>(c: &mut Criterion, label: &str)
where
    B: Backend<OwnedBuf = Vec<u8>>,
    Module<B>: VecZnxMulXpMinusOneAssignBackend<B> + VecZnxMulXpMinusOneAssignTmpBytes + ModuleNew<B> + VecZnxAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let group_name: String = format!("vec_znx_mul_xp_minus_one_inplace::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B>(params: [usize; 3]) -> impl FnMut()
    where
        B: Backend<OwnedBuf = Vec<u8>>,
        Module<B>: VecZnxMulXpMinusOneAssignBackend<B> + ModuleNew<B> + VecZnxMulXpMinusOneAssignTmpBytes + VecZnxAlloc<B>,
        ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut res: VecZnx<Vec<u8>> = module.vec_znx_alloc(cols, size);

        let mut scratch = ScratchOwned::alloc(module.vec_znx_mul_xp_minus_one_assign_tmp_bytes());

        // Fill a with random i64
        res.fill_uniform(50, &mut source);

        move || {
            let mut res_backend = <VecZnx<Vec<u8>> as VecZnxToBackendMut<B>>::to_backend_mut(&mut res);
            for i in 0..cols {
                module.vec_znx_mul_xp_minus_one_assign_backend(-7, &mut res_backend, i, &mut scratch.borrow());
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
