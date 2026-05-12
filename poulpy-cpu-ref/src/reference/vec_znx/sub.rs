use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};

use crate::{
    api::{ModuleNew, VecZnxAlloc, VecZnxSubAssignBackend, VecZnxSubBackend, VecZnxSubNegateAssignBackend},
    layouts::{
        Backend, FillUniform, HostDataMut, HostDataRef, Module, VecZnx, VecZnxBackendMut, VecZnxBackendRef, VecZnxToBackendMut,
        VecZnxToBackendRef, ZnxView, ZnxViewMut,
    },
    reference::znx::{ZnxCopy, ZnxNegate, ZnxNegateAssign, ZnxSub, ZnxSubAssign, ZnxSubNegateAssign, ZnxZero},
    source::Source,
};

pub fn vec_znx_sub<'r, 'a, BE>(
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_col: usize,
    b: &VecZnxBackendRef<'a, BE>,
    b_col: usize,
) where
    BE: Backend + ZnxSub + ZnxNegate + ZnxZero + ZnxCopy,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
        assert_eq!(b.n(), res.n());
    }

    let res_size: usize = res.size();
    let a_size: usize = a.size();
    let b_size: usize = b.size();

    if a_size <= b_size {
        let sum_size: usize = a_size.min(res_size);
        let cpy_size: usize = b_size.min(res_size);

        for j in 0..sum_size {
            BE::znx_sub(res.at_mut(res_col, j), a.at(a_col, j), b.at(b_col, j));
        }

        for j in sum_size..cpy_size {
            BE::znx_negate(res.at_mut(res_col, j), b.at(b_col, j));
        }

        for j in cpy_size..res_size {
            BE::znx_zero(res.at_mut(res_col, j));
        }
    } else {
        let sum_size: usize = b_size.min(res_size);
        let cpy_size: usize = a_size.min(res_size);

        for j in 0..sum_size {
            BE::znx_sub(res.at_mut(res_col, j), a.at(a_col, j), b.at(b_col, j));
        }

        for j in sum_size..cpy_size {
            BE::znx_copy(res.at_mut(res_col, j), a.at(a_col, j));
        }

        for j in cpy_size..res_size {
            BE::znx_zero(res.at_mut(res_col, j));
        }
    }
}

pub fn vec_znx_sub_assign<'r, 'a, BE>(
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_col: usize,
) where
    BE: Backend + ZnxSubAssign,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
    }

    let res_size: usize = res.size();
    let a_size: usize = a.size();

    let sum_size: usize = a_size.min(res_size);

    for j in 0..sum_size {
        BE::znx_sub_assign(res.at_mut(res_col, j), a.at(a_col, j));
    }
}

pub fn vec_znx_sub_negate_assign<'r, 'a, BE>(
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_col: usize,
) where
    BE: Backend + ZnxSubNegateAssign + ZnxNegateAssign,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
    }

    let res_size: usize = res.size();
    let a_size: usize = a.size();

    let sum_size: usize = a_size.min(res_size);

    for j in 0..sum_size {
        BE::znx_sub_negate_assign(res.at_mut(res_col, j), a.at(a_col, j));
    }

    for j in sum_size..res_size {
        BE::znx_negate_assign(res.at_mut(res_col, j));
    }
}

pub fn bench_vec_znx_sub<B>(c: &mut Criterion, label: &str)
where
    B: Backend<OwnedBuf = Vec<u8>>,
    Module<B>: VecZnxSubBackend<B> + ModuleNew<B> + VecZnxAlloc<B>,
{
    let group_name: String = format!("vec_znx_sub::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend<OwnedBuf = Vec<u8>>>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxSubBackend<B> + ModuleNew<B> + VecZnxAlloc<B>,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnx<Vec<u8>> = module.vec_znx_alloc(cols, size);
        let mut b: VecZnx<Vec<u8>> = module.vec_znx_alloc(cols, size);
        let mut c: VecZnx<Vec<u8>> = module.vec_znx_alloc(cols, size);

        // Fill a with random i64
        a.fill_uniform(50, &mut source);
        b.fill_uniform(50, &mut source);

        move || {
            for i in 0..cols {
                module.vec_znx_sub_backend(
                    &mut <VecZnx<Vec<u8>> as VecZnxToBackendMut<B>>::to_backend_mut(&mut c),
                    i,
                    &<VecZnx<Vec<u8>> as VecZnxToBackendRef<B>>::to_backend_ref(&a),
                    i,
                    &<VecZnx<Vec<u8>> as VecZnxToBackendRef<B>>::to_backend_ref(&b),
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

pub fn bench_vec_znx_sub_inplace<B>(c: &mut Criterion, label: &str)
where
    B: Backend<OwnedBuf = Vec<u8>>,
    Module<B>: VecZnxSubAssignBackend<B> + ModuleNew<B> + VecZnxAlloc<B>,
{
    let group_name: String = format!("vec_znx_sub_inplace::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend<OwnedBuf = Vec<u8>>>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxSubAssignBackend<B> + ModuleNew<B> + VecZnxAlloc<B>,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnx<Vec<u8>> = module.vec_znx_alloc(cols, size);
        let mut b: VecZnx<Vec<u8>> = module.vec_znx_alloc(cols, size);

        // Fill a with random i64
        a.fill_uniform(50, &mut source);
        b.fill_uniform(50, &mut source);

        move || {
            for i in 0..cols {
                module.vec_znx_sub_assign_backend(
                    &mut <VecZnx<Vec<u8>> as VecZnxToBackendMut<B>>::to_backend_mut(&mut b),
                    i,
                    &<VecZnx<Vec<u8>> as VecZnxToBackendRef<B>>::to_backend_ref(&a),
                    i,
                );
            }
            black_box(());
        }
    }

    for params in [[10, 2, 2], [11, 2, 4], [12, 2, 8], [13, 2, 16], [14, 2, 32]] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << params[0], params[1], params[2]));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}

pub fn bench_vec_znx_sub_negate_inplace<B>(c: &mut Criterion, label: &str)
where
    B: Backend<OwnedBuf = Vec<u8>>,
    Module<B>: VecZnxSubNegateAssignBackend<B> + ModuleNew<B> + VecZnxAlloc<B>,
{
    let group_name: String = format!("vec_znx_sub_negate_inplace::{label}");

    let mut group = c.benchmark_group(group_name);

    fn runner<B: Backend<OwnedBuf = Vec<u8>>>(params: [usize; 3]) -> impl FnMut()
    where
        Module<B>: VecZnxSubNegateAssignBackend<B> + ModuleNew<B> + VecZnxAlloc<B>,
    {
        let n: usize = 1 << params[0];
        let cols: usize = params[1];
        let size: usize = params[2];

        let module: Module<B> = Module::<B>::new(n as u64);

        let mut source: Source = Source::new([0u8; 32]);

        let mut a: VecZnx<Vec<u8>> = module.vec_znx_alloc(cols, size);
        let mut b: VecZnx<Vec<u8>> = module.vec_znx_alloc(cols, size);

        // Fill a with random i64
        a.fill_uniform(50, &mut source);
        b.fill_uniform(50, &mut source);

        move || {
            for i in 0..cols {
                module.vec_znx_sub_negate_assign_backend(
                    &mut <VecZnx<Vec<u8>> as VecZnxToBackendMut<B>>::to_backend_mut(&mut b),
                    i,
                    &<VecZnx<Vec<u8>> as VecZnxToBackendRef<B>>::to_backend_ref(&a),
                    i,
                );
            }
            black_box(());
        }
    }

    for params in [[10, 2, 2], [11, 2, 4], [12, 2, 8], [13, 2, 16], [14, 2, 32]] {
        let id: BenchmarkId = BenchmarkId::from_parameter(format!("{}x({}x{})", 1 << params[0], params[1], params[2]));
        let mut runner = runner::<B>(params);
        group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
    }

    group.finish();
}
