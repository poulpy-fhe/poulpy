use std::hint::black_box;

use criterion::{Bencher, measurement::Measurement};

use poulpy_hal::{
    api::{CnvPVecAlloc, Convolution, ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxBigAlloc, VecZnxDftAlloc},
    layouts::{
        Backend, CnvPVecLOwned, CnvPVecLToBackendMut, CnvPVecLToBackendRef, CnvPVecROwned, CnvPVecRToBackendMut,
        CnvPVecRToBackendRef, Module, ScratchOwned, VecZnxBigOwned, VecZnxBigToBackendMut, VecZnxDftToBackendMut,
    },
    source::Source,
};

use crate::hal::helpers::{
    random_backend_cnv_pvec_left, random_backend_cnv_pvec_right, random_host_vec_znx, upload_host_vec_znx, vec_znx_backend_ref,
};
use crate::hal::params::CnvSweepParms;

pub fn runner_cnv_prepare_left<BE, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &CnvSweepParms)
where
    BE: Backend<ZnxWord = i64> + 'static,
    Module<BE>: ModuleNew<BE> + Convolution<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let mut source: Source = Source::new([0u8; 32]);

    let c_size: usize = sweep.size + sweep.size - 1;

    let module: Module<BE> = Module::<BE>::new(sweep.n as u64);

    let mut a_prep: CnvPVecLOwned<BE> = module.cnv_pvec_left_alloc(1, sweep.size);

    let a = random_host_vec_znx(module.n(), 1, sweep.size, &mut source);
    let a = upload_host_vec_znx::<BE>(&a);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.cnv_prepare_left_tmp_bytes(c_size, sweep.size));

    bencher.iter(|| {
        let mut a_prep_backend = a_prep.to_backend_mut();
        let a_backend = vec_znx_backend_ref::<BE, _>(&a);
        module.cnv_prepare_left(&mut a_prep_backend, &a_backend, !0i64, &mut scratch.borrow());
        black_box(());
    });
}

pub fn runner_cnv_prepare_right<BE, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &CnvSweepParms)
where
    BE: Backend<ZnxWord = i64> + 'static,
    Module<BE>: ModuleNew<BE> + Convolution<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let mut source: Source = Source::new([0u8; 32]);

    let c_size: usize = sweep.size + sweep.size - 1;

    let module: Module<BE> = Module::<BE>::new(sweep.n as u64);

    let mut a_prep: CnvPVecROwned<BE> = module.cnv_pvec_right_alloc(1, sweep.size);

    let a = random_host_vec_znx(module.n(), 1, sweep.size, &mut source);
    let a = upload_host_vec_znx::<BE>(&a);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.cnv_prepare_right_tmp_bytes(c_size, sweep.size));

    bencher.iter(|| {
        let mut a_prep_backend = a_prep.to_backend_mut();
        let a_backend = vec_znx_backend_ref::<BE, _>(&a);
        module.cnv_prepare_right(&mut a_prep_backend, &a_backend, !0i64, &mut scratch.borrow());
        black_box(());
    });
}

pub fn runner_cnv_apply_dft<BE, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &CnvSweepParms)
where
    BE: Backend<ZnxWord = i64> + 'static,
    Module<BE>: ModuleNew<BE> + Convolution<BE> + VecZnxDftAlloc<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let mut source: Source = Source::new([0u8; 32]);

    let c_size: usize = sweep.size + sweep.size - 1;

    let module: Module<BE> = Module::<BE>::new(sweep.n as u64);

    let a_prep: CnvPVecLOwned<BE> = random_backend_cnv_pvec_left::<BE>(module.n(), 1, sweep.size, &mut source);
    let b_prep: CnvPVecROwned<BE> = random_backend_cnv_pvec_right::<BE>(module.n(), 1, sweep.size, &mut source);
    let mut c_dft = module.vec_znx_dft_alloc(1, c_size);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .cnv_apply_dft_tmp_bytes(0, c_size, sweep.size, sweep.size)
            .max(module.cnv_prepare_left_tmp_bytes(c_size, sweep.size))
            .max(module.cnv_prepare_right_tmp_bytes(c_size, sweep.size)),
    );

    bencher.iter(|| {
        let mut c_dft_backend = c_dft.to_backend_mut();
        module.cnv_apply_dft(
            0,
            &mut c_dft_backend,
            0,
            &a_prep.to_backend_ref(),
            0,
            &b_prep.to_backend_ref(),
            0,
            &mut scratch.borrow(),
        );
        black_box(());
    });
}

pub fn runner_cnv_apply_dft_accumulate<BE, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &CnvSweepParms)
where
    BE: Backend<ZnxWord = i64> + 'static,
    Module<BE>: ModuleNew<BE> + Convolution<BE> + VecZnxDftAlloc<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let mut source: Source = Source::new([0u8; 32]);

    let c_size: usize = sweep.size + sweep.size - 1;

    let module: Module<BE> = Module::<BE>::new(sweep.n as u64);

    let a_prep: CnvPVecLOwned<BE> = random_backend_cnv_pvec_left::<BE>(module.n(), 1, sweep.size, &mut source);
    let b_prep: CnvPVecROwned<BE> = random_backend_cnv_pvec_right::<BE>(module.n(), 1, sweep.size, &mut source);
    let mut c_dft = module.vec_znx_dft_alloc(1, c_size);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .cnv_apply_dft_tmp_bytes(0, c_size, sweep.size, sweep.size)
            .max(module.cnv_prepare_left_tmp_bytes(c_size, sweep.size))
            .max(module.cnv_prepare_right_tmp_bytes(c_size, sweep.size)),
    );
    {
        let mut c_dft_backend = c_dft.to_backend_mut();
        module.cnv_apply_dft(
            0,
            &mut c_dft_backend,
            0,
            &a_prep.to_backend_ref(),
            0,
            &b_prep.to_backend_ref(),
            0,
            &mut scratch.borrow(),
        );
    }

    bencher.iter(|| {
        let mut c_dft_backend = c_dft.to_backend_mut();
        module.cnv_apply_dft_accumulate(
            0,
            &mut c_dft_backend,
            0,
            &a_prep.to_backend_ref(),
            0,
            &b_prep.to_backend_ref(),
            0,
            &mut scratch.borrow(),
        );
        black_box(());
    });
}

pub fn runner_cnv_pairwise_apply_dft<BE, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &CnvSweepParms)
where
    BE: Backend<ZnxWord = i64> + 'static,
    Module<BE>: ModuleNew<BE> + Convolution<BE> + VecZnxDftAlloc<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let mut source: Source = Source::new([0u8; 32]);

    let module: Module<BE> = Module::<BE>::new(sweep.n as u64);

    let cols = 2;
    let c_size: usize = sweep.size + sweep.size - 1;

    let a_prep: CnvPVecLOwned<BE> = random_backend_cnv_pvec_left::<BE>(module.n(), cols, sweep.size, &mut source);
    let b_prep: CnvPVecROwned<BE> = random_backend_cnv_pvec_right::<BE>(module.n(), cols, sweep.size, &mut source);
    let mut c_dft = module.vec_znx_dft_alloc(1, c_size);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .cnv_pairwise_apply_dft_tmp_bytes(0, c_size, sweep.size, sweep.size)
            .max(module.cnv_prepare_left_tmp_bytes(c_size, sweep.size))
            .max(module.cnv_prepare_right_tmp_bytes(c_size, sweep.size)),
    );

    bencher.iter(|| {
        let mut c_dft_backend = c_dft.to_backend_mut();
        module.cnv_pairwise_apply_dft(
            0,
            &mut c_dft_backend,
            0,
            &a_prep.to_backend_ref(),
            &b_prep.to_backend_ref(),
            0,
            1,
            &mut scratch.borrow(),
        );
        black_box(());
    });
}

pub fn runner_cnv_by_const_apply<BE, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &CnvSweepParms)
where
    BE: Backend<ZnxWord = i64> + 'static,
    Module<BE>: ModuleNew<BE> + Convolution<BE> + VecZnxBigAlloc<BE> + CnvPVecAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let mut source: Source = Source::new([0u8; 32]);

    let module: Module<BE> = Module::<BE>::new(sweep.n as u64);

    let cols = 2;
    let c_size: usize = sweep.size + sweep.size - 1;

    let a = random_host_vec_znx(module.n(), cols, sweep.size, &mut source);
    let a = upload_host_vec_znx::<BE>(&a);
    let mut c_big: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(1, c_size);

    let b = random_host_vec_znx(module.n(), 1, sweep.size, &mut source);
    let b = upload_host_vec_znx::<BE>(&b);

    let mut scratch: ScratchOwned<BE> =
        ScratchOwned::alloc(module.cnv_by_const_apply_tmp_bytes(0, c_size, sweep.size, sweep.size));

    bencher.iter(|| {
        let mut c_big_backend = c_big.to_backend_mut();
        let a_backend = vec_znx_backend_ref::<BE, _>(&a);
        let b_backend = vec_znx_backend_ref::<BE, _>(&b);
        module.cnv_by_const_apply(
            0,
            &mut c_big_backend,
            0,
            &a_backend,
            0,
            &b_backend,
            0,
            0,
            &mut scratch.borrow(),
        );
        black_box(());
    });
}
