use poulpy_core::layouts::GLWEToBackendMut;
use poulpy_core::layouts::LWEInfos;
use poulpy_hal::api::VecZnxFillUniformSource;
use std::hint::black_box;

use criterion::{Bencher, measurement::Measurement};
use poulpy_core::{
    GLWEMaskFill, GLWETensoring,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GLWELayout, GLWETensorKey, GLWETensorKeyLayout, GLWETensorKeyPreparedFactory,
        ModuleCoreAlloc, Rank, TorusPrecision,
    },
    test_suite::keys::fill_by_digit,
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, ScratchOwned},
    source::Source,
};

use crate::core::params::{CoreParams, key_dnum_k_aux};

fn glwe_layout(cp: &CoreParams) -> GLWELayout {
    GLWELayout {
        n: Degree(cp.n),
        base2k: Base2K(cp.base2k),
        k: TorusPrecision(cp.k),
        rank: Rank(cp.rank),
    }
}

fn tensor_key_layout(cp: &CoreParams) -> GLWETensorKeyLayout {
    let (dnum, k_aux) = key_dnum_k_aux(cp.k, cp.base2k, cp.dsize);
    GLWETensorKeyLayout {
        n: Degree(cp.n),
        base2k: Base2K(cp.base2k),
        k_aux: TorusPrecision(k_aux),
        rank: Rank(cp.rank),
        dnum: Dnum(dnum),
        dsize: Dsize(cp.dsize),
    }
}

/// Relinearization (the keyswitch phase of `ckks_mul`).
///
/// Operands are uniform noise filled through the backend.
pub fn runner_glwe_tensor_relinearize<BE: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, cp: &CoreParams)
where
    Module<BE>: ModuleNew<BE>
        + GLWETensoring<BE>
        + GLWETensorKeyPreparedFactory<BE>
        + GLWEMaskFill<BE>
        + VecZnxFillUniformSource<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let glwe_infos = glwe_layout(cp);
    let tsk_infos = tensor_key_layout(cp);

    let module = Module::<BE>::new(cp.n as u64);
    let mut source = Source::new([0u8; 32]);

    let mut res = module.glwe_alloc_from_infos(&glwe_infos);
    let mut tensor = module.glwe_tensor_alloc_from_infos(&glwe_infos);
    module.vec_znx_fill_uniform_source(
        cp.base2k as usize,
        tensor.k().as_usize(),
        GLWEToBackendMut::<BE>::to_backend_mut(&mut tensor).data_mut(),
        0,
        &mut source,
    );
    module.fill_glwe_mask_from_source(&mut tensor, &mut source);

    let mut tsk_coeffs: GLWETensorKey<BE::OwnedBuf, i64> = module.glwe_tensor_key_alloc_from_infos(&tsk_infos);
    fill_by_digit(&module, &mut tsk_coeffs, 1, &mut source);

    let mut tsk = module.alloc_tensor_key_prepared_from_infos(&tsk_infos);
    let mut scratch = ScratchOwned::<BE>::alloc(
        module
            .prepare_tensor_key_tmp_bytes(&tsk_infos)
            .max(module.glwe_tensor_relinearize_tmp_bytes(&res, &tensor, &tsk)),
    );
    module.prepare_tensor_key(&mut tsk, &tsk_coeffs, &mut scratch.borrow());

    bencher.iter(|| {
        module.glwe_tensor_relinearize(&mut res, &tensor, &tsk, &mut scratch.borrow());
        black_box(());
    });
}

pub fn runner_glwe_tensor_apply<BE: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, cp: &CoreParams)
where
    Module<BE>: ModuleNew<BE>
        + GLWETensoring<BE>
        + GLWEMaskFill<BE>
        + VecZnxFillUniformSource<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let glwe_infos = glwe_layout(cp);
    let module = Module::<BE>::new(cp.n as u64);
    let mut source = Source::new([0u8; 32]);

    let mut a = module.glwe_alloc_from_infos(&glwe_infos);
    let mut b = module.glwe_alloc_from_infos(&glwe_infos);
    let mut tensor = module.glwe_tensor_alloc_from_infos(&glwe_infos);
    module.vec_znx_fill_uniform_source(
        cp.base2k as usize,
        a.k().as_usize(),
        GLWEToBackendMut::<BE>::to_backend_mut(&mut a).data_mut(),
        0,
        &mut source,
    );
    module.fill_glwe_mask_from_source(&mut a, &mut source);
    module.vec_znx_fill_uniform_source(
        cp.base2k as usize,
        b.k().as_usize(),
        GLWEToBackendMut::<BE>::to_backend_mut(&mut b).data_mut(),
        0,
        &mut source,
    );
    module.fill_glwe_mask_from_source(&mut b, &mut source);
    let mut scratch = ScratchOwned::<BE>::alloc(module.glwe_tensor_apply_tmp_bytes(&tensor, &a, &b));

    bencher.iter(|| {
        module.glwe_tensor_apply(0, &mut tensor, &a, &b, &mut scratch.borrow());
        black_box(());
    });
}

pub fn runner_glwe_tensor_square_apply<BE: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, cp: &CoreParams)
where
    Module<BE>: ModuleNew<BE>
        + GLWETensoring<BE>
        + GLWEMaskFill<BE>
        + VecZnxFillUniformSource<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let glwe_infos = glwe_layout(cp);
    let module = Module::<BE>::new(cp.n as u64);
    let mut source = Source::new([0u8; 32]);

    let mut a = module.glwe_alloc_from_infos(&glwe_infos);
    let mut tensor = module.glwe_tensor_alloc_from_infos(&glwe_infos);
    module.vec_znx_fill_uniform_source(
        cp.base2k as usize,
        a.k().as_usize(),
        GLWEToBackendMut::<BE>::to_backend_mut(&mut a).data_mut(),
        0,
        &mut source,
    );
    module.fill_glwe_mask_from_source(&mut a, &mut source);
    let mut scratch = ScratchOwned::<BE>::alloc(module.glwe_tensor_square_apply_tmp_bytes(&tensor, &a));

    bencher.iter(|| {
        module.glwe_tensor_square_apply(0, &mut tensor, &a, &mut scratch.borrow());
        black_box(());
    });
}
