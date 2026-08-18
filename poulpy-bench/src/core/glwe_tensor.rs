use std::hint::black_box;

use criterion::{Bencher, measurement::Measurement};
use poulpy_core::{
    GLWETensoring,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGLWEAtBackendMut, GLWELayout, GLWETensorKey, GLWETensorKeyLayout,
        GLWETensorKeyPreparedFactory, ModuleCoreAlloc, Rank, TorusPrecision,
    },
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxFillUniformSourceBackend},
    layouts::{Backend, Module, ScratchOwned},
    source::Source,
};

use crate::core::fill::{fill_gglwe, fill_glwe, fill_glwe_tensor};
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
/// Operands are uniform noise filled through the backend; see [`crate::core::fill`].
pub fn runner_glwe_tensor_relinearize<BE: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, cp: &CoreParams)
where
    Module<BE>: ModuleNew<BE>
        + GLWETensoring<BE>
        + GLWETensorKeyPreparedFactory<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>
        + VecZnxFillUniformSourceBackend<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    GLWETensorKey<BE::OwnedBuf, i64>: GGLWEAtBackendMut<BE>,
{
    let glwe_infos = glwe_layout(cp);
    let tsk_infos = tensor_key_layout(cp);

    let module = Module::<BE>::new(cp.n as u64);
    let mut source = Source::new([0u8; 32]);

    let mut res = module.glwe_alloc_from_infos(&glwe_infos);
    let mut tensor = module.glwe_tensor_alloc_from_infos(&glwe_infos);
    fill_glwe_tensor(&module, &mut tensor, &mut source);

    let mut tsk_coeffs: GLWETensorKey<BE::OwnedBuf, i64> = module.glwe_tensor_key_alloc_from_infos(&tsk_infos);
    fill_gglwe(&module, &mut tsk_coeffs, &mut source);

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
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>
        + VecZnxFillUniformSourceBackend<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let glwe_infos = glwe_layout(cp);
    let module = Module::<BE>::new(cp.n as u64);
    let mut source = Source::new([0u8; 32]);

    let mut a = module.glwe_alloc_from_infos(&glwe_infos);
    let mut b = module.glwe_alloc_from_infos(&glwe_infos);
    let mut tensor = module.glwe_tensor_alloc_from_infos(&glwe_infos);
    fill_glwe(&module, &mut a, &mut source);
    fill_glwe(&module, &mut b, &mut source);
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
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>
        + VecZnxFillUniformSourceBackend<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let glwe_infos = glwe_layout(cp);
    let module = Module::<BE>::new(cp.n as u64);
    let mut source = Source::new([0u8; 32]);

    let mut a = module.glwe_alloc_from_infos(&glwe_infos);
    let mut tensor = module.glwe_tensor_alloc_from_infos(&glwe_infos);
    fill_glwe(&module, &mut a, &mut source);
    let mut scratch = ScratchOwned::<BE>::alloc(module.glwe_tensor_square_apply_tmp_bytes(&tensor, &a));

    bencher.iter(|| {
        module.glwe_tensor_square_apply(0, &mut tensor, &a, &mut scratch.borrow());
        black_box(());
    });
}
