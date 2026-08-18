use poulpy_core::{
    GLWEExternalProduct,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGSW, GGSWAtBackendMut, GGSWLayout, GLWELayout, ModuleCoreAlloc, Rank, TorusPrecision,
        prepared::{GGSWPrepared, GGSWPreparedFactory},
    },
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxFillUniformSourceBackend},
    layouts::{Backend, Module, ScratchOwned},
    source::Source,
};
use std::hint::black_box;

use criterion::{Bencher, measurement::Measurement};

use crate::core::fill::{fill_ggsw, fill_glwe};
use crate::core::params::{CoreParams, key_dnum_k_aux};

fn layouts(cp: &CoreParams) -> (GLWELayout, GGSWLayout) {
    let glwe_infos = GLWELayout {
        n: Degree(cp.n),
        base2k: Base2K(cp.base2k),
        k: TorusPrecision(cp.k),
        rank: Rank(cp.rank),
    };
    let (dnum, k_aux) = key_dnum_k_aux(cp.k, cp.base2k, cp.dsize);
    let ggsw_infos = GGSWLayout {
        n: Degree(cp.n),
        base2k: Base2K(cp.base2k),
        k_aux: TorusPrecision(k_aux),
        rank: Rank(cp.rank),
        dnum: Dnum(dnum),
        dsize: Dsize(cp.dsize),
    };
    (glwe_infos, ggsw_infos)
}

/// Times the GLWE x GGSW external product.
///
/// Operands are uniform noise filled through the backend; see [`crate::core::fill`].
pub fn runner_glwe_external_product<BE: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, cp: &CoreParams)
where
    Module<BE>: ModuleNew<BE>
        + GLWEExternalProduct<BE>
        + GGSWPreparedFactory<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>
        + VecZnxFillUniformSourceBackend<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    GGSW<BE::OwnedBuf, i64>: GGSWAtBackendMut<BE>,
{
    let (glwe_infos, ggsw_infos) = layouts(cp);

    let module: Module<BE> = Module::<BE>::new(cp.n as u64);
    let mut source = Source::new([0u8; 32]);

    let mut ct_glwe_in = module.glwe_alloc_from_infos(&glwe_infos);
    let mut ct_glwe_out = module.glwe_alloc_from_infos(&glwe_infos);
    let mut ct_ggsw: GGSW<BE::OwnedBuf, i64> = module.ggsw_alloc_from_infos(&ggsw_infos);
    fill_glwe(&module, &mut ct_glwe_in, &mut source);
    fill_ggsw(&module, &mut ct_ggsw, &mut source);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .ggsw_prepare_tmp_bytes(&ggsw_infos)
            .max(module.glwe_external_product_tmp_bytes(&glwe_infos, &glwe_infos, &ggsw_infos)),
    );

    let mut ggsw_prepared: GGSWPrepared<BE::OwnedBuf, BE> = module.ggsw_prepared_alloc_from_infos(&ct_ggsw);
    module.ggsw_prepare(&mut ggsw_prepared, &ct_ggsw, &mut scratch.borrow());

    bencher.iter(|| {
        module.glwe_external_product(&mut ct_glwe_out, &ct_glwe_in, &ggsw_prepared, &mut scratch.borrow());
        black_box(());
    });
}

/// Times the in-place GLWE x GGSW external product.
pub fn runner_glwe_external_product_assign<BE: Backend<ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CoreParams,
) where
    Module<BE>: ModuleNew<BE>
        + GLWEExternalProduct<BE>
        + GGSWPreparedFactory<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>
        + VecZnxFillUniformSourceBackend<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    GGSW<BE::OwnedBuf, i64>: GGSWAtBackendMut<BE>,
{
    let (glwe_infos, ggsw_infos) = layouts(cp);

    let module: Module<BE> = Module::<BE>::new(cp.n as u64);
    let mut source = Source::new([0u8; 32]);

    let mut ct_glwe = module.glwe_alloc_from_infos(&glwe_infos);
    let mut ct_ggsw: GGSW<BE::OwnedBuf, i64> = module.ggsw_alloc_from_infos(&ggsw_infos);
    fill_glwe(&module, &mut ct_glwe, &mut source);
    fill_ggsw(&module, &mut ct_ggsw, &mut source);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .ggsw_prepare_tmp_bytes(&ggsw_infos)
            .max(module.glwe_external_product_tmp_bytes(&glwe_infos, &glwe_infos, &ggsw_infos)),
    );

    let mut ggsw_prepared: GGSWPrepared<BE::OwnedBuf, BE> = module.ggsw_prepared_alloc_from_infos(&ct_ggsw);
    module.ggsw_prepare(&mut ggsw_prepared, &ct_ggsw, &mut scratch.borrow());

    bencher.iter(|| {
        module.glwe_external_product_assign(&mut ct_glwe, &ggsw_prepared, &mut scratch.borrow());
        black_box(());
    });
}
