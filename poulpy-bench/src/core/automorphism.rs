use poulpy_core::layouts::GLWEToBackendMut;
use poulpy_core::layouts::LWEInfos;
use poulpy_core::{
    GLWEAutomorphism, GLWEMaskFill,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGLWELayout, GLWEAutomorphismKey, GLWELayout, ModuleCoreAlloc, Rank, SetGaloisElement,
        TorusPrecision,
        prepared::{GLWEAutomorphismKeyPrepared, GLWEAutomorphismKeyPreparedFactory},
    },
    test_suite::keys::fill_by_digit,
};
use poulpy_hal::api::VecZnxFillUniformSource;
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, ScratchOwned},
    source::Source,
};
use std::hint::black_box;

use criterion::{Bencher, measurement::Measurement};

use crate::core::params::{CoreParams, key_dnum_k_aux};
use poulpy_core::layouts::prepared::GLWEAutomorphismKeyPreparedToBackendRef;

/// Times the GLWE automorphism with a fixed Galois element (`X -> X^3`).
///
/// Operands are uniform noise filled through the backend.
pub fn runner_glwe_automorphism<BE: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, cp: &CoreParams)
where
    Module<BE>: ModuleNew<BE>
        + GLWEAutomorphism<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWEMaskFill<BE>
        + VecZnxFillUniformSource<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    const P: i64 = 3;

    let glwe_infos = GLWELayout {
        n: Degree(cp.n),
        base2k: Base2K(cp.base2k),
        k: TorusPrecision(cp.k),
        rank: Rank(cp.rank),
    };
    let (dnum, k_aux) = key_dnum_k_aux(cp.k, cp.base2k, cp.dsize);
    let atk_infos = GGLWELayout {
        n: Degree(cp.n),
        base2k: Base2K(cp.base2k),
        k_aux: TorusPrecision(k_aux),
        rank_in: Rank(cp.rank),
        rank_out: Rank(cp.rank),
        dnum: Dnum(dnum),
        dsize: Dsize(cp.dsize),
        stride: 1,
    };

    let module: Module<BE> = Module::<BE>::new(cp.n as u64);
    let mut source = Source::new([0u8; 32]);

    let mut atk: GLWEAutomorphismKey<BE::OwnedBuf, i64> = module.glwe_automorphism_key_alloc_from_infos(&atk_infos);
    fill_by_digit(&module, &mut atk, 1, &mut source);
    atk.set_p(P);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .glwe_automorphism_key_prepare_tmp_bytes(&atk_infos)
            .max(module.glwe_automorphism_tmp_bytes(&glwe_infos, &glwe_infos, &atk_infos)),
    );

    let mut atk_prepared: GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE> =
        module.glwe_automorphism_key_prepared_alloc_from_infos(&atk);
    module.glwe_automorphism_key_prepare(&mut atk_prepared, &atk, &mut scratch.borrow());

    let mut ct_in = module.glwe_alloc_from_infos(&glwe_infos);
    let mut ct_out = module.glwe_alloc_from_infos(&glwe_infos);
    module.vec_znx_fill_uniform_source(
        cp.base2k as usize,
        ct_in.k().as_usize(),
        GLWEToBackendMut::<BE>::to_backend_mut(&mut ct_in).data_mut(),
        0,
        &mut source,
    );
    module.fill_glwe_mask_from_source(&mut ct_in, &mut source);

    bencher.iter(|| {
        module.glwe_automorphism(&mut ct_out, &ct_in, &atk_prepared.to_backend_ref(), &mut scratch.borrow());
        black_box(());
    });
}
