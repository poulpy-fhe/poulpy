use poulpy_core::api::TransferInto;
use poulpy_core::{
    GLWEAutomorphism,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGLWEAtBackendMut, GGLWELayout, GLWEAutomorphismKey, GLWELayout, ModuleCoreAlloc, Rank,
        SetGaloisElement, TorusPrecision,
        prepared::{GLWEAutomorphismKeyPrepared, GLWEAutomorphismKeyPreparedFactory},
    },
};
use poulpy_hal::layouts::CopyFromHost;
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, ScratchOwned},
    source::Source,
};
use std::hint::black_box;

use criterion::{Bencher, measurement::Measurement};

use crate::core::fill::{host_glwe, host_glwe_automorphism_key, staging};
use crate::core::params::{CoreParams, key_dnum_k_aux};

/// Times the GLWE automorphism with a fixed Galois element (`X -> X^3`).
///
/// Operands are uniform noise filled through the backend; see [`crate::core::fill`].
pub fn runner_glwe_automorphism<BE: Backend<ZnxWord = i64, OwnedBuf: CopyFromHost>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CoreParams,
) where
    Module<BE>: ModuleNew<BE>
        + GLWEAutomorphism<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    GLWEAutomorphismKey<BE::OwnedBuf, i64>: GGLWEAtBackendMut<BE>,
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
    };

    let module: Module<BE> = Module::<BE>::new(cp.n as u64);
    let mut source = Source::new([0u8; 32]);

    let host = staging(cp.n as usize);
    let mut atk: GLWEAutomorphismKey<BE::OwnedBuf, i64> = module.glwe_automorphism_key_alloc_from_infos(&atk_infos);
    host_glwe_automorphism_key(&host, &atk_infos, &mut source).transfer_into(&mut atk);
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
    host_glwe(&host, &glwe_infos, &mut source).transfer_into(&mut ct_in);

    bencher.iter(|| {
        module.glwe_automorphism(&mut ct_out, &ct_in, P, &atk_prepared, &mut scratch.borrow());
        black_box(());
    });
}
