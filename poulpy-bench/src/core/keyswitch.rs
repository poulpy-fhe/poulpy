use poulpy_core::{
    GLWEKeyswitch,
    layouts::LWEInfos,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGLWE, GGLWEInfos, GGLWELayout, GLWE, GLWEInfos, GLWELayout, ModuleCoreAlloc, Rank,
        TorusPrecision, prepared::GGLWEPreparedFactory,
    },
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxFillUniformSourceBackend},
    layouts::{Backend, MatZnx, MatZnxAtBackendMut, Module, ScratchOwned, VecZnxBackendMut, vec_znx_backend_mut},
    source::Source,
};

use std::hint::black_box;

use criterion::{Bencher, measurement::Measurement};

use crate::core::params::{CoreParams, key_dnum_k_aux};

/// Fills one GLWE with uniform noise, through the backend.
///
/// Takes the backend view rather than the `GLWE` so that a GGLWE row, which is
/// a GLWE, can reuse it. The view carries its own column count: `rank + 1` for a
/// ciphertext, `rank_out + 1` for a key row.
fn fill_glwe_view<BE>(module: &Module<BE>, glwe: &mut VecZnxBackendMut<'_, BE>, base2k: usize, source: &mut Source)
where
    BE: Backend<ZnxWord = i64>,
    Module<BE>: VecZnxFillUniformSourceBackend<BE>,
{
    for col in 0..glwe.cols() {
        module.vec_znx_fill_uniform_source_backend(base2k, glwe, col, source);
    }
}

/// Fills a GLWE ciphertext with uniform noise.
fn fill_glwe<BE>(module: &Module<BE>, ct: &mut GLWE<BE::OwnedBuf, BE::ZnxWord>, source: &mut Source)
where
    BE: Backend<ZnxWord = i64>,
    Module<BE>: VecZnxFillUniformSourceBackend<BE>,
    GLWE<BE::OwnedBuf, BE::ZnxWord>: GLWEInfos,
{
    let base2k: usize = ct.base2k().into();
    fill_glwe_view(module, &mut vec_znx_backend_mut::<BE>(ct.data_mut()), base2k, source);
}

/// Fills a GGLWE with uniform noise, one GLWE row at a time.
fn fill_gglwe<BE>(module: &Module<BE>, key: &mut GGLWE<BE::OwnedBuf, BE::ZnxWord>, source: &mut Source)
where
    BE: Backend<ZnxWord = i64>,
    Module<BE>: VecZnxFillUniformSourceBackend<BE>,
    GGLWE<BE::OwnedBuf, BE::ZnxWord>: GGLWEInfos,
    MatZnx<BE::OwnedBuf, BE::ZnxWord>: MatZnxAtBackendMut<BE>,
{
    let base2k: usize = key.base2k().into();
    let rows: usize = key.dnum().as_usize();
    let cols_in: usize = key.rank_in().as_usize();
    let data = key.data_mut();
    for row in 0..rows {
        for col_in in 0..cols_in {
            fill_glwe_view(module, &mut data.at_backend_mut(row, col_in), base2k, source);
        }
    }
}

/// Times `glwe_keyswitch` alone.
///
/// The operands are uniform noise filled in through the backend rather than
/// genuine ciphertexts. Nothing is decrypted, and the arithmetic is
/// data-independent: an encryption of zero is uniform limbs too, so this is the
/// same input distribution reached without secrets, encryption or a transfer.
/// It is also what keeps the runner open to a device backend, which needs only
/// allocation, the prepared-key factory, the backend fill and the operation.
///
/// Setup is outside `bencher.iter`, so a device backend measures the kernel and
/// not the bus.
pub fn runner_glwe_keyswitch<BE: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, cp: &CoreParams)
where
    Module<BE>: ModuleNew<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>
        + GLWEKeyswitch<BE>
        + GGLWEPreparedFactory<BE>
        + VecZnxFillUniformSourceBackend<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    MatZnx<BE::OwnedBuf, i64>: MatZnxAtBackendMut<BE>,
{
    let glwe = GLWELayout {
        n: Degree(cp.n),
        base2k: Base2K(cp.base2k),
        k: TorusPrecision(cp.k),
        rank: Rank(cp.rank),
    };
    let (dnum, k_aux) = key_dnum_k_aux(cp.k + cp.dsize * cp.base2k, cp.base2k, cp.dsize);
    let key_infos = GGLWELayout {
        n: Degree(cp.n),
        base2k: Base2K(cp.base2k),
        k_aux: TorusPrecision(k_aux),
        rank_in: Rank(cp.rank),
        rank_out: Rank(cp.rank),
        dnum: Dnum(dnum),
        dsize: Dsize(cp.dsize),
    };

    let glwe_in = &glwe;
    let glwe_out = &glwe;

    let module: Module<BE> = Module::<BE>::new(cp.n as u64);
    let mut source: Source = Source::new([0u8; 32]);

    let mut ct_in = module.glwe_alloc_from_infos(glwe_in);
    let mut ct_out = module.glwe_alloc_from_infos(glwe_out);
    let mut key_coeffs = module.gglwe_alloc_from_infos(&key_infos);

    fill_glwe(&module, &mut ct_in, &mut source);
    fill_gglwe(&module, &mut key_coeffs, &mut source);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .gglwe_prepare_tmp_bytes(&key_infos)
            .max(module.glwe_keyswitch_tmp_bytes(glwe_out, glwe_in, &key_infos)),
    );

    let mut key = module.gglwe_prepared_alloc_from_infos(&key_infos);
    module.gglwe_prepare(&mut key, &key_coeffs, &mut scratch.borrow());

    bencher.iter(|| {
        module.glwe_keyswitch(&mut ct_out, &ct_in, &key, &mut scratch.borrow());
        black_box(());
    });
}
