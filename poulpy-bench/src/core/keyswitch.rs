use poulpy_core::api::TransferInto;
use poulpy_core::layouts::prepared::GGLWEPreparedToBackendRef;
use poulpy_core::{
    GLWEKeyswitch,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGLWELayout, GLWELayout, ModuleCoreAlloc, Rank, TorusPrecision,
        prepared::GGLWEPreparedFactory,
    },
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, CopyFromHost, Module, ScratchOwned},
    source::Source,
};

use std::hint::black_box;

use criterion::{Bencher, measurement::Measurement};

use crate::core::fill::{host_gglwe, host_glwe, staging};
use crate::core::params::{CoreParams, key_dnum_k_aux};

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
pub fn runner_glwe_keyswitch<BE: Backend<ZnxWord = i64, OwnedBuf: CopyFromHost>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CoreParams,
) where
    Module<BE>:
        ModuleNew<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64> + GLWEKeyswitch<BE> + GGLWEPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
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

    let host = staging(cp.n as usize);
    let mut ct_in = module.glwe_alloc_from_infos(glwe_in);
    let mut ct_out = module.glwe_alloc_from_infos(glwe_out);
    let mut key_coeffs = module.gglwe_alloc_from_infos(&key_infos);

    host_glwe(&host, glwe_in, &mut source).transfer_into(&mut ct_in);
    host_gglwe(&host, &key_infos, &mut source).transfer_into(&mut key_coeffs);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .gglwe_prepare_tmp_bytes(&key_infos)
            .max(module.glwe_keyswitch_tmp_bytes(glwe_out, glwe_in, &key_infos)),
    );

    let mut key = module.gglwe_prepared_alloc_from_infos(&key_infos);
    module.gglwe_prepare(&mut key, &key_coeffs, &mut scratch.borrow());

    bencher.iter(|| {
        module.glwe_keyswitch(&mut ct_out, &ct_in, &key.to_backend_ref(), &mut scratch.borrow());
        black_box(());
    });
}
