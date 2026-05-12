use poulpy_core::{
    DEFAULT_SIGMA_XE, EncryptionLayout, GGSWNoise, GLWEDecrypt, GLWEEncryptSk, GLWENoise,
    layouts::{GGSWInfos, GGSWLayout, GLWEInfos, GLWELayout, GLWESecretPreparedFactory, LWEInfos, prepared::GLWESecretPrepared},
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostBackend, HostDataMut, HostDataRef, Module, ScratchArena, ScratchOwned, Stats},
    source::Source,
};
use rand::Rng;

use crate::{
    bdd_arithmetic::{
        BDDKeyEncryptSk, BDDKeyPrepared, BDDKeyPreparedFactory, ExecuteBDDCircuit2WTo1W, FheUint, FheUintPrepare,
        FheUintPrepareDebug, FheUintPreparedDebug, FheUintPreparedEncryptSk, FheUintPreparedFactory,
        tests::test_suite::{TEST_GGSW_INFOS, TEST_GLWE_INFOS, TestContext},
    },
    blind_rotation::BlindRotationAlgo,
};

pub fn test_bdd_prepare<BRA: BlindRotationAlgo, BE: Backend<OwnedBuf = Vec<u8>> + HostBackend>(
    test_context: &TestContext<BRA, BE>,
) where
    Module<BE>: ModuleNew<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEDecrypt<BE>
        + GLWENoise<BE>
        + FheUintPreparedFactory<u32, BE>
        + FheUintPreparedEncryptSk<u32, BE>
        + FheUintPrepareDebug<BRA, u32, BE>
        + BDDKeyEncryptSk<BRA, BE>
        + BDDKeyPreparedFactory<BRA, BE>
        + GGSWNoise<BE>
        + FheUintPrepare<BRA, BE>
        + ExecuteBDDCircuit2WTo1W<BE>
        + GLWEEncryptSk<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> ScratchArena<'a, BE>: poulpy_core::ScratchArenaTakeCore<'a, BE>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let glwe_infos: GLWELayout = TEST_GLWE_INFOS;
    let ggsw_infos: GGSWLayout = TEST_GGSW_INFOS;

    let module: &Module<BE> = &test_context.module;
    let sk_glwe_prep: &GLWESecretPrepared<BE::OwnedBuf, BE> = &test_context.sk_glwe;
    let bdd_key_prepared: &BDDKeyPrepared<BE::OwnedBuf, BRA, BE> = &test_context.bdd_key;

    let mut source: Source = Source::new([6u8; 32]);

    let mut source_xa: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([3u8; 32]);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 22);

    let glwe_enc_infos = EncryptionLayout::new_from_default_sigma(glwe_infos).unwrap();

    // GLWE(value)
    let mut c_enc: FheUint<Vec<u8>, u32> = FheUint::alloc_from_infos(module, &glwe_infos);
    let value: u32 = source.next_u32();
    c_enc.encrypt_sk(
        module,
        value,
        sk_glwe_prep,
        &glwe_enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    // GGSW(0)
    let mut c_enc_prep_debug: FheUintPreparedDebug<Vec<u8>, u32> =
        FheUintPreparedDebug::<Vec<u8>, u32>::alloc_from_infos(module, &ggsw_infos);

    let mut scratch_2 = ScratchOwned::alloc(module.fhe_uint_prepare_tmp_bytes(7, 1, &c_enc_prep_debug, &c_enc, bdd_key_prepared));

    // GGSW(value)
    c_enc_prep_debug.prepare(module, &c_enc, bdd_key_prepared, &mut scratch_2.borrow());

    let max_noise = |col_i: usize| {
        let mut noise: f64 = -(ggsw_infos.size() as f64 * ggsw_infos.base2k().as_usize() as f64) + DEFAULT_SIGMA_XE.log2() + 2.0;
        noise += 0.5 * ggsw_infos.log_n() as f64;
        if col_i != 0 {
            noise += 0.5 * ggsw_infos.log_n() as f64
        }
        noise
    };

    for row in 0..c_enc_prep_debug.dnum().as_usize() {
        for col in 0..c_enc_prep_debug.rank().as_usize() + 1 {
            let stats: Vec<Stats> = c_enc_prep_debug.noise(module, row, col, value, sk_glwe_prep, &mut scratch.borrow());
            for (i, stat) in stats.iter().enumerate() {
                let noise_have: f64 = stat.std().log2();
                let noise_max: f64 = max_noise(col);
                assert!(
                    noise_have <= noise_max,
                    "bit: {i} noise_have: {noise_have} > noise_max: {noise_max}"
                )
            }
        }
    }
}
