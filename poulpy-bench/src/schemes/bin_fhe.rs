use std::hint::black_box;

use criterion::{Bencher, measurement::Measurement};
use poulpy_core::{
    EncryptionLayout, GGSWNoise, GLWEDecrypt, GLWEEncryptSk, GLWEExternalProduct, LWEEncryptSk,
    layouts::{
        Base2K, Dnum, Dsize, GGLWEToGGSWKeyLayout, GGSW, GGSWLayout, GGSWPreparedFactory, GLWE, GLWEAutomorphismKeyLayout,
        GLWELayout, GLWESecret, GLWESecretPrepared, GLWESecretPreparedFactory, GLWESecretSampling, LWE, LWEInfos, LWELayout,
        LWESecret, LWESecretSampling, ModuleCoreAlloc, TorusPrecision,
    },
};
use poulpy_hal::{
    api::{ModuleN, ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxRotateAssignBackend},
    layouts::{Backend, FillUniform, HostBackend, Module, ScratchOwned},
    source::Source,
};

use poulpy_bin_fhe::{
    blind_rotation::{
        BlindRotationAlgo, BlindRotationExecute, BlindRotationKey, BlindRotationKeyEncryptSk, BlindRotationKeyInfos,
        BlindRotationKeyLayout, BlindRotationKeyPrepared, BlindRotationKeyPreparedFactory, LookUpTableLayout, LookupTable,
        LookupTableFactory,
    },
    circuit_bootstrapping::{
        CircuitBootstrappingEncryptionInfos, CircuitBootstrappingExecute, CircuitBootstrappingKey,
        CircuitBootstrappingKeyEncryptSk, CircuitBootstrappingKeyLayout, CircuitBootstrappingKeyPrepared,
        CircuitBootstrappingKeyPreparedFactory,
    },
};

use crate::params::{BlindRotateBenchParams, CircuitBootstrappingBenchParam};

pub fn runner_blind_rotate<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, BRA: BlindRotationAlgo, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    params: &BlindRotateBenchParams,
) where
    Module<BE>: ModuleN
        + ModuleNew<BE>
        + BlindRotationKeyEncryptSk<BRA, BE>
        + BlindRotationKeyPreparedFactory<BRA, BE>
        + BlindRotationExecute<BRA, BE>
        + LookupTableFactory<BE::OwnedBuf, BE::ZnxWord>
        + GLWESecretPreparedFactory<BE>
        + GLWEDecrypt<BE>
        + LWEEncryptSk<BE>
        + GLWESecretSampling<BE>
        + LWESecretSampling<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let message_modulus: usize = 1 << params.log_message_modulus;

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 24);
    let module: Module<BE> = Module::<BE>::new(params.bin_fhe_params.n_glwe as u64);

    let mut source_xs: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([2u8; 32]);
    let mut source_xa: Source = Source::new([1u8; 32]);

    let brk_infos: BlindRotationKeyLayout = BlindRotationKeyLayout {
        n_glwe: params.bin_fhe_params.n_glwe.into(),
        n_lwe: params.bin_fhe_params.n_lwe.into(),
        base2k: Base2K(params.bin_fhe_params.base2k),
        k_aux: TorusPrecision(params.bin_fhe_params.k_aux),
        dnum: Dnum(1),
        rank: params.bin_fhe_params.rank.into(),
    };
    let glwe_infos: GLWELayout = GLWELayout {
        n: params.bin_fhe_params.n_glwe.into(),
        base2k: Base2K(params.bin_fhe_params.base2k),
        k: TorusPrecision(params.bin_fhe_params.k_aux),
        rank: params.bin_fhe_params.rank.into(),
    };
    let lwe_infos: LWELayout = LWELayout {
        n: params.bin_fhe_params.n_lwe.into(),
        k: TorusPrecision(params.bin_fhe_params.k_aux),
        base2k: Base2K(params.bin_fhe_params.base2k),
    };

    let mut sk_glwe: GLWESecret<Vec<u8>, i64> = module.glwe_secret_alloc_from_infos(&glwe_infos);
    module.glwe_secret_fill_ternary_prob(&mut sk_glwe, 0.5, &mut source_xs);
    let mut sk_glwe_dft: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&glwe_infos);
    module.glwe_secret_prepare(&mut sk_glwe_dft, &sk_glwe);

    let mut sk_lwe: LWESecret<Vec<u8>, i64> = module.lwe_secret_alloc(params.bin_fhe_params.n_lwe.into());
    module.lwe_secret_fill_binary_block(&mut sk_lwe, params.block_size, &mut source_xs);

    let brk_enc_infos = EncryptionLayout::new_from_default_sigma(brk_infos).unwrap();

    let mut brk: BlindRotationKey<BE::OwnedBuf, BRA, BE::ZnxWord> =
        BlindRotationKey::<BE::OwnedBuf, BRA, BE::ZnxWord>::alloc(&module, &brk_infos);
    module.blind_rotation_key_encrypt_sk(
        &mut brk,
        &sk_glwe_dft,
        &sk_lwe,
        &brk_enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    let mut brk_prepared: BlindRotationKeyPrepared<BE::OwnedBuf, BRA, BE> = BlindRotationKeyPrepared::alloc(&module, &brk);
    brk_prepared.prepare(&module, &brk, &mut scratch.borrow());

    let mut res: GLWE<Vec<u8>, i64> = module.glwe_alloc_from_infos(&glwe_infos);
    res.data_mut().fill_uniform(glwe_infos.base2k().as_usize(), &mut source_xa);
    let mut lwe: LWE<Vec<u8>, i64> = module.lwe_alloc_from_infos(&lwe_infos);
    lwe.fill_uniform(lwe_infos.base2k().as_usize(), &mut source_xa);

    let mut f_vec: Vec<i64> = vec![0i64; message_modulus];
    f_vec.iter_mut().enumerate().for_each(|(i, x)| *x = 2 * i as i64 + 1);

    let lut_infos = LookUpTableLayout {
        n: module.n().into(),
        extension_factor: params.extension_factor,
        k: TorusPrecision(2),
        base2k: Base2K(17),
    };
    let mut lut: LookupTable<BE::OwnedBuf, BE::ZnxWord> = LookupTable::alloc(&module, &lut_infos);
    lut.set(&module, &f_vec, params.log_message_modulus + 1);

    bencher.iter(|| {
        brk_prepared.execute(&module, &mut res, &lwe, &lut, &mut scratch.borrow());
        black_box(());
    });
}

pub fn runner_circuit_bootstrapping<
    BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64> + HostBackend,
    BRA: BlindRotationAlgo,
    M: Measurement,
>(
    bencher: &mut Bencher<'_, M>,
    params: &CircuitBootstrappingBenchParam,
) where
    Module<BE>: ModuleNew<BE>
        + ModuleN
        + GLWESecretPreparedFactory<BE>
        + GLWEExternalProduct<BE>
        + GLWEDecrypt<BE>
        + LWEEncryptSk<BE>
        + CircuitBootstrappingKeyEncryptSk<BRA, BE>
        + CircuitBootstrappingKeyPreparedFactory<BRA, BE>
        + CircuitBootstrappingExecute<BRA, BE>
        + GGSWPreparedFactory<BE>
        + GGSWNoise<BE>
        + GLWEEncryptSk<BE>
        + VecZnxRotateAssignBackend<BE>
        + GLWESecretSampling<BE>
        + LWESecretSampling<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let cbt_infos: CircuitBootstrappingKeyLayout = CircuitBootstrappingKeyLayout {
        brk_layout: BlindRotationKeyLayout {
            n_glwe: params.bin_fhe_params.n_glwe.into(),
            n_lwe: params.bin_fhe_params.n_lwe.into(),
            base2k: params.bin_fhe_params.base2k.into(),
            k_aux: params.bin_fhe_params.k_aux.into(),
            dnum: Dnum(params.brk_dnum),
            rank: params.bin_fhe_params.rank.into(),
        },
        atk_layout: GLWEAutomorphismKeyLayout {
            n: params.bin_fhe_params.n_glwe.into(),
            base2k: params.bin_fhe_params.base2k.into(),
            k_aux: params.bin_fhe_params.k_aux.into(),
            dnum: Dnum(params.atk_dnum),
            dsize: Dsize(params.atk_dsize),
            rank: params.bin_fhe_params.rank.into(),
        },
        tsk_layout: GGLWEToGGSWKeyLayout {
            n: params.bin_fhe_params.n_glwe.into(),
            base2k: params.bin_fhe_params.base2k.into(),
            k_aux: params.bin_fhe_params.k_aux.into(),
            dnum: Dnum(params.tsk_dnum),
            dsize: Dsize(params.tsk_dsize),
            rank: params.bin_fhe_params.rank.into(),
        },
    };
    let ggsw_infos: GGSWLayout = GGSWLayout {
        n: params.bin_fhe_params.n_glwe.into(),
        base2k: params.bin_fhe_params.base2k.into(),
        k_aux: params.bin_fhe_params.k_aux.into(),
        dnum: Dnum(params.ggsw_dnum),
        dsize: Dsize(params.ggsw_dsize),
        rank: params.bin_fhe_params.rank.into(),
    };
    let lwe_infos: LWELayout = LWELayout {
        n: params.bin_fhe_params.n_lwe.into(),
        k: params.bin_fhe_params.k_aux.into(),
        base2k: params.bin_fhe_params.base2k.into(),
    };

    let n_glwe = cbt_infos.brk_layout.n_glwe();
    let n_lwe = cbt_infos.brk_layout.n_lwe();
    let rank = cbt_infos.brk_layout.rank;

    let module: Module<BE> = Module::<BE>::new(n_glwe.as_u32() as u64);

    let mut source_xs: Source = Source::new([1u8; 32]);
    let mut source_xa: Source = Source::new([1u8; 32]);
    let mut source_xe: Source = Source::new([1u8; 32]);

    let mut sk_lwe: LWESecret<Vec<u8>, i64> = module.lwe_secret_alloc(n_lwe);
    module.lwe_secret_fill_binary_block(&mut sk_lwe, 7, &mut source_xs);

    let mut sk_glwe: GLWESecret<Vec<u8>, i64> = module.glwe_secret_alloc(rank);
    module.glwe_secret_fill_ternary_prob(&mut sk_glwe, 0.5, &mut source_xs);

    let ct_lwe: LWE<Vec<u8>, i64> = module.lwe_alloc_from_infos(&lwe_infos);
    let mut res: GGSW<Vec<u8>, i64> = module.ggsw_alloc_from_infos(&ggsw_infos);

    let cbt_enc_infos = CircuitBootstrappingEncryptionInfos::from_default_sigma(&cbt_infos).unwrap();

    let mut scratch: ScratchOwned<BE> =
        ScratchOwned::alloc(module.circuit_bootstrapping_execute_tmp_bytes(7, params.extension_factor, &res, &cbt_infos));
    let mut cbt_key: CircuitBootstrappingKey<BE::OwnedBuf, BRA, BE::ZnxWord> =
        CircuitBootstrappingKey::alloc_from_infos(&module, &cbt_infos);
    module.circuit_bootstrapping_key_encrypt_sk(
        &mut cbt_key,
        &sk_lwe,
        &sk_glwe,
        &cbt_enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    let mut cbt_prepared: CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE> =
        CircuitBootstrappingKeyPrepared::alloc_from_infos(&module, &cbt_infos);
    cbt_prepared.prepare(&module, &cbt_key, &mut scratch.borrow());

    bencher.iter(|| {
        cbt_prepared.execute_to_constant(
            &module,
            &mut res,
            &ct_lwe,
            params.log_domain,
            params.extension_factor,
            &mut scratch.borrow(),
        );
        black_box(());
    });
}
