use poulpy_hal::AlignedBuf;
use std::time::Instant;

use poulpy_hal::{
    api::{ModuleN, ScalarZnxAlloc, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxRotateAssign},
    layouts::{Backend, HostBackend, HostDataMut, HostDataRef, ScalarZnx, ScratchOwned, ZnxView, ZnxViewMut},
    source::Source,
};

use crate::{
    blind_rotation::{BlindRotationAlgo, BlindRotationKeyLayout, BlindRotationKeyPreparedFactory},
    circuit_bootstrapping::{
        CircuitBootstrappingExecute, CircuitBootstrappingKey, CircuitBootstrappingKeyEncryptSk, CircuitBootstrappingKeyLayout,
        CircuitBootstrappingKeyPrepared, CircuitBootstrappingKeyPreparedFactory,
    },
};

use crate::test_suite::parity::{assert_short_scratch_panics, snapshot_ggsw};
use poulpy_core::test_suite::noise::glwe_decrypt_checked;
use poulpy_core::test_suite::with_scratch;
use poulpy_core::{
    GGSWNoise, GLWEDecrypt, GLWEEncryptSk, GLWEExternalProduct, LWEEncryptSk,
    api::{GGLWEToGGSWKeyPreparedFactory, GGSWPreparedFactory, GLWEAutomorphismKeyPreparedFactory, GLWESecretPreparedFactory},
    layouts::{
        GGLWEToGGSWKeyLayout, GGSWAlloc, GGSWInfos, GGSWLayout, GLWEAlloc, GLWEAutomorphismKeyLayout, GLWEInfos,
        GLWEPlaintextAlloc, GLWESecretAlloc, LWEAlloc, LWELayout, LWEPlaintextAlloc, LWESecretAlloc, ModuleCoreAlloc,
    },
};

use poulpy_core::api::{GLWESecretSampling, LWESecretSampling};
use poulpy_core::layouts::prepared::GGSWPreparedToBackendRef;
use poulpy_core::layouts::{
    GGLWEToGGSWKeyAlloc, GGSW, GLWE, GLWEAutomorphismKeyAlloc, GLWEPlaintext, GLWESecret, LWE, LWEPlaintext, LWESecret,
    prepared::{GGSWPrepared, GLWESecretPrepared},
};

pub fn test_circuit_bootstrapping_to_exponent<
    BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64> + HostBackend,
    M,
    BRA: BlindRotationAlgo,
>(
    module: &M,
) where
    M: GLWEAutomorphismKeyAlloc
        + GGLWEToGGSWKeyAlloc
        + ModuleN
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + GGSWAlloc
        + GLWEAlloc
        + GLWEPlaintextAlloc
        + GLWESecretAlloc
        + LWEAlloc
        + LWEPlaintextAlloc
        + LWESecretAlloc
        + GLWESecretPreparedFactory<BE>
        + GLWEExternalProduct<BE>
        + GLWEDecrypt<BE>
        + LWEEncryptSk<BE>
        + CircuitBootstrappingKeyEncryptSk<BRA, BE>
        + CircuitBootstrappingKeyPreparedFactory<BRA, BE>
        + BlindRotationKeyPreparedFactory<BRA, BE>
        + CircuitBootstrappingExecute<BRA, BE>
        + GGSWPreparedFactory<BE>
        + GGLWEToGGSWKeyPreparedFactory<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GGSWNoise<BE>
        + GLWEEncryptSk<BE>
        + ScalarZnxAlloc<BE>
        + VecZnxRotateAssign<BE>
        + GLWESecretSampling<BE>
        + LWESecretSampling<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    BE::OwnedBuf: HostDataRef + HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut + AsMut<[u8]> + AsRef<[u8]> + Sync,
{
    let n_glwe: usize = module.n();
    let res_base2k: usize = 15;
    let base2k_lwe: usize = 14;
    let base2k_brk: usize = 13;
    let tsk_base2k: usize = 12;
    let a_base2ktk: usize = 11;
    let extension_factor: usize = 1;
    let rank: usize = 1;

    let n_lwe: usize = 77;
    let k_lwe_pt: usize = 4;
    let k_lwe_ct: usize = 22;
    let block_size: usize = 7;

    let rows_ggsw_res: usize = 3;

    let rows_brk: usize = 4;

    let rows_atk: usize = 4;

    // Covers the output GGSW, which the row expansion switches.
    let rows_tsk: usize = 6;

    let lwe_infos: LWELayout = LWELayout {
        n: n_lwe.into(),
        k: k_lwe_ct.into(),
        base2k: base2k_lwe.into(),
    };

    let cbt_infos: CircuitBootstrappingKeyLayout = CircuitBootstrappingKeyLayout {
        brk_layout: BlindRotationKeyLayout {
            n_glwe: n_glwe.into(),
            n_lwe: n_lwe.into(),
            base2k: base2k_brk.into(),
            dnum: rows_brk.into(),
            k_aux: (base2k_brk + n_glwe.ilog2() as usize).into(),
            rank: rank.into(),
        },
        atk_layout: GLWEAutomorphismKeyLayout {
            n: n_glwe.into(),
            base2k: a_base2ktk.into(),
            dnum: rows_atk.into(),
            k_aux: (a_base2ktk + n_glwe.ilog2() as usize).into(),
            rank: rank.into(),
            k_digit: a_base2ktk.into(),
        },
        tsk_layout: GGLWEToGGSWKeyLayout {
            n: n_glwe.into(),
            base2k: tsk_base2k.into(),
            dnum: rows_tsk.into(),
            k_aux: (tsk_base2k + n_glwe.ilog2() as usize).into(),
            k_digit: tsk_base2k.into(),
            rank: rank.into(),
        },
    };

    let ggsw_infos: GGSWLayout = GGSWLayout {
        n: n_glwe.into(),
        base2k: res_base2k.into(),
        k_aux: (res_base2k + n_glwe.ilog2() as usize).into(),
        dnum: rows_ggsw_res.into(),
        k_digit: res_base2k.into(),
        rank: rank.into(),
    };

    // Setup: key generation and preparation, encryptions, and the noise, product and decryption checks.
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .lwe_encrypt_sk_tmp_bytes(&lwe_infos)
            .max(module.circuit_bootstrapping_key_encrypt_sk_tmp_bytes(&cbt_infos))
            .max(module.circuit_bootstrapping_key_prepare_tmp_bytes(&cbt_infos))
            .max(module.ggsw_noise_tmp_bytes(&ggsw_infos))
            .max(module.glwe_encrypt_sk_tmp_bytes(&ggsw_infos))
            .max(module.ggsw_prepare_tmp_bytes(&ggsw_infos))
            .max(module.glwe_external_product_tmp_bytes(&ggsw_infos, &ggsw_infos, &ggsw_infos))
            .max(module.glwe_decrypt_tmp_bytes(&ggsw_infos)),
    );

    let mut source_xs: Source = Source::new([1u8; 32]);
    let mut source_xa: Source = Source::new([1u8; 32]);
    let mut source_xe: Source = Source::new([1u8; 32]);

    let mut sk_lwe: LWESecret<AlignedBuf, i64> = module.lwe_secret_alloc(n_lwe.into());
    module.lwe_secret_fill_binary_block(&mut sk_lwe, block_size, &mut source_xs);

    let mut sk_glwe: GLWESecret<AlignedBuf, i64> = module.glwe_secret_alloc(rank.into());
    module.glwe_secret_fill_ternary_prob(&mut sk_glwe, 0.5, &mut source_xs);

    let mut sk_glwe_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
    module.glwe_secret_prepare(&mut sk_glwe_prepared, &sk_glwe);

    let data: i64 = 1;

    let mut pt_lwe: LWEPlaintext<AlignedBuf, i64> = module.lwe_plaintext_alloc(base2k_lwe.into(), k_lwe_pt.into());
    pt_lwe.encode_i64(data, (k_lwe_pt + 1).into());

    println!("pt_lwe: {pt_lwe}");

    let mut ct_lwe: LWE<AlignedBuf, i64> = module.lwe_alloc_from_infos(&lwe_infos);
    module.lwe_encrypt_sk(
        &mut ct_lwe,
        &pt_lwe,
        &sk_lwe,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    let now: Instant = Instant::now();
    let mut cbt_key: CircuitBootstrappingKey<AlignedBuf, BRA, i64> =
        CircuitBootstrappingKey::alloc_from_infos(module, &cbt_infos);
    println!("CBT-ALLOC: {} ms", now.elapsed().as_millis());

    let now: Instant = Instant::now();
    module.circuit_bootstrapping_key_encrypt_sk(
        &mut cbt_key,
        &sk_lwe,
        &sk_glwe,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );
    println!("CBT-ENCRYPT: {} ms", now.elapsed().as_millis());

    let mut res: GGSW<AlignedBuf, i64> = module.ggsw_alloc_from_infos(&ggsw_infos);

    let log_gap_out = 1;

    let mut cbt_prepared: CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE> =
        CircuitBootstrappingKeyPrepared::alloc_from_infos(module, &cbt_infos);
    cbt_prepared.prepare(module, &cbt_key, &mut scratch.borrow());

    let now: Instant = Instant::now();
    let execute_bytes = module.circuit_bootstrapping_execute_to_exponent_tmp_bytes(
        log_gap_out,
        k_lwe_pt,
        block_size,
        extension_factor,
        &ggsw_infos,
        &cbt_infos,
    );
    let before = snapshot_ggsw::<BE, _>(&res);
    assert_short_scratch_panics::<BE, _>(execute_bytes, "insufficient circuit-bootstrapping scratch", |s| {
        cbt_prepared.execute_to_exponent(module, log_gap_out, k_lwe_pt, extension_factor, &mut res, &ct_lwe, s)
    });
    assert_eq!(before, snapshot_ggsw::<BE, _>(&res));
    with_scratch::<BE, _>(execute_bytes, |s| {
        cbt_prepared.execute_to_exponent(module, log_gap_out, k_lwe_pt, extension_factor, &mut res, &ct_lwe, s)
    });
    println!("CBT: {} ms", now.elapsed().as_millis());

    // X^{data * 2^log_gap_out}
    let mut pt_ggsw: ScalarZnx<AlignedBuf, i64> = module.scalar_znx_alloc(module.n(), 1);
    pt_ggsw.at_mut(0, 0)[data as usize * (1 << log_gap_out)] = 1;
    let pt_ggsw_ref = ScalarZnx::from_data(pt_ggsw.data.as_slice(), pt_ggsw.n(), pt_ggsw.cols());

    for row in 0..res.dnum().as_usize() {
        for col in 0..res.rank().as_usize() + 1 {
            println!(
                "row:{row} col:{col} -> {}",
                res.noise_stats(module, row, col, &pt_ggsw_ref, &sk_glwe_prepared, &mut scratch.borrow())
                    .std()
                    .log2()
            )
        }
    }

    let mut ct_glwe: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&ggsw_infos);
    let mut pt_glwe: GLWEPlaintext<AlignedBuf, i64> = module.glwe_plaintext_alloc_from_infos(&ggsw_infos);
    pt_glwe.data_mut().at_mut(0, 0)[0] = 1 << (res_base2k - 2);

    module.glwe_encrypt_sk(
        &mut ct_glwe,
        &pt_glwe,
        &sk_glwe_prepared,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    let mut res_prepared: GGSWPrepared<BE::OwnedBuf, BE> = module.ggsw_prepared_alloc_from_infos(&res);
    module.ggsw_prepare(&mut res_prepared, &res, &mut scratch.borrow());

    {
        module.glwe_external_product_assign(&mut ct_glwe, &res_prepared.to_backend_ref(), &mut scratch.borrow());
    }

    let mut pt_res: GLWEPlaintext<AlignedBuf, i64> = module.glwe_plaintext_alloc_from_infos(&ggsw_infos);
    glwe_decrypt_checked(module, &ct_glwe, &mut pt_res, &sk_glwe_prepared, &mut scratch.borrow());

    // Parameters are set such that the first limb should be noiseless.
    let mut pt_want: Vec<i64> = vec![0i64; module.n()];
    pt_want[data as usize * (1 << log_gap_out)] = pt_glwe.data().at(0, 0)[0];
    assert_eq!(pt_res.data().at(0, 0), pt_want);
}

pub fn test_circuit_bootstrapping_to_constant<
    BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64> + HostBackend,
    M,
    BRA: BlindRotationAlgo,
>(
    module: &M,
) where
    M: GLWEAutomorphismKeyAlloc
        + GGLWEToGGSWKeyAlloc
        + ModuleN
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + GGSWAlloc
        + GLWEAlloc
        + GLWEPlaintextAlloc
        + GLWESecretAlloc
        + LWEAlloc
        + LWEPlaintextAlloc
        + LWESecretAlloc
        + GLWESecretPreparedFactory<BE>
        + GLWEExternalProduct<BE>
        + GLWEDecrypt<BE>
        + LWEEncryptSk<BE>
        + CircuitBootstrappingKeyEncryptSk<BRA, BE>
        + CircuitBootstrappingKeyPreparedFactory<BRA, BE>
        + BlindRotationKeyPreparedFactory<BRA, BE>
        + CircuitBootstrappingExecute<BRA, BE>
        + GGSWPreparedFactory<BE>
        + GGLWEToGGSWKeyPreparedFactory<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GGSWNoise<BE>
        + GLWEEncryptSk<BE>
        + ScalarZnxAlloc<BE>
        + VecZnxRotateAssign<BE>
        + GLWESecretSampling<BE>
        + LWESecretSampling<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    BE::OwnedBuf: HostDataRef + HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut + AsMut<[u8]> + AsRef<[u8]> + Sync,
{
    let n_glwe: usize = module.n();
    let res_base2k: usize = 15;
    let base2k_lwe: usize = 14;
    let base2k_brk: usize = 13;
    let tsk_base2k: usize = 12;
    let a_base2ktk: usize = 11;
    let extension_factor: usize = 1;
    let rank: usize = 1;

    let n_lwe: usize = 77;
    let k_lwe_pt: usize = 1;
    let k_lwe_ct: usize = 13;
    let block_size: usize = 7;

    let rows_ggsw_res: usize = 3;

    let rows_brk: usize = 4;

    let rows_atk: usize = 4;

    // Covers the output GGSW, which the row expansion switches.
    let rows_tsk: usize = 6;

    let lwe_infos: LWELayout = LWELayout {
        n: n_lwe.into(),
        k: k_lwe_ct.into(),
        base2k: base2k_lwe.into(),
    };

    let cbt_infos: CircuitBootstrappingKeyLayout = CircuitBootstrappingKeyLayout {
        brk_layout: BlindRotationKeyLayout {
            n_glwe: n_glwe.into(),
            n_lwe: n_lwe.into(),
            base2k: base2k_brk.into(),
            dnum: rows_brk.into(),
            k_aux: (base2k_brk + n_glwe.ilog2() as usize).into(),
            rank: rank.into(),
        },
        atk_layout: GLWEAutomorphismKeyLayout {
            n: n_glwe.into(),
            base2k: a_base2ktk.into(),
            dnum: rows_atk.into(),
            k_aux: (a_base2ktk + n_glwe.ilog2() as usize).into(),
            rank: rank.into(),
            k_digit: a_base2ktk.into(),
        },
        tsk_layout: GGLWEToGGSWKeyLayout {
            n: n_glwe.into(),
            base2k: tsk_base2k.into(),
            dnum: rows_tsk.into(),
            k_aux: (tsk_base2k + n_glwe.ilog2() as usize).into(),
            k_digit: tsk_base2k.into(),
            rank: rank.into(),
        },
    };

    let ggsw_infos: GGSWLayout = GGSWLayout {
        n: n_glwe.into(),
        base2k: res_base2k.into(),
        k_aux: (res_base2k + n_glwe.ilog2() as usize).into(),
        dnum: rows_ggsw_res.into(),
        k_digit: res_base2k.into(),
        rank: rank.into(),
    };

    // Setup: key generation and preparation, encryptions, and the noise, product and decryption checks.
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .lwe_encrypt_sk_tmp_bytes(&lwe_infos)
            .max(module.circuit_bootstrapping_key_encrypt_sk_tmp_bytes(&cbt_infos))
            .max(module.circuit_bootstrapping_key_prepare_tmp_bytes(&cbt_infos))
            .max(module.ggsw_noise_tmp_bytes(&ggsw_infos))
            .max(module.glwe_encrypt_sk_tmp_bytes(&ggsw_infos))
            .max(module.ggsw_prepare_tmp_bytes(&ggsw_infos))
            .max(module.glwe_external_product_tmp_bytes(&ggsw_infos, &ggsw_infos, &ggsw_infos))
            .max(module.glwe_decrypt_tmp_bytes(&ggsw_infos)),
    );

    let mut source_xs: Source = Source::new([1u8; 32]);
    let mut source_xa: Source = Source::new([1u8; 32]);
    let mut source_xe: Source = Source::new([1u8; 32]);

    let mut sk_lwe: LWESecret<AlignedBuf, i64> = module.lwe_secret_alloc(n_lwe.into());
    module.lwe_secret_fill_binary_block(&mut sk_lwe, block_size, &mut source_xs);

    let mut sk_glwe: GLWESecret<AlignedBuf, i64> = module.glwe_secret_alloc(rank.into());
    module.glwe_secret_fill_ternary_prob(&mut sk_glwe, 0.5, &mut source_xs);

    let mut sk_glwe_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
    module.glwe_secret_prepare(&mut sk_glwe_prepared, &sk_glwe);

    let data: i64 = 1;

    let mut pt_lwe: LWEPlaintext<AlignedBuf, i64> = module.lwe_plaintext_alloc(base2k_lwe.into(), k_lwe_pt.into());
    pt_lwe.encode_i64(data, (k_lwe_pt + 1).into());

    println!("pt_lwe: {pt_lwe}");

    let mut ct_lwe: LWE<AlignedBuf, i64> = module.lwe_alloc_from_infos(&lwe_infos);
    module.lwe_encrypt_sk(
        &mut ct_lwe,
        &pt_lwe,
        &sk_lwe,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    let now: Instant = Instant::now();
    let mut cbt_key: CircuitBootstrappingKey<AlignedBuf, BRA, i64> =
        CircuitBootstrappingKey::alloc_from_infos(module, &cbt_infos);
    println!("CBT-ALLOC: {} ms", now.elapsed().as_millis());

    let now: Instant = Instant::now();
    module.circuit_bootstrapping_key_encrypt_sk(
        &mut cbt_key,
        &sk_lwe,
        &sk_glwe,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );
    println!("CBT-ENCRYPT: {} ms", now.elapsed().as_millis());

    let mut res: GGSW<AlignedBuf, i64> = module.ggsw_alloc_from_infos(&ggsw_infos);

    let mut cbt_prepared: CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE> =
        CircuitBootstrappingKeyPrepared::alloc_from_infos(module, &cbt_infos);
    cbt_prepared.prepare(module, &cbt_key, &mut scratch.borrow());

    let now: Instant = Instant::now();
    let execute_bytes =
        module.circuit_bootstrapping_execute_to_constant_tmp_bytes(block_size, extension_factor, &ggsw_infos, &cbt_infos);
    let before = snapshot_ggsw::<BE, _>(&res);
    assert_short_scratch_panics::<BE, _>(execute_bytes, "insufficient circuit-bootstrapping scratch", |s| {
        cbt_prepared.execute_to_constant(module, k_lwe_pt, extension_factor, &mut res, &ct_lwe, s)
    });
    assert_eq!(before, snapshot_ggsw::<BE, _>(&res));
    with_scratch::<BE, _>(execute_bytes, |s| {
        cbt_prepared.execute_to_constant(module, k_lwe_pt, extension_factor, &mut res, &ct_lwe, s)
    });
    println!("CBT: {} ms", now.elapsed().as_millis());

    // X^{data * 2^log_gap_out}
    let mut pt_ggsw: ScalarZnx<AlignedBuf, i64> = module.scalar_znx_alloc(module.n(), 1);
    pt_ggsw.at_mut(0, 0)[0] = data;
    let pt_ggsw_ref = ScalarZnx::from_data(pt_ggsw.data.as_slice(), pt_ggsw.n(), pt_ggsw.cols());

    for row in 0..res.dnum().as_usize() {
        for col in 0..res.rank().as_usize() + 1 {
            println!(
                "row:{row} col:{col} -> {}",
                res.noise_stats(module, row, col, &pt_ggsw_ref, &sk_glwe_prepared, &mut scratch.borrow())
                    .std()
                    .log2()
            )
        }
    }

    let mut ct_glwe: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&ggsw_infos);
    let mut pt_glwe: GLWEPlaintext<AlignedBuf, i64> = module.glwe_plaintext_alloc_from_infos(&ggsw_infos);
    pt_glwe.data_mut().at_mut(0, 0)[0] = 1 << (res_base2k - k_lwe_pt - 1);

    module.glwe_encrypt_sk(
        &mut ct_glwe,
        &pt_glwe,
        &sk_glwe_prepared,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    let mut res_prepared: GGSWPrepared<BE::OwnedBuf, BE> = module.ggsw_prepared_alloc_from_infos(&res);
    module.ggsw_prepare(&mut res_prepared, &res, &mut scratch.borrow());

    {
        module.glwe_external_product_assign(&mut ct_glwe, &res_prepared.to_backend_ref(), &mut scratch.borrow());
    }

    let mut pt_res: GLWEPlaintext<AlignedBuf, i64> = module.glwe_plaintext_alloc_from_infos(&ggsw_infos);
    glwe_decrypt_checked(module, &ct_glwe, &mut pt_res, &sk_glwe_prepared, &mut scratch.borrow());

    // Parameters are set such that the first limb should be noiseless.
    let mut pt_want: Vec<i64> = vec![0i64; module.n()];
    pt_want[0] = pt_glwe.data().at(0, 0)[0].wrapping_mul(data);
    println!("pt_res: {pt_res}");
    assert_eq!(pt_res.data().at(0, 0), pt_want);
}

/// The LWE secret of another dimension than the blind rotation key, then the GLWE secret of another degree than the
/// blind rotation, automorphism and tensor keys, each refused before any write or draw.
pub fn test_circuit_bootstrapping_key_encrypt_sk_rejects_secret_shape<
    BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64> + poulpy_hal::oep::HalModuleImpl,
    BRA: BlindRotationAlgo,
>(
    module: &poulpy_hal::layouts::Module<BE>,
) where
    poulpy_hal::layouts::Module<BE>: poulpy_hal::api::ModuleNew<BE>
        + CircuitBootstrappingKeyEncryptSk<BRA, BE>
        + LWESecretSampling<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>
        + GLWESecretAlloc
        + LWESecretAlloc,
{
    use poulpy_hal::layouts::WriterTo;

    let n = module.n();
    let half = poulpy_hal::layouts::Module::<BE>::new((n / 2) as u64);
    let layout = |n_brk: usize, n_atk: usize, n_tsk: usize| CircuitBootstrappingKeyLayout {
        brk_layout: BlindRotationKeyLayout {
            n_glwe: n_brk.into(),
            n_lwe: 8_usize.into(),
            base2k: 13_usize.into(),
            dnum: 2_usize.into(),
            k_aux: 20_usize.into(),
            rank: 1_usize.into(),
        },
        atk_layout: GLWEAutomorphismKeyLayout {
            n: n_atk.into(),
            base2k: 11_usize.into(),
            dnum: 2_usize.into(),
            k_aux: 20_usize.into(),
            rank: 1_usize.into(),
            k_digit: 11_usize.into(),
        },
        tsk_layout: GGLWEToGGSWKeyLayout {
            n: n_tsk.into(),
            base2k: 12_usize.into(),
            dnum: 2_usize.into(),
            k_aux: 20_usize.into(),
            k_digit: 12_usize.into(),
            rank: 1_usize.into(),
        },
    };
    let mut source_xs = Source::new([1u8; 32]);
    let mut sk_lwe = |n_lwe: usize| {
        let mut sk: LWESecret<AlignedBuf, i64> = module.lwe_secret_alloc(n_lwe.into());
        module.lwe_secret_fill_binary_block(&mut sk, 1, &mut source_xs);
        sk
    };
    let sk_full: GLWESecret<AlignedBuf, i64> = module.glwe_secret_alloc(1_usize.into());
    let sk_half: GLWESecret<AlignedBuf, i64> = half.glwe_secret_alloc(1_usize.into());
    let refusals = [
        "the LWE secret dimension differs from the blind rotation key's",
        "the GLWE secret degree differs from the blind rotation key's",
        "the GLWE secret degree differs from the automorphism key's",
        "the GLWE secret degree differs from the tensor key's",
    ];
    for (case, message) in refusals.into_iter().enumerate() {
        let infos = match case {
            2 => layout(n, n / 2, n),
            3 => layout(n, n, n / 2),
            _ => layout(n, n, n),
        };
        let sk_lwe = sk_lwe(if case == 0 { 9 } else { 8 });
        let sk_glwe = if case == 1 { &sk_half } else { &sk_full };
        let mut res: CircuitBootstrappingKey<AlignedBuf, BRA, i64> = CircuitBootstrappingKey::alloc_from_infos(module, &infos);
        let bytes = |key: &CircuitBootstrappingKey<AlignedBuf, BRA, i64>| {
            let mut out = Vec::new();
            key.write_to(&mut out).unwrap();
            out
        };
        let before = bytes(&res);
        let mut source_xe = Source::new([3u8; 32]);
        let mut source_xa = Source::new([2u8; 32]);
        crate::test_suite::parity::assert_panics_with(message, || {
            with_scratch::<BE, _>(0, |s| {
                module.circuit_bootstrapping_key_encrypt_sk(&mut res, &sk_lwe, sk_glwe, &mut source_xe, &mut source_xa, s)
            })
        });
        assert!(bytes(&res) == before, "a refused key encryption changed its destination");
        assert!(
            source_xe.new_seed() == Source::new([3u8; 32]).new_seed()
                && source_xa.new_seed() == Source::new([2u8; 32]).new_seed(),
            "a refused key encryption drew from its sources"
        );
    }
}
