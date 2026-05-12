use std::time::Instant;

use poulpy_hal::{
    api::{ModuleN, ScalarZnxAlloc, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxRotateAssignBackend},
    layouts::{Backend, HostBackend, HostDataMut, HostDataRef, ScalarZnx, ScratchOwned, ZnxView, ZnxViewMut},
    source::Source,
};

use crate::{
    blind_rotation::{BlindRotationAlgo, BlindRotationKeyLayout},
    circuit_bootstrapping::{
        CircuitBootstrappingEncryptionInfos, CircuitBootstrappingExecute, CircuitBootstrappingKey,
        CircuitBootstrappingKeyEncryptSk, CircuitBootstrappingKeyLayout, CircuitBootstrappingKeyPrepared,
        CircuitBootstrappingKeyPreparedFactory,
    },
};

use poulpy_core::{
    EncryptionLayout, GGSWNoise, GLWEDecrypt, GLWEEncryptSk, GLWEExternalProduct, LWEEncryptSk,
    layouts::{
        Dsize, GGLWEToGGSWKeyLayout, GGSWInfos, GGSWLayout, GGSWPreparedFactory, GLWEAutomorphismKeyLayout, GLWEInfos,
        GLWESecretPreparedFactory, LWEInfos, LWELayout, ModuleCoreAlloc,
    },
};

use poulpy_core::layouts::{
    GGSW, GLWE, GLWEPlaintext, GLWESecret, LWE, LWEPlaintext, LWESecret,
    prepared::{GGSWPrepared, GLWESecretPrepared},
};

pub fn test_circuit_bootstrapping_to_exponent<BE: Backend<OwnedBuf = Vec<u8>> + HostBackend, M, BRA: BlindRotationAlgo>(
    module: &M,
) where
    M: ModuleN
        + ModuleCoreAlloc
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
        + ScalarZnxAlloc<BE>
        + VecZnxRotateAssignBackend<BE>,
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

    let k_ggsw_res: usize = 4 * res_base2k;
    let rows_ggsw_res: usize = 3;

    let k_brk: usize = k_ggsw_res + base2k_brk;
    let rows_brk: usize = 4;

    let k_atk: usize = k_ggsw_res + tsk_base2k;
    let rows_atk: usize = 4;

    let k_tsk: usize = k_ggsw_res + a_base2ktk;
    let rows_tsk: usize = 4;

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
            k: k_brk.into(),
            dnum: rows_brk.into(),
            rank: rank.into(),
        },
        atk_layout: GLWEAutomorphismKeyLayout {
            n: n_glwe.into(),
            base2k: a_base2ktk.into(),
            k: k_atk.into(),
            dnum: rows_atk.into(),
            rank: rank.into(),
            dsize: Dsize(1),
        },
        tsk_layout: GGLWEToGGSWKeyLayout {
            n: n_glwe.into(),
            base2k: tsk_base2k.into(),
            k: k_tsk.into(),
            dnum: rows_tsk.into(),
            dsize: Dsize(1),
            rank: rank.into(),
        },
    };

    let ggsw_infos: GGSWLayout = GGSWLayout {
        n: n_glwe.into(),
        base2k: res_base2k.into(),
        k: k_ggsw_res.into(),
        dnum: rows_ggsw_res.into(),
        dsize: Dsize(1),
        rank: rank.into(),
    };

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 23);

    let mut source_xs: Source = Source::new([1u8; 32]);
    let mut source_xa: Source = Source::new([1u8; 32]);
    let mut source_xe: Source = Source::new([1u8; 32]);

    let mut sk_lwe: LWESecret<Vec<u8>> = module.lwe_secret_alloc(n_lwe.into());
    sk_lwe.fill_binary_block(block_size, &mut source_xs);

    let mut sk_glwe: GLWESecret<Vec<u8>> = module.glwe_secret_alloc(rank.into());
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_glwe_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
    module.glwe_secret_prepare(&mut sk_glwe_prepared, &sk_glwe);

    let data: i64 = 1;

    let mut pt_lwe: LWEPlaintext<Vec<u8>> = module.lwe_plaintext_alloc(base2k_lwe.into(), k_lwe_pt.into());
    pt_lwe.encode_i64(data, (k_lwe_pt + 1).into());

    println!("pt_lwe: {pt_lwe}");

    let lwe_enc_infos = EncryptionLayout::new_from_default_sigma(lwe_infos).unwrap();
    let mut ct_lwe: LWE<Vec<u8>> = module.lwe_alloc_from_infos(&lwe_infos);
    module.lwe_encrypt_sk(
        &mut ct_lwe,
        &pt_lwe,
        &sk_lwe,
        &lwe_enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    let now: Instant = Instant::now();
    let mut cbt_key: CircuitBootstrappingKey<Vec<u8>, BRA> = CircuitBootstrappingKey::alloc_from_infos(module, &cbt_infos);
    println!("CBT-ALLOC: {} ms", now.elapsed().as_millis());

    let cbt_enc_infos = CircuitBootstrappingEncryptionInfos::from_default_sigma(&cbt_infos).unwrap();
    let now: Instant = Instant::now();
    module.circuit_bootstrapping_key_encrypt_sk(
        &mut cbt_key,
        &sk_lwe,
        &sk_glwe,
        &cbt_enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );
    println!("CBT-ENCRYPT: {} ms", now.elapsed().as_millis());

    let mut res: GGSW<Vec<u8>> = module.ggsw_alloc_from_infos(&ggsw_infos);

    let log_gap_out = 1;

    let mut cbt_prepared: CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE> =
        CircuitBootstrappingKeyPrepared::alloc_from_infos(module, &cbt_infos);
    cbt_prepared.prepare(module, &cbt_key, &mut scratch.borrow());

    let now: Instant = Instant::now();
    cbt_prepared.execute_to_exponent(
        module,
        log_gap_out,
        &mut res,
        &ct_lwe,
        k_lwe_pt,
        extension_factor,
        &mut scratch.borrow(),
    );
    println!("CBT: {} ms", now.elapsed().as_millis());

    // X^{data * 2^log_gap_out}
    let mut pt_ggsw: ScalarZnx<Vec<u8>> = module.scalar_znx_alloc(1);
    pt_ggsw.at_mut(0, 0)[data as usize * (1 << log_gap_out)] = 1;
    let pt_ggsw_ref = ScalarZnx::from_data(pt_ggsw.data.as_slice(), pt_ggsw.n(), pt_ggsw.cols());

    for row in 0..res.dnum().as_usize() {
        for col in 0..res.rank().as_usize() + 1 {
            println!(
                "row:{row} col:{col} -> {}",
                res.noise(module, row, col, &pt_ggsw_ref, &sk_glwe_prepared, &mut scratch.borrow())
                    .std()
                    .log2()
            )
        }
    }
    let glwe_enc_infos = EncryptionLayout::new_from_default_sigma(ggsw_infos).unwrap();
    let mut ct_glwe: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(&ggsw_infos);
    let mut pt_glwe: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&ggsw_infos);
    pt_glwe.data.at_mut(0, 0)[0] = 1 << (res_base2k - 2);

    module.glwe_encrypt_sk(
        &mut ct_glwe,
        &pt_glwe,
        &sk_glwe_prepared,
        &glwe_enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    let mut res_prepared: GGSWPrepared<BE::OwnedBuf, BE> = module.ggsw_prepared_alloc_from_infos(&res);
    module.ggsw_prepare(&mut res_prepared, &res, &mut scratch.borrow());

    {
        module.glwe_external_product_assign(&mut ct_glwe, &res_prepared, res_prepared.size(), &mut scratch.borrow());
    }

    let mut pt_res: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&ggsw_infos);
    module.glwe_decrypt(&ct_glwe, &mut pt_res, &sk_glwe_prepared, &mut scratch.borrow());

    // Parameters are set such that the first limb should be noiseless.
    let mut pt_want: Vec<i64> = vec![0i64; module.n()];
    pt_want[data as usize * (1 << log_gap_out)] = pt_glwe.data.at(0, 0)[0];
    assert_eq!(pt_res.data.at(0, 0), pt_want);
}

pub fn test_circuit_bootstrapping_to_constant<BE: Backend<OwnedBuf = Vec<u8>> + HostBackend, M, BRA: BlindRotationAlgo>(
    module: &M,
) where
    M: ModuleN
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
        + ScalarZnxAlloc<BE>
        + VecZnxRotateAssignBackend<BE>,
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

    let k_ggsw_res: usize = 4 * res_base2k;
    let rows_ggsw_res: usize = 3;

    let k_brk: usize = k_ggsw_res + base2k_brk;
    let rows_brk: usize = 4;

    let k_atk: usize = k_ggsw_res + tsk_base2k;
    let rows_atk: usize = 4;

    let k_tsk: usize = k_ggsw_res + a_base2ktk;
    let rows_tsk: usize = 4;

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
            k: k_brk.into(),
            dnum: rows_brk.into(),
            rank: rank.into(),
        },
        atk_layout: GLWEAutomorphismKeyLayout {
            n: n_glwe.into(),
            base2k: a_base2ktk.into(),
            k: k_atk.into(),
            dnum: rows_atk.into(),
            rank: rank.into(),
            dsize: Dsize(1),
        },
        tsk_layout: GGLWEToGGSWKeyLayout {
            n: n_glwe.into(),
            base2k: tsk_base2k.into(),
            k: k_tsk.into(),
            dnum: rows_tsk.into(),
            dsize: Dsize(1),
            rank: rank.into(),
        },
    };

    let ggsw_infos: GGSWLayout = GGSWLayout {
        n: n_glwe.into(),
        base2k: res_base2k.into(),
        k: k_ggsw_res.into(),
        dnum: rows_ggsw_res.into(),
        dsize: Dsize(1),
        rank: rank.into(),
    };

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 23);

    let mut source_xs: Source = Source::new([1u8; 32]);
    let mut source_xa: Source = Source::new([1u8; 32]);
    let mut source_xe: Source = Source::new([1u8; 32]);

    let mut sk_lwe: LWESecret<Vec<u8>> = module.lwe_secret_alloc(n_lwe.into());
    sk_lwe.fill_binary_block(block_size, &mut source_xs);

    let mut sk_glwe: GLWESecret<Vec<u8>> = module.glwe_secret_alloc(rank.into());
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_glwe_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
    module.glwe_secret_prepare(&mut sk_glwe_prepared, &sk_glwe);

    let data: i64 = 1;

    let mut pt_lwe: LWEPlaintext<Vec<u8>> = module.lwe_plaintext_alloc(base2k_lwe.into(), k_lwe_pt.into());
    pt_lwe.encode_i64(data, (k_lwe_pt + 1).into());

    println!("pt_lwe: {pt_lwe}");

    let lwe_enc_infos = EncryptionLayout::new_from_default_sigma(lwe_infos).unwrap();
    let mut ct_lwe: LWE<Vec<u8>> = module.lwe_alloc_from_infos(&lwe_infos);
    module.lwe_encrypt_sk(
        &mut ct_lwe,
        &pt_lwe,
        &sk_lwe,
        &lwe_enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    let now: Instant = Instant::now();
    let mut cbt_key: CircuitBootstrappingKey<Vec<u8>, BRA> = CircuitBootstrappingKey::alloc_from_infos(module, &cbt_infos);
    println!("CBT-ALLOC: {} ms", now.elapsed().as_millis());

    let cbt_enc_infos = CircuitBootstrappingEncryptionInfos::from_default_sigma(&cbt_infos).unwrap();
    let now: Instant = Instant::now();
    module.circuit_bootstrapping_key_encrypt_sk(
        &mut cbt_key,
        &sk_lwe,
        &sk_glwe,
        &cbt_enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );
    println!("CBT-ENCRYPT: {} ms", now.elapsed().as_millis());

    let mut res: GGSW<Vec<u8>> = module.ggsw_alloc_from_infos(&ggsw_infos);

    let mut cbt_prepared: CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE> =
        CircuitBootstrappingKeyPrepared::alloc_from_infos(module, &cbt_infos);
    cbt_prepared.prepare(module, &cbt_key, &mut scratch.borrow());

    let now: Instant = Instant::now();
    cbt_prepared.execute_to_constant(module, &mut res, &ct_lwe, k_lwe_pt, extension_factor, &mut scratch.borrow());
    println!("CBT: {} ms", now.elapsed().as_millis());

    // X^{data * 2^log_gap_out}
    let mut pt_ggsw: ScalarZnx<Vec<u8>> = module.scalar_znx_alloc(1);
    pt_ggsw.at_mut(0, 0)[0] = data;
    let pt_ggsw_ref = ScalarZnx::from_data(pt_ggsw.data.as_slice(), pt_ggsw.n(), pt_ggsw.cols());

    for row in 0..res.dnum().as_usize() {
        for col in 0..res.rank().as_usize() + 1 {
            println!(
                "row:{row} col:{col} -> {}",
                res.noise(module, row, col, &pt_ggsw_ref, &sk_glwe_prepared, &mut scratch.borrow())
                    .std()
                    .log2()
            )
        }
    }

    let glwe_enc_infos = EncryptionLayout::new_from_default_sigma(ggsw_infos).unwrap();
    let mut ct_glwe: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(&ggsw_infos);
    let mut pt_glwe: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&ggsw_infos);
    pt_glwe.data.at_mut(0, 0)[0] = 1 << (res_base2k - k_lwe_pt - 1);

    module.glwe_encrypt_sk(
        &mut ct_glwe,
        &pt_glwe,
        &sk_glwe_prepared,
        &glwe_enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    let mut res_prepared: GGSWPrepared<BE::OwnedBuf, BE> = module.ggsw_prepared_alloc_from_infos(&res);
    module.ggsw_prepare(&mut res_prepared, &res, &mut scratch.borrow());

    {
        module.glwe_external_product_assign(&mut ct_glwe, &res_prepared, res_prepared.size(), &mut scratch.borrow());
    }

    let mut pt_res: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&ggsw_infos);
    module.glwe_decrypt(&ct_glwe, &mut pt_res, &sk_glwe_prepared, &mut scratch.borrow());

    // Parameters are set such that the first limb should be noiseless.
    let mut pt_want: Vec<i64> = vec![0i64; module.n()];
    pt_want[0] = pt_glwe.data.at(0, 0)[0] * data;
    println!("pt_res: {pt_res}");
    assert_eq!(pt_res.data.at(0, 0), pt_want);
}
