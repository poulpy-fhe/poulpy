use poulpy_core::{
    DEFAULT_BOUND_XE, DEFAULT_SIGMA_XE, GLWEDecrypt, GLWEEncryptSk, GLWEExternalProduct, LWEEncryptSk,
    layouts::{
        GGLWEToGGSWKeyLayout, GGSW, GGSWInfos, GGSWLayout, GLWE, GLWEAutomorphismKeyLayout, GLWEInfos, GLWELayout, GLWEPlaintext,
        GLWESecret, LWE, LWEInfos, LWELayout, LWEPlaintext, LWESecret, ModuleCoreAlloc,
        prepared::{GGSWPrepared, GGSWPreparedFactory, GLWESecretPrepared, GLWESecretPreparedFactory},
    },
};
use poulpy_hal::layouts::NoiseInfos;
use std::time::Instant;

#[cfg(all(
    feature = "enable-avx",
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
use poulpy_cpu_avx::FFT64Avx as BackendImpl;

#[cfg(not(all(
    feature = "enable-avx",
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
)))]
use poulpy_cpu_ref::FFT64Ref as BackendImpl;

use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalizeAssignBackend},
    layouts::{Backend, Module, ScalarZnx, ScratchOwned, VecZnxToBackendMut, ZnxView, ZnxViewMut},
    source::Source,
};

use poulpy_bin_fhe::{
    blind_rotation::{BlindRotationKeyLayout, CGGI},
    circuit_bootstrapping::{
        CircuitBootstrappingEncryptionInfos, CircuitBootstrappingKey, CircuitBootstrappingKeyEncryptSk,
        CircuitBootstrappingKeyLayout, CircuitBootstrappingKeyPrepared,
    },
};

fn main() {
    // GLWE ring degree
    let n_glwe: usize = 1024;

    // Module provides access to the backend arithmetic
    let module: Module<BackendImpl> = Module::<BackendImpl>::new(n_glwe as u64);

    // Base 2 loga
    let base2k: usize = 13;

    // Lookup table extension factor
    let extension_factor: usize = 1;

    // GLWE rank
    let rank: usize = 1;

    // LWE degree
    let n_lwe: usize = 574;

    // LWE plaintext modulus
    let k_lwe_pt: usize = 1;

    // LWE ciphertext modulus
    let k_lwe_ct: usize = 13;

    // LWE block binary key block size
    let block_size: usize = 7;

    // GGSW output number of dnum
    let rows_ggsw_res: usize = 2;

    // GGSW output modulus
    let k_ggsw_res: usize = (rows_ggsw_res + 1) * base2k;

    // Blind rotation key GGSW number of dnum
    let rows_brk: usize = rows_ggsw_res + 1;

    // Blind rotation key GGSW modulus
    let k_brk: usize = (rows_brk + 1) * base2k;

    // GGLWE automorphism keys number of dnum
    let rows_trace: usize = rows_ggsw_res + 1;

    // GGLWE automorphism keys modulus
    let k_trace: usize = (rows_trace + 1) * base2k;

    // GGLWE tensor key number of dnum
    let rows_tsk: usize = rows_ggsw_res + 1;

    // GGLWE tensor key modulus
    let k_tsk: usize = (rows_tsk + 1) * base2k;

    let cbt_layout: CircuitBootstrappingKeyLayout = CircuitBootstrappingKeyLayout {
        brk_layout: BlindRotationKeyLayout {
            n_glwe: n_glwe.into(),
            n_lwe: n_lwe.into(),
            base2k: base2k.into(),
            k: k_brk.into(),
            dnum: rows_brk.into(),
            rank: rank.into(),
        },
        atk_layout: GLWEAutomorphismKeyLayout {
            n: n_glwe.into(),
            base2k: base2k.into(),
            k: k_trace.into(),
            dnum: rows_trace.into(),
            dsize: 1_u32.into(),
            rank: rank.into(),
        },
        tsk_layout: GGLWEToGGSWKeyLayout {
            n: n_glwe.into(),
            base2k: base2k.into(),
            k: k_tsk.into(),
            dnum: rows_tsk.into(),
            dsize: 1_u32.into(),
            rank: rank.into(),
        },
    };

    let ggsw_infos: GGSWLayout = GGSWLayout {
        n: n_glwe.into(),
        base2k: base2k.into(),
        k: k_ggsw_res.into(),
        dnum: rows_ggsw_res.into(),
        dsize: 1_u32.into(),
        rank: rank.into(),
    };

    let lwe_infos = LWELayout {
        n: n_lwe.into(),
        k: k_lwe_ct.into(),
        base2k: base2k.into(),
    };

    // Scratch space (4MB)
    let mut scratch: ScratchOwned<BackendImpl> = ScratchOwned::alloc(1 << 22);

    // Secret key sampling source
    let mut source_xs: Source = Source::new([1u8; 32]);

    // Public randomness sampling source
    let mut source_xa: Source = Source::new([1u8; 32]);

    // Noise sampling source
    let mut source_xe: Source = Source::new([1u8; 32]);

    // LWE secret
    let mut sk_lwe: LWESecret<Vec<u8>> = module.lwe_secret_alloc(n_lwe.into());
    sk_lwe.fill_binary_block(block_size, &mut source_xs);
    // sk_lwe.fill_zero(); // for testing

    // GLWE secret
    let mut sk_glwe: GLWESecret<Vec<u8>> = module.glwe_secret_alloc(rank.into());
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);
    // sk_glwe.fill_zero(); // for testing

    // GLWE secret prepared (opaque backend dependant write only struct)
    let mut sk_glwe_prepared: GLWESecretPrepared<<BackendImpl as Backend>::OwnedBuf, BackendImpl> =
        module.glwe_secret_prepared_alloc(rank.into());
    module.glwe_secret_prepare(&mut sk_glwe_prepared, &sk_glwe);

    // Plaintext value to circuit bootstrap
    let data: i64 = 1 % (1 << k_lwe_pt);

    // LWE plaintext
    let mut pt_lwe: LWEPlaintext<Vec<u8>> = module.lwe_plaintext_alloc(base2k.into(), k_lwe_pt.into());

    // LWE plaintext(data * 2^{- (k_lwe_pt + 1)})
    pt_lwe.encode_i64(data, (k_lwe_pt + 1).into()); // +1 for padding bit

    // Normalize plaintext to nicely print coefficients
    module.vec_znx_normalize_assign_backend(
        base2k,
        &mut <poulpy_hal::layouts::VecZnx<Vec<u8>> as VecZnxToBackendMut<BackendImpl>>::to_backend_mut(pt_lwe.data_mut()),
        0,
        &mut scratch.borrow(),
    );
    println!("pt_lwe: {pt_lwe}");

    // LWE ciphertext
    let mut ct_lwe: LWE<Vec<u8>> = module.lwe_alloc_from_infos(&lwe_infos);

    let lwe_enc_infos = NoiseInfos::new(lwe_infos.max_k().as_usize(), DEFAULT_SIGMA_XE, DEFAULT_BOUND_XE).unwrap();
    let cbt_enc_infos = CircuitBootstrappingEncryptionInfos::from_default_sigma(&cbt_layout).unwrap();

    // Encrypt LWE Plaintext
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

    // Circuit bootstrapping evaluation key
    let mut cbt_key: CircuitBootstrappingKey<Vec<u8>, CGGI> = CircuitBootstrappingKey::alloc_from_infos(&module, &cbt_layout);

    module.circuit_bootstrapping_key_encrypt_sk(
        &mut cbt_key,
        &sk_lwe,
        &sk_glwe,
        &cbt_enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    println!("CBT-KGEN: {} ms", now.elapsed().as_millis());

    // Output GGSW
    let mut res: GGSW<Vec<u8>> = module.ggsw_alloc_from_infos(&ggsw_infos);

    // Circuit bootstrapping key prepared (opaque backend dependant write only struct)
    let mut cbt_prepared: CircuitBootstrappingKeyPrepared<<BackendImpl as Backend>::OwnedBuf, CGGI, BackendImpl> =
        CircuitBootstrappingKeyPrepared::alloc_from_infos(&module, &cbt_layout);
    cbt_prepared.prepare(&module, &cbt_key, &mut scratch.borrow());

    // Apply circuit bootstrapping: LWE(data * 2^{- (k_lwe_pt + 2)}) -> GGSW(data)
    let now: Instant = Instant::now();
    cbt_prepared.execute_to_constant(&module, &mut res, &ct_lwe, k_lwe_pt, extension_factor, &mut scratch.borrow());
    println!("CBT: {} ms", now.elapsed().as_millis());

    // Allocate "ideal" GGSW(data) plaintext
    let mut pt_ggsw: ScalarZnx<Vec<u8>> = module.scalar_znx_alloc(1);
    pt_ggsw.at_mut(0, 0)[0] = data;
    let pt_ggsw_ref = ScalarZnx::from_data(pt_ggsw.data.as_slice(), pt_ggsw.n(), pt_ggsw.cols());

    // Prints noise of GGSW(data)
    for row in 0..res.dnum().as_usize() {
        for col in 0..res.rank().as_usize() + 1 {
            println!(
                "row:{row} col:{col} -> {}",
                res.noise(&module, row, col, &pt_ggsw_ref, &sk_glwe_prepared, &mut scratch.borrow())
                    .std()
                    .log2()
            )
        }
    }

    // Tests GLWE(1) * GGSW(data)

    let glwe_infos: GLWELayout = GLWELayout {
        n: n_glwe.into(),
        base2k: base2k.into(),
        k: (k_ggsw_res - base2k).into(),
        rank: rank.into(),
    };

    // GLWE ciphertext modulus
    let mut ct_glwe: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(&glwe_infos);

    // Some GLWE plaintext with signed data
    let k_glwe_pt: usize = 3;
    let mut pt_glwe: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
    let mut data_vec: Vec<i64> = vec![0i64; n_glwe];
    data_vec
        .iter_mut()
        .enumerate()
        .for_each(|(x, y)| *y = (x % (1 << (k_glwe_pt - 1))) as i64 - (1 << (k_glwe_pt - 2)));

    pt_glwe.encode_vec_i64(&data_vec, (k_lwe_pt + 2).into());
    module.vec_znx_normalize_assign_backend(
        base2k,
        &mut <poulpy_hal::layouts::VecZnx<Vec<u8>> as VecZnxToBackendMut<BackendImpl>>::to_backend_mut(pt_glwe.data_mut()),
        0,
        &mut scratch.borrow(),
    );

    println!("{}", pt_glwe);

    // Encrypt
    let glwe_enc_infos = NoiseInfos::new(glwe_infos.max_k().as_usize(), DEFAULT_SIGMA_XE, DEFAULT_BOUND_XE).unwrap();
    module.glwe_encrypt_sk(
        &mut ct_glwe,
        &pt_glwe,
        &sk_glwe_prepared,
        &glwe_enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    // Prepare GGSW output of circuit bootstrapping (opaque backend dependant write only struct)
    let mut res_prepared: GGSWPrepared<<BackendImpl as Backend>::OwnedBuf, BackendImpl> =
        module.ggsw_prepared_alloc_from_infos(&res);
    module.ggsw_prepare(&mut res_prepared, &res, &mut scratch.borrow());

    // Apply GLWE x GGSW
    {
        let size = ct_glwe.size() + res_prepared.dsize().as_usize();
        module.glwe_external_product_assign(&mut ct_glwe, &res_prepared, size, &mut scratch.borrow());
    }

    // Decrypt
    let mut pt_res: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
    module.glwe_decrypt(&ct_glwe, &mut pt_res, &sk_glwe_prepared, &mut scratch.borrow());

    println!("pt_res: {:?}", &pt_res.data.at(0, 0)[..64]);
}
