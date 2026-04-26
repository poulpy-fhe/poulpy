use std::collections::HashMap;

use poulpy_core::{
    EncryptionLayout, GLWECopy, GLWEDecrypt, GLWEEncryptSk, GLWEExternalProduct, LWEEncryptSk, ScratchTakeCore,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGLWEToGGSWKeyLayout, GGSWLayout, GGSWPreparedFactory, GLWEAutomorphismKeyLayout,
        GLWELayout, GLWESecret, GLWESecretPreparedFactory, GLWESwitchingKeyLayout, GLWEToLWEKeyLayout, GLWEToMut, GLWEToRef,
        LWESecret, Rank, TorusPrecision,
    },
};
use poulpy_hal::{
    api::{ModuleN, ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxRotateAssign},
    layouts::{Backend, DeviceBuf, Module, Scratch, ScratchOwned},
    source::Source,
};
use poulpy_schemes::bin_fhe::{
    bdd_arithmetic::{
        BDDEncryptionInfos, BDDKey, BDDKeyEncryptSk, BDDKeyLayout, BDDKeyPrepared, BDDKeyPreparedFactory,
        ExecuteBDDCircuit2WTo1W, FheUint, FheUintPrepare, FheUintPrepared, GLWEBlindSelection, Sltu,
    },
    blind_rotation::{BlindRotationAlgo, BlindRotationKeyLayout, CGGI},
    circuit_bootstrapping::CircuitBootstrappingKeyLayout,
};
use rand::RngExt;

#[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
use poulpy_cpu_avx::FFT64Avx;
#[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
use poulpy_cpu_ref::FFT64Ref;

// This example demonstrates and end-to-end example usage of the BDD arithmetic API
// to compute the maximum of an array of integers.

fn example_max_array<BE: Backend, BRA: BlindRotationAlgo>()
where
    Module<BE>: ModuleNew<BE>
        + ModuleN
        + GLWESecretPreparedFactory<BE>
        + GLWEExternalProduct<BE>
        + GLWEDecrypt<BE>
        + LWEEncryptSk<BE>
        + GGSWPreparedFactory<BE>
        + GLWEEncryptSk<BE>
        + VecZnxRotateAssign<BE>
        + BDDKeyEncryptSk<BRA, BE>
        + BDDKeyPreparedFactory<BRA, BE>
        + FheUintPrepare<BRA, BE>
        + ExecuteBDDCircuit2WTo1W<BE>
        + GLWEBlindSelection<u32, BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    ////////// Parameter Selection
    const N_GLWE: u32 = 1024;
    const N_LWE: u32 = 567;
    const BINARY_BLOCK_SIZE: u32 = 7;
    const BASE2K: u32 = 17;
    const RANK: u32 = 1;

    // GLWE layout, used to generate GLWE Ciphertexts, keys, switching keys, etc
    let glwe_layout = GLWELayout {
        n: Degree(N_GLWE),
        base2k: Base2K(BASE2K),
        k: TorusPrecision(2 * BASE2K),
        rank: Rank(RANK),
    };

    // Used to generate GGSW Ciphertexts
    let ggsw_layout = GGSWLayout {
        n: Degree(N_GLWE),
        base2k: Base2K(BASE2K),
        k: TorusPrecision(3 * BASE2K),
        rank: Rank(RANK),
        dnum: Dnum(3),
        dsize: Dsize(1),
    };

    // Used to generate BDD Keys, for the arithmetic operations
    let bdd_layout = BDDKeyLayout {
        cbt_layout: CircuitBootstrappingKeyLayout {
            brk_layout: BlindRotationKeyLayout {
                n_glwe: Degree(N_GLWE),
                n_lwe: Degree(N_LWE),
                base2k: Base2K(BASE2K),
                k: TorusPrecision(4 * BASE2K),
                dnum: Dnum(4),
                rank: Rank(RANK),
            },
            atk_layout: GLWEAutomorphismKeyLayout {
                n: Degree(N_GLWE),
                base2k: Base2K(BASE2K),
                k: TorusPrecision(4 * BASE2K),
                dnum: Dnum(4),
                dsize: Dsize(1),
                rank: Rank(RANK),
            },
            tsk_layout: GGLWEToGGSWKeyLayout {
                n: Degree(N_GLWE),
                base2k: Base2K(BASE2K),
                k: TorusPrecision(4 * BASE2K),
                dnum: Dnum(4),
                dsize: Dsize(1),
                rank: Rank(RANK),
            },
        },
        ks_glwe_layout: Some(GLWESwitchingKeyLayout {
            n: Degree(N_GLWE),
            base2k: Base2K(BASE2K),
            k: TorusPrecision(4 * BASE2K),
            dnum: Dnum(4),
            dsize: Dsize(1),
            rank_in: Rank(RANK),
            rank_out: Rank(1),
        }),
        ks_lwe_layout: GLWEToLWEKeyLayout {
            n: Degree(N_GLWE),
            base2k: Base2K(BASE2K),
            k: TorusPrecision(4 * BASE2K),
            rank_in: Rank(RANK),
            dnum: Dnum(4),
        },
    };

    let module = Module::<BE>::new(N_GLWE as u64);

    // Secret key sampling source
    let mut source_xs: Source = Source::new([1u8; 32]);

    // Public randomness sampling source
    let mut source_xa: Source = Source::new([1u8; 32]);

    // Noise sampling source
    let mut source_xe: Source = Source::new([1u8; 32]);

    // Scratch space (4MB)
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 22);

    ////////// Key Generation and Preparation
    // Generating the GLWE and LWE key
    let mut sk_glwe = GLWESecret::alloc_from_infos(&glwe_layout);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_lwe = LWESecret::alloc(Degree(N_LWE));
    sk_lwe.fill_binary_block(BINARY_BLOCK_SIZE as usize, &mut source_xs);

    // Preparing the private keys
    let mut sk_glwe_prepared = module.glwe_secret_prepared_alloc_from_infos(&glwe_layout);
    module.glwe_secret_prepare(&mut sk_glwe_prepared, &sk_glwe);

    // Creating the public BDD Key
    // This key is required to prepare all Fhe Integers for operations,
    // and for performing the operations themselves
    let bdd_enc_infos = BDDEncryptionInfos::from_default_sigma(&bdd_layout).unwrap();
    let glwe_enc_infos = EncryptionLayout::new_from_default_sigma(glwe_layout).unwrap();

    let mut bdd_key: BDDKey<Vec<u8>, BRA> = BDDKey::alloc_from_infos(&bdd_layout);
    bdd_key.encrypt_sk(
        &module,
        &sk_lwe,
        &sk_glwe,
        &bdd_enc_infos,
        &mut source_xe,
        &mut source_xa,
        scratch.borrow(),
    );

    ////////// Input Encryption
    // Encrypting the inputs
    let mut rng = rand::rng();
    let inputs: Vec<u32> = (0..3).map(|_| rng.random_range(0..u32::MAX - 1)).collect();

    let mut inputs_enc: Vec<FheUint<Vec<u8>, u32>> = Vec::new();
    for input in &inputs {
        let mut next_input = FheUint::alloc_from_infos(&glwe_layout);
        next_input.encrypt_sk(
            &module,
            *input,
            &sk_glwe_prepared,
            &glwe_enc_infos,
            &mut source_xe,
            &mut source_xa,
            scratch.borrow(),
        );
        inputs_enc.push(next_input);
    }

    //////// Homomorphic computation starts here ////////

    // Preparing the BDD Key
    // The BDD key must be prepared once before any operation is performed
    let mut bdd_key_prepared: BDDKeyPrepared<DeviceBuf<BE>, BRA, BE> = BDDKeyPrepared::alloc_from_infos(&module, &bdd_layout);
    bdd_key_prepared.prepare(&module, &bdd_key, scratch.borrow());

    let mut max_enc: FheUint<Vec<u8>, u32> = FheUint::alloc_from_infos(&glwe_layout);
    max_enc.encrypt_sk(
        &module,
        0,
        &sk_glwe_prepared,
        &glwe_enc_infos,
        &mut source_xe,
        &mut source_xa,
        scratch.borrow(),
    );
    // Copy of max_enc for the HashMap
    let mut max_enc_copy: FheUint<Vec<u8>, u32> = FheUint::alloc_from_infos(&glwe_layout);

    // Allocating the intermediate ciphertext c_enc
    let mut compare_enc: FheUint<Vec<u8>, u32> = FheUint::alloc_from_infos(&glwe_layout);
    let mut compare_enc_prepared: FheUintPrepared<DeviceBuf<BE>, u32, BE> =
        FheUintPrepared::alloc_from_infos(&module, &ggsw_layout);

    for input_i in inputs_enc.iter_mut() {
        let mut max_enc_prepared: FheUintPrepared<DeviceBuf<BE>, u32, BE> =
            FheUintPrepared::alloc_from_infos(&module, &ggsw_layout);
        max_enc_prepared.prepare(&module, &max_enc, &bdd_key_prepared, scratch.borrow());

        let mut input_i_enc_prepared: FheUintPrepared<DeviceBuf<BE>, u32, BE> =
            FheUintPrepared::alloc_from_infos(&module, &ggsw_layout);
        input_i_enc_prepared.prepare(&module, input_i, &bdd_key_prepared, scratch.borrow());

        // b = (input_i < max)
        compare_enc.sltu(
            &module,
            &input_i_enc_prepared,
            &max_enc_prepared,
            &bdd_key_prepared,
            scratch.borrow(),
        );

        compare_enc_prepared.prepare(&module, &compare_enc, &bdd_key_prepared, scratch.borrow());

        module.glwe_copy(&mut max_enc_copy.to_mut(), &max_enc.to_ref());

        let cts = HashMap::from([(0, input_i), (1, &mut max_enc_copy)]);

        <Module<BE> as GLWEBlindSelection<u32, BE>>::glwe_blind_selection(
            &module,
            &mut max_enc,
            cts,
            &compare_enc_prepared,
            0,
            1,
            scratch.borrow(),
        );
    }

    //////// Homomorphic computation ends here ////////

    // Decrypting the result
    let result_dec = max_enc.decrypt(&module, &sk_glwe_prepared, scratch.borrow());

    // result = max of inputs
    let result_correct = inputs.iter().max().unwrap();
    println!("Result: {} == {}", result_dec, result_correct);
}

fn main() {
    #[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
    example_max_array::<FFT64Avx, CGGI>();

    #[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
    example_max_array::<FFT64Ref, CGGI>();
}
