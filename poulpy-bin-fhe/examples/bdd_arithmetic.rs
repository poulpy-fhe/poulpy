use std::collections::HashMap;

use poulpy_bin_fhe::{
    bdd_arithmetic::{
        Add, BDDEncryptionInfos, BDDKey, BDDKeyEncryptSk, BDDKeyLayout, BDDKeyPrepared, BDDKeyPreparedFactory,
        ExecuteBDDCircuit2WTo1W, FheUint, FheUintPrepare, FheUintPrepared, GLWEBlindSelection, Xor,
    },
    blind_rotation::{BlindRotationAlgo, BlindRotationKeyLayout, CGGI},
    circuit_bootstrapping::CircuitBootstrappingKeyLayout,
};
use poulpy_core::{
    EncryptionLayout, GLWEDecrypt, GLWEEncryptSk, ScratchArenaTakeCore,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGLWEToGGSWKeyLayout, GGSWLayout, GGSWPreparedFactory, GLWEAutomorphismKeyLayout,
        GLWELayout, GLWESecretPreparedFactory, GLWESwitchingKeyLayout, GLWEToLWEKeyLayout, ModuleCoreAlloc, Rank, TorusPrecision,
    },
};
use poulpy_hal::{
    api::{ModuleN, ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostBackend, HostDataMut, HostDataRef, Module, ScratchArena, ScratchOwned},
    source::Source,
};
use rand::RngExt;

#[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
use poulpy_cpu_avx::FFT64Avx;
#[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
use poulpy_cpu_ref::FFT64Ref;

// This example demonstrates and end-to-end example usage of the BDD arithmetic API
// It includes all steps including:
//
// - Parameter Selection
// - Key Generation
// - Input Encryption
//
// - Key preparation
// - Input Preparation
// - Operation Execution
//
// - Result Decryption
//
// There also is an example use of the GLWE Blind Selection operation,
// which can choose between any number of encrypted fheuint inputs

fn example_bdd_arithmetic<BE, BRA: BlindRotationAlgo>()
where
    BE: Backend<OwnedBuf = Vec<u8>> + HostBackend + 'static,
    Module<BE>: ModuleNew<BE>
        + ModuleN
        + GLWESecretPreparedFactory<BE>
        + GLWEDecrypt<BE>
        + GGSWPreparedFactory<BE>
        + GLWEEncryptSk<BE>
        + BDDKeyEncryptSk<BRA, BE>
        + BDDKeyPreparedFactory<BRA, BE>
        + FheUintPrepare<BRA, BE>
        + ExecuteBDDCircuit2WTo1W<BE>
        + GLWEBlindSelection<u32, BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    BE::OwnedBuf: HostDataRef + HostDataMut,
    for<'a> BE: Backend<BufMut<'a> = &'a mut [u8], BufRef<'a> = &'a [u8]> + 'static,
    for<'a> BE::BufMut<'a>: AsMut<[u8]> + AsRef<[u8]> + Sync,
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
            rank_in: Rank(1),
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
    let mut sk_glwe = module.glwe_secret_alloc_from_infos(&glwe_layout);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_lwe = module.lwe_secret_alloc(bdd_layout.cbt_layout.brk_layout.n_lwe);
    sk_lwe.fill_binary_block(BINARY_BLOCK_SIZE as usize, &mut source_xs);

    // Preparing the private keys
    let mut sk_glwe_prepared = module.glwe_secret_prepared_alloc_from_infos(&glwe_layout);
    module.glwe_secret_prepare(&mut sk_glwe_prepared, &sk_glwe);

    // Creating the public BDD Key
    // This key is required to prepare all Fhe Integers for operations,
    // and for performing the operations themselves
    let bdd_enc_infos = BDDEncryptionInfos::from_default_sigma(&bdd_layout).unwrap();

    let mut bdd_key: BDDKey<Vec<u8>, BRA> = BDDKey::alloc_from_infos(&module, &bdd_layout);
    bdd_key.encrypt_sk(
        &module,
        &sk_lwe,
        &sk_glwe,
        &bdd_enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    ////////// Input Encryption
    // Encrypting the inputs
    let input_a = 255_u32;
    let input_b = 30_u32;

    let glwe_enc_infos = EncryptionLayout::new_from_default_sigma(glwe_layout).unwrap();

    let mut a_enc: FheUint<Vec<u8>, u32> = FheUint::alloc_from_infos(&module, &glwe_layout);
    a_enc.encrypt_sk(
        &module,
        input_a,
        &sk_glwe_prepared,
        &glwe_enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    let mut b_enc: FheUint<Vec<u8>, u32> = FheUint::alloc_from_infos(&module, &glwe_layout);
    b_enc.encrypt_sk(
        &module,
        input_b,
        &sk_glwe_prepared,
        &glwe_enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    //////// Homomorphic computation starts here ////////

    // Preparing the BDD Key
    // The BDD key must be prepared once before any operation is performed
    let mut bdd_key_prepared: BDDKeyPrepared<BE::OwnedBuf, BRA, BE> = BDDKeyPrepared::alloc_from_infos(&module, &bdd_layout);
    bdd_key_prepared.prepare(&module, &bdd_key, &mut scratch.borrow());

    // Input Preparation
    // Before each operation, the inputs to that operation must be prepared
    // Preparation extracts each bit of the integer into a seperate GLWE ciphertext and bootstraps it into a GGSW ciphertext
    let mut a_enc_prepared: FheUintPrepared<BE::OwnedBuf, u32, BE> = FheUintPrepared::alloc_from_infos(&module, &ggsw_layout);
    a_enc_prepared.prepare(&module, &a_enc, &bdd_key_prepared, &mut scratch.borrow());

    let mut b_enc_prepared: FheUintPrepared<BE::OwnedBuf, u32, BE> = FheUintPrepared::alloc_from_infos(&module, &ggsw_layout);
    b_enc_prepared.prepare(&module, &b_enc, &bdd_key_prepared, &mut scratch.borrow());

    // Allocating the intermediate ciphertext c_enc
    let mut c_enc: FheUint<Vec<u8>, u32> = FheUint::alloc_from_infos(&module, &glwe_layout);

    // Performing the operation
    c_enc.add(
        &module,
        &a_enc_prepared,
        &b_enc_prepared,
        &bdd_key_prepared,
        &mut scratch.borrow(),
    );

    // Preparing the intermediate result ciphertext, c_enc, for the next operation
    let mut c_enc_prepared: FheUintPrepared<BE::OwnedBuf, u32, BE> = FheUintPrepared::alloc_from_infos(&module, &ggsw_layout);
    c_enc_prepared.prepare(&module, &c_enc, &bdd_key_prepared, &mut scratch.borrow());

    // Creating the output ciphertext d_enc
    let mut selected_enc: FheUint<Vec<u8>, u32> = FheUint::alloc_from_infos(&module, &glwe_layout);
    selected_enc.xor(
        &module,
        &c_enc_prepared,
        &a_enc_prepared,
        &bdd_key_prepared,
        &mut scratch.borrow(),
    );

    //////// Homomorphic computation ends here ////////

    // Decrypting the result
    let d_dec = selected_enc.decrypt(&module, &sk_glwe_prepared, &mut scratch.borrow());

    // d = (a + b) ^ a
    let d_correct = (input_a.wrapping_add(input_b)) ^ input_a;
    println!("Result: {} == {}", d_dec, d_correct);

    // List of available operations are:
    // - add: addition
    // - sub: subtraction
    // - sll: left shift logical
    // - sra: right shift arithmetic
    // - srl: right shift logical
    // - slt: less than
    // - sltu: less than unsigned
    // - and: bitwise and
    // - or: bitwise or
    // - xor: bitwise xor

    ///////////////////////////// GLWE Blind Selection
    // This example demonstrates the use of the GLWE Blind Selection operation
    // It can choose between any number of encrypted fheuint inputs
    // using an encrypted fheuint selector

    let log_2_number_of_inputs: usize = 5;
    let number_of_inputs: usize = 1 << log_2_number_of_inputs;
    let inputs_a_vec: Vec<u32> = (0..number_of_inputs)
        .map(|_| rand::rng().random_range(0..u32::MAX - 1))
        .collect();
    let input_selector: u32 = rand::rng().random_range(0..number_of_inputs as u32);

    let mut inputs_a_enc_vec: Vec<FheUint<Vec<u8>, u32>> = Vec::new();
    for input in &inputs_a_vec {
        let mut next_input: FheUint<Vec<u8>, u32> = FheUint::alloc_from_infos(&module, &glwe_layout);
        next_input.encrypt_sk(
            &module,
            *input,
            &sk_glwe_prepared,
            &glwe_enc_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );
        inputs_a_enc_vec.push(next_input);
    }

    let mut inputs_a_enc_vec_map: HashMap<usize, &mut FheUint<Vec<u8>, u32>> = HashMap::new();
    for (i, input) in inputs_a_enc_vec.iter_mut().enumerate() {
        inputs_a_enc_vec_map.insert(i, input);
    }

    let mut input_selector_enc: FheUint<Vec<u8>, u32> = FheUint::alloc_from_infos(&module, &glwe_layout);
    input_selector_enc.encrypt_sk(
        &module,
        input_selector,
        &sk_glwe_prepared,
        &glwe_enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );
    let mut input_selector_enc_prepared: FheUintPrepared<BE::OwnedBuf, u32, BE> =
        FheUintPrepared::alloc_from_infos(&module, &ggsw_layout);
    input_selector_enc_prepared.prepare(&module, &input_selector_enc, &bdd_key_prepared, &mut scratch.borrow());

    module.glwe_blind_selection(
        &mut selected_enc,
        inputs_a_enc_vec_map,
        &input_selector_enc_prepared,
        0,
        log_2_number_of_inputs,
        &mut scratch.borrow(),
    );

    let selected_dec = selected_enc.decrypt(&module, &sk_glwe_prepared, &mut scratch.borrow());
    let selected_correct = inputs_a_vec[input_selector as usize];
    println!("Result: {} == {}", selected_dec, selected_correct);
}

fn main() {
    #[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
    example_bdd_arithmetic::<FFT64Avx, CGGI>();

    #[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
    example_bdd_arithmetic::<FFT64Ref, CGGI>();
}
