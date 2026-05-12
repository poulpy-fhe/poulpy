use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use poulpy_core::{
    EncryptionLayout, GLWEDecrypt, GLWEEncryptSk, LWEEncryptSk,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGLWEToGGSWKeyLayout, GGSWLayout, GGSWPreparedFactory, GLWEAutomorphismKeyLayout,
        GLWELayout, GLWESecret, GLWESecretPreparedFactory, GLWESwitchingKeyLayout, GLWEToLWEKeyLayout, LWESecret,
        ModuleCoreAlloc, Rank, TorusPrecision,
    },
};

use poulpy_bin_fhe::{
    bdd_arithmetic::{
        Add, And, BDDEncryptionInfos, BDDKey, BDDKeyEncryptSk, BDDKeyLayout, BDDKeyPrepared, BDDKeyPreparedFactory,
        ExecuteBDDCircuit2WTo1W, FheUint, FheUintPrepare, FheUintPrepared, Or, Sll, Slt, Sltu, Sra, Srl, Sub, Xor,
    },
    blind_rotation::{BlindRotationAlgo, BlindRotationKeyInfos, BlindRotationKeyLayout, CGGI},
    circuit_bootstrapping::{
        CircuitBootstrappingEncryptionInfos, CircuitBootstrappingKey, CircuitBootstrappingKeyEncryptSk,
        CircuitBootstrappingKeyLayout, CircuitBootstrappingKeyPrepared,
    },
};
use poulpy_hal::{
    api::{ModuleN, ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostBackend, HostDataMut, HostDataRef, Module, ScratchArena, ScratchOwned},
    source::Source,
};

// Common setup data structure
struct BenchmarkSetup<BE: Backend<OwnedBuf = Vec<u8>> + HostBackend, BRA: BlindRotationAlgo> {
    module: Module<BE>,
    scratch: ScratchOwned<BE>,
    a_enc_prepared: FheUintPrepared<BE::OwnedBuf, u32, BE>,
    b_enc_prepared: FheUintPrepared<BE::OwnedBuf, u32, BE>,
    bdd_key_prepared: BDDKeyPrepared<BE::OwnedBuf, BRA, BE>,
    glwe_layout: GLWELayout,
}

struct Params {
    name: String,
    block_size: usize,
    glwe_layout: GLWELayout,
    ggsw_layout: GGSWLayout,
    bdd_layout: BDDKeyLayout,
}

fn setup_benchmark<BE: Backend<OwnedBuf = Vec<u8>> + HostBackend, BRA: BlindRotationAlgo>(
    params: &Params,
) -> BenchmarkSetup<BE, BRA>
where
    Module<BE>: ModuleNew<BE>
        + ModuleN
        + GLWESecretPreparedFactory<BE>
        + GLWEDecrypt<BE>
        + CircuitBootstrappingKeyEncryptSk<BRA, BE>
        + GGSWPreparedFactory<BE>
        + GLWEEncryptSk<BE>
        + BDDKeyEncryptSk<BRA, BE>
        + BDDKeyPreparedFactory<BRA, BE>
        + FheUintPrepare<BRA, BE>
        + ExecuteBDDCircuit2WTo1W<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    BE::OwnedBuf: HostDataRef + HostDataMut,
    for<'a> BE::BufMut<'a>: AsRef<[u8]> + AsMut<[u8]> + Sync,
    for<'a> BE::BufRef<'a>: AsRef<[u8]> + Send,
{
    // Scratch space (16MB)
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 24);

    let n_glwe: poulpy_core::layouts::Degree = params.bdd_layout.cbt_layout.brk_layout.n_glwe();
    let n_lwe: poulpy_core::layouts::Degree = params.bdd_layout.cbt_layout.brk_layout.n_lwe();
    let rank: poulpy_core::layouts::Rank = params.bdd_layout.cbt_layout.brk_layout.rank;

    let module: Module<BE> = Module::<BE>::new(n_glwe.as_u32() as u64);

    let mut source_xs: Source = Source::new([1u8; 32]);
    let mut source_xa: Source = Source::new([1u8; 32]);
    let mut source_xe: Source = Source::new([1u8; 32]);

    let mut sk_lwe: LWESecret<Vec<u8>> = module.lwe_secret_alloc(n_lwe);
    sk_lwe.fill_binary_block(params.block_size, &mut source_xs);

    let mut sk_glwe: GLWESecret<Vec<u8>> = module.glwe_secret_alloc(rank);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);

    // Circuit bootstrapping evaluation key
    let cbt_enc_infos = CircuitBootstrappingEncryptionInfos::from_default_sigma(&params.bdd_layout.cbt_layout).unwrap();
    let mut cbt_key: CircuitBootstrappingKey<Vec<u8>, BRA> =
        CircuitBootstrappingKey::alloc_from_infos(&module, &params.bdd_layout.cbt_layout);
    cbt_key.encrypt_sk(
        &module,
        &sk_lwe,
        &sk_glwe,
        &cbt_enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    let mut cbt_key_prepared: CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE> =
        CircuitBootstrappingKeyPrepared::alloc_from_infos(&module, &params.bdd_layout.cbt_layout);
    cbt_key_prepared.prepare(&module, &cbt_key, &mut scratch.borrow());

    let mut sk_glwe_prepared = module.glwe_secret_prepared_alloc_from_infos(&params.glwe_layout);
    module.glwe_secret_prepare(&mut sk_glwe_prepared, &sk_glwe);

    let bdd_enc_infos = BDDEncryptionInfos::from_default_sigma(&params.bdd_layout).unwrap();
    let glwe_enc_infos = EncryptionLayout::new_from_default_sigma(params.glwe_layout).unwrap();
    let mut bdd_key: BDDKey<Vec<u8>, BRA> = BDDKey::alloc_from_infos(&module, &params.bdd_layout);
    bdd_key.encrypt_sk(
        &module,
        &sk_lwe,
        &sk_glwe,
        &bdd_enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    let input_a = 255_u32;
    let input_b = 30_u32;

    let mut a_enc: FheUint<Vec<u8>, u32> = FheUint::alloc_from_infos(&module, &params.glwe_layout);
    a_enc.encrypt_sk(
        &module,
        input_a,
        &sk_glwe_prepared,
        &glwe_enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    let mut b_enc: FheUint<Vec<u8>, u32> = FheUint::alloc_from_infos(&module, &params.glwe_layout);
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
    let mut bdd_key_prepared: BDDKeyPrepared<BE::OwnedBuf, BRA, BE> =
        BDDKeyPrepared::alloc_from_infos(&module, &params.bdd_layout);
    bdd_key_prepared.prepare(&module, &bdd_key, &mut scratch.borrow());

    // Input Preparation
    // Before each operation, the inputs to that operation must be prepared
    // Preparation extracts each bit of the integer into a seperate GLWE ciphertext and bootstraps it into a GGSW ciphertext
    let mut a_enc_prepared: FheUintPrepared<BE::OwnedBuf, u32, BE> =
        FheUintPrepared::alloc_from_infos(&module, &params.ggsw_layout);
    a_enc_prepared.prepare(&module, &a_enc, &bdd_key_prepared, &mut scratch.borrow());

    let mut b_enc_prepared: FheUintPrepared<BE::OwnedBuf, u32, BE> =
        FheUintPrepared::alloc_from_infos(&module, &params.ggsw_layout);
    b_enc_prepared.prepare(&module, &b_enc, &bdd_key_prepared, &mut scratch.borrow());

    BenchmarkSetup {
        module,
        scratch,
        a_enc_prepared,
        b_enc_prepared,
        bdd_key_prepared,
        glwe_layout: params.glwe_layout,
    }
}

fn create_runner<BE: Backend<OwnedBuf = Vec<u8>> + HostBackend, BRA: BlindRotationAlgo, F>(
    setup: BenchmarkSetup<BE, BRA>,
    operation: F,
) -> impl FnMut()
where
    Module<BE>: ExecuteBDDCircuit2WTo1W<BE>,
    ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
    F: Fn(
        &mut FheUint<Vec<u8>, u32>,
        &Module<BE>,
        &FheUintPrepared<BE::OwnedBuf, u32, BE>,
        &FheUintPrepared<BE::OwnedBuf, u32, BE>,
        &BDDKeyPrepared<BE::OwnedBuf, BRA, BE>,
        &mut ScratchArena<'_, BE>,
    ),
{
    let BenchmarkSetup {
        module,
        mut scratch,
        a_enc_prepared,
        b_enc_prepared,
        bdd_key_prepared,
        glwe_layout,
    } = setup;

    let mut c_enc: FheUint<Vec<u8>, u32> = FheUint::alloc_from_infos(&module, &glwe_layout);

    move || {
        operation(
            &mut c_enc,
            &module,
            &a_enc_prepared,
            &b_enc_prepared,
            &bdd_key_prepared,
            &mut scratch.borrow(),
        );
        black_box(());
    }
}

fn bench_operation<BE: Backend<OwnedBuf = Vec<u8>> + HostBackend, BRA: BlindRotationAlgo, F>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    params: &Params,
    operation_name: &str,
    operation: F,
) where
    Module<BE>: ModuleNew<BE>
        + ModuleN
        + GLWESecretPreparedFactory<BE>
        + GLWEDecrypt<BE>
        + LWEEncryptSk<BE>
        + CircuitBootstrappingKeyEncryptSk<BRA, BE>
        + GGSWPreparedFactory<BE>
        + GLWEEncryptSk<BE>
        + BDDKeyEncryptSk<BRA, BE>
        + BDDKeyPreparedFactory<BRA, BE>
        + FheUintPrepare<BRA, BE>
        + ExecuteBDDCircuit2WTo1W<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    BE::OwnedBuf: HostDataRef + HostDataMut,
    for<'a> BE::BufMut<'a>: AsRef<[u8]> + AsMut<[u8]> + Sync,
    for<'a> BE::BufRef<'a>: AsRef<[u8]> + Send,
    F: Fn(
            &mut FheUint<Vec<u8>, u32>,
            &Module<BE>,
            &FheUintPrepared<BE::OwnedBuf, u32, BE>,
            &FheUintPrepared<BE::OwnedBuf, u32, BE>,
            &BDDKeyPrepared<BE::OwnedBuf, BRA, BE>,
            &mut ScratchArena<'_, BE>,
        ) + 'static,
{
    let setup = setup_benchmark::<BE, BRA>(params);
    let mut runner = create_runner(setup, operation);
    let id = BenchmarkId::from_parameter(format!("{}_{}", params.name, operation_name));
    group.bench_with_input(id, &(), |b, _| b.iter(&mut runner));
}

pub fn benc_bdd_arithmetic<BE: Backend<OwnedBuf = Vec<u8>> + HostBackend, BRA: BlindRotationAlgo>(c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE>
        + ModuleN
        + GLWESecretPreparedFactory<BE>
        + GLWEDecrypt<BE>
        + LWEEncryptSk<BE>
        + CircuitBootstrappingKeyEncryptSk<BRA, BE>
        + GGSWPreparedFactory<BE>
        + GLWEEncryptSk<BE>
        + BDDKeyEncryptSk<BRA, BE>
        + BDDKeyPreparedFactory<BRA, BE>
        + FheUintPrepare<BRA, BE>
        + ExecuteBDDCircuit2WTo1W<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    BE::OwnedBuf: HostDataRef + HostDataMut,
    for<'a> BE::BufMut<'a>: AsRef<[u8]> + AsMut<[u8]> + Sync,
    for<'a> BE::BufRef<'a>: AsRef<[u8]> + Send,
{
    let group_name: String = format!("bdd_arithmetic::{label}");

    let mut group = c.benchmark_group(group_name);

    const N_GLWE: u32 = 1024;
    const N_LWE: u32 = 679;
    const BINARY_BLOCK_SIZE: u32 = 7;
    const BASE2K: u32 = 15;
    const RANK: u32 = 2;

    let params: Params = Params {
        name: format!("n_glwe={N_GLWE}"),
        block_size: BINARY_BLOCK_SIZE as usize,
        glwe_layout: GLWELayout {
            n: Degree(N_GLWE),
            base2k: Base2K(BASE2K),
            k: TorusPrecision(2 * BASE2K),
            rank: Rank(RANK),
        },
        ggsw_layout: GGSWLayout {
            n: Degree(N_GLWE),
            base2k: Base2K(BASE2K),
            k: TorusPrecision(3 * BASE2K),
            rank: Rank(RANK),
            dnum: Dnum(3),
            dsize: Dsize(1),
        },
        bdd_layout: BDDKeyLayout {
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
        },
    };

    // Benchmark each operation
    bench_operation::<BE, BRA, _>(&mut group, &params, "add", |c_enc, module, a, b, key, scratch| {
        c_enc.add(module, a, b, key, scratch);
    });

    bench_operation::<BE, BRA, _>(&mut group, &params, "sub", |c_enc, module, a, b, key, scratch| {
        c_enc.sub(module, a, b, key, scratch);
    });

    bench_operation::<BE, BRA, _>(&mut group, &params, "sll", |c_enc, module, a, b, key, scratch| {
        c_enc.sll(module, a, b, key, scratch);
    });

    bench_operation::<BE, BRA, _>(&mut group, &params, "sra", |c_enc, module, a, b, key, scratch| {
        c_enc.sra(module, a, b, key, scratch);
    });

    bench_operation::<BE, BRA, _>(&mut group, &params, "srl", |c_enc, module, a, b, key, scratch| {
        c_enc.srl(module, a, b, key, scratch);
    });

    bench_operation::<BE, BRA, _>(&mut group, &params, "slt", |c_enc, module, a, b, key, scratch| {
        c_enc.slt(module, a, b, key, scratch);
    });

    bench_operation::<BE, BRA, _>(&mut group, &params, "sltu", |c_enc, module, a, b, key, scratch| {
        c_enc.sltu(module, a, b, key, scratch);
    });

    bench_operation::<BE, BRA, _>(&mut group, &params, "or", |c_enc, module, a, b, key, scratch| {
        c_enc.or(module, a, b, key, scratch);
    });

    bench_operation::<BE, BRA, _>(&mut group, &params, "and", |c_enc, module, a, b, key, scratch| {
        c_enc.and(module, a, b, key, scratch);
    });

    bench_operation::<BE, BRA, _>(&mut group, &params, "xor", |c_enc, module, a, b, key, scratch| {
        c_enc.xor(module, a, b, key, scratch);
    });

    group.finish();
}

fn bench_bdd_arithmetic(c: &mut Criterion) {
    benc_bdd_arithmetic::<poulpy_cpu_ref::FFT64Ref, CGGI>(c, "fft64-ref");
    #[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
    benc_bdd_arithmetic::<poulpy_cpu_avx::FFT64Avx, CGGI>(c, "fft64-avx");
    #[cfg(all(feature = "enable-avx512f", target_arch = "x86_64"))]
    benc_bdd_arithmetic::<poulpy_cpu_avx512::FFT64Avx512, CGGI>(c, "fft64-avx512");
}

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_bdd_arithmetic
}
criterion_main!(benches);
