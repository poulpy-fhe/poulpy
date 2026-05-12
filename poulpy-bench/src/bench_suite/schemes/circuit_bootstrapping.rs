use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};
use poulpy_core::{
    GGSWNoise, GLWEDecrypt, GLWEEncryptSk, GLWEExternalProduct, LWEEncryptSk,
    layouts::{
        Dsize, GGLWEToGGSWKeyLayout, GGSW, GGSWLayout, GGSWPreparedFactory, GLWEAutomorphismKeyLayout, GLWESecret,
        GLWESecretPreparedFactory, LWE, LWELayout, LWESecret, ModuleCoreAlloc,
    },
};
use poulpy_hal::{
    api::{ModuleN, ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxRotateAssignBackend},
    layouts::{Backend, HostBackend, Module, ScratchOwned},
    source::Source,
};

use poulpy_bin_fhe::{
    blind_rotation::{BlindRotationAlgo, BlindRotationKeyInfos, BlindRotationKeyLayout},
    circuit_bootstrapping::{
        CircuitBootstrappingEncryptionInfos, CircuitBootstrappingExecute, CircuitBootstrappingKey,
        CircuitBootstrappingKeyEncryptSk, CircuitBootstrappingKeyLayout, CircuitBootstrappingKeyPrepared,
        CircuitBootstrappingKeyPreparedFactory,
    },
};

pub fn bench_circuit_bootstrapping<BE: Backend<OwnedBuf = Vec<u8>> + HostBackend, BRA: BlindRotationAlgo>(
    c: &mut Criterion,
    label: &str,
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
        + VecZnxRotateAssignBackend<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let group_name: String = format!("circuit_bootstrapping::{label}");
    let mut group = c.benchmark_group(group_name);

    let cbt_infos: CircuitBootstrappingKeyLayout = CircuitBootstrappingKeyLayout {
        brk_layout: BlindRotationKeyLayout {
            n_glwe: 1024_u32.into(),
            n_lwe: 574_u32.into(),
            base2k: 13_u32.into(),
            k: 52_u32.into(),
            dnum: 3_u32.into(),
            rank: 2_u32.into(),
        },
        atk_layout: GLWEAutomorphismKeyLayout {
            n: 1024_u32.into(),
            base2k: 13_u32.into(),
            k: 52_u32.into(),
            dnum: 3_u32.into(),
            dsize: Dsize(1),
            rank: 2_u32.into(),
        },
        tsk_layout: GGLWEToGGSWKeyLayout {
            n: 1024_u32.into(),
            base2k: 13_u32.into(),
            k: 52_u32.into(),
            dnum: 3_u32.into(),
            dsize: Dsize(1),
            rank: 2_u32.into(),
        },
    };
    let ggsw_infos: GGSWLayout = GGSWLayout {
        n: 1024_u32.into(),
        base2k: 13_u32.into(),
        k: 26_u32.into(),
        dnum: 2_u32.into(),
        dsize: 1_u32.into(),
        rank: 2_u32.into(),
    };
    let lwe_infos: LWELayout = LWELayout {
        n: 574_u32.into(),
        k: 13_u32.into(),
        base2k: 13_u32.into(),
    };

    let n_glwe = cbt_infos.brk_layout.n_glwe();
    let n_lwe = cbt_infos.brk_layout.n_lwe();
    let rank = cbt_infos.brk_layout.rank;

    let module: Module<BE> = Module::<BE>::new(n_glwe.as_u32() as u64);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 22);

    let mut source_xs: Source = Source::new([1u8; 32]);
    let mut source_xa: Source = Source::new([1u8; 32]);
    let mut source_xe: Source = Source::new([1u8; 32]);

    let mut sk_lwe: LWESecret<Vec<u8>> = module.lwe_secret_alloc(n_lwe);
    sk_lwe.fill_binary_block(7, &mut source_xs);

    let mut sk_glwe: GLWESecret<Vec<u8>> = module.glwe_secret_alloc(rank);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);

    let ct_lwe: LWE<Vec<u8>> = module.lwe_alloc_from_infos(&lwe_infos);

    let cbt_enc_infos = CircuitBootstrappingEncryptionInfos::from_default_sigma(&cbt_infos).unwrap();

    let mut cbt_key: CircuitBootstrappingKey<Vec<u8>, BRA> = CircuitBootstrappingKey::alloc_from_infos(&module, &cbt_infos);
    module.circuit_bootstrapping_key_encrypt_sk(
        &mut cbt_key,
        &sk_lwe,
        &sk_glwe,
        &cbt_enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    let mut res: GGSW<Vec<u8>> = module.ggsw_alloc_from_infos(&ggsw_infos);
    let mut cbt_prepared: CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE> =
        CircuitBootstrappingKeyPrepared::alloc_from_infos(&module, &cbt_infos);
    cbt_prepared.prepare(&module, &cbt_key, &mut scratch.borrow());

    let id: BenchmarkId = BenchmarkId::from_parameter("1-bit");
    group.bench_with_input(id, &(), |b, _| {
        b.iter(|| {
            cbt_prepared.execute_to_constant(&module, &mut res, &ct_lwe, 1, 1, &mut scratch.borrow());
            black_box(());
        })
    });
    group.finish();
}
