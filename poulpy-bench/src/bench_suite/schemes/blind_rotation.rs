use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};
use poulpy_core::{
    EncryptionLayout, GLWEDecrypt, LWEEncryptSk,
    layouts::{
        Base2K, Dnum, GLWE, GLWELayout, GLWESecret, GLWESecretPrepared, GLWESecretPreparedFactory, LWE, LWEInfos, LWELayout,
        LWESecret, ModuleCoreAlloc, TorusPrecision,
    },
};
use poulpy_hal::{
    api::{ModuleN, ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, FillUniform, Module, ScratchOwned},
    source::Source,
};

use poulpy_bin_fhe::blind_rotation::{
    BlindRotationAlgo, BlindRotationExecute, BlindRotationKey, BlindRotationKeyEncryptSk, BlindRotationKeyLayout,
    BlindRotationKeyPrepared, BlindRotationKeyPreparedFactory, LookUpTableLayout, LookupTable, LookupTableFactory,
};

pub fn bench_blind_rotate<BE: Backend<OwnedBuf = Vec<u8>>, BRA: BlindRotationAlgo>(c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleN
        + ModuleNew<BE>
        + BlindRotationKeyEncryptSk<BRA, BE>
        + BlindRotationKeyPreparedFactory<BRA, BE>
        + BlindRotationExecute<BRA, BE>
        + LookupTableFactory
        + GLWESecretPreparedFactory<BE>
        + GLWEDecrypt<BE>
        + LWEEncryptSk<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let group_name: String = format!("blind_rotate::{label}");
    let mut group = c.benchmark_group(group_name);

    let n_glwe: usize = 2048;
    let n_lwe: usize = 1160;
    let rank: usize = 1;
    let block_size: usize = 8;
    let extension_factor: usize = 8;
    let log_message_modulus: usize = 8;
    let message_modulus: usize = 1 << log_message_modulus;

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(1 << 24);
    let module: Module<BE> = Module::<BE>::new(n_glwe as u64);

    let mut source_xs: Source = Source::new([2u8; 32]);
    let mut source_xe: Source = Source::new([2u8; 32]);
    let mut source_xa: Source = Source::new([1u8; 32]);

    let brk_infos: BlindRotationKeyLayout = BlindRotationKeyLayout {
        n_glwe: n_glwe.into(),
        n_lwe: n_lwe.into(),
        base2k: Base2K(18),
        k: TorusPrecision(36),
        dnum: Dnum(1),
        rank: rank.into(),
    };
    let glwe_infos: GLWELayout = GLWELayout {
        n: n_glwe.into(),
        base2k: Base2K(18),
        k: TorusPrecision(18),
        rank: rank.into(),
    };
    let lwe_infos: LWELayout = LWELayout {
        n: n_lwe.into(),
        k: TorusPrecision(18),
        base2k: Base2K(18),
    };

    let mut sk_glwe: GLWESecret<Vec<u8>> = module.glwe_secret_alloc_from_infos(&glwe_infos);
    sk_glwe.fill_ternary_prob(0.5, &mut source_xs);
    let mut sk_glwe_dft: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&glwe_infos);
    module.glwe_secret_prepare(&mut sk_glwe_dft, &sk_glwe);

    let mut sk_lwe: LWESecret<Vec<u8>> = module.lwe_secret_alloc(n_lwe.into());
    sk_lwe.fill_binary_block(block_size, &mut source_xs);

    let brk_enc_infos = EncryptionLayout::new_from_default_sigma(brk_infos).unwrap();

    let mut brk: BlindRotationKey<Vec<u8>, BRA> = BlindRotationKey::<Vec<u8>, BRA>::alloc(&module, &brk_infos);
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

    let mut res: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(&glwe_infos);
    res.data_mut().fill_uniform(glwe_infos.base2k().as_usize(), &mut source_xa);
    let mut lwe: LWE<Vec<u8>> = module.lwe_alloc_from_infos(&lwe_infos);
    lwe.data_mut().fill_uniform(lwe_infos.base2k().as_usize(), &mut source_xa);

    let mut f_vec: Vec<i64> = vec![0i64; message_modulus];
    f_vec.iter_mut().enumerate().for_each(|(i, x)| *x = 2 * i as i64 + 1);

    let lut_infos = LookUpTableLayout {
        n: module.n().into(),
        extension_factor,
        k: TorusPrecision(2),
        base2k: Base2K(17),
    };
    let mut lut: LookupTable = LookupTable::alloc(&module, &lut_infos);
    lut.set(&module, &f_vec, log_message_modulus + 1);

    let id: BenchmarkId = BenchmarkId::from_parameter(format!("{n_glwe} / {n_lwe}"));
    group.bench_with_input(id, &(), |b, _| {
        b.iter(|| {
            brk_prepared.execute(&module, &mut res, &lwe, &lut, &mut scratch.borrow());
            black_box(())
        })
    });
    group.finish();
}
