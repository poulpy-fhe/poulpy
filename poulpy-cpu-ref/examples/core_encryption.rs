use poulpy_core::{
    DEFAULT_SIGMA_XE, EncryptionLayout, GLWEDecrypt, GLWEEncryptSk, GLWESub,
    layouts::{
        Base2K, Degree, GLWE, GLWELayout, GLWEPlaintext, GLWEPlaintextLayout, GLWESecret, Rank, TorusPrecision,
        prepared::{GLWESecretPrepared, GLWESecretPreparedFactory},
    },
};
use poulpy_cpu_ref::FFT64Ref as BackendImpl;
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxFillUniform},
    layouts::{DeviceBuf, Module, ScratchOwned},
    source::Source,
};

fn main() {
    let log_n: usize = 10;
    let n: Degree = Degree(1 << log_n);
    let base2k: Base2K = Base2K(14);
    let k_xe: TorusPrecision = TorusPrecision(27);
    let k_pt: TorusPrecision = TorusPrecision(base2k.into());
    let rank: Rank = Rank(1);

    let module: Module<BackendImpl> = Module::<BackendImpl>::new(n.0 as u64);

    let glwe_ct_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
        n,
        base2k,
        k: k_xe,
        rank,
    })
    .unwrap();

    let glwe_pt_infos: GLWEPlaintextLayout = GLWEPlaintextLayout { n, base2k, k: k_pt };

    let mut ct: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_ct_infos);
    let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_pt_infos);
    let mut pt_have: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_pt_infos);

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([1u8; 32]);
    let mut source_xa: Source = Source::new([2u8; 32]);

    let mut scratch: ScratchOwned<BackendImpl> =
        ScratchOwned::alloc(module.glwe_encrypt_sk_tmp_bytes(&glwe_ct_infos) | module.glwe_decrypt_tmp_bytes(&glwe_ct_infos));

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(&glwe_ct_infos);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_prepared: GLWESecretPrepared<DeviceBuf<BackendImpl>, BackendImpl> = module.glwe_secret_prepared_alloc(rank);
    module.glwe_secret_prepare(&mut sk_prepared, &sk);

    module.vec_znx_fill_uniform(base2k.into(), &mut pt_want.data, 0, &mut source_xa);

    module.glwe_encrypt_sk(
        &mut ct,
        &pt_want,
        &sk_prepared,
        &glwe_ct_infos,
        &mut source_xe,
        &mut source_xa,
        scratch.borrow(),
    );

    module.glwe_decrypt(&ct, &mut pt_have, &sk_prepared, scratch.borrow());

    module.glwe_sub_assign(&mut pt_want, &pt_have);

    let noise_have: f64 = pt_want.data.stats(base2k.into(), 0).std() * (k_xe.as_u32() as f64).exp2();
    let noise_want: f64 = DEFAULT_SIGMA_XE;

    assert!(noise_have <= noise_want + 0.2);
}
