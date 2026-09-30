use poulpy_core::{
    Distribution, EncryptionLayout, GetDistributionMut,
    layouts::{
        Base2K, Dnum, Dsize, GGLWELayout, GLWELayout, GLWEPublicKey, GLWEPublicKeyPrepared, GLWEPublicKeyPreparedFactory,
        GLWESecret, GLWESecretPrepared, GLWESecretPreparedFactory, GLWESecretSampling, ModuleCoreAlloc, Rank, TorusPrecision,
    },
};
use poulpy_hal::{
    AlignedBuf,
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddScalarAssign},
    layouts::{Backend, Module, ScalarZnxAsVecZnxBackendMut, ScalarZnxToBackendRef, ScratchOwned},
    source::Source,
};

use crate::{api::GLWEPublicKeyProtocol, layouts::MHEModuleAlloc};

pub(crate) const BASE2K: Base2K = Base2K(12);
pub(crate) const K: TorusPrecision = TorusPrecision(33);
pub(crate) const RANK: Rank = Rank(2);
pub(crate) const DNUM: Dnum = Dnum(3);
pub(crate) const DSIZE: Dsize = Dsize(1);
pub(crate) const SEEDS: [[u8; 32]; 2] = [[1u8; 32], [2u8; 32]];
pub(crate) const PARTIES: usize = 3;

/// Galois element of the automorphism key tests.
pub(crate) const P: i64 = -5;

pub(crate) type Secret<BE> = (GLWESecret<AlignedBuf, i64>, GLWESecretPrepared<AlignedBuf, BE>);

pub(crate) fn gglwe_layout<BE: Backend>(module: &Module<BE>) -> GGLWELayout {
    GGLWELayout {
        n: module.n().into(),
        base2k: BASE2K,
        dnum: DNUM,
        k_aux: TorusPrecision(DSIZE.0 * BASE2K.0 + module.log_n() as u32),
        rank_in: RANK,
        rank_out: RANK,
        dsize: DSIZE,
        stride: 1,
    }
}

pub(crate) fn secret_from_seed<BE>(module: &Module<BE>, seed: [u8; 32]) -> Secret<BE>
where
    BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: GLWESecretSampling<BE> + GLWESecretPreparedFactory<BE>,
{
    let mut sk: GLWESecret<AlignedBuf, i64> = module.glwe_secret_alloc(RANK);
    module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut Source::new(seed));
    let mut sk_prepared: GLWESecretPrepared<AlignedBuf, BE> = module.glwe_secret_prepared_alloc(RANK);
    module.glwe_secret_prepare(&mut sk_prepared, &sk);
    (sk, sk_prepared)
}

/// One secret per party, each from its own seed.
pub(crate) fn party_secrets<BE>(module: &Module<BE>) -> Vec<Secret<BE>>
where
    BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: GLWESecretSampling<BE> + GLWESecretPreparedFactory<BE>,
{
    (0..PARTIES).map(|i| secret_from_seed(module, [100 + i as u8; 32])).collect()
}

/// One secret-shaped GGLWE message per party, distinct from every party secret.
pub(crate) fn party_messages<BE>(module: &Module<BE>) -> Vec<Secret<BE>>
where
    BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: GLWESecretSampling<BE> + GLWESecretPreparedFactory<BE>,
{
    (0..PARTIES).map(|i| secret_from_seed(module, [70 + i as u8; 32])).collect()
}

/// The coefficient-wise sum of the secrets of `parties`.
pub(crate) fn secret_sum<BE>(module: &Module<BE>, parties: &[Secret<BE>]) -> GLWESecret<AlignedBuf, i64>
where
    BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: VecZnxAddScalarAssign<BE>,
{
    let mut sum: GLWESecret<AlignedBuf, i64> = module.glwe_secret_alloc(RANK);
    for (sk, _) in parties {
        for col in 0..RANK.as_usize() {
            module.vec_znx_add_scalar_assign(
                &mut ScalarZnxAsVecZnxBackendMut::<BE>::as_vec_znx_backend_mut(sum.data_mut()),
                col,
                0,
                &ScalarZnxToBackendRef::<BE>::to_backend_ref(sk.data()),
                col,
            );
        }
    }
    sum
}

/// The ideal secret: the sum of the parties' secrets. It is not ternary; the
/// tag only has to be valid, since decryption ignores it.
pub(crate) fn ideal_secret<BE>(module: &Module<BE>, parties: &[Secret<BE>]) -> GLWESecretPrepared<AlignedBuf, BE>
where
    BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: GLWESecretPreparedFactory<BE> + VecZnxAddScalarAssign<BE>,
{
    let mut sum: GLWESecret<AlignedBuf, i64> = secret_sum(module, parties);
    *sum.dist_mut() = Distribution::TernaryProb(0.5);
    let mut sum_prepared: GLWESecretPrepared<AlignedBuf, BE> = module.glwe_secret_prepared_alloc(RANK);
    module.glwe_secret_prepare(&mut sum_prepared, &sum);
    sum_prepared
}

/// The collective public key of `parties` at `layout`: the public key protocol
/// under `SEEDS[0]`, finalized and prepared.
pub(crate) fn collective_public_key<BE>(
    module: &Module<BE>,
    parties: &[Secret<BE>],
    layout: &GLWELayout,
) -> GLWEPublicKeyPrepared<AlignedBuf, BE>
where
    BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + GLWEPublicKeyProtocol<BE> + GLWEPublicKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let enc_infos = EncryptionLayout::new_from_default_sigma(*layout).unwrap();
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .glwe_public_key_gen_tmp_bytes(layout)
            .max(module.glwe_public_key_finalize_tmp_bytes())
            .max(module.glwe_public_key_prepare_tmp_bytes(layout)),
    );
    let mut acc = module.glwe_public_key_share_alloc_from_infos(layout);
    let mut share = module.glwe_public_key_share_alloc_from_infos(layout);
    for (i, (_, sk)) in parties.iter().enumerate() {
        let dst = if i == 0 { &mut acc } else { &mut share };
        let mut source_xe = Source::new([40 + i as u8; 32]);
        module.glwe_public_key_gen(dst, sk, SEEDS[0], &enc_infos, &mut source_xe, &mut scratch.borrow());
        if i > 0 {
            module.glwe_public_key_aggregate(&mut acc, &share);
        }
    }
    let mut pk: GLWEPublicKey<AlignedBuf, i64> = module.glwe_public_key_alloc_from_infos(layout);
    module.glwe_public_key_finalize(&mut pk, &acc, Distribution::TernaryProb(0.5), &mut scratch.borrow());
    let mut pk_prepared: GLWEPublicKeyPrepared<AlignedBuf, BE> = module.glwe_public_key_prepared_alloc_from_infos(layout);
    module.glwe_public_key_prepare(&mut pk_prepared, &pk, &mut scratch.borrow());
    pk_prepared
}

/// Assert a protocol boundary rejects a layout with the exact static message.
pub(crate) fn assert_panics_with(expected: &str, f: impl FnOnce()) {
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).expect_err("expected a layout rejection");
    let message = panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str));
    assert_eq!(message, Some(expected));
}
