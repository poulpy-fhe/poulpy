use poulpy_core::{
    Distribution, EncryptionLayout, GetDistributionMut,
    layouts::{
        Base2K, Dnum, Dsize, GGLWELayout, GLWELayout, GLWEPublicKey, GLWEPublicKeyPrepared, GLWEPublicKeyPreparedFactory,
        GLWESecret, GLWESecretPrepared, GLWESecretPreparedFactory, GLWESecretSampling, ModuleCoreAlloc, Rank, TorusPrecision,
    },
};
use poulpy_hal::{
    AlignedBuf,
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, ScratchOwned, WriterTo, ZnxView, ZnxViewMut},
    source::Source,
};

use crate::{
    api::{GLWEPublicKeyShare, PatAggregate},
    layouts::MHEModuleAlloc,
};

pub(crate) const BASE2K: Base2K = Base2K(12);
pub(crate) const K: TorusPrecision = TorusPrecision(33);
pub(crate) const RANK: Rank = Rank(2);
pub(crate) const DNUM: Dnum = Dnum(3);
pub(crate) const DSIZE: Dsize = Dsize(1);
pub(crate) const SEEDS: [[u8; 32]; 2] = [[1u8; 32], [2u8; 32]];
pub(crate) const SEED_XE: [u8; 32] = [3u8; 32];
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

pub(crate) fn secret<BE>(module: &Module<BE>) -> Secret<BE>
where
    BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: GLWESecretSampling<BE> + GLWESecretPreparedFactory<BE>,
{
    secret_from_seed(module, [0u8; 32])
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

/// The coefficient-wise sum of the parties' secrets. It is not ternary; the
/// tag only has to be valid, since decryption ignores it.
pub(crate) fn secret_sum<BE>(module: &Module<BE>, parties: &[Secret<BE>]) -> GLWESecret<AlignedBuf, i64>
where
    BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
{
    let mut sum: GLWESecret<AlignedBuf, i64> = module.glwe_secret_alloc(RANK);
    for (sk, _) in parties {
        for col in 0..RANK.as_usize() {
            for (s, x) in sum.data_mut().at_mut(col, 0).iter_mut().zip(sk.data().at(col, 0)) {
                *s += x;
            }
        }
    }
    *sum.dist_mut() = Distribution::TernaryProb(0.5);
    sum
}

/// The ideal secret: [`secret_sum`] prepared.
pub(crate) fn ideal_secret<BE>(module: &Module<BE>, parties: &[Secret<BE>]) -> GLWESecretPrepared<AlignedBuf, BE>
where
    BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: GLWESecretPreparedFactory<BE>,
{
    let mut sum_prepared: GLWESecretPrepared<AlignedBuf, BE> = module.glwe_secret_prepared_alloc(RANK);
    module.glwe_secret_prepare(&mut sum_prepared, &secret_sum(module, parties));
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
    Module<BE>: MHEModuleAlloc<BE> + GLWEPublicKeyShare<BE> + PatAggregate<BE> + GLWEPublicKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let enc_infos = EncryptionLayout::new_from_default_sigma(*layout).unwrap();
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .glwe_public_key_share_tmp_bytes(layout)
            .max(module.glwe_public_key_finalize_tmp_bytes())
            .max(module.glwe_public_key_prepare_tmp_bytes(layout)),
    );
    let mut acc = module.glwe_pat_compressed_alloc_from_infos(layout);
    let mut share = module.glwe_pat_compressed_alloc_from_infos(layout);
    for (i, (_, sk)) in parties.iter().enumerate() {
        let dst = if i == 0 { &mut acc } else { &mut share };
        let mut source_xe = Source::new([40 + i as u8; 32]);
        module.glwe_public_key_share(dst, sk, SEEDS[0], &enc_infos, &mut source_xe, &mut scratch.borrow());
        if i > 0 {
            module.glwe_pat_compressed_aggregate_assign(&mut acc, &share);
        }
    }
    let mut pk: GLWEPublicKey<AlignedBuf, i64> = module.glwe_public_key_alloc_from_infos(layout);
    module.glwe_public_key_finalize(&mut pk, &acc, Distribution::TernaryProb(0.5), &mut scratch.borrow());
    let mut pk_prepared: GLWEPublicKeyPrepared<AlignedBuf, BE> = module.glwe_public_key_prepared_alloc_from_infos(layout);
    module.glwe_public_key_prepare(&mut pk_prepared, &pk, &mut scratch.borrow());
    pk_prepared
}

pub(crate) fn assert_write_rejects<T: WriterTo>(value: &T) {
    let mut bytes: Vec<u8> = Vec::new();
    let err = value.write_to(&mut bytes).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    assert!(bytes.is_empty());
}
