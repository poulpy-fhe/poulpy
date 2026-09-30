use poulpy_core::{
    Distribution, GetDistributionMut,
    layouts::{
        Base2K, Dnum, Dsize, GGLWELayout, GLWESecret, GLWESecretPrepared, GLWESecretPreparedFactory, GLWESecretSampling,
        ModuleCoreAlloc, Rank, TorusPrecision,
    },
};
use poulpy_hal::{
    AlignedBuf,
    api::VecZnxAddScalarAssign,
    layouts::{Backend, Module, ScalarZnxAsVecZnxBackendMut, ScalarZnxToBackendRef},
    source::Source,
};

pub(crate) const BASE2K: Base2K = Base2K(12);
pub(crate) const K: TorusPrecision = TorusPrecision(33);
pub(crate) const RANK: Rank = Rank(2);
pub(crate) const DNUM: Dnum = Dnum(3);
pub(crate) const DSIZE: Dsize = Dsize(1);
pub(crate) const SEEDS: [[u8; 32]; 2] = [[1u8; 32], [2u8; 32]];
pub(crate) const PARTIES: usize = 3;

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

/// The ideal secret: the coefficient-wise sum of the parties' secrets. It is
/// not ternary; the tag only has to be valid, since decryption ignores it.
pub(crate) fn ideal_secret<BE>(module: &Module<BE>, parties: &[Secret<BE>]) -> GLWESecretPrepared<AlignedBuf, BE>
where
    BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: GLWESecretPreparedFactory<BE> + VecZnxAddScalarAssign<BE>,
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
    *sum.dist_mut() = Distribution::TernaryProb(0.5);
    let mut sum_prepared: GLWESecretPrepared<AlignedBuf, BE> = module.glwe_secret_prepared_alloc(RANK);
    module.glwe_secret_prepare(&mut sum_prepared, &sum);
    sum_prepared
}

/// The common seed of public key entry `entry`, distinct per entry.
pub(crate) fn pk_entry_seed(entry: usize) -> [u8; 32] {
    let mut seed = SEEDS[0];
    seed[0] = 0x40 + entry as u8;
    seed
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
