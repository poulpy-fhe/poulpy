//! Shamir thresholdization: every party shares its secret, every party
//! aggregates the shares it receives, and any active set of at least the
//! threshold combines its shares into additive shares of the secrets' sum.

use poulpy_core::{
    GLWEAdd, GLWENormalize,
    layouts::{Base2K, GLWESecretPreparedFactory, GLWESecretSampling, TorusPrecision},
};
use poulpy_hal::{
    AlignedBuf,
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{HostBackend, HostDataMut, HostDataRef, Module, ScratchOwned, ZnxView},
    source::Source,
};

use super::fixtures::{RANK, Secret, secret_from_seed};
use crate::{
    api::GLWEShamirMHEProtocol,
    layouts::{GLWEShamirLayout, GLWEShamirShareOwned, GLWEWideSecret, MHEModuleAlloc},
};

/// A precision that is not a multiple of the base: the last limb is partial.
const K_SHARE: TorusPrecision = TorusPrecision(73);
const B_SHARE: Base2K = Base2K(13);

pub fn test_glwe_threshold<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>
        + GLWEShamirMHEProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEAdd<BE>
        + GLWENormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    threshold_combine(module, 5, 3, &[&[1, 2, 3], &[2, 4, 5], &[1, 3, 4, 5]]);
}

pub fn test_glwe_threshold_all_active<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>
        + GLWEShamirMHEProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEAdd<BE>
        + GLWENormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    threshold_combine(module, 3, 3, &[&[1, 2, 3]]);
}

/// Combining with fewer active parties than the threshold panics.
pub fn test_glwe_threshold_too_few_actives<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + GLWEShamirMHEProtocol<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = shamir_layout(module, 3, 3);
    let share = module.glwe_shamir_share_alloc(&layout);
    let mut res = module.glwe_wide_secret_alloc(&layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.mhe_glwe_shamir_share_finalize_tmp_bytes());
    module.mhe_glwe_shamir_share_finalize(&mut res, &share, 1, &[1, 2], &mut scratch.borrow());
}

/// Combining for a party outside the active set panics.
pub fn test_glwe_threshold_own_not_active<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + GLWEShamirMHEProtocol<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = shamir_layout(module, 3, 3);
    let share = module.glwe_shamir_share_alloc(&layout);
    let mut res = module.glwe_wide_secret_alloc(&layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.mhe_glwe_shamir_share_finalize_tmp_bytes());
    module.mhe_glwe_shamir_share_finalize(&mut res, &share, 1, &[2, 3, 4], &mut scratch.borrow());
}

/// Aggregating shares of different thresholds panics.
pub fn test_glwe_threshold_aggregate_mismatch<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + GLWEShamirMHEProtocol<BE>,
{
    let mut res = module.glwe_shamir_share_alloc(&shamir_layout(module, 3, 3));
    let share = module.glwe_shamir_share_alloc(&shamir_layout(module, 2, 3));
    module.mhe_glwe_shamir_share_aggregate(&mut res, &share);
}

fn shamir_layout<BE: poulpy_hal::layouts::Backend>(module: &Module<BE>, threshold: usize, gr_degree: usize) -> GLWEShamirLayout {
    GLWEShamirLayout {
        n: module.n().into(),
        base2k: B_SHARE,
        k: K_SHARE,
        rank: RANK,
        gr_degree,
        threshold,
    }
}

/// Thresholdizes one secret per party, then checks, for every active set,
/// that the combined additive shares sum to the secrets' sum.
fn threshold_combine<BE>(module: &Module<BE>, parties: usize, threshold: usize, active_sets: &[&[u32]])
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>
        + GLWEShamirMHEProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEAdd<BE>
        + GLWENormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let (secrets, layout, shares) = thresholdize(module, parties, threshold);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .glwe_normalize_tmp_bytes()
            .max(module.mhe_glwe_shamir_share_finalize_tmp_bytes()),
    );
    let n = module.n();
    for &actives in active_sets {
        let mut sum = module.glwe_wide_secret_alloc(&layout);
        let mut part = module.glwe_wide_secret_alloc(&layout);
        for &own in actives {
            module.mhe_glwe_shamir_share_finalize(&mut part, &shares[own as usize - 1], own, actives, &mut scratch.borrow());
            assert_canonical(&part);
            module.glwe_add_assign(&mut sum.inner, &part.inner);
        }
        module.glwe_normalize_assign(&mut sum.inner, &mut scratch.borrow());
        for col in 0..RANK.as_usize() {
            let mut got = vec![0i128; n];
            sum.data()
                .decode_vec_i128(B_SHARE.as_usize(), col, K_SHARE.as_usize(), &mut got);
            for (i, &g) in got.iter().enumerate() {
                let want: i64 = secrets.iter().map(|(sk, _)| sk.data().at(col, 0)[i]).sum();
                assert_eq!(g, want as i128, "combined shares do not sum to the secret");
            }
        }
    }
}

/// One secret per party, thresholdized: every party's aggregated share of
/// the secrets' sum, recipient `i` at index `i - 1`.
fn thresholdize<BE>(
    module: &Module<BE>,
    parties: usize,
    threshold: usize,
) -> (Vec<Secret<BE>>, GLWEShamirLayout, Vec<GLWEShamirShareOwned<BE>>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>
        + GLWEShamirMHEProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEAdd<BE>
        + GLWENormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    // The smallest Galois ring degree with a point per party.
    let gr_degree = (usize::BITS - parties.leading_zeros()) as usize;
    let layout = shamir_layout(module, threshold, gr_degree);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .mhe_glwe_shamir_polynomial_gen_tmp_bytes(&layout)
            .max(module.mhe_glwe_shamir_share_gen_tmp_bytes(&layout))
            .max(module.mhe_glwe_shamir_share_finalize_tmp_bytes()),
    );
    let secrets: Vec<Secret<BE>> = (0..parties).map(|i| secret_from_seed(module, [200 + i as u8; 32])).collect();

    let polys: Vec<_> = secrets
        .iter()
        .enumerate()
        .map(|(j, (sk, _))| {
            let mut poly = module.glwe_shamir_polynomial_alloc(&layout);
            module.mhe_glwe_shamir_polynomial_gen(&mut poly, sk, &mut Source::new([210 + j as u8; 32]), &mut scratch.borrow());
            poly
        })
        .collect();
    let shares: Vec<GLWEShamirShareOwned<BE>> = (1..=parties as u32)
        .map(|recipient| {
            let mut acc = module.glwe_shamir_share_alloc(&layout);
            let mut share = module.glwe_shamir_share_alloc(&layout);
            for (j, poly) in polys.iter().enumerate() {
                let dst = if j == 0 { &mut acc } else { &mut share };
                module.mhe_glwe_shamir_share_gen(dst, poly, recipient, &mut scratch.borrow());
                if j > 0 {
                    module.mhe_glwe_shamir_share_aggregate(&mut acc, &share);
                }
            }
            acc
        })
        .collect();
    (secrets, layout, shares)
}

/// The limbs of `secret` are balanced digits and its last limb has no bits
/// below its precision.
fn assert_canonical(secret: &GLWEWideSecret<AlignedBuf, i64>) {
    let (base2k, k) = (secret.base2k().as_usize(), secret.k().as_usize());
    let size = k.div_ceil(base2k);
    let (half, pad) = (1i64 << (base2k - 1), size * base2k - k);
    for col in 0..secret.rank().as_usize() {
        for limb in 0..size {
            for &x in secret.data().at(col, limb) {
                assert!((-half..half).contains(&x), "combined share digit out of range");
                if limb == size - 1 {
                    assert_eq!(x % (1 << pad), 0, "combined share has bits below its precision");
                }
            }
        }
    }
}
