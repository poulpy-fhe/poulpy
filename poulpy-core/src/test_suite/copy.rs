//! Exact integer checks for value-preserving GLWE copies and rounded narrowing.

use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{HostDataMut, HostDataRef, Module, ScratchOwned, VecZnx, ZnxView, ZnxViewMut},
    test_suite::TestParams,
};

use crate::{
    GLWEAdd, GLWECopy, GLWENormalize, GLWERotate, GLWEShift, GLWESub,
    layouts::{GLWE, GLWEInfos, LWEInfos, ModuleCoreAlloc, SetK},
    test_suite::noise::glwe_tensor::assert_canonical,
};

/// Decode the centered digits independently of the normalization implementation.
fn numerator(data: &VecZnx<impl HostDataRef, i64>, col: usize, j: usize, base: usize, k: usize) -> i128 {
    let mut value = 0i128;
    for limb in 0..k.div_ceil(base) {
        let digit = i128::from(data.at(col, limb)[j]);
        let position = (limb + 1) * base;
        value += if position <= k {
            digit << (k - position)
        } else {
            digit >> (position - k)
        };
    }
    value.rem_euclid(1i128 << k)
}

pub fn test_glwe_copy<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GLWECopy<BE> + GLWERotate<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base = params.base2k;
    let provenance = |rank| {
        Some(crate::ComponentNoise::from_secret(
            crate::Distribution::TernaryProb(0.5),
            rank,
        ))
    };
    let stale_provenance = |rank| {
        Some(crate::ComponentNoise::from_secret(
            crate::Distribution::TernaryProb(1.0),
            rank,
        ))
    };
    let half = 1i64 << (base - 1);
    // Extremal digits, carries, both signs and rounding ties.
    let digits = [-half, -half + 1, -9, -8, -7, -1, 0, 1, 7, 8, 9, half - 1];
    for src_k in [2 * base, 2 * base - 3] {
        // Rank-zero shares may carry body noise. Copying them into a larger
        // rank preserves that term and appends zero-noise mask components.
        for (src_rank, dst_rank, tagged) in [
            (0usize, 0usize, false),
            (0, 0, true),
            (0, 2, false),
            (0, 2, true),
            (1, 1, true),
            (2, 2, true),
        ] {
            let mut src: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc(base.into(), src_k.into(), src_rank.into());
            src.noise = if tagged { provenance(src_rank) } else { None };
            for col in 0..=src_rank {
                for limb in 0..src_k.div_ceil(base) {
                    let padding = ((limb + 1) * base).saturating_sub(src_k);
                    for (j, coeff) in src.data.at_mut(col, limb).iter_mut().enumerate() {
                        *coeff = (digits[(j + col * 3 + limb * 5) % digits.len()] >> padding) << padding;
                    }
                }
            }
            let original = src.clone();
            for dst_base in [base - 1, base, base + 1] {
                for dst_k in [base - 3, src_k - 3, src_k, src_k + 3] {
                    // Dirty spare limbs must be cleared, including when only
                    // logical precision changes within the same physical limb.
                    let mut dst: GLWE<BE::OwnedBuf, BE::ZnxWord> =
                        module.glwe_alloc(dst_base.into(), (dst_k + dst_base).into(), dst_rank.into());
                    dst.set_k(dst_k.into());
                    dst.noise = stale_provenance(dst_rank);
                    dst.data.raw_mut().fill(0x55);
                    let layout = dst.glwe_layout();
                    let tmp_bytes = module.glwe_copy_tmp_bytes(&dst, &src);
                    if dst_base == base && dst_k >= src_k {
                        assert_eq!(tmp_bytes, 0, "a lossless same-radix copy needs no scratch");
                    }
                    let mut scratch = ScratchOwned::<BE>::alloc(tmp_bytes);
                    module.glwe_copy(&mut dst, &src, &mut scratch.borrow());
                    assert_eq!(dst.glwe_layout(), layout, "copy changed destination layout");
                    let expected_noise = if dst_k < src_k {
                        None
                    } else {
                        src.noise().map(|noise| noise.with_rank(dst_rank))
                    };
                    assert!(
                        dst.noise() == expected_noise,
                        "copy noise: rank {src_rank} -> {dst_rank}, tagged={tagged}"
                    );
                    assert!(src == original, "copy changed the source");
                    assert_canonical(&dst.data, dst_base, dst_k);
                    for col in 0..=dst_rank {
                        for j in 0..module.n() {
                            let expected = if col > src_rank {
                                0
                            } else {
                                let value = numerator(&src.data, col, j, base, src_k);
                                if dst_k < src_k {
                                    // Round ties toward +infinity, modulo the torus.
                                    let drop = src_k - dst_k;
                                    ((value + (1i128 << (drop - 1))) >> drop).rem_euclid(1i128 << dst_k)
                                } else {
                                    value << (dst_k - src_k)
                                }
                            };
                            assert_eq!(
                                numerator(&dst.data, col, j, dst_base, dst_k),
                                expected,
                                "copy ({base}, {src_k}, {src_rank}) -> ({dst_base}, {dst_k}, {dst_rank}), col={col}, j={j}"
                            );
                        }
                    }
                }
            }
        }
    }

    // CMux and sign-extension copy unnormalized intermediates of the same
    // layout before further arithmetic. This fast path must stay bit-exact.
    let mut src: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc(base.into(), (2 * base).into(), 1usize.into());
    for (j, digit) in src.data.raw_mut().iter_mut().enumerate() {
        *digit = if j % 2 == 0 { half + 7 } else { -half - 9 };
    }
    let mut dst: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&src);
    assert_eq!(module.glwe_copy_tmp_bytes(&dst, &src), 0);
    let mut scratch = ScratchOwned::<BE>::alloc(0);
    module.glwe_copy(&mut dst, &src, &mut scratch.borrow());
    assert!(dst == src, "values differ");

    // Every evaluation clears freshness, even matching inputs or an identity.
    src.noise = provenance(1);
    module.glwe_rotate(1, &mut dst, &src);
    assert_eq!(dst.noise(), None);
    module.glwe_add_into(&mut dst, &src, &src);
    assert_eq!(dst.noise(), None);
    module.glwe_sub_assign(&mut dst, &src);
    assert_eq!(dst.noise(), None);
    src.noise = stale_provenance(1);
    module.glwe_add_assign(&mut dst, &src);
    assert_eq!(dst.noise(), None);
    let mut scratch = ScratchOwned::<BE>::alloc(module.glwe_normalize_tmp_bytes());
    module.glwe_normalize(&mut dst, &src, &mut scratch.borrow());
    assert_eq!(dst.noise(), None);

    // Plaintext arithmetic also invalidates the fresh-encryption estimate.
    src.noise = provenance(1);
    let mut scratch = ScratchOwned::<BE>::alloc(module.glwe_shift_tmp_bytes(dst.data.size()));
    for rank in [0usize, 1] {
        let mut other: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc(base.into(), (2 * base).into(), rank.into());
        for metadata in [None, provenance(rank), stale_provenance(rank)] {
            other.noise = metadata.clone();
            let expected = None;
            module.glwe_add_into(&mut dst, &src, &other);
            assert_eq!(dst.noise(), expected, "add: rank={rank}, metadata={metadata:?}");
            module.glwe_add_into(&mut dst, &other, &src);
            assert_eq!(dst.noise(), expected, "reversed add: rank={rank}");
            module.glwe_sub(&mut dst, &src, &other);
            assert_eq!(dst.noise(), expected, "sub: rank={rank}");
            module.glwe_sub(&mut dst, &other, &src);
            assert_eq!(dst.noise(), expected, "reversed sub: rank={rank}");
            dst.noise = provenance(1);
            module.glwe_add_assign(&mut dst, &other);
            assert_eq!(dst.noise(), expected, "add assign: rank={rank}");
            dst.noise = provenance(1);
            module.glwe_sub_assign(&mut dst, &other);
            assert_eq!(dst.noise(), expected, "sub assign: rank={rank}");
            dst.noise = provenance(1);
            module.glwe_sub_negate_assign(&mut dst, &other);
            assert_eq!(dst.noise(), expected, "sub negate assign: rank={rank}");
            let mask = dst.data.at(1, 0).to_vec();
            dst.noise = provenance(1);
            module.glwe_lsh_add(&mut dst, &other, 1, &mut scratch.borrow());
            assert_eq!(dst.noise(), expected, "shift add: rank={rank}");
            dst.noise = provenance(1);
            module.glwe_lsh_sub(&mut dst, &other, 1, &mut scratch.borrow());
            assert_eq!(dst.noise(), expected, "shift sub: rank={rank}");
            if rank == 0 {
                assert!(dst.data.at(1, 0) == mask, "plaintext shifts changed the mask");
            }
        }
    }
}
