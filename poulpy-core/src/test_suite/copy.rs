//! Exact integer checks for value-preserving GLWE copies and rounded narrowing.

use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{HostDataMut, HostDataRef, Module, ScratchOwned, VecZnx, ZnxView, ZnxViewMut},
    test_suite::TestParams,
};

use crate::{
    GLWECopy,
    layouts::{GLWE, GLWEInfos, ModuleCoreAlloc, SetK},
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
    Module<BE>: GLWECopy<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base = params.base2k;
    let half = 1i64 << (base - 1);
    // Extremal digits, carries, both signs and rounding ties.
    let digits = [-half, -half + 1, -9, -8, -7, -1, 0, 1, 7, 8, 9, half - 1];
    for src_k in [2 * base, 2 * base - 3] {
        for (src_rank, dst_rank) in [(0usize, 0usize), (0, 2), (1, 1), (2, 2)] {
            let mut src: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc(base.into(), src_k.into(), src_rank.into());
            for col in 0..=src_rank {
                for limb in 0..src_k.div_ceil(base) {
                    let padding = ((limb + 1) * base).saturating_sub(src_k);
                    for (j, coeff) in src.data.at_mut(col, limb).iter_mut().enumerate() {
                        *coeff = (digits[(j + col * 3 + limb * 5) % digits.len()] >> padding) << padding;
                    }
                }
            }
            let original = src.data.raw().to_vec();
            for dst_base in [base - 1, base, base + 1] {
                for dst_k in [base - 3, src_k - 3, src_k, src_k + 3] {
                    // Dirty spare limbs must be cleared, including when only
                    // logical precision changes within the same physical limb.
                    let mut dst: GLWE<BE::OwnedBuf, BE::ZnxWord> =
                        module.glwe_alloc(dst_base.into(), (dst_k + dst_base).into(), dst_rank.into());
                    dst.set_k(dst_k.into());
                    dst.data.raw_mut().fill(0x55);
                    let layout = dst.glwe_layout();
                    let tmp_bytes = module.glwe_copy_tmp_bytes(&dst, &src);
                    if dst_base == base && dst_k >= src_k {
                        assert_eq!(tmp_bytes, 0, "a lossless same-radix copy needs no scratch");
                    }
                    let mut scratch = ScratchOwned::<BE>::alloc(tmp_bytes);
                    module.glwe_copy(&mut dst, &src, &mut scratch.borrow());
                    assert_eq!(dst.glwe_layout(), layout, "copy changed destination metadata");
                    assert_eq!(src.data.raw(), original, "copy changed the source");
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
    assert_eq!(dst.data.raw(), src.data.raw());
}
