use poulpy_hal::{
    api::{
        CnvPVecAlloc, Convolution, ModuleNew, ScratchOwnedAlloc, VecZnxBigAlloc, VecZnxDftAlloc, VecZnxDftApply,
        VecZnxIdftApplyTmpA,
    },
    layouts::{
        Backend, CnvPVecLToBackendMut, CnvPVecLToBackendRef, CnvPVecRToBackendMut, CnvPVecRToBackendRef, HostBytesBackend,
        HostDataRef, Module, PrepareHint, ScratchOwned, VecZnxBigToBackendMut, VecZnxDftToBackendMut, ZnxView, ZnxViewMut,
    },
    test_suite::{upload_vec_znx, vec_znx_backend_ref},
};

/// Exact roundtrip and negacyclic convolution checks with linear-time integer oracles.
pub fn test_ntt_ring_degree<BE>(n: usize, dense_bits: u32)
where
    BE: Backend<ZnxWord = i64, BigWord = i128>,
    BE::OwnedBuf: HostDataRef,
    Module<BE>: ModuleNew<BE>
        + CnvPVecAlloc<BE>
        + Convolution<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxIdftApplyTmpA<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let module = Module::<BE>::new(n as u64);
    let host = Module::<HostBytesBackend>::new(n as u64);
    let mut a = host.vec_znx_alloc(1, 2);
    for limb in 0..2 {
        for (i, x) in a.at_mut(0, limb).iter_mut().enumerate() {
            *x = match i % 4 {
                0 => i64::MIN,
                1 => i64::MAX,
                _ => (i as u64).wrapping_mul(0x9e3779b97f4a7c15).rotate_left(limb as u32) as i64,
            };
        }
    }
    let input = upload_vec_znx::<BE>(&a);
    let mut dft = module.vec_znx_dft_alloc(1, 2);
    let mut big = module.vec_znx_big_alloc(1, 2);
    module.vec_znx_dft_apply(1, 0, &mut dft.to_backend_mut(), 0, &vec_znx_backend_ref::<BE>(&input), 0);
    module.vec_znx_idft_apply_tmpa(&mut big.to_backend_mut(), 0, &mut dft.to_backend_mut(), 0);
    for limb in 0..2 {
        for (i, (&got, &want)) in big.at(0, limb).iter().zip(a.at(0, limb)).enumerate() {
            assert_eq!(got, want as i128, "roundtrip n={n} limb={limb} i={i}");
        }
    }

    let mut b = host.vec_znx_alloc(1, 2);
    let mut left = module.cnv_pvec_left_alloc(1, 2, PrepareHint::Reuse);
    let mut right = module.cnv_pvec_right_alloc(1, 2, PrepareHint::Reuse);
    let mut product = module.vec_znx_dft_alloc(2, 4);
    let mut result = module.vec_znx_big_alloc(1, 4);
    let mut scratch = ScratchOwned::<BE>::alloc(
        module
            .cnv_prepare_left_tmp_bytes(4, 2)
            .max(module.cnv_prepare_right_tmp_bytes(4, 2))
            .max(module.cnv_apply_dft_tmp_bytes(0, 4, 2, 2)),
    );
    for dense in [false, true] {
        let mut expected = vec![vec![0i128; n]; 4];
        for limb in 0..2 {
            for (i, x) in a.at_mut(0, limb).iter_mut().enumerate() {
                *x = if dense {
                    (1i64 << dense_bits) * if limb == 0 { 1 } else { -1 }
                } else {
                    ((i as u64).wrapping_mul(0xd1342543de82ef95).rotate_left(limb as u32) as i64) >> 12
                };
            }
            b.at_mut(0, limb).fill(0);
            if dense {
                b.at_mut(0, limb).fill(if limb == 0 {
                    1i64 << dense_bits
                } else {
                    -(1i64 << (dense_bits - 1))
                });
            } else {
                for (index, value) in [(0, 3), (1, -5), (n / 2 - 1, 7), (n / 2, -11), (n - 1, 13)] {
                    b.at_mut(0, limb)[index] = value * (1i64 << (43 + limb));
                }
            }
        }
        for a_limb in 0..2 {
            for b_limb in 0..2 {
                let out = &mut expected[a_limb + b_limb];
                if dense {
                    let scale = a.at(0, a_limb)[0] as i128 * b.at(0, b_limb)[0] as i128;
                    for (i, x) in out.iter_mut().enumerate() {
                        *x += (2 * i as i128 + 2 - n as i128) * scale;
                    }
                } else {
                    for (j, &y) in b.at(0, b_limb).iter().enumerate().filter(|(_, y)| **y != 0) {
                        for (i, &x) in a.at(0, a_limb).iter().enumerate() {
                            let k = i + j;
                            out[k % n] += x as i128 * y as i128 * if k < n { 1 } else { -1 };
                        }
                    }
                }
            }
        }
        let a_backend = upload_vec_znx::<BE>(&a);
        let b_backend = upload_vec_znx::<BE>(&b);
        module.cnv_prepare_left(
            &mut left.to_backend_mut(),
            &vec_znx_backend_ref::<BE>(&a_backend),
            &mut scratch.arena(),
        );
        module.cnv_prepare_right(
            &mut right.to_backend_mut(),
            &vec_znx_backend_ref::<BE>(&b_backend),
            &mut scratch.arena(),
        );
        module.cnv_apply_dft(
            0,
            &mut product.to_backend_mut(),
            1,
            &left.to_backend_ref(),
            0,
            &right.to_backend_ref(),
            0,
            &mut scratch.arena(),
        );
        module.vec_znx_idft_apply_tmpa(&mut result.to_backend_mut(), 0, &mut product.to_backend_mut(), 1);
        for (limb, want) in expected.iter().enumerate() {
            for (i, (&got, &want)) in result.at(0, limb).iter().zip(want).enumerate() {
                assert_eq!(got, want, "convolution n={n} dense={dense} limb={limb} i={i}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn large_ring_ntt_log17() {
        super::test_ntt_ring_degree::<crate::NTT4x30Ref>(1 << 17, 50);
    }

    #[test]
    fn large_ring_ntt_log18() {
        super::test_ntt_ring_degree::<crate::NTT4x30Ref>(1 << 18, 50);
    }
}
