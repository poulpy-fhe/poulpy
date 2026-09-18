use poulpy_hal::{
    api::{ModuleNew, VecZnxBigAlloc, VecZnxDftAlloc, VecZnxDftApply, VecZnxIdftApplyTmpA},
    layouts::{
        Backend, HostBytesBackend, HostDataRef, Module, VecZnxBigToBackendMut, VecZnxDftToBackendMut, ZnxView, ZnxViewMut,
    },
    test_suite::{upload_vec_znx, vec_znx_backend_ref},
};

/// Exact DFT/IDFT roundtrip at the requested ring degree, including signed i64 limits.
pub fn test_ntt_ring_degree<BE>(n: usize)
where
    BE: Backend<ZnxWord = i64, BigWord = i128>,
    BE::OwnedBuf: HostDataRef,
    Module<BE>: ModuleNew<BE> + VecZnxBigAlloc<BE> + VecZnxDftAlloc<BE> + VecZnxDftApply<BE> + VecZnxIdftApplyTmpA<BE>,
{
    let module = Module::<BE>::new(n as u64);
    let host = Module::<HostBytesBackend>::new(n as u64);
    let mut a = host.vec_znx_alloc(n, 1, 2);
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
    let mut dft = module.vec_znx_dft_alloc(n, 1, 2);
    let mut big = module.vec_znx_big_alloc(n, 1, 2);
    module.vec_znx_dft_apply(1, 0, &mut dft.to_backend_mut(), 0, &vec_znx_backend_ref::<BE>(&input), 0);
    module.vec_znx_idft_apply_tmpa(&mut big.to_backend_mut(), 0, &mut dft.to_backend_mut(), 0);
    for limb in 0..2 {
        for (i, (&got, &want)) in big.at(0, limb).iter().zip(a.at(0, limb)).enumerate() {
            assert_eq!(got, want as i128, "roundtrip n={n} limb={limb} i={i}");
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn large_ring_ntt_log17() {
        super::test_ntt_ring_degree::<crate::NTT4x30Ref>(1 << 17);
    }

    #[test]
    fn large_ring_ntt_log18() {
        super::test_ntt_ring_degree::<crate::NTT4x30Ref>(1 << 18);
    }
}
