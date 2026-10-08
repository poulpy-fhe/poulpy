//! Rayon-scheduled wrapper for the NEON NTT4x30 backend.

use poulpy_cpu_rayon::ntt4x30::{PackedNtt4x30Base, PackedWord};
use poulpy_hal::{
    execution::TaskExecutor,
    layouts::{
        CnvDftAccTerm, CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecRBackendMut, CnvPVecRBackendRef, ConjugateInvariant,
        HostDataMut, HostDataRef, Module, Ring, Standard, VecZnxBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef,
        VmpPMatBackendRef,
    },
    oep::HalVecZnxDftImpl,
};

use super::{NTT4x30Neon, NTT4x30NeonRayon, convolution, vec_znx_dft, vmp};

/// The drivers of the NEON backend over one ring.
macro_rules! impl_packed_base {
    ($ring:ty) => {
        #[allow(clippy::too_many_arguments)]
        impl PackedNtt4x30Base for NTT4x30Neon<$ring> {
            fn idft_tmp_words(n: usize) -> usize {
                vec_znx_dft::idft_tmp_words(n)
            }

            fn dft_limb<E: TaskExecutor>(module: &Module<Self>, n: usize, dst: &mut [u32], src: Option<&[i64]>) {
                vec_znx_dft::dft_limb::<$ring, E>(module, n, dst, src)
            }

            fn idft_limb<E: TaskExecutor>(module: &Module<Self>, n: usize, dst: &mut [i128], src: &[u32], tmp: &mut [u64]) {
                vec_znx_dft::idft_limb::<$ring, E>(module, n, dst, src, tmp)
            }

            fn idft_limb_tmpa<E: TaskExecutor>(module: &Module<Self>, n: usize, dst: &mut [i128], src: &mut [u32]) {
                vec_znx_dft::idft_limb_tmpa::<$ring, E>(module, n, dst, src)
            }

            fn idft_limb_compact<E: TaskExecutor>(module: &Module<Self>, n: usize, slot: &mut [u32], tmp: &mut [u64]) {
                vec_znx_dft::idft_limb_compact::<$ring, E>(module, n, slot, tmp)
            }

            fn vmp_apply_tmp_bytes(a_size: usize, b_rows: usize, b_cols_in: usize) -> usize {
                vmp::vmp_apply_tmp_bytes_neon(a_size, b_rows, b_cols_in)
            }

            fn vmp_apply_dft_to_dft<E: TaskExecutor>(
                module: &Module<Self>,
                res: &mut VecZnxDftBackendMut<'_, Self>,
                a: &VecZnxDftBackendRef<'_, Self>,
                pmat: &VmpPMatBackendRef<'_, Self>,
                limb_offset: usize,
                tmp: &mut [u64],
            ) {
                vmp::vmp_apply_dft_to_dft_neon::<$ring, E>(module, res, a, pmat, limb_offset, tmp)
            }

            fn vmp_apply_dft_to_dft_add<E: TaskExecutor>(
                module: &Module<Self>,
                res: &mut VecZnxDftBackendMut<'_, Self>,
                a: &VecZnxDftBackendRef<'_, Self>,
                pmat: &VmpPMatBackendRef<'_, Self>,
                limb_offset: usize,
                tmp: &mut [u64],
            ) {
                vmp::vmp_apply_dft_to_dft_add_neon::<$ring, E>(module, res, a, pmat, limb_offset, tmp)
            }

            fn vec_znx_dft_add<E: TaskExecutor>(
                res: &mut VecZnxDftBackendMut<'_, Self>,
                res_col: usize,
                a: &VecZnxDftBackendRef<'_, Self>,
                a_col: usize,
                b: &VecZnxDftBackendRef<'_, Self>,
                b_col: usize,
            ) {
                vec_znx_dft::vec_znx_dft_add::<$ring, E>(res, res_col, a, a_col, b, b_col)
            }

            fn vec_znx_dft_add_assign<E: TaskExecutor>(
                res: &mut VecZnxDftBackendMut<'_, Self>,
                res_col: usize,
                a: &VecZnxDftBackendRef<'_, Self>,
                a_col: usize,
            ) {
                vec_znx_dft::vec_znx_dft_add_assign::<$ring, E>(res, res_col, a, a_col)
            }

            fn vec_znx_dft_sub<E: TaskExecutor>(
                res: &mut VecZnxDftBackendMut<'_, Self>,
                res_col: usize,
                a: &VecZnxDftBackendRef<'_, Self>,
                a_col: usize,
                b: &VecZnxDftBackendRef<'_, Self>,
                b_col: usize,
            ) {
                vec_znx_dft::vec_znx_dft_sub::<$ring, E>(res, res_col, a, a_col, b, b_col)
            }

            fn vec_znx_dft_sub_assign<E: TaskExecutor>(
                res: &mut VecZnxDftBackendMut<'_, Self>,
                res_col: usize,
                a: &VecZnxDftBackendRef<'_, Self>,
                a_col: usize,
            ) {
                vec_znx_dft::vec_znx_dft_sub_assign::<$ring, E>(res, res_col, a, a_col)
            }

            fn vec_znx_dft_sub_negate_assign<E: TaskExecutor>(
                res: &mut VecZnxDftBackendMut<'_, Self>,
                res_col: usize,
                a: &VecZnxDftBackendRef<'_, Self>,
                a_col: usize,
            ) {
                vec_znx_dft::vec_znx_dft_sub_negate_assign::<$ring, E>(res, res_col, a, a_col)
            }

            fn vec_znx_dft_copy<E: TaskExecutor>(
                step: usize,
                offset: usize,
                res: &mut VecZnxDftBackendMut<'_, Self>,
                res_col: usize,
                a: &VecZnxDftBackendRef<'_, Self>,
                a_col: usize,
            ) {
                vec_znx_dft::vec_znx_dft_copy::<$ring, E>(step, offset, res, res_col, a, a_col)
            }

            fn vec_znx_dft_automorphism_add<E: TaskExecutor>(
                plan: &<Self as HalVecZnxDftImpl>::AutomorphismPlan,
                res: &mut VecZnxDftBackendMut<'_, Self>,
                res_col: usize,
                a: &VecZnxDftBackendRef<'_, Self>,
                a_col: usize,
            ) {
                vec_znx_dft::vec_znx_dft_automorphism_add::<$ring, E>(plan, res, res_col, a, a_col)
            }

            fn cnv_prepare_tmp_bytes(n: usize) -> usize {
                convolution::cnv_prepare_tmp_bytes(n)
            }

            fn cnv_prepare_left<BE: PackedWord, E: TaskExecutor>(
                base: &Module<Self>,
                module: &Module<BE>,
                res: &mut CnvPVecLBackendMut<'_, BE>,
                a: &VecZnxBackendRef<'_, BE>,
                tmp: &mut [u64],
            ) where
                for<'a> BE::BufRef<'a>: HostDataRef,
                for<'a> BE::BufMut<'a>: HostDataMut,
            {
                convolution::cnv_prepare_left::<BE, E>(module, res, a, tmp, |n, dst, src, prepared| {
                    vec_znx_dft::dft_limb_scaled::<$ring, E>(base, n, dst, src, prepared)
                })
            }

            fn cnv_prepare_right<BE: PackedWord, E: TaskExecutor>(
                base: &Module<Self>,
                module: &Module<BE>,
                res: &mut CnvPVecRBackendMut<'_, BE>,
                a: &VecZnxBackendRef<'_, BE>,
                tmp: &mut [u64],
            ) where
                for<'a> BE::BufRef<'a>: HostDataRef,
                for<'a> BE::BufMut<'a>: HostDataMut,
            {
                convolution::cnv_prepare_right::<BE, E>(module, res, a, tmp, |n, dst, src, prepared| {
                    vec_znx_dft::dft_limb_scaled::<$ring, E>(base, n, dst, src, prepared)
                })
            }

            fn cnv_prepare_self<BE: PackedWord, E: TaskExecutor>(
                base: &Module<Self>,
                module: &Module<BE>,
                left: &mut CnvPVecLBackendMut<'_, BE>,
                right: &mut CnvPVecRBackendMut<'_, BE>,
                a: &VecZnxBackendRef<'_, BE>,
                tmp: &mut [u64],
            ) where
                for<'a> BE::BufRef<'a>: HostDataRef,
                for<'a> BE::BufMut<'a>: HostDataMut,
            {
                convolution::cnv_prepare_self::<BE, E>(module, left, right, a, tmp, |n, dst, src, prepared| {
                    vec_znx_dft::dft_limb_scaled::<$ring, E>(base, n, dst, src, prepared)
                })
            }

            fn cnv_apply_tmp_words(_res_size: usize) -> usize {
                // The stages of the NEON convolution live on the stack of each task.
                0
            }

            fn cnv_apply_dft<BE: PackedWord, E: TaskExecutor>(
                module: &Module<BE>,
                cnv_offset: usize,
                res: &mut VecZnxDftBackendMut<'_, BE>,
                res_col: usize,
                a: &CnvPVecLBackendRef<'_, BE>,
                a_col: usize,
                b: &CnvPVecRBackendRef<'_, BE>,
                b_col: usize,
                tmp: &mut [u32],
            ) where
                for<'a> BE::BufRef<'a>: HostDataRef,
                for<'a> BE::BufMut<'a>: HostDataMut,
            {
                let _ = tmp;
                unsafe { convolution::cnv_apply_dft::<BE, E>(module, cnv_offset, res, res_col, a, a_col, b, b_col) }
            }

            fn cnv_apply_dft_add<BE: PackedWord, E: TaskExecutor>(
                module: &Module<BE>,
                cnv_offset: usize,
                res: &mut VecZnxDftBackendMut<'_, BE>,
                res_col: usize,
                a: &CnvPVecLBackendRef<'_, BE>,
                a_col: usize,
                b: &CnvPVecRBackendRef<'_, BE>,
                b_col: usize,
                tmp: &mut [u32],
            ) where
                for<'a> BE::BufRef<'a>: HostDataRef,
                for<'a> BE::BufMut<'a>: HostDataMut,
            {
                let _ = tmp;
                unsafe { convolution::cnv_apply_dft_add::<BE, E>(module, cnv_offset, res, res_col, a, a_col, b, b_col) }
            }

            fn cnv_apply_dft_sum<BE: PackedWord, E: TaskExecutor>(
                module: &Module<BE>,
                cnv_offset: usize,
                res: &mut VecZnxDftBackendMut<'_, BE>,
                res_col: usize,
                terms: &[CnvDftAccTerm<'_, BE>],
                tmp: &mut [u32],
            ) where
                for<'a> BE::BufRef<'a>: HostDataRef,
                for<'a> BE::BufMut<'a>: HostDataMut,
            {
                let _ = tmp;
                unsafe { convolution::cnv_apply_dft_sum_neon::<BE, E>(module, cnv_offset, res, res_col, terms, &mut []) }
            }

            fn cnv_pairwise_apply_dft<BE: PackedWord, E: TaskExecutor>(
                module: &Module<BE>,
                cnv_offset: usize,
                res: &mut VecZnxDftBackendMut<'_, BE>,
                res_col: usize,
                a: &CnvPVecLBackendRef<'_, BE>,
                b: &CnvPVecRBackendRef<'_, BE>,
                i: usize,
                j: usize,
                tmp: &mut [u32],
            ) where
                for<'a> BE::BufRef<'a>: HostDataRef,
                for<'a> BE::BufMut<'a>: HostDataMut,
            {
                let _ = tmp;
                unsafe { convolution::cnv_pairwise_apply_dft::<BE, E>(module, cnv_offset, res, res_col, a, b, i, j) }
            }
        }
    };
}

impl_packed_base!(Standard);
impl_packed_base!(ConjugateInvariant);

mod standard {
    use poulpy_cpu_portable::kernels::znx::ZnxAutomorphismRotate;

    use super::{NTT4x30Neon, NTT4x30NeonRayon};

    poulpy_cpu_rayon::impl_ntt4x30_rayon_backend!(NTT4x30NeonRayon, NTT4x30Neon);

    unsafe impl poulpy_hal::oep::HalVecZnxMonomialImpl for NTT4x30NeonRayon {
        poulpy_cpu_portable::hal_impl_vec_znx_monomial!();
    }

    unsafe impl poulpy_hal::oep::HalVecZnxCIImpl for NTT4x30NeonRayon {
        poulpy_cpu_portable::hal_impl_vec_znx_ci!();
    }

    impl ZnxAutomorphismRotate for NTT4x30NeonRayon {
        #[inline(always)]
        fn znx_automorphism_rotate(p: i64, k: i64, res: &mut [i64], a: &[i64]) {
            <NTT4x30Neon as ZnxAutomorphismRotate>::znx_automorphism_rotate(p, k, res, a)
        }
    }
}

mod conjugate_invariant {
    use poulpy_hal::layouts::ConjugateInvariant;

    use super::{NTT4x30Neon, NTT4x30NeonRayon};

    poulpy_cpu_rayon::impl_ntt4x30_rayon_backend!(NTT4x30NeonRayon<ConjugateInvariant>, NTT4x30Neon<ConjugateInvariant>);
}

impl<R: Ring> poulpy_hal::execution::ScratchWorkers for NTT4x30NeonRayon<R> {
    const PREPARE: usize = 32;
    const APPLY: usize = 32;
    const VMP: usize = 32;
    const IDFT: usize = 32;
}

impl<R: Ring> poulpy_cpu_rayon::RayonTuning for NTT4x30NeonRayon<R> {
    const COEFF_MIN_LEN: usize = 1 << 17;
    const COEFF_MIN_TASK: usize = 1 << 13;
    const NORMALIZE_MIN_TASK: usize = 1 << 12;
}

#[cfg(test)]
mod tests {
    use poulpy_cpu_portable::kernels::znx::ZnxAdd;
    use poulpy_hal::{layouts::Module, test_suite::convolution::test_convolution_by_const};

    use super::NTT4x30NeonRayon;

    #[test]
    fn coefficient_add_matches_wrapping_arithmetic() {
        let a = vec![i64::MAX; 1 << 16];
        let b = vec![1; 1 << 16];
        let mut actual = vec![0; 1 << 16];
        <NTT4x30NeonRayon as ZnxAdd>::znx_add(&mut actual, &a, &b);
        assert!(actual.iter().all(|&x| x == i64::MIN));
    }

    #[test]
    fn convolution_by_const() {
        test_convolution_by_const(&Module::<NTT4x30NeonRayon>::new(1 << 8), 1 << 8, 50);
    }
}
