//! Rayon-scheduled wrapper for the portable NTT4x30 backend.

use poulpy_cpu_portable::{NTT4x30Portable, ntt4x30::drivers};
use poulpy_hal::{
    execution::TaskExecutor,
    layouts::{
        CnvDftAccTerm, CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecRBackendMut, CnvPVecRBackendRef, ConjugateInvariant,
        HostDataMut, HostDataRef, Module, Standard, VecZnxBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef,
        VmpPMatBackendRef,
    },
    oep::HalVecZnxDftImpl,
};

use super::NTT4x30PortableRayon;
use crate::ntt4x30::{PackedNtt4x30Base, PackedWord};

/// The drivers of the portable backend over one ring.
macro_rules! impl_packed_base {
    ($ring:ty) => {
        #[allow(clippy::too_many_arguments)]
        impl PackedNtt4x30Base<$ring> for NTT4x30Portable {
            fn idft_tmp_words(n: usize) -> usize {
                drivers::idft_tmp_words(n)
            }

            fn dft_limb<E: TaskExecutor>(module: &Module<Self>, n: usize, dst: &mut [u32], src: Option<&[i64]>) {
                drivers::dft_limb::<$ring, E>(module, n, dst, src)
            }

            fn idft_limb<E: TaskExecutor>(module: &Module<Self>, n: usize, dst: &mut [i128], src: &[u32], tmp: &mut [u64]) {
                drivers::idft_limb::<$ring, E>(module, n, dst, src, tmp)
            }

            fn idft_limb_compact<E: TaskExecutor>(module: &Module<Self>, n: usize, slot: &mut [u32], tmp: &mut [u64]) {
                drivers::idft_limb_compact::<$ring, E>(module, n, slot, tmp)
            }

            fn vmp_apply_tmp_bytes(a_size: usize, b_rows: usize, b_cols_in: usize) -> usize {
                drivers::vmp_apply_tmp_bytes(a_size, b_rows, b_cols_in)
            }

            fn vmp_apply_dft_to_dft<E: TaskExecutor>(
                _module: &Module<Self>,
                res: &mut VecZnxDftBackendMut<'_, Self, $ring>,
                a: &VecZnxDftBackendRef<'_, Self, $ring>,
                pmat: &VmpPMatBackendRef<'_, Self, $ring>,
                limb_offset: usize,
                tmp: &mut [u64],
            ) {
                drivers::vmp_apply_dft_to_dft::<$ring, E>(res, a, pmat, limb_offset, tmp)
            }

            fn vmp_apply_dft_to_dft_add<E: TaskExecutor>(
                _module: &Module<Self>,
                res: &mut VecZnxDftBackendMut<'_, Self, $ring>,
                a: &VecZnxDftBackendRef<'_, Self, $ring>,
                pmat: &VmpPMatBackendRef<'_, Self, $ring>,
                limb_offset: usize,
                tmp: &mut [u64],
            ) {
                drivers::vmp_apply_dft_to_dft_add::<$ring, E>(res, a, pmat, limb_offset, tmp)
            }

            fn vec_znx_dft_add<E: TaskExecutor>(
                res: &mut VecZnxDftBackendMut<'_, Self, $ring>,
                res_col: usize,
                a: &VecZnxDftBackendRef<'_, Self, $ring>,
                a_col: usize,
                b: &VecZnxDftBackendRef<'_, Self, $ring>,
                b_col: usize,
            ) {
                drivers::vec_znx_dft_add::<$ring, E>(res, res_col, a, a_col, b, b_col)
            }

            fn vec_znx_dft_add_assign<E: TaskExecutor>(
                res: &mut VecZnxDftBackendMut<'_, Self, $ring>,
                res_col: usize,
                a: &VecZnxDftBackendRef<'_, Self, $ring>,
                a_col: usize,
            ) {
                drivers::vec_znx_dft_add_assign::<$ring, E>(res, res_col, a, a_col)
            }

            fn vec_znx_dft_sub<E: TaskExecutor>(
                res: &mut VecZnxDftBackendMut<'_, Self, $ring>,
                res_col: usize,
                a: &VecZnxDftBackendRef<'_, Self, $ring>,
                a_col: usize,
                b: &VecZnxDftBackendRef<'_, Self, $ring>,
                b_col: usize,
            ) {
                drivers::vec_znx_dft_sub::<$ring, E>(res, res_col, a, a_col, b, b_col)
            }

            fn vec_znx_dft_sub_assign<E: TaskExecutor>(
                res: &mut VecZnxDftBackendMut<'_, Self, $ring>,
                res_col: usize,
                a: &VecZnxDftBackendRef<'_, Self, $ring>,
                a_col: usize,
            ) {
                drivers::vec_znx_dft_sub_assign::<$ring, E>(res, res_col, a, a_col)
            }

            fn vec_znx_dft_sub_negate_assign<E: TaskExecutor>(
                res: &mut VecZnxDftBackendMut<'_, Self, $ring>,
                res_col: usize,
                a: &VecZnxDftBackendRef<'_, Self, $ring>,
                a_col: usize,
            ) {
                drivers::vec_znx_dft_sub_negate_assign::<$ring, E>(res, res_col, a, a_col)
            }

            fn vec_znx_dft_copy<E: TaskExecutor>(
                step: usize,
                offset: usize,
                res: &mut VecZnxDftBackendMut<'_, Self, $ring>,
                res_col: usize,
                a: &VecZnxDftBackendRef<'_, Self, $ring>,
                a_col: usize,
            ) {
                drivers::vec_znx_dft_copy::<$ring, E>(step, offset, res, res_col, a, a_col)
            }

            fn vec_znx_dft_automorphism_add<E: TaskExecutor>(
                plan: &<Self as HalVecZnxDftImpl<$ring>>::AutomorphismPlan,
                res: &mut VecZnxDftBackendMut<'_, Self, $ring>,
                res_col: usize,
                a: &VecZnxDftBackendRef<'_, Self, $ring>,
                a_col: usize,
            ) {
                drivers::vec_znx_dft_automorphism_add::<$ring, E>(plan, res, res_col, a, a_col)
            }

            fn cnv_pvec_prepare_tmp_bytes(n: usize) -> usize {
                drivers::cnv_pvec_prepare_tmp_bytes(n)
            }

            fn cnv_pvec_left_prepare<BE: PackedWord, E: TaskExecutor>(
                base: &Module<Self>,
                module: &Module<BE>,
                res: &mut CnvPVecLBackendMut<'_, BE, $ring>,
                a: &VecZnxBackendRef<'_, BE, $ring>,
                tmp: &mut [u64],
            ) where
                for<'a> BE::BufRef<'a>: HostDataRef,
                for<'a> BE::BufMut<'a>: HostDataMut,
            {
                drivers::cnv_pvec_left_prepare::<BE, E, $ring>(module, res, a, tmp, |n, dst, src, prepared| {
                    drivers::dft_limb_scaled::<$ring, E>(base, n, dst, src, prepared)
                })
            }

            fn cnv_pvec_right_prepare<BE: PackedWord, E: TaskExecutor>(
                base: &Module<Self>,
                module: &Module<BE>,
                res: &mut CnvPVecRBackendMut<'_, BE, $ring>,
                a: &VecZnxBackendRef<'_, BE, $ring>,
                tmp: &mut [u64],
            ) where
                for<'a> BE::BufRef<'a>: HostDataRef,
                for<'a> BE::BufMut<'a>: HostDataMut,
            {
                drivers::cnv_pvec_right_prepare::<BE, E, $ring>(module, res, a, tmp, |n, dst, src, prepared| {
                    drivers::dft_limb_scaled::<$ring, E>(base, n, dst, src, prepared)
                })
            }

            fn cnv_pvec_self_prepare<BE: PackedWord, E: TaskExecutor>(
                base: &Module<Self>,
                module: &Module<BE>,
                left: &mut CnvPVecLBackendMut<'_, BE, $ring>,
                right: &mut CnvPVecRBackendMut<'_, BE, $ring>,
                a: &VecZnxBackendRef<'_, BE, $ring>,
                tmp: &mut [u64],
            ) where
                for<'a> BE::BufRef<'a>: HostDataRef,
                for<'a> BE::BufMut<'a>: HostDataMut,
            {
                drivers::cnv_pvec_self_prepare::<BE, E, $ring>(module, left, right, a, tmp, |n, dst, src, prepared| {
                    drivers::dft_limb_scaled::<$ring, E>(base, n, dst, src, prepared)
                })
            }

            fn cnv_apply_tmp_words(res_size: usize) -> usize {
                drivers::cnv_apply_tmp_words(res_size)
            }

            fn cnv_apply_dft<BE: PackedWord, E: TaskExecutor>(
                _base: &Module<Self>,
                module: &Module<BE>,
                cnv_offset: usize,
                res: &mut VecZnxDftBackendMut<'_, BE, $ring>,
                res_col: usize,
                a: &CnvPVecLBackendRef<'_, BE, $ring>,
                a_col: usize,
                b: &CnvPVecRBackendRef<'_, BE, $ring>,
                b_col: usize,
                b_galois: i64,
                tmp: &mut [u32],
            ) where
                for<'a> BE::BufRef<'a>: HostDataRef,
                for<'a> BE::BufMut<'a>: HostDataMut,
            {
                drivers::cnv_apply_dft::<BE, E, $ring>(module, cnv_offset, res, res_col, a, a_col, b, b_col, b_galois, tmp)
            }

            fn cnv_apply_dft_add<BE: PackedWord, E: TaskExecutor>(
                _base: &Module<Self>,
                module: &Module<BE>,
                cnv_offset: usize,
                res: &mut VecZnxDftBackendMut<'_, BE, $ring>,
                res_col: usize,
                a: &CnvPVecLBackendRef<'_, BE, $ring>,
                a_col: usize,
                b: &CnvPVecRBackendRef<'_, BE, $ring>,
                b_col: usize,
                b_galois: i64,
                tmp: &mut [u32],
            ) where
                for<'a> BE::BufRef<'a>: HostDataRef,
                for<'a> BE::BufMut<'a>: HostDataMut,
            {
                drivers::cnv_apply_dft_add::<BE, E, $ring>(module, cnv_offset, res, res_col, a, a_col, b, b_col, b_galois, tmp)
            }

            fn cnv_apply_dft_sum<BE: PackedWord, E: TaskExecutor>(
                _base: &Module<Self>,
                module: &Module<BE>,
                cnv_offset: usize,
                res: &mut VecZnxDftBackendMut<'_, BE, $ring>,
                res_col: usize,
                terms: &[CnvDftAccTerm<'_, BE, $ring>],
                tmp: &mut [u32],
            ) where
                for<'a> BE::BufRef<'a>: HostDataRef,
                for<'a> BE::BufMut<'a>: HostDataMut,
            {
                drivers::cnv_apply_dft_sum::<BE, E, $ring>(module, cnv_offset, res, res_col, terms, tmp)
            }

            fn cnv_pairwise_apply_dft<BE: PackedWord, E: TaskExecutor>(
                _base: &Module<Self>,
                module: &Module<BE>,
                cnv_offset: usize,
                res: &mut VecZnxDftBackendMut<'_, BE, $ring>,
                res_col: usize,
                a: &CnvPVecLBackendRef<'_, BE, $ring>,
                b: &CnvPVecRBackendRef<'_, BE, $ring>,
                i: usize,
                j: usize,
                tmp: &mut [u32],
            ) where
                for<'a> BE::BufRef<'a>: HostDataRef,
                for<'a> BE::BufMut<'a>: HostDataMut,
            {
                drivers::cnv_pairwise_apply_dft::<BE, E, $ring>(module, cnv_offset, res, res_col, a, b, i, j, tmp)
            }
        }
    };
}

impl_packed_base!(Standard);
impl_packed_base!(ConjugateInvariant);

mod backend {
    use poulpy_cpu_portable::NTT4x30Portable;
    use poulpy_cpu_portable::kernels::znx::ZnxAutomorphismRotate;
    use poulpy_hal::layouts::Standard;

    use super::NTT4x30PortableRayon;

    crate::impl_ntt4x30_rayon_backend!(NTT4x30PortableRayon, NTT4x30Portable);

    unsafe impl poulpy_hal::oep::HalVecZnxMonomialImpl for NTT4x30PortableRayon {
        poulpy_cpu_portable::hal_impl_vec_znx_monomial!();
    }

    unsafe impl poulpy_hal::oep::HalVecZnxCIImpl for NTT4x30PortableRayon {
        poulpy_cpu_portable::hal_impl_vec_znx_ci!();
    }

    unsafe impl poulpy_hal::oep::HalScalarZnxCIImpl for NTT4x30PortableRayon {}

    impl ZnxAutomorphismRotate<Standard> for NTT4x30PortableRayon {
        #[inline(always)]
        fn znx_automorphism_rotate(p: i64, k: i64, res: &mut [i64], a: &[i64]) {
            <NTT4x30Portable as ZnxAutomorphismRotate<Standard>>::znx_automorphism_rotate(p, k, res, a)
        }
    }
}

// Starting values, those of the NEON NTT4x30 wrapper.
impl poulpy_hal::execution::ScratchWorkers for NTT4x30PortableRayon {
    const PREPARE: usize = 32;
    const APPLY: usize = 32;
    const VMP: usize = 32;
    const IDFT: usize = 32;
}

impl crate::RayonTuning for NTT4x30PortableRayon {
    const COEFF_MIN_LEN: usize = 1 << 17;
    const COEFF_MIN_TASK: usize = 1 << 13;
    const NORMALIZE_MIN_TASK: usize = 1 << 12;
}
