//! Rayon-scheduled wrapper for the AVX2 NTT4x30 backend.

use std::mem::size_of;

use poulpy_cpu_rayon::RayonTaskExecutor;
use poulpy_hal::layouts::{
    ConjugateInvariant, DataView, DataViewMut, Module, Ring, ScratchArena, Standard, VecZnxDft, VecZnxDftBackendMut,
    VecZnxDftBackendRef, VmpPMat, VmpPMatBackendRef,
};

use super::{NTT4x30Avx, NTT4x30AvxRayon, convolution, vec_znx_dft, vmp};

poulpy_cpu_rayon::impl_packed_ntt4x30_base!(NTT4x30Avx, Standard, drivers);
poulpy_cpu_rayon::impl_packed_ntt4x30_base!(NTT4x30Avx, ConjugateInvariant, drivers);

/// The drivers bound by [`poulpy_cpu_rayon::impl_packed_ntt4x30_base!`], under the names and arguments of `PackedNtt4x30Base`.
#[allow(clippy::too_many_arguments)]
mod drivers {
    use poulpy_cpu_portable::kernels::ntt4x30::{
        NttDFTExecute,
        ntt::{NttTable, NttTableInv},
        primes::Primes30,
        vec_znx_dft::NttModuleHandle,
    };
    use poulpy_cpu_rayon::ntt4x30::PackedWord;
    use poulpy_hal::{
        execution::TaskExecutor,
        layouts::{
            CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecRBackendMut, CnvPVecRBackendRef, HostDataMut, HostDataRef, Module,
            Ring, VecZnxBackendRef, VecZnxDftBackendMut,
        },
    };

    use super::{NTT4x30Avx, convolution, vec_znx_dft};

    pub(super) use super::convolution::cnv_prepare_tmp_bytes;
    pub(super) use super::vec_znx_dft::{
        vec_znx_dft_add, vec_znx_dft_add_assign, vec_znx_dft_automorphism_add, vec_znx_dft_copy, vec_znx_dft_sub,
        vec_znx_dft_sub_assign, vec_znx_dft_sub_negate_assign,
    };
    pub(super) use super::vmp::{
        vmp_apply_dft_to_dft_add_avx as vmp_apply_dft_to_dft_add, vmp_apply_dft_to_dft_avx as vmp_apply_dft_to_dft,
        vmp_apply_tmp_bytes_avx as vmp_apply_tmp_bytes,
    };

    /// A limb takes four `u64` per coefficient while it is transformed.
    pub(super) fn idft_tmp_words(n: usize) -> usize {
        4 * n
    }

    /// The convolution applies take no staging scratch.
    pub(super) fn cnv_apply_tmp_words(_res_size: usize) -> usize {
        0
    }

    // `E` is part of the binding contract: the interleaved limb is transformed as one task.
    #[allow(clippy::extra_unused_type_parameters)]
    pub(super) fn dft_limb<R: Ring, E: TaskExecutor>(
        module: &Module<NTT4x30Avx<R>>,
        n: usize,
        dst: &mut [u32],
        src: Option<&[i64]>,
    ) where
        NTT4x30Avx<R>: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>>,
    {
        vec_znx_dft::dft_limb(module, n, dst, src)
    }

    // `E` is part of the binding contract: the interleaved limb is transformed as one task.
    #[allow(clippy::extra_unused_type_parameters)]
    pub(super) fn idft_limb<R: Ring, E: TaskExecutor>(
        module: &Module<NTT4x30Avx<R>>,
        n: usize,
        dst: &mut [i128],
        src: &[u32],
        tmp: &mut [u64],
    ) where
        NTT4x30Avx<R>: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>>,
    {
        vec_znx_dft::idft_limb(module, n, dst, src, tmp)
    }

    // `E` is part of the binding contract: the interleaved limb is transformed as one task.
    #[allow(clippy::extra_unused_type_parameters)]
    pub(super) fn idft_limb_compact<R: Ring, E: TaskExecutor>(
        module: &Module<NTT4x30Avx<R>>,
        n: usize,
        slot: &mut [u32],
        tmp: &mut [u64],
    ) where
        NTT4x30Avx<R>: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>>,
    {
        vec_znx_dft::idft_compact_limb(module, n, slot, tmp)
    }

    pub(super) fn cnv_prepare_left<R: Ring, BE: PackedWord, E: TaskExecutor>(
        base: &Module<NTT4x30Avx<R>>,
        module: &Module<BE>,
        res: &mut CnvPVecLBackendMut<'_, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        tmp: &mut [u64],
    ) where
        NTT4x30Avx<R>: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>>,
        for<'a> BE::BufRef<'a>: HostDataRef,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        convolution::cnv_prepare_left::<BE, E>(module, res, a, tmp, |n, dst, src| {
            vec_znx_dft::dft_limb_wide(base, n, dst, src)
        })
    }

    pub(super) fn cnv_prepare_right<R: Ring, BE: PackedWord, E: TaskExecutor>(
        base: &Module<NTT4x30Avx<R>>,
        module: &Module<BE>,
        res: &mut CnvPVecRBackendMut<'_, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        tmp: &mut [u64],
    ) where
        NTT4x30Avx<R>: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>>,
        for<'a> BE::BufRef<'a>: HostDataRef,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        convolution::cnv_prepare_right::<BE, E>(module, res, a, tmp, |n, dst, src| {
            vec_znx_dft::dft_limb_wide(base, n, dst, src)
        })
    }

    pub(super) fn cnv_prepare_self<R: Ring, BE: PackedWord, E: TaskExecutor>(
        base: &Module<NTT4x30Avx<R>>,
        module: &Module<BE>,
        left: &mut CnvPVecLBackendMut<'_, BE>,
        right: &mut CnvPVecRBackendMut<'_, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        tmp: &mut [u64],
    ) where
        NTT4x30Avx<R>: NttDFTExecute<NttTable<Primes30, R>> + NttDFTExecute<NttTableInv<Primes30, R>>,
        for<'a> BE::BufRef<'a>: HostDataRef,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        convolution::cnv_prepare_self::<BE, E>(module, left, right, a, tmp, |n, dst, src| {
            vec_znx_dft::dft_limb_wide(base, n, dst, src)
        })
    }

    pub(super) fn cnv_apply_dft<R: Ring, BE: PackedWord, E: TaskExecutor>(
        base: &Module<NTT4x30Avx<R>>,
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, BE>,
        a_col: usize,
        b: &CnvPVecRBackendRef<'_, BE>,
        b_col: usize,
        b_galois: i64,
        _tmp: &mut [u32],
    ) where
        for<'a> BE::BufRef<'a>: HostDataRef,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        unsafe {
            convolution::cnv_apply_dft::<BE, E>(
                module,
                base.get_bbc_meta(),
                cnv_offset,
                res,
                res_col,
                a,
                a_col,
                b,
                b_col,
                b_galois,
            )
        }
    }

    pub(super) fn cnv_apply_dft_add<R: Ring, BE: PackedWord, E: TaskExecutor>(
        base: &Module<NTT4x30Avx<R>>,
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, BE>,
        a_col: usize,
        b: &CnvPVecRBackendRef<'_, BE>,
        b_col: usize,
        b_galois: i64,
        _tmp: &mut [u32],
    ) where
        for<'a> BE::BufRef<'a>: HostDataRef,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        unsafe {
            convolution::cnv_apply_dft_add::<BE, E>(
                module,
                base.get_bbc_meta(),
                cnv_offset,
                res,
                res_col,
                a,
                a_col,
                b,
                b_col,
                b_galois,
            )
        }
    }

    pub(super) fn cnv_pairwise_apply_dft<R: Ring, BE: PackedWord, E: TaskExecutor>(
        base: &Module<NTT4x30Avx<R>>,
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, BE>,
        b: &CnvPVecRBackendRef<'_, BE>,
        i: usize,
        j: usize,
        _tmp: &mut [u32],
    ) where
        for<'a> BE::BufRef<'a>: HostDataRef,
        for<'a> BE::BufMut<'a>: HostDataMut,
    {
        unsafe { convolution::cnv_pairwise_apply_dft::<BE, E>(module, base.get_bbc_meta(), cnv_offset, res, res_col, a, b, i, j) }
    }
}

mod standard {
    use poulpy_cpu_portable::kernels::znx::ZnxAutomorphismRotate;

    use super::{NTT4x30Avx, NTT4x30AvxRayon};

    poulpy_cpu_rayon::impl_ntt4x30_rayon_backend!(NTT4x30AvxRayon, NTT4x30Avx);

    unsafe impl poulpy_hal::oep::HalVecZnxMonomialImpl for NTT4x30AvxRayon {
        poulpy_cpu_portable::hal_impl_vec_znx_monomial!();
    }

    unsafe impl poulpy_hal::oep::HalVecZnxCIImpl for NTT4x30AvxRayon {
        poulpy_cpu_portable::hal_impl_vec_znx_ci!();
    }

    impl ZnxAutomorphismRotate for NTT4x30AvxRayon {
        #[inline(always)]
        fn znx_automorphism_rotate(p: i64, k: i64, res: &mut [i64], a: &[i64]) {
            <NTT4x30Avx as ZnxAutomorphismRotate>::znx_automorphism_rotate(p, k, res, a)
        }
    }
}

mod conjugate_invariant {
    use poulpy_hal::layouts::ConjugateInvariant;

    use super::{NTT4x30Avx, NTT4x30AvxRayon};

    poulpy_cpu_rayon::impl_ntt4x30_rayon_backend!(NTT4x30AvxRayon<ConjugateInvariant>, NTT4x30Avx<ConjugateInvariant>);
}

/// Interleaved-digit product of the Rayon backend over one ring, on per-worker scratch.
macro_rules! impl_digits_strided {
    ($ring:ty) => {
        unsafe impl poulpy_core::oep::GGLWEProductDigitsStridedImpl for NTT4x30AvxRayon<$ring> {
            fn gglwe_product_digits_strided_tmp_bytes(
                _module: &Module<Self>,
                _res_size: usize,
                a_cols: usize,
                a_size: usize,
                dsize: usize,
                pmat_rows: usize,
                pmat_cols_in: usize,
                _pmat_cols_out: usize,
                _pmat_size: usize,
            ) -> usize {
                vmp::vmp_apply_digits_strided_tmp_bytes_avx(
                    a_cols,
                    a_size,
                    dsize,
                    pmat_rows,
                    pmat_cols_in,
                    poulpy_cpu_rayon::workers(<Self as poulpy_hal::execution::ScratchWorkers>::VMP),
                )
            }

            fn gglwe_product_digits_strided(
                module: &Module<Self>,
                res: &mut VecZnxDftBackendMut<'_, Self>,
                a: &VecZnxDftBackendRef<'_, Self>,
                dsize: usize,
                product_limbs: usize,
                pmat: &VmpPMatBackendRef<'_, Self>,
                scratch: &mut ScratchArena<'_, Self>,
            ) {
                let metadata_bytes = 4 * dsize * size_of::<u64>();
                let per_worker =
                    vmp::vmp_apply_digits_strided_tmp_bytes_avx(a.cols(), a.size(), dsize, pmat.rows(), pmat.cols_in(), 1)
                        - metadata_bytes;
                let workers = poulpy_cpu_rayon::workers_within(
                    <Self as poulpy_hal::execution::ScratchWorkers>::VMP,
                    per_worker,
                    scratch.available().saturating_sub(metadata_bytes),
                );
                let bytes = metadata_bytes + workers * per_worker;
                let (tmp, _) = crate::hal_impl::take_host_typed::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
                let res_shape = res.shape();
                vmp::vmp_apply_dft_to_dft_digits_strided_avx::<$ring, RayonTaskExecutor>(
                    module.reinterpret(),
                    &mut VecZnxDft::from_shape(&mut **res.data_mut(), res_shape),
                    &VecZnxDft::from_shape(&**a.data(), a.shape()),
                    dsize,
                    product_limbs,
                    &VmpPMat::from_data(
                        &**pmat.data(),
                        pmat.n(),
                        pmat.rows(),
                        pmat.cols_in(),
                        pmat.cols_out(),
                        pmat.size(),
                        pmat.hint(),
                    ),
                    tmp,
                );
            }

            ::poulpy_core::impl_gglwe_product_step_hooks_reference!();
        }
    };
}

impl_digits_strided!(Standard);
impl_digits_strided!(ConjugateInvariant);

impl<R: Ring> poulpy_hal::execution::ScratchWorkers for NTT4x30AvxRayon<R> {
    const PREPARE: usize = 4;
    const APPLY: usize = 8;
    const VMP: usize = 8;
    const IDFT: usize = 8;
}

impl<R: Ring> poulpy_cpu_rayon::RayonTuning for NTT4x30AvxRayon<R> {
    const COEFF_MIN_LEN: usize = 1 << 15;
    const COEFF_MIN_TASK: usize = 1 << 13;
    const NORMALIZE_MIN_TASK: usize = 1 << 12;
}

#[cfg(test)]
mod tests {
    use poulpy_hal::{layouts::Module, test_suite::convolution::test_convolution_by_const};

    use super::NTT4x30AvxRayon;

    #[test]
    fn convolution_by_const() {
        test_convolution_by_const(&Module::<NTT4x30AvxRayon>::new(1 << 8), 1 << 8, 50);
    }
}
