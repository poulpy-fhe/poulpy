//! Rayon-scheduled wrapper shared by the NTT4x30 backends that store the transform domain as packed `u32` residues.
//!
//! A serial backend describes its drivers through [`PackedNtt4x30Base`], and [`impl_ntt4x30_rayon_backend!`]
//! builds the Rayon variant on top of it: the coefficient kernels are forwarded or split by ranges, and the
//! transform-domain operations run the base drivers on [`RayonTaskExecutor`](crate::RayonTaskExecutor).

use poulpy_cpu_portable::kernels::ntt4x30::primes::Primes30;
use poulpy_hal::{
    execution::TaskExecutor,
    layouts::{
        Backend, CnvDftAccTerm, CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecRBackendMut, CnvPVecRBackendRef, CrtWord,
        HostDataMut, HostDataRef, Module, VecZnxBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef, VecZnxInfos,
        VmpPMatBackendRef, ZnxViewMut,
    },
    oep::HalVecZnxDftImpl,
};

/// Panics unless `a` has degree `n` and a column `col`, as the base kernels check before they slice a limb.
///
/// The parallel paths of [`impl_ntt4x30_rayon_backend!`] slice limbs themselves, so they check first.
#[doc(hidden)]
#[inline]
pub fn check_packed_operand(n: usize, a: &impl VecZnxInfos, col: usize) {
    assert_eq!(a.n(), n, "operand degree differs from the result degree");
    assert!(col < a.cols(), "operand column out of range");
}

/// A backend whose transform domain a wrapper can retag as its own: the packed NTT4x30 word.
pub trait PackedWord: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64> {}
impl<BE: Backend<DftWord = CrtWord<Primes30, u32>, ZnxWord = i64>> PackedWord for BE {}

/// Drivers a serial packed NTT4x30 backend hands to its Rayon variant.
///
/// A limb of a transform-domain vector is `4 * n` consecutive `u32`, limb `l` of column `c` starting at `4 * n * (l * cols + c)`.
/// The operations that split their work take the executor as `E` and run serially on `SerialTaskExecutor`.
/// The limb transforms may split one limb further, into its planes.
/// The convolution drivers are generic over the backend `BE` that tags their operands, so that the wrapper passes its own layouts.
/// The inverse limb transforms receive `tmp` per task, zero words when the backend transforms in place.
#[allow(clippy::too_many_arguments)]
pub trait PackedNtt4x30Base: PackedWord + HalVecZnxDftImpl {
    /// `u64` words of scratch the inverse transform of one limb of degree `n` needs.
    fn idft_tmp_words(n: usize) -> usize;

    /// Forward transform of `src` into the limb `dst`, or zeros when `src` is `None`.
    ///
    /// `vec_znx_dft_apply` carries no scratch, so a backend that needs working memory keeps its own.
    fn dft_limb<E: TaskExecutor>(module: &Module<Self>, n: usize, dst: &mut [u32], src: Option<&[i64]>);

    /// Inverse transform of the limb `src` into `dst`, with [`Self::idft_tmp_words`] words in `tmp`.
    fn idft_limb<E: TaskExecutor>(module: &Module<Self>, n: usize, dst: &mut [i128], src: &[u32], tmp: &mut [u64]);

    /// Inverse transform of the limb `slot` into coefficients that take its place.
    fn idft_limb_compact<E: TaskExecutor>(module: &Module<Self>, n: usize, slot: &mut [u32], tmp: &mut [u64]);

    /// Scratch (in bytes) of the vector-matrix products, per worker.
    fn vmp_apply_tmp_bytes(a_size: usize, b_rows: usize, b_cols_in: usize) -> usize;

    fn vmp_apply_dft_to_dft<E: TaskExecutor>(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        pmat: &VmpPMatBackendRef<'_, Self>,
        limb_offset: usize,
        tmp: &mut [u64],
    );

    fn vmp_apply_dft_to_dft_add<E: TaskExecutor>(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        pmat: &VmpPMatBackendRef<'_, Self>,
        limb_offset: usize,
        tmp: &mut [u64],
    );

    fn vec_znx_dft_add<E: TaskExecutor>(
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, Self>,
        b_col: usize,
    );

    fn vec_znx_dft_add_assign<E: TaskExecutor>(
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    );

    fn vec_znx_dft_sub<E: TaskExecutor>(
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, Self>,
        b_col: usize,
    );

    fn vec_znx_dft_sub_assign<E: TaskExecutor>(
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    );

    fn vec_znx_dft_sub_negate_assign<E: TaskExecutor>(
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    );

    fn vec_znx_dft_copy<E: TaskExecutor>(
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    );

    fn vec_znx_dft_automorphism_add<E: TaskExecutor>(
        plan: &<Self as HalVecZnxDftImpl>::AutomorphismPlan,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    );

    /// Scratch (in bytes) of the convolution prepares, per worker.
    fn cnv_prepare_tmp_bytes(n: usize) -> usize;

    /// `base` is the module of this backend, whose tables transform the limbs of `a`.
    fn cnv_prepare_left<BE: PackedWord, E: TaskExecutor>(
        base: &Module<Self>,
        module: &Module<BE>,
        res: &mut CnvPVecLBackendMut<'_, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        tmp: &mut [u64],
    ) where
        for<'a> BE::BufRef<'a>: HostDataRef,
        for<'a> BE::BufMut<'a>: HostDataMut;

    fn cnv_prepare_right<BE: PackedWord, E: TaskExecutor>(
        base: &Module<Self>,
        module: &Module<BE>,
        res: &mut CnvPVecRBackendMut<'_, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        tmp: &mut [u64],
    ) where
        for<'a> BE::BufRef<'a>: HostDataRef,
        for<'a> BE::BufMut<'a>: HostDataMut;

    fn cnv_prepare_self<BE: PackedWord, E: TaskExecutor>(
        base: &Module<Self>,
        module: &Module<BE>,
        left: &mut CnvPVecLBackendMut<'_, BE>,
        right: &mut CnvPVecRBackendMut<'_, BE>,
        a: &VecZnxBackendRef<'_, BE>,
        tmp: &mut [u64],
    ) where
        for<'a> BE::BufRef<'a>: HostDataRef,
        for<'a> BE::BufMut<'a>: HostDataMut;

    /// `u32` words of scratch of the convolution applies for a result of `res_size` limbs, per worker.
    fn cnv_apply_tmp_words(res_size: usize) -> usize;

    fn cnv_apply_dft<BE: PackedWord, E: TaskExecutor>(
        base: &Module<Self>,
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, BE>,
        a_col: usize,
        b: &CnvPVecRBackendRef<'_, BE>,
        b_col: usize,
        b_galois: i64,
        tmp: &mut [u32],
    ) where
        for<'a> BE::BufRef<'a>: HostDataRef,
        for<'a> BE::BufMut<'a>: HostDataMut;

    fn cnv_apply_dft_add<BE: PackedWord, E: TaskExecutor>(
        base: &Module<Self>,
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &CnvPVecLBackendRef<'_, BE>,
        a_col: usize,
        b: &CnvPVecRBackendRef<'_, BE>,
        b_col: usize,
        b_galois: i64,
        tmp: &mut [u32],
    ) where
        for<'a> BE::BufRef<'a>: HostDataRef,
        for<'a> BE::BufMut<'a>: HostDataMut;

    /// `res[res_col] = sum of a_t (x) sigma_{b_galois_t}(b_t)` over `terms`.
    ///
    /// The default overwrites with the first term and adds the others, and zeroes the column when `terms` is empty.
    /// A backend with a fused kernel overrides it, reading every term's `b_galois` and rejecting an even one before
    /// its first write.
    fn cnv_apply_dft_sum<BE: PackedWord, E: TaskExecutor>(
        base: &Module<Self>,
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
        let Some((first, rest)) = terms.split_first() else {
            for limb in 0..res.size() {
                bytemuck::cast_slice_mut::<_, u32>(res.at_mut(res_col, limb)).fill(0);
            }
            return;
        };
        terms.iter().for_each(|term| poulpy_hal::layouts::check_galois(term.b_galois));
        Self::cnv_apply_dft::<BE, E>(
            base,
            module,
            cnv_offset,
            res,
            res_col,
            &first.a,
            first.a_col,
            &first.b,
            first.b_col,
            first.b_galois,
            tmp,
        );
        for term in rest {
            Self::cnv_apply_dft_add::<BE, E>(
                base,
                module,
                cnv_offset,
                res,
                res_col,
                &term.a,
                term.a_col,
                &term.b,
                term.b_col,
                term.b_galois,
                tmp,
            );
        }
    }

    fn cnv_pairwise_apply_dft<BE: PackedWord, E: TaskExecutor>(
        base: &Module<Self>,
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
        for<'a> BE::BufMut<'a>: HostDataMut;
}

#[doc(hidden)]
#[macro_export]
macro_rules! rayon_forward_i128_big {
    ($base:ty, $method:ident($($arg:ident: $atype:ty),* $(,)?)) => {
        fn $method($($arg: $atype),*) {
            <$base as I128BigOps>::$method($($arg),*)
        }
    };
}

/// Implements [`PackedNtt4x30Base`] for `$backend<$ring>` by forwarding every method to the driver module `$drivers`.
///
/// The driver module provides, for each method of the trait except `cnv_apply_dft_sum`, a function of the same name and arguments.
/// The functions that the trait makes generic take the ring first: `<R, E>`, or `<R, BE, E>` for the convolution drivers.
/// `cnv_apply_dft_sum` keeps the default body of the trait.
/// The backend owns the module, so its own adaptations stay in its crate.
#[doc(hidden)]
#[macro_export]
macro_rules! impl_packed_ntt4x30_base {
    ($backend:ident, $ring:ty, $drivers:ident) => {
        #[allow(clippy::too_many_arguments)]
        impl $crate::ntt4x30::PackedNtt4x30Base for $backend<$ring> {
            fn idft_tmp_words(n: usize) -> usize {
                $drivers::idft_tmp_words(n)
            }

            fn dft_limb<E: $crate::__private::poulpy_hal::execution::TaskExecutor>(
                module: &$crate::__private::poulpy_hal::layouts::Module<Self>,
                n: usize,
                dst: &mut [u32],
                src: Option<&[i64]>,
            ) {
                $drivers::dft_limb::<$ring, E>(module, n, dst, src)
            }

            fn idft_limb<E: $crate::__private::poulpy_hal::execution::TaskExecutor>(
                module: &$crate::__private::poulpy_hal::layouts::Module<Self>,
                n: usize,
                dst: &mut [i128],
                src: &[u32],
                tmp: &mut [u64],
            ) {
                $drivers::idft_limb::<$ring, E>(module, n, dst, src, tmp)
            }

            fn idft_limb_compact<E: $crate::__private::poulpy_hal::execution::TaskExecutor>(
                module: &$crate::__private::poulpy_hal::layouts::Module<Self>,
                n: usize,
                slot: &mut [u32],
                tmp: &mut [u64],
            ) {
                $drivers::idft_limb_compact::<$ring, E>(module, n, slot, tmp)
            }

            fn vmp_apply_tmp_bytes(a_size: usize, b_rows: usize, b_cols_in: usize) -> usize {
                $drivers::vmp_apply_tmp_bytes(a_size, b_rows, b_cols_in)
            }

            fn vmp_apply_dft_to_dft<E: $crate::__private::poulpy_hal::execution::TaskExecutor>(
                module: &$crate::__private::poulpy_hal::layouts::Module<Self>,
                res: &mut $crate::__private::poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
                a: &$crate::__private::poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
                pmat: &$crate::__private::poulpy_hal::layouts::VmpPMatBackendRef<'_, Self>,
                limb_offset: usize,
                tmp: &mut [u64],
            ) {
                $drivers::vmp_apply_dft_to_dft::<$ring, E>(module, res, a, pmat, limb_offset, tmp)
            }

            fn vmp_apply_dft_to_dft_add<E: $crate::__private::poulpy_hal::execution::TaskExecutor>(
                module: &$crate::__private::poulpy_hal::layouts::Module<Self>,
                res: &mut $crate::__private::poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
                a: &$crate::__private::poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
                pmat: &$crate::__private::poulpy_hal::layouts::VmpPMatBackendRef<'_, Self>,
                limb_offset: usize,
                tmp: &mut [u64],
            ) {
                $drivers::vmp_apply_dft_to_dft_add::<$ring, E>(module, res, a, pmat, limb_offset, tmp)
            }

            fn vec_znx_dft_add<E: $crate::__private::poulpy_hal::execution::TaskExecutor>(
                res: &mut $crate::__private::poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
                res_col: usize,
                a: &$crate::__private::poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
                a_col: usize,
                b: &$crate::__private::poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
                b_col: usize,
            ) {
                $drivers::vec_znx_dft_add::<$ring, E>(res, res_col, a, a_col, b, b_col)
            }

            fn vec_znx_dft_add_assign<E: $crate::__private::poulpy_hal::execution::TaskExecutor>(
                res: &mut $crate::__private::poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
                res_col: usize,
                a: &$crate::__private::poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
                a_col: usize,
            ) {
                $drivers::vec_znx_dft_add_assign::<$ring, E>(res, res_col, a, a_col)
            }

            fn vec_znx_dft_sub<E: $crate::__private::poulpy_hal::execution::TaskExecutor>(
                res: &mut $crate::__private::poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
                res_col: usize,
                a: &$crate::__private::poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
                a_col: usize,
                b: &$crate::__private::poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
                b_col: usize,
            ) {
                $drivers::vec_znx_dft_sub::<$ring, E>(res, res_col, a, a_col, b, b_col)
            }

            fn vec_znx_dft_sub_assign<E: $crate::__private::poulpy_hal::execution::TaskExecutor>(
                res: &mut $crate::__private::poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
                res_col: usize,
                a: &$crate::__private::poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
                a_col: usize,
            ) {
                $drivers::vec_znx_dft_sub_assign::<$ring, E>(res, res_col, a, a_col)
            }

            fn vec_znx_dft_sub_negate_assign<E: $crate::__private::poulpy_hal::execution::TaskExecutor>(
                res: &mut $crate::__private::poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
                res_col: usize,
                a: &$crate::__private::poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
                a_col: usize,
            ) {
                $drivers::vec_znx_dft_sub_negate_assign::<$ring, E>(res, res_col, a, a_col)
            }

            fn vec_znx_dft_copy<E: $crate::__private::poulpy_hal::execution::TaskExecutor>(
                step: usize,
                offset: usize,
                res: &mut $crate::__private::poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
                res_col: usize,
                a: &$crate::__private::poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
                a_col: usize,
            ) {
                $drivers::vec_znx_dft_copy::<$ring, E>(step, offset, res, res_col, a, a_col)
            }

            fn vec_znx_dft_automorphism_add<E: $crate::__private::poulpy_hal::execution::TaskExecutor>(
                plan: &<Self as $crate::__private::poulpy_hal::oep::HalVecZnxDftImpl>::AutomorphismPlan,
                res: &mut $crate::__private::poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
                res_col: usize,
                a: &$crate::__private::poulpy_hal::layouts::VecZnxDftBackendRef<'_, Self>,
                a_col: usize,
            ) {
                $drivers::vec_znx_dft_automorphism_add::<$ring, E>(plan, res, res_col, a, a_col)
            }

            fn cnv_prepare_tmp_bytes(n: usize) -> usize {
                $drivers::cnv_prepare_tmp_bytes(n)
            }

            fn cnv_prepare_left<BE: $crate::ntt4x30::PackedWord, E: $crate::__private::poulpy_hal::execution::TaskExecutor>(
                base: &$crate::__private::poulpy_hal::layouts::Module<Self>,
                module: &$crate::__private::poulpy_hal::layouts::Module<BE>,
                res: &mut $crate::__private::poulpy_hal::layouts::CnvPVecLBackendMut<'_, BE>,
                a: &$crate::__private::poulpy_hal::layouts::VecZnxBackendRef<'_, BE>,
                tmp: &mut [u64],
            ) where
                for<'a> BE::BufRef<'a>: $crate::__private::poulpy_hal::layouts::HostDataRef,
                for<'a> BE::BufMut<'a>: $crate::__private::poulpy_hal::layouts::HostDataMut,
            {
                $drivers::cnv_prepare_left::<$ring, BE, E>(base, module, res, a, tmp)
            }

            fn cnv_prepare_right<BE: $crate::ntt4x30::PackedWord, E: $crate::__private::poulpy_hal::execution::TaskExecutor>(
                base: &$crate::__private::poulpy_hal::layouts::Module<Self>,
                module: &$crate::__private::poulpy_hal::layouts::Module<BE>,
                res: &mut $crate::__private::poulpy_hal::layouts::CnvPVecRBackendMut<'_, BE>,
                a: &$crate::__private::poulpy_hal::layouts::VecZnxBackendRef<'_, BE>,
                tmp: &mut [u64],
            ) where
                for<'a> BE::BufRef<'a>: $crate::__private::poulpy_hal::layouts::HostDataRef,
                for<'a> BE::BufMut<'a>: $crate::__private::poulpy_hal::layouts::HostDataMut,
            {
                $drivers::cnv_prepare_right::<$ring, BE, E>(base, module, res, a, tmp)
            }

            fn cnv_prepare_self<BE: $crate::ntt4x30::PackedWord, E: $crate::__private::poulpy_hal::execution::TaskExecutor>(
                base: &$crate::__private::poulpy_hal::layouts::Module<Self>,
                module: &$crate::__private::poulpy_hal::layouts::Module<BE>,
                left: &mut $crate::__private::poulpy_hal::layouts::CnvPVecLBackendMut<'_, BE>,
                right: &mut $crate::__private::poulpy_hal::layouts::CnvPVecRBackendMut<'_, BE>,
                a: &$crate::__private::poulpy_hal::layouts::VecZnxBackendRef<'_, BE>,
                tmp: &mut [u64],
            ) where
                for<'a> BE::BufRef<'a>: $crate::__private::poulpy_hal::layouts::HostDataRef,
                for<'a> BE::BufMut<'a>: $crate::__private::poulpy_hal::layouts::HostDataMut,
            {
                $drivers::cnv_prepare_self::<$ring, BE, E>(base, module, left, right, a, tmp)
            }

            fn cnv_apply_tmp_words(res_size: usize) -> usize {
                $drivers::cnv_apply_tmp_words(res_size)
            }

            fn cnv_apply_dft<BE: $crate::ntt4x30::PackedWord, E: $crate::__private::poulpy_hal::execution::TaskExecutor>(
                base: &$crate::__private::poulpy_hal::layouts::Module<Self>,
                module: &$crate::__private::poulpy_hal::layouts::Module<BE>,
                cnv_offset: usize,
                res: &mut $crate::__private::poulpy_hal::layouts::VecZnxDftBackendMut<'_, BE>,
                res_col: usize,
                a: &$crate::__private::poulpy_hal::layouts::CnvPVecLBackendRef<'_, BE>,
                a_col: usize,
                b: &$crate::__private::poulpy_hal::layouts::CnvPVecRBackendRef<'_, BE>,
                b_col: usize,
                b_galois: i64,
                tmp: &mut [u32],
            ) where
                for<'a> BE::BufRef<'a>: $crate::__private::poulpy_hal::layouts::HostDataRef,
                for<'a> BE::BufMut<'a>: $crate::__private::poulpy_hal::layouts::HostDataMut,
            {
                $drivers::cnv_apply_dft::<$ring, BE, E>(
                    base, module, cnv_offset, res, res_col, a, a_col, b, b_col, b_galois, tmp,
                )
            }

            fn cnv_apply_dft_add<BE: $crate::ntt4x30::PackedWord, E: $crate::__private::poulpy_hal::execution::TaskExecutor>(
                base: &$crate::__private::poulpy_hal::layouts::Module<Self>,
                module: &$crate::__private::poulpy_hal::layouts::Module<BE>,
                cnv_offset: usize,
                res: &mut $crate::__private::poulpy_hal::layouts::VecZnxDftBackendMut<'_, BE>,
                res_col: usize,
                a: &$crate::__private::poulpy_hal::layouts::CnvPVecLBackendRef<'_, BE>,
                a_col: usize,
                b: &$crate::__private::poulpy_hal::layouts::CnvPVecRBackendRef<'_, BE>,
                b_col: usize,
                b_galois: i64,
                tmp: &mut [u32],
            ) where
                for<'a> BE::BufRef<'a>: $crate::__private::poulpy_hal::layouts::HostDataRef,
                for<'a> BE::BufMut<'a>: $crate::__private::poulpy_hal::layouts::HostDataMut,
            {
                $drivers::cnv_apply_dft_add::<$ring, BE, E>(
                    base, module, cnv_offset, res, res_col, a, a_col, b, b_col, b_galois, tmp,
                )
            }

            fn cnv_pairwise_apply_dft<
                BE: $crate::ntt4x30::PackedWord,
                E: $crate::__private::poulpy_hal::execution::TaskExecutor,
            >(
                base: &$crate::__private::poulpy_hal::layouts::Module<Self>,
                module: &$crate::__private::poulpy_hal::layouts::Module<BE>,
                cnv_offset: usize,
                res: &mut $crate::__private::poulpy_hal::layouts::VecZnxDftBackendMut<'_, BE>,
                res_col: usize,
                a: &$crate::__private::poulpy_hal::layouts::CnvPVecLBackendRef<'_, BE>,
                b: &$crate::__private::poulpy_hal::layouts::CnvPVecRBackendRef<'_, BE>,
                i: usize,
                j: usize,
                tmp: &mut [u32],
            ) where
                for<'a> BE::BufRef<'a>: $crate::__private::poulpy_hal::layouts::HostDataRef,
                for<'a> BE::BufMut<'a>: $crate::__private::poulpy_hal::layouts::HostDataMut,
            {
                $drivers::cnv_pairwise_apply_dft::<$ring, BE, E>(base, module, cnv_offset, res, res_col, a, b, i, j, tmp)
            }
        }
    };
}

/// Implements the Rayon-scheduled NTT4x30 backend `$rayon` on top of the serial backend `$base`, which implements [`PackedNtt4x30Base`].
///
/// The caller provides what differs per backend and per ring: `ScratchWorkers`, `RayonTuning`, and on the standard ring `HalVecZnxMonomialImpl`, `HalVecZnxCIImpl` and `ZnxAutomorphismRotate`.
///
/// Beyond the trait, the macro forwards to the HAL implementations of `$base` (vector, big, transform, scalar and vector-matrix products) and to its coefficient kernels, so `$base` must implement them.
/// It assumes that the scalar products of `$base` take no scratch.
/// The items land in a module named `ntt4x30_rayon_backend`, so two invocations must sit in separate modules.
#[macro_export]
macro_rules! impl_ntt4x30_rayon_backend {
    ($rayon:ty, $base:ty) => {
        mod ntt4x30_rayon_backend {
            #[allow(unused_imports)]
            use super::*;

use std::mem::size_of;

use $crate::__private::bytemuck::{cast_slice, cast_slice_mut};
use $crate::__private::rayon::prelude::*;

use $crate::__private::poulpy_cpu_portable::{
    hal_defaults::{BigWordHadamardProduct, HalVecZnxDefault, NTT4x30ModuleDefault, NTT4x30VecZnxBigDefault},
    kernels::{
        ntt4x30::{
            I128BigOps, NttDFTExecute,
            ntt::{NttTable, NttTableInv},
            primes::Primes30,
        },
        znx::{
            ZnxAdd, ZnxAddAssign, ZnxAutomorphism, ZnxCopy, ZnxMulAddPowerOfTwo, ZnxMulPowerOfTwo, ZnxMulPowerOfTwoAssign,
            ZnxNegate, ZnxNegateAssign, ZnxRotate, ZnxSub, ZnxSubAssign, ZnxSubNegateAssign, ZnxSwitchRing, ZnxZero,
        },
        znx_normalize::{Limbs, LimbsMut, ZnxNormalize},
    },
};
use $crate::__private::poulpy_hal::{
    execution::{SerialTaskExecutor, TaskExecutor},
    layouts::{
        Backend, DataView, DataViewMut, MatZnxBackendRef, Module, ScalarZnx, ScalarZnxBackendRef, ScratchArena, SvpPPol,
        SvpPPolBackendMut, SvpPPolBackendRef, VecZnx, VecZnxBackendMut, VecZnxBackendRef, VecZnxBig, VecZnxBigBackendMut,
        VecZnxDft, VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMat, VmpPMatBackendMut, VmpPMatBackendRef, ZnxView, ZnxViewMut,
    },
    oep::{HalConvolutionImpl, HalModuleImpl, HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl, HalVmpImpl},
};

use $crate::{RayonTaskExecutor, SendPtr, parallel_limb_tasks};

$crate::__private::poulpy_hal::impl_backend_from!($rayon, $base, $crate::RayonTaskExecutor);

impl $crate::__private::poulpy_hal::layouts::MaxBase2k for $rayon
where
    $base: $crate::__private::poulpy_hal::layouts::MaxBase2k,
{
    fn max_base2k(n: usize, products: usize, failure_bits: usize, squaring: bool) -> usize {
        <$base as $crate::__private::poulpy_hal::layouts::MaxBase2k>::max_base2k(n, products, failure_bits, squaring)
    }
}

fn base_module(module: &Module<$rayon>) -> &Module<$base> {
    module.reinterpret()
}

fn base_dft_ref<'a>(a: &'a VecZnxDftBackendRef<'_, $rayon>) -> VecZnxDftBackendRef<'a, $base> {
    VecZnxDft::from_shape(&**a.data(), a.shape())
}

fn base_dft_mut<'a>(a: &'a mut VecZnxDftBackendMut<'_, $rayon>) -> VecZnxDftBackendMut<'a, $base> {
    let shape = a.shape();
    VecZnxDft::from_shape(&mut **a.data_mut(), shape)
}

fn base_znx_ref<'a>(a: &'a VecZnxBackendRef<'_, $rayon>) -> VecZnxBackendRef<'a, $base> {
    VecZnx::from_shape(&**a.data(), a.shape())
}

fn base_scalar_ref<'a>(a: &'a ScalarZnxBackendRef<'_, $rayon>) -> ScalarZnxBackendRef<'a, $base> {
    ScalarZnx::from_data(&**a.data(), a.n(), a.cols())
}

fn base_svp_ref<'a>(a: &'a SvpPPolBackendRef<'_, $rayon>) -> SvpPPolBackendRef<'a, $base> {
    SvpPPol::from_data(&**a.data(), a.n(), a.cols(), a.hint())
}

fn base_svp_mut<'a>(a: &'a mut SvpPPolBackendMut<'_, $rayon>) -> SvpPPolBackendMut<'a, $base> {
    let (n, cols, hint) = (a.n(), a.cols(), a.hint());
    SvpPPol::from_data(&mut **a.data_mut(), n, cols, hint)
}

fn base_big_mut<'a>(a: &'a mut VecZnxBigBackendMut<'_, $rayon>) -> VecZnxBigBackendMut<'a, $base> {
    let shape = a.shape();
    VecZnxBig::from_shape(&mut **a.data_mut(), shape)
}

fn base_big_ref<'a>(
    a: &'a $crate::__private::poulpy_hal::layouts::VecZnxBigBackendRef<'_, $rayon>,
) -> $crate::__private::poulpy_hal::layouts::VecZnxBigBackendRef<'a, $base> {
    VecZnxBig::from_shape(&**a.data(), a.shape())
}

fn base_vmp_ref<'a>(a: &'a VmpPMatBackendRef<'_, $rayon>) -> VmpPMatBackendRef<'a, $base> {
    VmpPMat::from_data(&**a.data(), a.n(), a.rows(), a.cols_in(), a.cols_out(), a.size(), a.hint())
}

fn base_vmp_mut<'a>(a: &'a mut VmpPMatBackendMut<'_, $rayon>) -> VmpPMatBackendMut<'a, $base> {
    let (n, rows, cols_in, cols_out, size, hint) = (a.n(), a.rows(), a.cols_in(), a.cols_out(), a.size(), a.hint());
    VmpPMat::from_data(&mut **a.data_mut(), n, rows, cols_in, cols_out, size, hint)
}

/// Ring degree from which the limbs of a transform-domain vector are worth one task each.
const LIMB_TASKS_MIN_N: usize = 1 << 13;

#[inline]
fn limb_tasks(n: usize, size: usize) -> bool {
    n >= LIMB_TASKS_MIN_N && parallel_limb_tasks(size) && !RayonTaskExecutor::should_serialize_inner()
}

/// Scratch (in bytes) of the convolution applies: one stage per worker.
fn apply_tmp_bytes(res_size: usize) -> usize {
    $crate::workers(<$rayon as $crate::__private::poulpy_hal::execution::ScratchWorkers>::APPLY)
        * <$base as $crate::ntt4x30::PackedNtt4x30Base>::cnv_apply_tmp_words(res_size)
        * size_of::<u32>()
}

/// The stages of the convolution applies, for as many workers as the arena holds.
fn apply_scratch<'a>(res_size: usize, scratch: &'a mut ScratchArena<'_, $rayon>) -> &'a mut [u32] {
    let per_worker = <$base as $crate::ntt4x30::PackedNtt4x30Base>::cnv_apply_tmp_words(res_size);
    if per_worker == 0 {
        return &mut [];
    }
    let workers = $crate::workers_within(
        <$rayon as $crate::__private::poulpy_hal::execution::ScratchWorkers>::APPLY,
        per_worker * size_of::<u32>(),
        scratch.available(),
    );
    $crate::take_scratch::<$rayon, u32>(scratch.borrow(), workers * per_worker).0
}

$crate::rayon_parallel_binary!($rayon, $base, ZnxAdd, znx_add);
$crate::rayon_parallel_assign!($rayon, $base, ZnxAddAssign, znx_add_assign);
$crate::rayon_parallel_binary!($rayon, $base, ZnxSub, znx_sub);
$crate::rayon_parallel_assign!($rayon, $base, ZnxSubAssign, znx_sub_assign);
$crate::rayon_parallel_assign!($rayon, $base, ZnxSubNegateAssign, znx_sub_negate_assign);
$crate::rayon_parallel_shift!($rayon, $base, ZnxMulAddPowerOfTwo, znx_muladd_power_of_two);
$crate::rayon_parallel_shift!($rayon, $base, ZnxMulPowerOfTwo, znx_mul_power_of_two);
impl ZnxMulPowerOfTwoAssign for $rayon {
    #[inline(always)]
    fn znx_mul_power_of_two_assign(k: i64, res: &mut [i64]) {
        let Some(chunk) = $crate::parallel_chunk_len::<$rayon>(res.len()) else {
            return <$base as ZnxMulPowerOfTwoAssign>::znx_mul_power_of_two_assign(k, res);
        };
        res.par_chunks_mut(chunk)
            .for_each(|res| <$base as ZnxMulPowerOfTwoAssign>::znx_mul_power_of_two_assign(k, res));
    }
}
impl ZnxAutomorphism for $rayon {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        <$base as ZnxAutomorphism>::znx_automorphism(p, res, a)
    }

    #[inline(always)]
    fn znx_automorphism_i128(p: i64, res: &mut [i128], a: &[i128]) {
        <$base as ZnxAutomorphism>::znx_automorphism_i128(p, res, a)
    }
}
$crate::rayon_parallel_assign!($rayon, $base, ZnxCopy, znx_copy);
$crate::rayon_parallel_assign!($rayon, $base, ZnxNegate, znx_negate);
$crate::rayon_parallel_unary!($rayon, $base, ZnxNegateAssign, znx_negate_assign);
$crate::rayon_forward_znx!($rayon, $base, ZnxRotate, znx_rotate(p: i64, res: &mut [i64], src: &[i64]));
$crate::rayon_parallel_unary!($rayon, $base, ZnxZero, znx_zero);
$crate::rayon_forward_znx!($rayon, $base, ZnxSwitchRing, znx_switch_ring(res: &mut [i64], a: &[i64]));

impl ZnxNormalize for $rayon {
    unsafe fn znx_normalize(res: LimbsMut<i64>, res_base2k: usize, res_k: usize, offset: i64, a: Limbs<i64>, a_base2k: usize, len: usize) {
        unsafe { <$base as ZnxNormalize>::znx_normalize(res, res_base2k, res_k, offset, a, a_base2k, len) }
    }
    unsafe fn znx_normalize_big(
        res: LimbsMut<i64>,
        res_base2k: usize,
        res_k: usize,
        offset: i64,
        a: Limbs<<$base as Backend>::BigWord>,
        a_base2k: usize,
        len: usize,
    ) {
        unsafe { <$base as ZnxNormalize>::znx_normalize_big(res, res_base2k, res_k, offset, a, a_base2k, len) }
    }
}

impl NttDFTExecute<NttTable<Primes30, <$base as Backend>::Ring>> for $rayon
{
    fn ntt_dft_execute(table: &NttTable<Primes30, <$base as Backend>::Ring>, data: &mut [u64]) {
        <$base as NttDFTExecute<NttTable<Primes30, <$base as Backend>::Ring>>>::ntt_dft_execute(table, data)
    }

    fn ntt_automorphism_plan(n: usize, p: i64) -> $crate::__private::poulpy_cpu_portable::kernels::ntt4x30::vec_znx_dft::NttAutomorphismPlan {
        <$base as NttDFTExecute<NttTable<Primes30, <$base as Backend>::Ring>>>::ntt_automorphism_plan(n, p)
    }
}

impl NttDFTExecute<NttTableInv<Primes30, <$base as Backend>::Ring>> for $rayon
{
    fn ntt_dft_execute(table: &NttTableInv<Primes30, <$base as Backend>::Ring>, data: &mut [u64]) {
        <$base as NttDFTExecute<NttTableInv<Primes30, <$base as Backend>::Ring>>>::ntt_dft_execute(table, data)
    }

    fn ntt_automorphism_plan(n: usize, p: i64) -> $crate::__private::poulpy_cpu_portable::kernels::ntt4x30::vec_znx_dft::NttAutomorphismPlan {
        <$base as NttDFTExecute<NttTableInv<Primes30, <$base as Backend>::Ring>>>::ntt_automorphism_plan(n, p)
    }
}

impl I128BigOps for $rayon {
    $crate::rayon_forward_i128_big!($base, i128_hadamard_product_i64(res: &mut [i128], a: &[i64], b: &[i64]));
    $crate::rayon_forward_i128_big!($base, i128_add(res: &mut [i128], a: &[i128], b: &[i128]));
    $crate::rayon_forward_i128_big!($base, i128_add_assign(res: &mut [i128], a: &[i128]));
    $crate::rayon_forward_i128_big!($base, i128_add_small(res: &mut [i128], a: &[i128], b: &[i64]));
    $crate::rayon_forward_i128_big!($base, i128_add_small_assign(res: &mut [i128], a: &[i64]));
    $crate::rayon_forward_i128_big!($base, i128_sub(res: &mut [i128], a: &[i128], b: &[i128]));
    $crate::rayon_forward_i128_big!($base, i128_sub_assign(res: &mut [i128], a: &[i128]));
    $crate::rayon_forward_i128_big!($base, i128_sub_negate_assign(res: &mut [i128], a: &[i128]));
    $crate::rayon_forward_i128_big!($base, i128_sub_small_a(res: &mut [i128], a: &[i64], b: &[i128]));
    $crate::rayon_forward_i128_big!($base, i128_sub_small_b(res: &mut [i128], a: &[i128], b: &[i64]));
    $crate::rayon_forward_i128_big!($base, i128_sub_small_assign(res: &mut [i128], a: &[i64]));
    $crate::rayon_forward_i128_big!($base, i128_sub_small_negate_assign(res: &mut [i128], a: &[i64]));
    $crate::rayon_forward_i128_big!($base, i128_negate(res: &mut [i128], a: &[i128]));
    $crate::rayon_forward_i128_big!($base, i128_negate_assign(res: &mut [i128]));
    $crate::rayon_forward_i128_big!($base, i128_neg_from_small(res: &mut [i128], a: &[i64]));
    $crate::rayon_forward_i128_big!($base, i128_from_small(res: &mut [i128], a: &[i64]));
}

impl BigWordHadamardProduct for $rayon {
    fn big_word_hadamard_product(res: &mut [i128], a: &[i64], b: &[i64]) {
        <Self as I128BigOps>::i128_hadamard_product_i64(res, a, b)
    }
}

unsafe impl $crate::__private::poulpy_hal::oep::HalMatZnxImpl for $rayon {}

unsafe impl HalVecZnxImpl for $rayon
{
    $crate::__private::poulpy_cpu_portable::hal_impl_vec_znx_without_normalize!();

    fn vec_znx_normalize(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_base2k: usize,
        res_k: usize,
        res_offset: i64,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_base2k: usize,
        a_col: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        $crate::normalize::vec_znx_normalize_par::<$base, Self>(res, res_base2k, res_k, res_offset, res_col, a, a_base2k, a_col);
    }

    fn vec_znx_normalize_assign(
        _module: &Module<Self>,
        base2k: usize,
        k: usize,
        a_offset: i64,
        a: &mut VecZnxBackendMut<'_, Self>,
        a_col: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        $crate::normalize::vec_znx_normalize_assign_par::<$base, Self>(base2k, k, a_offset, a, a_col);
    }
}
unsafe impl HalModuleImpl for $rayon
{
    $crate::__private::poulpy_cpu_portable::hal_impl_module!(NTT4x30ModuleDefault);
}
unsafe impl HalVmpImpl for $rayon
{
    fn vmp_prepare_tmp_bytes(module: &Module<Self>, _rows: usize, _cols_in: usize, _cols_out: usize, _size: usize) -> usize {
        <$base>::vmp_prepare_tmp_bytes(base_module(module), _rows, _cols_in, _cols_out, _size)
    }

    fn vmp_prepare(
        module: &Module<Self>,
        res: &mut VmpPMatBackendMut<'_, Self>,
        a: &MatZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let mut base_scratch = scratch.borrow().into_backend::<$base>();
        <$base>::vmp_prepare(base_module(module), &mut base_vmp_mut(res), a, &mut base_scratch);
    }

    fn vmp_apply_dft_to_dft_tmp_bytes(
        _module: &Module<Self>,
        _res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        _b_cols_out: usize,
        _b_size: usize,
    ) -> usize {
        $crate::workers(<Self as $crate::__private::poulpy_hal::execution::ScratchWorkers>::VMP)
            * <$base as $crate::ntt4x30::PackedNtt4x30Base>::vmp_apply_tmp_bytes(a_size, b_rows, b_cols_in)
    }

    fn vmp_apply_dft_to_dft(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        b: &VmpPMatBackendRef<'_, Self>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let per_worker = <$base as $crate::ntt4x30::PackedNtt4x30Base>::vmp_apply_tmp_bytes(a.size(), b.rows(), b.cols_in());
        let bytes = $crate::workers_within(
            <Self as $crate::__private::poulpy_hal::execution::ScratchWorkers>::VMP,
            per_worker,
            scratch.available(),
        ) * per_worker;
        let (tmp, _) = $crate::take_scratch::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        if RayonTaskExecutor::should_serialize_inner() {
            <$base as $crate::ntt4x30::PackedNtt4x30Base>::vmp_apply_dft_to_dft::<SerialTaskExecutor>(
                base_module(module),
                &mut base_dft_mut(res),
                &base_dft_ref(a),
                &base_vmp_ref(b),
                limb_offset,
                tmp,
            );
        } else {
            <$base as $crate::ntt4x30::PackedNtt4x30Base>::vmp_apply_dft_to_dft::<RayonTaskExecutor>(
                base_module(module),
                &mut base_dft_mut(res),
                &base_dft_ref(a),
                &base_vmp_ref(b),
                limb_offset,
                tmp,
            );
        }
    }

    fn vmp_apply_dft_to_dft_add_tmp_bytes(
        _module: &Module<Self>,
        _res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        _b_cols_out: usize,
        _b_size: usize,
    ) -> usize {
        $crate::workers(<Self as $crate::__private::poulpy_hal::execution::ScratchWorkers>::VMP)
            * <$base as $crate::ntt4x30::PackedNtt4x30Base>::vmp_apply_tmp_bytes(a_size, b_rows, b_cols_in)
    }

    fn vmp_apply_dft_to_dft_add(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        b: &VmpPMatBackendRef<'_, Self>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let per_worker = <$base as $crate::ntt4x30::PackedNtt4x30Base>::vmp_apply_tmp_bytes(a.size(), b.rows(), b.cols_in());
        let bytes = $crate::workers_within(
            <Self as $crate::__private::poulpy_hal::execution::ScratchWorkers>::VMP,
            per_worker,
            scratch.available(),
        ) * per_worker;
        let (tmp, _) = $crate::take_scratch::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        if RayonTaskExecutor::should_serialize_inner() {
            <$base as $crate::ntt4x30::PackedNtt4x30Base>::vmp_apply_dft_to_dft_add::<SerialTaskExecutor>(
                base_module(module),
                &mut base_dft_mut(res),
                &base_dft_ref(a),
                &base_vmp_ref(b),
                limb_offset,
                tmp,
            );
        } else {
            <$base as $crate::ntt4x30::PackedNtt4x30Base>::vmp_apply_dft_to_dft_add::<RayonTaskExecutor>(
                base_module(module),
                &mut base_dft_mut(res),
                &base_dft_ref(a),
                &base_vmp_ref(b),
                limb_offset,
                tmp,
            );
        }
    }

    fn vmp_extract_selected_rows(
        module: &Module<Self>,
        res: &mut VmpPMatBackendMut<'_, Self>,
        a: &VmpPMatBackendRef<'_, Self>,
        first_row: usize,
        row_step: usize,
    ) {
        <$base>::vmp_extract_selected_rows(
            base_module(module),
            &mut base_vmp_mut(res),
            &base_vmp_ref(a),
            first_row,
            row_step,
        )
    }

    fn vmp_zero(_module: &Module<Self>, res: &mut VmpPMatBackendMut<'_, Self>) {
        res.data_mut().fill(Default::default());
    }
}

unsafe impl HalConvolutionImpl for $rayon
{
    fn cnv_prepare_left_tmp_bytes(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
        $crate::workers(<Self as $crate::__private::poulpy_hal::execution::ScratchWorkers>::PREPARE)
            * <$base as $crate::ntt4x30::PackedNtt4x30Base>::cnv_prepare_tmp_bytes(module.n())
    }

    fn cnv_prepare_left(
        module: &Module<Self>,
        res: &mut $crate::__private::poulpy_hal::layouts::CnvPVecLBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let per_worker = <$base as $crate::ntt4x30::PackedNtt4x30Base>::cnv_prepare_tmp_bytes(res.n());
        let bytes = $crate::workers_within(
            res.size().min(<Self as $crate::__private::poulpy_hal::execution::ScratchWorkers>::PREPARE),
            per_worker,
            scratch.available(),
        ) * per_worker;
        let (tmp, _) = $crate::take_scratch::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        <$base as $crate::ntt4x30::PackedNtt4x30Base>::cnv_prepare_left::<Self, RayonTaskExecutor>(base_module(module), module, res, a, tmp);
    }

    fn cnv_prepare_right_tmp_bytes(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
        $crate::workers(<Self as $crate::__private::poulpy_hal::execution::ScratchWorkers>::PREPARE)
            * <$base as $crate::ntt4x30::PackedNtt4x30Base>::cnv_prepare_tmp_bytes(module.n())
    }

    fn cnv_prepare_right(
        module: &Module<Self>,
        res: &mut $crate::__private::poulpy_hal::layouts::CnvPVecRBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let per_worker = <$base as $crate::ntt4x30::PackedNtt4x30Base>::cnv_prepare_tmp_bytes(res.n());
        let bytes = $crate::workers_within(
            res.size().min(<Self as $crate::__private::poulpy_hal::execution::ScratchWorkers>::PREPARE),
            per_worker,
            scratch.available(),
        ) * per_worker;
        let (tmp, _) = $crate::take_scratch::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        <$base as $crate::ntt4x30::PackedNtt4x30Base>::cnv_prepare_right::<Self, RayonTaskExecutor>(base_module(module), module, res, a, tmp);
    }

    fn cnv_apply_dft_tmp_bytes(
        _module: &Module<Self>,
        _cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        let _ = (a_size, b_size);
        apply_tmp_bytes(res_size)
    }

    fn cnv_by_const_apply_tmp_bytes(
        module: &Module<Self>,
        cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        let _ = (module, cnv_offset);
        let _ = (res_size, a_size, b_size);
        0
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_by_const_apply(
        _module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, Self>,
        b_col: usize,
        b_coeff: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        if RayonTaskExecutor::should_serialize_inner() {
            $crate::__private::poulpy_cpu_portable::kernels::ntt4x30::convolution::ntt4x30_cnv_by_const_apply_portable::<Self, SerialTaskExecutor>(
                cnv_offset,
                res,
                res_col,
                a,
                a_col,
                b,
                b_col,
                b_coeff,
                &mut [],
            );
        } else {
            $crate::__private::poulpy_cpu_portable::kernels::ntt4x30::convolution::ntt4x30_cnv_by_const_apply_portable::<Self, RayonTaskExecutor>(
                cnv_offset,
                res,
                res_col,
                a,
                a_col,
                b,
                b_col,
                b_coeff,
                &mut [],
            );
        }
    }

    fn cnv_by_const_apply_add_tmp_bytes(
        module: &Module<Self>,
        cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        Self::cnv_by_const_apply_tmp_bytes(module, cnv_offset, res_size, a_size, b_size)
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_by_const_apply_add(
        _module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, Self>,
        b_col: usize,
        b_coeff: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        if RayonTaskExecutor::should_serialize_inner() {
            $crate::__private::poulpy_cpu_portable::kernels::ntt4x30::convolution::ntt4x30_cnv_by_const_apply_add_portable::<Self, SerialTaskExecutor>(
                cnv_offset,
                res,
                res_col,
                a,
                a_col,
                b,
                b_col,
                b_coeff,
                &mut [],
            );
        } else {
            $crate::__private::poulpy_cpu_portable::kernels::ntt4x30::convolution::ntt4x30_cnv_by_const_apply_add_portable::<Self, RayonTaskExecutor>(
                cnv_offset,
                res,
                res_col,
                a,
                a_col,
                b,
                b_col,
                b_coeff,
                &mut [],
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_apply_dft(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &$crate::__private::poulpy_hal::layouts::CnvPVecLBackendRef<'_, Self>,
        a_col: usize,
        b: &$crate::__private::poulpy_hal::layouts::CnvPVecRBackendRef<'_, Self>,
        b_col: usize,
        b_galois: i64,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let tmp = apply_scratch(res.size(), scratch);
        <$base as $crate::ntt4x30::PackedNtt4x30Base>::cnv_apply_dft::<Self, RayonTaskExecutor>(base_module(module), module, cnv_offset, res, res_col, a, a_col, b, b_col, b_galois, tmp);
    }

    fn cnv_apply_dft_add_tmp_bytes(
        module: &Module<Self>,
        cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        Self::cnv_apply_dft_tmp_bytes(module, cnv_offset, res_size, a_size, b_size)
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_apply_dft_add(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &$crate::__private::poulpy_hal::layouts::CnvPVecLBackendRef<'_, Self>,
        a_col: usize,
        b: &$crate::__private::poulpy_hal::layouts::CnvPVecRBackendRef<'_, Self>,
        b_col: usize,
        b_galois: i64,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let tmp = apply_scratch(res.size(), scratch);
        <$base as $crate::ntt4x30::PackedNtt4x30Base>::cnv_apply_dft_add::<Self, RayonTaskExecutor>(base_module(module), module, cnv_offset, res, res_col, a, a_col, b, b_col, b_galois, tmp);
    }

    fn cnv_apply_dft_sum_tmp_bytes(
        _module: &Module<Self>,
        _cnv_offset: usize,
        res_size: usize,
        _a_size: usize,
        _b_size: usize,
    ) -> usize {
        apply_tmp_bytes(res_size)
    }

    fn cnv_apply_dft_sum(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        terms: &[$crate::__private::poulpy_hal::layouts::CnvDftAccTerm<'_, Self>],
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let tmp = apply_scratch(res.size(), scratch);
        <$base as $crate::ntt4x30::PackedNtt4x30Base>::cnv_apply_dft_sum::<Self, RayonTaskExecutor>(base_module(module), module, cnv_offset, res, res_col, terms, tmp);
    }

    fn cnv_pairwise_apply_dft_tmp_bytes(
        _module: &Module<Self>,
        _cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize {
        let _ = (a_size, b_size);
        apply_tmp_bytes(res_size)
    }

    #[allow(clippy::too_many_arguments)]
    fn cnv_pairwise_apply_dft(
        module: &Module<Self>,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &$crate::__private::poulpy_hal::layouts::CnvPVecLBackendRef<'_, Self>,
        b: &$crate::__private::poulpy_hal::layouts::CnvPVecRBackendRef<'_, Self>,
        i: usize,
        j: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let tmp = apply_scratch(res.size(), scratch);
        <$base as $crate::ntt4x30::PackedNtt4x30Base>::cnv_pairwise_apply_dft::<Self, RayonTaskExecutor>(base_module(module), module, cnv_offset, res, res_col, a, b, i, j, tmp);
    }

    fn cnv_prepare_self_tmp_bytes(module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
        $crate::workers(<Self as $crate::__private::poulpy_hal::execution::ScratchWorkers>::PREPARE)
            * <$base as $crate::ntt4x30::PackedNtt4x30Base>::cnv_prepare_tmp_bytes(module.n())
    }

    fn cnv_prepare_self(
        module: &Module<Self>,
        left: &mut $crate::__private::poulpy_hal::layouts::CnvPVecLBackendMut<'_, Self>,
        right: &mut $crate::__private::poulpy_hal::layouts::CnvPVecRBackendMut<'_, Self>,
        a: &VecZnxBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let per_worker = <$base as $crate::ntt4x30::PackedNtt4x30Base>::cnv_prepare_tmp_bytes(left.n());
        let bytes = $crate::workers_within(
            left.size().min(<Self as $crate::__private::poulpy_hal::execution::ScratchWorkers>::PREPARE),
            per_worker,
            scratch.available(),
        ) * per_worker;
        let (tmp, _) = $crate::take_scratch::<Self, u64>(scratch.borrow(), bytes / size_of::<u64>());
        <$base as $crate::ntt4x30::PackedNtt4x30Base>::cnv_prepare_self::<Self, RayonTaskExecutor>(base_module(module), module, left, right, a, tmp);
    }
}
unsafe impl HalVecZnxBigImpl for $rayon
{
    $crate::__private::poulpy_cpu_portable::hal_impl_vec_znx_big_without_normalize!(NTT4x30VecZnxBigDefault);

    fn vec_znx_big_normalize(
        _module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_base2k: usize,
        res_k: usize,
        res_offset: i64,
        res_col: usize,
        a: &$crate::__private::poulpy_hal::layouts::VecZnxBigBackendRef<'_, Self>,
        a_base2k: usize,
        a_col: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        $crate::normalize::vec_znx_big_normalize_par::<$base, Self>(
            res,
            res_base2k,
            res_k,
            res_offset,
            res_col,
            &base_big_ref(a),
            a_base2k,
            a_col,
        );
    }
}
unsafe impl HalSvpImpl for $rayon
{
    fn svp_prepare(
        module: &Module<Self>,
        res: &mut SvpPPolBackendMut<'_, Self>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <$base>::svp_prepare(
            base_module(module),
            &mut base_svp_mut(res),
            res_col,
            &base_scalar_ref(a),
            a_col,
        );
    }

    fn svp_ppol_copy(
        module: &Module<Self>,
        res: &mut SvpPPolBackendMut<'_, Self>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <$base>::svp_ppol_copy(
            base_module(module),
            &mut base_svp_mut(res),
            res_col,
            &base_svp_ref(a),
            a_col,
        );
    }

    fn svp_apply_dft_tmp_bytes(_module: &Module<Self>, _b_size: usize) -> usize {
        0
    }

    #[allow(clippy::too_many_arguments)]
    fn svp_apply_dft(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, Self>,
        b_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let mut base_scratch = scratch.borrow().into_backend::<$base>();
        <$base>::svp_apply_dft(
            base_module(module),
            &mut base_dft_mut(res),
            res_col,
            &base_svp_ref(a),
            a_col,
            &base_znx_ref(b),
            b_col,
            &mut base_scratch,
        );
    }

    fn svp_apply_dft_to_dft(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, Self>,
        b_col: usize,
    ) {
        <$base>::svp_apply_dft_to_dft(
            base_module(module),
            &mut base_dft_mut(res),
            res_col,
            &base_svp_ref(a),
            a_col,
            &base_dft_ref(b),
            b_col,
        );
    }

    fn svp_apply_dft_to_dft_assign(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <$base>::svp_apply_dft_to_dft_assign(
            base_module(module),
            &mut base_dft_mut(res),
            res_col,
            &base_svp_ref(a),
            a_col,
        );
    }
}
unsafe impl HalVecZnxDftImpl for $rayon
{
    fn vec_znx_idft_normalize_tmp_bytes(module: &Module<Self>, _res_size: usize, a_size: usize) -> usize {
        let workers = $crate::workers(<Self as $crate::__private::poulpy_hal::execution::ScratchWorkers>::IDFT).min(a_size.max(1));
        workers * <$base as $crate::ntt4x30::PackedNtt4x30Base>::idft_tmp_words(module.n()) * size_of::<u64>()
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_idft_normalize(
        module: &Module<Self>,
        res: &mut $crate::__private::poulpy_hal::layouts::VecZnxBackendMut<'_, Self>,
        res_base2k: usize,
        res_k: usize,
        res_offset: i64,
        res_col: usize,
        a: &mut VecZnxDftBackendMut<'_, Self>,
        a_col: usize,
        a_base2k: usize,
        addend: Option<(&VecZnxBackendRef<'_, Self>, usize)>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        $crate::ntt4x30::check_packed_operand(res.n(), res, res_col);
        $crate::ntt4x30::check_packed_operand(res.n(), a, a_col);
        let n = a.n();
        $crate::__private::poulpy_hal::layouts::check_degree::<$base>(module.n(), n);
        let cols = a.cols();
        let size = a.size();
        let per_worker = <$base as $crate::ntt4x30::PackedNtt4x30Base>::idft_tmp_words(n);
        let arena = scratch.borrow();
        let workers = $crate::workers_within(
            size.clamp(1, <Self as $crate::__private::poulpy_hal::execution::ScratchWorkers>::IDFT),
            per_worker * size_of::<u64>(),
            arena.available(),
        );
        let (worker_tmp, _) = $crate::take_scratch::<Self, u64>(arena, workers * per_worker);

        // Each limb is replaced by its coefficients: the planes move to the worker's buffer during the transform.
        let base = base_module(module);
        let data = SendPtr::new(cast_slice_mut::<_, u32>(a.raw_mut()).as_mut_ptr());
        RayonTaskExecutor::for_each_chunked(size, worker_tmp, per_worker, |tmp, limb| {
            // Tasks take distinct limbs.
            let slot = unsafe { std::slice::from_raw_parts_mut(data.get().add(4 * n * (limb * cols + a_col)), 4 * n) };
            <$base as $crate::ntt4x30::PackedNtt4x30Base>::idft_limb_compact::<RayonTaskExecutor>(base, n, slot, tmp);
        });

        if let Some((add, add_col)) = addend {
            if add.n() == n {
                let big: &mut [i128] = cast_slice_mut(a.raw_mut());
                big.par_chunks_mut(n * cols)
                    .take(size.min(add.size()))
                    .enumerate()
                    .for_each(|(limb, group)| {
                        <$base as I128BigOps>::i128_add_small_assign(
                            &mut group[n * a_col..][..n],
                            add.at(add_col, limb),
                        );
                    });
            } else {
                let a_shape = a.shape();
                let mut big: VecZnxBigBackendMut<'_, Self> = VecZnxBig::from_shape(&mut **a.data_mut(), a_shape);
                let mut big_ref = &mut big;
                $crate::__private::poulpy_cpu_portable::kernels::ntt4x30::vec_znx_big::ntt4x30_vec_znx_big_add_small_assign_portable::<_, _, Self>(
                    &mut big_ref,
                    a_col,
                    &add,
                    add_col,
                );
            }
        }
        let a_shape = a.shape();
        let big_ref: $crate::__private::poulpy_hal::layouts::VecZnxBigBackendRef<'_, $base> = VecZnxBig::from_shape(&**a.data(), a_shape);
        $crate::normalize::vec_znx_big_normalize_par::<$base, Self>(res, res_base2k, res_k, res_offset, res_col, &big_ref, a_base2k, a_col);
    }
    fn vec_znx_dft_apply(
        module: &Module<Self>,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        $crate::ntt4x30::check_packed_operand(res.n(), res, res_col);
        $crate::ntt4x30::check_packed_operand(res.n(), a, a_col);
        $crate::__private::poulpy_hal::layouts::assert_dense(a, "vec_znx_dft_apply");
        assert!(step >= 1, "vec_znx_dft_apply: step must be >= 1");
        if !parallel_limb_tasks(res.size()) {
            return <$base>::vec_znx_dft_apply(
                base_module(module),
                step,
                offset,
                &mut base_dft_mut(res),
                res_col,
                &base_znx_ref(a),
                a_col,
            );
        }

        $crate::__private::poulpy_hal::layouts::check_degree::<$base>(module.n(), res.n());
        let n = res.n();
        let cols = res.cols();
        let a_size = a.size();
        let base = base_module(module);
        let data: &mut [u32] = cast_slice_mut(res.raw_mut());
        data.par_chunks_mut(4 * n * cols).enumerate().for_each(|(limb, group)| {
            let src_limb = offset + limb * step;
            <$base as $crate::ntt4x30::PackedNtt4x30Base>::dft_limb::<RayonTaskExecutor>(
                base,
                n,
                &mut group[4 * n * res_col..][..4 * n],
                (src_limb < a_size).then(|| a.at(a_col, src_limb)),
            );
        });
    }

    fn vec_znx_idft_apply_tmp_bytes(module: &Module<Self>) -> usize {
        <$base>::vec_znx_idft_apply_tmp_bytes(base_module(module)).max(
            $crate::workers(<Self as $crate::__private::poulpy_hal::execution::ScratchWorkers>::IDFT)
                * <$base as $crate::ntt4x30::PackedNtt4x30Base>::idft_tmp_words(module.n())
                * size_of::<u64>(),
        )
    }

    fn vec_znx_idft_apply(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        $crate::ntt4x30::check_packed_operand(res.n(), res, res_col);
        $crate::ntt4x30::check_packed_operand(res.n(), a, a_col);
        if !parallel_limb_tasks(res.size()) {
            let mut base_scratch = scratch.borrow().into_backend::<$base>();
            return <$base>::vec_znx_idft_apply(
                base_module(module),
                &mut base_big_mut(res),
                res_col,
                &base_dft_ref(a),
                a_col,
                &mut base_scratch,
            );
        }

        $crate::__private::poulpy_hal::layouts::check_degree::<$base>(module.n(), res.n());
        let n = res.n();
        let res_cols = res.cols();
        let a_cols = a.cols();
        let size = res.size();
        let min_size = size.min(a.size());
        let a_data: &[u32] = cast_slice(a.raw());
        let per_worker = <$base as $crate::ntt4x30::PackedNtt4x30Base>::idft_tmp_words(n);
        let workers = $crate::workers_within(
            size.min(<Self as $crate::__private::poulpy_hal::execution::ScratchWorkers>::IDFT),
            per_worker * size_of::<u64>(),
            scratch.available(),
        );
        let (worker_tmp, _) = $crate::take_scratch::<Self, u64>(scratch.borrow(), workers * per_worker);
        let res_ptr = SendPtr::new(res.raw_mut().as_mut_ptr());
        let module = base_module(module);
        RayonTaskExecutor::for_each_chunked(size, worker_tmp, per_worker, |tmp, limb| {
            let dst = unsafe { std::slice::from_raw_parts_mut(res_ptr.get().add(n * (limb * res_cols + res_col)), n) };
            if limb < min_size {
                let src = &a_data[4 * n * (limb * a_cols + a_col)..][..4 * n];
                <$base as $crate::ntt4x30::PackedNtt4x30Base>::idft_limb::<RayonTaskExecutor>(module, n, dst, src, tmp);
            } else {
                dst.fill(0);
            }
        });
    }

    fn vec_znx_dft_add(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, Self>,
        b_col: usize,
    ) {
        if limb_tasks(res.n(), res.size()) {
            <$base as $crate::ntt4x30::PackedNtt4x30Base>::vec_znx_dft_add::<RayonTaskExecutor>(
                &mut base_dft_mut(res),
                res_col,
                &base_dft_ref(a),
                a_col,
                &base_dft_ref(b),
                b_col,
            )
        } else {
            <$base as $crate::ntt4x30::PackedNtt4x30Base>::vec_znx_dft_add::<SerialTaskExecutor>(
                &mut base_dft_mut(res),
                res_col,
                &base_dft_ref(a),
                a_col,
                &base_dft_ref(b),
                b_col,
            )
        }
    }

    fn vec_znx_dft_add_assign(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        if limb_tasks(res.n(), res.size()) {
            <$base as $crate::ntt4x30::PackedNtt4x30Base>::vec_znx_dft_add_assign::<RayonTaskExecutor>(
                &mut base_dft_mut(res),
                res_col,
                &base_dft_ref(a),
                a_col,
            )
        } else {
            <$base as $crate::ntt4x30::PackedNtt4x30Base>::vec_znx_dft_add_assign::<SerialTaskExecutor>(
                &mut base_dft_mut(res),
                res_col,
                &base_dft_ref(a),
                a_col,
            )
        }
    }

    fn vec_znx_dft_sub(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, Self>,
        b_col: usize,
    ) {
        if limb_tasks(res.n(), res.size()) {
            <$base as $crate::ntt4x30::PackedNtt4x30Base>::vec_znx_dft_sub::<RayonTaskExecutor>(
                &mut base_dft_mut(res),
                res_col,
                &base_dft_ref(a),
                a_col,
                &base_dft_ref(b),
                b_col,
            )
        } else {
            <$base as $crate::ntt4x30::PackedNtt4x30Base>::vec_znx_dft_sub::<SerialTaskExecutor>(
                &mut base_dft_mut(res),
                res_col,
                &base_dft_ref(a),
                a_col,
                &base_dft_ref(b),
                b_col,
            )
        }
    }

    fn vec_znx_dft_sub_assign(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        if limb_tasks(res.n(), res.size()) {
            <$base as $crate::ntt4x30::PackedNtt4x30Base>::vec_znx_dft_sub_assign::<RayonTaskExecutor>(
                &mut base_dft_mut(res),
                res_col,
                &base_dft_ref(a),
                a_col,
            )
        } else {
            <$base as $crate::ntt4x30::PackedNtt4x30Base>::vec_znx_dft_sub_assign::<SerialTaskExecutor>(
                &mut base_dft_mut(res),
                res_col,
                &base_dft_ref(a),
                a_col,
            )
        }
    }

    fn vec_znx_dft_sub_negate_assign(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        if limb_tasks(res.n(), res.size()) {
            <$base as $crate::ntt4x30::PackedNtt4x30Base>::vec_znx_dft_sub_negate_assign::<RayonTaskExecutor>(
                &mut base_dft_mut(res),
                res_col,
                &base_dft_ref(a),
                a_col,
            )
        } else {
            <$base as $crate::ntt4x30::PackedNtt4x30Base>::vec_znx_dft_sub_negate_assign::<SerialTaskExecutor>(
                &mut base_dft_mut(res),
                res_col,
                &base_dft_ref(a),
                a_col,
            )
        }
    }

    fn vec_znx_dft_copy(
        _module: &Module<Self>,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        if limb_tasks(res.n(), res.size()) {
            <$base as $crate::ntt4x30::PackedNtt4x30Base>::vec_znx_dft_copy::<RayonTaskExecutor>(
                step,
                offset,
                &mut base_dft_mut(res),
                res_col,
                &base_dft_ref(a),
                a_col,
            )
        } else {
            <$base as $crate::ntt4x30::PackedNtt4x30Base>::vec_znx_dft_copy::<SerialTaskExecutor>(
                step,
                offset,
                &mut base_dft_mut(res),
                res_col,
                &base_dft_ref(a),
                a_col,
            )
        }
    }

    fn vec_znx_dft_zero(module: &Module<Self>, res: &mut VecZnxDftBackendMut<'_, Self>, res_col: usize) {
        <$base>::vec_znx_dft_zero(base_module(module), &mut base_dft_mut(res), res_col)
    }

    type AutomorphismPlan = <$base as HalVecZnxDftImpl>::AutomorphismPlan;

    fn vec_znx_dft_automorphism_plan(module: &Module<Self>, n: usize, p: i64) -> Self::AutomorphismPlan {
        <$base>::vec_znx_dft_automorphism_plan(base_module(module), n, p)
    }

    fn vec_znx_dft_automorphism_with_plan(
        _module: &Module<Self>,
        plan: &Self::AutomorphismPlan,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        <$base>::vec_znx_dft_automorphism_with_plan(base_module(_module), plan, &mut base_dft_mut(res), res_col, &base_dft_ref(a), a_col);
    }

    fn vec_znx_dft_automorphism_add_with_plan_tmp_bytes(_module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
        0
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_dft_automorphism_add_with_plan(
        _module: &Module<Self>,
        plan: &Self::AutomorphismPlan,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let _ = scratch;
        if RayonTaskExecutor::should_serialize_inner() {
            <$base as $crate::ntt4x30::PackedNtt4x30Base>::vec_znx_dft_automorphism_add::<SerialTaskExecutor>(
                plan,
                &mut base_dft_mut(res),
                res_col,
                &base_dft_ref(a),
                a_col,
            );
        } else {
            <$base as $crate::ntt4x30::PackedNtt4x30Base>::vec_znx_dft_automorphism_add::<RayonTaskExecutor>(
                plan,
                &mut base_dft_mut(res),
                res_col,
                &base_dft_ref(a),
                a_col,
            );
        }
    }
}

#[cfg(test)]
mod ntt4x30_rayon_tests {
    #[allow(unused_imports)]
    use super::*;

    /// Above the parallel floor of every tuning, so the kernels run as tasks on a pool of more than one thread.
    const COEFF_LEN: usize = 1 << 18;

    #[test]
    fn coefficient_kernels_wrap_above_the_parallel_floor() {
        let max = vec![i64::MAX; COEFF_LEN];
        let min = vec![i64::MIN; COEFF_LEN];
        let one = vec![1; COEFF_LEN];
        let mut actual = vec![0; COEFF_LEN];
        <$rayon as ZnxAdd>::znx_add(&mut actual, &max, &one);
        assert!(actual.iter().all(|&x| x == i64::MIN));
        <$rayon as ZnxSub>::znx_sub(&mut actual, &min, &one);
        assert!(actual.iter().all(|&x| x == i64::MAX));
        <$rayon as ZnxNegate>::znx_negate(&mut actual, &min);
        assert!(actual.iter().all(|&x| x == i64::MIN));
    }

    /// The element-wise operations split their limbs across tasks from `LIMB_TASKS_MIN_N` on.
    /// One call of each at that degree must give the words of the serial base, on a pool of more than one thread.
    /// The operands come from one forward transform each, so the check stays cheap under emulation.
    #[test]
    fn test_vec_znx_dft_element_wise_on_parallel_limbs() {
        use $crate::__private::poulpy_hal::{api::*, layouts::*, source::Source};

        let n = LIMB_TASKS_MIN_N;
        let cols = 2;
        let base = Module::<$base>::new(n as u64);
        let rayon = Module::<$rayon>::new(n as u64);
        let dft = |seed: u8, size: usize| {
            let mut coeffs = base.vec_znx_alloc(n, cols, size);
            base.vec_znx_fill_uniform_source_all(50, size * 50, &mut coeffs, &mut Source::new([seed; 32]));
            let mut dft = base.vec_znx_dft_alloc(n, cols, size);
            for col in 0..cols {
                base.vec_znx_dft_apply(1, 0, &mut dft.to_backend_mut(), col, &vec_znx_backend_ref::<$base>(&coeffs), col);
            }
            dft
        };
        // The two backends share the transform-domain layout, so the Rayon operands are byte copies.
        macro_rules! lift {
            ($dft:expr) => {{
                let mut lifted = rayon.vec_znx_dft_alloc(n, cols, $dft.size());
                lifted.raw_mut().copy_from_slice($dft.raw());
                lifted
            }};
        }
        let (a, b) = (dft(1, 4), dft(2, 3));
        let (a_rayon, b_rayon) = (lift!(a), lift!(b));
        macro_rules! check {
            ($name:literal, |$m:ident, $res:ident, $x:ident, $y:ident, $col:ident| $body:expr) => {{
                let mut want = dft(3, 4);
                let mut got = lift!(want);
                for $col in 0..cols {
                    {
                        let ($m, $res, $x, $y) = (&base, &mut want, &a, &b);
                        $body;
                    }
                    {
                        let ($m, $res, $x, $y) = (&rayon, &mut got, &a_rayon, &b_rayon);
                        $body;
                    }
                }
                assert!(want.raw() == got.raw(), "{} differs from the serial base", $name);
            }};
        }

        check!("vec_znx_dft_add", |m, res, x, y, col| m.vec_znx_dft_add(
            &mut res.to_backend_mut(),
            col,
            &x.to_backend_ref(),
            col,
            &y.to_backend_ref(),
            col
        ));
        check!("vec_znx_dft_add_assign", |m, res, x, _y, col| m
            .vec_znx_dft_add_assign(&mut res.to_backend_mut(), col, &x.to_backend_ref(), col));
        check!("vec_znx_dft_sub", |m, res, x, y, col| m.vec_znx_dft_sub(
            &mut res.to_backend_mut(),
            col,
            &x.to_backend_ref(),
            col,
            &y.to_backend_ref(),
            col
        ));
        check!("vec_znx_dft_sub_assign", |m, res, x, _y, col| m
            .vec_znx_dft_sub_assign(&mut res.to_backend_mut(), col, &x.to_backend_ref(), col));
        check!("vec_znx_dft_sub_negate_assign", |m, res, x, _y, col| m
            .vec_znx_dft_sub_negate_assign(&mut res.to_backend_mut(), col, &x.to_backend_ref(), col));
        check!("vec_znx_dft_copy", |m, res, x, _y, col| m.vec_znx_dft_copy(
            2,
            1,
            &mut res.to_backend_mut(),
            col,
            &x.to_backend_ref(),
            col
        ));
        check!("vec_znx_dft_automorphism_add_with_plan", |m, res, x, _y, col| {
            let plan = m.vec_znx_dft_automorphism_plan(n, 5);
            let mut scratch = ScratchOwned::alloc(m.vec_znx_dft_automorphism_add_with_plan_tmp_bytes(res.size(), x.size()));
            m.vec_znx_dft_automorphism_add_with_plan(&plan, &mut res.to_backend_mut(), col, &x.to_backend_ref(), col, &mut scratch.arena())
        });
    }
}

        }
    };
}
