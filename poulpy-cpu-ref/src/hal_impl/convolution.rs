//! `HalConvolutionImpl` building blocks.
//!
//! Each macro emits one independent group of methods and expands on its own; a
//! backend composes the groups it wants inside its `impl` block and writes the
//! rest by hand. No macro here invokes another.

/// Operand prep, one group per tier: `$tier` is the method stem (`pvec`),
/// `$Tier` the container stem (`PVec`).
#[macro_export]
/// Operand prep for the pvec tier, forwarded to the defaults trait.
macro_rules! cnv_impl_prepares_pvec {
    ($defaults:ident) => {
        fn cnv_prepare_left_pvec_tmp_bytes(module: &Module<Self>, res_size: usize, a_size: usize) -> usize {
            <Self as $defaults<Self>>::cnv_prepare_left_pvec_tmp_bytes_default(module, res_size, a_size)
        }

        fn cnv_prepare_left_pvec(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::CnvPVecLBackendMut<'_, Self>,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            mask: i64,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as $defaults<Self>>::cnv_prepare_left_pvec_default(module, res, a, mask, &mut scratch);
        }

        fn cnv_prepare_right_pvec_tmp_bytes(module: &Module<Self>, res_size: usize, a_size: usize) -> usize {
            <Self as $defaults<Self>>::cnv_prepare_right_pvec_tmp_bytes_default(module, res_size, a_size)
        }

        fn cnv_prepare_right_pvec(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::CnvPVecRBackendMut<'_, Self>,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            mask: i64,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as $defaults<Self>>::cnv_prepare_right_pvec_default(module, res, a, mask, &mut scratch);
        }

        fn cnv_prepare_self_pvec_tmp_bytes(module: &Module<Self>, res_size: usize, a_size: usize) -> usize {
            <Self as $defaults<Self>>::cnv_prepare_self_pvec_tmp_bytes_default(module, res_size, a_size)
        }

        fn cnv_prepare_self_pvec(
            module: &Module<Self>,
            left: &mut poulpy_hal::layouts::CnvPVecLBackendMut<'_, Self>,
            right: &mut poulpy_hal::layouts::CnvPVecRBackendMut<'_, Self>,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            mask: i64,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as $defaults<Self>>::cnv_prepare_self_pvec_default(module, left, right, a, mask, &mut scratch);
        }
    };
}

/// Operand prep for the tvec tier, forwarded to the defaults trait.
#[macro_export]
macro_rules! cnv_impl_prepares_tvec {
    ($defaults:ident) => {
        fn cnv_prepare_left_tvec_tmp_bytes(module: &Module<Self>, res_size: usize, a_size: usize) -> usize {
            <Self as $defaults<Self>>::cnv_prepare_left_tvec_tmp_bytes_default(module, res_size, a_size)
        }

        fn cnv_prepare_left_tvec(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::CnvTVecLBackendMut<'_, Self>,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            mask: i64,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as $defaults<Self>>::cnv_prepare_left_tvec_default(module, res, a, mask, &mut scratch);
        }

        fn cnv_prepare_right_tvec_tmp_bytes(module: &Module<Self>, res_size: usize, a_size: usize) -> usize {
            <Self as $defaults<Self>>::cnv_prepare_right_tvec_tmp_bytes_default(module, res_size, a_size)
        }

        fn cnv_prepare_right_tvec(
            module: &Module<Self>,
            res: &mut poulpy_hal::layouts::CnvTVecRBackendMut<'_, Self>,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            mask: i64,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as $defaults<Self>>::cnv_prepare_right_tvec_default(module, res, a, mask, &mut scratch);
        }

        fn cnv_prepare_self_tvec_tmp_bytes(module: &Module<Self>, res_size: usize, a_size: usize) -> usize {
            <Self as $defaults<Self>>::cnv_prepare_self_tvec_tmp_bytes_default(module, res_size, a_size)
        }

        fn cnv_prepare_self_tvec(
            module: &Module<Self>,
            left: &mut poulpy_hal::layouts::CnvTVecLBackendMut<'_, Self>,
            right: &mut poulpy_hal::layouts::CnvTVecRBackendMut<'_, Self>,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            mask: i64,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as $defaults<Self>>::cnv_prepare_self_tvec_default(module, left, right, a, mask, &mut scratch);
        }
    };
}

/// Apply methods for the pvec tier, forwarded to the defaults trait.
/// All operands and results stay in the DFT domain.
#[macro_export]
macro_rules! cnv_impl_apply_pvec {
    ($defaults:ident) => {
        fn cnv_apply_pvec_to_dft_tmp_bytes(
            module: &Module<Self>,
            cnv_offset: usize,
            res_size: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            <Self as $defaults<Self>>::cnv_apply_pvec_to_dft_tmp_bytes_default(module, cnv_offset, res_size, a_size, b_size)
        }

        #[allow(clippy::too_many_arguments)]
        fn cnv_apply_pvec_to_dft(
            module: &Module<Self>,
            cnv_offset: usize,
            mut res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::CnvPVecLBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::CnvPVecRBackendRef<'_, Self>,
            b_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as $defaults<Self>>::cnv_apply_pvec_to_dft_default(
                module,
                cnv_offset,
                &mut res,
                res_col,
                a,
                a_col,
                b,
                b_col,
                &mut scratch,
            );
        }

        fn cnv_apply_pvec_to_dft_accumulate_tmp_bytes(
            module: &Module<Self>,
            cnv_offset: usize,
            res_size: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            <Self as $defaults<Self>>::cnv_apply_pvec_to_dft_accumulate_tmp_bytes_default(
                module, cnv_offset, res_size, a_size, b_size,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn cnv_apply_pvec_to_dft_accumulate(
            module: &Module<Self>,
            cnv_offset: usize,
            mut res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::CnvPVecLBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::CnvPVecRBackendRef<'_, Self>,
            b_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as $defaults<Self>>::cnv_apply_pvec_to_dft_accumulate_default(
                module,
                cnv_offset,
                &mut res,
                res_col,
                a,
                a_col,
                b,
                b_col,
                &mut scratch,
            );
        }

        fn cnv_accumulate_pvec_to_dft_tmp_bytes(
            module: &Module<Self>,
            cnv_offset: usize,
            res_size: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            <Self as $defaults<Self>>::cnv_accumulate_pvec_to_dft_tmp_bytes_default(module, cnv_offset, res_size, a_size, b_size)
        }

        fn cnv_accumulate_pvec_to_dft<'a>(
            module: &Module<Self>,
            cnv_offset: usize,
            mut res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            terms: &[poulpy_hal::layouts::CnvDftAccTermPvec<'a, Self>],
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) where
            Self: 'a,
        {
            let mut scratch = scratch.borrow();
            <Self as $defaults<Self>>::cnv_accumulate_pvec_to_dft_default(
                module,
                cnv_offset,
                &mut res,
                res_col,
                terms,
                &mut scratch,
            );
        }

        fn cnv_pairwise_apply_pvec_to_dft_tmp_bytes(
            module: &Module<Self>,
            cnv_offset: usize,
            res_size: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            <Self as $defaults<Self>>::cnv_pairwise_apply_pvec_to_dft_tmp_bytes_default(
                module, cnv_offset, res_size, a_size, b_size,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn cnv_pairwise_apply_pvec_to_dft(
            module: &Module<Self>,
            cnv_offset: usize,
            mut res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::CnvPVecLBackendRef<'_, Self>,
            b: &poulpy_hal::layouts::CnvPVecRBackendRef<'_, Self>,
            i: usize,
            j: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as $defaults<Self>>::cnv_pairwise_apply_pvec_to_dft_default(
                module,
                cnv_offset,
                &mut res,
                res_col,
                a,
                b,
                i,
                j,
                &mut scratch,
            );
        }
    };
}

/// Apply methods for the tvec tier, forwarded to the defaults trait.
/// All operands and results stay in the DFT domain.
#[macro_export]
macro_rules! cnv_impl_apply_tvec {
    ($defaults:ident) => {
        fn cnv_apply_tvec_to_dft_tmp_bytes(
            module: &Module<Self>,
            cnv_offset: usize,
            res_size: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            <Self as $defaults<Self>>::cnv_apply_tvec_to_dft_tmp_bytes_default(module, cnv_offset, res_size, a_size, b_size)
        }

        #[allow(clippy::too_many_arguments)]
        fn cnv_apply_tvec_to_dft(
            module: &Module<Self>,
            cnv_offset: usize,
            mut res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::CnvTVecLBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::CnvTVecRBackendRef<'_, Self>,
            b_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as $defaults<Self>>::cnv_apply_tvec_to_dft_default(
                module,
                cnv_offset,
                &mut res,
                res_col,
                a,
                a_col,
                b,
                b_col,
                &mut scratch,
            );
        }

        fn cnv_apply_tvec_to_dft_accumulate_tmp_bytes(
            module: &Module<Self>,
            cnv_offset: usize,
            res_size: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            <Self as $defaults<Self>>::cnv_apply_tvec_to_dft_accumulate_tmp_bytes_default(
                module, cnv_offset, res_size, a_size, b_size,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn cnv_apply_tvec_to_dft_accumulate(
            module: &Module<Self>,
            cnv_offset: usize,
            mut res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::CnvTVecLBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::CnvTVecRBackendRef<'_, Self>,
            b_col: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as $defaults<Self>>::cnv_apply_tvec_to_dft_accumulate_default(
                module,
                cnv_offset,
                &mut res,
                res_col,
                a,
                a_col,
                b,
                b_col,
                &mut scratch,
            );
        }

        fn cnv_accumulate_tvec_to_dft_tmp_bytes(
            module: &Module<Self>,
            cnv_offset: usize,
            res_size: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            <Self as $defaults<Self>>::cnv_accumulate_tvec_to_dft_tmp_bytes_default(module, cnv_offset, res_size, a_size, b_size)
        }

        fn cnv_accumulate_tvec_to_dft<'a>(
            module: &Module<Self>,
            cnv_offset: usize,
            mut res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            terms: &[poulpy_hal::layouts::CnvDftAccTermTvec<'a, Self>],
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) where
            Self: 'a,
        {
            let mut scratch = scratch.borrow();
            <Self as $defaults<Self>>::cnv_accumulate_tvec_to_dft_default(
                module,
                cnv_offset,
                &mut res,
                res_col,
                terms,
                &mut scratch,
            );
        }

        fn cnv_pairwise_apply_tvec_to_dft_tmp_bytes(
            module: &Module<Self>,
            cnv_offset: usize,
            res_size: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            <Self as $defaults<Self>>::cnv_pairwise_apply_tvec_to_dft_tmp_bytes_default(
                module, cnv_offset, res_size, a_size, b_size,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn cnv_pairwise_apply_tvec_to_dft(
            module: &Module<Self>,
            cnv_offset: usize,
            mut res: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::CnvTVecLBackendRef<'_, Self>,
            b: &poulpy_hal::layouts::CnvTVecRBackendRef<'_, Self>,
            i: usize,
            j: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as $defaults<Self>>::cnv_pairwise_apply_tvec_to_dft_default(
                module,
                cnv_offset,
                &mut res,
                res_col,
                a,
                b,
                i,
                j,
                &mut scratch,
            );
        }
    };
}

/// `cnv_by_const_apply`, forwarded to the defaults trait.
#[macro_export]
macro_rules! cnv_impl_by_const {
    ($defaults:ident) => {
        fn cnv_by_const_apply_tmp_bytes(
            module: &Module<Self>,
            cnv_offset: usize,
            res_size: usize,
            a_size: usize,
            b_size: usize,
        ) -> usize {
            <Self as $defaults<Self>>::cnv_by_const_apply_tmp_bytes_default(module, cnv_offset, res_size, a_size, b_size)
        }

        #[allow(clippy::too_many_arguments)]
        fn cnv_by_const_apply(
            module: &Module<Self>,
            cnv_offset: usize,
            mut res: &mut poulpy_hal::layouts::VecZnxBigBackendMut<'_, Self>,
            res_col: usize,
            a: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            a_col: usize,
            b: &poulpy_hal::layouts::VecZnxBackendRef<'_, Self>,
            b_col: usize,
            b_coeff: usize,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
        ) {
            let mut scratch = scratch.borrow();
            <Self as $defaults<Self>>::cnv_by_const_apply_default(
                module,
                cnv_offset,
                &mut res,
                res_col,
                a,
                a_col,
                b,
                b_col,
                b_coeff,
                &mut scratch,
            );
        }
    };
}
