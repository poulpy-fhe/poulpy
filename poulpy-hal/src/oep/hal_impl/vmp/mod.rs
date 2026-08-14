//! Open extension points for the VMP family.

#![allow(clippy::too_many_arguments)]

/// Emits the OEP trait for one VMP matrix prep tier.
///
/// `$Tier`/`$stem` are the tier's camel and snake tokens (`PMat`/`pmat`), from
/// which every method name and the matrix view types are derived.
macro_rules! hal_vmp_tier_impl {
    (
        $(#[$meta:meta])*
        $Trait:ident, $Tier:ident, $stem:ident
    ) => {
        paste::paste! {
        $(#[$meta])*
        ///
        /// # Safety
        ///
        /// Implementations must uphold the backend safety contract; see
        /// [`HalVmpImpl`](super::HalVmpImpl).
        pub unsafe trait $Trait<BE: Backend>: Backend {
            fn [<vmp_prepare_ $stem _tmp_bytes>](
                module: &Module<BE>,
                rows: usize,
                cols_in: usize,
                cols_out: usize,
                size: usize,
            ) -> usize;
            fn [<vmp_prepare_ $stem>](
                module: &Module<BE>,
                res: &mut crate::layouts::[<Vmp $Tier BackendMut>]<'_, BE>,
                a: &crate::layouts::MatZnxBackendRef<'_, BE>,
                scratch: &mut ScratchArena<'_, BE>,
            );

            hal_vmp_tier_apply!($Tier, $stem, small_to_dft, VecZnxDftBackendMut, VecZnxBackendRef);
            hal_vmp_tier_apply!($Tier, $stem, dft_to_dft, VecZnxDftBackendMut, VecZnxDftBackendRef, limb_offset);
            hal_vmp_tier_apply!($Tier, $stem, small_to_dft_accumulate, VecZnxDftBackendMut, VecZnxBackendRef);
            hal_vmp_tier_apply!($Tier, $stem, dft_to_dft_accumulate, VecZnxDftBackendMut, VecZnxDftBackendRef, limb_offset);
            hal_vmp_tier_apply!($Tier, $stem, small_to_big, VecZnxBigBackendMut, VecZnxBackendRef);
            hal_vmp_tier_apply!($Tier, $stem, dft_to_big, VecZnxBigBackendMut, VecZnxDftBackendRef, limb_offset);
            hal_vmp_tier_norm_apply!($Tier, $stem, small_to_small, VecZnxBackendRef);
            hal_vmp_tier_norm_apply!($Tier, $stem, dft_to_small, VecZnxDftBackendRef, limb_offset);
        }
        }
    };
}

/// One `*_tmp_bytes` + apply method pair whose result is a DFT or big vector.
macro_rules! hal_vmp_tier_apply {
    ($Tier:ident, $stem:ident, $variant:ident, $Res:ident, $B:ident $(, $off:ident)?) => {
        paste::paste! {
            fn [<vmp_apply_ $stem _ $variant _tmp_bytes>](
                module: &Module<BE>,
                res_size: usize,
                a_rows: usize,
                a_cols_in: usize,
                a_cols_out: usize,
                a_size: usize,
                b_size: usize,
            ) -> usize;
            fn [<vmp_apply_ $stem _ $variant>](
                module: &Module<BE>,
                res: &mut crate::layouts::$Res<'_, BE>,
                a: &crate::layouts::[<Vmp $Tier BackendRef>]<'_, BE>,
                b: &crate::layouts::$B<'_, BE>,
                $($off: usize,)?
                scratch: &mut crate::layouts::ScratchArena<'_, BE>,
            );
        }
    };
}

/// One `*_tmp_bytes` + apply method pair whose result is a normalized vector.
macro_rules! hal_vmp_tier_norm_apply {
    ($Tier:ident, $stem:ident, $variant:ident, $B:ident $(, $off:ident)?) => {
        paste::paste! {
            fn [<vmp_apply_ $stem _ $variant _tmp_bytes>](
                module: &Module<BE>,
                res_size: usize,
                a_rows: usize,
                a_cols_in: usize,
                a_cols_out: usize,
                a_size: usize,
                b_size: usize,
            ) -> usize;
            fn [<vmp_apply_ $stem _ $variant>](
                module: &Module<BE>,
                res: &mut crate::layouts::VecZnxBackendMut<'_, BE>,
                res_base2k: usize,
                res_offset: i64,
                a: &crate::layouts::[<Vmp $Tier BackendRef>]<'_, BE>,
                b: &crate::layouts::$B<'_, BE>,
                b_base2k: usize,
                $($off: usize,)?
                scratch: &mut crate::layouts::ScratchArena<'_, BE>,
            );
        }
    };
}

mod pmat;
mod tmat;

pub use pmat::HalVmpPMatImpl;
pub use tmat::HalVmpTMatImpl;

use crate::layouts::{Backend, Module};

/// Vector-matrix product family extension point.
///
/// # Safety
/// Implementations must uphold the backend safety contract for prepared matrix
/// layouts, scratch usage, and arithmetic correctness.
pub unsafe trait HalVmpImpl<BE: Backend>: HalVmpPMatImpl<BE> + HalVmpTMatImpl<BE> {
    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_small_to_dft_tmp_bytes(
        module: &Module<BE>,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_small_to_dft(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        a: &crate::layouts::MatZnxBackendRef<'_, BE>,
        b: &crate::layouts::VecZnxBackendRef<'_, BE>,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_dft_to_dft_tmp_bytes(
        module: &Module<BE>,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_dft_to_dft(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        a: &crate::layouts::MatZnxBackendRef<'_, BE>,
        b: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        limb_offset: usize,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_small_to_dft_accumulate_tmp_bytes(
        module: &Module<BE>,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_small_to_dft_accumulate(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        a: &crate::layouts::MatZnxBackendRef<'_, BE>,
        b: &crate::layouts::VecZnxBackendRef<'_, BE>,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_dft_to_dft_accumulate_tmp_bytes(
        module: &Module<BE>,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_dft_to_dft_accumulate(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        a: &crate::layouts::MatZnxBackendRef<'_, BE>,
        b: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        limb_offset: usize,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_small_to_big_tmp_bytes(
        module: &Module<BE>,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_small_to_big(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        a: &crate::layouts::MatZnxBackendRef<'_, BE>,
        b: &crate::layouts::VecZnxBackendRef<'_, BE>,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_dft_to_big_tmp_bytes(
        module: &Module<BE>,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_dft_to_big(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        a: &crate::layouts::MatZnxBackendRef<'_, BE>,
        b: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        limb_offset: usize,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_small_to_small_tmp_bytes(
        module: &Module<BE>,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_small_to_small(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBackendMut<'_, BE>,
        res_base2k: usize,
        res_offset: i64,
        a: &crate::layouts::MatZnxBackendRef<'_, BE>,
        b: &crate::layouts::VecZnxBackendRef<'_, BE>,
        b_base2k: usize,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_dft_to_small_tmp_bytes(
        module: &Module<BE>,
        res_size: usize,
        a_rows: usize,
        a_cols_in: usize,
        a_cols_out: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_small_dft_to_small(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBackendMut<'_, BE>,
        res_base2k: usize,
        res_offset: i64,
        a: &crate::layouts::MatZnxBackendRef<'_, BE>,
        b: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        b_base2k: usize,
        limb_offset: usize,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );
}
