//! Open extension points for the SVP family.

#![allow(clippy::too_many_arguments)]

/// Emits the OEP trait for one SVP scalar prep tier.
///
/// `$Tier`/`$stem` are the tier's camel and snake tokens (`PPol`/`ppol`), from
/// which every method name and the scalar view types are derived.
macro_rules! hal_svp_tier_impl {
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
        /// [`HalSvpImpl`](super::HalSvpImpl).
        pub unsafe trait $Trait<BE: Backend>: Backend {
            fn [<svp_prepare_ $stem>](
                module: &Module<BE>,
                res: &mut crate::layouts::[<Svp $Tier BackendMut>]<'_, BE>,
                res_col: usize,
                a: &ScalarZnxBackendRef<'_, BE>,
                a_col: usize,
            );
            fn [<svp_ $stem _copy_backend>](
                module: &Module<BE>,
                res: &mut crate::layouts::[<Svp $Tier BackendMut>]<'_, BE>,
                res_col: usize,
                a: &crate::layouts::[<Svp $Tier BackendRef>]<'_, BE>,
                a_col: usize,
            );

            hal_svp_tier_apply!($Tier, $stem, small_to_dft, VecZnxDftBackendMut, VecZnxBackendRef);
            hal_svp_tier_apply!($Tier, $stem, dft_to_dft, VecZnxDftBackendMut, VecZnxDftBackendRef);
            hal_svp_tier_apply!($Tier, $stem, small_to_big, VecZnxBigBackendMut, VecZnxBackendRef, scratch);
            hal_svp_tier_apply!($Tier, $stem, dft_to_big, VecZnxBigBackendMut, VecZnxDftBackendRef, scratch);
            hal_svp_tier_norm_apply!($Tier, $stem, small_to_small, VecZnxBackendRef);
            hal_svp_tier_norm_apply!($Tier, $stem, dft_to_small, VecZnxDftBackendRef);

            fn [<svp_apply_ $stem _dft_to_dft_assign>](
                module: &Module<BE>,
                res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
                res_col: usize,
                a: &crate::layouts::[<Svp $Tier BackendRef>]<'_, BE>,
                a_col: usize,
            );
        }
        }
    };
}

/// One apply method whose result is a DFT or big vector. A trailing `scratch`
/// token adds the scratch parameter the `_to_big` variants carry.
macro_rules! hal_svp_tier_apply {
    ($Tier:ident, $stem:ident, $variant:ident, $Res:ident, $B:ident $(, $scratch:ident)?) => {
        paste::paste! {
            #[allow(clippy::too_many_arguments)]
            fn [<svp_apply_ $stem _ $variant>](
                module: &Module<BE>,
                res: &mut crate::layouts::$Res<'_, BE>,
                res_col: usize,
                a: &crate::layouts::[<Svp $Tier BackendRef>]<'_, BE>,
                a_col: usize,
                b: &crate::layouts::$B<'_, BE>,
                b_col: usize,
                $($scratch: &mut crate::layouts::ScratchArena<'_, BE>,)?
            );
        }
    };
}

/// One apply method whose result is a normalized vector.
macro_rules! hal_svp_tier_norm_apply {
    ($Tier:ident, $stem:ident, $variant:ident, $B:ident) => {
        paste::paste! {
            #[allow(clippy::too_many_arguments)]
            fn [<svp_apply_ $stem _ $variant>](
                module: &Module<BE>,
                res: &mut crate::layouts::VecZnxBackendMut<'_, BE>,
                res_base2k: usize,
                res_offset: i64,
                res_col: usize,
                a: &crate::layouts::[<Svp $Tier BackendRef>]<'_, BE>,
                a_col: usize,
                b: &crate::layouts::$B<'_, BE>,
                b_base2k: usize,
                b_col: usize,
                scratch: &mut crate::layouts::ScratchArena<'_, BE>,
            );
        }
    };
}

mod ppol;
mod tpol;

pub use ppol::HalSvpPPolImpl;
pub use tpol::HalSvpTPolImpl;

use crate::layouts::{Backend, Module};

/// Scalar-vector product family extension point.
///
/// # Safety
/// Implementations must uphold the backend safety contract for prepared
/// polynomial layouts and arithmetic correctness.
pub unsafe trait HalSvpImpl<BE: Backend>: HalSvpPPolImpl<BE> + HalSvpTPolImpl<BE> {
    fn svp_apply_to_big_tmp_bytes(module: &Module<BE>, res_size: usize) -> usize;

    fn svp_apply_to_small_tmp_bytes(module: &Module<BE>, b_size: usize) -> usize;

    #[allow(clippy::too_many_arguments)]
    fn svp_apply_small_small_to_dft(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxBackendRef<'_, BE>,
        b_col: usize,
    );

    #[allow(clippy::too_many_arguments)]
    fn svp_apply_small_dft_to_dft(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        b_col: usize,
    );

    #[allow(clippy::too_many_arguments)]
    fn svp_apply_small_small_to_big(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxBackendRef<'_, BE>,
        b_col: usize,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );

    #[allow(clippy::too_many_arguments)]
    fn svp_apply_small_dft_to_big(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        b_col: usize,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );

    #[allow(clippy::too_many_arguments)]
    fn svp_apply_small_small_to_small(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBackendMut<'_, BE>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &crate::layouts::ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxBackendRef<'_, BE>,
        b_base2k: usize,
        b_col: usize,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );

    #[allow(clippy::too_many_arguments)]
    fn svp_apply_small_dft_to_small(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBackendMut<'_, BE>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &crate::layouts::ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        b_base2k: usize,
        b_col: usize,
        scratch: &mut crate::layouts::ScratchArena<'_, BE>,
    );

    fn svp_apply_small_dft_to_dft_assign(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
    );
}
