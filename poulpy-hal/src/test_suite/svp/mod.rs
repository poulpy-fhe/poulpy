//! Cross-backend tests for the SVP apply matrix.
//!
//! One exported test per `svp_apply_<scalar>_<vector>_to_<output>` variant, each
//! bounded on that variant alone (plus its tier's prepare and the DFT/normalize
//! machinery used to settle the result). A backend implementing a subset of the
//! matrix can register exactly the tests it supports.

use super::{
    TestParams, download_vec_znx, scalar_znx_backend_ref, upload_scalar_znx, upload_vec_znx, vec_znx_backend_mut,
    vec_znx_backend_ref,
};
use crate::layouts::SvpPPolToBackendMut;
use crate::layouts::SvpPPolToBackendRef;
use crate::layouts::SvpTPolToBackendMut;
use crate::layouts::SvpTPolToBackendRef;
use crate::layouts::VecZnxBigOwned;
use crate::layouts::VecZnxBigToBackendMut;
use crate::layouts::VecZnxBigToBackendRef;
use crate::layouts::VecZnxDftOwned;
use crate::layouts::VecZnxDftToBackendMut;
use crate::layouts::VecZnxDftToBackendRef;
use crate::layouts::ZnxView;

use crate::{
    api::{
        ScratchOwnedAlloc, SvpApplyPPolDftToBig, SvpApplyPPolDftToDft, SvpApplyPPolDftToDftAssign, SvpApplyPPolDftToSmall,
        SvpApplyPPolSmallToBig, SvpApplyPPolSmallToDft, SvpApplyPPolSmallToSmall, SvpApplySmallDftToBig, SvpApplySmallDftToDft,
        SvpApplySmallDftToDftAssign, SvpApplySmallDftToSmall, SvpApplySmallSmallToBig, SvpApplySmallSmallToDft,
        SvpApplySmallSmallToSmall, SvpApplyTPolDftToBig, SvpApplyTPolDftToDft, SvpApplyTPolDftToDftAssign,
        SvpApplyTPolDftToSmall, SvpApplyTPolSmallToBig, SvpApplyTPolSmallToDft, SvpApplyTPolSmallToSmall, SvpApplyToBigTmpBytes,
        SvpApplyToSmallTmpBytes, SvpPPolAlloc, SvpPreparePPol, SvpPrepareTPol, SvpTPolAlloc, VecZnxBigAlloc, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxDftAlloc, VecZnxDftApply, VecZnxIdftApplyTmpA,
    },
    layouts::{Backend, FillUniform, HostBytesBackend, Module, ScratchOwned, SvpPPolOwned, SvpTPolOwned, VecZnx},
    source::Source,
};

const COLS: usize = 2;
const SIZES: [usize; 4] = [1, 2, 3, 4];

fn idft_into_alloc<BE>(module: &Module<BE>, a: &mut VecZnxDftOwned<BE>) -> VecZnxBigOwned<BE>
where
    BE: Backend,
    Module<BE>: VecZnxIdftApplyTmpA<BE>,
{
    let cols = a.cols();
    let size = a.size();
    let mut res = module.vec_znx_big_alloc(cols, size);
    for j in 0..cols {
        let mut res_backend = res.to_backend_mut();
        let mut a_backend = a.to_backend_mut();
        module.vec_znx_idft_apply_tmpa(&mut res_backend, j, &mut a_backend, j);
    }
    res
}

/// Lifts `a` into the DFT domain, column by column.
fn dft_of<BE>(module: &Module<BE>, a: &VecZnx<BE::OwnedBuf, i64>, cols: usize, size: usize) -> VecZnxDftOwned<BE>
where
    BE: crate::test_suite::TestBackend,
    Module<BE>: VecZnxDftApply<BE>,
{
    let mut res: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(cols, size);
    for j in 0..cols {
        module.vec_znx_dft_apply(1, 0, &mut res.to_backend_mut(), j, &vec_znx_backend_ref::<BE>(a), j);
    }
    res
}

/// Normalizes a `VecZnxBig` result down to coefficients and downloads it.
fn settle_big<BE>(
    module_host: &Module<HostBytesBackend>,
    module: &Module<BE>,
    a: &VecZnxBigOwned<BE>,
    cols: usize,
    size: usize,
    base2k: usize,
) -> VecZnx<Vec<u8>, i64>
where
    BE: crate::test_suite::TestBackend,
    Module<BE>: VecZnxBigNormalize<BE> + VecZnxBigNormalizeTmpBytes,
{
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.vec_znx_big_normalize_tmp_bytes());
    let mut out = upload_vec_znx::<BE>(&module_host.vec_znx_alloc(cols, size));
    for j in 0..cols {
        module.vec_znx_big_normalize(
            &mut vec_znx_backend_mut::<BE>(&mut out),
            base2k,
            0,
            j,
            &a.to_backend_ref(),
            base2k,
            j,
            &mut scratch.arena(),
        );
    }
    download_vec_znx::<BE>(&out)
}

/// Inverse-transforms a `VecZnxDft` result, then settles it.
fn settle_dft<BE>(
    module_host: &Module<HostBytesBackend>,
    module: &Module<BE>,
    a: &mut VecZnxDftOwned<BE>,
    base2k: usize,
) -> VecZnx<Vec<u8>, i64>
where
    BE: crate::test_suite::TestBackend,
    Module<BE>: VecZnxIdftApplyTmpA<BE> + VecZnxBigNormalize<BE> + VecZnxBigNormalizeTmpBytes,
{
    let (cols, size) = (a.cols(), a.size());
    let big = idft_into_alloc(module, a);
    settle_big(module_host, module, &big, cols, size, base2k)
}

/// Guards against a vacuous pass: two backends agreeing on all-zero proves nothing.
fn assert_non_zero(a: &VecZnx<Vec<u8>, i64>, label: &str) {
    let nonzero = (0..a.cols()).any(|j| (0..a.size()).any(|k| a.at(j, k).iter().any(|&c| c != 0)));
    assert!(nonzero, "{label}: product is all zero, the comparison proves nothing");
}

/// Prepares the scalar into this variant's tier, or yields `()` for the `small` tier.
macro_rules! svp_prepared {
    (small, $module:expr, $be:ty, $scalar:expr, $cols:expr) => {
        ()
    };
    (tpol, $module:expr, $be:ty, $scalar:expr, $cols:expr) => {{
        let mut prepared: SvpTPolOwned<$be> = $module.svp_tpol_alloc($cols);
        for j in 0..$cols {
            $module.svp_prepare_tpol(&mut prepared.to_backend_mut(), j, &scalar_znx_backend_ref::<$be>(&$scalar), j);
        }
        prepared
    }};
    (ppol, $module:expr, $be:ty, $scalar:expr, $cols:expr) => {{
        let mut prepared: SvpPPolOwned<$be> = $module.svp_ppol_alloc($cols);
        for j in 0..$cols {
            $module.svp_prepare_ppol(&mut prepared.to_backend_mut(), j, &scalar_znx_backend_ref::<$be>(&$scalar), j);
        }
        prepared
    }};
}

/// The scalar operand this variant passes to its apply method.
macro_rules! svp_scalar {
    (small, $be:ty, $scalar:expr, $prepared:expr) => {
        scalar_znx_backend_ref::<$be>(&$scalar)
    };
    (tpol, $be:ty, $scalar:expr, $prepared:expr) => {
        $prepared.to_backend_ref()
    };
    (ppol, $be:ty, $scalar:expr, $prepared:expr) => {
        $prepared.to_backend_ref()
    };
}

/// The vector operand this variant passes to its apply method.
macro_rules! svp_vector {
    (small, $be:ty, $a:expr, $a_dft:expr) => {
        vec_znx_backend_ref::<$be>(&$a)
    };
    (dft, $be:ty, $a:expr, $a_dft:expr) => {
        $a_dft.to_backend_ref()
    };
}

/// One `svp_apply_<scalar>_<vector>_to_dft` variant, on both backends.
macro_rules! svp_to_dft_test {
    ($name:ident, $method:ident, $scalar:ident, $vector:ident, [$($tier:ident),*], $apply:ident) => {
        pub fn $name<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
            params: &TestParams,
            module_host: &Module<HostBytesBackend>,
            module_ref: &Module<BR>,
            module_test: &Module<BT>,
        ) where
            Module<BR>: $($tier<BR> +)* $apply<BR>
                + VecZnxDftAlloc<BR>
                + VecZnxDftApply<BR>
                + VecZnxBigAlloc<BR>
                + VecZnxIdftApplyTmpA<BR>
                + VecZnxBigNormalize<BR>
                + VecZnxBigNormalizeTmpBytes,
            Module<BT>: $($tier<BT> +)* $apply<BT>
                + VecZnxDftAlloc<BT>
                + VecZnxDftApply<BT>
                + VecZnxBigAlloc<BT>
                + VecZnxIdftApplyTmpA<BT>
                + VecZnxBigNormalize<BT>
                + VecZnxBigNormalizeTmpBytes,
            ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
            ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
        {
            let base2k = params.base2k;
            assert_eq!(module_ref.n(), module_test.n());
            let mut source: Source = Source::new([0u8; 32]);

            let mut scalar = module_host.scalar_znx_alloc(COLS);
            scalar.fill_uniform(base2k, &mut source);
            let scalar_ref = upload_scalar_znx::<BR>(&scalar);
            let scalar_test = upload_scalar_znx::<BT>(&scalar);
            let _prepared_ref = svp_prepared!($scalar, module_ref, BR, scalar_ref, COLS);
            let _prepared_test = svp_prepared!($scalar, module_test, BT, scalar_test, COLS);

            for a_size in SIZES {
                let mut a = module_host.vec_znx_alloc(COLS, a_size);
                a.fill_uniform(base2k, &mut source);
                let a_ref = upload_vec_znx::<BR>(&a);
                let a_test = upload_vec_znx::<BT>(&a);
                let _a_dft_ref = dft_of(module_ref, &a_ref, COLS, a_size);
                let _a_dft_test = dft_of(module_test, &a_test, COLS, a_size);

                for res_size in SIZES {
                    let mut res_ref: VecZnxDftOwned<BR> = module_ref.vec_znx_dft_alloc(COLS, res_size);
                    let mut res_test: VecZnxDftOwned<BT> = module_test.vec_znx_dft_alloc(COLS, res_size);

                    for j in 0..COLS {
                        module_ref.$method(
                            &mut res_ref.to_backend_mut(),
                            j,
                            &svp_scalar!($scalar, BR, scalar_ref, _prepared_ref),
                            j,
                            &svp_vector!($vector, BR, a_ref, _a_dft_ref),
                            j,
                        );
                        module_test.$method(
                            &mut res_test.to_backend_mut(),
                            j,
                            &svp_scalar!($scalar, BT, scalar_test, _prepared_test),
                            j,
                            &svp_vector!($vector, BT, a_test, _a_dft_test),
                            j,
                        );
                    }

                    let got_ref = settle_dft(module_host, module_ref, &mut res_ref, base2k);
                    let got_test = settle_dft(module_host, module_test, &mut res_test, base2k);
                    assert_non_zero(&got_ref, stringify!($name));
                    assert_eq!(got_ref, got_test);
                }
            }
        }
    };
}

/// One `svp_apply_<scalar>_dft_to_dft_assign` variant, on both backends.
macro_rules! svp_assign_test {
    ($name:ident, $method:ident, $scalar:ident, [$($tier:ident),*], $apply:ident) => {
        pub fn $name<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
            params: &TestParams,
            module_host: &Module<HostBytesBackend>,
            module_ref: &Module<BR>,
            module_test: &Module<BT>,
        ) where
            Module<BR>: $($tier<BR> +)* $apply<BR>
                + VecZnxDftAlloc<BR>
                + VecZnxDftApply<BR>
                + VecZnxBigAlloc<BR>
                + VecZnxIdftApplyTmpA<BR>
                + VecZnxBigNormalize<BR>
                + VecZnxBigNormalizeTmpBytes,
            Module<BT>: $($tier<BT> +)* $apply<BT>
                + VecZnxDftAlloc<BT>
                + VecZnxDftApply<BT>
                + VecZnxBigAlloc<BT>
                + VecZnxIdftApplyTmpA<BT>
                + VecZnxBigNormalize<BT>
                + VecZnxBigNormalizeTmpBytes,
            ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
            ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
        {
            let base2k = params.base2k;
            assert_eq!(module_ref.n(), module_test.n());
            let mut source: Source = Source::new([0u8; 32]);

            let mut scalar = module_host.scalar_znx_alloc(COLS);
            scalar.fill_uniform(base2k, &mut source);
            let scalar_ref = upload_scalar_znx::<BR>(&scalar);
            let scalar_test = upload_scalar_znx::<BT>(&scalar);
            let _prepared_ref = svp_prepared!($scalar, module_ref, BR, scalar_ref, COLS);
            let _prepared_test = svp_prepared!($scalar, module_test, BT, scalar_test, COLS);

            for size in SIZES {
                let mut a = module_host.vec_znx_alloc(COLS, size);
                a.fill_uniform(base2k, &mut source);
                let a_ref = upload_vec_znx::<BR>(&a);
                let a_test = upload_vec_znx::<BT>(&a);

                // The accumulator starts as the vector operand, then is multiplied in place.
                let mut res_ref = dft_of(module_ref, &a_ref, COLS, size);
                let mut res_test = dft_of(module_test, &a_test, COLS, size);

                for j in 0..COLS {
                    module_ref.$method(
                        &mut res_ref.to_backend_mut(),
                        j,
                        &svp_scalar!($scalar, BR, scalar_ref, _prepared_ref),
                        j,
                    );
                    module_test.$method(
                        &mut res_test.to_backend_mut(),
                        j,
                        &svp_scalar!($scalar, BT, scalar_test, _prepared_test),
                        j,
                    );
                }

                let got_ref = settle_dft(module_host, module_ref, &mut res_ref, base2k);
                let got_test = settle_dft(module_host, module_test, &mut res_test, base2k);
                assert_non_zero(&got_ref, stringify!($name));
                assert_eq!(got_ref, got_test);
            }
        }
    };
}

/// One `svp_apply_<scalar>_<vector>_to_big` variant, on both backends.
macro_rules! svp_to_big_test {
    ($name:ident, $method:ident, $scalar:ident, $vector:ident, [$($tier:ident),*], $apply:ident) => {
        pub fn $name<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
            params: &TestParams,
            module_host: &Module<HostBytesBackend>,
            module_ref: &Module<BR>,
            module_test: &Module<BT>,
        ) where
            Module<BR>: $($tier<BR> +)* $apply<BR>
                + SvpApplyToBigTmpBytes
                + VecZnxDftAlloc<BR>
                + VecZnxDftApply<BR>
                + VecZnxBigAlloc<BR>
                + VecZnxBigNormalize<BR>
                + VecZnxBigNormalizeTmpBytes,
            Module<BT>: $($tier<BT> +)* $apply<BT>
                + SvpApplyToBigTmpBytes
                + VecZnxDftAlloc<BT>
                + VecZnxDftApply<BT>
                + VecZnxBigAlloc<BT>
                + VecZnxBigNormalize<BT>
                + VecZnxBigNormalizeTmpBytes,
            ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
            ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
        {
            let base2k = params.base2k;
            assert_eq!(module_ref.n(), module_test.n());
            let mut source: Source = Source::new([0u8; 32]);

            let mut scalar = module_host.scalar_znx_alloc(COLS);
            scalar.fill_uniform(base2k, &mut source);
            let scalar_ref = upload_scalar_znx::<BR>(&scalar);
            let scalar_test = upload_scalar_znx::<BT>(&scalar);
            let _prepared_ref = svp_prepared!($scalar, module_ref, BR, scalar_ref, COLS);
            let _prepared_test = svp_prepared!($scalar, module_test, BT, scalar_test, COLS);

            for a_size in SIZES {
                let mut a = module_host.vec_znx_alloc(COLS, a_size);
                a.fill_uniform(base2k, &mut source);
                let a_ref = upload_vec_znx::<BR>(&a);
                let a_test = upload_vec_znx::<BT>(&a);
                let _a_dft_ref = dft_of(module_ref, &a_ref, COLS, a_size);
                let _a_dft_test = dft_of(module_test, &a_test, COLS, a_size);

                for res_size in SIZES {
                    let mut scratch_ref: ScratchOwned<BR> =
                        ScratchOwned::alloc(module_ref.svp_apply_to_big_tmp_bytes(res_size));
                    let mut scratch_test: ScratchOwned<BT> =
                        ScratchOwned::alloc(module_test.svp_apply_to_big_tmp_bytes(res_size));

                    let mut res_ref = module_ref.vec_znx_big_alloc(COLS, res_size);
                    let mut res_test = module_test.vec_znx_big_alloc(COLS, res_size);

                    for j in 0..COLS {
                        module_ref.$method(
                            &mut res_ref.to_backend_mut(),
                            j,
                            &svp_scalar!($scalar, BR, scalar_ref, _prepared_ref),
                            j,
                            &svp_vector!($vector, BR, a_ref, _a_dft_ref),
                            j,
                            &mut scratch_ref.arena(),
                        );
                        module_test.$method(
                            &mut res_test.to_backend_mut(),
                            j,
                            &svp_scalar!($scalar, BT, scalar_test, _prepared_test),
                            j,
                            &svp_vector!($vector, BT, a_test, _a_dft_test),
                            j,
                            &mut scratch_test.arena(),
                        );
                    }

                    let got_ref = settle_big(module_host, module_ref, &res_ref, COLS, res_size, base2k);
                    let got_test = settle_big(module_host, module_test, &res_test, COLS, res_size, base2k);
                    assert_non_zero(&got_ref, stringify!($name));
                    assert_eq!(got_ref, got_test);
                }
            }
        }
    };
}

/// One `svp_apply_<scalar>_<vector>_to_small` variant, on both backends.
macro_rules! svp_to_small_test {
    ($name:ident, $method:ident, $scalar:ident, $vector:ident, [$($tier:ident),*], $apply:ident) => {
        pub fn $name<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
            params: &TestParams,
            module_host: &Module<HostBytesBackend>,
            module_ref: &Module<BR>,
            module_test: &Module<BT>,
        ) where
            Module<BR>: $($tier<BR> +)* $apply<BR> + SvpApplyToSmallTmpBytes + VecZnxDftAlloc<BR> + VecZnxDftApply<BR>,
            Module<BT>: $($tier<BT> +)* $apply<BT> + SvpApplyToSmallTmpBytes + VecZnxDftAlloc<BT> + VecZnxDftApply<BT>,
            ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
            ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
        {
            let base2k = params.base2k;
            assert_eq!(module_ref.n(), module_test.n());
            let mut source: Source = Source::new([0u8; 32]);

            let mut scalar = module_host.scalar_znx_alloc(COLS);
            scalar.fill_uniform(base2k, &mut source);
            let scalar_ref = upload_scalar_znx::<BR>(&scalar);
            let scalar_test = upload_scalar_znx::<BT>(&scalar);
            let _prepared_ref = svp_prepared!($scalar, module_ref, BR, scalar_ref, COLS);
            let _prepared_test = svp_prepared!($scalar, module_test, BT, scalar_test, COLS);

            for a_size in SIZES {
                let mut a = module_host.vec_znx_alloc(COLS, a_size);
                a.fill_uniform(base2k, &mut source);
                let a_ref = upload_vec_znx::<BR>(&a);
                let a_test = upload_vec_znx::<BT>(&a);
                let _a_dft_ref = dft_of(module_ref, &a_ref, COLS, a_size);
                let _a_dft_test = dft_of(module_test, &a_test, COLS, a_size);

                let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(module_ref.svp_apply_to_small_tmp_bytes(a_size));
                let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(module_test.svp_apply_to_small_tmp_bytes(a_size));

                for res_size in SIZES {
                    let template = module_host.vec_znx_alloc(COLS, res_size);
                    let mut res_ref = upload_vec_znx::<BR>(&template);
                    let mut res_test = upload_vec_znx::<BT>(&template);

                    for j in 0..COLS {
                        module_ref.$method(
                            &mut vec_znx_backend_mut::<BR>(&mut res_ref),
                            base2k,
                            0,
                            j,
                            &svp_scalar!($scalar, BR, scalar_ref, _prepared_ref),
                            j,
                            &svp_vector!($vector, BR, a_ref, _a_dft_ref),
                            base2k,
                            j,
                            &mut scratch_ref.arena(),
                        );
                        module_test.$method(
                            &mut vec_znx_backend_mut::<BT>(&mut res_test),
                            base2k,
                            0,
                            j,
                            &svp_scalar!($scalar, BT, scalar_test, _prepared_test),
                            j,
                            &svp_vector!($vector, BT, a_test, _a_dft_test),
                            base2k,
                            j,
                            &mut scratch_test.arena(),
                        );
                    }

                    let got_ref = download_vec_znx::<BR>(&res_ref);
                    let got_test = download_vec_znx::<BT>(&res_test);
                    assert_non_zero(&got_ref, stringify!($name));
                    assert_eq!(got_ref, got_test);
                }
            }
        }
    };
}

svp_to_dft_test!(
    test_svp_apply_small_small_to_dft,
    svp_apply_small_small_to_dft,
    small,
    small,
    [],
    SvpApplySmallSmallToDft
);
svp_to_dft_test!(
    test_svp_apply_small_dft_to_dft,
    svp_apply_small_dft_to_dft,
    small,
    dft,
    [],
    SvpApplySmallDftToDft
);
svp_assign_test!(
    test_svp_apply_small_dft_to_dft_assign,
    svp_apply_small_dft_to_dft_assign,
    small,
    [],
    SvpApplySmallDftToDftAssign
);
svp_to_big_test!(
    test_svp_apply_small_small_to_big,
    svp_apply_small_small_to_big,
    small,
    small,
    [],
    SvpApplySmallSmallToBig
);
svp_to_big_test!(
    test_svp_apply_small_dft_to_big,
    svp_apply_small_dft_to_big,
    small,
    dft,
    [],
    SvpApplySmallDftToBig
);
svp_to_small_test!(
    test_svp_apply_small_small_to_small,
    svp_apply_small_small_to_small,
    small,
    small,
    [],
    SvpApplySmallSmallToSmall
);
svp_to_small_test!(
    test_svp_apply_small_dft_to_small,
    svp_apply_small_dft_to_small,
    small,
    dft,
    [],
    SvpApplySmallDftToSmall
);

mod ppol;
mod tpol;

pub use ppol::*;
pub use tpol::*;
