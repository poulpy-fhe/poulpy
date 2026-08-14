//! Cross-backend tests for the VMP apply matrix.
//!
//! One exported test per `vmp_apply_<matrix>_<vector>_to_<output>` variant, each
//! bounded on that variant alone (plus its tier's prepare and the DFT/normalize
//! machinery used to settle the result). A backend implementing a subset of the
//! matrix can register exactly the tests it supports.

use super::{TestParams, download_vec_znx, upload_mat_znx, upload_vec_znx, vec_znx_backend_mut, vec_znx_backend_ref};
use crate::layouts::MatZnxBackendRef;
use crate::layouts::VecZnxBigOwned;
use crate::layouts::VecZnxBigToBackendMut;
use crate::layouts::VecZnxBigToBackendRef;
use crate::layouts::VecZnxDftOwned;
use crate::layouts::VecZnxDftToBackendMut;
use crate::layouts::VecZnxDftToBackendRef;
use crate::layouts::VmpPMatOwned;
use crate::layouts::VmpPMatToBackendMut;
use crate::layouts::VmpPMatToBackendRef;
use crate::layouts::VmpTMatOwned;
use crate::layouts::VmpTMatToBackendMut;
use crate::layouts::VmpTMatToBackendRef;
use crate::layouts::ZnxView;
use crate::{
    api::{
        ScratchOwnedAlloc, VecZnxBigAlloc, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAlloc, VecZnxDftApply,
        VecZnxIdftApplyTmpA, VmpApplyPMatDftToBig, VmpApplyPMatDftToBigTmpBytes, VmpApplyPMatDftToDft,
        VmpApplyPMatDftToDftAccumulate, VmpApplyPMatDftToDftAccumulateTmpBytes, VmpApplyPMatDftToDftTmpBytes,
        VmpApplyPMatDftToSmall, VmpApplyPMatDftToSmallTmpBytes, VmpApplyPMatSmallToBig, VmpApplyPMatSmallToBigTmpBytes,
        VmpApplyPMatSmallToDft, VmpApplyPMatSmallToDftAccumulate, VmpApplyPMatSmallToDftAccumulateTmpBytes,
        VmpApplyPMatSmallToDftTmpBytes, VmpApplyPMatSmallToSmall, VmpApplyPMatSmallToSmallTmpBytes, VmpApplySmallDftToBig,
        VmpApplySmallDftToBigTmpBytes, VmpApplySmallDftToDft, VmpApplySmallDftToDftAccumulate,
        VmpApplySmallDftToDftAccumulateTmpBytes, VmpApplySmallDftToDftTmpBytes, VmpApplySmallDftToSmall,
        VmpApplySmallDftToSmallTmpBytes, VmpApplySmallSmallToBig, VmpApplySmallSmallToBigTmpBytes, VmpApplySmallSmallToDft,
        VmpApplySmallSmallToDftAccumulate, VmpApplySmallSmallToDftAccumulateTmpBytes, VmpApplySmallSmallToDftTmpBytes,
        VmpApplySmallSmallToSmall, VmpApplySmallSmallToSmallTmpBytes, VmpApplyTMatDftToBig, VmpApplyTMatDftToBigTmpBytes,
        VmpApplyTMatDftToDft, VmpApplyTMatDftToDftAccumulate, VmpApplyTMatDftToDftAccumulateTmpBytes,
        VmpApplyTMatDftToDftTmpBytes, VmpApplyTMatDftToSmall, VmpApplyTMatDftToSmallTmpBytes, VmpApplyTMatSmallToBig,
        VmpApplyTMatSmallToBigTmpBytes, VmpApplyTMatSmallToDft, VmpApplyTMatSmallToDftAccumulate,
        VmpApplyTMatSmallToDftAccumulateTmpBytes, VmpApplyTMatSmallToDftTmpBytes, VmpApplyTMatSmallToSmall,
        VmpApplyTMatSmallToSmallTmpBytes, VmpPMatAlloc, VmpPreparePMat, VmpPreparePMatTmpBytes, VmpPrepareTMat,
        VmpPrepareTMatTmpBytes, VmpTMatAlloc,
    },
    layouts::{Backend, FillUniform, HostBytesBackend, MatZnx, MatZnxToBackendRef, Module, ScratchOwned, VecZnx},
    source::Source,
};

const MAX_COLS: usize = 2;
const MAX_SIZE: usize = 4;

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

fn mat_ref<BE: Backend>(a: &MatZnx<BE::OwnedBuf, BE::ZnxWord>) -> MatZnxBackendRef<'_, BE> {
    <MatZnx<BE::OwnedBuf, BE::ZnxWord> as MatZnxToBackendRef<BE>>::to_backend_ref(a)
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

/// Prepares the matrix into this variant's tier, or yields `()` for the `small` tier.
macro_rules! vmp_prepared {
    (small, $module:expr, $be:ty, $mat:expr, $rows:expr, $cols_in:expr, $cols_out:expr, $size:expr) => {
        ()
    };
    (tmat, $module:expr, $be:ty, $mat:expr, $rows:expr, $cols_in:expr, $cols_out:expr, $size:expr) => {{
        let mut scratch: ScratchOwned<$be> =
            ScratchOwned::alloc($module.vmp_prepare_tmat_tmp_bytes($rows, $cols_in, $cols_out, $size));
        let mut prepared: VmpTMatOwned<$be> = $module.vmp_tmat_alloc($rows, $cols_in, $cols_out, $size);
        $module.vmp_prepare_tmat(&mut prepared.to_backend_mut(), &mat_ref::<$be>(&$mat), &mut scratch.arena());
        prepared
    }};
    (pmat, $module:expr, $be:ty, $mat:expr, $rows:expr, $cols_in:expr, $cols_out:expr, $size:expr) => {{
        let mut scratch: ScratchOwned<$be> =
            ScratchOwned::alloc($module.vmp_prepare_pmat_tmp_bytes($rows, $cols_in, $cols_out, $size));
        let mut prepared: VmpPMatOwned<$be> = $module.vmp_pmat_alloc($rows, $cols_in, $cols_out, $size);
        $module.vmp_prepare_pmat(&mut prepared.to_backend_mut(), &mat_ref::<$be>(&$mat), &mut scratch.arena());
        prepared
    }};
}

/// The matrix operand this variant passes to its apply method.
macro_rules! vmp_matrix {
    (small, $be:ty, $mat:expr, $prepared:expr) => {
        mat_ref::<$be>(&$mat)
    };
    (tmat, $be:ty, $mat:expr, $prepared:expr) => {
        $prepared.to_backend_ref()
    };
    (pmat, $be:ty, $mat:expr, $prepared:expr) => {
        $prepared.to_backend_ref()
    };
}

/// The vector operand this variant passes to its apply method.
macro_rules! vmp_vector {
    (small, $be:ty, $a:expr, $a_dft:expr) => {
        vec_znx_backend_ref::<$be>(&$a)
    };
    (dft, $be:ty, $a:expr, $a_dft:expr) => {
        $a_dft.to_backend_ref()
    };
}

/// Applies the variant. The `dft` vector shapes carry an extra `limb_offset`.
macro_rules! vmp_apply {
    (small, $module:expr, $method:ident, $res:expr, $matrix:expr, $vector:expr, $scratch:expr) => {
        $module.$method(&mut $res, &$matrix, &$vector, &mut $scratch)
    };
    (dft, $module:expr, $method:ident, $res:expr, $matrix:expr, $vector:expr, $scratch:expr) => {
        $module.$method(&mut $res, &$matrix, &$vector, 0, &mut $scratch)
    };
}

/// Applies a `_to_small` variant, whose result carries its own base-2k framing.
macro_rules! vmp_apply_to_small {
    (small, $module:expr, $method:ident, $res:expr, $base2k:expr, $matrix:expr, $vector:expr, $scratch:expr) => {
        $module.$method(&mut $res, $base2k, 0, &$matrix, &$vector, $base2k, &mut $scratch)
    };
    (dft, $module:expr, $method:ident, $res:expr, $base2k:expr, $matrix:expr, $vector:expr, $scratch:expr) => {
        $module.$method(&mut $res, $base2k, 0, &$matrix, &$vector, $base2k, 0, &mut $scratch)
    };
}

/// The result accumulator: zeroed for plain applies, seeded for accumulating ones.
macro_rules! vmp_seed_dft {
    (no, $module_host:expr, $module_ref:expr, $module_test:expr, $base2k:expr, $source:expr, $cols:expr, $size:expr) => {{
        let res_ref: VecZnxDftOwned<BR> = $module_ref.vec_znx_dft_alloc($cols, $size);
        let res_test: VecZnxDftOwned<BT> = $module_test.vec_znx_dft_alloc($cols, $size);
        (res_ref, res_test)
    }};
    (yes, $module_host:expr, $module_ref:expr, $module_test:expr, $base2k:expr, $source:expr, $cols:expr, $size:expr) => {{
        let mut seed = $module_host.vec_znx_alloc($cols, $size);
        seed.fill_uniform($base2k, $source);
        let seed_ref = upload_vec_znx::<BR>(&seed);
        let seed_test = upload_vec_znx::<BT>(&seed);
        (
            dft_of($module_ref, &seed_ref, $cols, $size),
            dft_of($module_test, &seed_test, $cols, $size),
        )
    }};
}

/// Shared preamble: operands, prepared matrix, and per-backend scratch.
macro_rules! vmp_operands {
    ($module_host:expr, $module_ref:expr, $module_test:expr, $matrix:ident, $base2k:expr, $source:expr,
     $rows:expr, $cols_in:expr, $cols_out:expr, $size_in:expr, $size_out:expr) => {{
        let mut a = $module_host.vec_znx_alloc($cols_in, $size_in);
        a.fill_uniform($base2k, $source);
        let a_ref = upload_vec_znx::<BR>(&a);
        let a_test = upload_vec_znx::<BT>(&a);

        let mut mat = $module_host.mat_znx_alloc($rows, $cols_in, $cols_out, $size_out);
        mat.fill_uniform($base2k, $source);
        let mat_ref_backend = upload_mat_znx::<BR>(&mat);
        let mat_test_backend = upload_mat_znx::<BT>(&mat);

        let prepared_ref = vmp_prepared!(
            $matrix,
            $module_ref,
            BR,
            mat_ref_backend,
            $rows,
            $cols_in,
            $cols_out,
            $size_out
        );
        let prepared_test = vmp_prepared!(
            $matrix,
            $module_test,
            BT,
            mat_test_backend,
            $rows,
            $cols_in,
            $cols_out,
            $size_out
        );

        (a_ref, a_test, mat_ref_backend, mat_test_backend, prepared_ref, prepared_test)
    }};
}

/// One `vmp_apply_<matrix>_<vector>_to_dft` variant, on both backends.
///
/// `accumulate: yes` seeds the result with a random DFT vector first, so the
/// fused accumulate variants are exercised on a non-zero accumulator.
macro_rules! vmp_to_dft_test {
    ($name:ident, $method:ident, $tmp_bytes:ident, $matrix:ident, $vector:ident, accumulate: $acc:ident,
     [$($t1:ident),*], [$($t0:ident),*]) => {
        pub fn $name<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
            params: &TestParams,
            module_host: &Module<HostBytesBackend>,
            module_ref: &Module<BR>,
            module_test: &Module<BT>,
        ) where
            Module<BR>: $($t1<BR> +)* $($t0 +)*
                VecZnxDftAlloc<BR>
                + VecZnxDftApply<BR>
                + VecZnxBigAlloc<BR>
                + VecZnxIdftApplyTmpA<BR>
                + VecZnxBigNormalize<BR>
                + VecZnxBigNormalizeTmpBytes,
            Module<BT>: $($t1<BT> +)* $($t0 +)*
                VecZnxDftAlloc<BT>
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

            for cols_in in 1..MAX_COLS + 1 {
                for cols_out in 1..MAX_COLS + 1 {
                    for size_in in 1..MAX_SIZE + 1 {
                        for size_out in 1..MAX_SIZE + 1 {
                            let rows = size_in;
                            let (a_ref, a_test, _mat_ref_backend, _mat_test_backend, _prepared_ref, _prepared_test) =
                                vmp_operands!(
                                    module_host, module_ref, module_test, $matrix, base2k, &mut source,
                                    rows, cols_in, cols_out, size_in, size_out
                                );
                            let _a_dft_ref = dft_of(module_ref, &a_ref, cols_in, size_in);
                            let _a_dft_test = dft_of(module_test, &a_test, cols_in, size_in);

                            let (mut res_ref, mut res_test) = vmp_seed_dft!(
                                $acc, module_host, module_ref, module_test, base2k, &mut source, cols_out, size_out
                            );

                            let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(
                                module_ref.$tmp_bytes(size_out, rows, cols_in, cols_out, size_out, size_in),
                            );
                            let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(
                                module_test.$tmp_bytes(size_out, rows, cols_in, cols_out, size_out, size_in),
                            );

                            vmp_apply!(
                                $vector, module_ref, $method,
                                res_ref.to_backend_mut(),
                                vmp_matrix!($matrix, BR, _mat_ref_backend, _prepared_ref),
                                vmp_vector!($vector, BR, a_ref, _a_dft_ref),
                                scratch_ref.arena()
                            );
                            vmp_apply!(
                                $vector, module_test, $method,
                                res_test.to_backend_mut(),
                                vmp_matrix!($matrix, BT, _mat_test_backend, _prepared_test),
                                vmp_vector!($vector, BT, a_test, _a_dft_test),
                                scratch_test.arena()
                            );

                            let got_ref = settle_dft(module_host, module_ref, &mut res_ref, base2k);
                            let got_test = settle_dft(module_host, module_test, &mut res_test, base2k);
                            assert_non_zero(&got_ref, stringify!($name));
                            assert_eq!(got_ref, got_test);
                        }
                    }
                }
            }
        }
    };
}

/// One `vmp_apply_<matrix>_<vector>_to_big` variant, on both backends.
macro_rules! vmp_to_big_test {
    ($name:ident, $method:ident, $tmp_bytes:ident, $matrix:ident, $vector:ident, [$($t1:ident),*], [$($t0:ident),*]) => {
        pub fn $name<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
            params: &TestParams,
            module_host: &Module<HostBytesBackend>,
            module_ref: &Module<BR>,
            module_test: &Module<BT>,
        ) where
            Module<BR>: $($t1<BR> +)* $($t0 +)*
                VecZnxDftAlloc<BR>
                + VecZnxDftApply<BR>
                + VecZnxBigAlloc<BR>
                + VecZnxBigNormalize<BR>
                + VecZnxBigNormalizeTmpBytes,
            Module<BT>: $($t1<BT> +)* $($t0 +)*
                VecZnxDftAlloc<BT>
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

            for cols_in in 1..MAX_COLS + 1 {
                for cols_out in 1..MAX_COLS + 1 {
                    for size_in in 1..MAX_SIZE + 1 {
                        for size_out in 1..MAX_SIZE + 1 {
                            let rows = size_in;
                            let (a_ref, a_test, _mat_ref_backend, _mat_test_backend, _prepared_ref, _prepared_test) =
                                vmp_operands!(
                                    module_host, module_ref, module_test, $matrix, base2k, &mut source,
                                    rows, cols_in, cols_out, size_in, size_out
                                );
                            let _a_dft_ref = dft_of(module_ref, &a_ref, cols_in, size_in);
                            let _a_dft_test = dft_of(module_test, &a_test, cols_in, size_in);

                            let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(
                                module_ref.$tmp_bytes(size_out, rows, cols_in, cols_out, size_out, size_in),
                            );
                            let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(
                                module_test.$tmp_bytes(size_out, rows, cols_in, cols_out, size_out, size_in),
                            );

                            let mut res_ref = module_ref.vec_znx_big_alloc(cols_out, size_out);
                            let mut res_test = module_test.vec_znx_big_alloc(cols_out, size_out);

                            vmp_apply!(
                                $vector, module_ref, $method,
                                res_ref.to_backend_mut(),
                                vmp_matrix!($matrix, BR, _mat_ref_backend, _prepared_ref),
                                vmp_vector!($vector, BR, a_ref, _a_dft_ref),
                                scratch_ref.arena()
                            );
                            vmp_apply!(
                                $vector, module_test, $method,
                                res_test.to_backend_mut(),
                                vmp_matrix!($matrix, BT, _mat_test_backend, _prepared_test),
                                vmp_vector!($vector, BT, a_test, _a_dft_test),
                                scratch_test.arena()
                            );

                            let got_ref = settle_big(module_host, module_ref, &res_ref, cols_out, size_out, base2k);
                            let got_test = settle_big(module_host, module_test, &res_test, cols_out, size_out, base2k);
                            assert_non_zero(&got_ref, stringify!($name));
                            assert_eq!(got_ref, got_test);
                        }
                    }
                }
            }
        }
    };
}

/// One `vmp_apply_<matrix>_<vector>_to_small` variant, on both backends.
macro_rules! vmp_to_small_test {
    ($name:ident, $method:ident, $tmp_bytes:ident, $matrix:ident, $vector:ident, [$($t1:ident),*], [$($t0:ident),*]) => {
        pub fn $name<BR: crate::test_suite::TestBackend, BT: crate::test_suite::TestBackend>(
            params: &TestParams,
            module_host: &Module<HostBytesBackend>,
            module_ref: &Module<BR>,
            module_test: &Module<BT>,
        ) where
            Module<BR>: $($t1<BR> +)* $($t0 +)* VecZnxDftAlloc<BR> + VecZnxDftApply<BR>,
            Module<BT>: $($t1<BT> +)* $($t0 +)* VecZnxDftAlloc<BT> + VecZnxDftApply<BT>,
            ScratchOwned<BR>: ScratchOwnedAlloc<BR>,
            ScratchOwned<BT>: ScratchOwnedAlloc<BT>,
        {
            let base2k = params.base2k;
            assert_eq!(module_ref.n(), module_test.n());
            let mut source: Source = Source::new([0u8; 32]);

            for cols_in in 1..MAX_COLS + 1 {
                for cols_out in 1..MAX_COLS + 1 {
                    for size_in in 1..MAX_SIZE + 1 {
                        for size_out in 1..MAX_SIZE + 1 {
                            let rows = size_in;
                            let (a_ref, a_test, _mat_ref_backend, _mat_test_backend, _prepared_ref, _prepared_test) =
                                vmp_operands!(
                                    module_host, module_ref, module_test, $matrix, base2k, &mut source,
                                    rows, cols_in, cols_out, size_in, size_out
                                );
                            let _a_dft_ref = dft_of(module_ref, &a_ref, cols_in, size_in);
                            let _a_dft_test = dft_of(module_test, &a_test, cols_in, size_in);

                            let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(
                                module_ref.$tmp_bytes(size_out, rows, cols_in, cols_out, size_out, size_in),
                            );
                            let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(
                                module_test.$tmp_bytes(size_out, rows, cols_in, cols_out, size_out, size_in),
                            );

                            let template = module_host.vec_znx_alloc(cols_out, size_out);
                            let mut res_ref = upload_vec_znx::<BR>(&template);
                            let mut res_test = upload_vec_znx::<BT>(&template);

                            vmp_apply_to_small!(
                                $vector, module_ref, $method,
                                vec_znx_backend_mut::<BR>(&mut res_ref), base2k,
                                vmp_matrix!($matrix, BR, _mat_ref_backend, _prepared_ref),
                                vmp_vector!($vector, BR, a_ref, _a_dft_ref),
                                scratch_ref.arena()
                            );
                            vmp_apply_to_small!(
                                $vector, module_test, $method,
                                vec_znx_backend_mut::<BT>(&mut res_test), base2k,
                                vmp_matrix!($matrix, BT, _mat_test_backend, _prepared_test),
                                vmp_vector!($vector, BT, a_test, _a_dft_test),
                                scratch_test.arena()
                            );

                            let got_ref = download_vec_znx::<BR>(&res_ref);
                            let got_test = download_vec_znx::<BT>(&res_test);
                            assert_non_zero(&got_ref, stringify!($name));
                            assert_eq!(got_ref, got_test);
                        }
                    }
                }
            }
        }
    };
}

vmp_to_dft_test!(
    test_vmp_apply_small_small_to_dft,
    vmp_apply_small_small_to_dft,
    vmp_apply_small_small_to_dft_tmp_bytes,
    small,
    small,
    accumulate: no,
    [VmpApplySmallSmallToDft],
    [VmpApplySmallSmallToDftTmpBytes]
);
vmp_to_dft_test!(
    test_vmp_apply_small_dft_to_dft,
    vmp_apply_small_dft_to_dft,
    vmp_apply_small_dft_to_dft_tmp_bytes,
    small,
    dft,
    accumulate: no,
    [VmpApplySmallDftToDft],
    [VmpApplySmallDftToDftTmpBytes]
);
vmp_to_dft_test!(
    test_vmp_apply_small_small_to_dft_accumulate,
    vmp_apply_small_small_to_dft_accumulate,
    vmp_apply_small_small_to_dft_accumulate_tmp_bytes,
    small,
    small,
    accumulate: yes,
    [VmpApplySmallSmallToDftAccumulate],
    [VmpApplySmallSmallToDftAccumulateTmpBytes]
);
vmp_to_dft_test!(
    test_vmp_apply_small_dft_to_dft_accumulate,
    vmp_apply_small_dft_to_dft_accumulate,
    vmp_apply_small_dft_to_dft_accumulate_tmp_bytes,
    small,
    dft,
    accumulate: yes,
    [VmpApplySmallDftToDftAccumulate],
    [VmpApplySmallDftToDftAccumulateTmpBytes]
);
vmp_to_big_test!(
    test_vmp_apply_small_small_to_big,
    vmp_apply_small_small_to_big,
    vmp_apply_small_small_to_big_tmp_bytes,
    small,
    small,
    [VmpApplySmallSmallToBig],
    [VmpApplySmallSmallToBigTmpBytes]
);
vmp_to_big_test!(
    test_vmp_apply_small_dft_to_big,
    vmp_apply_small_dft_to_big,
    vmp_apply_small_dft_to_big_tmp_bytes,
    small,
    dft,
    [VmpApplySmallDftToBig],
    [VmpApplySmallDftToBigTmpBytes]
);
vmp_to_small_test!(
    test_vmp_apply_small_small_to_small,
    vmp_apply_small_small_to_small,
    vmp_apply_small_small_to_small_tmp_bytes,
    small,
    small,
    [VmpApplySmallSmallToSmall],
    [VmpApplySmallSmallToSmallTmpBytes]
);
vmp_to_small_test!(
    test_vmp_apply_small_dft_to_small,
    vmp_apply_small_dft_to_small,
    vmp_apply_small_dft_to_small_tmp_bytes,
    small,
    dft,
    [VmpApplySmallDftToSmall],
    [VmpApplySmallDftToSmallTmpBytes]
);

mod pmat;
mod tmat;

pub use pmat::*;
pub use tmat::*;
