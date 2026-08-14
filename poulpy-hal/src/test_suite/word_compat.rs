//! Word-compatibility tests: execute declared cross-backend layout
//! compatibility for word-keyed prepared containers.
//!
//! Instantiate via [`cross_backend_test_suite!`](crate::cross_backend_test_suite)
//! with a pair of backends declaring the same `DftWord` (enforced at
//! compile time by the equality bounds on each test function):
//!
//! - The `*_bytes` tests assert **byte-identical** prepared/DFT buffers and
//!   are only valid for exact-arithmetic words (NTT/CRT families), where both
//!   backends must produce the same bytes for the same input.
//! - [`test_word_compat_dft_cross_idft`] asserts **cross-consumption**: a DFT
//!   buffer produced by one backend is consumed by the other. This holds for
//!   every pair that declares [`VecZnxDftLayoutCompatible`](crate::layouts::VecZnxDftLayoutCompatible),
//!   including `f64` FFT backends whose DFT-domain values may differ in final ulps.
//!
//! A backend whose byte layout deviates in any aspect must either mint a new
//! word type or omit the corresponding layout-compatibility marker; these tests
//! make declared compatibility executable.

use super::{
    TestParams, scalar_znx_backend_ref, upload_mat_znx, upload_scalar_znx,
    vec_znx_dft::{dft_of_uploaded_vec_znx, idft_apply_to_host},
};
use crate::layouts::SvpPPolToBackendMut;
use crate::layouts::SvpTPolToBackendMut;
use crate::layouts::VmpPMatToBackendMut;
use crate::layouts::VmpTMatToBackendMut;

use crate::{
    api::{
        ScratchOwnedAlloc, SvpPPolAlloc, SvpPreparePPol, SvpPrepareTPol, SvpTPolAlloc, VecZnxBigAlloc, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxDftAlloc, VecZnxDftApply, VecZnxIdftApply, VmpPMatAlloc, VmpPreparePMat,
        VmpPreparePMatTmpBytes, VmpPrepareTMat, VmpPrepareTMatTmpBytes, VmpTMatAlloc,
    },
    layouts::{
        DataView, FillUniform, HostBytesBackend, MatZnx, MatZnxToBackendRef, Module, ScratchOwned, SvpPPolLayoutCompatible,
        SvpPPolOwned, SvpTPolLayoutCompatible, SvpTPolOwned, VecZnxDftLayoutCompatible, VecZnxDftOwned, VmpPMatLayoutCompatible,
        VmpPMatOwned, VmpTMatLayoutCompatible, VmpTMatOwned,
    },
    source::Source,
};

/// Same input, DFT on each backend: the resulting `VecZnxDft` buffers must be
/// byte-identical. Exact-arithmetic (NTT/CRT) words only.
pub fn test_word_compat_dft_bytes<BA, BB>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_a: &Module<BA>,
    module_b: &Module<BB>,
) where
    BA: crate::test_suite::TestBackend + VecZnxDftLayoutCompatible<BB>,
    BB: crate::test_suite::TestBackend,
    Module<BA>: VecZnxDftAlloc<BA> + VecZnxDftApply<BA>,
    Module<BB>: VecZnxDftAlloc<BB> + VecZnxDftApply<BB>,
{
    let base2k = params.base2k;
    assert_eq!(module_a.n(), module_b.n());
    let cols = 2;
    let mut source = Source::new([0u8; 32]);

    for size in [1, 2, 3, 4] {
        let mut a = module_host.vec_znx_alloc(cols, size);
        a.fill_uniform(base2k, &mut source);
        let dft_a = dft_of_uploaded_vec_znx(module_a, &a, 1, 0);
        let dft_b = dft_of_uploaded_vec_znx(module_b, &a, 1, 0);
        assert!(
            BA::to_host_bytes(&dft_a.data) == BB::to_host_bytes(&dft_b.data),
            "shared DftWord but different DFT buffer bytes (size={size}): one backend violates the word contract"
        );
    }
}

/// Same input, `svp_prepare_ppol` on each backend: the resulting `SvpPPol` buffers
/// must be byte-identical. Exact-arithmetic (NTT/CRT) words only.
pub fn test_word_compat_svp_prepare_ppol_bytes<BA, BB>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_a: &Module<BA>,
    module_b: &Module<BB>,
) where
    BA: crate::test_suite::TestBackend + SvpPPolLayoutCompatible<BB>,
    BB: crate::test_suite::TestBackend,
    Module<BA>: SvpPPolAlloc<BA> + SvpPreparePPol<BA>,
    Module<BB>: SvpPPolAlloc<BB> + SvpPreparePPol<BB>,
{
    let base2k = params.base2k;
    assert_eq!(module_a.n(), module_b.n());
    let cols = 2;
    let mut source = Source::new([0u8; 32]);

    let mut scalar = module_host.scalar_znx_alloc(cols);
    scalar.fill_uniform(base2k, &mut source);
    let scalar_a = upload_scalar_znx::<BA>(&scalar);
    let scalar_b = upload_scalar_znx::<BB>(&scalar);

    let mut svp_a: SvpPPolOwned<BA> = module_a.svp_ppol_alloc(cols);
    let mut svp_b: SvpPPolOwned<BB> = module_b.svp_ppol_alloc(cols);
    for j in 0..cols {
        module_a.svp_prepare_ppol(&mut svp_a.to_backend_mut(), j, &scalar_znx_backend_ref::<BA>(&scalar_a), j);
        module_b.svp_prepare_ppol(&mut svp_b.to_backend_mut(), j, &scalar_znx_backend_ref::<BB>(&scalar_b), j);
    }
    assert!(
        BA::to_host_bytes(&svp_a.data) == BB::to_host_bytes(&svp_b.data),
        "shared DftWord but different SvpPPol buffer bytes: one backend violates the word contract"
    );
}

/// Same input, `svp_prepare_tpol` on each backend: the resulting `SvpTPol` buffers
/// must be byte-identical. Exact-arithmetic (NTT/CRT) words only.
pub fn test_word_compat_svp_prepare_tpol_bytes<BA, BB>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_a: &Module<BA>,
    module_b: &Module<BB>,
) where
    BA: crate::test_suite::TestBackend + SvpTPolLayoutCompatible<BB>,
    BB: crate::test_suite::TestBackend,
    Module<BA>: SvpTPolAlloc<BA> + SvpPrepareTPol<BA>,
    Module<BB>: SvpTPolAlloc<BB> + SvpPrepareTPol<BB>,
{
    let base2k = params.base2k;
    assert_eq!(module_a.n(), module_b.n());
    let cols = 2;
    let mut source = Source::new([0u8; 32]);

    let mut scalar = module_host.scalar_znx_alloc(cols);
    scalar.fill_uniform(base2k, &mut source);
    let scalar_a = upload_scalar_znx::<BA>(&scalar);
    let scalar_b = upload_scalar_znx::<BB>(&scalar);

    let mut svp_a: SvpTPolOwned<BA> = module_a.svp_tpol_alloc(cols);
    let mut svp_b: SvpTPolOwned<BB> = module_b.svp_tpol_alloc(cols);
    for j in 0..cols {
        module_a.svp_prepare_tpol(&mut svp_a.to_backend_mut(), j, &scalar_znx_backend_ref::<BA>(&scalar_a), j);
        module_b.svp_prepare_tpol(&mut svp_b.to_backend_mut(), j, &scalar_znx_backend_ref::<BB>(&scalar_b), j);
    }
    assert!(
        BA::to_host_bytes(&svp_a.data) == BB::to_host_bytes(&svp_b.data),
        "shared DftWord but different SvpTPol buffer bytes: one backend violates the word contract"
    );
}

/// Same input, `vmp_prepare_pmat` on each backend: the resulting `VmpPMat` buffers
/// must be byte-identical. Exact-arithmetic (NTT/CRT) words only.
///
/// A backend pair that packs `VmpPMat` differently must not declare
/// [`VmpPMatLayoutCompatible`] and therefore cannot instantiate this test —
/// notably the accelerated NTT4x30 backends (prime-major planar layout)
/// against the reference backend (block-interleaved q120c): their divergence
/// under the shared `Q120bScalar` word is now prevented by construction, the
/// backends being distinct container types with no `VmpPMat` marker.
pub fn test_word_compat_vmp_prepare_pmat_bytes<BA, BB>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_a: &Module<BA>,
    module_b: &Module<BB>,
) where
    BA: crate::test_suite::TestBackend + VmpPMatLayoutCompatible<BB>,
    BB: crate::test_suite::TestBackend,
    Module<BA>: VmpPMatAlloc<BA> + VmpPreparePMat<BA> + VmpPreparePMatTmpBytes,
    Module<BB>: VmpPMatAlloc<BB> + VmpPreparePMat<BB> + VmpPreparePMatTmpBytes,
    ScratchOwned<BA>: ScratchOwnedAlloc<BA>,
    ScratchOwned<BB>: ScratchOwnedAlloc<BB>,
{
    let base2k = params.base2k;
    assert_eq!(module_a.n(), module_b.n());
    let (rows, cols_in, cols_out, size) = (2, 2, 2, 3);
    let mut source = Source::new([0u8; 32]);

    let mut scratch_a: ScratchOwned<BA> = ScratchOwned::alloc(module_a.vmp_prepare_pmat_tmp_bytes(rows, cols_in, cols_out, size));
    let mut scratch_b: ScratchOwned<BB> = ScratchOwned::alloc(module_b.vmp_prepare_pmat_tmp_bytes(rows, cols_in, cols_out, size));

    let mut mat = module_host.mat_znx_alloc(rows, cols_in, cols_out, size);
    mat.fill_uniform(base2k, &mut source);
    let mat_a = upload_mat_znx::<BA>(&mat);
    let mat_b = upload_mat_znx::<BB>(&mat);

    let mut pmat_a: VmpPMatOwned<BA> = module_a.vmp_pmat_alloc(rows, cols_in, cols_out, size);
    let mut pmat_b: VmpPMatOwned<BB> = module_b.vmp_pmat_alloc(rows, cols_in, cols_out, size);
    module_a.vmp_prepare_pmat(
        &mut pmat_a.to_backend_mut(),
        &<MatZnx<BA::OwnedBuf, BA::ZnxWord> as MatZnxToBackendRef<BA>>::to_backend_ref(&mat_a),
        &mut scratch_a.arena(),
    );
    module_b.vmp_prepare_pmat(
        &mut pmat_b.to_backend_mut(),
        &<MatZnx<BB::OwnedBuf, BB::ZnxWord> as MatZnxToBackendRef<BB>>::to_backend_ref(&mat_b),
        &mut scratch_b.arena(),
    );
    assert!(
        BA::to_host_bytes(pmat_a.data()) == BB::to_host_bytes(pmat_b.data()),
        "shared DftWord but different VmpPMat buffer bytes: one backend violates the word contract"
    );
}

/// Same input, `vmp_prepare_tmat` on each backend: the resulting `VmpTMat` buffers
/// must be byte-identical. Exact-arithmetic (NTT/CRT) words only.
///
/// A backend pair that packs `VmpTMat` differently must not declare
/// [`VmpTMatLayoutCompatible`] and therefore cannot instantiate this test —
/// notably the accelerated NTT4x30 backends (prime-major planar layout)
/// against the reference backend (block-interleaved q120c): their divergence
/// under the shared `Q120bScalar` word is now prevented by construction, the
/// backends being distinct container types with no `VmpTMat` marker.
pub fn test_word_compat_vmp_prepare_tmat_bytes<BA, BB>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_a: &Module<BA>,
    module_b: &Module<BB>,
) where
    BA: crate::test_suite::TestBackend + VmpTMatLayoutCompatible<BB>,
    BB: crate::test_suite::TestBackend,
    Module<BA>: VmpTMatAlloc<BA> + VmpPrepareTMat<BA> + VmpPrepareTMatTmpBytes,
    Module<BB>: VmpTMatAlloc<BB> + VmpPrepareTMat<BB> + VmpPrepareTMatTmpBytes,
    ScratchOwned<BA>: ScratchOwnedAlloc<BA>,
    ScratchOwned<BB>: ScratchOwnedAlloc<BB>,
{
    let base2k = params.base2k;
    assert_eq!(module_a.n(), module_b.n());
    let (rows, cols_in, cols_out, size) = (2, 2, 2, 3);
    let mut source = Source::new([0u8; 32]);

    let mut scratch_a: ScratchOwned<BA> = ScratchOwned::alloc(module_a.vmp_prepare_tmat_tmp_bytes(rows, cols_in, cols_out, size));
    let mut scratch_b: ScratchOwned<BB> = ScratchOwned::alloc(module_b.vmp_prepare_tmat_tmp_bytes(rows, cols_in, cols_out, size));

    let mut mat = module_host.mat_znx_alloc(rows, cols_in, cols_out, size);
    mat.fill_uniform(base2k, &mut source);
    let mat_a = upload_mat_znx::<BA>(&mat);
    let mat_b = upload_mat_znx::<BB>(&mat);

    let mut pmat_a: VmpTMatOwned<BA> = module_a.vmp_tmat_alloc(rows, cols_in, cols_out, size);
    let mut pmat_b: VmpTMatOwned<BB> = module_b.vmp_tmat_alloc(rows, cols_in, cols_out, size);
    module_a.vmp_prepare_tmat(
        &mut pmat_a.to_backend_mut(),
        &<MatZnx<BA::OwnedBuf, BA::ZnxWord> as MatZnxToBackendRef<BA>>::to_backend_ref(&mat_a),
        &mut scratch_a.arena(),
    );
    module_b.vmp_prepare_tmat(
        &mut pmat_b.to_backend_mut(),
        &<MatZnx<BB::OwnedBuf, BB::ZnxWord> as MatZnxToBackendRef<BB>>::to_backend_ref(&mat_b),
        &mut scratch_b.arena(),
    );
    assert!(
        BA::to_host_bytes(pmat_a.data()) == BB::to_host_bytes(pmat_b.data()),
        "shared DftWord but different VmpTMat buffer bytes: one backend violates the word contract"
    );
}

/// Cross-consumption: a DFT buffer produced by one backend is consumed
/// (IDFT + normalize) by the other, in both directions, and must yield the
/// same coefficient-domain result as native consumption. Valid for every
/// shared-word pair, including `f64` FFT backends.
pub fn test_word_compat_dft_cross_idft<BA, BB>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_a: &Module<BA>,
    module_b: &Module<BB>,
) where
    BA: crate::test_suite::TestBackend + VecZnxDftLayoutCompatible<BB>,
    BB: crate::test_suite::TestBackend<OwnedBuf = BA::OwnedBuf, DftWord = BA::DftWord, ZnxWord = BA::ZnxWord>
        + VecZnxDftLayoutCompatible<BA>,
    Module<BA>: VecZnxDftAlloc<BA>
        + VecZnxDftApply<BA>
        + VecZnxBigAlloc<BA>
        + VecZnxIdftApply<BA>
        + VecZnxBigNormalize<BA>
        + VecZnxBigNormalizeTmpBytes,
    Module<BB>: VecZnxDftAlloc<BB>
        + VecZnxDftApply<BB>
        + VecZnxBigAlloc<BB>
        + VecZnxIdftApply<BB>
        + VecZnxBigNormalize<BB>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BA>: ScratchOwnedAlloc<BA>,
    ScratchOwned<BB>: ScratchOwnedAlloc<BB>,
{
    let base2k = params.base2k;
    assert_eq!(module_a.n(), module_b.n());
    let cols = 2;
    let mut source = Source::new([0u8; 32]);
    let mut scratch_a: ScratchOwned<BA> = ScratchOwned::alloc(module_a.vec_znx_big_normalize_tmp_bytes());
    let mut scratch_b: ScratchOwned<BB> = ScratchOwned::alloc(module_b.vec_znx_big_normalize_tmp_bytes());

    for size in [1, 2, 3, 4] {
        let mut a = module_host.vec_znx_alloc(cols, size);
        a.fill_uniform(base2k, &mut source);

        let dft_a = dft_of_uploaded_vec_znx(module_a, &a, 1, 0);
        let dft_b = dft_of_uploaded_vec_znx(module_b, &a, 1, 0);

        // Native consumption on each backend first (the re-tag consumes the
        // buffer), then each backend consuming the other's buffer via the
        // marker-guarded zero-copy re-tag: all four must agree in the
        // coefficient domain.
        let res_aa = idft_apply_to_host(module_a, base2k, &dft_a, size, &mut scratch_a);
        let res_bb = idft_apply_to_host(module_b, base2k, &dft_b, size, &mut scratch_b);
        let dft_ab: VecZnxDftOwned<BB> = dft_a.into_backend::<BB>();
        let dft_ba: VecZnxDftOwned<BA> = dft_b.into_backend::<BA>();
        let res_ab = idft_apply_to_host(module_b, base2k, &dft_ab, size, &mut scratch_b);
        let res_ba = idft_apply_to_host(module_a, base2k, &dft_ba, size, &mut scratch_a);

        assert_eq!(res_aa, res_ab, "consuming A's DFT buffer on B diverges (size={size})");
        assert_eq!(res_bb, res_ba, "consuming B's DFT buffer on A diverges (size={size})");
    }
}
