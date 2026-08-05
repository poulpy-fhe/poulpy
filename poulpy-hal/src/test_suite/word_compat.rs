//! Word-compatibility tests: execute the word contract of
//! [`DftWord`](crate::layouts::DftWord) — type equality of word-keyed
//! containers means buffer interchangeability across backends.
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
//!   every shared-word pair, including `f64` FFT backends whose DFT-domain
//!   values may differ in final ulps.
//!
//! A backend whose byte layout deviates in any aspect must mint a new word
//! type; these tests are what makes that rule executable.

use super::{
    TestParams, scalar_znx_backend_ref, upload_mat_znx, upload_scalar_znx,
    vec_znx_dft::{dft_of_uploaded_vec_znx, idft_apply_to_host},
};

use crate::{
    api::{
        ScratchOwnedAlloc, SvpPPolAlloc, SvpPrepare, VecZnxBigAlloc, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxDftAlloc, VecZnxDftApply, VecZnxIdftApply, VmpPMatAlloc, VmpPrepare, VmpPrepareTmpBytes,
    },
    layouts::{
        Backend, FillUniform, HostBytesBackend, MatZnx, MatZnxToBackendRef, Module, ScalarZnx, ScratchOwned, SvpPPolOwned,
        VecZnx, VmpPMatOwned,
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
    BA: Backend,
    BB: Backend<OwnedBuf = BA::OwnedBuf, DftWord = BA::DftWord>,
    Module<BA>: VecZnxDftAlloc<BA> + VecZnxDftApply<BA>,
    Module<BB>: VecZnxDftAlloc<BB> + VecZnxDftApply<BB>,
{
    let base2k = params.base2k;
    assert_eq!(module_a.n(), module_b.n());
    let cols = 2;
    let mut source = Source::new([0u8; 32]);

    for size in [1, 2, 3, 4] {
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, size);
        a.fill_uniform(base2k, &mut source);
        let dft_a = dft_of_uploaded_vec_znx(module_a, &a, 1, 0);
        let dft_b = dft_of_uploaded_vec_znx(module_b, &a, 1, 0);
        assert!(
            dft_a == dft_b,
            "shared DftWord but different DFT buffer bytes (size={size}): one backend violates the word contract"
        );
    }
}

/// Same input, `svp_prepare` on each backend: the resulting `SvpPPol` buffers
/// must be byte-identical. Exact-arithmetic (NTT/CRT) words only.
pub fn test_word_compat_svp_prepare_bytes<BA, BB>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_a: &Module<BA>,
    module_b: &Module<BB>,
) where
    BA: Backend,
    BB: Backend<OwnedBuf = BA::OwnedBuf, DftWord = BA::DftWord>,
    Module<BA>: SvpPPolAlloc<BA> + SvpPrepare<BA>,
    Module<BB>: SvpPPolAlloc<BB> + SvpPrepare<BB>,
{
    let base2k = params.base2k;
    assert_eq!(module_a.n(), module_b.n());
    let cols = 2;
    let mut source = Source::new([0u8; 32]);

    let mut scalar: ScalarZnx<Vec<u8>> = module_host.scalar_znx_alloc(cols);
    scalar.fill_uniform(base2k, &mut source);
    let scalar_a = upload_scalar_znx::<BA>(&scalar);
    let scalar_b = upload_scalar_znx::<BB>(&scalar);

    let mut svp_a: SvpPPolOwned<BA> = module_a.svp_ppol_alloc(cols);
    let mut svp_b: SvpPPolOwned<BB> = module_b.svp_ppol_alloc(cols);
    for j in 0..cols {
        module_a.svp_prepare(
            &mut svp_a.to_backend_mut::<BA>(),
            j,
            &scalar_znx_backend_ref::<BA>(&scalar_a),
            j,
        );
        module_b.svp_prepare(
            &mut svp_b.to_backend_mut::<BB>(),
            j,
            &scalar_znx_backend_ref::<BB>(&scalar_b),
            j,
        );
    }
    assert!(
        svp_a == svp_b,
        "shared DftWord but different SvpPPol buffer bytes: one backend violates the word contract"
    );
}

/// Same input, `vmp_prepare` on each backend: the resulting `VmpPMat` buffers
/// must be byte-identical. Exact-arithmetic (NTT/CRT) words only.
///
/// A backend that packs its `VmpPMat` differently (e.g. via a
/// `bytes_of_vmp_pmat` override) must declare its own word type and therefore
/// cannot be paired here — that is the contract, not a limitation.
///
/// KNOWN VIOLATION: the accelerated NTT4x30 backends (AVX/AVX-512/NEON)
/// prepare `VmpPMat` in a prime-major planar layout while the reference
/// backend uses block-interleaved q120c, all under the shared `Q120bScalar`
/// word. Their instantiations of this test are `#[ignore]`d until `VmpPMat`
/// gets its own per-backend prepared-word declaration.
pub fn test_word_compat_vmp_prepare_bytes<BA, BB>(
    params: &TestParams,
    module_host: &Module<HostBytesBackend>,
    module_a: &Module<BA>,
    module_b: &Module<BB>,
) where
    BA: Backend,
    BB: Backend<OwnedBuf = BA::OwnedBuf, DftWord = BA::DftWord>,
    Module<BA>: VmpPMatAlloc<BA> + VmpPrepare<BA> + VmpPrepareTmpBytes,
    Module<BB>: VmpPMatAlloc<BB> + VmpPrepare<BB> + VmpPrepareTmpBytes,
    ScratchOwned<BA>: ScratchOwnedAlloc<BA>,
    ScratchOwned<BB>: ScratchOwnedAlloc<BB>,
{
    let base2k = params.base2k;
    assert_eq!(module_a.n(), module_b.n());
    let (rows, cols_in, cols_out, size) = (2, 2, 2, 3);
    let mut source = Source::new([0u8; 32]);

    let mut scratch_a: ScratchOwned<BA> = ScratchOwned::alloc(module_a.vmp_prepare_tmp_bytes(rows, cols_in, cols_out, size));
    let mut scratch_b: ScratchOwned<BB> = ScratchOwned::alloc(module_b.vmp_prepare_tmp_bytes(rows, cols_in, cols_out, size));

    let mut mat: MatZnx<Vec<u8>> = module_host.mat_znx_alloc(rows, cols_in, cols_out, size);
    mat.fill_uniform(base2k, &mut source);
    let mat_a = upload_mat_znx::<BA>(&mat);
    let mat_b = upload_mat_znx::<BB>(&mat);

    let mut pmat_a: VmpPMatOwned<BA> = module_a.vmp_pmat_alloc(rows, cols_in, cols_out, size);
    let mut pmat_b: VmpPMatOwned<BB> = module_b.vmp_pmat_alloc(rows, cols_in, cols_out, size);
    module_a.vmp_prepare(
        &mut pmat_a.to_backend_mut::<BA>(),
        &<MatZnx<BA::OwnedBuf> as MatZnxToBackendRef<BA>>::to_backend_ref(&mat_a),
        &mut scratch_a.arena(),
    );
    module_b.vmp_prepare(
        &mut pmat_b.to_backend_mut::<BB>(),
        &<MatZnx<BB::OwnedBuf> as MatZnxToBackendRef<BB>>::to_backend_ref(&mat_b),
        &mut scratch_b.arena(),
    );
    assert!(
        pmat_a == pmat_b,
        "shared DftWord but different VmpPMat buffer bytes: one backend violates the word contract"
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
    BA: Backend,
    BB: Backend<OwnedBuf = BA::OwnedBuf, DftWord = BA::DftWord>,
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
        let mut a: VecZnx<Vec<u8>> = module_host.vec_znx_alloc(cols, size);
        a.fill_uniform(base2k, &mut source);

        let dft_a = dft_of_uploaded_vec_znx(module_a, &a, 1, 0);
        let dft_b = dft_of_uploaded_vec_znx(module_b, &a, 1, 0);

        // Native consumption on each backend, then each backend consuming the
        // other's buffer: all four must agree in the coefficient domain.
        let res_aa = idft_apply_to_host(module_a, base2k, &dft_a, size, &mut scratch_a);
        let res_ab = idft_apply_to_host(module_b, base2k, &dft_a, size, &mut scratch_b);
        let res_bb = idft_apply_to_host(module_b, base2k, &dft_b, size, &mut scratch_b);
        let res_ba = idft_apply_to_host(module_a, base2k, &dft_b, size, &mut scratch_a);

        assert_eq!(res_aa, res_ab, "consuming A's DFT buffer on B diverges (size={size})");
        assert_eq!(res_bb, res_ba, "consuming B's DFT buffer on A diverges (size={size})");
    }
}
