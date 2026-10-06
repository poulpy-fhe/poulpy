use super::{DelegatingFFT64Portable, FFT64Portable};
use poulpy_core::test_suite::parity::{ParityShapes, test_glwe_external_product_parity};
use poulpy_hal::{layouts::Module, test_suite::TestParams};
use std::cell::Cell;

thread_local! {
    static DFT_CALLS: Cell<usize> = const { Cell::new(0) };
}
const DFT_EXTRA_SCRATCH: usize = 4096;

unsafe impl poulpy_core::oep::GLWEExternalProductImpl for DelegatingFFT64Portable {
    fn glwe_external_product_internal_tmp_bytes<R, A, B>(
        module: &::poulpy_hal::layouts::Module<Self>,
        res_infos: &R,
        a_infos: &A,
        b_infos: &B,
    ) -> usize
    where
        R: poulpy_core::layouts::GLWEInfos,
        A: poulpy_core::layouts::GLWEInfos,
        B: poulpy_core::layouts::GGSWInfos,
    {
        DFT_EXTRA_SCRATCH + poulpy_core::reference::external_product::glwe::GLWEExternalProductDftReference::glwe_external_product_internal_tmp_bytes_reference(module, res_infos, a_infos, b_infos)
    }

    fn glwe_external_product_dft<'r, A>(
        module: &::poulpy_hal::layouts::Module<Self>,
        res_dft: &mut poulpy_hal::layouts::VecZnxDftBackendMut<'r, Self>,
        a: &A,
        ggsw: &poulpy_core::layouts::GGSWPreparedBackendRef<'_, Self>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
    ) where
        A: poulpy_core::layouts::GLWEToBackendRef<Self>,
    {
        DFT_CALLS.set(DFT_CALLS.get() + 1);
        let (marker, mut remaining) = scratch.borrow().take_region(DFT_EXTRA_SCRATCH);
        marker.fill(0xA6);
        poulpy_core::reference::external_product::glwe::GLWEExternalProductDftReference::glwe_external_product_dft_reference(
            module,
            res_dft,
            a,
            ggsw,
            &mut remaining,
        );
        assert!(marker.iter().all(|&byte| byte == 0xA6));
    }
    fn glwe_external_product_tmp_bytes<R, A, G>(
        module: &::poulpy_hal::layouts::Module<DelegatingFFT64Portable>,
        res_infos: &R,
        a_infos: &A,
        ggsw_infos: &G,
    ) -> usize
    where
        R: poulpy_core::layouts::GLWEInfos,
        A: poulpy_core::layouts::GLWEInfos,
        G: poulpy_core::layouts::GGSWInfos,
    {
        poulpy_core::reference::external_product::glwe::glwe_external_product_tmp_bytes_reference::<
            DelegatingFFT64Portable,
            _,
            _,
            _,
            _,
        >(module, res_infos, a_infos, ggsw_infos)
    }

    fn glwe_external_product<R, A>(
        module: &::poulpy_hal::layouts::Module<DelegatingFFT64Portable>,
        res: &mut R,
        a: &A,
        ggsw: &poulpy_core::layouts::prepared::GGSWPreparedBackendRef<'_, DelegatingFFT64Portable>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<DelegatingFFT64Portable>,
    ) where
        R: poulpy_core::layouts::GLWEToBackendMut<DelegatingFFT64Portable> + poulpy_core::layouts::GLWEInfos,
        A: poulpy_core::layouts::GLWEToBackendRef<DelegatingFFT64Portable> + poulpy_core::layouts::GLWEInfos,
    {
        let before = DFT_CALLS.get();
        poulpy_core::reference::external_product::glwe::glwe_external_product_reference::<DelegatingFFT64Portable, _, _, _>(
            module, res, a, ggsw, scratch,
        );
        assert_eq!(DFT_CALLS.get(), before + 1);
    }

    fn glwe_external_product_assign<R>(
        module: &::poulpy_hal::layouts::Module<DelegatingFFT64Portable>,
        res: &mut R,
        ggsw: &poulpy_core::layouts::prepared::GGSWPreparedBackendRef<'_, DelegatingFFT64Portable>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<DelegatingFFT64Portable>,
    ) where
        R: poulpy_core::layouts::GLWEToBackendMut<DelegatingFFT64Portable> + poulpy_core::layouts::GLWEInfos,
    {
        let before = DFT_CALLS.get();
        poulpy_core::reference::external_product::glwe::glwe_external_product_assign_reference::<DelegatingFFT64Portable, _, _>(
            module, res, ggsw, scratch,
        );
        assert_eq!(DFT_CALLS.get(), before + 1);
    }
}

#[test]
fn external_product_dispatches_dft_with_selected_scratch() {
    DFT_CALLS.set(0);
    test_glwe_external_product_parity(
        &TestParams {
            size: 64,
            n: 64,
            base2k: 17,
        },
        &ParityShapes {
            ranks: vec![1, 2],
            dsizes: Some(vec![1, 2, 3]),
        },
        &Module::<FFT64Portable>::new(64),
        &Module::<DelegatingFFT64Portable>::new(64),
    );
    assert!(DFT_CALLS.get() > 0);
}
