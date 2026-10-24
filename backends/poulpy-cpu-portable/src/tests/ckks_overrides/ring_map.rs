//! The conjugate-invariant ring maps enter the selected core maps, are sized by
//! their queries and match the portable backend.
use super::OverrideBackend;
use super::carry::with_marked_region;
use crate::{FFT64CIPortable, FFT64Portable};
use poulpy_ckks::{
    SlotsKind,
    api::CKKSCIRingMapOps,
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc},
    test_suite::{
        CKKSTestParams,
        parity::{
            helpers::{Snapshot, fixture_ciphertext, snapshot, with_scratch},
            layout,
        },
    },
};
use poulpy_core::{
    GLWEBytesOf, GLWEKeyswitch,
    api::GLWECIKeyPreparedFactory,
    layouts::{
        GLWECIEmbedKeyPrepared, GLWECITraceKeyPrepared, GLWEInfos, GLWESecretToBackendMut, GLWESecretToBackendRef,
        GLWEToBackendMut, GLWEToBackendRef, ModuleCoreAlloc,
    },
};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, ScratchArena, ScratchOwned},
};
use std::cell::Cell;

thread_local! {
    static CI_EMBED_CALLS: Cell<usize> = const { Cell::new(0) };
    static CI_TRACE_CALLS: Cell<usize> = const { Cell::new(0) };
}
const CI_TRACE_EXTRA: usize = 640;

unsafe impl poulpy_hal::oep::HalVecZnxCIImpl for OverrideBackend {
    crate::hal_impl_vec_znx_ci!();
}

unsafe impl poulpy_core::oep::GLWECIConversionImpl for OverrideBackend {
    fn glwe_ci_embed<R, A>(module: &Module<Self>, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos,
    {
        CI_EMBED_CALLS.set(CI_EMBED_CALLS.get() + 1);
        poulpy_core::reference::ci_conversion::glwe_ci_embed_reference::<Self, _, _, _>(module, res, a)
    }

    fn glwe_secret_ci_embed<R, S>(module: &Module<Self>, res: &mut R, src: &S)
    where
        R: GLWESecretToBackendMut<Self> + poulpy_core::GetDistributionMut,
        S: GLWESecretToBackendRef<Self>,
    {
        poulpy_core::reference::ci_conversion::glwe_secret_ci_embed_reference::<Self, _, _>(module, res, src)
    }

    fn glwe_ci_trace_tmp_bytes<R, A>(module: &Module<Self>, res_infos: &R, a_infos: &A) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
    {
        CI_TRACE_EXTRA
            + poulpy_core::reference::ci_conversion::glwe_ci_trace_tmp_bytes_reference::<Self, _, _, _>(
                module, res_infos, a_infos,
            )
    }

    fn glwe_ci_trace<R, A>(module: &Module<Self>, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos,
    {
        CI_TRACE_CALLS.set(CI_TRACE_CALLS.get() + 1);
        with_marked_region::<CI_TRACE_EXTRA, _>(scratch, |rest| {
            poulpy_core::reference::ci_conversion::glwe_ci_trace_reference::<Self, _, _, _>(module, res, a, rest)
        })
    }
}
poulpy_ckks::impl_ckks_ring_map_derived_full!(OverrideBackend);

fn params(n: usize) -> CKKSTestParams {
    CKKSTestParams {
        n,
        base2k: 16,
        k: 49,
        prec_meta: Default::default(),
        prec_log_budget: 16,
        hw: 8,
        dsize: 1,
        rank: 1,
    }
}

/// Zero embed and trace keys prepared on `module`: the switch they apply is deterministic.
fn keys<B: Backend>(module: &Module<B>) -> (GLWECIEmbedKeyPrepared<B::OwnedBuf, B>, GLWECITraceKeyPrepared<B::OwnedBuf, B>)
where
    Module<B>: GLWECIKeyPreparedFactory<B> + ModuleCoreAlloc<OwnedBuf = B::OwnedBuf, ZnxWord = B::ZnxWord>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let infos = params(64).ksk_layout(49);
    let mut scratch = ScratchOwned::<B>::alloc(module.glwe_ci_key_prepare_tmp_bytes(&infos));
    let mut embed = module.glwe_ci_embed_key_prepared_alloc_from_infos(&infos);
    module.glwe_ci_key_prepare(
        &mut embed,
        &module.glwe_ci_embed_key_alloc_from_infos(&infos),
        &mut scratch.borrow(),
    );
    let mut trace = module.glwe_ci_trace_key_prepared_alloc_from_infos(&infos);
    module.glwe_ci_key_prepare(
        &mut trace,
        &module.glwe_ci_trace_key_alloc_from_infos(&infos),
        &mut scratch.borrow(),
    );
    (embed, trace)
}

/// Embeds a conjugate-invariant fixture into a standard ciphertext, then traces it back.
fn round_trip<B>(module: &Module<B>, ci: &Module<FFT64CIPortable>) -> (Snapshot, Snapshot)
where
    B: Backend<ZnxWord = i64, OwnedBuf = poulpy_hal::AlignedBuf>,
    Module<B>: CKKSCIRingMapOps<B>
        + GLWECIKeyPreparedFactory<B>
        + ModuleCoreAlloc<OwnedBuf = B::OwnedBuf, ZnxWord = B::ZnxWord>
        + CKKSModuleAlloc<B>,
    ScratchOwned<B>: ScratchOwnedAlloc<B> + ScratchOwnedBorrow<B>,
{
    let (embed_key, trace_key) = keys(module);
    let src = fixture_ciphertext(ci, &layout(params(32), 1, 49, 16, 0, SlotsKind::Real), 51);
    let mut extended: CKKSCiphertextOwned<B> = module.ckks_ciphertext_alloc(16usize.into(), 49usize.into());
    let bytes = module.ckks_ci_embed_tmp_bytes(&extended, &embed_key);
    with_scratch::<B, _>(bytes, |s| module.ckks_ci_embed(&mut extended, &src, &embed_key, s)).unwrap();
    let mut back = ci.ckks_ciphertext_alloc(16usize.into(), 48usize.into());
    let bytes = module.ckks_ci_trace_tmp_bytes(&back, &extended, &trace_key);
    with_scratch::<B, _>(bytes, |s| module.ckks_ci_trace(&mut back, &extended, &trace_key, s)).unwrap();
    (snapshot::<B, _>(&extended), snapshot::<FFT64CIPortable, _>(&back))
}

#[test]
fn ring_maps_enter_the_selected_core_maps() {
    let ci = Module::<FFT64CIPortable>::new(32);
    let reference = Module::<FFT64Portable>::new(64);
    let module = Module::<OverrideBackend>::new(64);

    let (embed_key, trace_key) = keys(&module);
    let extended = module.ckks_ciphertext_alloc(16usize.into(), 49usize.into());
    let back = ci.ckks_ciphertext_alloc(16usize.into(), 48usize.into());
    assert_eq!(
        module.ckks_ci_embed_tmp_bytes(&extended, &embed_key),
        module.glwe_keyswitch_tmp_bytes(&extended, &extended, &embed_key)
    );
    let trace_bytes = module.ckks_ci_trace_tmp_bytes(&back, &extended, &trace_key);
    assert_eq!(
        trace_bytes,
        module.glwe_bytes_of_from_infos(&extended)
            + module
                .glwe_keyswitch_tmp_bytes(&extended, &extended, &trace_key)
                .max(poulpy_core::GLWECITrace::glwe_ci_trace_tmp_bytes(&module, &back, &extended))
    );
    assert!(
        poulpy_core::GLWECITrace::glwe_ci_trace_tmp_bytes(&module, &back, &extended)
            == poulpy_core::GLWECITrace::glwe_ci_trace_tmp_bytes(&reference, &back, &extended) + CI_TRACE_EXTRA
    );

    let want = round_trip(&reference, &ci);
    CI_EMBED_CALLS.set(0);
    CI_TRACE_CALLS.set(0);
    let got = round_trip(&module, &ci);
    assert_eq!(CI_EMBED_CALLS.get(), 1);
    assert_eq!(CI_TRACE_CALLS.get(), 1);
    assert_eq!(got, want);

    // A conjugate-invariant output of the wrong degree is refused before the trace is entered.
    let mut wrong = Module::<FFT64CIPortable>::new(16).ckks_ciphertext_alloc(16usize.into(), 48usize.into());
    let unchanged = snapshot::<FFT64CIPortable, _>(&wrong);
    CI_TRACE_CALLS.set(0);
    assert!(
        with_scratch::<OverrideBackend, _>(trace_bytes, |s| module.ckks_ci_trace(&mut wrong, &extended, &trace_key, s)).is_err()
    );
    assert_eq!(CI_TRACE_CALLS.get(), 0);
    assert_eq!(snapshot::<FFT64CIPortable, _>(&wrong), unchanged);
}
