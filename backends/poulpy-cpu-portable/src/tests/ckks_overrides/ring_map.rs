//! The conjugate-invariant ring maps enter the selected core maps, are sized by
//! their queries and match the oracle.
use super::OverrideBackend;
use super::carry::with_marked_region;
use crate::FFT64CIPortable;
use poulpy_ckks::{
    CKKSCompositionError, SlotsKind,
    api::CKKSCIRingMapOps,
    layouts::CKKSCiphertext,
    test_suite::{
        CKKSTestParams,
        parity::{
            helpers::{Snapshot, assert_untagged, fixture_ciphertext, snapshot, with_scratch},
            layout,
        },
    },
};
use poulpy_core::{
    GLWEBytesOf, GLWECIKeyEncryptSk, GLWEKeyswitch, GLWESecretSampling,
    api::GLWECIKeyPreparedFactory,
    layouts::{
        GLWECIEmbedKey, GLWECIEmbedKeyPrepared, GLWECITraceKey, GLWECITraceKeyPrepared, GLWEInfos, GLWELayout,
        GLWESecretToBackendMut, GLWESecretToBackendRef, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, ModuleCoreAlloc,
    },
};
use poulpy_cpu_oracle::FFT64Oracle;
use poulpy_hal::{
    AlignedBuf,
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, ConjugateInvariant, Module, ScratchArena, ScratchOwned, Standard},
    source::Source,
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
poulpy_ckks::impl_ckks_ci_ring_map_reference_full!(OverrideBackend);

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

/// Coefficient-domain inputs every backend starts from: a uniform conjugate-invariant
/// source, uniform outputs to overwrite and switching keys encrypted on the oracle.
struct Fixtures {
    src: CKKSCiphertext<AlignedBuf, i64, ConjugateInvariant>,
    extended: CKKSCiphertext<AlignedBuf, i64, Standard>,
    back: CKKSCiphertext<AlignedBuf, i64, ConjugateInvariant>,
    embed: GLWECIEmbedKey<AlignedBuf, i64>,
    trace: GLWECITraceKey<AlignedBuf, i64>,
}

fn fixtures(module_ref: &Module<FFT64Oracle>, ci: &Module<FFT64CIPortable>) -> Fixtures {
    let infos = params(64).ksk_layout(49);
    let mut ci_sk = ci.glwe_secret_alloc_from_infos(&params(32).glwe_layout());
    ci.glwe_secret_fill_ternary_hw(&mut ci_sk, 8, &mut Source::new([1; 32]));
    let mut sk = module_ref.glwe_secret_alloc_from_infos(&params(64).glwe_layout());
    module_ref.glwe_secret_fill_ternary_hw(&mut sk, 16, &mut Source::new([2; 32]));
    let (mut xe, mut xa) = (Source::new([3; 32]), Source::new([4; 32]));
    let mut scratch = ScratchOwned::<FFT64Oracle>::alloc(module_ref.glwe_ci_key_encrypt_sk_tmp_bytes(&infos));
    let mut embed = module_ref.glwe_ci_embed_key_alloc_from_infos(&infos);
    module_ref.glwe_ci_embed_key_encrypt_sk(&mut embed, &ci_sk, &sk, &mut xe, &mut xa, &mut scratch.borrow());
    let mut trace = module_ref.glwe_ci_trace_key_alloc_from_infos(&infos);
    module_ref.glwe_ci_trace_key_encrypt_sk(&mut trace, &ci_sk, &sk, &mut xe, &mut xa, &mut scratch.borrow());
    Fixtures {
        src: fixture_ciphertext(ci, &layout(params(32), 1, 49, 16, 0, SlotsKind::Real), 51),
        extended: fixture_ciphertext(module_ref, &layout(params(64), 1, 49, 16, 0, SlotsKind::Complex), 52),
        back: fixture_ciphertext(ci, &layout(params(32), 1, 48, 16, 0, SlotsKind::Complex), 53),
        embed,
        trace,
    }
}

/// The fixture keys prepared on `module`.
fn prepared_keys<BE>(module: &Module<BE>, fixtures: &Fixtures) -> PreparedKeys<BE>
where
    BE: Backend<ZnxWord = i64, OwnedBuf = AlignedBuf>,
    Module<BE>: GLWECIKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let infos = params(64).ksk_layout(49);
    let mut scratch = ScratchOwned::<BE>::alloc(module.glwe_ci_key_prepare_tmp_bytes(&infos));
    let mut embed = module.glwe_ci_embed_key_prepared_alloc_from_infos(&infos);
    module.glwe_ci_key_prepare(&mut embed, &fixtures.embed, &mut scratch.borrow());
    let mut trace = module.glwe_ci_trace_key_prepared_alloc_from_infos(&infos);
    module.glwe_ci_key_prepare(&mut trace, &fixtures.trace, &mut scratch.borrow());
    (embed, trace)
}

type PreparedKeys<BE> = (GLWECIEmbedKeyPrepared<AlignedBuf, BE>, GLWECITraceKeyPrepared<AlignedBuf, BE>);

/// Embeds the source into the standard fixture, then traces it back into the conjugate-invariant one, each map
/// on its own query.
fn round_trip<BE>(module: &Module<BE>, fixtures: &Fixtures) -> (Snapshot, Snapshot)
where
    BE: Backend<ZnxWord = i64, OwnedBuf = AlignedBuf, Ring = Standard>,
    Module<BE>: CKKSCIRingMapOps<BE> + GLWECIKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let (embed_key, trace_key) = prepared_keys(module, fixtures);
    let mut extended = fixtures.extended.clone();
    let bytes = module.ckks_ci_embed_tmp_bytes(&extended, &embed_key);
    with_scratch::<BE, _>(bytes, |s| module.ckks_ci_embed(&mut extended, &fixtures.src, &embed_key, s)).unwrap();
    let mut back = fixtures.back.clone();
    let bytes = module.ckks_ci_trace_tmp_bytes(&back, &extended, &trace_key);
    with_scratch::<BE, _>(bytes, |s| module.ckks_ci_trace(&mut back, &extended, &trace_key, s)).unwrap();
    (snapshot::<BE, _>(&extended), snapshot::<FFT64CIPortable, _>(&back))
}

#[test]
fn ring_maps_enter_the_selected_core_maps() {
    let ci = Module::<FFT64CIPortable>::new(32);
    let module_ref = Module::<FFT64Oracle>::new(64);
    let module_test = Module::<OverrideBackend>::new(64);
    let fixtures = fixtures(&module_ref, &ci);

    let (extended, back) = (&fixtures.extended, &fixtures.back);
    let key_infos = params(64).ksk_layout(49);
    let allocated = GLWELayout {
        n: extended.n(),
        base2k: extended.base2k(),
        k: extended.max_k(),
        rank: extended.rank(),
    };
    assert_eq!(
        module_test.ckks_ci_embed_tmp_bytes(extended, &key_infos),
        module_test.glwe_keyswitch_tmp_bytes(&allocated, &allocated, &key_infos)
    );
    let trace_bytes = module_test.ckks_ci_trace_tmp_bytes(back, extended, &key_infos);
    assert_eq!(
        trace_bytes,
        OverrideBackend::scratch_aligned(module_test.glwe_bytes_of_from_infos(extended))
            + module_test.glwe_keyswitch_tmp_bytes(extended, extended, &key_infos).max(
                poulpy_core::GLWECITrace::glwe_ci_trace_tmp_bytes(&module_test, back, extended)
            )
    );
    assert!(
        poulpy_core::GLWECITrace::glwe_ci_trace_tmp_bytes(&module_test, back, extended)
            == poulpy_core::GLWECITrace::glwe_ci_trace_tmp_bytes(&module_ref, back, extended) + CI_TRACE_EXTRA
    );

    let want = round_trip(&module_ref, &fixtures);
    assert_untagged("ring maps", [&want.0, &want.1]);
    CI_EMBED_CALLS.set(0);
    CI_TRACE_CALLS.set(0);
    let got = round_trip(&module_test, &fixtures);
    assert_eq!(CI_EMBED_CALLS.get(), 1);
    assert_eq!(CI_TRACE_CALLS.get(), 1);
    assert_eq!(got, want);

    // A conjugate-invariant output of the wrong degree is refused before the trace is entered.
    let (_, trace_key) = prepared_keys(&module_test, &fixtures);
    let mut wrong = fixture_ciphertext(
        &Module::<FFT64CIPortable>::new(16),
        &layout(params(16), 1, 48, 16, 0, SlotsKind::Complex),
        54,
    );
    let unchanged = snapshot::<FFT64CIPortable, _>(&wrong);
    CI_TRACE_CALLS.set(0);
    let err =
        with_scratch::<OverrideBackend, _>(0, |s| module_test.ckks_ci_trace(&mut wrong, extended, &trace_key, s)).unwrap_err();
    assert!(
        matches!(
            err.composition(),
            Some(CKKSCompositionError::InvalidOperand { op: "ckks_ci_trace", .. })
        ),
        "wrong output degree: {err}"
    );
    assert_eq!(CI_TRACE_CALLS.get(), 0);
    assert_eq!(snapshot::<FFT64CIPortable, _>(&wrong), unchanged);
}
