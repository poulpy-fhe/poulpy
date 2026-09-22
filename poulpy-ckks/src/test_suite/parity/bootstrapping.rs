//! Encapsulated modulus raising, including optimized and fallback shapes.
use super::{
    helpers::{Snapshot, fixture_ciphertext, snapshot, with_scratch},
    keys::{fixture_gglwe, key_layout},
};
use crate::{CKKSInfos, CKKSLayout, CKKSMeta, SlotsKind, oep::CKKSEncapsulatedModUpImpl, test_suite::CKKSTestParams};
use poulpy_core::layouts::{GGLWEPreparedFactory, GLWELayout, LWEInfos, prepared::GGLWEPreparedToBackendRef};
use poulpy_hal::layouts::{Backend, Module};

fn run<B>(params: CKKSTestParams, module: &Module<B>) -> Vec<(Result<(), String>, Snapshot, Snapshot)>
where
    B: Backend<ZnxWord = i64> + CKKSEncapsulatedModUpImpl,
    Module<B>: GGLWEPreparedFactory<B>,
{
    let b = params.base2k;
    let small = 3 * b + 1;
    let mut results = Vec::new();
    // Matching radix and multi-limb digits permit the optimized path. Other
    // cases require the general composition; widths also cover partial limbs,
    // no zero-prefix limbs, several full zero-prefix limbs, and rejected raises.
    for (dsize, key_b, extra, scale) in [
        (2, b, 4 * b + 3, 0),
        (2, b, 3 * b + 5, b + 1),
        (2, b, 5, 4),
        (1, b, 2 * b + 1, 1),
        (2, b - 1, 2 * b + 3, 0),
        (2, b, 0, 1),
    ] {
        let src_layout = CKKSLayout {
            ring_kind: crate::CKKSRingKind::Standard,
            glwe_layout: GLWELayout {
                n: module.n().into(),
                base2k: b.into(),
                k: small.into(),
                rank: params.rank.into(),
            },
            meta: CKKSMeta {
                log_delta: b,
                log_sparsity: 1,
                slots: SlotsKind::Complex,
            },
        };
        let dst_layout = CKKSLayout {
            ring_kind: crate::CKKSRingKind::Standard,
            glwe_layout: GLWELayout {
                k: (small + extra).into(),
                ..src_layout.glwe_layout
            },
            ..src_layout
        };
        let d2s_layout = key_layout(module.n(), b, small, 1, params.rank, params.rank);
        let s2d_layout = key_layout(module.n(), key_b, small + extra, dsize, params.rank, params.rank);
        let mut d2s = module.gglwe_prepared_alloc_from_infos(&d2s_layout);
        let mut s2d = module.gglwe_prepared_alloc_from_infos(&s2d_layout);
        for (prepared, layout, seed) in [(&mut d2s, &d2s_layout, 31), (&mut s2d, &s2d_layout, 37)] {
            let coefficients = fixture_gglwe(module, layout, seed);
            with_scratch::<B, _>(module.gglwe_prepare_tmp_bytes(layout), |scratch| {
                module.gglwe_prepare(prepared, &coefficients, scratch)
            });
        }
        let bytes = B::ckks_encapsulated_mod_up_tmp_bytes(module, &dst_layout, &src_layout, &d2s_layout, &s2d_layout);
        for seed in [41, 43] {
            let mut src = fixture_ciphertext(module, &src_layout, seed);
            let mut dst = fixture_ciphertext(module, &dst_layout, 47);
            let result = with_scratch::<B, _>(bytes, |scratch| {
                B::ckks_encapsulated_mod_up(
                    module,
                    &mut dst,
                    &mut src,
                    scale,
                    &d2s.to_backend_ref(),
                    &s2d.to_backend_ref(),
                    scratch,
                )
            });
            assert_eq!(result.is_ok(), extra >= scale, "modulus raise acceptance changed");
            if result.is_ok() {
                assert_eq!(
                    dst.meta(),
                    CKKSMeta {
                        log_delta: src_layout.meta.log_delta + scale,
                        ..src_layout.meta
                    }
                );
                assert_eq!(dst.k(), dst_layout.glwe_layout.k);
            }
            results.push((
                result.map_err(|error| error.to_string()),
                snapshot::<B, _>(&src),
                snapshot::<B, _>(&dst),
            ));
        }
    }
    results
}

/// Compare selected implementations using the same canonical inputs and key
/// coefficients. Source mutations, destination metadata, errors, and each
/// implementation's own exact guarded scratch budget are part of the contract.
pub fn test_encapsulated_mod_up_parity<BR, BT, F>(params: CKKSTestParams, reference: &Module<BR>, tested: &Module<BT>)
where
    BR: Backend<ZnxWord = i64> + CKKSEncapsulatedModUpImpl,
    BT: Backend<ZnxWord = i64> + CKKSEncapsulatedModUpImpl,
    Module<BR>: GGLWEPreparedFactory<BR>,
    Module<BT>: GGLWEPreparedFactory<BT>,
{
    assert_eq!(reference.n(), tested.n());
    assert_eq!(run(params, reference), run(params, tested), "encapsulated ModUp differs");
}
