//! Paired matrix generation and homomorphic transforms in every output format.
use super::{
    helpers::{Snapshot, fixture_ciphertext, snapshot, with_scratch},
    keys::{key_layout, prepared_automorphism_key},
};
use crate::{
    CKKSLayout, CKKSMeta, CoeffsMeta, SlotsKind,
    api::{CKKSAllOpsTmpBytes, CKKSDFTMatrixOps, CKKSDFTOps, CKKSEncodingHostOps, CKKSEncodingScalar},
    layouts::{DFTOutputFormat, DFTPlan, DFTType, Decode, Encode, Repack, Split, Standard},
    oep::{CKKSEncodingImpl, DFTImpl, DFTMatrixImpl},
    test_suite::CKKSTestParams,
};
use poulpy_core::layouts::{GLWEAutomorphismKeyPreparedFactory, GLWELayout};
use poulpy_hal::layouts::{Backend, CyclotomicOrder, Module};
use std::collections::HashMap;

// Compact diagonal storage may stop at a backend-specific minimum degree.
// Compare the public polynomial after the common ring embedding, not its
// allocation degree or unused coefficient slots.
fn diagonal_snapshot<B, A>(value: &A, ring_degree: usize) -> Snapshot
where
    B: Backend<ZnxWord = i64>,
    A: crate::CKKSInfos + poulpy_core::layouts::GLWEInfos + poulpy_core::layouts::GLWEToBackendRef<B>,
{
    let mut out = snapshot::<B, _>(value);
    let degree = out.layout.glwe_layout.n.as_usize();
    assert!(ring_degree.is_multiple_of(degree));
    let gap = ring_degree / degree;
    let mut embedded = vec![0; out.digits.len() * gap];
    for (index, coefficient) in out.digits.iter().enumerate() {
        embedded[index * gap] = *coefficient;
    }
    out.digits = embedded;
    out.layout.glwe_layout.n = ring_degree.into();
    out
}

fn run<B, F>(params: CKKSTestParams, module: &Module<B>) -> Vec<Snapshot>
where
    B: Backend<ZnxWord = i64> + DFTImpl + DFTMatrixImpl<F> + CKKSEncodingImpl<F>,
    F: CKKSEncodingScalar,
    Module<B>: CKKSAllOpsTmpBytes<B> + GLWEAutomorphismKeyPreparedFactory<B>,
{
    let b = params.base2k;
    let k = 8 * b + 7;
    let key = key_layout(module.n(), b, k, 1, 1, 1);
    let pt = CKKSLayout {
        glwe_layout: GLWELayout {
            n: module.n().into(),
            base2k: b.into(),
            k: 20usize.into(),
            rank: 0usize.into(),
        },
        meta: CKKSMeta {
            log_delta: 8,
            log_sparsity: 0,
            slots: SlotsKind::Complex,
        },
    };
    let mut results = Vec::new();
    let log_max_slots = module.n().ilog2() as usize - 1;
    for log_slots in [log_max_slots, log_max_slots - 2] {
        let layout = CKKSLayout {
            glwe_layout: GLWELayout {
                n: module.n().into(),
                base2k: b.into(),
                k: k.into(),
                rank: 1usize.into(),
            },
            meta: CKKSMeta {
                log_delta: 16,
                log_sparsity: log_max_slots - log_slots,
                slots: SlotsKind::Complex,
            },
        };
        let shared = module.ckks_all_ops_with_atk_tmp_bytes(&layout, &key, &key, &pt);
        let encoding = <Module<B> as CKKSEncodingHostOps<B, F>>::ckks_reim_tmp_bytes(module, module.n() / 2);
        macro_rules! prepare {
            ($dir:ty, $fmt:ty, $kind:expr, $format:expr) => {{
                let plan = DFTPlan::new(
                    $kind,
                    vec![(1, 2); log_slots],
                    $format,
                    CoeffsMeta::from_delta_budget(8, 12),
                )
                .unwrap();
                let dft = with_scratch::<B, _>(encoding, |scratch| {
                    <Module<B> as CKKSDFTMatrixOps<B, F>>::ckks_new_dft_matrix::<$dir, $fmt>(module, b.into(), &plan, scratch)
                })
                .unwrap();
                // Quantized diagonals are public logical operands; prepared
                // transforms are independently allocated and never byte-compared.
                for factor in dft.factor_operands() {
                    for giant in &factor.giant_steps {
                        for diagonal in &giant.diagonals {
                            results.push(diagonal_snapshot::<B, _>(&diagonal.plaintext, module.n()));
                        }
                    }
                }
                let mut keys = HashMap::new();
                for p in plan
                    .galois_elements(module.n().ilog2() as usize, module.cyclotomic_order())
                    .into_iter()
                    .chain([-1])
                {
                    keys.insert(
                        p,
                        prepared_automorphism_key(module, &key, p, (p as u8).wrapping_add(53)),
                    );
                }
                let prepared = with_scratch::<B, _>(shared, |scratch| module.ckks_prepare_dft_matrix(&dft, scratch)).unwrap();
                let mut raw = fixture_ciphertext(module, &layout, 61);
                with_scratch::<B, _>(shared, |scratch| {
                    module.ckks_dft_evaluate_assign(&mut raw, &dft, &keys, scratch)
                })
                .unwrap();
                let mut resident = fixture_ciphertext(module, &layout, 61);
                with_scratch::<B, _>(shared, |scratch| {
                    module.ckks_dft_evaluate_assign(&mut resident, &prepared, &keys, scratch)
                })
                .unwrap();
                assert_eq!(
                    snapshot::<B, _>(&raw),
                    snapshot::<B, _>(&resident),
                    "prepared DFT differs from streamed DFT"
                );
                results.push(snapshot::<B, _>(&raw));
                (dft, prepared, keys)
            }};
        }
        macro_rules! standard {
            ($dir:ty, $kind:expr, $method:ident) => {{
                let (dft, prepared, keys) = prepare!($dir, Standard, $kind, DFTOutputFormat::Standard);
                let mut out = fixture_ciphertext(module, &layout, 67);
                with_scratch::<B, _>(shared, |scratch| module.$method(&mut out, &dft, &keys, scratch)).unwrap();
                let mut resident = fixture_ciphertext(module, &layout, 67);
                with_scratch::<B, _>(shared, |scratch| {
                    module.$method(&mut resident, &prepared, &keys, scratch)
                })
                .unwrap();
                assert_eq!(snapshot::<B, _>(&out), snapshot::<B, _>(&resident));
                results.push(snapshot::<B, _>(&out));
            }};
        }
        standard!(Encode, DFTType::Encode, ckks_coeffs_to_slots);
        standard!(Decode, DFTType::Decode, ckks_slots_to_coeffs);
        {
            let (dft, prepared, keys) = prepare!(Encode, Split, DFTType::Encode, DFTOutputFormat::SplitRealAndImag);
            let src = fixture_ciphertext(module, &layout, 71);
            let before = snapshot::<B, _>(&src);
            let mut re = fixture_ciphertext(module, &layout, 73);
            let mut im = fixture_ciphertext(module, &layout, 79);
            with_scratch::<B, _>(shared, |scratch| {
                module.ckks_coeffs_to_slots_split(&mut re, &mut im, &src, &dft, &keys, scratch)
            })
            .unwrap();
            let mut resident_re = fixture_ciphertext(module, &layout, 73);
            let mut resident_im = fixture_ciphertext(module, &layout, 79);
            with_scratch::<B, _>(shared, |scratch| {
                module.ckks_coeffs_to_slots_split(&mut resident_re, &mut resident_im, &src, &prepared, &keys, scratch)
            })
            .unwrap();
            assert_eq!(snapshot::<B, _>(&re), snapshot::<B, _>(&resident_re));
            assert_eq!(snapshot::<B, _>(&im), snapshot::<B, _>(&resident_im));
            assert_eq!(before, snapshot::<B, _>(&src));
            results.push(snapshot::<B, _>(&re));
            results.push(snapshot::<B, _>(&im));
        }
        {
            let (dft, prepared, keys) = prepare!(Decode, Split, DFTType::Decode, DFTOutputFormat::SplitRealAndImag);
            let real_layout = CKKSLayout {
                meta: CKKSMeta {
                    slots: SlotsKind::Real,
                    ..layout.meta
                },
                ..layout
            };
            let re = fixture_ciphertext(module, &real_layout, 83);
            let im = fixture_ciphertext(module, &real_layout, 89);
            let mut out = fixture_ciphertext(module, &layout, 97);
            with_scratch::<B, _>(shared, |scratch| {
                module.ckks_slots_to_coeffs_split(&mut out, &re, &im, &dft, &keys, scratch)
            })
            .unwrap();
            let mut resident = fixture_ciphertext(module, &layout, 97);
            with_scratch::<B, _>(shared, |scratch| {
                module.ckks_slots_to_coeffs_split(&mut resident, &re, &im, &prepared, &keys, scratch)
            })
            .unwrap();
            assert_eq!(snapshot::<B, _>(&out), snapshot::<B, _>(&resident));
            results.push(snapshot::<B, _>(&out));
            assert_eq!(
                snapshot::<B, _>(&re),
                snapshot::<B, _>(&fixture_ciphertext(module, &real_layout, 83))
            );
            assert_eq!(
                snapshot::<B, _>(&im),
                snapshot::<B, _>(&fixture_ciphertext(module, &real_layout, 89))
            );
        }
        if log_slots < log_max_slots {
            let (dft, prepared, keys) = prepare!(Encode, Repack, DFTType::Encode, DFTOutputFormat::RepackImagAsReal);
            let src = fixture_ciphertext(module, &layout, 101);
            let mut out = fixture_ciphertext(module, &layout, 103);
            with_scratch::<B, _>(shared, |scratch| {
                module.ckks_coeffs_to_slots_repack(&mut out, &src, &dft, &keys, scratch)
            })
            .unwrap();
            let mut resident = fixture_ciphertext(module, &layout, 103);
            with_scratch::<B, _>(shared, |scratch| {
                module.ckks_coeffs_to_slots_repack(&mut resident, &src, &prepared, &keys, scratch)
            })
            .unwrap();
            assert_eq!(snapshot::<B, _>(&out), snapshot::<B, _>(&resident));
            results.push(snapshot::<B, _>(&out));
            assert_eq!(
                snapshot::<B, _>(&src),
                snapshot::<B, _>(&fixture_ciphertext(module, &layout, 101))
            );
            let (dft, prepared, keys) = prepare!(Decode, Repack, DFTType::Decode, DFTOutputFormat::RepackImagAsReal);
            let repacked = CKKSLayout {
                meta: CKKSMeta {
                    log_sparsity: layout.meta.log_sparsity - 1,
                    slots: SlotsKind::Real,
                    ..layout.meta
                },
                ..layout
            };
            let src = fixture_ciphertext(module, &repacked, 107);
            let mut out = fixture_ciphertext(module, &layout, 109);
            with_scratch::<B, _>(shared, |scratch| {
                module.ckks_slots_to_coeffs_repack(&mut out, &src, &dft, &keys, scratch)
            })
            .unwrap();
            let mut resident = fixture_ciphertext(module, &layout, 109);
            with_scratch::<B, _>(shared, |scratch| {
                module.ckks_slots_to_coeffs_repack(&mut resident, &src, &prepared, &keys, scratch)
            })
            .unwrap();
            assert_eq!(snapshot::<B, _>(&out), snapshot::<B, _>(&resident));
            results.push(snapshot::<B, _>(&out));
            assert_eq!(
                snapshot::<B, _>(&src),
                snapshot::<B, _>(&fixture_ciphertext(module, &repacked, 107))
            );
        } else {
            let invalid = DFTPlan::new(
                DFTType::Encode,
                vec![(1, 2); log_slots],
                DFTOutputFormat::RepackImagAsReal,
                CoeffsMeta::from_delta_budget(8, 12),
            )
            .unwrap();
            assert!(
                with_scratch::<B, _>(
                    encoding,
                    |scratch| <Module<B> as CKKSDFTMatrixOps<B, F>>::ckks_new_dft_matrix::<Encode, Repack>(
                        module,
                        b.into(),
                        &invalid,
                        scratch
                    )
                )
                .is_err()
            );
        }
    }
    results
}

/// Checks all DFT dispatch methods, both directions, each valid format, sparse
/// and dense layouts, prepared and streamed factors, and the shared scratch contract.
pub fn test_dft_parity<BR, BT, F>(params: CKKSTestParams, reference: &Module<BR>, tested: &Module<BT>)
where
    BR: Backend<ZnxWord = i64> + DFTImpl + DFTMatrixImpl<F> + CKKSEncodingImpl<F>,
    BT: Backend<ZnxWord = i64> + DFTImpl + DFTMatrixImpl<F> + CKKSEncodingImpl<F>,
    F: CKKSEncodingScalar,
    Module<BR>: CKKSAllOpsTmpBytes<BR> + GLWEAutomorphismKeyPreparedFactory<BR>,
    Module<BT>: CKKSAllOpsTmpBytes<BT> + GLWEAutomorphismKeyPreparedFactory<BT>,
{
    assert_eq!(reference.n(), tested.n());
    assert_eq!(
        run::<BR, F>(params, reference),
        run::<BT, F>(params, tested),
        "DFT parity differs"
    );
}
