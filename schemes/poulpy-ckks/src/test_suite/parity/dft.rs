//! Paired matrix generation and homomorphic transforms in every output format.
use super::{
    helpers::{Snapshot, assert_untagged, fixture_ciphertext, snapshot, with_scratch},
    keys::{key_layout, prepared_automorphism_key},
};
use crate::Scale;
use crate::{
    CKKSLayout, CoeffsMeta, GLWEPlaintextMeta, SlotLayout, SlotsKind,
    api::{CKKSDFTMatrixOps, CKKSDFTOps, CKKSEncodingHostOps, CKKSEncodingScalar},
    layouts::{CKKSModuleAlloc, DFTOutputFormat, DFTPlan, DFTType, Decode, Encode, Repack, Split, Standard},
    oep::{CKKSDFTImpl, CKKSDFTMatrixImpl, CKKSEncodingImpl},
    test_suite::CKKSTestParams,
};
use poulpy_core::{GLWEMaskFill, api::GLWEAutomorphismKeyPreparedFactory, layouts::GLWELayout};
use poulpy_hal::layouts::{Backend, CyclotomicOrder, Module};
use std::collections::HashMap;

// A diagonal stored at its period's degree may stop at a backend-specific minimum degree.
// Compare the public polynomial after the common ring embedding, not its
// allocation degree or unused coefficient slots.
fn diagonal_snapshot<BE, A>(value: &A, ring_degree: usize) -> Snapshot
where
    BE: Backend<ZnxWord = i64>,
    A: crate::CKKSInfos + poulpy_core::layouts::GLWEInfos + poulpy_core::layouts::GLWEToBackendRef<BE>,
{
    let mut out = snapshot::<BE, _>(value);
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

fn run<BE, F>(params: CKKSTestParams, module: &Module<BE>) -> Vec<Snapshot>
where
    BE: Backend<ZnxWord = i64> + CKKSDFTImpl + CKKSDFTMatrixImpl<F> + CKKSEncodingImpl<F>,
    F: CKKSEncodingScalar,
    Module<BE>: GLWEAutomorphismKeyPreparedFactory<BE> + GLWEMaskFill<BE>,
{
    let b = params.base2k;
    let k = 8 * b + 7;
    let key = key_layout(module.n(), b, k, 1, 1, 1);
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
            meta: GLWEPlaintextMeta {
                scale: Scale::Log(16),
                slots: SlotsKind::Complex,
                log_sparsity: log_max_slots - log_slots,
                slot_layout: SlotLayout::Full,
            },
        };
        let encoding = <Module<BE> as CKKSEncodingHostOps<BE, F>>::ckks_reim_tmp_bytes(module, module.n() / 2);
        macro_rules! prepare {
            ($dir:ty, $fmt:ty, $kind:expr, $format:expr) => {{
                let plan = DFTPlan::new(
                    $kind,
                    vec![(1, 2); log_slots],
                    $format,
                    CoeffsMeta::from_delta_budget(8, 12),
                )
                .unwrap();
                let dft = with_scratch::<BE, _>(encoding, |scratch| {
                    <Module<BE> as CKKSDFTMatrixOps<BE, F>>::ckks_new_dft_matrix::<$dir, $fmt>(module, b.into(), &plan, scratch)
                })
                .unwrap();
                // Quantized diagonals are public logical operands; prepared
                // transforms are independently allocated and never byte-compared.
                for factor in dft.factor_operands() {
                    for giant in &factor.giant_steps {
                        for diagonal in &giant.diagonals {
                            results.push(diagonal_snapshot::<BE, _>(&diagonal.plaintext, module.n()));
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
                let mut prepared = module.ckks_dft_matrix_prepared_alloc_from_infos(&dft);
                with_scratch::<BE, _>(module.ckks_prepare_dft_matrix_tmp_bytes(&dft), |scratch| {
                    module.ckks_prepare_dft_matrix(&mut prepared, &dft, scratch)
                })
                .unwrap();
                let mut raw = fixture_ciphertext(module, &layout, 61);
                with_scratch::<BE, _>(module.ckks_dft_tmp_bytes(&layout, &layout, &dft, &key), |scratch| {
                    module.ckks_dft_evaluate_assign(&mut raw, &dft, &keys, scratch)
                })
                .unwrap();
                let mut resident = fixture_ciphertext(module, &layout, 61);
                with_scratch::<BE, _>(
                    module.ckks_dft_tmp_bytes(&layout, &layout, &prepared, &key),
                    |scratch| module.ckks_dft_evaluate_assign(&mut resident, &prepared, &keys, scratch),
                )
                .unwrap();
                assert_eq!(
                    snapshot::<BE, _>(&raw),
                    snapshot::<BE, _>(&resident),
                    "prepared DFT differs from streamed DFT"
                );
                results.push(snapshot::<BE, _>(&raw));
                (dft, prepared, keys)
            }};
        }
        macro_rules! standard {
            ($dir:ty, $kind:expr, $method:ident) => {{
                let (dft, prepared, keys) = prepare!($dir, Standard, $kind, DFTOutputFormat::Standard);
                let mut out = fixture_ciphertext(module, &layout, 67);
                with_scratch::<BE, _>(module.ckks_dft_tmp_bytes(&layout, &layout, &dft, &key), |scratch| {
                    module.$method(&mut out, &dft, &keys, scratch)
                })
                .unwrap();
                let mut resident = fixture_ciphertext(module, &layout, 67);
                with_scratch::<BE, _>(
                    module.ckks_dft_tmp_bytes(&layout, &layout, &prepared, &key),
                    |scratch| module.$method(&mut resident, &prepared, &keys, scratch),
                )
                .unwrap();
                assert_eq!(snapshot::<BE, _>(&out), snapshot::<BE, _>(&resident));
                results.push(snapshot::<BE, _>(&out));
            }};
        }
        standard!(Encode, DFTType::Encode, ckks_coeffs_to_slots);
        standard!(Decode, DFTType::Decode, ckks_slots_to_coeffs);
        {
            let (dft, prepared, keys) = prepare!(Encode, Split, DFTType::Encode, DFTOutputFormat::SplitRealAndImag);
            let src = fixture_ciphertext(module, &layout, 71);
            let before = snapshot::<BE, _>(&src);
            let mut re = fixture_ciphertext(module, &layout, 73);
            let mut im = fixture_ciphertext(module, &layout, 79);
            with_scratch::<BE, _>(module.ckks_dft_tmp_bytes(&layout, &layout, &dft, &key), |scratch| {
                module.ckks_coeffs_to_slots_split(&mut re, &mut im, &src, &dft, &keys, scratch)
            })
            .unwrap();
            let mut resident_re = fixture_ciphertext(module, &layout, 73);
            let mut resident_im = fixture_ciphertext(module, &layout, 79);
            with_scratch::<BE, _>(module.ckks_dft_tmp_bytes(&layout, &layout, &prepared, &key), |scratch| {
                module.ckks_coeffs_to_slots_split(&mut resident_re, &mut resident_im, &src, &prepared, &keys, scratch)
            })
            .unwrap();
            assert_eq!(snapshot::<BE, _>(&re), snapshot::<BE, _>(&resident_re));
            assert_eq!(snapshot::<BE, _>(&im), snapshot::<BE, _>(&resident_im));
            assert_eq!(before, snapshot::<BE, _>(&src));
            results.push(snapshot::<BE, _>(&re));
            results.push(snapshot::<BE, _>(&im));
        }
        {
            let (dft, prepared, keys) = prepare!(Decode, Split, DFTType::Decode, DFTOutputFormat::SplitRealAndImag);
            let real_layout = CKKSLayout {
                meta: GLWEPlaintextMeta {
                    slots: SlotsKind::Real,
                    ..layout.meta
                },
                ..layout
            };
            let re = fixture_ciphertext(module, &real_layout, 83);
            let im = fixture_ciphertext(module, &real_layout, 89);
            let mut out = fixture_ciphertext(module, &layout, 97);
            with_scratch::<BE, _>(module.ckks_dft_tmp_bytes(&layout, &layout, &dft, &key), |scratch| {
                module.ckks_slots_to_coeffs_split(&mut out, &re, &im, &dft, &keys, scratch)
            })
            .unwrap();
            let mut resident = fixture_ciphertext(module, &layout, 97);
            with_scratch::<BE, _>(module.ckks_dft_tmp_bytes(&layout, &layout, &prepared, &key), |scratch| {
                module.ckks_slots_to_coeffs_split(&mut resident, &re, &im, &prepared, &keys, scratch)
            })
            .unwrap();
            assert_eq!(snapshot::<BE, _>(&out), snapshot::<BE, _>(&resident));
            results.push(snapshot::<BE, _>(&out));
            assert_eq!(
                snapshot::<BE, _>(&re),
                snapshot::<BE, _>(&fixture_ciphertext(module, &real_layout, 83))
            );
            assert_eq!(
                snapshot::<BE, _>(&im),
                snapshot::<BE, _>(&fixture_ciphertext(module, &real_layout, 89))
            );
        }
        if log_slots < log_max_slots {
            let (dft, prepared, keys) = prepare!(Encode, Repack, DFTType::Encode, DFTOutputFormat::RepackImagAsReal);
            let src = fixture_ciphertext(module, &layout, 101);
            let mut out = fixture_ciphertext(module, &layout, 103);
            with_scratch::<BE, _>(module.ckks_dft_tmp_bytes(&layout, &layout, &dft, &key), |scratch| {
                module.ckks_coeffs_to_slots_repack(&mut out, &src, &dft, &keys, scratch)
            })
            .unwrap();
            let mut resident = fixture_ciphertext(module, &layout, 103);
            with_scratch::<BE, _>(module.ckks_dft_tmp_bytes(&layout, &layout, &prepared, &key), |scratch| {
                module.ckks_coeffs_to_slots_repack(&mut resident, &src, &prepared, &keys, scratch)
            })
            .unwrap();
            assert_eq!(snapshot::<BE, _>(&out), snapshot::<BE, _>(&resident));
            results.push(snapshot::<BE, _>(&out));
            assert_eq!(
                snapshot::<BE, _>(&src),
                snapshot::<BE, _>(&fixture_ciphertext(module, &layout, 101))
            );
            let (dft, prepared, keys) = prepare!(Decode, Repack, DFTType::Decode, DFTOutputFormat::RepackImagAsReal);
            let repacked = CKKSLayout {
                meta: GLWEPlaintextMeta {
                    slots: SlotsKind::Real,
                    log_sparsity: layout.meta.log_sparsity - 1,
                    ..layout.meta
                },
                ..layout
            };
            let src = fixture_ciphertext(module, &repacked, 107);
            let mut out = fixture_ciphertext(module, &layout, 109);
            with_scratch::<BE, _>(module.ckks_dft_tmp_bytes(&layout, &layout, &dft, &key), |scratch| {
                module.ckks_slots_to_coeffs_repack(&mut out, &src, &dft, &keys, scratch)
            })
            .unwrap();
            let mut resident = fixture_ciphertext(module, &layout, 109);
            with_scratch::<BE, _>(module.ckks_dft_tmp_bytes(&layout, &layout, &prepared, &key), |scratch| {
                module.ckks_slots_to_coeffs_repack(&mut resident, &src, &prepared, &keys, scratch)
            })
            .unwrap();
            assert_eq!(snapshot::<BE, _>(&out), snapshot::<BE, _>(&resident));
            results.push(snapshot::<BE, _>(&out));
            assert_eq!(
                snapshot::<BE, _>(&src),
                snapshot::<BE, _>(&fixture_ciphertext(module, &repacked, 107))
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
                with_scratch::<BE, _>(encoding, |scratch| {
                    <Module<BE> as CKKSDFTMatrixOps<BE, F>>::ckks_new_dft_matrix::<Encode, Repack>(
                        module,
                        b.into(),
                        &invalid,
                        scratch,
                    )
                })
                .is_err()
            );
        }
    }
    results
}

/// Checks all DFT dispatch methods, both directions, each valid format, sparse
/// and dense layouts, prepared and streamed factors, and the shared scratch contract.
pub fn test_dft_parity<BR, BT, F>(params: CKKSTestParams, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    BR: Backend<ZnxWord = i64> + CKKSDFTImpl + CKKSDFTMatrixImpl<F> + CKKSEncodingImpl<F>,
    BT: Backend<ZnxWord = i64> + CKKSDFTImpl + CKKSDFTMatrixImpl<F> + CKKSEncodingImpl<F>,
    F: CKKSEncodingScalar,
    Module<BR>: GLWEAutomorphismKeyPreparedFactory<BR> + GLWEMaskFill<BR>,
    Module<BT>: GLWEAutomorphismKeyPreparedFactory<BT> + GLWEMaskFill<BT>,
{
    assert_eq!(module_ref.n(), module_test.n());
    let want = run::<BR, F>(params, module_ref);
    assert_untagged("DFT", &want);
    assert_eq!(want, run::<BT, F>(params, module_test), "DFT parity differs");
}
