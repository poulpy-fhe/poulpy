//! Plaintext extraction on independently uploaded coefficients and metadata.
use super::{arithmetic::layout, helpers::*};
use crate::{CKKSInfos, SlotsKind, oep::CKKSPlaintextZnxImpl, test_suite::CKKSTestParams};
use poulpy_core::layouts::LWEInfos;
use poulpy_hal::layouts::{Backend, Module};

fn exercise<B: Backend<ZnxWord = i64> + CKKSPlaintextZnxImpl>(params: CKKSTestParams, module: &Module<B>) -> Vec<Snapshot> {
    let b = params.base2k;
    let mut results = Vec::new();
    for sparse in [0, 2] {
        for slots in [SlotsKind::Real, SlotsKind::Complex] {
            let input = layout(params, 0, 3 * b + 1, b, sparse, slots);
            let a = fixture_plaintext(module, &input, 43);
            let before = snapshot::<B, _>(&a);
            for delta in [b - 3, b, b + 3] {
                for budget in [0, b + 1] {
                    let output = layout(params, 0, delta + budget, delta, sparse, slots);
                    let mut out = fixture_plaintext(module, &output, 99);
                    with_scratch::<B, _>(B::ckks_extract_pt_tmp_bytes_impl(module, out.max_size()), |scratch| {
                        B::ckks_extract_pt_impl(module, &mut out, &a, scratch)
                    })
                    .unwrap();
                    assert_eq!(out.meta(), output.meta);
                    results.push(snapshot::<B, _>(&out));
                    assert_eq!(before, snapshot::<B, _>(&a));
                }
            }
            let mut invalid = input;
            invalid.glwe_layout.base2k = (b - 1).into();
            let mut out = fixture_plaintext(module, &invalid, 99);
            let before_out = snapshot::<B, _>(&out);
            assert!(
                with_scratch::<B, _>(B::ckks_extract_pt_tmp_bytes_impl(module, out.max_size()), |scratch| {
                    B::ckks_extract_pt_impl(module, &mut out, &a, scratch)
                })
                .is_err()
            );
            assert_eq!(before_out, snapshot::<B, _>(&out), "failed extraction mutated output");
        }
    }
    results
}

pub fn test_plaintext_parity<BR, BT, F>(params: CKKSTestParams, r: &Module<BR>, t: &Module<BT>)
where
    BR: Backend<ZnxWord = i64> + CKKSPlaintextZnxImpl,
    BT: Backend<ZnxWord = i64> + CKKSPlaintextZnxImpl,
{
    let _scalar = std::marker::PhantomData::<F>;
    assert_eq!(exercise(params, r), exercise(params, t));
}
