//! Plaintext extraction on coefficients each backend samples from the same seed, and metadata.
use super::{arithmetic::layout, helpers::*};
use crate::{CKKSInfos, SlotsKind, oep::CKKSPlaintextVecImpl, test_suite::CKKSTestParams};
use poulpy_core::{GLWEFill, layouts::LWEInfos};
use poulpy_hal::layouts::{Backend, Module};

fn exercise<BE: Backend<ZnxWord = i64> + CKKSPlaintextVecImpl>(params: CKKSTestParams, module: &Module<BE>) -> Vec<Snapshot>
where
    Module<BE>: GLWEFill<BE>,
{
    let b = params.base2k;
    let mut results = Vec::new();
    for sparse in [0, 2] {
        for slots in [SlotsKind::Real, SlotsKind::Complex] {
            let input = layout(params, 0, 3 * b + 1, b, sparse, slots);
            let a = fixture_plaintext(module, &input, 43);
            let before = snapshot::<BE, _>(&a);
            for delta in [b - 3, b, b + 3] {
                for budget in [0, b + 1] {
                    let output = layout(params, 0, delta + budget, delta, sparse, slots);
                    let mut out = fixture_plaintext(module, &output, 99);
                    with_scratch::<BE, _>(BE::ckks_extract_pt_tmp_bytes(module, out.max_size()), |scratch| {
                        BE::ckks_extract_pt(module, &mut out, &a, scratch)
                    })
                    .unwrap();
                    assert_eq!(out.meta(), output.meta);
                    results.push(snapshot::<BE, _>(&out));
                    assert_eq!(before, snapshot::<BE, _>(&a));
                }
            }
            let mut invalid = input;
            invalid.glwe_layout.base2k = (b - 1).into();
            let mut out = fixture_plaintext(module, &invalid, 99);
            let before_out = snapshot::<BE, _>(&out);
            assert!(with_scratch::<BE, _>(0, |scratch| BE::ckks_extract_pt(module, &mut out, &a, scratch)).is_err());
            assert_eq!(before_out, snapshot::<BE, _>(&out), "failed extraction mutated output");
        }
    }
    results
}

pub fn test_plaintext_parity<BR, BT, F>(params: CKKSTestParams, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    BR: Backend<ZnxWord = i64> + CKKSPlaintextVecImpl,
    BT: Backend<ZnxWord = i64> + CKKSPlaintextVecImpl,
    Module<BR>: GLWEFill<BR>,
    Module<BT>: GLWEFill<BT>,
{
    let _scalar = std::marker::PhantomData::<F>;
    let want = exercise(params, module_ref);
    assert_untagged("plaintext", &want);
    assert_eq!(want, exercise(params, module_test));
}
