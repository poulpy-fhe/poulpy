//! Plaintext extraction on coefficients each backend samples from the same seed, and metadata.
use super::{arithmetic::layout, helpers::*};
use crate::{CKKSInfos, SlotsKind, oep::CKKSPlaintextZnxImpl, test_suite::CKKSTestParams};
use poulpy_core::{GLWEMaskFill, layouts::LWEInfos};
use poulpy_hal::layouts::{Backend, Module};

fn exercise<B: Backend<ZnxWord = i64> + CKKSPlaintextZnxImpl>(params: CKKSTestParams, module: &Module<B>) -> Vec<HostCiphertext>
where
    Module<B>: GLWEMaskFill<B>,
{
    let b = params.base2k;
    let mut results = Vec::new();
    for sparse in [0, 2] {
        for slots in [SlotsKind::Real, SlotsKind::Complex] {
            let input = layout(params, 0, 3 * b + 1, b, sparse, slots);
            let a = fixture_plaintext(module, &input, 43);
            let before = host_ciphertext::<B, _>(&a);
            for delta in [b - 3, b, b + 3] {
                for budget in [0, b + 1] {
                    let output = layout(params, 0, delta + budget, delta, sparse, slots);
                    let mut out = fixture_plaintext(module, &output, 99);
                    with_scratch::<B, _>(B::ckks_extract_pt_tmp_bytes_impl(module, out.max_size()), |scratch| {
                        B::ckks_extract_pt_impl(module, &mut out, &a, scratch)
                    })
                    .unwrap();
                    assert_eq!(out.meta(), output.meta);
                    results.push(host_ciphertext::<B, _>(&out));
                    assert!(before == host_ciphertext::<B, _>(&a), "plaintext result differs");
                }
            }
            let mut invalid = input;
            invalid.glwe_layout.base2k = (b - 1).into();
            let mut out = fixture_plaintext(module, &invalid, 99);
            let before_out = host_ciphertext::<B, _>(&out);
            assert!(
                with_scratch::<B, _>(B::ckks_extract_pt_tmp_bytes_impl(module, out.max_size()), |scratch| {
                    B::ckks_extract_pt_impl(module, &mut out, &a, scratch)
                })
                .is_err()
            );
            assert!(
                before_out == host_ciphertext::<B, _>(&out),
                "failed extraction mutated output"
            );
        }
    }
    results
}

pub fn test_plaintext_parity<BR, BT, F>(params: CKKSTestParams, r: &Module<BR>, t: &Module<BT>)
where
    BR: Backend<ZnxWord = i64> + CKKSPlaintextZnxImpl,
    BT: Backend<ZnxWord = i64> + CKKSPlaintextZnxImpl,
    Module<BR>: GLWEMaskFill<BR>,
    Module<BT>: GLWEMaskFill<BT>,
{
    let _scalar = std::marker::PhantomData::<F>;
    assert!(exercise(params, r) == exercise(params, t), "plaintext result differs");
}
