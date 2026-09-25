//! Encryption parity given identical realized draws. The caller selects both
//! backends and, when their streams differ, installs a controlled-sampling adapter.
use super::{arithmetic::layout, helpers::*};
use crate::{CKKSInfos, SetCKKSInfos, SlotsKind, oep::CKKSEncryptionImpl, test_suite::CKKSTestParams};
use poulpy_core::{Distribution, EncryptionLayout, GetDistributionMut, layouts::*};
use poulpy_hal::api::VecZnxFillUniformSourceAll;
use poulpy_hal::{
    layouts::{Backend, Module},
    source::Source,
};

fn exercise<B>(params: CKKSTestParams, module: &Module<B>) -> (Vec<Snapshot>, Vec<[u8; 32]>)
where
    B: Backend<ZnxWord = i64> + CKKSEncryptionImpl,
    Module<B>: GLWESecretPreparedFactory<B> + VecZnxFillUniformSourceAll<B>,
{
    let mut results = Vec::new();
    let mut sources = Vec::new();
    {
        let rank = params.rank;
        let mut sk = module.glwe_secret_alloc(rank.into());
        let digits: Vec<i64> = (0..module.n() * rank).map(|i| (i % 3) as i64 - 1).collect();
        B::copy_from_host(&mut sk.data_mut().data, bytemuck::cast_slice(&digits));
        *sk.dist_mut() = Distribution::TernaryProb(2.0 / 3.0);
        let mut prepared = module.glwe_secret_prepared_alloc_from_infos(&sk);
        module.glwe_secret_prepare(&mut prepared, &sk);
        for sparse in [0, 2] {
            for slots in [SlotsKind::Real, SlotsKind::Complex] {
                let ct_layout = layout(params, rank, 4 * params.base2k + 3, params.base2k, sparse, slots);
                let pt_layout = layout(params, 0, params.base2k + 3, params.base2k, sparse, slots);
                let pt = fixture_plaintext(module, &pt_layout, 67);
                let before = snapshot::<B, _>(&pt);
                let enc = EncryptionLayout::new_from_default_sigma(ct_layout.glwe_layout).unwrap();
                let mut ct = fixture_ciphertext(module, &ct_layout, 99);
                let mut e = Source::new([71; 32]);
                let mut a = Source::new([72; 32]);
                with_scratch::<B, _>(B::ckks_encrypt_sk_tmp_bytes_impl(module, &ct), |scratch| {
                    B::ckks_encrypt_sk_impl(module, &mut ct, &pt, &prepared, &enc, &mut e, &mut a, scratch)
                })
                .unwrap();
                assert_eq!(before, snapshot::<B, _>(&pt), "encryption changed plaintext");
                results.push(snapshot::<B, _>(&ct));
                sources.extend([e.new_seed(), a.new_seed()]);
                let before_ct = snapshot::<B, _>(&ct);
                for delta in [params.base2k - 2, params.base2k, params.base2k + 2] {
                    let out_layout = layout(params, 0, delta + 1, delta, sparse, slots);
                    let mut out = fixture_plaintext(module, &out_layout, 98);
                    with_scratch::<B, _>(B::ckks_decrypt_tmp_bytes_impl(module, &out, &ct), |scratch| {
                        B::ckks_decrypt_impl(module, &mut out, &ct, &prepared, scratch)
                    })
                    .unwrap();
                    assert_eq!(out.meta(), out_layout.meta);
                    results.push(snapshot::<B, _>(&out));
                    assert_eq!(before_ct, snapshot::<B, _>(&ct), "decryption changed ciphertext");
                }
                // Preserve a wide allocation while lowering only meaningful
                // precision: extraction still writes every allocated limb.
                let wide_layout = layout(params, 0, 16 * params.base2k, params.base2k, sparse, slots);
                let mut out = fixture_plaintext(module, &wide_layout, 97);
                SetCKKSInfos::set_k(&mut out, (2 * params.base2k).into());
                assert!(out.max_size() > ct.max_size());
                let before_meta = out.meta();
                with_scratch::<B, _>(B::ckks_decrypt_tmp_bytes_impl(module, &out, &ct), |scratch| {
                    B::ckks_decrypt_impl(module, &mut out, &ct, &prepared, scratch)
                })
                .unwrap();
                assert_eq!(out.meta(), before_meta);
                results.push(snapshot::<B, _>(&out));
                assert_eq!(before_ct, snapshot::<B, _>(&ct), "decryption changed ciphertext");
                let mut bad_layout = pt_layout;
                bad_layout.glwe_layout.base2k = (params.base2k - 1).into();
                let mut out = fixture_plaintext(module, &bad_layout, 98);
                let unchanged = snapshot::<B, _>(&out);
                assert!(
                    with_scratch::<B, _>(B::ckks_decrypt_tmp_bytes_impl(module, &out, &ct), |scratch| {
                        B::ckks_decrypt_impl(module, &mut out, &ct, &prepared, scratch)
                    })
                    .is_err()
                );
                assert_eq!(unchanged, snapshot::<B, _>(&out), "failed decryption changed plaintext");
            }
        }
    }
    (results, sources)
}

/// Both backends must receive identical draws, either through matching streams
/// or a caller-installed controlled-sampling scope. Equal seeds alone do not
/// establish that condition. Each backend prepares the same secret independently.
pub fn test_encryption_parity<BR, BT>(params: CKKSTestParams, r: &Module<BR>, t: &Module<BT>)
where
    BR: Backend<ZnxWord = i64> + CKKSEncryptionImpl,
    BT: Backend<ZnxWord = i64> + CKKSEncryptionImpl,
    Module<BR>: GLWESecretPreparedFactory<BR> + VecZnxFillUniformSourceAll<BR>,
    Module<BT>: GLWESecretPreparedFactory<BT> + VecZnxFillUniformSourceAll<BT>,
{
    assert_eq!(exercise(params, r), exercise(params, t));
}

/// Registers encryption parity for caller-selected backends, exposing the tested
/// backend's draws to an optional comparison sampling adapter.
#[macro_export]
macro_rules! ckks_encryption_parity_test_suite {
    (mod $name:ident, backend_ref=$reference:ty, backend_test=$tested:ty, params=$params:expr $(,)?) => {
        mod $name {
            #[test]
            fn encryption() {
                let params = $params;
                ::poulpy_core::test_suite::parity::controlled_sampling::with_backend_samples(
                    ::poulpy_hal::layouts::Module::<$tested>::new(params.n as u64),
                    |tested| {
                        let reference = ::poulpy_hal::layouts::Module::<$reference>::new(params.n as u64);
                        $crate::test_suite::parity::test_encryption_parity::<$reference, $tested>(params, &reference, tested);
                    },
                );
            }
        }
    };
}
