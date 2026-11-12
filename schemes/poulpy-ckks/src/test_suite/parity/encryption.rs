//! Encryption parity given identical realized draws. The caller selects both
//! backends and, when their streams differ, installs a controlled-sampling adapter.
use super::{arithmetic::layout, helpers::*};
use crate::{CKKSInfos, SlotLayout, SlotsKind, oep::CKKSEncryptionImpl, shaping::ShapingUnit, test_suite::CKKSTestParams};
use poulpy_core::{Distribution, GLWEFill, GLWESecretPreparedFactory, GetDistributionMut, layouts::*};
use poulpy_hal::{
    layouts::{Backend, Module, Ring},
    source::Source,
};

fn exercise<BE, Rg: Ring>(params: CKKSTestParams, module: &Module<BE>) -> (Vec<Snapshot>, Vec<[u8; 32]>)
where
    BE: Backend<ZnxWord = i64> + CKKSEncryptionImpl<Rg>,
    Module<BE>: GLWESecretPreparedFactory<BE, Rg> + GLWEFill<BE, Rg>,
{
    let mut results = Vec::new();
    let mut sources = Vec::new();
    {
        let rank = params.rank;
        let mut sk = module.glwe_secret_alloc(rank.into());
        let digits: Vec<i64> = (0..module.n() * rank).map(|i| (i % 3) as i64 - 1).collect();
        BE::copy_from_host(&mut sk.data_mut().data, bytemuck::cast_slice(&digits));
        *sk.dist_mut() = Distribution::TernaryProb(2.0 / 3.0);
        let mut prepared = module.glwe_secret_prepared_alloc_from_infos(&sk);
        module.glwe_secret_prepare(&mut prepared, &sk);
        for sparse in [0, 2] {
            for slots in [SlotsKind::Real, SlotsKind::Complex] {
                let ct_layout = layout(params, rank, 4 * params.base2k + 3, params.base2k, sparse, slots);
                // A plaintext below the ring degree: encryption reads its sparsity off the degrees.
                let mut pt_layout = layout(params, 0, params.base2k + 3, params.base2k, sparse, slots);
                pt_layout.glwe_layout.n = (params.n >> sparse).into();
                let pt = fixture_plaintext(module, &pt_layout, 67);
                let before = snapshot::<BE, _, Rg>(&pt);

                let mut ct = fixture_ciphertext(module, &ct_layout, 99);
                let mut e = Source::new([71; 32]);
                let mut a = Source::new([72; 32]);
                with_scratch::<BE, _>(BE::ckks_encrypt_sk_tmp_bytes(module, &ct), |scratch| {
                    BE::ckks_encrypt_sk(module, &mut ct, &pt, &prepared, &mut e, &mut a, scratch)
                })
                .unwrap();
                assert_eq!(before, snapshot::<BE, _, Rg>(&pt), "encryption changed plaintext");
                results.push(snapshot::<BE, _, Rg>(&ct));
                sources.extend([e.new_seed(), a.new_seed()]);
                let before_ct = snapshot::<BE, _, Rg>(&ct);
                for delta in [params.base2k - 2, params.base2k, params.base2k + 2] {
                    let out_layout = layout(params, 0, delta + 1, delta, sparse, slots);
                    let mut out = fixture_plaintext(module, &out_layout, 98);
                    with_scratch::<BE, _>(BE::ckks_decrypt_tmp_bytes(module, &out, &ct), |scratch| {
                        BE::ckks_decrypt(module, &mut out, &ct, &prepared, scratch)
                    })
                    .unwrap();
                    assert_eq!(out.meta(), out_layout.meta);
                    results.push(snapshot::<BE, _, Rg>(&out));
                    assert_eq!(before_ct, snapshot::<BE, _, Rg>(&ct), "decryption changed ciphertext");
                }
                // Preserve a wide allocation while lowering only meaningful
                // precision: extraction still writes every allocated limb.
                let wide_layout = layout(params, 0, 16 * params.base2k, params.base2k, sparse, slots);
                let mut out = fixture_plaintext(module, &wide_layout, 97);
                SetK::set_k(&mut out, (2 * params.base2k).into());
                assert!(out.max_size() > ct.max_size());
                let before_meta = out.meta();
                with_scratch::<BE, _>(BE::ckks_decrypt_tmp_bytes(module, &out, &ct), |scratch| {
                    BE::ckks_decrypt(module, &mut out, &ct, &prepared, scratch)
                })
                .unwrap();
                assert_eq!(out.meta(), before_meta);
                results.push(snapshot::<BE, _, Rg>(&out));
                assert_eq!(before_ct, snapshot::<BE, _, Rg>(&ct), "decryption changed ciphertext");
                let mut bad_layout = pt_layout;
                bad_layout.glwe_layout.base2k = (params.base2k - 1).into();
                let mut out = fixture_plaintext(module, &bad_layout, 98);
                let unchanged = snapshot::<BE, _, Rg>(&out);
                assert!(
                    with_scratch::<BE, _>(BE::ckks_decrypt_tmp_bytes(module, &out, &ct), |scratch| {
                        BE::ckks_decrypt(module, &mut out, &ct, &prepared, scratch)
                    })
                    .is_err()
                );
                assert_eq!(unchanged, snapshot::<BE, _, Rg>(&out), "failed decryption changed plaintext");
            }
        }
        // A replicated plaintext encrypts `max_pass_bits` wider, then rounds with shaping.
        let ct_layout = layout(params, rank, 4 * params.base2k + 3, params.base2k, 0, SlotsKind::Complex);
        let mut pt_layout = layout(params, 0, params.base2k + 3, params.base2k, 0, SlotsKind::Complex);
        pt_layout.meta.slot_layout = SlotLayout::Replicated(ShapingUnit::U16_20.layout(1).unwrap());
        let pt = fixture_plaintext(module, &pt_layout, 66);
        let mut ct = fixture_ciphertext(module, &ct_layout, 96);
        let mut e = Source::new([73; 32]);
        let mut a = Source::new([74; 32]);
        with_scratch::<BE, _>(BE::ckks_encrypt_sk_tmp_bytes(module, &ct), |scratch| {
            BE::ckks_encrypt_sk(module, &mut ct, &pt, &prepared, &mut e, &mut a, scratch)
        })
        .unwrap();
        assert_eq!(ct.meta().slot_layout, pt_layout.meta.slot_layout);
        results.push(snapshot::<BE, _, Rg>(&ct));
        sources.extend([e.new_seed(), a.new_seed()]);
    }
    (results, sources)
}

/// Both backends must receive identical draws, either through matching streams
/// or a caller-installed controlled-sampling scope. Equal seeds alone do not
/// establish that condition. Each backend prepares the same secret independently.
pub fn test_encryption_parity<BR, BT, Rg: Ring>(params: CKKSTestParams, module_ref: &Module<BR>, module_test: &Module<BT>)
where
    BR: Backend<ZnxWord = i64> + CKKSEncryptionImpl<Rg>,
    BT: Backend<ZnxWord = i64> + CKKSEncryptionImpl<Rg>,
    Module<BR>: GLWESecretPreparedFactory<BR, Rg> + GLWEFill<BR, Rg>,
    Module<BT>: GLWESecretPreparedFactory<BT, Rg> + GLWEFill<BT, Rg>,
{
    assert_eq!(exercise(params, module_ref), exercise(params, module_test));
}

/// Registers encryption parity for caller-selected backends, exposing the tested
/// backend's draws to an optional comparison sampling adapter. The optional
/// `ring = ..` argument is `::poulpy_hal::layouts::Standard` when absent, and the
/// conjugate-invariant lists pass `ring = ::poulpy_hal::layouts::ConjugateInvariant`.
#[macro_export]
macro_rules! ckks_encryption_parity_test_suite {
    (mod $name:ident, backend_ref=$reference:ty, backend_test=$tested:ty, params=$params:expr $(,)?) => {
        $crate::ckks_encryption_parity_test_suite!(
            mod $name,
            backend_ref = $reference,
            backend_test = $tested,
            ring = ::poulpy_hal::layouts::Standard,
            params = $params,
        );
    };
    (mod $name:ident, backend_ref=$reference:ty, backend_test=$tested:ty, ring=$ring:ty, params=$params:expr $(,)?) => {
        mod $name {
            #[test]
            fn encrypt_sk_rejects_degree_mismatch() {
                $crate::test_suite::encryption::test_encrypt_sk_rejects_degree_mismatch::<$tested, $ring>();
            }

            #[test]
            fn encryption() {
                let params = $params;
                ::poulpy_core::test_suite::parity::controlled_sampling::with_backend_samples(
                    ::poulpy_hal::layouts::Module::<$tested>::new(params.n as u64),
                    |tested| {
                        let reference = ::poulpy_hal::layouts::Module::<$reference>::new(params.n as u64);
                        $crate::test_suite::parity::test_encryption_parity::<$reference, $tested, $ring>(
                            params, &reference, tested,
                        );
                    },
                );
            }
        }
    };
}
