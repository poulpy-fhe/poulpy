//! Fold and unfold round trip, checked on ring coefficients: the fold only moves
//! coefficients, so the folded message has a cleartext model.

use std::collections::HashMap;

use poulpy_core::layouts::{GGLWEInfos, GLWEAutomorphismKeyPrepared, GLWEInfos, GLWELayout, GLWESecretPrepared, LWEInfos};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostBytesBackend, HostDataMut, HostDataRef, Module, ScratchOwned, Standard},
    source::Source,
};

use crate::{
    CKKSInfos, CKKSLayout, CKKSMeta, SetCKKSInfos, SlotsKind,
    api::{CKKSFoldLayoutOps, CKKSFoldOps},
    layouts::{CKKSCiphertextOwned, CKKSFoldKeysLayout, CKKSModuleAlloc, CKKSPlaintextVecHostCodec, RingSwitchKeys},
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextModule, TestContextSharedModule, TestScalar, alloc_scratch, assert_precision,
            ckks_decrypt_with_prec, ckks_encrypt_coeffs, gen_atk, gen_sk_with_raw,
        },
    },
};

type AutomorphismKeys<BE> = HashMap<i64, GLWEAutomorphismKeyPrepared<<BE as Backend>::OwnedBuf, BE>>;

/// Folds a complex input and two real pairs at the input width, decrypts and checks
/// the folded messages, re-encrypts them wider as a bootstrap would, then unfolds and
/// checks the inputs return: under the fold secret at its degree, dense and sparse,
/// and under their own secret at half of it, paired then merged two per folded ciphertext.
pub fn test_fold_unfold<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE> + CKKSFoldLayoutOps<BE> + CKKSFoldOps<BE, Standard>,
    F: TestScalar,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let n = module.n();
    let half_params = CKKSTestParams {
        n: n / 2,
        hw: params.hw.min(n / 2),
        ..params
    };
    let k_in = params.prec().k().as_usize();
    let k_out = 2 * k_in;
    let mut scratch = alloc_scratch(&params, module);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [31; 32]);
    let (half_sk_raw, half_sk) = gen_sk_with_raw(&half_params, module, host_module, [32; 32]);
    let ring_switch = RingSwitchKeys {
        inbound: params.ksk_layout(k_in).layout,
        outbound: params.ksk_layout(k_out).layout,
    }
    .generate(
        module,
        &half_sk_raw,
        &sk_raw,
        &mut Source::new([33; 32]),
        &mut Source::new([34; 32]),
        &mut scratch.borrow(),
    )
    .unwrap()
    .prepare(module, &mut scratch.borrow())
    .unwrap();
    // Conjugation splits real pairs; two sparse elements exercise both split levels.
    let sparse_real = CKKSLayout {
        meta: CKKSMeta {
            log_sparsity: 2,
            slots: SlotsKind::Real,
            ..params.prec_meta
        },
        ..params.prec()
    };
    let automorphisms: AutomorphismKeys<BE> = CKKSFoldOps::<_, Standard>::ckks_unfold_galois_elements(module, &sparse_real)
        .into_iter()
        .map(|p| (p, gen_atk(&params, module, p, &sk_raw, &mut scratch.borrow())))
        .collect();
    let half_automorphisms = HashMap::from([(-1, gen_atk(&half_params, module, -1, &half_sk_raw, &mut scratch.borrow()))]);
    for (input_params, input_sk, ring_switch, automorphisms, log_sparsity) in [
        (&params, &sk, None, Some(&automorphisms), 0),
        (&params, &sk, None, Some(&automorphisms), 2),
        (&half_params, &half_sk, Some(&ring_switch), Some(&half_automorphisms), 0),
    ] {
        let n_in = input_params.n;
        let log_delta = params.prec_meta.log_delta;
        let label = format!("degree {n_in}, sparsity {log_sparsity}");
        let mut scratch = alloc_scratch(input_params, module);
        let slots = [
            SlotsKind::Complex,
            SlotsKind::Real,
            SlotsKind::Real,
            SlotsKind::Real,
            SlotsKind::Real,
        ];
        let msgs: Vec<Vec<F>> = slots
            .iter()
            .enumerate()
            .map(|(seed, &slots)| message(n_in, slots, log_sparsity, seed))
            .collect();
        let ins: Vec<_> = msgs
            .iter()
            .zip(slots)
            .map(|(msg, slots)| {
                let prec = CKKSLayout {
                    meta: CKKSMeta {
                        log_sparsity,
                        slots,
                        ..input_params.prec_meta
                    },
                    ..input_params.prec()
                };
                ckks_encrypt_coeffs(
                    input_params,
                    module,
                    host_module,
                    input_sk,
                    k_in,
                    msg,
                    prec,
                    &mut scratch.borrow(),
                )
            })
            .collect();

        // Real inputs pair at either degree; a folded ciphertext holds
        // `g = n / n_in` positions, with four sparse parts in the same-ring sparse case.
        let units = [(0, None), (1, Some(2)), (3, Some(4))];
        let g = n / n_in;
        let span = g << log_sparsity;
        let degree = n.into();
        let count = module.ckks_fold_count(&ins, degree);
        assert_eq!(count, units.len().div_ceil(span), "folded count, {label}");
        let keys_layout = CKKSFoldKeysLayout {
            ring_switch: ring_switch.map(RingSwitchKeys::gglwe_layout),
            automorphism: automorphisms
                .and_then(|keys| keys.values().next())
                .map(|key| key.gglwe_layout()),
        };
        let output = CKKSLayout {
            glwe_layout: GLWELayout {
                k: k_out.into(),
                ..ins[0].glwe_layout()
            },
            meta: ins[0].meta(),
        };
        let folded_layout = module.ckks_fold_layout(&ins[0], degree, &keys_layout);
        let mut fold_scratch = ScratchOwned::<BE>::alloc(module.ckks_fold_tmp_bytes(&output, &ins[0], degree, &keys_layout));
        let mut folded: Vec<_> = (0..count)
            .map(|_| module.ckks_ciphertext_alloc_from_glwe_infos(&folded_layout))
            .collect();
        module
            .ckks_fold(
                &mut folded,
                &ins,
                ring_switch.map(|keys| &keys.inbound),
                &mut fold_scratch.borrow(),
            )
            .unwrap();

        let mut refresh_scratch = alloc_scratch(&params, module);
        let mut refreshed: Vec<_> = folded
            .iter()
            .zip(units.chunks(span))
            .enumerate()
            .map(|(i, (ct, group))| {
                assert_eq!(
                    ct.meta(),
                    CKKSMeta {
                        log_delta,
                        log_sparsity: 0,
                        slots: SlotsKind::Complex,
                    },
                    "folded metadata {i}, {label}"
                );
                let got = decrypt_coeffs::<BE, F>(module, &params, ct, &sk, &mut refresh_scratch);
                assert_precision(
                    &format!("folded, {label}"),
                    &got,
                    &folded_message(&msgs, group, g, n),
                    log_delta,
                    n,
                );
                let prec = CKKSLayout {
                    meta: ct.meta(),
                    ..params.prec()
                };
                ckks_encrypt_coeffs(
                    &params,
                    module,
                    host_module,
                    &sk,
                    k_out,
                    &got,
                    prec,
                    &mut refresh_scratch.borrow(),
                )
            })
            .collect();

        let mut outs: Vec<_> = ins
            .iter()
            .map(|ct| {
                let mut out = module.ckks_ciphertext_alloc_from_glwe_infos(&output);
                out.set_meta(ct.meta());
                out
            })
            .collect();
        module
            .ckks_unfold(
                &mut outs,
                &mut refreshed,
                ring_switch.map(|keys| &keys.outbound),
                automorphisms,
                &mut fold_scratch.borrow(),
            )
            .unwrap();
        for (i, (out, msg)) in outs.iter().zip(&msgs).enumerate() {
            assert_eq!(out.meta(), ins[i].meta(), "unfolded {i}, {label}");
            let got = decrypt_coeffs::<BE, F>(module, input_params, out, input_sk, &mut scratch);
            assert_precision(&format!("unfolded {i}, {label}"), &got, msg, log_delta, n_in);
        }
    }
}

/// Coefficients of a message of degree `n` in `Z[X^(2^log_sparsity)]`; real slots
/// are self-conjugate, `c_(n−i) = −c_i`.
pub(crate) fn message<F: TestScalar>(n: usize, slots: SlotsKind, log_sparsity: usize, seed: usize) -> Vec<F> {
    let mut c: Vec<f64> = (0..n)
        .map(|i| {
            if i.is_multiple_of(1 << log_sparsity) {
                ((i * 37 + seed * 101) % 255) as f64 / 255.0 - 0.5
            } else {
                0.0
            }
        })
        .collect();
    if slots == SlotsKind::Real {
        c[n / 2] = 0.0;
        for i in n / 2 + 1..n {
            c[i] = -c[n - i];
        }
    }
    c.into_iter().map(|x| F::from_f64(x).unwrap()).collect()
}

/// `Σ_j X^j·(re_j + X^(n/2)·im_j)(X^g)` of degree `n`, over the units of `group`.
pub(crate) fn folded_message<F: TestScalar>(msgs: &[Vec<F>], group: &[(usize, Option<usize>)], g: usize, n: usize) -> Vec<F> {
    let mut want = vec![F::zero(); n];
    for (j, &(re, im)) in group.iter().enumerate() {
        for (offset, part) in [(0, Some(re)), (n / 2, im)] {
            for (i, &c) in part.map_or(&[][..], |part| &msgs[part]).iter().enumerate() {
                // Negacyclic: `X^n = −1`.
                let e = j + i * g + offset;
                want[e % n] = want[e % n] + if (e / n).is_multiple_of(2) { c } else { -c };
            }
        }
    }
    want
}

pub(crate) fn decrypt_coeffs<BE, F>(
    module: &Module<BE>,
    params: &CKKSTestParams,
    ct: &CKKSCiphertextOwned<BE>,
    sk: &GLWESecretPrepared<BE::OwnedBuf, BE>,
    scratch: &mut ScratchOwned<BE>,
) -> Vec<F>
where
    BE: TestContextBackend,
    Module<BE>: TestContextSharedModule<BE>,
    F: TestScalar,
{
    let prec = CKKSLayout {
        meta: ct.meta(),
        ..params.prec()
    };
    let pt = ckks_decrypt_with_prec(module, ct, sk, prec, &mut scratch.borrow()).unwrap();
    let mut coeffs = vec![F::zero(); ct.n().as_usize()];
    pt.decode_host_floats(&mut coeffs).unwrap();
    coeffs
}
