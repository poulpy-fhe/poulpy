//! Tests for `CKKSInnerSumOps`: sums of rotations by multiples of a power-of-two batch, and their reverse.
//!
//! # Test inventory
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_inner_sum`] | inner sums over windows and all the slots, summed or normalized, replicates, out of place and in place, rejections |

use std::collections::HashMap;

use poulpy_core::layouts::{GLWEAutomorphismKeyPrepared, LWEInfos, SetGLWEPlaintextInfos};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module, Standard},
};

use crate::{
    CKKSInfos, SlotsKind,
    api::{CKKSCopyOps, CKKSInnerSumOps, CKKSModuleInfos},
    test_suite::{
        CKKSTestParams,
        parity::helpers::{snapshot, with_scratch},
        reference_encoder::ReferenceEncoder,
    },
};

use super::helpers::{
    TestContextBackend, TestContextModule, TestScalar, alloc_ct, alloc_scratch, assert_ct_meta, assert_decrypt_precision,
    ckks_encrypt, gen_atk, gen_sk_with_raw, test_vector_1, want_rotate,
};

/// `dec(inner_sum(enc(v), log_batch, log_count))` is `sum_{j < 2^log_count} rot(v, j 2^log_batch)`, and under
/// `normalize` its mean at `log_delta + log_count`, and `dec(replicate(enc(v), log_batch, log_count))` is
/// `sum_{j < 2^log_count} rot(v, -j 2^log_batch)`: windows from batches of one and two slots, and the inner sum over
/// all the slots, out of place and in place, each within exactly the scratch its query reports, and into a narrower
/// destination. A missing key, `normalize` on Integer slots and a window past the slots are rejected before `dst`
/// changes.
pub fn test_inner_sum<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let log_m = m.ilog2() as usize;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let (re, im) = test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re,
        &im,
        &mut scratch.borrow(),
    );
    let log_count = 2;
    let batches = [0, 1, log_m - log_count];
    let mut atks = HashMap::new();
    for &log_batch in &batches {
        for i in 0..log_count {
            for sign in [1, -1] {
                let gal = module.ckks_galois_element(sign << (log_batch + i));
                atks.entry(gal)
                    .or_insert_with(|| gen_atk(&params, module, gal, &sk_raw, &mut scratch.borrow()));
            }
        }
    }
    let atk = params.atk_layout();
    let want = |log_batch: usize, sign: i64, normalize: bool| {
        let (mut want_re, mut want_im) = (vec![F::zero(); m], vec![F::zero(); m]);
        for j in 0..1i64 << log_count {
            let (r, i) = want_rotate(&re, &im, sign * (j << log_batch), m);
            for s in 0..m {
                want_re[s] = want_re[s] + r[s];
                want_im[s] = want_im[s] + i[s];
            }
        }
        if normalize {
            let scale = F::from_f64(1.0 / (1u64 << log_count) as f64).unwrap();
            want_re.iter_mut().chain(want_im.iter_mut()).for_each(|x| *x = *x * scale);
        }
        (want_re, want_im)
    };

    for log_batch in batches {
        for (replicate, normalize) in [(false, false), (false, true), (true, false)] {
            let (want_re, want_im) = want(log_batch, if replicate { -1 } else { 1 }, normalize);
            let shift = if normalize { log_count } else { 0 };
            let label = format!("log_batch={log_batch} replicate={replicate} normalize={normalize}");

            let mut ct_res = alloc_ct(&params, module, params.k);
            let bytes = if replicate {
                module.ckks_replicate_into_tmp_bytes(&ct_res, &ct, &atk)
            } else {
                module.ckks_inner_sum_into_tmp_bytes(&ct_res, &ct, &atk)
            };
            with_scratch::<BE, _>(bytes, |s| {
                if replicate {
                    module.ckks_replicate_into(&mut ct_res, &ct, log_batch, log_count, &atks, s)
                } else {
                    module.ckks_inner_sum_into(&mut ct_res, &ct, log_batch, log_count, normalize, &atks, s)
                }
            })
            .unwrap();
            assert_ct_meta(&label, &ct_res, ct.log_delta() + shift, ct.log_budget() - shift);
            assert_decrypt_precision(
                &label,
                &params,
                module,
                &encoder,
                &ct_res,
                &sk,
                &want_re,
                &want_im,
                &mut scratch.borrow(),
            );

            let mut ct_assign = alloc_ct(&params, module, params.k);
            module.ckks_copy(&mut ct_assign, &ct, &mut scratch.borrow()).unwrap();
            let bytes = if replicate {
                module.ckks_replicate_assign_tmp_bytes(&ct_assign, &atk)
            } else {
                module.ckks_inner_sum_assign_tmp_bytes(&ct_assign, &atk)
            };
            with_scratch::<BE, _>(bytes, |s| {
                if replicate {
                    module.ckks_replicate_assign(&mut ct_assign, log_batch, log_count, &atks, s)
                } else {
                    module.ckks_inner_sum_assign(&mut ct_assign, log_batch, log_count, normalize, &atks, s)
                }
            })
            .unwrap();
            assert!(
                snapshot::<BE, _>(&ct_assign) == snapshot::<BE, _>(&ct_res),
                "{label}: in place differs from out of place"
            );
        }
    }

    // Into a destination one limb and a bit narrower than `ct`: the copy spends the difference, at whose width every key
    // is looked up.
    let narrow_k = params.k - params.base2k - 1;
    let offset = ct.k().as_usize() - narrow_k;
    let (want_re, want_im) = want(0, 1, true);
    let mut ct_narrow = alloc_ct(&params, module, narrow_k);
    with_scratch::<BE, _>(module.ckks_inner_sum_into_tmp_bytes(&ct_narrow, &ct, &atk), |s| {
        module.ckks_inner_sum_into(&mut ct_narrow, &ct, 0, log_count, true, &atks, s)
    })
    .unwrap();
    assert_ct_meta(
        "inner_sum narrower",
        &ct_narrow,
        ct.log_delta() + log_count,
        ct.log_budget() - offset - log_count,
    );
    assert_decrypt_precision(
        "inner_sum narrower",
        &params,
        module,
        &encoder,
        &ct_narrow,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );

    let mut ct_res = alloc_ct(&params, module, params.k);
    let before = snapshot::<BE, _>(&ct_res);
    let no_keys: HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>> = HashMap::new();
    for (label, result) in [
        (
            "a missing key",
            module.ckks_inner_sum_into(&mut ct_res, &ct, 0, log_count, false, &no_keys, &mut scratch.borrow()),
        ),
        (
            "a window past the slots",
            module.ckks_replicate_into(&mut ct_res, &ct, log_m - 1, log_count, &atks, &mut scratch.borrow()),
        ),
    ] {
        assert!(result.is_err(), "{label} was accepted");
        assert!(snapshot::<BE, _>(&ct_res) == before, "{label} wrote dst");
    }
    let mut ct_int = alloc_ct(&params, module, params.k);
    module.ckks_copy(&mut ct_int, &ct, &mut scratch.borrow()).unwrap();
    ct_int.set_slots(SlotsKind::Integer);
    let before = snapshot::<BE, _>(&ct_int);
    assert!(
        module
            .ckks_inner_sum_assign(&mut ct_int, 0, log_count, true, &atks, &mut scratch.borrow())
            .is_err(),
        "normalize on Integer slots was accepted"
    );
    assert!(snapshot::<BE, _>(&ct_int) == before, "normalize on Integer slots wrote dst");
}
