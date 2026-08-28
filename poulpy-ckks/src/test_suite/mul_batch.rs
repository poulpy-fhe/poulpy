//! Dependency-frontier batch multiplication: exact parity with the ordered
//! scalar operations, at batch lengths 0 to 4.
//!
//! Cross-item destination aliasing needs no runtime test: the batch takes `&mut`
//! destinations, so it is a borrow-check error. That is pinned by the
//! `compile_fail` doctest on
//! [`CKKSMulOps::ckks_mul_into_batch`](crate::api::CKKSMulOps::ckks_mul_into_batch).

use poulpy_core::layouts::{Dsize, GGLWEInfos, GLWERelinearizationKeyHelper, GLWERelinearizationKeyLayoutHelper, TorusPrecision};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module, ScratchOwned},
};

use crate::{
    CKKSCompositionError, CKKSInfos, SetCKKSInfos,
    api::{CKKSMulIntoItem, CKKSMulOps, CKKSPreparedMulAssignItem, CKKSSquareAssignItem, CKKSSquareIntoItem},
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc, CKKSPreparedRight},
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextModule, TestScalar, alloc_ct, alloc_scratch, assert_ckks_error, ckks_encrypt,
            gen_sk_with_raw, gen_tsk,
        },
        polynomial_evaluation::assert_ct_identical,
        reference_encoder::ReferenceEncoder,
    },
};

struct TwoRelinearizationKeys<K> {
    pivot: TorusPrecision,
    pivot_key: K,
    other_key: K,
}

impl<K> TwoRelinearizationKeys<K> {
    fn key_for(&self, k: TorusPrecision) -> &K {
        if k == self.pivot { &self.pivot_key } else { &self.other_key }
    }
}

impl<K: GGLWEInfos> GLWERelinearizationKeyHelper for TwoRelinearizationKeys<K> {
    type Key = K;

    fn get_relinearization_key_for(&self, k: TorusPrecision) -> poulpy_core::Result<(&K, Dsize)> {
        let key = self.key_for(k);
        Ok((key, key.effective_dsize()))
    }
}

impl<K: GGLWEInfos> GLWERelinearizationKeyLayoutHelper for TwoRelinearizationKeys<K> {
    type Layout = K;

    fn get_relinearization_key_layout_for(&self, k: TorusPrecision) -> poulpy_core::Result<(&K, Dsize)> {
        let key = self.key_for(k);
        Ok((key, key.effective_dsize()))
    }
}

/// Four source ciphertexts at three different widths, so the items in a batch
/// carry different effective widths and therefore different convolution offsets.
const WIDTH_STEPS: [usize; 4] = [0, 1, 2, 0];

/// Every batch operation agrees with its ordered scalar counterpart, limb for
/// limb and metadata for metadata, at lengths 0 through 4.
///
/// Coverage in one sweep: heterogeneous widths, a read-only operand repeated
/// across items, a prepared operand repeated across items, and the intentional
/// intra-item aliasing of the two `_assign` forms.
pub fn test_mul_batches_match_scalar<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<ZnxWord = i64>,
    BE::OwnedBuf: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    // Two relinearization-key shapes: the suite's own `dsize` and the active
    // EvalMod one, which gives a different digit count.
    for dsize in [params.dsize, 8] {
        mul_batches_case::<BE, F, E>(
            CKKSTestParams { dsize, ..params },
            module,
            host_module,
            &format!("dsize={dsize}"),
        );
    }
}

fn mul_batches_case<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>, label: &str)
where
    BE: TestContextBackend<ZnxWord = i64>,
    BE::OwnedBuf: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let (re, im) = super::helpers::test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let sources: Vec<CKKSCiphertextOwned<BE>> = WIDTH_STEPS
        .iter()
        .map(|step| {
            ckks_encrypt(
                &params,
                module,
                host_module,
                &encoder,
                &sk,
                params.k - step * params.base2k,
                &re,
                &im,
                &mut scratch.borrow(),
            )
        })
        .collect();
    let prepared: Vec<CKKSPreparedRight<BE>> = sources
        .iter()
        .map(|source| module.ckks_prepare_right(source, &mut scratch.borrow()).unwrap())
        .collect();

    for len in 0..=4usize {
        // ── mul_into: item `i` multiplies source `i` by the shared source 0,
        // so one read-only operand repeats across every item.
        let mut want: Vec<CKKSCiphertextOwned<BE>> = (0..len).map(|_| alloc_ct(&params, module, params.k)).collect();
        for (i, dst) in want.iter_mut().enumerate() {
            module
                .ckks_mul_into(dst, &sources[i], &sources[0], &tsk, &mut scratch.borrow())
                .unwrap();
        }
        let mut have: Vec<CKKSCiphertextOwned<BE>> = (0..len).map(|_| alloc_ct(&params, module, params.k)).collect();
        {
            let mut items: Vec<CKKSMulIntoItem<&mut CKKSCiphertextOwned<BE>, _, _>> = have
                .iter_mut()
                .enumerate()
                .map(|(i, dst)| CKKSMulIntoItem {
                    dst,
                    a: &sources[i],
                    b: &sources[0],
                })
                .collect();
            module.ckks_mul_into_batch(&mut items, &tsk, &mut scratch.borrow()).unwrap();
        }
        for i in 0..len {
            assert_ct_identical::<BE>(&format!("{label}: mul_into_batch len={len} item={i}"), &want[i], &have[i]);
        }

        // ── square_into
        let mut want: Vec<CKKSCiphertextOwned<BE>> = (0..len).map(|_| alloc_ct(&params, module, params.k)).collect();
        for (i, dst) in want.iter_mut().enumerate() {
            module
                .ckks_square_into(dst, &sources[i], &tsk, &mut scratch.borrow())
                .unwrap();
        }
        let mut have: Vec<CKKSCiphertextOwned<BE>> = (0..len).map(|_| alloc_ct(&params, module, params.k)).collect();
        {
            let mut items: Vec<CKKSSquareIntoItem<&mut CKKSCiphertextOwned<BE>, _>> = have
                .iter_mut()
                .enumerate()
                .map(|(i, dst)| CKKSSquareIntoItem { dst, a: &sources[i] })
                .collect();
            module
                .ckks_square_into_batch(&mut items, &tsk, &mut scratch.borrow())
                .unwrap();
        }
        for i in 0..len {
            assert_ct_identical::<BE>(&format!("{label}: square_into_batch len={len} item={i}"), &want[i], &have[i]);
        }

        // ── square_assign: every item aliases its own destination on purpose.
        let mut want: Vec<CKKSCiphertextOwned<BE>> = sources[..len].iter().map(clone_ct::<BE>).collect();
        for dst in want.iter_mut() {
            module.ckks_square_assign(dst, &tsk, &mut scratch.borrow()).unwrap();
        }
        let mut have: Vec<CKKSCiphertextOwned<BE>> = sources[..len].iter().map(clone_ct::<BE>).collect();
        {
            let mut items: Vec<CKKSSquareAssignItem<&mut CKKSCiphertextOwned<BE>>> =
                have.iter_mut().map(|dst| CKKSSquareAssignItem { dst }).collect();
            module
                .ckks_square_assign_batch(&mut items, &tsk, &mut scratch.borrow())
                .unwrap();
        }
        for i in 0..len {
            assert_ct_identical::<BE>(
                &format!("{label}: square_assign_batch len={len} item={i}"),
                &want[i],
                &have[i],
            );
        }

        // ── mul_prepared_assign: prepared operand 0 repeats on the even items.
        let prepared_for = |i: usize| if i.is_multiple_of(2) { &prepared[0] } else { &prepared[i] };
        let mut want: Vec<CKKSCiphertextOwned<BE>> = sources[..len].iter().map(clone_ct::<BE>).collect();
        for (i, dst) in want.iter_mut().enumerate() {
            module
                .ckks_mul_prepared_assign(dst, prepared_for(i), &tsk, &mut scratch.borrow())
                .unwrap();
        }
        let mut have: Vec<CKKSCiphertextOwned<BE>> = sources[..len].iter().map(clone_ct::<BE>).collect();
        {
            let mut items: Vec<CKKSPreparedMulAssignItem<&mut CKKSCiphertextOwned<BE>, &CKKSPreparedRight<BE>>> = have
                .iter_mut()
                .enumerate()
                .map(|(i, dst)| CKKSPreparedMulAssignItem {
                    dst,
                    prepared: prepared_for(i),
                })
                .collect();
            module
                .ckks_mul_prepared_assign_batch(&mut items, &tsk, &mut scratch.borrow())
                .unwrap();
        }
        for i in 0..len {
            assert_ct_identical::<BE>(
                &format!("{label}: mul_prepared_assign_batch len={len} item={i}"),
                &want[i],
                &have[i],
            );
        }
    }
}

/// Every batch runs inside exactly the bytes its `*_batch_tmp_bytes` advertises.
pub fn test_mul_batches_exact_scratch<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<ZnxWord = i64>,
    BE::OwnedBuf: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let (re, im) = super::helpers::test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let sources: Vec<CKKSCiphertextOwned<BE>> = WIDTH_STEPS
        .iter()
        .map(|step| {
            ckks_encrypt(
                &params,
                module,
                host_module,
                &encoder,
                &sk,
                params.k - step * params.base2k,
                &re,
                &im,
                &mut scratch.borrow(),
            )
        })
        .collect();
    let prepared = module.ckks_prepare_right(&sources[1], &mut scratch.borrow()).unwrap();

    let mut dst: Vec<CKKSCiphertextOwned<BE>> = (0..4).map(|_| alloc_ct(&params, module, params.k)).collect();

    let query: Vec<CKKSMulIntoItem<&CKKSCiphertextOwned<BE>, _, _>> = dst
        .iter()
        .enumerate()
        .map(|(i, d)| CKKSMulIntoItem {
            dst: d,
            a: &sources[i],
            b: &sources[0],
        })
        .collect();
    let bytes = module.ckks_mul_into_batch_tmp_bytes(&query, &tsk);
    drop(query);
    let mut exact = ScratchOwned::<BE>::alloc(bytes);
    let mut items: Vec<CKKSMulIntoItem<&mut CKKSCiphertextOwned<BE>, _, _>> = dst
        .iter_mut()
        .enumerate()
        .map(|(i, d)| CKKSMulIntoItem {
            dst: d,
            a: &sources[i],
            b: &sources[0],
        })
        .collect();
    module.ckks_mul_into_batch(&mut items, &tsk, &mut exact.borrow()).unwrap();
    drop(items);

    let query: Vec<CKKSSquareIntoItem<&CKKSCiphertextOwned<BE>, _>> = dst
        .iter()
        .enumerate()
        .map(|(i, d)| CKKSSquareIntoItem { dst: d, a: &sources[i] })
        .collect();
    let bytes = module.ckks_square_into_batch_tmp_bytes(&query, &tsk);
    drop(query);
    let mut exact = ScratchOwned::<BE>::alloc(bytes);
    let mut items: Vec<CKKSSquareIntoItem<&mut CKKSCiphertextOwned<BE>, _>> = dst
        .iter_mut()
        .enumerate()
        .map(|(i, d)| CKKSSquareIntoItem { dst: d, a: &sources[i] })
        .collect();
    module.ckks_square_into_batch(&mut items, &tsk, &mut exact.borrow()).unwrap();
    drop(items);

    let mut assign: Vec<CKKSCiphertextOwned<BE>> = sources.iter().map(clone_ct::<BE>).collect();
    let query: Vec<CKKSSquareAssignItem<&CKKSCiphertextOwned<BE>>> =
        assign.iter().map(|d| CKKSSquareAssignItem { dst: d }).collect();
    let bytes = module.ckks_square_assign_batch_tmp_bytes(&query, &tsk);
    drop(query);
    let mut exact = ScratchOwned::<BE>::alloc(bytes);
    let mut items: Vec<CKKSSquareAssignItem<&mut CKKSCiphertextOwned<BE>>> =
        assign.iter_mut().map(|d| CKKSSquareAssignItem { dst: d }).collect();
    module
        .ckks_square_assign_batch(&mut items, &tsk, &mut exact.borrow())
        .unwrap();
    drop(items);

    let mut assign: Vec<CKKSCiphertextOwned<BE>> = sources.iter().map(clone_ct::<BE>).collect();
    let query: Vec<CKKSPreparedMulAssignItem<&CKKSCiphertextOwned<BE>, &CKKSPreparedRight<BE>>> = assign
        .iter()
        .map(|d| CKKSPreparedMulAssignItem {
            dst: d,
            prepared: &prepared,
        })
        .collect();
    let bytes = module.ckks_mul_prepared_assign_batch_tmp_bytes(&query, &tsk);
    drop(query);
    let mut exact = ScratchOwned::<BE>::alloc(bytes);
    let mut items: Vec<CKKSPreparedMulAssignItem<&mut CKKSCiphertextOwned<BE>, &CKKSPreparedRight<BE>>> = assign
        .iter_mut()
        .map(|d| CKKSPreparedMulAssignItem {
            dst: d,
            prepared: &prepared,
        })
        .collect();
    module
        .ckks_mul_prepared_assign_batch(&mut items, &tsk, &mut exact.borrow())
        .unwrap();
}

/// A prepared operand of the wrong layout is rejected by the batch, and no
/// destination is touched: validation runs over the whole slice first.
pub fn test_mul_prepared_assign_batch_rejects_layout_mismatch<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<ZnxWord = i64>,
    BE::OwnedBuf: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let (re, im) = super::helpers::test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let source = ckks_encrypt(
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
    let good = module.ckks_prepare_right(&source, &mut scratch.borrow()).unwrap();

    // Prepared from an operand with a different limb radix.
    let mismatched_base2k = params.base2k / 2;
    let mut other = module.ckks_ciphertext_alloc(mismatched_base2k.into(), params.k.into());
    other.set_meta(params.prec().meta);
    let bad = module.ckks_prepare_right(&other, &mut scratch.borrow()).unwrap();

    let mut dst: Vec<CKKSCiphertextOwned<BE>> = (0..2).map(|_| clone_ct::<BE>(&source)).collect();
    let untouched: Vec<CKKSCiphertextOwned<BE>> = dst.iter().map(clone_ct::<BE>).collect();

    // The invalid operand is the *last* item: the first must not be mutated.
    let err = {
        let (head, tail) = dst.split_at_mut(1);
        let mut items = [
            CKKSPreparedMulAssignItem {
                dst: &mut head[0],
                prepared: &good,
            },
            CKKSPreparedMulAssignItem {
                dst: &mut tail[0],
                prepared: &bad,
            },
        ];
        module
            .ckks_mul_prepared_assign_batch(&mut items, &tsk, &mut scratch.borrow())
            .unwrap_err()
    };
    assert_ckks_error(
        "mul_prepared_assign_batch_layout_mismatch",
        &err,
        CKKSCompositionError::PreparedOperandLayoutMismatch {
            op: "mul_prepared",
            dst_n: params.n,
            dst_base2k: params.base2k,
            dst_rank: 1,
            prep_n: params.n,
            prep_base2k: mismatched_base2k,
            prep_rank: 1,
        },
    );
    for i in 0..2 {
        assert_ct_identical::<BE>(
            &format!("mul_prepared_assign_batch validation-before-mutation item={i}"),
            &untouched[i],
            &dst[i],
        );
    }
}

/// A mixed frontier may resolve equal-shaped lanes to different physical keys,
/// and may contain an Empty lane. The batch must preserve each exact pairing,
/// dispatch only active lanes through the batch OEP, and match scalar calls.
pub fn test_mul_batch_distinct_keys_and_empty_lane<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<ZnxWord = i64>,
    BE::OwnedBuf: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let (re, im) = super::helpers::test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    // Independently allocated prepared keys with exactly the same physical
    // shape. Pointer identity must not enter the batch contract.
    let keys = TwoRelinearizationKeys {
        pivot: params.k.into(),
        pivot_key: gen_tsk(&params, module, &sk_raw, &mut scratch.borrow()),
        other_key: gen_tsk(&params, module, &sk_raw, &mut scratch.borrow()),
    };
    let full = ckks_encrypt(
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
    let narrow = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k - params.base2k,
        &re,
        &im,
        &mut scratch.borrow(),
    );
    let mut zero = clone_ct::<BE>(&narrow);
    zero.set_log_delta(0);
    zero.set_log_budget(0);
    let sources = [&full, &narrow, &zero];

    let mut want: Vec<_> = (0..sources.len()).map(|_| alloc_ct(&params, module, params.k)).collect();
    for (dst, source) in want.iter_mut().zip(sources) {
        module.ckks_square_into(dst, source, &keys, &mut scratch.borrow()).unwrap();
    }

    let mut have: Vec<_> = (0..sources.len()).map(|_| alloc_ct(&params, module, params.k)).collect();
    let query: Vec<_> = have
        .iter()
        .zip(sources)
        .map(|(dst, source)| CKKSSquareIntoItem { dst, a: source })
        .collect();
    let bytes = module.ckks_square_into_batch_tmp_bytes(&query, &keys);
    drop(query);
    let mut exact = ScratchOwned::<BE>::alloc(bytes.max(1));
    let mut items: Vec<_> = have
        .iter_mut()
        .zip(sources)
        .map(|(dst, source)| CKKSSquareIntoItem { dst, a: source })
        .collect();
    module.ckks_square_into_batch(&mut items, &keys, &mut exact.borrow()).unwrap();
    drop(items);

    for i in 0..sources.len() {
        assert_ct_identical::<BE>(&format!("distinct physical keys / Empty lane item={i}"), &want[i], &have[i]);
    }
}

/// A late prepared-multiply budget failure is discovered during the global
/// planning pass, before even the first valid destination is touched.
pub fn test_mul_prepared_assign_batch_rejects_late_budget_without_writes<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<ZnxWord = i64>,
    BE::OwnedBuf: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let (re, im) = super::helpers::test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let source = ckks_encrypt(
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
    let prepared = module.ckks_prepare_right(&source, &mut scratch.borrow()).unwrap();

    let mut dst = [clone_ct::<BE>(&source), clone_ct::<BE>(&source)];
    dst[1].set_log_budget(8);
    let untouched = [clone_ct::<BE>(&dst[0]), clone_ct::<BE>(&dst[1])];
    let err = {
        let (head, tail) = dst.split_at_mut(1);
        let mut items = [
            CKKSPreparedMulAssignItem {
                dst: &mut head[0],
                prepared: &prepared,
            },
            CKKSPreparedMulAssignItem {
                dst: &mut tail[0],
                prepared: &prepared,
            },
        ];
        module
            .ckks_mul_prepared_assign_batch(&mut items, &tsk, &mut scratch.borrow())
            .unwrap_err()
    };
    assert_ckks_error(
        "mul_prepared_assign_batch_late_budget",
        &err,
        CKKSCompositionError::MultiplicationPrecisionUnderflow {
            op: "mul",
            lhs_log_budget: 8,
            rhs_log_budget: source.log_budget(),
            lhs_log_delta: source.log_delta(),
            rhs_log_delta: source.log_delta(),
        },
    );
    for i in 0..dst.len() {
        assert_ct_identical::<BE>(&format!("late prepared budget preserves item={i}"), &untouched[i], &dst[i]);
    }
}

/// A helper may resolve a valid key for an early lane and an insufficiently
/// deep key for a later one. Every key is bound before dispatch, so the later
/// coverage failure leaves both destinations byte-for-byte unchanged.
pub fn test_mul_batch_rejects_late_non_covering_key_without_writes<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<ZnxWord = i64>,
    BE::OwnedBuf: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let (re, im) = super::helpers::test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let full_key = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let digit = params
        .dsize
        .checked_mul(params.base2k)
        .expect("test decomposition digit overflows");
    let requested = params.k.div_ceil(digit);
    assert!(requested >= 2, "test parameters need at least two tensor-key digits");
    let short_params = CKKSTestParams {
        // `tsk_layout` adds one digit before dividing. This leaves exactly
        // `requested - 1` stored rows for a `requested`-digit input.
        k: (requested - 2) * digit,
        ..params
    };
    let short_key = gen_tsk(&short_params, module, &sk_raw, &mut scratch.borrow());

    let narrow_k = params.k - params.base2k;
    let keys = TwoRelinearizationKeys {
        pivot: narrow_k.into(),
        pivot_key: full_key,
        other_key: short_key,
    };
    let narrow = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        narrow_k,
        &re,
        &im,
        &mut scratch.borrow(),
    );
    let full = ckks_encrypt(
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

    // Scalar execution and its singleton batch have the same covering-key
    // contract. Neither may fall through to the permissive raw-key prefix use.
    let mut scalar_dst = alloc_ct(&params, module, params.k);
    let scalar_untouched = clone_ct::<BE>(&scalar_dst);
    let scalar_err = module
        .ckks_square_into(&mut scalar_dst, &full, &keys, &mut scratch.borrow())
        .unwrap_err();
    let scalar_message = scalar_err.to_string();
    assert!(
        scalar_message.contains("short of") && scalar_message.contains(&format!("input_k={}", params.k)),
        "unexpected scalar key-coverage error: {scalar_message}"
    );
    assert_ct_identical::<BE>(
        "scalar non-covering key preserves destination",
        &scalar_untouched,
        &scalar_dst,
    );

    let mut singleton_dst = alloc_ct(&params, module, params.k);
    let singleton_untouched = clone_ct::<BE>(&singleton_dst);
    let singleton_err = {
        let mut items = [CKKSSquareIntoItem {
            dst: &mut singleton_dst,
            a: &full,
        }];
        module
            .ckks_square_into_batch(&mut items, &keys, &mut scratch.borrow())
            .unwrap_err()
    };
    let singleton_message = singleton_err.to_string();
    assert!(
        singleton_message.contains("short of") && singleton_message.contains(&format!("input_k={}", params.k)),
        "unexpected singleton key-coverage error: {singleton_message}"
    );
    assert_ct_identical::<BE>(
        "singleton non-covering key preserves destination",
        &singleton_untouched,
        &singleton_dst,
    );

    let scalar_query = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        module.ckks_square_tmp_bytes(&scalar_dst, &full, &keys)
    }));
    assert!(scalar_query.is_err(), "scalar scratch query accepted a non-covering key");
    let singleton_query = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let items = [CKKSSquareIntoItem {
            dst: &singleton_dst,
            a: &full,
        }];
        module.ckks_square_into_batch_tmp_bytes(&items, &keys)
    }));
    assert!(
        singleton_query.is_err(),
        "singleton scratch query accepted a non-covering key"
    );

    let mut dst = [alloc_ct(&params, module, params.k), alloc_ct(&params, module, params.k)];
    let untouched = [clone_ct::<BE>(&dst[0]), clone_ct::<BE>(&dst[1])];
    let err = {
        let (head, tail) = dst.split_at_mut(1);
        let mut items = [
            CKKSSquareIntoItem {
                dst: &mut head[0],
                a: &narrow,
            },
            CKKSSquareIntoItem {
                dst: &mut tail[0],
                a: &full,
            },
        ];
        module
            .ckks_square_into_batch(&mut items, &keys, &mut scratch.borrow())
            .unwrap_err()
    };
    let message = err.to_string();
    assert!(
        message.contains("short of") && message.contains(&format!("input_k={}", params.k)),
        "unexpected late key-coverage error: {message}"
    );
    for i in 0..dst.len() {
        assert_ct_identical::<BE>(&format!("late non-covering key preserves item={i}"), &untouched[i], &dst[i]);
    }
}

/// Prepared widths are `usize` in cached metadata but exact tensor precision
/// is `u32`-backed. Both query and execution reject an unrepresentable late
/// lane before writing the earlier valid lane.
pub fn test_mul_prepared_batch_rejects_precision_overflow_without_writes<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<ZnxWord = i64>,
    BE::OwnedBuf: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let Some(overflow_k) = (u32::MAX as usize).checked_add(1) else {
        return;
    };
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let (re, im) = super::helpers::test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let source = ckks_encrypt(
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
    let good = module.ckks_prepare_right(&source, &mut scratch.borrow()).unwrap();
    let mut bad = module.ckks_prepare_right(&source, &mut scratch.borrow()).unwrap();
    bad.k = overflow_k;

    let mut dst = [clone_ct::<BE>(&source), clone_ct::<BE>(&source)];
    let query = [
        CKKSPreparedMulAssignItem {
            dst: &dst[0],
            prepared: &good,
        },
        CKKSPreparedMulAssignItem {
            dst: &dst[1],
            prepared: &bad,
        },
    ];
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        module.ckks_mul_prepared_assign_batch_tmp_bytes(&query, &tsk)
    }))
    .expect_err("prepared precision overflow must make the scratch query panic");
    let panic_message = panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&str>().copied())
        .unwrap_or("non-string panic");
    assert!(
        panic_message.contains("prepared precision") && panic_message.contains("exceeds u32"),
        "unexpected query panic: {panic_message}"
    );

    let untouched = [clone_ct::<BE>(&dst[0]), clone_ct::<BE>(&dst[1])];
    let err = {
        let (head, tail) = dst.split_at_mut(1);
        let mut items = [
            CKKSPreparedMulAssignItem {
                dst: &mut head[0],
                prepared: &good,
            },
            CKKSPreparedMulAssignItem {
                dst: &mut tail[0],
                prepared: &bad,
            },
        ];
        module
            .ckks_mul_prepared_assign_batch(&mut items, &tsk, &mut scratch.borrow())
            .unwrap_err()
    };
    let message = err.to_string();
    assert!(
        message.contains(&format!("prepared precision {overflow_k} exceeds u32")),
        "unexpected overflow error: {message}"
    );
    for i in 0..dst.len() {
        assert_ct_identical::<BE>(
            &format!("prepared precision overflow preserves item={i}"),
            &untouched[i],
            &dst[i],
        );
    }
}

/// The execution seam recomputes the whole frontier's exact requirement and
/// rejects a short shared arena before the first destination write.
pub fn test_mul_batch_rejects_short_scratch_without_writes<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<ZnxWord = i64>,
    BE::OwnedBuf: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let (re, im) = super::helpers::test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let full = ckks_encrypt(
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
    let narrow = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k - params.base2k,
        &re,
        &im,
        &mut scratch.borrow(),
    );
    let sources = [&full, &narrow];
    let mut dst = [alloc_ct(&params, module, params.k), alloc_ct(&params, module, params.k)];
    let query = [
        CKKSSquareIntoItem {
            dst: &dst[0],
            a: sources[0],
        },
        CKKSSquareIntoItem {
            dst: &dst[1],
            a: sources[1],
        },
    ];
    let required = module.ckks_square_into_batch_tmp_bytes(&query, &tsk);
    assert!(required > 0, "active square batch must require scratch");
    let mut short = ScratchOwned::<BE>::alloc(0);
    let untouched = [clone_ct::<BE>(&dst[0]), clone_ct::<BE>(&dst[1])];

    let err = {
        let (head, tail) = dst.split_at_mut(1);
        let mut items = [
            CKKSSquareIntoItem {
                dst: &mut head[0],
                a: sources[0],
            },
            CKKSSquareIntoItem {
                dst: &mut tail[0],
                a: sources[1],
            },
        ];
        module
            .ckks_square_into_batch(&mut items, &tsk, &mut short.borrow())
            .unwrap_err()
    };
    let message = err.to_string();
    assert!(
        message.contains("scratch.available()") && message.contains("ckks_square_into_batch"),
        "unexpected short-scratch error: {message}"
    );
    for i in 0..dst.len() {
        assert_ct_identical::<BE>(&format!("short batch scratch preserves item={i}"), &untouched[i], &dst[i]);
    }
}

/// Byte-for-byte copy of a ciphertext, used to give the scalar and batch runs
/// identical starting states.
fn clone_ct<BE>(source: &CKKSCiphertextOwned<BE>) -> CKKSCiphertextOwned<BE>
where
    BE: poulpy_hal::layouts::Backend<ZnxWord = i64>,
    BE::OwnedBuf: poulpy_hal::layouts::HostDataRef + Clone,
{
    source.clone()
}

/// `ckks_mul_prepared_assign` against a freshly prepared operand is the same
/// operation as `ckks_mul_assign` against that operand, limb for limb: same
/// parameters, same tensor width, same stamp.
///
/// The lockstep `*TimesInput` tail relies on this to issue its two multiplies as
/// one prepared-right frontier.
pub fn test_mul_prepared_assign_matches_mul_assign<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<ZnxWord = i64>,
    BE::OwnedBuf: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let (re, im) = super::helpers::test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    for step in WIDTH_STEPS {
        let dst = ckks_encrypt(
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
        let other = ckks_encrypt(
            &params,
            module,
            host_module,
            &encoder,
            &sk,
            params.k - step * params.base2k,
            &im,
            &re,
            &mut scratch.borrow(),
        );

        let mut want = clone_ct::<BE>(&dst);
        module
            .ckks_mul_assign(&mut want, &other, &tsk, &mut scratch.borrow())
            .unwrap();

        let prepared = module.ckks_prepare_right(&other, &mut scratch.borrow()).unwrap();
        let mut have = clone_ct::<BE>(&dst);
        module
            .ckks_mul_prepared_assign(&mut have, &prepared, &tsk, &mut scratch.borrow())
            .unwrap();

        assert_ct_identical::<BE>(&format!("mul_prepared_assign vs mul_assign (step={step})"), &want, &have);
    }
}
