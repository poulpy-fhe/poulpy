//! Dependency-frontier batch multiplication: exact parity with the ordered
//! scalar operations, at batch lengths 0 to 4.
//!
//! Cross-item destination aliasing needs no runtime test: the batch takes `&mut`
//! destinations, so it is a borrow-check error. That is pinned by the
//! `compile_fail` doctest on
//! [`CKKSMulOps::ckks_mul_into_batch`](crate::api::CKKSMulOps::ckks_mul_into_batch).

use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module, ScratchOwned},
};

use crate::{
    CKKSCompositionError, SetCKKSInfos,
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

/// Byte-for-byte copy of a ciphertext, used to give the scalar and batch runs
/// identical starting states.
fn clone_ct<BE>(source: &CKKSCiphertextOwned<BE>) -> CKKSCiphertextOwned<BE>
where
    BE: poulpy_hal::layouts::Backend<ZnxWord = i64>,
    BE::OwnedBuf: poulpy_hal::layouts::HostDataRef + Clone,
{
    source.clone()
}
