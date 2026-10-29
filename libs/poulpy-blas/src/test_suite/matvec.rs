//! Products of a vector by a matrix.

use poulpy_ckks::{
    CKKSInfos, CKKSLayout, CKKSModuleInfos, CKKSResult, CoeffsMeta, GLWEPlaintextMeta, SetCKKSInfos, SetGLWEPlaintextInfos,
    SlotsKind,
    api::{CKKSEncodingOps, CKKSLinearTransformationOps},
    layouts::CKKSModuleAlloc,
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextModule, TestContextSharedModule, TestScalar, alloc_scratch,
            assert_decrypt_precision_at_log_delta, ckks_encrypt, gen_sk_with_raw,
        },
        parity::helpers::{Snapshot, snapshot, with_scratch},
        reference_encoder::ReferenceEncoder,
    },
};
use poulpy_core::layouts::{
    Base2K, DiagonalArithmetic, GLWELayout, GLWEToBackendRef, LWEInfos, TorusPrecision, prepared::GLWESecretPrepared,
};
use poulpy_hal::{
    api::{CnvPVecAlloc, NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedBorrow},
    layouts::{Backend, CyclotomicOrder, HostBytesBackend, HostDataMut, HostDataRef, Module, ScratchArena, Standard},
};

use super::helpers::{
    Rejection::{self, Budget, InvalidOperand, MissingKey},
    TestKeys, assert_rejected, assert_short_scratch_panics, poisoned_scratch, rotation_keys, to_f,
};
use crate::{
    api::{BlasEncodingOps, BlasMatVecOperandPreparedFactory, BlasMatVecOps},
    layouts::{
        BlasMatVecOperand, BlasMatVecOperandLayout, BlasMatVecOperandUnprepared, BlasModuleAlloc, BlasVector, BlasVectorInfos,
        BlasVectorLayout, BlasVectorOwned, BlasVectorShape, blas_matvec_pt_key_uses, blas_matvec_pt_output_layout,
        matvec_pt_galois_elements,
    },
    oep::{BlasEncodingImpl, BlasMatVecImpl},
};

/// Entry `(row, col)` of the public matrices of these tests, dyadic in `[-1, 1]`.
pub fn matvec_entry(row: usize, col: usize) -> f64 {
    ((row * 7 + col * 3) % 17) as f64 / 8.0 - 1.0
}

/// Entry `e` of the input vectors of these tests, dyadic in `[-1, 1]`.
pub fn vector_entry(e: usize) -> f64 {
    ((e * 5 + 3) % 13) as f64 / 8.0 - 0.75
}

/// The slots of ciphertext `t` of a vector of `shape` holding `values`, repeated every `period` of `slots` slots, zero
/// elsewhere.
pub fn vector_slots(shape: BlasVectorShape, period: usize, slots: usize, t: usize, values: &[f64]) -> Vec<f64> {
    let mut x = vec![0.0; slots];
    for e in 0..shape.chunk {
        if let Some(&v) = values.get(t * shape.chunk + e) {
            for q in 0..slots / period {
                x[shape.offset + shape.stride * e + q * period] = v;
            }
        }
    }
    x
}

/// `W v` for the matrix of [`matvec_entry`] of `rows` rows.
pub fn matvec_model(rows: usize, v: &[f64]) -> Vec<f64> {
    (0..rows)
        .map(|row| v.iter().enumerate().map(|(col, x)| matvec_entry(row, col) * x).sum())
        .collect()
}

/// Encrypts `values` as a vector of `shape` at torus width `k`, each ciphertext labelled with the sparsity of `period`.
#[allow(clippy::too_many_arguments)]
pub fn encrypt_vector<BE, F, E>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend<BE::Ring>>,
    encoder: &ReferenceEncoder<E>,
    sk: &GLWESecretPrepared<BE::OwnedBuf, BE>,
    k: usize,
    shape: BlasVectorShape,
    period: usize,
    values: &[f64],
    scratch: &mut ScratchArena<'_, BE>,
) -> BlasVectorOwned<BE>
where
    BE: TestContextBackend,
    Module<BE>: TestContextSharedModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F>,
{
    let slots = module.ckks_max_slots();
    let label = (slots / period).ilog2() as usize;
    let zero = vec![0.0; slots];
    let cts = (0..shape.count())
        .map(|t| {
            let re = vector_slots(shape, period, slots, t, values);
            let mut ct = ckks_encrypt(
                params,
                module,
                host_module,
                encoder,
                sk,
                k,
                &to_f::<F>(&re),
                &to_f::<F>(&zero),
                scratch,
            );
            ct.set_log_sparsity(label);
            ct
        })
        .collect();
    BlasVector::from_cts(cts, shape).unwrap()
}

/// A vector of `shape` at torus width `k`, at the radix and rank of `params`, labelled dense.
pub fn alloc_vector<BE: Backend>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    shape: BlasVectorShape,
    k: usize,
) -> BlasVectorOwned<BE>
where
    Module<BE>: CKKSModuleAlloc<BE>,
{
    let ct = CKKSLayout {
        glwe_layout: GLWELayout {
            k: k.into(),
            ..params.glwe_layout()
        },
        meta: GLWEPlaintextMeta::default(),
    };
    module.blas_vector_alloc_from_infos(&BlasVectorLayout::new(ct, shape))
}

/// Decrypts every ciphertext of `v` and checks it against `want`, the entries of a vector repeated every `period` and
/// zero on the other slots, after its metadata against `layout`.
#[allow(clippy::too_many_arguments)]
pub fn assert_vector_decrypt<BE, F, E>(
    label: &str,
    params: &CKKSTestParams,
    module: &Module<BE>,
    encoder: &ReferenceEncoder<E>,
    v: &BlasVectorOwned<BE>,
    sk: &GLWESecretPrepared<BE::OwnedBuf, BE>,
    want: &[f64],
    period: usize,
    layout: &BlasVectorLayout,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextSharedModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    assert_eq!(v.blas_vector_layout(), *layout, "{label}: layout");
    let slots = module.ckks_max_slots();
    let shape = v.vector_shape();
    let log_prec = layout.ct.log_delta().min(params.prec().log_delta());
    let zero = vec![0.0; slots];
    for (t, ct) in v.cts().iter().enumerate() {
        assert!(ct.noise().is_none(), "{label} ciphertext {t}: kept a noise tag");
        assert_decrypt_precision_at_log_delta::<BE, F, E>(
            &format!("{label} ciphertext {t}"),
            params,
            module,
            encoder,
            ct,
            sk,
            &to_f::<F>(&vector_slots(shape, period, slots, t, want)),
            &to_f::<F>(&zero),
            log_prec,
            scratch,
        );
    }
}

/// Every ciphertext of `v`, digits and metadata.
pub fn vector_snapshot<BE>(v: &BlasVectorOwned<BE>) -> Vec<Snapshot>
where
    BE: Backend<ZnxWord = i64>,
    poulpy_ckks::layouts::CKKSCiphertextOwned<BE>: GLWEToBackendRef<BE>,
{
    v.cts().iter().map(snapshot::<BE, _>).collect()
}

/// An operand of `layout` holding [`matvec_entry`], encoded then prepared, each call on exactly its query's bytes.
pub fn matvec_operands<BE, F>(
    module: &Module<BE>,
    layout: &BlasMatVecOperandLayout,
) -> (BlasMatVecOperandUnprepared<BE>, BlasMatVecOperand<BE>)
where
    BE: Backend + BlasEncodingImpl,
    Module<BE>: CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F> + BlasEncodingOps<BE, F> + CnvPVecAlloc<BE>,
    F: DiagonalArithmetic + TestScalar,
{
    let mut streamed = module.blas_matvec_operand_alloc_from_infos(layout);
    with_scratch::<BE, _>(module.blas_matvec_operand_encode_tmp_bytes(&streamed), |s| {
        module
            .blas_matvec_operand_encode(&mut streamed, |row, col| F::from_f64(matvec_entry(row, col)).unwrap(), s)
            .unwrap()
    });
    let mut resident = module.blas_matvec_operand_prepared_alloc_from_infos(layout);
    with_scratch::<BE, _>(module.blas_matvec_operand_prepare_tmp_bytes(layout), |s| {
        module.blas_matvec_operand_prepare(&mut resident, &streamed, s)
    });
    (streamed, resident)
}

/// The operand layouts the tests run, on `slots` slots, at least 16, at `params`' radix and precision: one ciphertext
/// each side, dense; a replicated input into a dense output; a dense input into a replicated output; several
/// ciphertexts on each side, the last ones partial, both replicated.
pub fn matvec_layouts(params: &CKKSTestParams, slots: usize) -> Vec<BlasMatVecOperandLayout> {
    let (base2k, meta): (Base2K, CoeffsMeta) = (params.base2k.into(), params.prec().into());
    let at = |input, p_in, output, p_out| BlasMatVecOperandLayout::new(slots, input, p_in, output, p_out, base2k, meta);
    vec![
        at(
            BlasVectorShape::new(5, 5, 1, 3),
            slots,
            BlasVectorShape::new(3, 3, 0, 5),
            slots,
        ),
        at(BlasVectorShape::new(4, 4, 0, 2), 8, BlasVectorShape::new(5, 5, 1, 3), slots),
        at(BlasVectorShape::new(5, 5, 2, 3), slots, BlasVectorShape::new(4, 4, 0, 1), 4),
        at(BlasVectorShape::new(7, 3, 1, 2), 8, BlasVectorShape::new(5, 2, 0, 3), 8),
    ]
}

/// `dec(matvec_pt(W, enc(v)))` is `W v` at every repetition of the output's period, and zero elsewhere, for every
/// layout of [`matvec_layouts`], with the operand resident and streamed, which give the same bits, on the operand's key
/// set alone ([`matvec_pt_galois_elements`], the encoded maps' Galois elements), at the metadata
/// [`blas_matvec_pt_output_layout`] gives; `res` is one limb wider than the result. On a store answering only at the
/// widths [`blas_matvec_pt_key_uses`] lists, it runs into `res` wider and narrower than the result. One byte short of
/// its query, the call panics before it writes `res`.
pub fn test_blas_matvec_pt<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard> + BlasMatVecImpl + BlasEncodingImpl,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + BlasEncodingOps<BE, F>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let slots = params.n / 2;
    if slots < 16 {
        eprintln!("matvec_pt skipped: its vectors span 16 slots, the ring holds {slots}");
        return;
    }
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let p_w = params.prec().log_delta();
    for layout in matvec_layouts(&params, slots) {
        let label = format!("matvec_pt {layout:?}");
        let (streamed, resident) = matvec_operands::<BE, F>(module, &layout);
        let keys = TestKeys::<BE> {
            atks: rotation_keys(
                &params,
                module,
                &sk_raw,
                matvec_pt_galois_elements(module.cyclotomic_order(), &layout),
                &mut scratch.borrow(),
            ),
            tsk: None,
        };
        let v: Vec<f64> = (0..layout.input.len).map(vector_entry).collect();
        let a = encrypt_vector::<BE, F, E>(
            &params,
            module,
            host_module,
            &encoder,
            &sk,
            params.k,
            layout.input,
            layout.input_period,
            &v,
            &mut scratch.borrow(),
        );
        let res_k = params.k + params.base2k;
        let template = alloc_vector(&params, module, layout.output, res_k);
        let want_layout = blas_matvec_pt_output_layout(&template, &layout, &a);
        let mut want_ct = a.ct_layout();
        want_ct.glwe_layout.k = TorusPrecision((params.k - p_w) as u32);
        want_ct.meta.slots = want_ct.meta.slots.join(SlotsKind::Real);
        want_ct.meta.log_sparsity = (slots / layout.output_period).ilog2() as usize;
        assert_eq!(
            want_layout,
            BlasVectorLayout::new(want_ct, layout.output),
            "{label}: output layout"
        );
        let mut encoded: Vec<i64> = resident
            .block(0, 0)
            .galois_elements(module.cyclotomic_order())
            .into_iter()
            .filter(|&p| p != 1)
            .collect();
        encoded.sort_unstable();
        encoded.dedup();
        let mut set = matvec_pt_galois_elements(module.cyclotomic_order(), &layout);
        set.retain(|&p| p != 1);
        assert_eq!(set, encoded, "{label}: key set against the encoded maps");
        let run = |w: &dyn Fn(&mut BlasVectorOwned<BE>, &mut ScratchArena<'_, BE>) -> CKKSResult<()>, bytes: usize| {
            let mut res = alloc_vector(&params, module, layout.output, res_k);
            let mut op_scratch = poisoned_scratch::<BE>(bytes);
            w(&mut res, &mut op_scratch.borrow()).unwrap();
            res
        };
        let resident_bytes = module.blas_matvec_pt_tmp_bytes(&template, &resident, &a, &params.atk_layout());
        let streamed_bytes = module.blas_matvec_pt_tmp_bytes(&template, &streamed, &a, &params.atk_layout());
        let have = run(&|res, s| module.blas_matvec_pt(res, &resident, &a, &keys, s), resident_bytes);
        assert_vector_decrypt::<BE, F, E>(
            &label,
            &params,
            module,
            &encoder,
            &have,
            &sk,
            &matvec_model(layout.output.len, &v),
            layout.output_period,
            &want_layout,
            &mut scratch.borrow(),
        );
        // Keys at the widths the key-use query lists and no other, into `res` wider and narrower than the result.
        for k in [res_k, params.k - p_w - params.base2k] {
            let res = alloc_vector(&params, module, layout.output, k);
            let uses = blas_matvec_pt_key_uses(module, &res, &layout, &a);
            let strict = keys.at_widths(uses.rotations.iter().map(|&(p, k)| (p, k.as_usize())), []);
            let bytes = module.blas_matvec_pt_tmp_bytes(&res, &resident, &a, &params.atk_layout());
            let mut res = res;
            with_scratch::<BE, _>(bytes, |s| module.blas_matvec_pt(&mut res, &resident, &a, &strict, s))
                .unwrap_or_else(|e| panic!("{label}: res.k={k} on the listed keys: {e}"));
        }
        let have_streamed = run(&|res, s| module.blas_matvec_pt(res, &streamed, &a, &keys, s), streamed_bytes);
        assert_eq!(
            vector_snapshot::<BE>(&have_streamed),
            vector_snapshot::<BE>(&have),
            "{label}: streamed against resident"
        );
        for (form, bytes) in [("resident", resident_bytes), ("streamed", streamed_bytes)] {
            let mut res = alloc_vector(&params, module, layout.output, res_k);
            let before = vector_snapshot::<BE>(&res);
            assert_short_scratch_panics::<BE, _>("blas_matvec_pt", bytes, |s| {
                if form == "resident" {
                    module.blas_matvec_pt(&mut res, &resident, &a, &keys, s)
                } else {
                    module.blas_matvec_pt(&mut res, &streamed, &a, &keys, s)
                }
            });
            assert!(
                vector_snapshot::<BE>(&res) == before,
                "{label}: {form} wrote res before panicking"
            );
        }
    }
}

/// Runs `call` on `res` and asserts it fails with `rejection` raised by `op` and leaves `res` unchanged.
fn assert_vector_guard<BE>(
    label: &str,
    op: &str,
    rejection: Rejection,
    res: &mut BlasVectorOwned<BE>,
    call: impl FnOnce(&mut BlasVectorOwned<BE>) -> CKKSResult<()>,
) where
    BE: Backend<ZnxWord = i64>,
    poulpy_ckks::layouts::CKKSCiphertextOwned<BE>: GLWEToBackendRef<BE>,
{
    let before = vector_snapshot::<BE>(res);
    assert_rejected(label, op, rejection, call(res));
    assert!(vector_snapshot::<BE>(res) == before, "{label}: {op} changed its output");
}

/// `blas_matvec_pt` rejects, before any write: a vector whose ciphertexts do not tile its shape, share no layout, or
/// whose entries do not fit the period its label proves, as input and as `res`; a vector of another shape than the
/// operand's input or output; an input of another radix or rank than `res`, or narrower than the operand; an input
/// repeating with a longer period than the operand reads; a store lacking a key of the operand's set; and an input whose
/// budget is below the operand's scale, which poulpy-ckks's summed transformation rejects.
pub fn test_blas_matvec_pt_guards<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard> + BlasMatVecImpl + BlasEncodingImpl,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + BlasEncodingOps<BE, F>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let op = "blas_matvec_pt";
    let slots = params.n / 2;
    if slots < 16 {
        eprintln!("matvec_pt_guards skipped: its vectors span 16 slots, the ring holds {slots}");
        return;
    }
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    // Several ciphertexts on each side, the input replicated.
    let layout = matvec_layouts(&params, slots).pop().unwrap();
    let (_, w) = matvec_operands::<BE, F>(module, &layout);
    let elements = matvec_pt_galois_elements(module.cyclotomic_order(), &layout);
    let keys = TestKeys::<BE> {
        atks: rotation_keys(&params, module, &sk_raw, elements.iter().copied(), &mut scratch.borrow()),
        tsk: None,
    };
    let v: Vec<f64> = (0..layout.input.len).map(vector_entry).collect();
    let encrypt = |shape: BlasVectorShape, period: usize, k: usize, scratch: &mut ScratchArena<'_, BE>| {
        encrypt_vector::<BE, F, E>(&params, module, host_module, &encoder, &sk, k, shape, period, &v, scratch)
    };
    let a = encrypt(layout.input, layout.input_period, params.k, &mut scratch.borrow());
    let mut res = alloc_vector(&params, module, layout.output, params.k);
    let bytes = module.blas_matvec_pt_tmp_bytes(&res, &w, &a, &params.atk_layout());
    let call =
        |label: &str, rejection: Rejection, res: &mut BlasVectorOwned<BE>, a: &BlasVectorOwned<BE>, keys: &TestKeys<BE>| {
            assert_vector_guard::<BE>(label, op, rejection, res, |res| {
                with_scratch::<BE, _>(bytes, |s| module.blas_matvec_pt(res, &w, a, keys, s))
            });
        };

    // Vectors the operation rejects, as input, then as `res`.
    let mut rejected: Vec<(&str, BlasVectorOwned<BE>)> = Vec::new();
    let mut mixed = a.clone();
    mixed.ct_mut(1).set_log_delta(a.ct_layout().log_delta() - 1);
    rejected.push(("ciphertexts of two layouts", mixed));
    let mut sparse = a.clone();
    sparse
        .cts_mut()
        .iter_mut()
        .for_each(|ct| ct.set_log_sparsity((slots / 4).ilog2() as usize));
    rejected.push(("entries past the period the label proves", sparse));
    for (what, bad) in &rejected {
        call(&format!("input: {what}"), InvalidOperand, &mut res, bad, &keys);
        let mut bad_res = bad.clone();
        assert_vector_guard::<BE>(&format!("res: {what}"), op, InvalidOperand, &mut bad_res, |res| {
            with_scratch::<BE, _>(bytes, |s| module.blas_matvec_pt(res, &w, &a, &keys, s))
        });
    }

    // Shapes and layouts against the operand's.
    let other_shape = encrypt(
        BlasVectorShape::new(7, 3, 0, 2),
        layout.input_period,
        params.k,
        &mut scratch.borrow(),
    );
    call("input of another shape", InvalidOperand, &mut res, &other_shape, &keys);
    let mut other_res = alloc_vector(&params, module, BlasVectorShape::new(5, 2, 1, 3), params.k);
    assert_vector_guard::<BE>("res of another shape", op, InvalidOperand, &mut other_res, |res| {
        with_scratch::<BE, _>(bytes, |s| module.blas_matvec_pt(res, &w, &a, &keys, s))
    });
    let other_radix = CKKSTestParams {
        base2k: params.base2k - 1,
        ..params
    };
    let mut res_radix = alloc_vector(&other_radix, module, layout.output, params.k);
    assert_vector_guard::<BE>("input of another radix than res", op, InvalidOperand, &mut res_radix, |res| {
        with_scratch::<BE, _>(bytes, |s| module.blas_matvec_pt(res, &w, &a, &keys, s))
    });
    let mut wide_meta = layout.coeffs_meta;
    wide_meta.k = TorusPrecision(wide_meta.k.0 + 1);
    let (_, wide) = matvec_operands::<BE, F>(
        module,
        &BlasMatVecOperandLayout {
            coeffs_meta: wide_meta,
            ..layout
        },
    );
    let narrow = encrypt(
        layout.input,
        layout.input_period,
        layout.coeffs_meta.k.as_usize(),
        &mut scratch.borrow(),
    );
    assert_vector_guard::<BE>("input narrower than the operand", op, InvalidOperand, &mut res, |res| {
        with_scratch::<BE, _>(bytes, |s| module.blas_matvec_pt(res, &wide, &narrow, &keys, s))
    });
    let dense = encrypt(layout.input, slots, params.k, &mut scratch.borrow());
    call(
        "input repeating past the operand's period",
        InvalidOperand,
        &mut res,
        &dense,
        &keys,
    );

    // Keys: the operand's set without its first element.
    let missing = TestKeys::<BE> {
        atks: rotation_keys(
            &params,
            module,
            &sk_raw,
            elements.iter().copied().skip(1),
            &mut scratch.borrow(),
        ),
        tsk: None,
    };
    if !elements.is_empty() {
        call("a missing rotation key", MissingKey, &mut res, &a, &missing);
    }

    // Budget: an input at the operand's width, whose budget is below its scale.
    let low = encrypt(
        layout.input,
        layout.input_period,
        layout.coeffs_meta.k.as_usize(),
        &mut scratch.borrow(),
    );
    assert_vector_guard::<BE>(
        "an input budget below the operand's scale",
        "ckks_eval_linear_transformation_sum_into",
        Budget,
        &mut res,
        |res| with_scratch::<BE, _>(bytes, |s| module.blas_matvec_pt(res, &w, &low, &keys, s)),
    );
}
