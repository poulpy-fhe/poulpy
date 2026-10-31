//! Products of a vector by a matrix.

use poulpy_ckks::{
    CKKSInfos, CKKSLayout, CKKSModuleInfos, CKKSResult, CoeffsMeta, GLWEPlaintextInfos, GLWEPlaintextMeta, Scale, SetCKKSInfos,
    SetGLWEPlaintextInfos, SlotsKind,
    api::{CKKSEncodingOps, CKKSLinearTransformationOps, CKKSRightOperandPreparedFactory, LinearTransformationLayout},
    layouts::CKKSRightOperandPrepared,
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextModule, TestContextSharedModule, TestScalar, alloc_scratch,
            assert_decrypt_precision_at_log_delta, ckks_encrypt, encode_and_upload_pt, gen_sk_with_raw, gen_tsk,
        },
        parity::helpers::{Snapshot, snapshot, with_scratch},
        reference_encoder::ReferenceEncoder,
    },
};
use poulpy_core::layouts::{
    Base2K, DiagonalArithmetic, GLWEInfos, GLWELayout, GLWEToBackendRef, LWEInfos, LinearTransformationStrategy, TorusPrecision,
    prepared::GLWESecretPrepared,
};
use poulpy_hal::{
    api::{CnvPVecAlloc, ReimFFT, ReimIFFT, ReimNew, ScratchOwnedBorrow},
    layouts::{
        Backend, CyclotomicOrder, HostBytesBackend, HostDataMut, HostDataRef, Module, PrepareHint, ScratchArena, Standard,
    },
};

use super::helpers::{
    Rejection::{self, Budget, InvalidOperand, MissingKey, MissingTensorKey},
    TestKeys, assert_panics, assert_rejected, assert_short_scratch_panics, poisoned_scratch, rotation_keys, to_f,
};
use crate::{
    api::{BlasDiagonalCiphertextPreparedFactory, BlasEncodingOps, BlasMatVecOperandPreparedFactory, BlasMatVecOps},
    layouts::{
        BlasDiagonalCiphertext, BlasDiagonalCiphertextAlloc, BlasDiagonalCiphertextLayout, BlasDiagonalCiphertextPrepared,
        BlasMatVecOperand, BlasMatVecOperandAlloc, BlasMatVecOperandLayout, BlasMatVecOperandPrepared, BlasVector,
        BlasVectorAlloc, BlasVectorInfos, BlasVectorLayout, BlasVectorMut, BlasVectorOwned, BlasVectorRef, BlasVectorShape,
        append_col_diagonal_galois_elements, append_row_diagonal_galois_elements, blas_append_col_diagonal_key_uses,
        blas_append_row_diagonal_key_uses, blas_matvec_diagonal_key_uses, blas_matvec_diagonal_output_layout,
        blas_matvec_pt_output_layout, diagonal_ciphertext::AppendAxis, matvec_diagonal_galois_elements,
        matvec_pt_galois_elements,
    },
    oep::{BlasEncodingImpl, BlasMatVecImpl},
};

/// A call writing a vector on a scratch.
type VectorCall<'c, BE> = dyn Fn(&mut BlasVectorOwned<BE>, &mut ScratchArena<'_, BE>) -> CKKSResult<()> + 'c;

/// A change of a ciphertext layout.
type LayoutEdit = fn(&mut GLWELayout);

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
    E: ReimFFT<F> + ReimIFFT<F>,
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
) -> BlasVectorOwned<BE> {
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
    E: ReimFFT<F> + ReimIFFT<F>,
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
) -> (BlasMatVecOperand<BE>, BlasMatVecOperandPrepared<BE>)
where
    BE: Backend + BlasEncodingImpl,
    Module<BE>: CKKSEncodingOps<BE, F> + BlasEncodingOps<BE, F> + CnvPVecAlloc<BE>,
    F: DiagonalArithmetic + TestScalar,
{
    matvec_operands_of::<BE, F>(module, layout, matvec_entry)
}

/// [`matvec_operands`] holding `entry`.
pub(crate) fn matvec_operands_of<BE, F>(
    module: &Module<BE>,
    layout: &BlasMatVecOperandLayout,
    entry: fn(usize, usize) -> f64,
) -> (BlasMatVecOperand<BE>, BlasMatVecOperandPrepared<BE>)
where
    BE: Backend + BlasEncodingImpl,
    Module<BE>: CKKSEncodingOps<BE, F> + BlasEncodingOps<BE, F> + CnvPVecAlloc<BE>,
    F: DiagonalArithmetic + TestScalar,
{
    let mut streamed = module.blas_matvec_operand_alloc_from_infos(layout);
    with_scratch::<BE, _>(module.blas_matvec_operand_encode_tmp_bytes(&streamed), |s| {
        module
            .blas_matvec_operand_encode(&mut streamed, |row, col| F::from_f64(entry(row, col)).unwrap(), s)
            .unwrap()
    });
    let mut resident = module.blas_matvec_operand_prepared_alloc_from_infos(layout);
    with_scratch::<BE, _>(module.blas_matvec_operand_prepare_tmp_bytes(layout), |s| {
        module.blas_matvec_operand_prepare(&mut resident, &streamed, s)
    });
    (streamed, resident)
}

/// The operand layouts the tests run, on `slots` slots, at least 16, at `params`' radix and precision: one ciphertext
/// each side, dense, on the `Direct` schedule; a replicated input into a dense output on the BSGS schedule of giant step
/// 2; a dense input into a replicated output, and several ciphertexts on
/// each side, the last ones partial, both replicated, at the cost-optimal BSGS giant step.
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
        at(BlasVectorShape::new(4, 4, 0, 2), 8, BlasVectorShape::new(5, 5, 1, 3), slots)
            .with_strategy(LinearTransformationStrategy::Bsgs { giant_step: 2 }),
        at(BlasVectorShape::new(5, 5, 2, 3), slots, BlasVectorShape::new(4, 4, 0, 1), 4).with_optimal_bsgs(),
        at(BlasVectorShape::new(7, 3, 1, 2), 8, BlasVectorShape::new(5, 2, 0, 3), 8).with_optimal_bsgs(),
    ]
}

/// `dec(matvec_pt(W, enc(v)))` is `W v` at every repetition of the output's period, and zero elsewhere, for every
/// layout of [`matvec_layouts`], with the operand resident and streamed, which give the same bits, on the operand's key
/// set alone ([`matvec_pt_galois_elements`]), at the metadata
/// [`blas_matvec_pt_output_layout`] gives, into `res` one limb wider than the result and at its width. One byte short of
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
    E: ReimNew<F>,
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
        let run = |w: &VectorCall<'_, BE>, bytes: usize, k: usize| {
            let mut res = alloc_vector(&params, module, layout.output, k);
            let mut op_scratch = poisoned_scratch::<BE>(bytes);
            w(&mut res, &mut op_scratch.borrow()).unwrap();
            res
        };
        let resident_bytes = module.blas_matvec_pt_tmp_bytes(&template, &resident, &a, &params.atk_layout());
        let streamed_bytes = module.blas_matvec_pt_tmp_bytes(&template, &streamed, &a, &params.atk_layout());
        let have = run(
            &|res, s| module.blas_matvec_pt(res, &resident, &a, &keys, s),
            resident_bytes,
            res_k,
        );
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
        let template_at = alloc_vector(&params, module, layout.output, params.k - p_w);
        let bytes = module.blas_matvec_pt_tmp_bytes(&template_at, &resident, &a, &params.atk_layout());
        let at = run(
            &|res, s| module.blas_matvec_pt(res, &resident, &a, &keys, s),
            bytes,
            params.k - p_w,
        );
        assert_vector_decrypt::<BE, F, E>(
            &format!("{label} at the result's width"),
            &params,
            module,
            &encoder,
            &at,
            &sk,
            &matvec_model(layout.output.len, &v),
            layout.output_period,
            &blas_matvec_pt_output_layout(&template_at, &layout, &a),
            &mut scratch.borrow(),
        );
        let have_streamed = run(
            &|res, s| module.blas_matvec_pt(res, &streamed, &a, &keys, s),
            streamed_bytes,
            res_k,
        );
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
pub(crate) fn assert_vector_guard<BE>(
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

/// `v` read as a vector of `shape` whatever it holds: ciphertexts that do not tile the shape it reports, which every
/// operation rejects before `res` changes.
struct Untiled<BE: Backend> {
    v: BlasVectorOwned<BE>,
    shape: BlasVectorShape,
}

impl<BE: Backend> BlasVectorInfos for Untiled<BE> {
    fn blas_vector_layout(&self) -> BlasVectorLayout {
        BlasVectorLayout::new(self.v.ct_layout(), self.shape)
    }
}

impl<BE: Backend> BlasVectorRef<BE> for Untiled<BE>
where
    BlasVectorOwned<BE>: BlasVectorRef<BE>,
{
    type Ct = <BlasVectorOwned<BE> as BlasVectorRef<BE>>::Ct;

    fn cts(&self) -> &[Self::Ct] {
        BlasVectorRef::<BE>::cts(&self.v)
    }
}

impl<BE: Backend> BlasVectorMut<BE> for Untiled<BE>
where
    BlasVectorOwned<BE>: BlasVectorMut<BE>,
{
    fn cts_mut(&mut self) -> &mut [Self::Ct] {
        BlasVectorMut::<BE>::cts_mut(&mut self.v)
    }
}

/// `blas_matvec_pt` rejects, before any write, as input and as `res`: a vector whose ciphertexts do not tile its shape,
/// share no layout, have no slot between entries, carry a scale past their width, or whose entries do not fit the period
/// its label proves, and one of twice the degree, another radix or another rank than the other side. It also rejects a
/// vector of another shape than the operand's input or output; an input and `res` at a radix the operand is not at, or
/// at a rank its keys are not at; an operand of other slots; an input narrower than the operand; an input repeating with
/// a longer period than the operand reads; a store lacking a key of the operand's set; and an input whose budget is below
/// the operand's scale, which poulpy-ckks's summed transformation rejects.
pub fn test_blas_matvec_pt_guards<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard> + BlasMatVecImpl + BlasEncodingImpl,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + BlasEncodingOps<BE, F>
        + CKKSLinearTransformationOps<BE>
        + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: ReimNew<F>,
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
    rejected.push((
        "no slot between entries",
        BlasVector::from_cts(
            a.clone().into_cts(),
            BlasVectorShape {
                stride: 0,
                ..layout.input
            },
        )
        .unwrap(),
    ));
    let scale = a.ct_layout().k().as_usize() + 1;
    let mut past_width = a.clone();
    past_width.cts_mut().iter_mut().for_each(|ct| {
        let meta = GLWEPlaintextMeta {
            scale: Scale::Log(scale),
            ..ct.meta()
        };
        ct.set_meta(meta)
    });
    rejected.push(("a scale past the ciphertexts' width", past_width));
    for (what, bad) in &rejected {
        call(&format!("input: {what}"), InvalidOperand, &mut res, bad, &keys);
        let mut bad_res = bad.clone();
        assert_vector_guard::<BE>(&format!("res: {what}"), op, InvalidOperand, &mut bad_res, |res| {
            with_scratch::<BE, _>(bytes, |s| module.blas_matvec_pt(res, &w, &a, &keys, s))
        });
    }

    // A vector whose ciphertexts do not tile the shape it reports, as input, then as `res`.
    let untiled = |v: &BlasVectorOwned<BE>| Untiled {
        shape: BlasVectorShape {
            len: v.vector_shape().len + v.vector_shape().chunk,
            ..v.vector_shape()
        },
        v: v.clone(),
    };
    let before = vector_snapshot::<BE>(&res);
    let untiled_a = untiled(&a);
    assert_rejected(
        "input: ciphertexts that do not tile the shape",
        op,
        InvalidOperand,
        with_scratch::<BE, _>(bytes, |s| module.blas_matvec_pt(&mut res, &w, &untiled_a, &keys, s)),
    );
    assert!(
        vector_snapshot::<BE>(&res) == before,
        "input untiled: {op} changed its output"
    );
    let mut untiled_res = untiled(&res);
    assert_rejected(
        "res: ciphertexts that do not tile the shape",
        op,
        InvalidOperand,
        with_scratch::<BE, _>(bytes, |s| module.blas_matvec_pt(&mut untiled_res, &w, &a, &keys, s)),
    );
    assert!(
        vector_snapshot::<BE>(&untiled_res.v) == before,
        "res untiled: {op} changed its output"
    );

    // Foreign layouts, as input beside a regular `res`, then as `res` beside a regular input: twice the degree, one bit
    // less radix, one more rank.
    let foreign = |v: &BlasVectorOwned<BE>, edit: fn(&mut GLWELayout)| {
        let mut ct = v.ct_layout();
        edit(&mut ct.glwe_layout);
        module.blas_vector_alloc_from_infos(&BlasVectorLayout::new(ct, v.vector_shape()))
    };
    let edits: [(&str, LayoutEdit); 3] = [
        ("degree", |l| l.n = (2 * l.n.as_usize()).into()),
        ("base2k", |l| l.base2k = (l.base2k.as_usize() - 1).into()),
        ("rank", |l| l.rank = (l.rank.as_usize() + 1).into()),
    ];
    for (what, edit) in edits {
        call(
            &format!("input of another {what}"),
            InvalidOperand,
            &mut res,
            &foreign(&a, edit),
            &keys,
        );
        assert_vector_guard::<BE>(
            &format!("res of another {what}"),
            op,
            InvalidOperand,
            &mut foreign(&res, edit),
            |res| with_scratch::<BE, _>(bytes, |s| module.blas_matvec_pt(res, &w, &a, &keys, s)),
        );
    }
    // Input and `res` agreeing on a radix the operand is not at, then on a rank its keys are not at.
    for (what, edit) in [edits[1], edits[2]] {
        let (a_other, mut res_other) = (foreign(&a, edit), foreign(&res, edit));
        assert_vector_guard::<BE>(
            &format!("input and res of another {what} than the operand"),
            op,
            InvalidOperand,
            &mut res_other,
            |res| with_scratch::<BE, _>(bytes, |s| module.blas_matvec_pt(res, &w, &a_other, &keys, s)),
        );
    }
    // An operand encoded on a module of other slots.
    let double = Module::<BE>::new(2 * params.n as u64);
    let (_, wide_slots) = matvec_operands::<BE, F>(
        &double,
        &BlasMatVecOperandLayout {
            slots: 2 * slots,
            ..layout
        },
    );
    assert_vector_guard::<BE>("an operand of other slots", op, InvalidOperand, &mut res, |res| {
        with_scratch::<BE, _>(bytes, |s| module.blas_matvec_pt(res, &wide_slots, &a, &keys, s))
    });

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
        "ckks_linear_transformation_eval_sum_into",
        Budget,
        &mut res,
        |res| with_scratch::<BE, _>(bytes, |s| module.blas_matvec_pt(res, &w, &low, &keys, s)),
    );
}

/// The diagonals the diagonal-ciphertext tests hold, `0..8`, at giant step 4: four baby steps and two giant steps.
pub fn diagonal_ciphertext_layout(params: &CKKSTestParams, slots: usize, ct_k: usize) -> BlasDiagonalCiphertextLayout {
    let mut ct = params.prec();
    ct.glwe_layout.k = ct_k.into();
    BlasDiagonalCiphertextLayout::new(
        ct,
        LinearTransformationLayout {
            indexes: (0..8).collect(),
            slots,
            strategy: LinearTransformationStrategy::Bsgs { giant_step: 4 },
        },
    )
}

/// `layout` holding the diagonals `indexes` under `strategy`.
pub(crate) fn diagonal_ciphertext_with(
    layout: &BlasDiagonalCiphertextLayout,
    indexes: Vec<i64>,
    strategy: LinearTransformationStrategy,
) -> BlasDiagonalCiphertextLayout {
    BlasDiagonalCiphertextLayout::new(
        layout.ct,
        LinearTransformationLayout {
            indexes,
            slots: layout.lt.slots,
            strategy,
        },
    )
}

/// The Galois elements of every operation on a matrix of `layout`: its product, its row and its column appends.
pub(crate) fn diagonal_ciphertext_elements(order: i64, layout: &BlasDiagonalCiphertextLayout) -> std::collections::BTreeSet<i64> {
    matvec_diagonal_galois_elements(order, layout)
        .into_iter()
        .chain(append_row_diagonal_galois_elements(order, layout))
        .chain(append_col_diagonal_galois_elements(order, layout))
        .collect()
}

/// [`BlasMatVecOps::blas_append_row_diagonal`] at `at` for [`AppendAxis::Row`], else [`BlasMatVecOps::blas_append_col_diagonal`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn append<BE, V, K>(
    module: &Module<BE>,
    axis: AppendAxis,
    m: &mut BlasDiagonalCiphertext<BE>,
    at: i64,
    v: &V,
    mask: &CKKSRightOperandPrepared<BE>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<()>
where
    BE: Backend + BlasMatVecImpl,
    V: crate::layouts::BlasVectorRef<BE>,
    K: poulpy_core::layouts::GetAutomorphismKey<BE> + Sync,
{
    match axis {
        AppendAxis::Row => module.blas_append_row_diagonal(m, at, v, mask, keys, scratch),
        AppendAxis::Col => module.blas_append_col_diagonal(m, at, v, mask, keys, scratch),
    }
}

/// The scratch query of [`append`] along `axis`.
pub(crate) fn append_tmp_bytes<BE: Backend + BlasMatVecImpl>(
    module: &Module<BE>,
    axis: AppendAxis,
    m: &impl crate::layouts::BlasDiagonalCiphertextInfos,
    v: &impl BlasVectorInfos,
    mask: &impl poulpy_core::layouts::GLWEInfos,
    keys: &impl poulpy_core::layouts::GGLWEInfos,
) -> usize {
    match axis {
        AppendAxis::Row => module.blas_append_row_diagonal_tmp_bytes(m, v, mask, keys),
        AppendAxis::Col => module.blas_append_col_diagonal_tmp_bytes(m, v, mask, keys),
    }
}

/// The key-use query of [`append`] along `axis`.
pub(crate) fn append_key_uses<BE: Backend>(
    module: &Module<BE>,
    axis: AppendAxis,
    m: &impl crate::layouts::BlasDiagonalCiphertextInfos,
    v: &impl BlasVectorInfos,
) -> crate::layouts::BlasKeyUses {
    match axis {
        AppendAxis::Row => blas_append_row_diagonal_key_uses(module, m, v),
        AppendAxis::Col => blas_append_col_diagonal_key_uses(module, m, v),
    }
}

/// The layout of the product of a matrix of diagonals of layout `m` by `a` into `res` of width `res_k`, read off the
/// `blas_matvec_diagonal` doc: poulpy-ckks's ct x ct rule between `a` and a diagonal.
pub(crate) fn matvec_diagonal_layout(a: CKKSLayout, m: CKKSLayout, res_k: usize) -> CKKSLayout {
    let (a_delta, m_delta) = (a.log_delta(), m.log_delta());
    let log_delta = a_delta.min(m_delta);
    let budget = (a.k().as_usize() - a_delta).min(m.k().as_usize() - m_delta) - a_delta.max(m_delta);
    let mut ct = a;
    ct.glwe_layout.k = (budget + log_delta).min(res_k).into();
    ct.meta.scale = poulpy_ckks::Scale::Log(log_delta);
    ct.meta.log_sparsity = a.log_sparsity().min(m.log_sparsity());
    ct.meta.slots = a.slots().join(m.slots());
    ct
}

/// The cleartext of a matrix holding the diagonals `t` of `ts`: row `at` of `v` adds `v[at + t]` at `(at, at + t)`,
/// column `at` adds `v[at - t]` at `(at - t, at)`, on every slot of `mask` shifted by `at`; `matvec` sums
/// `D_t[i] x[i + t]`.
pub(crate) struct DiagonalCiphertextModel {
    slots: usize,
    ts: Vec<usize>,
    m: Vec<Vec<f64>>,
}

impl DiagonalCiphertextModel {
    pub(crate) fn new(slots: usize, ts: Vec<usize>) -> Self {
        Self {
            slots,
            ts,
            m: vec![vec![0.0; slots]; slots],
        }
    }

    pub(crate) fn append(&mut self, axis: AppendAxis, at: i64, mask: &[f64], v: &[f64]) {
        let s = self.slots;
        for (q0, &w) in mask.iter().enumerate() {
            let q = (q0 as i64 + at).rem_euclid(s as i64) as usize;
            for &t in &self.ts {
                match axis {
                    AppendAxis::Col => {
                        let i = (q + s - t) % s;
                        self.m[i][q] += w * v[i];
                    }
                    AppendAxis::Row => {
                        let j = (q + t) % s;
                        self.m[q][j] += w * v[j];
                    }
                }
            }
        }
    }

    pub(crate) fn matvec(&self, x: &[f64]) -> Vec<f64> {
        let s = self.slots;
        (0..s)
            .map(|i| self.ts.iter().map(|&t| self.m[i][(i + t) % s] * x[(i + t) % s]).sum())
            .collect()
    }
}

/// Every diagonal ciphertext of `m`, digits and metadata.
pub(crate) fn diagonal_ciphertext_snapshot<BE>(m: &BlasDiagonalCiphertext<BE>) -> Vec<Snapshot>
where
    BE: Backend<ZnxWord = i64>,
    poulpy_ckks::layouts::CKKSCiphertextOwned<BE>: GLWEToBackendRef<BE>,
{
    m.diagonals()
        .giant_steps
        .iter()
        .flat_map(|gs| gs.diagonals.iter())
        .map(|d| snapshot::<BE, _>(&d.plaintext))
        .collect()
}

/// Prepares `m` into `res` on exactly its query's bytes.
pub(crate) fn prepare_diagonal_ciphertext<BE>(
    module: &Module<BE>,
    res: &mut BlasDiagonalCiphertextPrepared<BE>,
    m: &BlasDiagonalCiphertext<BE>,
) where
    BE: Backend,
    Module<BE>: BlasDiagonalCiphertextPreparedFactory<BE>,
{
    let bytes = module.blas_diagonal_ciphertext_prepare_tmp_bytes(m);
    module.blas_diagonal_ciphertext_prepare(res, m, &mut poisoned_scratch::<BE>(bytes).borrow());
}

/// `m` prepared into a fresh allocation, on exactly the query's bytes.
pub(crate) fn prepared_of<BE>(module: &Module<BE>, m: &BlasDiagonalCiphertext<BE>) -> BlasDiagonalCiphertextPrepared<BE>
where
    BE: Backend,
    Module<BE>: BlasDiagonalCiphertextPreparedFactory<BE>,
{
    let mut prepared = module.blas_diagonal_ciphertext_prepared_alloc_from_infos(m);
    prepare_diagonal_ciphertext(module, &mut prepared, m);
    prepared
}

/// The appends of the diagonal-ciphertext tests: `(axis, at, mask index, seed)`, by the masks of one slot then two.
const APPENDS: [(AppendAxis, i64, usize, usize); 5] = [
    (AppendAxis::Row, 0, 0, 1),
    (AppendAxis::Row, 3, 0, 2),
    (AppendAxis::Col, 5, 0, 3),
    (AppendAxis::Row, -2, 1, 4),
    (AppendAxis::Col, 1, 1, 5),
];

/// A diagonal ciphertext over the slots, its diagonals `0..8` at giant step 4, by rotation and by powers of two: zero once
/// allocated, so its product is zero; then rows and columns appended at several positions, by a mask of one slot
/// prepared to reuse and one of two slots prepared once, and its product by a vector, prepared again into one
/// allocation after each append (resident) and streamed, with the same bits, decrypts to the cleartext matrix's,
/// at the metadata of its doc, which [`blas_matvec_diagonal_output_layout`] gives, on the keys of [`matvec_diagonal_galois_elements`],
/// [`append_row_diagonal_galois_elements`] and [`append_col_diagonal_galois_elements`] alone, then on a store answering only at the widths
/// [`blas_matvec_diagonal_key_uses`], [`blas_append_row_diagonal_key_uses`] and [`blas_append_col_diagonal_key_uses`] list. A matrix of the
/// diagonal `0` alone, which rotates nothing, appends and multiplies, resident and streamed, on the tensor key alone.
/// One byte short of its query, each call and the preparation panic before they write.
pub fn test_blas_matvec_diagonal<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard> + BlasMatVecImpl + BlasEncodingImpl,
    Module<BE>: TestContextModule<BE> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: ReimNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let slots = params.n / 2;
    if slots < 16 {
        eprintln!("matvec_diagonal skipped: its matrix holds diagonals up to 8 on at least 16 slots, the ring holds {slots}");
        return;
    }
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let log_delta = params.prec().log_delta();
    let by_rotation = diagonal_ciphertext_layout(&params, slots, params.k - log_delta);
    let by_powers = diagonal_ciphertext_with(
        &by_rotation,
        (0..8).collect(),
        LinearTransformationStrategy::BsgsPowersOfTwo { giant_step: 4 },
    );
    let order = module.cyclotomic_order();
    let whole = BlasVectorShape::new(slots, slots, 0, 1);
    let encrypt = |values: &[f64], scratch: &mut ScratchArena<'_, BE>| {
        encrypt_vector::<BE, F, E>(
            &params,
            module,
            host_module,
            &encoder,
            &sk,
            params.k,
            whole,
            slots,
            values,
            scratch,
        )
    };
    let mask_of = |slots_set: &[usize], hint: PrepareHint| {
        let mut re = vec![0.0; slots];
        slots_set.iter().for_each(|&s| re[s] = 1.0);
        let pt = encode_and_upload_pt(
            host_module,
            module,
            &encoder,
            params.base2k.into(),
            params.prec(),
            &to_f::<F>(&re),
            &to_f::<F>(&vec![0.0; slots]),
        );
        let prepared = with_scratch::<BE, _>(module.ckks_right_operand_prepare_tmp_bytes(&pt), |s| {
            module.ckks_right_operand_prepare_alloc(&pt, hint, s)
        })
        .unwrap();
        (re, prepared)
    };
    let masks = [
        mask_of(&[0], PrepareHint::Reuse),
        mask_of(&[0, slots / 2], PrepareHint::OneShot),
    ];
    let x: Vec<f64> = (0..slots).map(vector_entry).collect();
    let a = encrypt(&x, &mut scratch.borrow());
    let res_k = params.k;
    let want_layout = BlasVectorLayout::new(matvec_diagonal_layout(a.ct_layout(), by_rotation.ct, res_k), whole);

    for layout in [&by_rotation, &by_powers] {
        let name = format!("{:?}", layout.lt.strategy);
        let keys = TestKeys::<BE> {
            atks: rotation_keys(
                &params,
                module,
                &sk_raw,
                diagonal_ciphertext_elements(order, layout),
                &mut scratch.borrow(),
            ),
            tsk: Some(gen_tsk(&params, module, &sk_raw, &mut scratch.borrow())),
        };
        let mut m = module.blas_diagonal_ciphertext_alloc_from_infos(layout);
        // One prepared allocation, refilled after each append.
        let mut prepared = module.blas_diagonal_ciphertext_prepared_alloc_from_infos(layout);
        let mut model = DiagonalCiphertextModel::new(slots, (0..8).collect());
        let matvec = |m: &BlasDiagonalCiphertext<BE>,
                      prepared: &mut BlasDiagonalCiphertextPrepared<BE>,
                      model: &DiagonalCiphertextModel,
                      label: &str,
                      scratch: &mut ScratchArena<'_, BE>| {
            prepare_diagonal_ciphertext(module, prepared, m);
            let mut res = alloc_vector(&params, module, whole, res_k);
            assert_eq!(
                blas_matvec_diagonal_output_layout(&res, m, &a),
                want_layout,
                "{label}: output layout"
            );
            let bytes = module.blas_matvec_diagonal_tmp_bytes(&res, &*prepared, &a, &params.tsk_layout(), &params.atk_layout());
            module
                .blas_matvec_diagonal(&mut res, &*prepared, &a, &keys, &mut poisoned_scratch::<BE>(bytes).borrow())
                .unwrap();
            let mut streamed = alloc_vector(&params, module, whole, res_k);
            let bytes = module.blas_matvec_diagonal_tmp_bytes(&streamed, m, &a, &params.tsk_layout(), &params.atk_layout());
            module
                .blas_matvec_diagonal(&mut streamed, m, &a, &keys, &mut poisoned_scratch::<BE>(bytes).borrow())
                .unwrap();
            assert!(
                vector_snapshot::<BE>(&streamed) == vector_snapshot::<BE>(&res),
                "{label}: streamed against resident"
            );
            assert_vector_decrypt::<BE, F, E>(
                label,
                &params,
                module,
                &encoder,
                &res,
                &sk,
                &model.matvec(&x),
                slots,
                &want_layout,
                scratch,
            );
        };
        matvec(
            &m,
            &mut prepared,
            &model,
            &format!("{name}: a zero matrix"),
            &mut scratch.borrow(),
        );

        for (axis, at, mask, seed) in APPENDS {
            let (mask, mask_prepared) = &masks[mask];
            let v: Vec<f64> = (0..slots).map(|e| vector_entry(e * seed + seed)).collect();
            let ct_v = encrypt(&v, &mut scratch.borrow());
            let bytes = append_tmp_bytes(module, axis, &m, &ct_v, mask_prepared, &params.atk_layout());
            let mut op_scratch = poisoned_scratch::<BE>(bytes);
            append(
                module,
                axis,
                &mut m,
                at,
                &ct_v,
                mask_prepared,
                &keys,
                &mut op_scratch.borrow(),
            )
            .unwrap();
            model.append(axis, at, mask, &v);
            let label = format!("{name}: after {} at {at}", axis.op());
            assert!(
                m.diagonals()
                    .giant_steps
                    .iter()
                    .flat_map(|gs| &gs.diagonals)
                    .all(|d| CKKSLayout {
                        glwe_layout: d.plaintext.glwe_layout(),
                        meta: d.plaintext.meta(),
                    } == layout.ct
                        && d.plaintext.noise().is_none()),
                "{label}: the diagonals left the matrix's layout"
            );
            matvec(&m, &mut prepared, &model, &label, &mut scratch.borrow());

            // The append on a store answering only at the widths its key-use query lists, then one byte short.
            let uses = append_key_uses(module, axis, &m, &ct_v);
            let strict = keys.at_widths(uses.rotations.iter().map(|&(p, k)| (p, k.as_usize())), []);
            let mut copy = module.blas_diagonal_ciphertext_alloc_from_infos(layout);
            with_scratch::<BE, _>(bytes, |s| {
                append(module, axis, &mut copy, at, &ct_v, mask_prepared, &strict, s)
            })
            .unwrap_or_else(|e| panic!("{label}: on the listed keys: {e}"));
            let before = diagonal_ciphertext_snapshot::<BE>(&copy);
            assert_short_scratch_panics::<BE, _>(axis.op(), bytes, |s| {
                append(module, axis, &mut copy, at, &ct_v, mask_prepared, &keys, s)
            });
            assert!(
                diagonal_ciphertext_snapshot::<BE>(&copy) == before,
                "{label}: wrote the matrix before panicking"
            );
        }

        // The product, resident and streamed, on a store answering only at the listed widths, then one byte short; the
        // preparation one byte short.
        let res = alloc_vector(&params, module, whole, res_k);
        let uses = blas_matvec_diagonal_key_uses(module, &res, &m, &a);
        let strict = keys.at_widths(
            uses.rotations.iter().map(|&(p, k)| (p, k.as_usize())),
            uses.tensor.map(|k| k.as_usize()),
        );
        for form in ["resident", "streamed"] {
            let bytes = if form == "resident" {
                module.blas_matvec_diagonal_tmp_bytes(&res, &prepared, &a, &params.tsk_layout(), &params.atk_layout())
            } else {
                module.blas_matvec_diagonal_tmp_bytes(&res, &m, &a, &params.tsk_layout(), &params.atk_layout())
            };
            let mut out = alloc_vector(&params, module, whole, res_k);
            with_scratch::<BE, _>(bytes, |s| {
                if form == "resident" {
                    module.blas_matvec_diagonal(&mut out, &prepared, &a, &strict, s)
                } else {
                    module.blas_matvec_diagonal(&mut out, &m, &a, &strict, s)
                }
            })
            .unwrap_or_else(|e| panic!("{name}: {form} matvec_diagonal on the listed keys: {e}"));
            let before = vector_snapshot::<BE>(&out);
            assert_short_scratch_panics::<BE, _>("blas_matvec_diagonal", bytes, |s| {
                if form == "resident" {
                    module.blas_matvec_diagonal(&mut out, &prepared, &a, &keys, s)
                } else {
                    module.blas_matvec_diagonal(&mut out, &m, &a, &keys, s)
                }
            });
            assert!(
                vector_snapshot::<BE>(&out) == before,
                "{name}: {form} matvec_diagonal wrote res before panicking"
            );
        }
        let bytes = module.blas_diagonal_ciphertext_prepare_tmp_bytes(layout);
        assert_short_scratch_panics::<BE, _>("blas_diagonal_ciphertext_prepare", bytes, |s| {
            module.blas_diagonal_ciphertext_prepare(&mut prepared, &m, s)
        });
    }

    // The diagonal `0` alone: no rotation, so no rotation key, and every call still asserts its scratch.
    let diagonal = diagonal_ciphertext_with(&by_rotation, vec![0], LinearTransformationStrategy::Direct);
    assert!(
        diagonal_ciphertext_elements(order, &diagonal).is_empty(),
        "the diagonal 0 rotates nothing"
    );
    let tensor_only = TestKeys::<BE> {
        atks: Default::default(),
        tsk: Some(gen_tsk(&params, module, &sk_raw, &mut scratch.borrow())),
    };
    let mut m = module.blas_diagonal_ciphertext_alloc_from_infos(&diagonal);
    let mut model = DiagonalCiphertextModel::new(slots, vec![0]);
    let (mask, mask_prepared) = &masks[1];
    for (axis, at) in [(AppendAxis::Row, 2), (AppendAxis::Col, 5)] {
        let v: Vec<f64> = (0..slots).map(|e| vector_entry(e + at as usize)).collect();
        let ct_v = encrypt(&v, &mut scratch.borrow());
        let bytes = append_tmp_bytes(module, axis, &m, &ct_v, mask_prepared, &params.atk_layout());
        let mut copy = module.blas_diagonal_ciphertext_alloc_from_infos(&diagonal);
        assert_short_scratch_panics::<BE, _>(axis.op(), bytes, |s| {
            append(module, axis, &mut copy, at, &ct_v, mask_prepared, &tensor_only, s)
        });
        with_scratch::<BE, _>(bytes, |s| {
            append(module, axis, &mut m, at, &ct_v, mask_prepared, &tensor_only, s)
        })
        .unwrap();
        model.append(axis, at, mask, &v);
    }
    let prepared = prepared_of(module, &m);
    let mut streamed = alloc_vector(&params, module, whole, res_k);
    let bytes = module.blas_matvec_diagonal_tmp_bytes(&streamed, &m, &a, &params.tsk_layout(), &params.atk_layout());
    assert_short_scratch_panics::<BE, _>("blas_matvec_diagonal", bytes, |s| {
        module.blas_matvec_diagonal(&mut streamed, &m, &a, &tensor_only, s)
    });
    with_scratch::<BE, _>(bytes, |s| module.blas_matvec_diagonal(&mut streamed, &m, &a, &tensor_only, s)).unwrap();
    let mut res = alloc_vector(&params, module, whole, res_k);
    let bytes = module.blas_matvec_diagonal_tmp_bytes(&res, &prepared, &a, &params.tsk_layout(), &params.atk_layout());
    assert_short_scratch_panics::<BE, _>("blas_matvec_diagonal", bytes, |s| {
        module.blas_matvec_diagonal(&mut res, &prepared, &a, &tensor_only, s)
    });
    with_scratch::<BE, _>(bytes, |s| {
        module.blas_matvec_diagonal(&mut res, &prepared, &a, &tensor_only, s)
    })
    .unwrap();
    assert!(
        vector_snapshot::<BE>(&streamed) == vector_snapshot::<BE>(&res),
        "the diagonal 0: streamed against resident"
    );
    assert_vector_decrypt::<BE, F, E>(
        "the diagonal 0",
        &params,
        module,
        &encoder,
        &res,
        &sk,
        &model.matvec(&x),
        slots,
        &want_layout,
        &mut scratch.borrow(),
    );
}

/// `blas_matvec_diagonal`, `blas_append_row_diagonal` and `blas_append_col_diagonal` reject, before any write: a vector that is not a whole
/// ciphertext, as input and as `res`; an input of another radix than `res`, or at a radix above 62; a matrix of another
/// module's slots, or whose diagonals are of another radix or rank than the vectors; a mask that is not a plaintext or
/// of another radix; an append that would change the diagonals' slot kind or sparsity; a store lacking a rotation key
/// or the tensor key, and operands of another rank than the keys; and an appended vector whose product does not land in
/// the diagonals' frame, which poulpy-ckks's accumulate refuses. The resident product rejects the missing keys alike, a
/// preparation into another layout panics, and a matrix layout the module cannot hold panics at allocation.
pub fn test_blas_matvec_diagonal_guards<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard> + BlasMatVecImpl + BlasEncodingImpl,
    Module<BE>: TestContextModule<BE> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    F: TestScalar + DiagonalArithmetic,
    E: ReimNew<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let slots = params.n / 2;
    if slots < 16 {
        eprintln!(
            "matvec_diagonal_guards skipped: its matrix holds diagonals up to 8 on at least 16 slots, the ring holds {slots}"
        );
        return;
    }
    let encoder = ReferenceEncoder::<E>::new(slots).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let log_delta = params.prec().log_delta();
    let layout = diagonal_ciphertext_layout(&params, slots, params.k - log_delta);
    let order = module.cyclotomic_order();
    let keys = TestKeys::<BE> {
        atks: rotation_keys(
            &params,
            module,
            &sk_raw,
            diagonal_ciphertext_elements(order, &layout),
            &mut scratch.borrow(),
        ),
        tsk: Some(gen_tsk(&params, module, &sk_raw, &mut scratch.borrow())),
    };
    let no_tensor = TestKeys::<BE> {
        atks: rotation_keys(
            &params,
            module,
            &sk_raw,
            matvec_diagonal_galois_elements(order, &layout),
            &mut scratch.borrow(),
        ),
        tsk: None,
    };
    let empty = TestKeys::<BE> {
        atks: Default::default(),
        tsk: None,
    };
    let whole = BlasVectorShape::new(slots, slots, 0, 1);
    let values: Vec<f64> = (0..slots).map(vector_entry).collect();
    let encrypt = |shape: BlasVectorShape, k: usize, scratch: &mut ScratchArena<'_, BE>| {
        encrypt_vector::<BE, F, E>(&params, module, host_module, &encoder, &sk, k, shape, slots, &values, scratch)
    };
    let a = encrypt(whole, params.k, &mut scratch.borrow());
    let strided_shape = BlasVectorShape::new(slots / 2, slots / 2, 0, 2);
    let strided = encrypt(strided_shape, params.k, &mut scratch.borrow());
    let mask_at = |base2k: usize| {
        let mut re = vec![0.0; slots];
        re[0] = 1.0;
        let pt = encode_and_upload_pt(
            host_module,
            module,
            &encoder,
            base2k.into(),
            params.prec(),
            &to_f::<F>(&re),
            &to_f::<F>(&vec![0.0; slots]),
        );
        with_scratch::<BE, _>(module.ckks_right_operand_prepare_tmp_bytes(&pt), |s| {
            module.ckks_right_operand_prepare_alloc(&pt, PrepareHint::Reuse, s)
        })
        .unwrap()
    };
    let mask = mask_at(params.base2k);
    let ct_mask = with_scratch::<BE, _>(module.ckks_right_operand_prepare_tmp_bytes(&a.cts()[0]), |s| {
        module.ckks_right_operand_prepare_alloc(&a.cts()[0], PrepareHint::Reuse, s)
    })
    .unwrap();
    // Zero operands of another radix, then of another rank, than `params`'.
    let other_radix = CKKSTestParams {
        base2k: params.base2k - 1,
        ..params
    };
    let other_rank = CKKSTestParams {
        rank: params.rank + 1,
        ..params
    };
    let at_params = |p: &CKKSTestParams| {
        let mut l = diagonal_ciphertext_layout(p, slots, params.k - log_delta);
        l.ct.glwe_layout = GLWELayout {
            k: l.ct.k(),
            ..p.glwe_layout()
        };
        l
    };

    let m = module.blas_diagonal_ciphertext_alloc_from_infos(&layout);
    let mut res = alloc_vector(&params, module, whole, params.k);
    let bytes = module.blas_matvec_diagonal_tmp_bytes(&res, &m, &a, &params.tsk_layout(), &params.atk_layout());
    let product = |label: &str,
                   rejection: Rejection,
                   res: &mut BlasVectorOwned<BE>,
                   m: &BlasDiagonalCiphertext<BE>,
                   a: &BlasVectorOwned<BE>,
                   keys: &TestKeys<BE>| {
        assert_vector_guard::<BE>(label, "blas_matvec_diagonal", rejection, res, |res| {
            with_scratch::<BE, _>(bytes, |s| module.blas_matvec_diagonal(res, m, a, keys, s))
        });
    };
    product("a strided input", InvalidOperand, &mut res, &m, &strided, &keys);
    let mut strided_res = alloc_vector(&params, module, strided_shape, params.k);
    product("a strided res", InvalidOperand, &mut strided_res, &m, &a, &keys);
    let mut radix_res = alloc_vector(&other_radix, module, whole, params.k);
    product(
        "an input of another radix than res",
        InvalidOperand,
        &mut radix_res,
        &m,
        &a,
        &keys,
    );
    let wide_radix = CKKSTestParams { base2k: 63, ..params };
    let (mut wide_res, wide_a) = (
        alloc_vector(&wide_radix, module, whole, 126),
        alloc_vector(&wide_radix, module, whole, 126),
    );
    product("a radix above 62", InvalidOperand, &mut wide_res, &m, &wide_a, &keys);
    let doubled = Module::<BE>::new(2 * params.n as u64);
    let mut doubled_layout = diagonal_ciphertext_layout(&params, 2 * slots, params.k - log_delta);
    doubled_layout.ct.glwe_layout.n = (2 * params.n).into();
    let foreign = doubled.blas_diagonal_ciphertext_alloc_from_infos(&doubled_layout);
    product(
        "a matrix of another module's slots",
        InvalidOperand,
        &mut res,
        &foreign,
        &a,
        &keys,
    );
    let m_radix = module.blas_diagonal_ciphertext_alloc_from_infos(&at_params(&other_radix));
    product("diagonals of another radix", InvalidOperand, &mut res, &m_radix, &a, &keys);
    let m_rank = module.blas_diagonal_ciphertext_alloc_from_infos(&at_params(&other_rank));
    product("diagonals of another rank", InvalidOperand, &mut res, &m_rank, &a, &keys);
    if !matvec_diagonal_galois_elements(order, &layout).is_empty() {
        product("no rotation key", MissingKey, &mut res, &m, &a, &empty);
    }
    product("no tensor key", MissingTensorKey, &mut res, &m, &a, &no_tensor);
    let mut empty_res = alloc_vector(&params, module, whole, 0);
    let empty_bytes = module.blas_matvec_diagonal_tmp_bytes(&empty_res, &m, &a, &params.tsk_layout(), &params.atk_layout());
    assert_vector_guard::<BE>(
        "a res of no width",
        "ckks_linear_transformation_eval_into",
        Budget,
        &mut empty_res,
        |res| with_scratch::<BE, _>(empty_bytes, |s| module.blas_matvec_diagonal(res, &m, &a, &keys, s)),
    );
    // The resident form runs the same validator; its preparation refuses another layout.
    let mut prepared = prepared_of(module, &m);
    let resident_bytes = module.blas_matvec_diagonal_tmp_bytes(&res, &prepared, &a, &params.tsk_layout(), &params.atk_layout());
    for (label, rejection, keys) in [
        ("resident, no rotation key", MissingKey, &empty),
        ("resident, no tensor key", MissingTensorKey, &no_tensor),
    ] {
        if rejection == MissingKey && matvec_diagonal_galois_elements(order, &layout).is_empty() {
            continue;
        }
        assert_vector_guard::<BE>(label, "blas_matvec_diagonal", rejection, &mut res, |res| {
            with_scratch::<BE, _>(resident_bytes, |s| module.blas_matvec_diagonal(res, &prepared, &a, keys, s))
        });
    }
    let empty_bytes =
        module.blas_matvec_diagonal_tmp_bytes(&empty_res, &prepared, &a, &params.tsk_layout(), &params.atk_layout());
    assert_vector_guard::<BE>(
        "resident, a res of no width",
        "ckks_linear_transformation_eval_into",
        Budget,
        &mut empty_res,
        |res| with_scratch::<BE, _>(empty_bytes, |s| module.blas_matvec_diagonal(res, &prepared, &a, &keys, s)),
    );
    let other = diagonal_ciphertext_with(&layout, vec![0, 1], LinearTransformationStrategy::Direct);
    let m_other = module.blas_diagonal_ciphertext_alloc_from_infos(&other);
    assert_panics("a preparation into another layout", "another layout", || {
        with_scratch::<BE, _>(module.blas_diagonal_ciphertext_prepare_tmp_bytes(&layout), |s| {
            module.blas_diagonal_ciphertext_prepare(&mut prepared, &m_other, s)
        })
    });
    let (mut rank_res, rank_a) = (
        alloc_vector(&other_rank, module, whole, params.k),
        alloc_vector(&other_rank, module, whole, params.k),
    );
    product(
        "operands of another rank than the keys",
        InvalidOperand,
        &mut rank_res,
        &m_rank,
        &rank_a,
        &keys,
    );
    let diagonal = diagonal_ciphertext_with(&at_params(&other_rank), vec![0], LinearTransformationStrategy::Direct);
    let m_diagonal = module.blas_diagonal_ciphertext_alloc_from_infos(&diagonal);
    product(
        "operands of another rank than the tensor key",
        InvalidOperand,
        &mut rank_res,
        &m_diagonal,
        &rank_a,
        &keys,
    );

    let real = CKKSTestParams {
        prec_meta: GLWEPlaintextMeta {
            slots: SlotsKind::Real,
            ..params.prec_meta
        },
        ..params
    };
    let narrower = encrypt(whole, params.k - params.base2k, &mut scratch.borrow());
    let rank_v = alloc_vector(&other_rank, module, whole, params.k);
    for axis in [AppendAxis::Row, AppendAxis::Col] {
        let op = axis.op();
        let check = |label: &str,
                     rejection: Rejection,
                     raised: &str,
                     layout: &BlasDiagonalCiphertextLayout,
                     v: &BlasVectorOwned<BE>,
                     mask: &CKKSRightOperandPrepared<BE>,
                     keys: &TestKeys<BE>| {
            let mut m = module.blas_diagonal_ciphertext_alloc_from_infos(layout);
            let before = diagonal_ciphertext_snapshot::<BE>(&m);
            let bytes = append_tmp_bytes(module, axis, &m, v, mask, &params.atk_layout());
            assert_rejected(
                &format!("{op}: {label}"),
                raised,
                rejection,
                with_scratch::<BE, _>(bytes, |s| append(module, axis, &mut m, 1, v, mask, keys, s)),
            );
            assert!(
                diagonal_ciphertext_snapshot::<BE>(&m) == before,
                "{op}: {label}: the append wrote the matrix"
            );
        };
        check("a strided vector", InvalidOperand, op, &layout, &strided, &mask, &keys);
        check(
            "diagonals of another radix",
            InvalidOperand,
            op,
            &at_params(&other_radix),
            &a,
            &mask,
            &keys,
        );
        check(
            "diagonals of another rank",
            InvalidOperand,
            op,
            &at_params(&other_rank),
            &a,
            &mask,
            &keys,
        );
        check("a ciphertext mask", InvalidOperand, op, &layout, &a, &ct_mask, &keys);
        check(
            "a mask of another radix",
            InvalidOperand,
            op,
            &layout,
            &a,
            &mask_at(params.base2k - 1),
            &keys,
        );
        check(
            "complex slots into a real matrix",
            InvalidOperand,
            op,
            &at_params(&real),
            &a,
            &mask,
            &keys,
        );
        let mut sparse = layout.clone();
        sparse.ct.meta.log_sparsity = 1;
        check(
            "a dense vector into a sparse matrix",
            InvalidOperand,
            op,
            &sparse,
            &a,
            &mask,
            &keys,
        );
        let rotations = match axis {
            AppendAxis::Row => append_row_diagonal_galois_elements(order, &layout),
            AppendAxis::Col => append_col_diagonal_galois_elements(order, &layout),
        };
        if !rotations.is_empty() {
            check("no rotation key", MissingKey, op, &layout, &a, &mask, &empty);
            check(
                "a vector of another rank than the keys",
                InvalidOperand,
                op,
                &at_params(&other_rank),
                &rank_v,
                &mask,
                &keys,
            );
        }
        // One limb narrower: its products fall short of the diagonals' width.
        check(
            "a vector off the diagonals' frame",
            InvalidOperand,
            "ckks_dot_product_prepared_pt_vec_add",
            &layout,
            &narrower,
            &mask,
            &keys,
        );
    }

    // Layouts the module cannot hold.
    let alloc = |label: &str, edit: &dyn Fn(&mut BlasDiagonalCiphertextLayout)| {
        let mut bad = layout.clone();
        edit(&mut bad);
        assert_panics(label, "the matrix", || module.blas_diagonal_ciphertext_alloc_from_infos(&bad));
    };
    alloc("another slot count", &|l| l.lt.slots = 2 * slots);
    alloc("no diagonal", &|l| l.lt.indexes.clear());
    alloc("a diagonal past the slots", &|l| l.lt.indexes.push(slots as i64));
    alloc("another degree", &|l| l.ct.glwe_layout.n = (2 * params.n).into());
    alloc("plaintext diagonals", &|l| l.ct.glwe_layout.rank = 0usize.into());
    alloc("a radix above 62", &|l| l.ct.glwe_layout.base2k = 63usize.into());
}
