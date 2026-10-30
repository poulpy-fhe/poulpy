//! Runners for products of a vector by a public matrix: a `d x d` matrix by one ciphertext of `d` entries spread at
//! stride `S / d` over its `S` slots, the operand resident or streamed.

use criterion::{Bencher, measurement::Measurement};
use poulpy_blas::{
    api::{BlasDiagonalCiphertextPreparedFactory, BlasEncodingOps, BlasMatVecOperandPreparedFactory, BlasMatVecOps},
    layouts::{
        BlasDiagonalCiphertext, BlasDiagonalCiphertextLayout, BlasGridLine, BlasMatVecMaps, BlasMatVecOperandLayout,
        BlasModuleAlloc, BlasVector, BlasVectorLayout, BlasVectorShape, append_row_diagonal_galois_elements,
        blas_grid_to_vector_max_base2k, blas_matvec_diagonal_max_base2k, blas_matvec_pt_max_base2k,
        matvec_diagonal_galois_elements, matvec_pt_galois_elements,
    },
    oep::BlasMatVecImpl,
};
use poulpy_ckks::{
    CoeffsMeta, SetCKKSInfos,
    api::{CKKSEncodingHostOps, CKKSMulOps, LinearTransformationLayout, LinearTransformationStrategy, optimal_bsgs_giant_step},
    layouts::{CKKSModuleAlloc, CKKSPreparedRight},
};
use poulpy_core::{
    Result as CoreResult,
    layouts::{
        GetAutomorphismKey, GetTensorKey, TorusPrecision,
        prepared::{
            GLWEAutomorphismKeyPrepared, GLWEAutomorphismKeyPreparedBackendRef, GLWEAutomorphismKeyPreparedToBackendRef,
            GLWETensorKeyPrepared, GLWETensorKeyPreparedBackendRef, GLWETensorKeyPreparedToBackendRef,
        },
    },
};
use poulpy_hal::{
    api::{ModuleSynchronize, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, CyclotomicOrder, HostStaged, MaxBase2k, Module, PrepareHint, ScratchOwned, Standard},
};
use std::collections::HashMap;

use crate::{
    blas::{
        matmul::{BlasBenchModule, Client, FAILURE_BITS, HEADROOM, atk_layout, base2k, block_layout, entry, tsk_layout, verify},
        params::BlasBenchParams,
    },
    timed,
};

/// The operand's layout: `d` entries at stride `S / d` on each side, one ciphertext, dense, at the cost-optimal BSGS
/// giant step.
fn layout(bp: &BlasBenchParams) -> BlasMatVecOperandLayout {
    let (slots, d) = (bp.slots(), bp.d());
    let shape = BlasVectorShape::new(d, d, 0, slots / d);
    let meta = CoeffsMeta::from_delta_budget(bp.p, HEADROOM);
    BlasMatVecOperandLayout::new(slots, shape, slots, shape, slots, base2k(bp), meta).with_optimal_bsgs()
}

/// The public matrix, scaled so that the products stay in `[-1, 1)`.
fn w_entry(d: usize) -> impl Fn(usize, usize) -> f64 {
    move |r, c| entry(4, 0, r, c) / d as f64
}

/// The encrypted vector.
fn x_entry(e: usize) -> f64 {
    entry(5, 0, 0, e)
}

/// `blas_matvec_pt` by the operand, prepared when `resident`, else streamed from its encoding.
fn matvec<BE, M>(bencher: &mut Bencher<'_, M>, bp: &BlasBenchParams, op: &'static str, resident: bool)
where
    BE: Backend<Ring = Standard> + HostStaged + 'static + BlasMatVecImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    timed::<BE, M, _>(bencher, op, bp, || {
        let mut client = Client::<BE, BE>::new(bp);
        let (layout, d) = (layout(bp), bp.d());
        let module = &client.module;
        let mut streamed = module.blas_matvec_operand_alloc_from_infos(&layout);
        let mut prepared = module.blas_matvec_operand_prepared_alloc_from_infos(&layout);
        let mut arena = ScratchOwned::<BE>::alloc(
            module
                .blas_matvec_operand_encode_tmp_bytes(&layout)
                .max(module.blas_matvec_operand_prepare_tmp_bytes(&layout)),
        );
        module
            .blas_matvec_operand_encode(&mut streamed, w_entry(d), &mut arena.borrow())
            .unwrap();
        module.blas_matvec_operand_prepare(&mut prepared, &streamed, &mut arena.borrow());
        let atks = client.rotation_keys(bp, matvec_pt_galois_elements(client.module.cyclotomic_order(), &layout));
        let x: Vec<f64> = (0..d).map(x_entry).collect();
        let mut slots = vec![0.0; bp.slots()];
        x.iter().enumerate().for_each(|(e, &v)| slots[e * layout.input.stride] = v);
        let a = BlasVector::from_cts(vec![client.encrypt_slots(bp, &slots)], layout.input).unwrap();
        let max_base2k = blas_matvec_pt_max_base2k::<BE>(&layout, &a, FAILURE_BITS);
        assert!(
            max_base2k >= bp.base2k,
            "blas_matvec_pt: base2k exceeds the radix its lazy sums allow at 128 failure bits"
        );
        let mut res = client
            .module
            .blas_vector_alloc(layout.output, base2k(bp), TorusPrecision(bp.k as u32));
        let mut scratch = if resident {
            run(&client.module, &mut res, &prepared, &a, &atks, bp)
        } else {
            run(&client.module, &mut res, &streamed, &a, &atks, bp)
        };
        let (re, im) = client.decrypt_slots(bp, res.ct(0));
        let w = w_entry(d);
        let mut max_err = im.iter().fold(0.0f64, |err, v| err.max(v.abs()));
        for (s, have) in re.iter().enumerate() {
            let want = if s.is_multiple_of(layout.output.stride) {
                (0..d).map(|c| w(s / layout.output.stride, c) * x[c]).sum()
            } else {
                0.0
            };
            max_err = max_err.max((have - want).abs());
        }
        verify::<BE>(op, bp, max_err, Some(max_base2k), atks.len());
        let module = client.module;
        if resident {
            Box::new(move || {
                module
                    .blas_matvec_pt(&mut res, &prepared, &a, &atks, &mut scratch.borrow())
                    .unwrap();
                module.synchronize();
            })
        } else {
            Box::new(move || {
                module
                    .blas_matvec_pt(&mut res, &streamed, &a, &atks, &mut scratch.borrow())
                    .unwrap();
                module.synchronize();
            })
        }
    });
}

/// One call into `res`, on the scratch its query reports, which it returns.
fn run<BE, W, K>(
    module: &Module<BE>,
    res: &mut poulpy_blas::layouts::BlasVectorOwned<BE>,
    w: &W,
    a: &poulpy_blas::layouts::BlasVectorOwned<BE>,
    keys: &K,
    bp: &BlasBenchParams,
) -> ScratchOwned<BE>
where
    BE: Backend<Ring = Standard> + 'static + BlasMatVecImpl,
    W: BlasMatVecMaps<BE>,
    K: poulpy_core::layouts::GetAutomorphismKey<BE> + poulpy_core::layouts::GetTensorKey<BE> + Sync,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let bytes = module.blas_matvec_pt_tmp_bytes(&*res, w, a, &atk_layout(bp));
    let mut scratch = ScratchOwned::<BE>::alloc(bytes);
    module.blas_matvec_pt(res, w, a, keys, &mut scratch.borrow()).unwrap();
    scratch
}

/// `blas_matvec_pt` by the prepared operand.
pub fn runner_blas_matvec_pt_resident<BE, M>(bencher: &mut Bencher<'_, M>, bp: &BlasBenchParams)
where
    BE: Backend<Ring = Standard> + HostStaged + 'static + BlasMatVecImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    matvec::<BE, M>(bencher, bp, "blas_matvec_pt_resident", true);
}

/// `blas_matvec_pt` streaming the operand's encoded diagonals.
pub fn runner_blas_matvec_pt_streamed<BE, M>(bencher: &mut Bencher<'_, M>, bp: &BlasBenchParams)
where
    BE: Backend<Ring = Standard> + HostStaged + 'static + BlasMatVecImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    matvec::<BE, M>(bencher, bp, "blas_matvec_pt_streamed", false);
}

/// A diagonal ciphertext of the diagonals `0..d` over the slots, at the cost-optimal giant step, whose diagonals hold the
/// frame of a ciphertext of the point's width through a mask of scale `p`.
fn diagonal_ciphertext_layout(bp: &BlasBenchParams) -> BlasDiagonalCiphertextLayout {
    let (slots, d) = (bp.slots(), bp.d() as i64);
    let mut ct = block_layout::<Standard>(bp);
    ct.glwe_layout.k = TorusPrecision((bp.k - bp.p) as u32);
    BlasDiagonalCiphertextLayout::new(
        ct,
        LinearTransformationLayout {
            indexes: (0..d).collect(),
            slots,
            strategy: LinearTransformationStrategy::Bsgs {
                giant_step: optimal_bsgs_giant_step(0..d, slots),
            },
        },
    )
}

/// The prepared mask of slot 0, at scale `p`.
fn slot_mask<BE>(client: &Client<BE, BE>, bp: &BlasBenchParams) -> CKKSPreparedRight<BE>
where
    BE: Backend<Ring = Standard> + HostStaged + 'static,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let module = &client.module;
    let meta = CoeffsMeta::from_delta_budget(bp.p, HEADROOM);
    let mut pt = module.ckks_pt_vec_alloc(bp.slots(), base2k(bp), meta.k);
    SetCKKSInfos::set_meta(&mut pt, meta.meta);
    let mut re = vec![0.0; bp.slots()];
    re[0] = 1.0;
    let mut arena = ScratchOwned::<BE>::alloc(
        <Module<BE> as CKKSEncodingHostOps<BE, f64>>::ckks_reim_tmp_bytes(module, bp.slots())
            .max(module.ckks_prepare_tmp_bytes(&pt)),
    );
    module
        .ckks_encode_reim_into(&mut pt, &re, &vec![0.0; bp.slots()], &mut arena.borrow())
        .unwrap();
    module
        .ckks_prepare_right(&pt, PrepareHint::Reuse, &mut arena.borrow())
        .unwrap()
}

/// Row `p` of the diagonal ciphertext, `p < 3`: entry `j` is `M[p][j]`, scaled so that the products stay in `[-1, 1)`.
fn matrix_row(bp: &BlasBenchParams, p: usize) -> Vec<f64> {
    (0..bp.slots()).map(|j| entry(6, p, 0, j) / bp.d() as f64).collect()
}

/// The rotation keys and the tensor key of a product by a diagonal ciphertext, as the one source it reads.
struct ProductKeys<BE: Backend> {
    atks: HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>>,
    tsk: GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
}

impl<BE: Backend> GetAutomorphismKey<BE> for ProductKeys<BE>
where
    GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>: GLWEAutomorphismKeyPreparedToBackendRef<BE>,
{
    fn lookup_automorphism_key(&self, p: i64, k: TorusPrecision) -> CoreResult<GLWEAutomorphismKeyPreparedBackendRef<'_, BE>> {
        self.atks.lookup_automorphism_key(p, k)
    }
}

impl<BE: Backend> GetTensorKey<BE> for ProductKeys<BE>
where
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE>,
{
    fn lookup_tensor_key(&self, k: TorusPrecision) -> CoreResult<GLWETensorKeyPreparedBackendRef<'_, BE>> {
        self.tsk.lookup_tensor_key(k)
    }
}

/// The call a diagonal-ciphertext runner times.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Timed {
    Append,
    Resident,
    Streamed,
}

/// `blas_append_row_diagonal` of one more row into a matrix holding three appended rows, or the matrix's
/// `blas_matvec_diagonal`, prepared once or streamed, each on its own key set: the rows' rotations are baby steps of the
/// product's, so the product's keys append them too.
fn diagonal_ciphertext<BE, M>(bencher: &mut Bencher<'_, M>, bp: &BlasBenchParams, op: &'static str, timed_call: Timed)
where
    BE: Backend<Ring = Standard> + HostStaged + 'static + BlasMatVecImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    timed::<BE, M, _>(bencher, op, bp, || {
        let mut client = Client::<BE, BE>::new(bp);
        let (layout, slots, d) = (diagonal_ciphertext_layout(bp), bp.slots(), bp.d());
        let order = client.module.cyclotomic_order();
        let elements = if timed_call == Timed::Append {
            append_row_diagonal_galois_elements(order, &layout)
        } else {
            matvec_diagonal_galois_elements(order, &layout)
        };
        let atks = client.rotation_keys(bp, elements);
        let mask = slot_mask(&client, bp);
        let whole = BlasVectorShape::new(slots, slots, 0, 1);
        let mut m = client.module.blas_diagonal_ciphertext_alloc_from_infos(&layout);
        let rows: Vec<Vec<f64>> = (0..3).map(|p| matrix_row(bp, p)).collect();
        let append_bytes = client.module.blas_append_row_diagonal_tmp_bytes(
            &layout,
            &BlasVectorLayout::new(block_layout::<Standard>(bp), whole),
            &mask,
            &atk_layout(bp),
        );
        let mut append_scratch = ScratchOwned::<BE>::alloc(append_bytes);
        for (p, row) in rows.iter().enumerate() {
            let v = BlasVector::from_cts(vec![client.encrypt_slots(bp, row)], whole).unwrap();
            client
                .module
                .blas_append_row_diagonal(&mut m, p as i64, &v, &mask, &atks, &mut append_scratch.borrow())
                .unwrap();
        }
        let x: Vec<f64> = (0..slots).map(|e| entry(5, 0, 0, e)).collect();
        let a = BlasVector::from_cts(vec![client.encrypt_slots(bp, &x)], whole).unwrap();
        if timed_call == Timed::Append {
            let max_err = diagonals_error(&mut client, bp, &m, &rows);
            verify::<BE>(op, bp, max_err, None, atks.len());
            let module = client.module;
            return Box::new(move || {
                module
                    .blas_append_row_diagonal(&mut m, 0, &a, &mask, &atks, &mut append_scratch.borrow())
                    .unwrap();
                module.synchronize();
            });
        }
        let keys = ProductKeys {
            atks,
            tsk: client.tensor_key(bp),
        };
        let mut res = client
            .module
            .blas_vector_alloc(whole, base2k(bp), TorusPrecision(bp.k as u32));
        let module = &client.module;
        let mut prepared = module.blas_diagonal_ciphertext_prepared_alloc_from_infos(&layout);
        let mut arena = ScratchOwned::<BE>::alloc(module.blas_diagonal_ciphertext_prepare_tmp_bytes(&layout));
        module.blas_diagonal_ciphertext_prepare(&mut prepared, &m, &mut arena.borrow());
        let (tsk_layout, atk_layout) = (tsk_layout(bp), atk_layout(bp));
        let mut scratch = if timed_call == Timed::Resident {
            let mut scratch =
                ScratchOwned::<BE>::alloc(module.blas_matvec_diagonal_tmp_bytes(&res, &prepared, &a, &tsk_layout, &atk_layout));
            module
                .blas_matvec_diagonal(&mut res, &prepared, &a, &keys, &mut scratch.borrow())
                .unwrap();
            scratch
        } else {
            let mut scratch =
                ScratchOwned::<BE>::alloc(module.blas_matvec_diagonal_tmp_bytes(&res, &m, &a, &tsk_layout, &atk_layout));
            module
                .blas_matvec_diagonal(&mut res, &m, &a, &keys, &mut scratch.borrow())
                .unwrap();
            scratch
        };
        let max_base2k = blas_matvec_diagonal_max_base2k::<BE>(&m, &a, FAILURE_BITS);
        assert!(
            max_base2k >= bp.base2k,
            "blas_matvec_diagonal: base2k exceeds the radix its lazy sums allow at 128 failure bits"
        );
        let (re, im) = client.decrypt_slots(bp, res.ct(0));
        let mut max_err = im.iter().fold(0.0f64, |err, v| err.max(v.abs()));
        for (i, have) in re.iter().enumerate() {
            let want: f64 = rows
                .get(i)
                .map_or(0.0, |row| (0..d).map(|t| row[(i + t) % slots] * x[(i + t) % slots]).sum());
            max_err = max_err.max((have - want).abs());
        }
        verify::<BE>(op, bp, max_err, Some(max_base2k), keys.atks.len());
        let module = client.module;
        if timed_call == Timed::Resident {
            return Box::new(move || {
                module
                    .blas_matvec_diagonal(&mut res, &prepared, &a, &keys, &mut scratch.borrow())
                    .unwrap();
                module.synchronize();
            });
        }
        Box::new(move || {
            module
                .blas_matvec_diagonal(&mut res, &m, &a, &keys, &mut scratch.borrow())
                .unwrap();
            module.synchronize();
        })
    });
}

/// The largest distance of the decrypted diagonals of `m` to those of the matrix of the rows `rows`: entry `(b, g)`
/// holds `rot(D_(b + g), -g)`, `D_t[j] = M[j][j + t]`, and of their imaginary parts to zero.
fn diagonals_error<BE>(
    client: &mut Client<BE, BE>,
    bp: &BlasBenchParams,
    m: &BlasDiagonalCiphertext<BE>,
    rows: &[Vec<f64>],
) -> f64
where
    BE: Backend<Ring = Standard> + HostStaged + 'static,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let slots = bp.slots() as i64;
    let mut max_err = 0.0f64;
    for gs in &m.diagonals().giant_steps {
        for diagonal in &gs.diagonals {
            let t = diagonal.baby + gs.rot;
            let (re, im) = client.decrypt_slots(bp, &diagonal.plaintext);
            for (i, (have, imag)) in re.iter().zip(&im).enumerate() {
                let j = (i as i64 - gs.rot).rem_euclid(slots);
                let want = rows
                    .get(j as usize)
                    .map_or(0.0, |row| row[(j + t).rem_euclid(slots) as usize]);
                max_err = max_err.max((have - want).abs()).max(imag.abs());
            }
        }
    }
    max_err
}

/// `blas_matvec_diagonal` of a prepared diagonal ciphertext of `d` diagonals by a whole ciphertext.
pub fn runner_blas_matvec_diagonal_resident<BE, M>(bencher: &mut Bencher<'_, M>, bp: &BlasBenchParams)
where
    BE: Backend<Ring = Standard> + HostStaged + 'static + BlasMatVecImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    diagonal_ciphertext::<BE, M>(bencher, bp, "blas_matvec_diagonal_resident", Timed::Resident);
}

/// `blas_matvec_diagonal` of a diagonal ciphertext of `d` diagonals, each prepared at its use, by a whole ciphertext.
pub fn runner_blas_matvec_diagonal_streamed<BE, M>(bencher: &mut Bencher<'_, M>, bp: &BlasBenchParams)
where
    BE: Backend<Ring = Standard> + HostStaged + 'static + BlasMatVecImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    diagonal_ciphertext::<BE, M>(bencher, bp, "blas_matvec_diagonal_streamed", Timed::Streamed);
}

/// `blas_append_row_diagonal` of a whole ciphertext into a diagonal ciphertext of `d` diagonals.
pub fn runner_blas_append_row_diagonal<BE, M>(bencher: &mut Bencher<'_, M>, bp: &BlasBenchParams)
where
    BE: Backend<Ring = Standard> + HostStaged + 'static + BlasMatVecImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    diagonal_ciphertext::<BE, M>(bencher, bp, "blas_append_row_diagonal", Timed::Append);
}

/// `blas_grid_to_vector` of row 1 of the first matrix of a one-block grid holding the dense batch, through the identity,
/// the operand prepared at the cost-optimal BSGS giant step, into one ciphertext of `d` contiguous entries. The operand
/// stores every diagonal of a `d x d` matrix between these shapes, `2d - 1`, whatever its values.
pub fn runner_blas_grid_to_vector_resident<BE, M>(bencher: &mut Bencher<'_, M>, bp: &BlasBenchParams)
where
    BE: Backend<Ring = Standard> + 'static + BlasMatVecImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    let op = "blas_grid_to_vector_resident";
    timed::<BE, M, _>(bencher, op, bp, || {
        let mut client = Client::<BE>::new(bp);
        let (slots, d) = (bp.slots(), bp.d());
        let line = BlasGridLine::Row { mat_slot: 0, row: 1 };
        let a = client.encrypt(bp, 1, 1, bp.batch(), |s, r, c| entry(6, s, r, c));
        let output = BlasVectorShape::new(d, d, 0, 1);
        let meta = CoeffsMeta::from_delta_budget(bp.p, HEADROOM);
        let layout =
            BlasMatVecOperandLayout::new(slots, line.shape(&a), slots, output, slots, base2k(bp), meta).with_optimal_bsgs();
        let module = &client.module;
        let mut streamed = module.blas_matvec_operand_alloc_from_infos(&layout);
        let mut prepared = module.blas_matvec_operand_prepared_alloc_from_infos(&layout);
        let mut arena = ScratchOwned::<BE>::alloc(
            module
                .blas_matvec_operand_encode_tmp_bytes(&layout)
                .max(module.blas_matvec_operand_prepare_tmp_bytes(&layout)),
        );
        let identity = |r: usize, c: usize| f64::from(u8::from(r == c));
        module
            .blas_matvec_operand_encode(&mut streamed, identity, &mut arena.borrow())
            .unwrap();
        module.blas_matvec_operand_prepare(&mut prepared, &streamed, &mut arena.borrow());
        let atks = client.rotation_keys(bp, matvec_pt_galois_elements(client.module.cyclotomic_order(), &layout));
        let max_base2k = blas_grid_to_vector_max_base2k::<BE>(&layout, &a, FAILURE_BITS);
        assert!(
            max_base2k >= bp.base2k,
            "{op}: base2k exceeds the radix its lazy sums allow at 128 failure bits"
        );
        let module = &client.module;
        let mut res = module.blas_vector_alloc(output, base2k(bp), TorusPrecision(bp.k as u32));
        let mut scratch = ScratchOwned::<BE>::alloc(module.blas_grid_to_vector_tmp_bytes(&res, &prepared, &a, &atk_layout(bp)));
        module
            .blas_grid_to_vector(&mut res, &prepared, &a, line, &atks, &mut scratch.borrow())
            .unwrap();
        let (re, im) = client.decrypt_slots(bp, res.ct(0));
        let mut max_err = im.iter().fold(0.0f64, |err, v| err.max(v.abs()));
        for (s, have) in re.iter().enumerate() {
            let want = if s < d { entry(6, 0, 1, s) } else { 0.0 };
            max_err = max_err.max((have - want).abs());
        }
        verify::<BE>(op, bp, max_err, Some(max_base2k), atks.len());
        let module = client.module;
        Box::new(move || {
            module
                .blas_grid_to_vector(&mut res, &prepared, &a, line, &atks, &mut scratch.borrow())
                .unwrap();
            module.synchronize();
        })
    });
}
