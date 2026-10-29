//! Runners for products of a vector by a public matrix: a `d x d` matrix by one ciphertext of `d` entries spread at
//! stride `S / d` over its `S` slots, the operand resident or streamed.

use criterion::{Bencher, measurement::Measurement};
use poulpy_blas::{
    api::{BlasEncodingOps, BlasMatVecOperandPreparedFactory, BlasMatVecOps},
    layouts::{
        BlasMatVecMaps, BlasMatVecOperandLayout, BlasModuleAlloc, BlasVector, BlasVectorShape, blas_matvec_pt_max_base2k,
        matvec_pt_galois_elements,
    },
    oep::BlasMatVecImpl,
};
use poulpy_ckks::CoeffsMeta;
use poulpy_core::layouts::TorusPrecision;
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, CyclotomicOrder, MaxBase2k, Module, ScratchOwned, Standard},
};

use crate::{
    blas::{
        matmul::{BlasBenchModule, Client, FAILURE_BITS, HEADROOM, base2k, entry, verify},
        params::BlasBenchParams,
    },
    timed,
};

/// The operand's layout: `d` entries at stride `S / d` on each side, one ciphertext, dense.
fn layout(bp: &BlasBenchParams) -> BlasMatVecOperandLayout {
    let (slots, d) = (bp.slots(), bp.d());
    let shape = BlasVectorShape::new(d, d, 0, slots / d);
    let meta = CoeffsMeta::from_delta_budget(bp.p, HEADROOM);
    BlasMatVecOperandLayout::new(slots, shape, slots, shape, slots, base2k(bp), meta)
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
    BE: Backend<Ring = Standard> + 'static + BlasMatVecImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    timed::<BE, M, _>(bencher, op, bp, || {
        let mut client = Client::<BE>::new(bp);
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
                    .unwrap()
            })
        } else {
            Box::new(move || {
                module
                    .blas_matvec_pt(&mut res, &streamed, &a, &atks, &mut scratch.borrow())
                    .unwrap()
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
    let bytes = module.blas_matvec_pt_tmp_bytes(&*res, w, a, &super::matmul::atk_layout(bp));
    let mut scratch = ScratchOwned::<BE>::alloc(bytes);
    module.blas_matvec_pt(res, w, a, keys, &mut scratch.borrow()).unwrap();
    scratch
}

/// `blas_matvec_pt` by the prepared operand.
pub fn runner_blas_matvec_pt_resident<BE, M>(bencher: &mut Bencher<'_, M>, bp: &BlasBenchParams)
where
    BE: Backend<Ring = Standard> + 'static + BlasMatVecImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    matvec::<BE, M>(bencher, bp, "blas_matvec_pt_resident", true);
}

/// `blas_matvec_pt` streaming the operand's encoded diagonals.
pub fn runner_blas_matvec_pt_streamed<BE, M>(bencher: &mut Bencher<'_, M>, bp: &BlasBenchParams)
where
    BE: Backend<Ring = Standard> + 'static + BlasMatVecImpl + MaxBase2k,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    matvec::<BE, M>(bencher, bp, "blas_matvec_pt_streamed", false);
}
