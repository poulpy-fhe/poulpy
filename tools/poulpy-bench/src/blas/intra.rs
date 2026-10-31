//! Runners for the `poulpy-blas` intra operations added beside the transpose, on one `d x d` block per operand: the row
//! and column sums and the row rotation of a replicated grid, each point built and checked as in [`super::matmul`].

use criterion::{Bencher, measurement::Measurement};
use poulpy_blas::{
    api::BlasMatIntraOps,
    layouts::{
        BlasCiphertextOwned, BlasKeyUses, BlasPlan, blas_rotate_rows_key_uses, blas_sum_cols_key_uses, blas_sum_rows_key_uses,
        replicated,
    },
    oep::BlasMatIntraImpl,
};
use poulpy_hal::{
    api::{ModuleSynchronize, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostStaged, Module, ScratchOwned, Standard},
};

use super::matmul::{
    BlasBenchModule, BlasClientModule, BlasEvalModule, BlasPoint, Client, a_entry, atk_layout, b_entry, check, log_budget, verify,
};
use crate::{blas::params::BlasBenchParams, timed};
use poulpy_ckks::api::CKKSRightOperandPreparedFactory;

/// The point of `blas_sum_rows` (`rows`) or `blas_sum_cols` of `matrices` matrices, `bp.batch_on` distinct ones or one
/// encrypted from a plaintext of `d^2` slots, which proves it and sums its rows without the projection. Their entries
/// lie in `[-1/d, 1/d)`, so that the sums stay in `[-1, 1)`.
pub fn point_blas_reduction<BE, BC>(bp: &BlasBenchParams, op: &'static str, rows: bool, matrices: usize) -> BlasPoint
where
    BE: Backend + HostStaged + 'static + BlasMatIntraImpl,
    BC: Backend<Ring = BE::Ring> + HostStaged + 'static,
    Module<BE>: CKKSRightOperandPreparedFactory<BE> + BlasEvalModule<BE>,
    Module<BC>: BlasClientModule<BC>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    ScratchOwned<BC>: ScratchOwnedAlloc<BC> + ScratchOwnedBorrow<BC>,
{
    let mut client = Client::<BE, BC>::new(bp);
    let ctx = client.context(bp, BlasPlan::new(bp.log_d));
    let entry = b_entry(bp.d());
    let ct_a = client.encrypt(bp, 1, 1, matrices, &entry);
    assert!(
        replicated::<BE::Ring>(&ct_a) == (matrices == 1),
        "{op}: the grid is not the one the runner names"
    );
    let mut res = client.result(bp, 1, 1);
    // The keys of the path the grid's label selects, without the broadcast's on a replicated grid.
    let uses = if rows {
        blas_sum_rows_key_uses(&client.module, &res, &ct_a, &ctx)
    } else {
        blas_sum_cols_key_uses(&client.module, &res, &ct_a, &ctx)
    };
    let atks = client.rotation_keys(bp, elements_of(&uses));
    let atk = atk_layout(bp);
    let bytes = if rows {
        client.module.blas_sum_rows_tmp_bytes(&res, &ct_a, &ctx, &atk)
    } else {
        client.module.blas_sum_cols_tmp_bytes(&res, &ct_a, &ctx, &atk)
    };
    let mut scratch = ScratchOwned::<BE>::alloc(bytes);
    let keys = atks.len();
    let input_log_budget = log_budget::<BE>(&ct_a);
    let call = move |module: &Module<BE>, res: &mut BlasCiphertextOwned<BE>, scratch: &mut ScratchOwned<BE>| {
        if rows {
            module.blas_sum_rows(res, &ct_a, &ctx, &atks, &mut scratch.borrow()).unwrap()
        } else {
            module.blas_sum_cols(res, &ct_a, &ctx, &atks, &mut scratch.borrow()).unwrap()
        }
    };
    call(&client.module, &mut res, &mut scratch);
    let max_err = client.max_error(bp, &res, |s, r, c| {
        (0..bp.d())
            .map(|l| {
                if rows {
                    entry(s % matrices, l, c)
                } else {
                    entry(s % matrices, r, l)
                }
            })
            .sum()
    });
    let check = check::<BE>(bp, max_err, None, keys, input_log_budget, &res);
    let module = client.module;
    BlasPoint {
        call: Box::new(move || {
            call(&module, &mut res, &mut scratch);
            module.synchronize();
        }),
        check,
    }
}

fn reduction<BE, M>(bencher: &mut Bencher<'_, M>, bp: &BlasBenchParams, op: &'static str, rows: bool, matrices: usize)
where
    BE: Backend<Ring = Standard> + HostStaged + 'static + BlasMatIntraImpl,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    timed::<BE, M, _>(bencher, op, bp, || {
        point_blas_reduction::<BE, BE>(bp, op, rows, matrices).reported::<BE>(op, bp)
    });
}

/// The Galois elements of `uses`.
pub(super) fn elements_of(uses: &BlasKeyUses) -> Vec<i64> {
    uses.rotations.iter().map(|&(p, _)| p).collect()
}

/// `blas_sum_rows` of `bp.batch()` distinct matrices: needs `bp.batch() > 1`.
pub fn runner_blas_sum_rows<BE, M>(bencher: &mut Bencher<'_, M>, bp: &BlasBenchParams)
where
    BE: Backend<Ring = Standard> + HostStaged + 'static + BlasMatIntraImpl,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    reduction::<BE, M>(bencher, bp, "blas_sum_rows", true, bp.batch());
}

/// `blas_sum_rows` of one matrix in every matrix slot, at no scale.
pub fn runner_blas_sum_rows_replicated<BE, M>(bencher: &mut Bencher<'_, M>, bp: &BlasBenchParams)
where
    BE: Backend<Ring = Standard> + HostStaged + 'static + BlasMatIntraImpl,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    reduction::<BE, M>(bencher, bp, "blas_sum_rows_replicated", true, 1);
}

/// `blas_sum_cols` of `bp.batch()` matrices.
pub fn runner_blas_sum_cols<BE, M>(bencher: &mut Bencher<'_, M>, bp: &BlasBenchParams)
where
    BE: Backend<Ring = Standard> + HostStaged + 'static + BlasMatIntraImpl,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    reduction::<BE, M>(bencher, bp, "blas_sum_cols", false, bp.batch());
}

/// `blas_rotate_rows` by one row of one matrix in every matrix slot: a rotation by `d` slots, at no scale.
pub fn runner_blas_rotate_rows_replicated<BE, M>(bencher: &mut Bencher<'_, M>, bp: &BlasBenchParams)
where
    BE: Backend<Ring = Standard> + HostStaged + 'static + BlasMatIntraImpl,
    Module<BE>: BlasBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    let op = "blas_rotate_rows_replicated";
    timed::<BE, M, _>(bencher, op, bp, || {
        let mut client = Client::<BE, BE>::new(bp);
        let ctx = client.context(bp, BlasPlan::new(bp.log_d));
        let ct_a = client.encrypt(bp, 1, 1, 1, a_entry);
        assert!(replicated::<Standard>(&ct_a), "{op}: the grid does not hold one matrix");
        let mut res = client.result(bp, 1, 1);
        let uses = blas_rotate_rows_key_uses(&client.module, 1, &res, &ct_a, &ctx);
        let atks = client.rotation_keys(bp, elements_of(&uses));
        let bytes = client.module.blas_rotate_rows_tmp_bytes(&res, &ct_a, &ctx, &atk_layout(bp));
        let mut scratch = ScratchOwned::<BE>::alloc(bytes);
        client
            .module
            .blas_rotate_rows(1, &mut res, &ct_a, &ctx, &atks, &mut scratch.borrow())
            .unwrap();
        let d = bp.d();
        let max_err = client.max_error(bp, &res, |_, r, c| a_entry(0, (r + 1) % d, c));
        verify::<BE>(op, bp, max_err, None, atks.len());
        let module = client.module;
        Box::new(move || {
            module
                .blas_rotate_rows(1, &mut res, &ct_a, &ctx, &atks, &mut scratch.borrow())
                .unwrap();
            module.synchronize();
        })
    });
}
