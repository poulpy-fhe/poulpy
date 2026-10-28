//! Reference compositions over poulpy-ckks, the contract every override reproduces.

mod encoding;
mod inter;
mod intra;
mod jkls;
mod matmul;

pub use encoding::BlasEncodingReference;
pub use inter::BlasMatInterReference;
pub use intra::BlasMatIntraReference;
pub(crate) use intra::{sum_line, sum_line_tmp_bytes};
pub use jkls::BlasJklsReference;
pub use matmul::BlasMatMulReference;

use poulpy_ckks::{CKKSCtBounds, api::CKKSLinearTransformationOps};
use poulpy_core::layouts::{GGLWEInfos, LinearTransformationPlan};
use poulpy_hal::layouts::Backend;

/// Scratch of the baby steps of a block of layout `a` and of one linear transformation, summed when `summed`, from
/// blocks of layout `a` into a block of layout `res`.
pub(crate) fn lt_tmp_bytes<BE, M, K>(module: &M, res: &impl CKKSCtBounds, a: &impl CKKSCtBounds, key: &K, summed: bool) -> usize
where
    BE: Backend,
    M: CKKSLinearTransformationOps<BE> + ?Sized,
    K: GGLWEInfos,
{
    // A `res` of no width is never read: its map fails the budget check before any scratch.
    let live = res.k().as_usize() > 0;
    // The maps take a key per rotation, which every plan of that kind sizes alike.
    let plan = LinearTransformationPlan::default();
    let eval = if summed {
        let res_bytes = live.then(|| module.ckks_eval_linear_transformation_sum_tmp_bytes(res, &plan, key));
        module
            .ckks_eval_linear_transformation_sum_tmp_bytes(a, &plan, key)
            .max(res_bytes.unwrap_or(0))
    } else {
        let res_bytes = live.then(|| module.ckks_eval_linear_transformation_tmp_bytes(res, &plan, key));
        module
            .ckks_eval_linear_transformation_tmp_bytes(a, &plan, key)
            .max(res_bytes.unwrap_or(0))
    };
    module
        .ckks_prepare_linear_transformation_baby_steps_tmp_bytes(a, &plan, key)
        .max(eval)
}
