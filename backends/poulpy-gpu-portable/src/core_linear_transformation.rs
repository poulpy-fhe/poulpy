//! The giant-step products of a linear transformation, a batch per launch.
//!
//! The diagonals of a prepared transformation have one limb, so a giant-step
//! product is a sum of baby steps scaled slot by slot. The products of
//! several giant steps read the same baby steps: one launch per column reads
//! each baby step once and writes every product of the batch. The diagonals
//! it reads are first gathered side by side in the scratch.

use poulpy_core::{
    layouts::{
        LinearTransformationGiantStep, LinearTransformationPlan,
        prepared::{LinearTransformationBabySteps, PreparedDiagonal},
    },
    prod_dft_size,
};
use poulpy_hal::{
    api::ScratchArenaTakeBasic,
    layouts::{
        Backend, CnvPVecLToBackendRef, CnvPVecRToBackendRef, Module, Ring, ScratchArena, VecZnxDftBackendMut, check_cnv_dft_term,
        check_degree,
    },
};

use crate::{
    NTT4x30Gpu,
    hal_impl::{cnv_left_column, cnv_right_column, dft_column_mut},
    ntt4x30::{DftColumn, MANY_LEFTS, MANY_PRODUCTS, MANY_TERMS, Schedule, gather, scale_many},
    runtime::{GpuBuf, GpuRuntime},
};

/// Giant steps whose products one launch writes.
pub(crate) const GIANT_BATCH: usize = MANY_PRODUCTS;

/// Scratch of the gathered diagonals of [`accumulate_giant_steps`] for `diagonals` diagonals.
fn gathered_bytes<RT: GpuRuntime, R: Ring>(n: usize, diagonals: usize) -> usize {
    NTT4x30Gpu::<RT, R>::scratch_aligned(NTT4x30Gpu::<RT, R>::bytes_of_vec_znx_dft(n, 1, diagonals))
}

/// Scratch [`accumulate_giant_steps`] takes beside the reference's in an
/// evaluation by `plan`: a batch of its widest giant steps gathered. A wider
/// batch of another transformation finds too little and takes the reference.
pub(crate) fn giant_steps_tmp_bytes<RT: GpuRuntime, R: Ring>(
    module: &Module<NTT4x30Gpu<RT, R>>,
    plan: &LinearTransformationPlan,
) -> usize {
    let batch = GIANT_BATCH.min(plan.index.len());
    let widest = plan
        .index
        .iter()
        .map(Vec::len)
        .filter(|&count| count <= MANY_TERMS)
        .max()
        .unwrap_or(0);
    if batch < 2 {
        return 0;
    }
    gathered_bytes::<RT, R>(module.n(), batch * widest)
}

/// `prods[g] = sum_i baby(steps[g].diagonals[i]) * diagonal_i` for the steps
/// of one batch. Returns `false`, having done nothing, for shapes the batched
/// kernel does not take or a `scratch` that cannot hold the gathered
/// diagonals: the caller then accumulates each step on its own.
#[allow(clippy::type_complexity)]
pub(crate) fn accumulate_giant_steps<RT: GpuRuntime, R: Ring>(
    module: &Module<NTT4x30Gpu<RT, R>>,
    cnv_offset_hi: usize,
    prods: &mut [VecZnxDftBackendMut<'_, NTT4x30Gpu<RT, R>>],
    lhs: &LinearTransformationBabySteps<NTT4x30Gpu<RT, R>>,
    steps: &[LinearTransformationGiantStep<PreparedDiagonal<GpuBuf<RT>, NTT4x30Gpu<RT, R>>>],
    scratch: &mut ScratchArena<'_, NTT4x30Gpu<RT, R>>,
) -> bool {
    // A batch the reference refuses takes the reference, which refuses it.
    if prods.len() != steps.len() || steps.len() < 2 || steps.len() > MANY_PRODUCTS {
        return false;
    }
    let cols = lhs.cols();
    let mut schedule = Schedule {
        products: steps.len() as u32,
        lefts: 0,
        counts: [0; MANY_PRODUCTS],
        left: [[0; MANY_TERMS]; MANY_PRODUCTS],
        right: [[0; MANY_TERMS]; MANY_PRODUCTS],
    };
    let mut babies: Vec<i64> = Vec::new();
    let mut diagonals = Vec::new();
    for (g, step) in steps.iter().enumerate() {
        if step.diagonals.is_empty() || step.diagonals.len() > MANY_TERMS {
            return false;
        }
        schedule.counts[g] = step.diagonals.len() as u32;
        for (t, diagonal) in step.diagonals.iter().enumerate() {
            let left = babies.iter().position(|&baby| baby == diagonal.baby).unwrap_or_else(|| {
                babies.push(diagonal.baby);
                babies.len() - 1
            });
            schedule.left[g][t] = left as u32;
            schedule.right[g][t] = diagonals.len() as u32;
            diagonals.push(diagonal.plaintext.cnv());
        }
    }
    if babies.len() > MANY_LEFTS {
        return false;
    }
    schedule.lefts = babies.len() as u32;
    let first = diagonals[0];
    if diagonals
        .iter()
        .any(|diagonal| diagonal.size() != 1 || diagonal.n() != first.n())
    {
        return false;
    }
    let baby = lhs.baby_step(babies[0]);
    if babies.iter().any(|&rot| {
        let other = lhs.baby_step(rot);
        other.size() != baby.size() || other.cols() != cols || other.n() != baby.n()
    }) || prods
        .iter()
        .any(|prod| prod.size() != prod_dft_size(baby.size(), 1, cnv_offset_hi))
    {
        return false;
    }
    // The checks of the reference's sums, on every term before the first launch.
    for (prod, step) in prods.iter().zip(steps) {
        check_degree::<NTT4x30Gpu<RT, R>>(module.n(), prod.n());
        for diagonal in &step.diagonals {
            let (a, b) = (
                lhs.baby_step(diagonal.baby).to_backend_ref(),
                diagonal.plaintext.cnv().to_backend_ref(),
            );
            for col in 0..cols {
                check_cnv_dft_term::<NTT4x30Gpu<RT, R>>(prod.n(), &a, col, &b, 0, 1);
            }
        }
    }

    // The products are one vector after the other in one allocation.
    let columns: Option<Vec<DftColumn>> = prods.iter().map(|prod| dft_column_mut(prod, 0)).collect();
    let Some(columns) = columns else { return false };
    let res = &columns[0];
    if prods.iter().any(|prod| prod.cols() != cols) || columns[1].base <= res.base {
        return false;
    }
    let gap = columns[1].base - res.base;
    let evenly = columns.iter().enumerate().all(|(g, column)| {
        column.binding.id == res.binding.id
            && column.base as u64 == res.base as u64 + g as u64 * gap as u64
            && column.stride == res.stride
            && column.size == res.size
            && column.n == res.n
    });
    if !evenly {
        return false;
    }

    let n = first.n();
    if scratch.available() < gathered_bytes::<RT, R>(n, diagonals.len()) {
        return false;
    }
    // The diagonals one after the other, as the limbs of a one-column vector.
    let (gathered, _) = scratch.borrow().take_vec_znx_dft_scratch(n, 1, diagonals.len());
    let rights = dft_column_mut(&gathered, 0).expect("the gathered diagonals hold storage");
    let sources: Vec<DftColumn> = diagonals
        .iter()
        .map(|diagonal| cnv_right_column(*diagonal, 0).expect("a prepared diagonal holds storage"))
        .collect();
    gather::<RT>(&rights, &sources);
    let mut lefts = Vec::with_capacity(babies.len());
    for col in 0..cols {
        let res = dft_column_mut(&prods[0], col).expect("the products hold storage");
        lefts.clear();
        lefts.extend(
            babies
                .iter()
                .map(|&rot| cnv_left_column(lhs.baby_step(rot), col).expect("a prepared baby step holds storage")),
        );
        scale_many::<RT>(&res, gap, &lefts, &rights, schedule, cnv_offset_hi);
    }
    true
}

/// `LinearTransformationImpl` as `poulpy_core::impl_linear_transformation_reference_full!`
/// registers it, every method forwarding to its reference body, with the
/// giant steps accumulated in batches by [`accumulate_giant_steps`].
macro_rules! impl_linear_transformation {
    ($be:ty) => {
        unsafe impl ::poulpy_core::oep::LinearTransformationImpl for $be {
            fn glwe_eval_linear_transformation_tmp_bytes<R, A, B, K>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &R,
                a: &A,
                pt: &B,
                plan: &::poulpy_core::layouts::LinearTransformationPlan,
                key: &K,
            ) -> usize
            where
                R: ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEInfos,
                B: ::poulpy_core::layouts::GLWEInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                ::poulpy_core::reference::linear_transformation::glwe_eval_linear_transformation_tmp_bytes_reference(
                    module, res, a, pt, plan, key,
                ) + $crate::core_linear_transformation::giant_steps_tmp_bytes(module, plan)
            }

            fn glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes<R, A, B, K>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &R,
                a: &A,
                pt: &B,
                plan: &::poulpy_core::layouts::LinearTransformationPlan,
                key: &K,
            ) -> usize
            where
                R: ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEInfos,
                B: ::poulpy_core::layouts::GLWEInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                ::poulpy_core::reference::linear_transformation::glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes_reference(
                    module, res, a, pt, plan, key,
                )
            }

            fn glwe_eval_linear_transformation_sum_tmp_bytes<R, A, B, K>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &R,
                a: &A,
                pt: &B,
                plan: &::poulpy_core::layouts::LinearTransformationPlan,
                key: &K,
            ) -> usize
            where
                R: ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEInfos,
                B: ::poulpy_core::layouts::GLWEInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                ::poulpy_core::reference::linear_transformation::glwe_eval_linear_transformation_sum_tmp_bytes_reference(
                    module, res, a, pt, plan, key,
                ) + $crate::core_linear_transformation::giant_steps_tmp_bytes(module, plan)
            }

            fn glwe_prepare_linear_transformation_baby_steps_tmp_bytes<A, K>(
                module: &::poulpy_hal::layouts::Module<$be>,
                a: &A,
                plan: &::poulpy_core::layouts::LinearTransformationPlan,
                key: &K,
            ) -> usize
            where
                A: ::poulpy_core::layouts::GLWEInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                ::poulpy_core::reference::linear_transformation::glwe_prepare_linear_transformation_baby_steps_tmp_bytes_reference(
                    module, a, plan, key,
                )
            }

            fn glwe_prepare_linear_transformation_rhs_tmp_bytes<P>(
                module: &::poulpy_hal::layouts::Module<$be>,
                pt_infos: &P,
            ) -> usize
            where
                P: ::poulpy_core::layouts::LWEInfos,
            {
                ::poulpy_core::reference::linear_transformation::glwe_prepare_linear_transformation_rhs_tmp_bytes_reference(
                    module, pt_infos,
                )
            }

            fn glwe_prepare_linear_transformation_rhs<P>(
                module: &::poulpy_hal::layouts::Module<$be>,
                prepared: &mut ::poulpy_core::layouts::LinearTransformation<
                    ::poulpy_core::layouts::prepared::PreparedDiagonal<<$be as ::poulpy_hal::layouts::Backend>::OwnedBuf, $be>,
                >,
                lt: &::poulpy_core::layouts::LinearTransformation<P>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                P: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
            {
                ::poulpy_core::reference::linear_transformation::glwe_prepare_linear_transformation_rhs_reference(
                    module, prepared, lt, scratch,
                )
            }

            fn glwe_prepare_linear_transformation_baby_steps<A, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cache: &mut ::poulpy_core::layouts::prepared::LinearTransformationBabySteps<$be>,
                a: &A,
                keys: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                A: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                H: ::poulpy_core::layouts::GetAutomorphismKey<$be>,
            {
                ::poulpy_core::reference::linear_transformation::glwe_prepare_linear_transformation_baby_steps_reference(
                    module, cache, a, keys, scratch,
                )
            }

            fn glwe_eval_linear_transformation_into<R, P, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                lhs: &::poulpy_core::layouts::prepared::LinearTransformationBabySteps<$be>,
                rhs: &::poulpy_core::layouts::LinearTransformation<P>,
                keys: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                P: ::poulpy_core::DiagonalProd<$be>,
                H: ::poulpy_core::layouts::GetAutomorphismKey<$be>,
            {
                ::poulpy_core::reference::linear_transformation::glwe_eval_linear_transformation_into_reference(
                    module, cnv_offset, res, lhs, rhs, keys, scratch,
                )
            }

            fn glwe_eval_linear_transformation_sum_into<R, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                babies: &[&::poulpy_core::layouts::prepared::LinearTransformationBabySteps<$be>],
                lts: &[&::poulpy_core::layouts::prepared::LinearTransformationPrepared<$be>],
                keys: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<$be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                H: ::poulpy_core::layouts::GetAutomorphismKey<$be>,
            {
                ::poulpy_core::reference::linear_transformation::glwe_eval_linear_transformation_sum_into_reference(
                    module, cnv_offset, res, babies, lts, keys, scratch,
                )
            }

            fn glwe_giant_step_batch(_module: &::poulpy_hal::layouts::Module<$be>) -> usize {
                $crate::core_linear_transformation::GIANT_BATCH
            }

            fn glwe_accumulate_giant_steps(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset_hi: usize,
                prods: &mut [::poulpy_hal::layouts::VecZnxDftBackendMut<'_, $be>],
                lhs: &::poulpy_core::layouts::prepared::LinearTransformationBabySteps<$be>,
                steps: &[::poulpy_core::layouts::LinearTransformationGiantStep<
                    ::poulpy_core::layouts::prepared::PreparedDiagonal<<$be as ::poulpy_hal::layouts::Backend>::OwnedBuf, $be>,
                >],
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) {
                if !$crate::core_linear_transformation::accumulate_giant_steps(module, cnv_offset_hi, prods, lhs, steps, scratch) {
                    ::poulpy_core::reference::linear_transformation::glwe_accumulate_giant_steps_reference(
                        module,
                        cnv_offset_hi,
                        prods,
                        lhs,
                        steps,
                        scratch,
                    )
                }
            }
        }
    };
}
pub(crate) use impl_linear_transformation;
