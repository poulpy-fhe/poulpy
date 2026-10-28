use crate::NTT3x42IfmaRayon as BE;
use poulpy_cpu_rayon::RayonTaskExecutor as E;
use poulpy_hal::execution::TaskExecutor;

use poulpy_core::{
    gadget_dft_apply, gadget_input_size, gglwe_product_accumulation_output_size,
    layouts::*,
    reference::{
        keyswitching::glwe::GGLWEProductReference,
        linear_transformation::{
            glwe_eval_linear_transformation_into_reference, glwe_eval_linear_transformation_sum_into_reference,
        },
    },
};
use poulpy_hal::{api::*, layouts::*};
unsafe impl poulpy_core::oep::LinearTransformationImpl for BE {
    fn glwe_eval_linear_transformation_tmp_bytes<R, A, B, K>(
        module: &Module<BE>,
        res: &R,
        a: &A,
        pt: &B,
        plan: &poulpy_core::layouts::LinearTransformationPlan,
        key: &K,
    ) -> usize
    where
        R: poulpy_core::layouts::GLWEInfos,
        A: poulpy_core::layouts::GLWEInfos,
        B: poulpy_core::layouts::GLWEInfos,
        K: poulpy_core::layouts::GGLWEInfos,
    {
        poulpy_core::reference::linear_transformation::glwe_eval_linear_transformation_tmp_bytes_reference(
            module, res, a, pt, plan, key,
        )
    }

    fn glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes<R, A, B, K>(
        module: &Module<BE>,
        res: &R,
        a: &A,
        pt: &B,
        plan: &poulpy_core::layouts::LinearTransformationPlan,
        key: &K,
    ) -> usize
    where
        R: poulpy_core::layouts::GLWEInfos,
        A: poulpy_core::layouts::GLWEInfos,
        B: poulpy_core::layouts::GLWEInfos,
        K: poulpy_core::layouts::GGLWEInfos,
    {
        poulpy_core::reference::linear_transformation::glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes_reference(
            module, res, a, pt, plan, key,
        )
    }

    fn glwe_eval_linear_transformation_sum_tmp_bytes<R, A, B, K>(
        module: &Module<BE>,
        res: &R,
        a: &A,
        pt: &B,
        plan: &poulpy_core::layouts::LinearTransformationPlan,
        key: &K,
    ) -> usize
    where
        R: poulpy_core::layouts::GLWEInfos,
        A: poulpy_core::layouts::GLWEInfos,
        B: poulpy_core::layouts::GLWEInfos,
        K: poulpy_core::layouts::GGLWEInfos,
    {
        poulpy_core::reference::linear_transformation::glwe_eval_linear_transformation_sum_tmp_bytes_reference(
            module, res, a, pt, plan, key,
        )
    }

    fn glwe_prepare_linear_transformation_baby_steps_tmp_bytes<A, K>(
        module: &Module<BE>,
        a: &A,
        plan: &poulpy_core::layouts::LinearTransformationPlan,
        key: &K,
    ) -> usize
    where
        A: poulpy_core::layouts::GLWEInfos,
        K: poulpy_core::layouts::GGLWEInfos,
    {
        poulpy_core::reference::linear_transformation::glwe_prepare_linear_transformation_baby_steps_tmp_bytes_reference(
            module, a, plan, key,
        )
    }

    fn glwe_prepare_linear_transformation_rhs_tmp_bytes<P>(module: &Module<BE>, pt_infos: &P) -> usize
    where
        P: poulpy_core::layouts::LWEInfos,
    {
        poulpy_core::reference::linear_transformation::glwe_prepare_linear_transformation_rhs_tmp_bytes_reference(
            module, pt_infos,
        )
    }

    fn glwe_prepare_linear_transformation_rhs<P>(
        module: &Module<BE>,
        prepared: &mut poulpy_core::layouts::LinearTransformation<
            poulpy_core::layouts::prepared::PreparedDiagonal<<BE as ::poulpy_hal::layouts::Backend>::OwnedBuf, BE>,
        >,
        lt: &poulpy_core::layouts::LinearTransformation<P>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        P: poulpy_core::layouts::GLWEToBackendRef<BE> + poulpy_core::layouts::GLWEInfos,
    {
        poulpy_core::reference::linear_transformation::glwe_prepare_linear_transformation_rhs_reference(
            module, prepared, lt, scratch,
        )
    }

    fn glwe_prepare_linear_transformation_baby_steps<A, H>(
        module: &Module<BE>,
        cache: &mut poulpy_core::layouts::prepared::LinearTransformationBabySteps<BE>,
        a: &A,
        keys: &H,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        A: poulpy_core::layouts::GLWEToBackendRef<BE> + poulpy_core::layouts::GLWEInfos,
        H: poulpy_core::layouts::GetAutomorphismKey<BE>,
    {
        poulpy_core::reference::linear_transformation::glwe_prepare_linear_transformation_baby_steps_reference(
            module, cache, a, keys, scratch,
        )
    }

    fn glwe_eval_linear_transformation_into<R, P, H>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        lhs: &poulpy_core::layouts::prepared::LinearTransformationBabySteps<BE>,
        rhs: &poulpy_core::layouts::LinearTransformation<P>,
        keys: &H,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        R: poulpy_core::layouts::GLWEToBackendMut<BE> + poulpy_core::layouts::GLWEInfos,
        P: poulpy_core::DiagonalProd<BE>,
        H: poulpy_core::layouts::GetAutomorphismKey<BE>,
    {
        assert_linear_transformation_into(lhs, rhs);
        if !eval(module, cnv_offset, res, &[lhs], &[rhs], keys, scratch) {
            glwe_eval_linear_transformation_into_reference(module, cnv_offset, res, lhs, rhs, keys, scratch)
        }
    }

    fn glwe_eval_linear_transformation_sum_into<R, H>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        babies: &[&poulpy_core::layouts::prepared::LinearTransformationBabySteps<BE>],
        lts: &[&poulpy_core::layouts::prepared::LinearTransformationPrepared<BE>],
        keys: &H,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        R: poulpy_core::layouts::GLWEToBackendMut<BE> + poulpy_core::layouts::GLWEInfos,
        H: poulpy_core::layouts::GetAutomorphismKey<BE>,
    {
        assert_linear_transformation_sum(babies, lts);
        if !eval(module, cnv_offset, res, babies, lts, keys, scratch) {
            glwe_eval_linear_transformation_sum_into_reference(module, cnv_offset, res, babies, lts, keys, scratch)
        }
    }

    fn glwe_eval_ct_linear_transformation_tmp_bytes<R, A, B, K, T>(
        module: &Module<BE>,
        res: &R,
        a: &A,
        b: &B,
        plan: &poulpy_core::layouts::LinearTransformationPlan,
        key: &K,
        tsk: &T,
    ) -> usize
    where
        R: poulpy_core::layouts::GLWEInfos,
        A: poulpy_core::layouts::GLWEInfos,
        B: poulpy_core::layouts::GLWEInfos,
        K: poulpy_core::layouts::GGLWEInfos,
        T: poulpy_core::layouts::GGLWEInfos,
    {
        poulpy_core::reference::linear_transformation::glwe_eval_ct_linear_transformation_tmp_bytes_reference(
            module, res, a, b, plan, key, tsk,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_eval_ct_linear_transformation_into<R, D, T, H>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        lhs: &poulpy_core::layouts::prepared::LinearTransformationBabySteps<BE>,
        lhs_k: poulpy_core::layouts::TorusPrecision,
        rhs: &poulpy_core::layouts::LinearTransformation<D>,
        tsk: &T,
        keys: &H,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, BE>,
    ) where
        R: poulpy_core::layouts::GLWEToBackendMut<BE> + poulpy_core::layouts::GLWEInfos,
        D: ::poulpy_hal::layouts::CnvPVecRToBackendRef<BE> + poulpy_core::layouts::GLWEInfos,
        T: poulpy_core::layouts::GetTensorKey<BE>,
        H: poulpy_core::layouts::GetAutomorphismKey<BE>,
    {
        poulpy_core::reference::linear_transformation::glwe_eval_ct_linear_transformation_into_reference(
            module, cnv_offset, res, lhs, lhs_k, rhs, tsk, keys, scratch,
        )
    }

    fn glwe_giant_step_batch(module: &Module<BE>) -> usize {
        poulpy_core::reference::linear_transformation::glwe_giant_step_batch_reference(module)
    }

    fn glwe_accumulate_giant_steps(
        module: &Module<BE>,
        cnv_offset_hi: usize,
        prods: &mut [VecZnxDftBackendMut<'_, BE>],
        lhs: &poulpy_core::layouts::prepared::LinearTransformationBabySteps<BE>,
        steps: &[poulpy_core::layouts::LinearTransformationGiantStep<
            poulpy_core::layouts::prepared::PreparedDiagonal<<BE as Backend>::OwnedBuf, BE>,
        >],
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        poulpy_core::reference::linear_transformation::glwe_accumulate_giant_steps_reference(
            module,
            cnv_offset_hi,
            prods,
            lhs,
            steps,
            scratch,
        )
    }
}

/// Lowered under test so that the parity suites, whose destinations hold a few limbs, take the parallel path.
const PARALLEL_EVAL_MIN_SIZE: usize = if cfg!(test) { 2 } else { 16 };

/// The parallel evaluation of `Σ_t M_t(a_t)` over checked terms, each giant step's PROD summed over the terms as the
/// reference sums it. Returns `false`, with `res` untouched, where it does not apply.
fn eval<R, P, H>(
    module: &Module<BE>,
    cnv_offset: usize,
    res: &mut R,
    babies: &[&LinearTransformationBabySteps<BE>],
    lts: &[&LinearTransformation<P>],
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> bool
where
    R: GLWEToBackendMut<BE> + GLWEInfos,
    P: DiagonalProd<BE>,
    H: GetAutomorphismKey<BE>,
{
    let (lhs, rhs) = (babies[0], lts[0]);
    // The fused evaluator uses one key per giant step.
    // Power-of-two schedules require the reference rotation tree.
    if rhs.rotations != BsgsRotations::PerRotation
        || !E::is_parallel()
        || res.size() < PARALLEL_EVAL_MIN_SIZE
        || res.rank().as_usize() != 1
        || module.n() < 8
        || res.n().as_usize() != module.n()
    {
        return false;
    }
    let base = rhs.first_diagonal_plaintext().unwrap().base2k().as_usize();
    let cols = res.rank().as_usize() + 1;
    let (offset_hi, offset_lo) = poulpy_core::cnv_offset_to_limb_offset(cnv_offset, base);
    let prod_size = poulpy_core::prod_dft_size(lhs.size(), rhs.first_diagonal_plaintext().unwrap().size(), offset_hi);
    let term_count = rhs.giant_steps.iter().filter(|g| g.rot != 0).count();
    let giant_keys: Vec<_> = rhs
        .giant_steps
        .iter()
        .map(|gs| {
            (gs.rot != 0).then(|| {
                keys.get_automorphism_key(module.galois_element(gs.rot), res.k())
                    .unwrap_or_else(|e| panic!("giant-step rotation {}: {e}", gs.rot))
            })
        })
        .collect();
    if giant_keys.iter().flatten().any(|key| {
        key.base2k().as_usize() != base
            || key.base2k() != res.base2k()
            || key.stride() != 1
            || key.k_offset().0 != 0
            || key.dsize().as_usize() < 2
            || key.n().as_usize() != module.n()
            || key.rank_in().as_usize() != 1
            || key.rank_out().as_usize() != 1
    }) {
        return false;
    }
    // Preserve the extra live product limb without widening the key input window.
    let widened = GLWELayout {
        n: res.n(),
        base2k: res.base2k(),
        k: (res.k().as_usize() + prod_size.saturating_sub(res.size()) * base).into(),
        rank: res.rank(),
    };
    let output_size = giant_keys
        .iter()
        .flatten()
        .map(|key| gglwe_product_accumulation_output_size::<BE, _, _, _>(&widened, res, key, term_count))
        .max()
        .unwrap_or(res.size());
    let lazy_size = output_size.max(prod_size);
    if lazy_size * cols > 128 {
        return false;
    }
    let (mut prod0, arena) = scratch.borrow().take_vec_znx_dft_scratch(module.n(), cols, prod_size);
    let (mut acc0, mut work0) = arena.take_vec_znx_dft_scratch(module.n(), cols, lazy_size);
    for col in 0..cols {
        module.vec_znx_dft_zero(&mut acc0, col);
    }
    for (g, key) in giant_keys.iter().enumerate() {
        P::accumulate_giant_prod(module, offset_hi, &mut prod0, lhs, &rhs.giant_steps[g], &mut work0);
        for (&lhs, &rhs) in babies[1..].iter().zip(&lts[1..]) {
            let (mut bucket, mut bucket_work) = work0.borrow().take_vec_znx_dft_scratch(module.n(), cols, prod_size);
            P::accumulate_giant_prod(module, offset_hi, &mut bucket, lhs, &rhs.giant_steps[g], &mut bucket_work);
            for col in 0..cols {
                module.vec_znx_dft_add_assign(&mut prod0, col, &bucket.to_backend_ref(), col);
            }
        }
        finish_giant(
            module,
            &mut acc0,
            &mut prod0,
            base,
            rhs.giant_steps[g].rot,
            key.as_ref(),
            output_size,
            term_count,
            &mut work0,
        );
    }
    let res_base = res.base2k().as_usize();
    let res_k = res.k().as_usize();
    res.set_canonical(true);
    let mut result = res.to_backend_mut();
    for col in 0..cols {
        consume(
            module,
            result.data_mut(),
            res_base,
            res_k,
            offset_lo,
            col,
            &mut acc0,
            col,
            base,
            &mut work0,
        );
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn finish_giant(
    module: &Module<BE>,
    acc: &mut VecZnxDftBackendMut<'_, BE>,
    prod: &mut VecZnxDftBackendMut<'_, BE>,
    base: usize,
    rot: i64,
    key: Option<&GLWEAutomorphismKeyPreparedBackendRef<'_, BE>>,
    output_size: usize,
    term_count: usize,
    scratch: &mut ScratchArena<'_, BE>,
) {
    let Some(key) = key else {
        let prod = VecZnxDft::from_shape(&**prod.data(), prod.shape());
        for col in 0..acc.cols() {
            module.vec_znx_dft_add_assign(acc, col, &prod, col);
        }
        return;
    };
    let output_size = output_size.min(key.size());
    let mask_size = prod.size().min(output_size);
    // Each normalized mask column is the gadget operand, at most `mask_size` full limbs.
    let (k_digit, dnum) = (key.k_digit(), key.dnum());
    let mask_k = TorusPrecision((mask_size * base) as u32);
    let operand_size = gadget_input_size(key.base2k(), k_digit, dnum, mask_k, mask_size);
    let (mut a_dft, mut arena) = scratch
        .borrow()
        .take_vec_znx_dft_scratch(module.n(), acc.cols() - 1, operand_size);
    {
        let (mut small, mut work) = arena.borrow().take_vec_znx_scratch(module.n(), 1, mask_size);
        for col in 1..acc.cols() {
            module.vec_znx_idft_normalize(&mut small, base, mask_size * base, 0, 0, prod, col, base, None, &mut work);
            gadget_dft_apply(
                module,
                &mut a_dft,
                col - 1,
                &small.to_backend_ref(),
                0,
                key.base2k(),
                k_digit,
                dnum,
                mask_k,
                &mut work.borrow(),
            );
        }
    }

    let key = GGLWEPreparedToBackendRef::<BE>::to_backend_ref(&key);
    let plan = module.vec_znx_dft_automorphism_plan(module.n(), module.galois_element(rot));
    let product_terms = module
        .n()
        .saturating_mul(key.dnum().as_usize())
        .saturating_mul(key.dsize().as_usize())
        .saturating_mul(term_count.max(1));
    let growth = if product_terms <= 1 {
        0
    } else {
        (usize::BITS - (product_terms - 1).leading_zeros()) as usize
    };
    let product_limbs = (2 * base + growth).div_ceil(base);
    let a_shape = a_dft.shape();
    let acc_shape = acc.shape();
    let body_shape = prod.shape();
    let a = VecZnxDft::<_, _, crate::NTT3x42Ifma>::from_shape(&**a_dft.data(), a_shape);
    let mut acc = VecZnxDft::<_, _, crate::NTT3x42Ifma>::from_shape(&mut **acc.data_mut(), acc_shape);
    let body = VecZnxDft::<_, _, crate::NTT3x42Ifma>::from_shape(&**prod.data(), body_shape);
    let data = key.data();
    let pmat = VmpPMat::<_, _, crate::NTT3x42Ifma>::from_data(
        &**data.data(),
        data.n(),
        data.rows(),
        data.cols_in(),
        data.cols_out(),
        data.size(),
        data.hint(),
    );
    let bytes = module.gglwe_product_dft_tmp_bytes_reference(output_size, operand_size, &key);
    let (tmp, _) = crate::hal_impl::take_host_typed::<BE, u64>(arena, bytes / 8);
    super::vmp::vmp_rotate_add::<_, E>(
        &mut acc,
        &a,
        key.dsize().as_usize(),
        product_limbs,
        &pmat,
        tmp,
        &super::vmp::RotatedOutput {
            plan: &plan,
            body,
            output_size,
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn consume(
    module: &Module<BE>,
    res: &mut VecZnxBackendMut<'_, BE>,
    res_base: usize,
    res_k: usize,
    offset: i64,
    res_col: usize,
    a: &mut VecZnxDftBackendMut<'_, BE>,
    a_col: usize,
    base: usize,
    scratch: &mut ScratchArena<'_, BE>,
) {
    let n = a.n();
    let workers =
        poulpy_hal::execution::scratch_workers_within::<E>(a.size().min(8), 3 * n * size_of::<u64>(), scratch.available());
    let (tmp, _) = crate::hal_impl::take_host_typed::<BE, u64>(scratch.borrow(), workers * 3 * n);
    let shape = a.shape();
    let mut a = VecZnxDft::<_, _, crate::NTT3x42Ifma>::from_shape(&mut **a.data_mut(), shape);
    let shape = res.shape();
    let mut res = VecZnx::from_shape(&mut **res.data_mut(), shape);
    super::vec_znx_dft::idft_normalize_ifma::<_, E>(
        module.reinterpret(),
        &mut res,
        res_base,
        res_k,
        offset,
        res_col,
        &mut a,
        a_col,
        base,
        None,
        tmp,
    );
}
