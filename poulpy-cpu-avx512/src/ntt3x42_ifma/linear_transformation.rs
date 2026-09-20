use crate::NTT3x42IfmaRayon as BE;
use poulpy_cpu_rayon::RayonTaskExecutor as E;
use poulpy_hal::execution::TaskExecutor;

use poulpy_core::{
    layouts::*,
    reference::{
        keyswitching::glwe::{GGLWEProductReference, gglwe_product_accumulation_output_size},
        linear_transformation::{DiagonalProd, glwe_eval_linear_transformation_into_reference},
    },
};
use poulpy_hal::{api::*, layouts::*};
impl poulpy_core::oep::LinearTransformationReference<BE> for ::poulpy_hal::layouts::Module<BE> {
    fn glwe_eval_linear_transformation_tmp_bytes_reference<R, A, B, K>(&self, res: &R, a: &A, pt: &B, key: &K) -> usize
    where
        R: poulpy_core::layouts::GLWEInfos,
        A: poulpy_core::layouts::GLWEInfos,
        B: poulpy_core::layouts::GLWEInfos,
        K: poulpy_core::layouts::GGLWEInfos,
    {
        poulpy_core::reference::linear_transformation::glwe_eval_linear_transformation_tmp_bytes_reference::<BE, _, _, _, _, _>(
            self, res, a, pt, key,
        )
    }

    fn glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes_reference<R, A, B, K>(
        &self,
        res: &R,
        a: &A,
        pt: &B,
        key: &K,
    ) -> usize
    where
        R: poulpy_core::layouts::GLWEInfos,
        A: poulpy_core::layouts::GLWEInfos,
        B: poulpy_core::layouts::GLWEInfos,
        K: poulpy_core::layouts::GGLWEInfos,
    {
        poulpy_core::reference::linear_transformation::glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes_reference::<
            BE,
            _,
            _,
            _,
            _,
            _,
        >(self, res, a, pt, key)
    }

    fn glwe_prepare_linear_transformation_baby_steps_tmp_bytes_reference<A, K>(&self, a: &A, key: &K) -> usize
    where
        A: poulpy_core::layouts::GLWEInfos,
        K: poulpy_core::layouts::GGLWEInfos,
    {
        poulpy_core::reference::linear_transformation::glwe_prepare_linear_transformation_baby_steps_tmp_bytes_reference::<
            BE,
            _,
            _,
            _,
        >(self, a, key)
    }

    fn glwe_prepare_linear_transformation_rhs_tmp_bytes_reference<P>(&self, pt_infos: &P) -> usize
    where
        P: poulpy_core::layouts::LWEInfos,
    {
        poulpy_core::reference::linear_transformation::glwe_prepare_linear_transformation_rhs_tmp_bytes_reference::<BE, _, _>(
            self, pt_infos,
        )
    }

    fn glwe_prepare_linear_transformation_rhs_reference<P>(
        &self,
        prepared: &mut poulpy_core::layouts::LinearTransformation<
            poulpy_core::layouts::prepared::PreparedDiagonal<<BE as ::poulpy_hal::layouts::Backend>::OwnedBuf, BE>,
        >,
        lt: &poulpy_core::layouts::LinearTransformation<P>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        P: poulpy_core::layouts::GLWEToBackendRef<BE> + poulpy_core::layouts::GLWEInfos,
    {
        poulpy_core::reference::linear_transformation::glwe_prepare_linear_transformation_rhs_reference::<BE, _, _>(
            self, prepared, lt, scratch,
        )
    }

    fn glwe_prepare_linear_transformation_baby_steps_reference<A, H>(
        &self,
        cache: &mut poulpy_core::layouts::prepared::LinearTransformationBabySteps<BE>,
        a: &A,
        keys: &H,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        A: poulpy_core::layouts::GLWEToBackendRef<BE> + poulpy_core::layouts::GLWEInfos,
        H: poulpy_core::layouts::GetAutomorphismKey<BE>,
    {
        poulpy_core::reference::linear_transformation::glwe_prepare_linear_transformation_baby_steps_reference::<BE, _, _, _>(
            self, cache, a, keys, scratch,
        )
    }

    fn glwe_eval_linear_transformation_into_reference<R, P, H>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        lhs: &poulpy_core::layouts::prepared::LinearTransformationBabySteps<BE>,
        rhs: &poulpy_core::layouts::LinearTransformation<P>,
        keys: &H,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<BE>,
    ) where
        R: poulpy_core::layouts::GLWEToBackendMut<BE> + poulpy_core::layouts::GLWEInfos,
        P: poulpy_core::reference::linear_transformation::DiagonalProd<BE>,
        H: poulpy_core::layouts::GetAutomorphismKey<BE>,
    {
        eval(self, cnv_offset, res, lhs, rhs, keys, scratch)
    }
}

#[allow(clippy::too_many_arguments)]
fn eval<R, P, H>(
    module: &Module<BE>,
    cnv_offset: usize,
    res: &mut R,
    lhs: &LinearTransformationBabySteps<BE>,
    rhs: &LinearTransformation<P>,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GLWEToBackendMut<BE> + GLWEInfos,
    P: DiagonalProd<BE>,
    H: GetAutomorphismKey<BE>,
{
    if !E::is_parallel() || res.size() < 16 || res.rank().as_usize() != 1 || module.n() < 8 {
        return glwe_eval_linear_transformation_into_reference(module, cnv_offset, res, lhs, rhs, keys, scratch);
    }
    let base = rhs.first_diagonal_plaintext().unwrap().base2k().as_usize();
    let cols = res.rank().as_usize() + 1;
    let (offset_hi, offset_lo) = poulpy_core::reference::operations::cnv_offset_to_limb_offset(cnv_offset, base);
    let prod_size = lhs.size() + rhs.first_diagonal_plaintext().unwrap().size() - offset_hi;
    let term_count = rhs.giant_steps.iter().filter(|g| g.rot != 0).count();
    let giant_keys: Vec<_> = rhs
        .giant_steps
        .iter()
        .map(|gs| (gs.rot != 0).then(|| keys.get_automorphism_key(module.galois_element(gs.rot), res.k()).unwrap()))
        .collect();
    if giant_keys.iter().flatten().any(|key| {
        key.base2k().as_usize() != base
            || key.base2k() != res.base2k()
            || key.stride() != 1
            || key.dsize().as_usize() < 2
            || key.n().as_usize() != module.n()
            || key.rank_in().as_usize() != 1
            || key.rank_out().as_usize() != 1
    }) {
        return glwe_eval_linear_transformation_into_reference(module, cnv_offset, res, lhs, rhs, keys, scratch);
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
        return glwe_eval_linear_transformation_into_reference(module, cnv_offset, res, lhs, rhs, keys, scratch);
    }
    let (mut prod0, arena) = scratch.borrow().take_vec_znx_dft_scratch(module.n(), cols, prod_size);
    let (mut acc0, mut work0) = arena.take_vec_znx_dft_scratch(module.n(), cols, lazy_size);
    for col in 0..cols {
        module.vec_znx_dft_zero(&mut acc0, col);
    }
    for (g, key) in giant_keys.iter().enumerate() {
        P::accumulate_giant_prod(module, offset_hi, &mut prod0, lhs, &rhs.giant_steps[g], &mut work0);
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
    let (mut a_dft, mut arena) = scratch
        .borrow()
        .take_vec_znx_dft_scratch(module.n(), acc.cols() - 1, mask_size);
    {
        let (mut small, mut work) = arena.borrow().take_vec_znx_scratch(module.n(), 1, mask_size);
        for col in 1..acc.cols() {
            module.vec_znx_idft_normalize_consume(&mut small, base, mask_size * base, 0, prod, col, base, None, &mut work);
            module.vec_znx_dft_apply(1, 0, &mut a_dft, col - 1, &small.to_backend_ref(), 0);
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
    let bytes = module.gglwe_product_dft_tmp_bytes_reference(output_size, mask_size, &key);
    let (tmp, _) = crate::hal_impl::take_host_typed::<BE, u64>(arena, bytes / 8);
    super::vmp::vmp_rotate_add::<E>(
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
    let workers = poulpy_hal::execution::scratch_workers_within::<E>(
        a.size().min(8),
        3 * n * size_of::<u64>(),
        scratch.available().saturating_sub(3 * n * size_of::<i128>()),
    );
    let (tmp, arena) = crate::hal_impl::take_host_typed::<BE, u64>(scratch.borrow(), workers * 3 * n);
    let (carry, _) = crate::hal_impl::take_host_typed::<BE, i128>(arena, 3 * n);
    let shape = a.shape();
    let mut a = VecZnxDft::<_, _, crate::NTT3x42Ifma>::from_shape(&mut **a.data_mut(), shape);
    let shape = res.shape();
    let mut res = VecZnx::from_shape(&mut **res.data_mut(), shape);
    super::vec_znx_dft::idft_normalize_consume_ifma::<E>(
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
        carry,
    );
}
