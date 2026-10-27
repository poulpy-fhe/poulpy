//! Registration of the `poulpy-core` extension points.
//!
//! Every family forwards to its reference body, which composes HAL methods,
//! except the step hooks and the products that have a fused kernel, and
//! sampling, which draws on the device from the seed (`znx::sample`).

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use poulpy_core::{
    Distribution, Noise,
    layouts::{GGLWEInfos, prepared::GGLWEPreparedBackendRef},
    oep::{ConstTerm, GGLWEProductDigitsStridedImpl, SamplingImpl},
    reference::keyswitching::glwe::{
        gglwe_operand_from_dft_reference, gglwe_operand_from_dft_tmp_bytes_reference, gglwe_product_fused_reference,
        gglwe_product_fused_tmp_bytes_reference, gglwe_product_idft_normalize_reference,
        gglwe_product_idft_normalize_tmp_bytes_reference, gglwe_product_limbs,
    },
};
use poulpy_cpu_portable::kernels::noise::{gaussian_cumulative_table_portable, gaussian_rejection_constants_portable};
use poulpy_hal::layouts::{
    ConjugateInvariant, Module, Ring, ScalarZnxBackendMut, ScratchArena, VecZnxBackendMut, VecZnxBackendRef, VecZnxBigBackendMut,
    VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMatBackendRef,
};

use crate::{
    NTT4x30Gpu,
    hal_impl::{
        big_column_mut, column_mut, column_ref, gglwe_product_digits_strided, gglwe_product_fused, gglwe_products_fused,
        scalar_column,
    },
    module::handle,
    ntt4x30::AutomorphismPlan,
    runtime::{GpuBuf, GpuRuntime, access},
    znx::{
        CONST_TERMS, Column, KERNEL_RADIX, Term, convolution_by_const_sum, lincomb, normalize, normalize_products,
        sample::{Placement, add_rejection, add_table, add_uniform, fill_block, fill_fixed_weight, fill_probability},
    },
};

unsafe impl<RT: GpuRuntime, R: Ring> GGLWEProductDigitsStridedImpl for NTT4x30Gpu<RT, R> {
    fn gglwe_product_digits_strided_tmp_bytes(
        _module: &Module<Self>,
        _res_size: usize,
        _a_cols: usize,
        _a_size: usize,
        _dsize: usize,
        _pmat_rows: usize,
        _pmat_cols_in: usize,
        _pmat_cols_out: usize,
        _pmat_size: usize,
    ) -> usize {
        0
    }

    fn gglwe_product_digits_strided(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        dsize: usize,
        product_limbs: usize,
        pmat: &VmpPMatBackendRef<'_, Self>,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        gglwe_product_digits_strided(res, a, dsize, product_limbs, pmat);
    }

    fn gglwe_product_fused_tmp_bytes<K>(
        module: &Module<Self>,
        res_size: usize,
        product_size: usize,
        a_size: usize,
        key_infos: &K,
    ) -> usize
    where
        K: GGLWEInfos,
    {
        gglwe_product_fused_tmp_bytes_reference(module, res_size, product_size, a_size, key_infos)
    }

    // One launch: the product adds, rotates and accumulates as it stores.
    fn gglwe_product_fused(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        product_size: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        key: &GGLWEPreparedBackendRef<'_, Self>,
        term_count: usize,
        addends: &[Option<(&VecZnxDftBackendRef<'_, Self>, usize)>],
        p: i64,
        accumulate: bool,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        // The kernel takes a dense key, a source to multiply and an addend on each of the first two columns.
        if key.stride() != 1 || a.size() == 0 || addends.len() > 2 {
            return gglwe_product_fused_reference(module, res, product_size, a, key, term_count, addends, p, accumulate, scratch);
        }
        let dsize = key.dsize().as_usize();
        // One limb per digit is a plain vector-matrix product.
        let product_limbs = if dsize == 1 { 1 } else { gglwe_product_limbs(key, term_count) };
        // The identity stores without a rotation.
        let plan = (p != 1).then(|| AutomorphismPlan::new(res.n(), p, handle(module).tables().invariant()));
        let addend = |column: usize| addends.get(column).copied().flatten();
        gglwe_product_fused(
            res,
            product_size,
            a,
            dsize,
            product_limbs,
            key.data(),
            [addend(0), addend(1)],
            plan.as_ref(),
            accumulate,
        );
    }

    // The sources of one key in launches of up to `MANY_SOURCES`, each reading the key once.
    fn gglwe_products_fused(
        module: &Module<Self>,
        res: &mut [VecZnxDftBackendMut<'_, Self>],
        product_size: usize,
        a: &[VecZnxDftBackendRef<'_, Self>],
        key: &GGLWEPreparedBackendRef<'_, Self>,
        term_count: usize,
        addends: &[&[Option<(&VecZnxDftBackendRef<'_, Self>, usize)>]],
        p: i64,
        accumulate: &[bool],
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        assert!(
            a.len() == res.len() && addends.len() == res.len() && accumulate.len() == res.len(),
            "gglwe_products_fused: items of different lengths"
        );
        // The kernel takes a dense key, the same accumulation for all, and on output column 0
        // either an addend for every source, read from its own column 0, or none.
        let body = |item: &&[Option<(&VecZnxDftBackendRef<'_, Self>, usize)>]| match item {
            [] | [None] => Some(false),
            [Some((_, 0))] => Some(true),
            _ => None,
        };
        let bodies: Option<Vec<bool>> = addends.iter().map(body).collect();
        let shared = bodies
            .as_ref()
            .is_some_and(|bodies| bodies.iter().all(|&b| b == bodies[0]) && accumulate.iter().all(|&acc| acc == accumulate[0]));
        let dsize = key.dsize().as_usize();
        let product_limbs = if dsize == 1 { 1 } else { gglwe_product_limbs(key, term_count) };
        if res.is_empty() {
            // The reference's checks, then nothing to do.
            assert!(
                a.is_empty() && addends.is_empty() && accumulate.is_empty(),
                "gglwe_products_fused: items of different lengths"
            );
            return;
        }
        // The identity stores without a rotation.
        let plan = (p != 1).then(|| AutomorphismPlan::new(res[0].n(), p, handle(module).tables().invariant()));
        let mut start = 0;
        while start < res.len() {
            let end = (start + crate::ntt4x30::MANY_SOURCES).min(res.len());
            let launched = shared && key.stride() == 1 && end - start >= 2 && {
                let with_bodies = bodies.as_ref().unwrap()[0];
                let addend_refs: Vec<&VecZnxDftBackendRef<'_, Self>> = if with_bodies {
                    addends[start..end].iter().map(|item| item[0].unwrap().0).collect()
                } else {
                    Vec::new()
                };
                gglwe_products_fused(
                    &mut res[start..end],
                    product_size,
                    &a[start..end],
                    dsize,
                    product_limbs,
                    key.data(),
                    with_bodies.then_some(&addend_refs[..]),
                    plan.as_ref(),
                    accumulate[0],
                )
            };
            if !launched {
                for i in start..end {
                    Self::gglwe_product_fused(
                        module,
                        &mut res[i],
                        product_size,
                        &a[i],
                        key,
                        term_count,
                        addends[i],
                        p,
                        accumulate[i],
                        scratch,
                    );
                }
            }
            start = end;
        }
    }

    fn gglwe_operand_from_dft_tmp_bytes<K>(module: &Module<Self>, a_size: usize, output_size: usize, key_infos: &K) -> usize
    where
        K: GGLWEInfos,
    {
        gglwe_operand_from_dft_tmp_bytes_reference(module, a_size, output_size, key_infos)
    }

    fn gglwe_operand_from_dft(
        module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        a: &mut VecZnxDftBackendMut<'_, Self>,
        a_base2k: usize,
        key: &GGLWEPreparedBackendRef<'_, Self>,
        output_size: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        gglwe_operand_from_dft_reference(module, res, a, a_base2k, key, output_size, scratch)
    }

    fn gglwe_product_idft_normalize_tmp_bytes<K>(
        module: &Module<Self>,
        res_size: usize,
        product_size: usize,
        a_size: usize,
        key_infos: &K,
    ) -> usize
    where
        K: GGLWEInfos,
    {
        gglwe_product_idft_normalize_tmp_bytes_reference(module, res_size, product_size, a_size, key_infos)
    }

    // The product runs the first inverse launch of its result; each column then takes the
    // remaining launches and the normalization.
    fn gglwe_product_idft_normalize(
        module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_base2k: usize,
        res_k: usize,
        res_offset: i64,
        res_dft: &mut VecZnxDftBackendMut<'_, Self>,
        a: &VecZnxDftBackendRef<'_, Self>,
        key: &GGLWEPreparedBackendRef<'_, Self>,
        term_count: usize,
        addends: &[Option<(&VecZnxDftBackendRef<'_, Self>, usize)>],
        body: Option<(&VecZnxBackendRef<'_, Self>, usize)>,
        a_base2k: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        let dsize = key.dsize().as_usize();
        let product_limbs = if dsize == 1 { 1 } else { gglwe_product_limbs(key, term_count) };
        let addend = |column: usize| addends.get(column).copied().flatten();
        let fused = key.stride() == 1
            && a.size() != 0
            && addends.len() <= 2
            && crate::hal_impl::gglwe_product_digits_strided_inverse(
                module,
                res_dft,
                a,
                dsize,
                product_limbs,
                key.data(),
                [addend(0), addend(1)],
            );
        if !fused {
            return gglwe_product_idft_normalize_reference(
                module, res, res_base2k, res_k, res_offset, res_dft, a, key, term_count, addends, body, a_base2k, scratch,
            );
        }
        for col in 0..res_dft.cols() {
            crate::hal_impl::idft_tail_normalize(
                module,
                res,
                res_base2k,
                res_k,
                res_offset,
                col,
                res_dft,
                col,
                a_base2k,
                if col == 0 { body } else { None },
            );
        }
    }

    // The sources of one key in launches of up to `MANY_SOURCES` that read the key once and run
    // the first inverse launch; each column then takes the remaining launches and the
    // normalization.
    fn gglwe_products_idft_normalize(
        module: &Module<Self>,
        res: &mut [VecZnxBackendMut<'_, Self>],
        res_base2k: usize,
        res_k: usize,
        res_offset: i64,
        res_dft: &mut [VecZnxDftBackendMut<'_, Self>],
        a: &[VecZnxDftBackendRef<'_, Self>],
        key: &GGLWEPreparedBackendRef<'_, Self>,
        term_count: usize,
        addends: &[&[Option<(&VecZnxDftBackendRef<'_, Self>, usize)>]],
        body: &[Option<(&VecZnxBackendRef<'_, Self>, usize)>],
        a_base2k: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        assert!(
            res_dft.len() == res.len() && a.len() == res.len() && addends.len() == res.len() && body.len() == res.len(),
            "gglwe_products_idft_normalize: items of different lengths"
        );
        let dsize = key.dsize().as_usize();
        let product_limbs = if dsize == 1 { 1 } else { gglwe_product_limbs(key, term_count) };
        let plain = key.stride() == 1 && addends.iter().all(|item| item.iter().all(Option::is_none));
        let mut start = 0;
        while start < res.len() {
            let end = (start + crate::ntt4x30::MANY_SOURCES).min(res.len());
            let launched = plain
                && end - start >= 2
                && crate::hal_impl::gglwe_products_digits_strided_inverse(
                    module,
                    &mut res_dft[start..end],
                    &a[start..end],
                    dsize,
                    product_limbs,
                    key.data(),
                );
            if launched {
                for i in start..end {
                    for col in 0..res_dft[i].cols() {
                        crate::hal_impl::idft_tail_normalize(
                            module,
                            &mut res[i],
                            res_base2k,
                            res_k,
                            res_offset,
                            col,
                            &mut res_dft[i],
                            col,
                            a_base2k,
                            if col == 0 { body[i] } else { None },
                        );
                    }
                }
            } else {
                for i in start..end {
                    Self::gglwe_product_idft_normalize(
                        module,
                        &mut res[i],
                        res_base2k,
                        res_k,
                        res_offset,
                        &mut res_dft[i],
                        &a[i],
                        key,
                        term_count,
                        addends[i],
                        body[i],
                        a_base2k,
                        scratch,
                    );
                }
            }
            start = end;
        }
    }
}

/// Scratch of the accumulator of [`const_combination`], when the sum is not
/// normalized as it forms.
fn const_combination_acc_bytes<RT: GpuRuntime, R: Ring>(module: &Module<NTT4x30Gpu<RT, R>>, acc_size: usize) -> usize {
    use poulpy_hal::layouts::Backend;
    NTT4x30Gpu::<RT, R>::scratch_aligned(NTT4x30Gpu::<RT, R>::bytes_of_vec_znx_big(module.n(), 1, acc_size))
}

/// Scratch of [`shifted_const_combination`]: every constant of a combination
/// shifted at once, then the accumulator of [`const_combination`].
pub(crate) fn const_combination_shift_tmp_bytes<RT: GpuRuntime, R: Ring>(
    module: &Module<NTT4x30Gpu<RT, R>>,
    acc_size: usize,
    b_size: usize,
) -> usize {
    use poulpy_hal::{api::VecZnxRshTmpBytes, layouts::Backend};
    NTT4x30Gpu::<RT, R>::scratch_aligned(NTT4x30Gpu::<RT, R>::bytes_of_vec_znx(1, CONST_TERMS, b_size + 1))
        + module
            .vec_znx_rsh_tmp_bytes(b_size + 1)
            .max(const_combination_acc_bytes(module, acc_size))
}

/// [`const_combination`] of terms whose constants may need a sub-limb shift: the
/// shifted constants, laid out by `vec_znx_rsh` as the reference lays each out, are
/// carved from `scratch` side by side, one column per term. `false` when the sum
/// takes no fused launch or `scratch` cannot hold the shifted constants.
pub(crate) fn shifted_const_combination<RT: GpuRuntime, R: Ring>(
    module: &Module<NTT4x30Gpu<RT, R>>,
    res: &mut VecZnxBackendMut<'_, NTT4x30Gpu<RT, R>>,
    base2k: usize,
    res_k: usize,
    acc_size: usize,
    terms: &[ConstTerm<'_, NTT4x30Gpu<RT, R>>],
    scratch: &mut ScratchArena<'_, NTT4x30Gpu<RT, R>>,
) -> bool {
    use poulpy_hal::{
        api::{ScratchArenaTakeBasic, VecZnxRsh},
        layouts::{VecZnxToBackendMut, VecZnxToBackendRef, vec_znx_backend_ref_from_ref},
    };
    if terms.iter().all(|term| term.shift == 0) {
        return const_combination(module, res, base2k, res_k, acc_size, terms, scratch);
    }
    if terms.len() > CONST_TERMS || terms.iter().any(|term| term.shift >= base2k) {
        return false;
    }
    let b_size = terms.iter().map(|term| term.b.size()).max().unwrap_or(0);
    if scratch.available() < const_combination_shift_tmp_bytes(module, acc_size, b_size) {
        return false;
    }
    let (mut shifted, mut scratch) = scratch.borrow().take_vec_znx_scratch(1, terms.len(), b_size + 1);
    for (index, term) in terms.iter().enumerate() {
        if term.shift != 0 {
            module.vec_znx_rsh(
                base2k,
                term.shift,
                &mut shifted.to_backend_mut(),
                index,
                &vec_znx_backend_ref_from_ref::<NTT4x30Gpu<RT, R>>(&term.b).window_coeffs(term.b_coeff, 1),
                term.b_col,
                &mut scratch.borrow(),
            );
        }
    }
    let shifted = shifted.to_backend_ref();
    let terms: Vec<ConstTerm<'_, NTT4x30Gpu<RT, R>>> = terms
        .iter()
        .enumerate()
        .map(|(index, term)| {
            let (b, b_col, b_coeff) = if term.shift == 0 {
                (
                    vec_znx_backend_ref_from_ref::<NTT4x30Gpu<RT, R>>(&term.b),
                    term.b_col,
                    term.b_coeff,
                )
            } else {
                (vec_znx_backend_ref_from_ref::<NTT4x30Gpu<RT, R>>(&shifted), index, 0)
            };
            ConstTerm {
                a: vec_znx_backend_ref_from_ref::<NTT4x30Gpu<RT, R>>(&term.a),
                b,
                b_col,
                b_coeff,
                shift: 0,
                cnv_offset: term.cnv_offset,
            }
        })
        .collect();
    const_combination(module, res, base2k, res_k, acc_size, &terms, &mut scratch)
}

/// `GLWEMulConstImpl::glwe_const_combination` with the terms of a column
/// summed inside the launch that normalizes them, or into an accumulator
/// carved from `scratch` when the sum cannot be normalized as it forms.
/// Returns `false`, having done nothing, for more terms than one launch takes,
/// operands without storage or a `scratch` that cannot hold the accumulator.
pub(crate) fn const_combination<RT: GpuRuntime, R: Ring>(
    module: &Module<NTT4x30Gpu<RT, R>>,
    res: &mut VecZnxBackendMut<'_, NTT4x30Gpu<RT, R>>,
    base2k: usize,
    res_k: usize,
    acc_size: usize,
    terms: &[ConstTerm<'_, NTT4x30Gpu<RT, R>>],
    scratch: &mut ScratchArena<'_, NTT4x30Gpu<RT, R>>,
) -> bool {
    if terms.is_empty() || terms.len() > CONST_TERMS || acc_size == 0 {
        return false;
    }
    let n = res.n();
    let complete = terms.iter().all(|term| {
        term.a.n() == n
            && term.a.cols() >= res.cols()
            && column_ref(&term.a, 0).is_some()
            && column_ref(&term.b, term.b_col).is_some()
    });
    if !complete || column_mut(res, 0).is_none() {
        return false;
    }
    for term in terms {
        poulpy_hal::layouts::assert_dense(&term.a, "glwe_const_combination");
        poulpy_hal::layouts::assert_dense(&term.b, "glwe_const_combination");
    }
    let products = |col: usize| -> Vec<(Column, Column, usize, usize)> {
        terms
            .iter()
            .map(|term| {
                (
                    column_ref(&term.a, col).expect("checked above"),
                    column_ref(&term.b, term.b_col).expect("checked above"),
                    term.b_coeff,
                    term.cnv_offset,
                )
            })
            .collect()
    };
    // The normalization reads the sum as it forms it, unless the radix is one
    // the kernel leaves to the host or the destination is one of the operands.
    let res0 = column_mut(res, 0).expect("checked above");
    let apart = products(0)
        .iter()
        .all(|(a, b, _, _)| a.binding.id != res0.binding.id && b.binding.id != res0.binding.id);
    if base2k <= KERNEL_RADIX && apart {
        for col in 0..res.cols() {
            let res = column_mut(res, col).expect("checked above");
            normalize_products::<RT>(&res, base2k, res_k, &products(col), acc_size, base2k);
        }
        return true;
    }
    if scratch.available() < const_combination_acc_bytes(module, acc_size) {
        return false;
    }
    let (acc, _) = poulpy_hal::api::ScratchArenaTakeBasic::take_vec_znx_big_scratch(scratch.borrow(), n, 1, acc_size);
    let acc = big_column_mut(&acc, 0).expect("the accumulator holds storage");
    for col in 0..res.cols() {
        convolution_by_const_sum::<RT>(&acc, &products(col));
        let res = column_mut(res, col).expect("checked above");
        normalize::<RT>(&res, base2k, res_k, 0, &acc, base2k);
    }
    true
}

/// `GLWEMulConstImpl` as `poulpy_core::impl_glwe_mul_const_reference_full!`
/// registers it, with the constant combination summed in one launch per column.
macro_rules! impl_mul_const {
    ($be:ty) => {
        unsafe impl ::poulpy_core::oep::GLWEMulConstImpl for $be {
            fn glwe_mul_const_tmp_bytes<R, A, B>(module: &::poulpy_hal::layouts::Module<$be>, res: &R, a: &A, b: &B) -> usize
            where
                R: ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEInfos,
                B: ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWEMulConstReference<$be>>::glwe_mul_const_tmp_bytes_reference::<R, A, B>(module, res, a, b)
            }

            fn glwe_mul_const<R, A, B>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                a: &A,
                b: &B,
                b_coeff: usize,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                B: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWEMulConstReference<$be>>::glwe_mul_const_reference::<R, A, B>(module, cnv_offset, res, a, b, b_coeff, scratch)
            }

            fn glwe_mul_const_assign<R, B>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                b: &B,
                b_coeff: usize,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                B: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWEMulConstReference<$be>>::glwe_mul_const_assign_reference::<R, B>(module, cnv_offset, res, b, b_coeff, scratch)
            }

            fn glwe_const_combination_tmp_bytes(
                module: &::poulpy_hal::layouts::Module<$be>,
                acc_size: usize,
                a_size: usize,
                b_size: usize,
            ) -> usize
            where
                ::poulpy_hal::layouts::Module<$be>: ::poulpy_core::reference::operations::GLWEConstCombinationReference<$be>,
            {
                ::poulpy_core::reference::operations::glwe_const_combination_tmp_bytes_reference(module, acc_size, a_size, b_size)
                    .max($crate::core_impl::const_combination_shift_tmp_bytes(module, acc_size, b_size))
            }

            fn glwe_const_combination(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut ::poulpy_hal::layouts::VecZnxBackendMut<'_, $be>,
                base2k: usize,
                res_k: usize,
                acc_size: usize,
                terms: &[::poulpy_core::oep::ConstTerm<'_, $be>],
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                ::poulpy_hal::layouts::Module<$be>: ::poulpy_core::reference::operations::GLWEConstCombinationReference<$be>,
            {
                if !$crate::core_impl::shifted_const_combination(module, res, base2k, res_k, acc_size, terms, scratch) {
                    ::poulpy_core::reference::operations::glwe_const_combination_reference(
                        module, res, base2k, res_k, acc_size, terms, scratch,
                    )
                }
            }
        }
    };
}

/// The device form of a Gaussian, built once per module and sigma.
enum GaussianPlan<RT: GpuRuntime> {
    /// The cumulative table of a bound up to 64, four `u32` words per
    /// threshold, most significant first.
    Table { table: GpuBuf<RT>, entries: usize },
    /// The constants of the rejection sampler on `limbs` limbs each.
    Rejection { consts: GpuBuf<RT>, limbs: usize, v_max: u32 },
}

struct GaussianPlans<RT: GpuRuntime>(Mutex<HashMap<u64, Arc<GaussianPlan<RT>>>>);

fn gaussian_plan<RT: GpuRuntime, R: Ring>(module: &Module<NTT4x30Gpu<RT, R>>, sigma: f64) -> Arc<GaussianPlan<RT>> {
    let build = || {
        let (_, _, bound) = Noise::gaussian_parts(sigma);
        if u64::try_from(&bound).is_ok_and(|bound| bound <= 64) {
            let thresholds = gaussian_cumulative_table_portable(sigma);
            let words: Vec<u32> = thresholds
                .iter()
                .flat_map(|&t| [(t >> 96) as u32, (t >> 64) as u32, (t >> 32) as u32, t as u32])
                .collect();
            return GaussianPlan::Table {
                table: GpuBuf::from_host(bytemuck::cast_slice(&words)),
                entries: thresholds.len(),
            };
        }
        let constants = gaussian_rejection_constants_portable(sigma);
        let (t, bound) = (&constants.t, &constants.bound);
        // Every product the sampler forms, the series test's `den * j` with
        // `j < 2^32` included, and the shifted sample, fit `limbs` limbs.
        let largest = constants.largest(32).max(t << 32).max(bound << 64);
        let limbs = {
            // Bits of `largest`, from its little-endian bytes.
            let bytes = largest.to_le_bytes();
            let top = bytes
                .iter()
                .rposition(|&byte| byte != 0)
                .map_or(0, |at| 8 * at + 8 - bytes[at].leading_zeros() as usize);
            top.div_ceil(32) + 1
        };
        let v_max = u32::try_from(bound / t).expect("the Gaussian bound is a few times t");
        let words: Vec<u32> = [
            t,
            bound,
            &constants.variance_num,
            &constants.variance_den,
            &constants.acceptance_den,
        ]
        .into_iter()
        .flat_map(|value| {
            let mut bytes = value.to_le_bytes().to_vec();
            bytes.resize(4 * limbs, 0);
            bytes
                .chunks_exact(4)
                .map(|word| u32::from_le_bytes(word.try_into().expect("four bytes")))
                .collect::<Vec<_>>()
        })
        .collect();
        GaussianPlan::Rejection {
            consts: GpuBuf::from_host(bytemuck::cast_slice(&words)),
            limbs,
            v_max,
        }
    };
    // Building a plan uploads. The access is taken before the cache is
    // locked, as everywhere else: a thread that records holds the device.
    let _access = access::shared();
    handle(module)
        .plan_cache()
        .with_or_create::<GaussianPlans<RT>, _, _>(
            || Ok(GaussianPlans(Mutex::new(HashMap::new()))),
            |plans: &GaussianPlans<RT>| {
                let mut plans = plans.0.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                plans.entry(sigma.to_bits()).or_insert_with(|| Arc::new(build())).clone()
            },
        )
        .expect("the Gaussian plan cache is created infallibly")
}

/// Checks a noise placement as the portable sampler does, before anything is drawn.
#[allow(clippy::too_many_arguments)]
fn noise_placement(
    module_n: usize,
    n: usize,
    base2k: usize,
    k: usize,
    noise: Noise,
    res_size: usize,
    res_col: usize,
    res_cols: usize,
) -> Placement {
    assert!(n.is_power_of_two() && n <= module_n, "noise degree outside module");
    noise.validate();
    assert!((1..=63).contains(&base2k), "noise radix must be in 1..=63");
    assert!(k > 0, "noise precision must be positive");
    let size = k.div_ceil(base2k);
    assert!(size <= res_size, "noise precision exceeds destination allocation");
    assert!(res_col < res_cols, "noise column exceeds destination allocation");
    Placement {
        size,
        base2k,
        shift: (base2k - k % base2k) % base2k,
    }
}

/// Adds a draw of `noise` to `res`, an `i64` or `i128` column.
fn add_noise<RT: GpuRuntime, R: Ring>(
    module: &Module<NTT4x30Gpu<RT, R>>,
    res: &Column,
    place: &Placement,
    noise: Noise,
    seed: &[u8; 32],
) {
    match noise {
        Noise::Uniform { bits } => add_uniform::<RT>(res, place, bits, seed),
        Noise::Gaussian { sigma } => match &*gaussian_plan(module, sigma) {
            GaussianPlan::Table { table, entries } => {
                let binding = table.view().binding().expect("the table holds storage");
                // A sample within half a digit lands on the last limb alone.
                let first = if (*entries as u128) << place.shift < 1u128 << (place.base2k - 1) {
                    place.size - 1
                } else if place.base2k == 1 {
                    0
                } else {
                    let bound_bits = (u64::BITS - (*entries as u64).leading_zeros()) as usize;
                    place.size - place.size.min(2 + bound_bits / place.base2k)
                };
                add_table::<RT>(res, place, first, &binding, *entries, seed);
            }
            GaussianPlan::Rejection { consts, limbs, v_max } => {
                let binding = consts.view().binding().expect("the constants hold storage");
                add_rejection::<RT>(res, place, &binding, *limbs, *v_max, seed);
            }
        },
    }
}

/// The device samplers. A draw depends only on the seed, the degree, the law
/// and the placement: every coefficient has its own keystream.
unsafe impl<RT: GpuRuntime, R: Ring> SamplingImpl for NTT4x30Gpu<RT, R> {
    fn scalar_znx_fill_distribution(
        _module: &Module<Self>,
        res: &mut ScalarZnxBackendMut<'_, Self>,
        res_col: usize,
        dist: Distribution,
        seed: [u8; 32],
    ) {
        let n = res.n();
        assert!(res_col < res.cols(), "scalar_znx_fill_distribution: column out of range");
        let check_probability = |prob: f64| {
            assert!(
                (0.0..=1.0).contains(&prob),
                "scalar_znx_fill_distribution: probability outside [0, 1]"
            );
            // `floor(prob * 2^64)`, the product being exact: one is the only probability it does not fit.
            (prob < 1.0).then(|| (prob * 2f64.powi(64)) as u64)
        };
        match dist {
            Distribution::TernaryFixed(hw) | Distribution::BinaryFixed(hw) => {
                assert!(hw <= n, "scalar_znx_fill_distribution: Hamming weight above the degree")
            }
            Distribution::TernaryProb(prob) | Distribution::BinaryProb(prob) => {
                check_probability(prob);
            }
            Distribution::BinaryBlock(block) => assert!(
                block > 0 && n.is_multiple_of(block),
                "scalar_znx_fill_distribution: block size does not divide the degree"
            ),
            Distribution::ZERO => {}
            Distribution::NONE | Distribution::ENCAPSULATED(_) => {
                panic!("scalar_znx_fill_distribution: {dist:?} is not a sampleable distribution")
            }
        }
        let Some(column) = scalar_column(res, res_col) else { return };
        match dist {
            Distribution::TernaryProb(prob) => fill_probability::<RT>(&column, check_probability(prob), true, &seed),
            Distribution::BinaryProb(prob) => fill_probability::<RT>(&column, check_probability(prob), false, &seed),
            Distribution::BinaryBlock(block) => fill_block::<RT>(&column, block, &seed),
            Distribution::TernaryFixed(hw) | Distribution::BinaryFixed(hw) => {
                lincomb::<RT>(&column, 1, Term::zero(&column), Term::zero(&column));
                fill_fixed_weight::<RT>(&column, hw, matches!(dist, Distribution::TernaryFixed(_)), &seed);
            }
            _ => lincomb::<RT>(&column, 1, Term::zero(&column), Term::zero(&column)),
        }
    }

    fn vec_znx_add_noise(
        module: &Module<Self>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_col: usize,
        noise: Noise,
        seed: [u8; 32],
    ) {
        let place = noise_placement(module.n(), res.n(), base2k, k, noise, res.size(), res_col, res.cols());
        let Some(res) = column_mut(res, res_col) else { return };
        add_noise(module, &res, &place, noise, &seed);
    }

    fn vec_znx_big_add_noise(
        module: &Module<Self>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        noise: Noise,
        seed: [u8; 32],
    ) {
        let place = noise_placement(module.n(), res.n(), base2k, k, noise, res.size(), res_col, res.cols());
        let Some(res) = big_column_mut(res, res_col) else { return };
        add_noise(module, &res, &place, noise, &seed);
    }
}

/// The families registered through the upstream macros, which take a
/// concrete backend type: one invocation per runtime and ring.
macro_rules! impl_core {
    ($be:ty) => {
        ::poulpy_core::impl_automorphism_reference_full!($be);
        ::poulpy_core::impl_decompression_reference_full!($be);
        ::poulpy_core::impl_decryption_reference_full!($be);
        ::poulpy_core::impl_ggsw_conversion_reference_full!($be);
        ::poulpy_core::impl_glwe_keyswitch_reference_full!($be);
        ::poulpy_core::impl_gglwe_keyswitch_derived_full!($be);
        ::poulpy_core::impl_ggsw_keyswitch_derived_full!($be);
        ::poulpy_core::impl_lwe_keyswitch_reference_full!($be);
        ::poulpy_core::impl_encryption_reference_full!($be);
        ::poulpy_core::impl_glwe_external_product_reference_full!($be);
        ::poulpy_core::impl_gglwe_external_product_derived_full!($be);
        ::poulpy_core::impl_ggsw_external_product_derived_full!($be);
        $crate::core_linear_transformation::impl_linear_transformation!($be);
        impl_mul_const!($be);
        ::poulpy_core::impl_glwe_mul_plain_reference_full!($be);
        ::poulpy_core::impl_glwe_mul_prepared_plain_sum_reference_full!($be);
        ::poulpy_core::impl_glwe_add_reference_full!($be);
        ::poulpy_core::impl_glwe_sub_reference_full!($be);
        ::poulpy_core::impl_glwe_negate_reference_full!($be);
        ::poulpy_core::impl_glwe_zero_reference_full!($be);
        ::poulpy_core::impl_glwe_copy_reference_full!($be);
        ::poulpy_core::impl_gglwe_copy_reference_full!($be);
        ::poulpy_core::impl_ggsw_copy_reference_full!($be);
        ::poulpy_core::impl_glwe_shift_reference_full!($be);
        ::poulpy_core::impl_glwe_normalize_reference_full!($be);
        ::poulpy_core::impl_polynomial_evaluation_derived_full!($be);
        ::poulpy_core::impl_preparation_reference_full!($be);
        ::poulpy_core::impl_secret_reference_full!($be);
        ::poulpy_core::impl_glwe_trace_derived_full!($be);
    };
}

/// The families of the standard ring only.
macro_rules! impl_core_standard {
    ($be:ty) => {
        ::poulpy_core::impl_conversion_reference_full!($be);
        ::poulpy_core::impl_glwe_pack_derived_full!($be);
        ::poulpy_core::impl_glwe_rotate_reference_full!($be);
        ::poulpy_core::impl_ggsw_rotate_derived_full!($be);
        ::poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!($be);
        ::poulpy_core::impl_glwe_ci_conversion_reference_full!($be);
    };
}

/// `GLWETensoringImpl` as `poulpy_core::impl_glwe_tensoring_reference_full!` registers
/// it, every method forwarding to its reference body, except
/// `glwe_mul_columns`, which is the fused three-column product.
macro_rules! impl_tensoring {
    ($be:ty) => {
        unsafe impl ::poulpy_core::oep::GLWETensoringImpl for $be {
            fn glwe_tensor_apply_prepared_right_tmp_bytes<R, A>(
                module: &::poulpy_hal::layouts::Module<$be>, res: &R, a: &A, a_size: usize, b_size: usize,
            ) -> usize
            where
                R: ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_tensor_apply_prepared_right_tmp_bytes_reference(
                    module, res, a, a_size, b_size,
                )
            }

            fn glwe_tensor_apply_prepared_right<R, A, BP>(
                module: &::poulpy_hal::layouts::Module<$be>, cnv_offset: usize, res: &mut R, a: &A, b: &BP, b_size: usize,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                BP: ::poulpy_hal::layouts::CnvPVecRToBackendRef<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_tensor_apply_prepared_right_reference(
                    module, cnv_offset, res, a, b, b_size, scratch,
                )
            }

            fn glwe_tensor_apply_tmp_bytes<R, A, B>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &R,
                a: &A,
                b: &B,
            ) -> usize
            where
                R: ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEInfos,
                B: ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_tensor_apply_tmp_bytes_reference(
                    module, res, a, b,
                )
            }

            fn glwe_tensor_square_apply_tmp_bytes<R, A>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &R,
                a: &A,
            ) -> usize
            where
                R: ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_tensor_square_apply_tmp_bytes_reference(
                    module, res, a,
                )
            }

            fn glwe_tensor_apply<R, A, B>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                a: &A,
                b: &B,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                B: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_tensor_apply_reference(
                    module, cnv_offset, res, a, b, scratch,
                )
            }

            fn glwe_tensor_square_apply<R, A>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                a: &A,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_tensor_square_apply_reference(
                    module, cnv_offset, res, a, scratch,
                )
            }

            fn glwe_tensor_relinearize<R, A, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                a: &A,
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                H: ::poulpy_core::layouts::GetTensorKey<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_tensor_relinearize_reference(
                    module, res, a, tsk, scratch,
                )
            }

            fn glwe_tensor_relinearize_tmp_bytes<R, A, B>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &R,
                a: &A,
                tsk: &B,
            ) -> usize
            where
                R: ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEInfos,
                B: ::poulpy_core::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_tensor_relinearize_tmp_bytes_reference(
                    module, res, a, tsk,
                )
            }

            fn glwe_mul_relinearize_tmp_bytes<R, B>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &R,
                a_size: usize,
                b_size: usize,
                tensor_k: ::poulpy_core::layouts::TorusPrecision,
                tsk: &B,
            ) -> usize
            where
                R: ::poulpy_core::layouts::GLWEInfos,
                B: ::poulpy_core::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_mul_relinearize_tmp_bytes_reference(
                    module, res, a_size, b_size, tensor_k, tsk,
                )
            }

            #[allow(clippy::too_many_arguments)]
            fn glwe_mul_relinearize<R, A, B, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                res_k: ::poulpy_core::layouts::TorusPrecision,
                a: &A,
                b: &B,
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                B: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                H: ::poulpy_core::layouts::GetTensorKey<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_mul_relinearize_reference(
                    module, cnv_offset, res, res_k, a, b, tsk, scratch,
                )
            }

            fn glwe_mul_relinearize_assign<R, A, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                res_k: ::poulpy_core::layouts::TorusPrecision,
                a: &A,
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                H: ::poulpy_core::layouts::GetTensorKey<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_mul_relinearize_assign_reference(
                    module, cnv_offset, res, res_k, a, tsk, scratch,
                )
            }

            fn glwe_square_relinearize<R, A, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                res_k: ::poulpy_core::layouts::TorusPrecision,
                a: &A,
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                H: ::poulpy_core::layouts::GetTensorKey<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_square_relinearize_reference(
                    module, cnv_offset, res, res_k, a, tsk, scratch,
                )
            }

            fn glwe_square_relinearize_assign<R, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                res_k: ::poulpy_core::layouts::TorusPrecision,
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                H: ::poulpy_core::layouts::GetTensorKey<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_square_relinearize_assign_reference(
                    module, cnv_offset, res, res_k, tsk, scratch,
                )
            }

            #[allow(clippy::too_many_arguments)]
            fn glwe_mul_prepared_relinearize<R, A, BP, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                res_k: ::poulpy_core::layouts::TorusPrecision,
                a: &A,
                b: &BP,
                b_size: usize,
                b_k: ::poulpy_core::layouts::TorusPrecision,
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                BP: ::poulpy_hal::layouts::CnvPVecRToBackendRef<$be>,
                H: ::poulpy_core::layouts::GetTensorKey<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_mul_prepared_relinearize_reference(
                    module, cnv_offset, res, res_k, a, b, b_size, b_k, tsk, scratch,
                )
            }

            #[allow(clippy::too_many_arguments)]
            fn glwe_mul_prepared_relinearize_assign<R, AP, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                res_k: ::poulpy_core::layouts::TorusPrecision,
                a: &AP,
                a_size: usize,
                a_k: ::poulpy_core::layouts::TorusPrecision,
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                AP: ::poulpy_hal::layouts::CnvPVecRToBackendRef<$be>,
                H: ::poulpy_core::layouts::GetTensorKey<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_mul_prepared_relinearize_assign_reference(
                    module, cnv_offset, res, res_k, a, a_size, a_k, tsk, scratch,
                )
            }

            // Both prepared forms are the transform: a square prepares the left
            // one alone and reads it on both sides.
            fn glwe_square_prepare(
                module: &::poulpy_hal::layouts::Module<$be>,
                left: &mut ::poulpy_hal::layouts::CnvPVecLBackendMut<'_, $be>,
                right: &mut ::poulpy_hal::layouts::CnvPVecRBackendMut<'_, $be>,
                a: &::poulpy_hal::layouts::VecZnxBackendRef<'_, $be>,
                _scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) {
                crate::hal_impl::cnv_prepare_self_left(module, left, right, a);
            }

            fn glwe_square_columns(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset_hi: usize,
                diag: &mut ::poulpy_hal::layouts::VecZnxDftBackendMut<'_, $be>,
                cross: &mut ::poulpy_hal::layouts::VecZnxDftBackendMut<'_, $be>,
                left: &::poulpy_hal::layouts::CnvPVecLBackendRef<'_, $be>,
                right: &::poulpy_hal::layouts::CnvPVecRBackendRef<'_, $be>,
                _scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) {
                crate::hal_impl::glwe_mul_columns(module, cnv_offset_hi, diag, cross, left, right, true)
            }

            // The three products come from one fused launch instead of the reference's five.
            fn glwe_mul_columns(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset_hi: usize,
                diag: &mut ::poulpy_hal::layouts::VecZnxDftBackendMut<'_, $be>,
                cross: &mut ::poulpy_hal::layouts::VecZnxDftBackendMut<'_, $be>,
                a: &::poulpy_hal::layouts::CnvPVecLBackendRef<'_, $be>,
                b: &::poulpy_hal::layouts::CnvPVecRBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) {
                let _ = scratch;
                crate::hal_impl::glwe_mul_columns(module, cnv_offset_hi, diag, cross, a, b, false)
            }

            fn glwe_mul_relinearize_sum_tmp_bytes<R, B>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &R,
                a_size: usize,
                b_size: usize,
                tensor_k: ::poulpy_core::layouts::TorusPrecision,
                tsk: &B,
            ) -> usize
            where
                R: ::poulpy_core::layouts::GLWEInfos,
                B: ::poulpy_core::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_mul_relinearize_sum_tmp_bytes_reference(
                    module, res, a_size, b_size, tensor_k, tsk,
                )
            }

            fn glwe_mul_relinearize_sum<R, A, B, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                res_k: ::poulpy_core::layouts::TorusPrecision,
                a: &[&A],
                b: &[&B],
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                B: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos,
                H: ::poulpy_core::layouts::GetTensorKey<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_mul_relinearize_sum_reference(
                    module, cnv_offset, res, res_k, a, b, tsk, scratch,
                )
            }

            fn glwe_mul_prepared_relinearize_sum_tmp_bytes<R, B>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &R,
                a_size: usize,
                b_size: usize,
                tensor_k: ::poulpy_core::layouts::TorusPrecision,
                tsk: &B,
            ) -> usize
            where
                R: ::poulpy_core::layouts::GLWEInfos,
                B: ::poulpy_core::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_mul_prepared_relinearize_sum_tmp_bytes_reference(
                    module, res, a_size, b_size, tensor_k, tsk,
                )
            }

            #[allow(clippy::too_many_arguments)]
            fn glwe_mul_prepared_relinearize_sum<R, AP, BP, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                res_k: ::poulpy_core::layouts::TorusPrecision,
                a: &[&AP],
                b: &[&BP],
                a_size: usize,
                a_k: ::poulpy_core::layouts::TorusPrecision,
                b_size: usize,
                b_k: ::poulpy_core::layouts::TorusPrecision,
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                AP: ::poulpy_hal::layouts::CnvPVecLToBackendRef<$be>,
                BP: ::poulpy_hal::layouts::CnvPVecRToBackendRef<$be>,
                H: ::poulpy_core::layouts::GetTensorKey<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_mul_prepared_relinearize_sum_reference(
                    module, cnv_offset, res, res_k, a, b, a_size, a_k, b_size, b_k, tsk, scratch,
                )
            }

            fn glwe_tensor_apply_prepared_tmp_bytes<R>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &R,
                a_size: usize,
                b_size: usize,
                base2k: ::poulpy_core::layouts::Base2K,
            ) -> usize
            where
                R: ::poulpy_core::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_tensor_apply_prepared_tmp_bytes_reference(
                    module, res, a_size, b_size, base2k,
                )
            }

            #[allow(clippy::too_many_arguments)]
            fn glwe_tensor_apply_prepared<R, AP, BP>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                a: &AP,
                a_size: usize,
                b: &BP,
                b_size: usize,
                base2k: ::poulpy_core::layouts::Base2K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                AP: ::poulpy_hal::layouts::CnvPVecLToBackendRef<$be>,
                BP: ::poulpy_hal::layouts::CnvPVecRToBackendRef<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_tensor_apply_prepared_reference(
                    module, cnv_offset, res, a, a_size, b, b_size, base2k, scratch,
                )
            }
        }
    };
}

#[cfg(any(feature = "enable-metal", feature = "enable-vulkan", feature = "enable-wgpu"))]
mod wgpu {
    use super::ConjugateInvariant;
    impl_core!(crate::NTT4x30Wgpu);
    impl_tensoring!(crate::NTT4x30Wgpu);
    impl_core_standard!(crate::NTT4x30Wgpu);
    impl_core!(crate::NTT4x30Wgpu<ConjugateInvariant>);
    impl_tensoring!(crate::NTT4x30Wgpu<ConjugateInvariant>);
}

#[cfg(feature = "enable-cuda")]
mod cuda {
    use super::ConjugateInvariant;
    impl_core!(crate::NTT4x30Cuda);
    impl_tensoring!(crate::NTT4x30Cuda);
    impl_core_standard!(crate::NTT4x30Cuda);
    impl_core!(crate::NTT4x30Cuda<ConjugateInvariant>);
    impl_tensoring!(crate::NTT4x30Cuda<ConjugateInvariant>);
}

#[cfg(feature = "enable-hip")]
mod hip {
    use super::ConjugateInvariant;
    impl_core!(crate::NTT4x30Hip);
    impl_tensoring!(crate::NTT4x30Hip);
    impl_core_standard!(crate::NTT4x30Hip);
    impl_core!(crate::NTT4x30Hip<ConjugateInvariant>);
    impl_tensoring!(crate::NTT4x30Hip<ConjugateInvariant>);
}
