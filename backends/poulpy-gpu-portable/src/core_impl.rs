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
    gadget::{gadget_dft_apply_from_dft, gadget_dft_operand_size},
    layouts::{
        GGLWEInfos, LWEInfos, assert_gglwe_product, assert_gglwe_product_from_dft, assert_gglwe_product_idft_normalize,
        prepared::GGLWEPreparedBackendRef,
    },
    oep::{ConstTerm, GGLWEProductDigitsStridedImpl, SamplingImpl},
    reference::keyswitching::glwe::{
        gglwe_product_fused_reference, gglwe_product_fused_tmp_bytes_reference, gglwe_product_idft_normalize_reference,
        gglwe_product_idft_normalize_tmp_bytes_reference, gglwe_product_limbs, gglwe_products_from_dft_reference,
        gglwe_products_from_dft_tmp_bytes_reference,
    },
};
use poulpy_cpu_portable::kernels::noise::{gaussian_cumulative_table_portable, gaussian_rejection_constants_portable};
use poulpy_hal::layouts::{
    Backend, ConjugateInvariant, Module, Ring, ScalarZnxBackendMut, ScratchArena, VecZnxBackendMut, VecZnxBackendRef,
    VecZnxBigBackendMut, VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMatBackendRef, vec_znx_dft_backend_ref_from_mut,
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
        CONST_TERMS, Column, KERNEL_RADIX, Term, lincomb, normalize_products,
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
        assert_gglwe_product("gglwe_product_fused", res, product_size, a, key, addends);
        assert!(
            scratch.available() >= Self::gglwe_product_fused_tmp_bytes(module, res.size(), product_size, a.size(), key),
            "gglwe_product_fused: scratch below its query"
        );
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

    fn gglwe_products_from_dft_tmp_bytes<K>(
        module: &Module<Self>,
        count: usize,
        rank: usize,
        source_size: usize,
        key_infos: &K,
    ) -> usize
    where
        K: GGLWEInfos,
    {
        // The operands of every source side by side, then the work of one source.
        let operand_size = gadget_dft_operand_size(key_infos, source_size, key_infos.size());
        let operand = Self::scratch_aligned(Self::bytes_of_vec_znx_dft(module.n(), rank, operand_size));
        gglwe_products_from_dft_tmp_bytes_reference(module, count, rank, source_size, key_infos)
            + count.saturating_sub(1) * operand
    }

    // The operands of the sources laid out side by side, then their products in launches of up
    // to `MANY_SOURCES`, each reading the key once.
    fn gglwe_products_from_dft(
        module: &Module<Self>,
        p: i64,
        term_count: usize,
        res: &mut [VecZnxDftBackendMut<'_, Self>],
        output_size: usize,
        accumulate: &[bool],
        sources: &mut [VecZnxDftBackendMut<'_, Self>],
        source_sizes: &[usize],
        source_base2k: usize,
        key: &GGLWEPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) {
        use poulpy_hal::api::ScratchArenaTakeBasic;
        assert!(
            sources.len() == res.len() && source_sizes.len() == res.len() && accumulate.len() == res.len(),
            "gglwe_products_from_dft: items of different lengths"
        );
        for i in 0..res.len() {
            assert_gglwe_product_from_dft(
                "gglwe_products_from_dft",
                &res[i],
                &sources[i],
                source_sizes[i],
                source_base2k,
                key,
                output_size,
            );
        }
        let Some(first) = res.first() else { return };
        let (n, rank) = (first.n(), first.cols() - 1);
        let source_size = source_sizes[0];
        assert!(
            scratch.available() >= Self::gglwe_products_from_dft_tmp_bytes(module, res.len(), rank, source_size, key),
            "gglwe_products_from_dft: scratch below its query"
        );
        // The kernel takes a dense key, sources of one size and one accumulation for all.
        let alike = source_sizes.iter().all(|&size| size == source_size) && accumulate.iter().all(|&acc| acc == accumulate[0]);
        if key.stride() != 1 || res.len() < 2 || !alike {
            return gglwe_products_from_dft_reference(
                module,
                p,
                term_count,
                res,
                output_size,
                accumulate,
                sources,
                source_sizes,
                source_base2k,
                key,
                scratch,
            );
        }
        let output_size = output_size.min(key.size());
        let operand_size = gadget_dft_operand_size(key, source_size, output_size);
        let (mut operands, mut rest) = scratch
            .borrow()
            .take_vec_znx_dft_slice_scratch(n, res.len(), rank, operand_size);
        let mut sources: Vec<VecZnxDftBackendMut<'_, Self>> = sources
            .iter_mut()
            .map(|source| source.with_limb_range_mut(0, source_size))
            .collect();
        for (operand, source) in operands.iter_mut().zip(sources.iter_mut()) {
            gadget_dft_apply_from_dft(module, operand, source, source_base2k, key, output_size, &mut rest);
        }
        let operands: Vec<VecZnxDftBackendRef<'_, Self>> = operands
            .iter()
            .map(|operand| vec_znx_dft_backend_ref_from_mut(operand))
            .collect();
        let bodies: Vec<VecZnxDftBackendRef<'_, Self>> = sources.iter().map(vec_znx_dft_backend_ref_from_mut).collect();
        let dsize = key.dsize().as_usize();
        let product_limbs = if dsize == 1 { 1 } else { gglwe_product_limbs(key, term_count) };
        // The identity stores without a rotation.
        let plan = (p != 1).then(|| AutomorphismPlan::new(n, p, handle(module).tables().invariant()));
        let mut body_refs: Vec<&VecZnxDftBackendRef<'_, Self>> = Vec::with_capacity(crate::ntt4x30::MANY_SOURCES);
        let mut start = 0;
        while start < res.len() {
            let end = (start + crate::ntt4x30::MANY_SOURCES).min(res.len());
            body_refs.clear();
            body_refs.extend(&bodies[start..end]);
            let launched = end - start >= 2
                && gglwe_products_fused(
                    &mut res[start..end],
                    output_size,
                    &operands[start..end],
                    dsize,
                    product_limbs,
                    key.data(),
                    Some(&body_refs[..]),
                    plan.as_ref(),
                    accumulate[0],
                );
            if !launched {
                for i in start..end {
                    Self::gglwe_product_fused(
                        module,
                        &mut res[i],
                        output_size,
                        &operands[i],
                        key,
                        term_count,
                        &[Some((&bodies[i], 0))],
                        p,
                        accumulate[i],
                        &mut rest,
                    );
                }
            }
            start = end;
        }
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

    // At rank 1 and the kernel's radices, the product runs the first inverse launch of its
    // result; each column then takes the remaining launches and the normalization.
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
        assert_gglwe_product_idft_normalize("gglwe_product_idft_normalize", res, res_dft, a, key, addends, body);
        assert!(
            scratch.available()
                >= Self::gglwe_product_idft_normalize_tmp_bytes(module, res.size(), res_dft.size(), a.size(), key),
            "gglwe_product_idft_normalize: scratch below its query"
        );
        let dsize = key.dsize().as_usize();
        let product_limbs = if dsize == 1 { 1 } else { gglwe_product_limbs(key, term_count) };
        let addend = |column: usize| addends.get(column).copied().flatten();
        let fused = key.stride() == 1
            && a.size() != 0
            && res_dft.cols() == 2
            && res_base2k <= KERNEL_RADIX
            && a_base2k <= KERNEL_RADIX
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
            res_dft.len() == res.len()
                && a.len() == res.len()
                && (addends.is_empty() || addends.len() == res.len())
                && body.len() == res.len(),
            "gglwe_products_idft_normalize: items of different lengths"
        );
        let addend = |i: usize| addends.get(i).copied().unwrap_or(&[]);
        let mut need = 0;
        for i in 0..res.len() {
            assert_gglwe_product_idft_normalize(
                "gglwe_products_idft_normalize",
                &res[i],
                &res_dft[i],
                &a[i],
                key,
                addend(i),
                body[i],
            );
            need = need.max(Self::gglwe_product_idft_normalize_tmp_bytes(
                module,
                res[i].size(),
                res_dft[i].size(),
                a[i].size(),
                key,
            ));
        }
        assert!(
            scratch.available() >= need,
            "gglwe_products_idft_normalize: scratch below its query"
        );
        let dsize = key.dsize().as_usize();
        let product_limbs = if dsize == 1 { 1 } else { gglwe_product_limbs(key, term_count) };
        let plain = key.stride() == 1
            && res_base2k <= KERNEL_RADIX
            && a_base2k <= KERNEL_RADIX
            && res_dft.iter().all(|res_dft| res_dft.cols() == 2)
            && addends.iter().all(|item| item.iter().all(Option::is_none));
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
                        addend(i),
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

/// Scratch of [`shifted_const_combination`]: every constant of a combination
/// shifted at once.
pub(crate) fn const_combination_shift_tmp_bytes<RT: GpuRuntime, R: Ring>(
    module: &Module<NTT4x30Gpu<RT, R>>,
    b_size: usize,
) -> usize {
    use poulpy_hal::{api::VecZnxRshTmpBytes, layouts::Backend};
    NTT4x30Gpu::<RT, R>::scratch_aligned(NTT4x30Gpu::<RT, R>::bytes_of_vec_znx(1, CONST_TERMS, b_size + 1))
        + module.vec_znx_rsh_tmp_bytes(b_size + 1)
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
        return const_combination(res, base2k, res_k, acc_size, terms);
    }
    if terms.len() > CONST_TERMS || base2k > KERNEL_RADIX || terms.iter().any(|term| term.shift >= base2k) {
        return false;
    }
    let b_size = terms.iter().map(|term| term.b.size()).max().unwrap_or(0);
    if scratch.available() < const_combination_shift_tmp_bytes(module, b_size) {
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
    const_combination(res, base2k, res_k, acc_size, &terms)
}

/// `GLWEMulConstImpl::glwe_const_combination` with the terms of a column
/// summed inside the launch that normalizes them. Returns `false`, having done
/// nothing, for more terms than one launch takes, a radix the kernel does not
/// handle, operands without storage or a destination that is one of them.
pub(crate) fn const_combination<RT: GpuRuntime, R: Ring>(
    res: &mut VecZnxBackendMut<'_, NTT4x30Gpu<RT, R>>,
    base2k: usize,
    res_k: usize,
    acc_size: usize,
    terms: &[ConstTerm<'_, NTT4x30Gpu<RT, R>>],
) -> bool {
    if terms.is_empty() || terms.len() > CONST_TERMS || acc_size == 0 || base2k > KERNEL_RADIX {
        return false;
    }
    let n = res.n();
    let complete = terms.iter().all(|term| {
        term.a.n() == n
            && term.a.cols() >= res.cols()
            && column_ref(&term.a, 0).is_some()
            && column_ref(&term.b, term.b_col).is_some()
    });
    let Some(res0) = column_mut(res, 0).filter(|_| complete) else {
        return false;
    };
    // The normalization reads the sum as it forms it: the destination is apart from the operands.
    let apart = terms.iter().all(|term| {
        let (a, b) = (
            column_ref(&term.a, 0).expect("checked above"),
            column_ref(&term.b, term.b_col).expect("checked above"),
        );
        a.binding.id != res0.binding.id && b.binding.id != res0.binding.id
    });
    if !apart {
        return false;
    }
    for term in terms {
        poulpy_hal::layouts::assert_dense(&term.a, "cnv_by_const_apply");
        poulpy_hal::layouts::assert_dense(&term.b, "cnv_by_const_apply");
    }
    // The products of a column, refilled for each.
    let mut products: Vec<(Column, Column, usize, usize)> = Vec::with_capacity(terms.len());
    for col in 0..res.cols() {
        products.clear();
        products.extend(terms.iter().map(|term| {
            (
                column_ref(&term.a, col).expect("checked above"),
                column_ref(&term.b, term.b_col).expect("checked above"),
                term.b_coeff,
                term.cnv_offset,
            )
        }));
        let res = column_mut(res, col).expect("checked above");
        normalize_products::<RT>(&res, base2k, res_k, &products, acc_size, base2k);
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
                    .max($crate::core_impl::const_combination_shift_tmp_bytes(module, b_size))
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
            fn glwe_mul_prepared_relinearize_sum<'a, R, AP, BP, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut R,
                res_k: ::poulpy_core::layouts::TorusPrecision,
                a: impl IntoIterator<Item = &'a AP, IntoIter: ExactSizeIterator + Clone>,
                a_size: usize,
                a_k: ::poulpy_core::layouts::TorusPrecision,
                b: impl IntoIterator<Item = &'a BP, IntoIter: ExactSizeIterator + Clone>,
                b_size: usize,
                b_k: ::poulpy_core::layouts::TorusPrecision,
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEToBackendMut<$be> + ::poulpy_core::layouts::GLWEInfos,
                AP: ::poulpy_hal::layouts::CnvPVecLToBackendRef<$be> + 'a,
                BP: ::poulpy_hal::layouts::CnvPVecRToBackendRef<$be> + 'a,
                H: ::poulpy_core::layouts::GetTensorKey<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_mul_prepared_relinearize_sum_reference(
                    module, cnv_offset, res, res_k, a, a_size, a_k, b, b_size, b_k, tsk, scratch,
                )
            }

            fn glwe_mul_prepared_relinearize_sum_dft_tmp_bytes<R, B>(
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
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_mul_prepared_relinearize_sum_dft_tmp_bytes_reference(
                    module, res, a_size, b_size, tensor_k, tsk,
                )
            }

            #[allow(clippy::too_many_arguments)]
            fn glwe_mul_prepared_relinearize_sum_dft<'a, R, AP, BP, B, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                cnv_offset: usize,
                res: &mut ::poulpy_hal::layouts::VecZnxDftBackendMut<'_, $be>,
                res_infos: &R,
                shape: &::poulpy_core::reference::operations::OnePassShape,
                pairs: impl IntoIterator<Item = (&'a AP, ::poulpy_core::layouts::MulRightOperand<'a, BP, B>)>,
                a_size: usize,
                a_k: ::poulpy_core::layouts::TorusPrecision,
                b_size: usize,
                b_k: ::poulpy_core::layouts::TorusPrecision,
                summed: bool,
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: ::poulpy_core::layouts::GLWEInfos,
                AP: ::poulpy_hal::layouts::CnvPVecLToBackendRef<$be> + 'a,
                BP: ::poulpy_hal::layouts::CnvPVecRToBackendRef<$be> + 'a,
                B: ::poulpy_core::layouts::GLWEToBackendRef<$be> + ::poulpy_core::layouts::GLWEInfos + 'a,
                H: ::poulpy_core::layouts::GetTensorKey<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_mul_prepared_relinearize_sum_dft_reference(
                    module, cnv_offset, res, res_infos, shape, pairs, a_size, a_k, b_size, b_k, summed, tsk, scratch,
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
