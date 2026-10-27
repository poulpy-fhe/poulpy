//! Registration of the `poulpy-core` extension points.
//!
//! Every family forwards to its reference body, which composes HAL methods,
//! except the two products that have a fused kernel and sampling, which draws
//! on the device from the seed (`znx::sample`).

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use poulpy_core::{
    Distribution, Noise,
    oep::{GGLWEProductDigitsStridedImpl, SamplingImpl},
};
use poulpy_cpu_portable::kernels::noise::{gaussian_cumulative_table_portable, gaussian_rejection_constants_portable};
use poulpy_hal::layouts::{
    ConjugateInvariant, Module, Ring, ScalarZnxBackendMut, ScratchArena, VecZnxBackendMut, VecZnxBigBackendMut,
    VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMatBackendRef,
};

use crate::{
    NTT4x30Gpu,
    hal_impl::{big_column_mut, column_mut, gglwe_product_digits_strided, scalar_column},
    module::handle,
    runtime::{GpuBuf, GpuRuntime, access},
    znx::{
        Column, Term, lincomb,
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

    ::poulpy_core::impl_gglwe_product_step_hooks_reference!();
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
        ::poulpy_core::impl_linear_transformation_reference_full!($be);
        ::poulpy_core::impl_operations_reference_full!($be);
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
                crate::hal_impl::glwe_mul_columns(module, cnv_offset_hi, diag, cross, a, b)
            }

            fn glwe_square_prepare(
                module: &::poulpy_hal::layouts::Module<$be>,
                left: &mut ::poulpy_hal::layouts::CnvPVecLBackendMut<'_, $be>,
                right: &mut ::poulpy_hal::layouts::CnvPVecRBackendMut<'_, $be>,
                a: &::poulpy_hal::layouts::VecZnxBackendRef<'_, $be>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) {
                <::poulpy_hal::layouts::Module<$be> as ::poulpy_core::reference::operations::GLWETensoringReference<$be>>::glwe_square_prepare_reference(
                    module, left, right, a, scratch,
                )
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
