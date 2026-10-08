//! The upstream HAL suites, with `NTT4x30Portable` as the comparison backend.

#![cfg(any(feature = "enable-metal", feature = "enable-vulkan", feature = "enable-wgpu"))]

use poulpy_hal::{backend_test_suite, cross_backend_test_suite};

mod local;

type Gpu = crate::NTT4x30Wgpu;
type GpuCI = crate::NTT4x30Wgpu<poulpy_hal::layouts::ConjugateInvariant>;

cross_backend_test_suite! {
    mod vec_znx,
    backend_ref = poulpy_cpu_portable::NTT4x30Portable,
    backend_test = super::Gpu,
    params = TestParams { size: 1 << 8, base2k: 50, n: 8 },
    tests = {
        test_vec_znx_add_matches_reference => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_matches_reference,
        test_vec_znx_add_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_assign,
        test_vec_znx_add_scalar_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_scalar_assign,
        test_vec_znx_sub => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub,
        test_vec_znx_sub_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub_assign,
        test_vec_znx_sub_negate_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub_negate_assign,
        test_vec_znx_rsh => poulpy_hal::test_suite::vec_znx::test_vec_znx_rsh,
        test_vec_znx_rsh_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_rsh_assign,
        test_vec_znx_lsh => poulpy_hal::test_suite::vec_znx::test_vec_znx_lsh,
        test_vec_znx_lsh_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_lsh_assign,
        test_vec_znx_negate => poulpy_hal::test_suite::vec_znx::test_vec_znx_negate,
        test_vec_znx_negate_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_negate_assign,
        test_vec_znx_rotate => poulpy_hal::test_suite::vec_znx::test_vec_znx_rotate,
        test_vec_znx_rotate_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_rotate_assign,
        test_vec_znx_automorphism => poulpy_hal::test_suite::vec_znx::test_vec_znx_automorphism,
        test_vec_znx_automorphism_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_automorphism_assign,
        test_vec_znx_mul_xp_minus_one => poulpy_hal::test_suite::vec_znx::test_vec_znx_mul_xp_minus_one,
        test_vec_znx_mul_xp_minus_one_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_mul_xp_minus_one_assign,
        test_vec_znx_normalize => poulpy_hal::test_suite::vec_znx::test_vec_znx_normalize,
        test_vec_znx_normalize_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_normalize_assign,
        test_vec_znx_gadget_decompose => poulpy_hal::test_suite::vec_znx::test_vec_znx_gadget_decompose,
        test_vec_znx_switch_ring => poulpy_hal::test_suite::vec_znx::test_vec_znx_switch_ring,
        test_vec_znx_copy => poulpy_hal::test_suite::vec_znx::test_vec_znx_copy,
    }
}

// Digits of several limbs at a small radix, over a wide degree.
cross_backend_test_suite! {
    mod vec_znx_gadget_decompose_small_radix,
    backend_ref = poulpy_cpu_portable::NTT4x30Portable,
    backend_test = super::Gpu,
    params = TestParams { size: 1 << 11, base2k: 12, n: 8 },
    tests = {
        test_vec_znx_gadget_decompose => poulpy_hal::test_suite::vec_znx::test_vec_znx_gadget_decompose,
    }
}

backend_test_suite! {
    mod single,
    backend = super::Gpu,
    params = TestParams { size: 1 << 8, base2k: 50, n: 8 },
    tests = {
        test_vec_znx_fill_uniform => poulpy_hal::test_suite::vec_znx::test_vec_znx_fill_uniform,
        test_transfer_padded_lengths => poulpy_hal::test_suite::transfer::test_transfer_padded_lengths,
        test_vec_znx_window_ops => poulpy_hal::test_suite::window::test_vec_znx_window_ops,
        test_vec_znx_window_normalize_ops => poulpy_hal::test_suite::window::test_vec_znx_window_normalize_ops,
        test_vec_znx_embed_add_sub => poulpy_hal::test_suite::embed::test_vec_znx_embed_add_sub,
    }
}

cross_backend_test_suite! {
    mod vec_znx_big,
    backend_ref = poulpy_cpu_portable::NTT4x30Portable,
    backend_test = super::Gpu,
    params = TestParams { size: 1 << 8, base2k: 50, n: 8 },
    tests = {
        test_vec_znx_big_add => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add,
        test_vec_znx_big_add_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add_assign,
        test_vec_znx_big_add_small => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add_small,
        test_vec_znx_big_add_small_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add_small_assign,
        test_vec_znx_big_automorphism => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_automorphism,
        test_vec_znx_big_automorphism_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_automorphism_assign,
        test_vec_znx_big_negate => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_negate,
        test_vec_znx_big_negate_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_negate_assign,
        test_vec_znx_big_normalize => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_normalize,
        test_vec_znx_big_sub => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub,
        test_vec_znx_big_sub_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_assign,
        test_vec_znx_big_sub_negate_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_negate_assign,
        test_vec_znx_big_sub_small_a => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_small_a,
        test_vec_znx_big_sub_small_b => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_small_b,
        test_vec_znx_big_sub_small_a_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_small_a_assign,
        test_vec_znx_big_sub_small_b_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_small_b_assign,
        test_vec_znx_big_from_small => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_from_small,
        test_vec_znx_big_inner_sum => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_inner_sum,
        test_vec_znx_big_col_weighted_sum => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_col_weighted_sum,
        test_vec_znx_scalar_product => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_scalar_product,
    }
}

backend_test_suite! {
    mod single_big,
    backend = super::Gpu,
    params = TestParams { size: 1 << 8, base2k: 50, n: 8 },
    tests = {
        test_vec_znx_big_window_ops => poulpy_hal::test_suite::window::test_vec_znx_big_window_ops,
        test_vec_znx_big_window_normalize => poulpy_hal::test_suite::window::test_vec_znx_big_window_normalize,
        test_vec_znx_big_embed_add_sub => poulpy_hal::test_suite::embed::test_vec_znx_big_embed_add_sub,
    }
}

cross_backend_test_suite! {
    mod vec_znx_dft,
    backend_ref = poulpy_cpu_portable::NTT4x30Portable,
    backend_test = super::Gpu,
    params = TestParams { size: 1 << 8, base2k: 50, n: 8 },
    tests = {
        test_vec_znx_dft_copy => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_copy,
        test_vec_znx_dft_add => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_add,
        test_vec_znx_dft_add_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_add_assign,
        test_vec_znx_dft_sub => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub,
        test_vec_znx_dft_sub_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub_assign,
        test_vec_znx_dft_sub_negate_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub_negate_assign,
        test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
        test_vec_znx_dft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_apply,
        test_vec_znx_dft_zero => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_zero,
        test_vec_znx_dft_automorphism => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_automorphism,
        test_vec_znx_dft_automorphism_add => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_automorphism_add,
        test_vec_znx_idft_normalize => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_normalize,
    }
}

backend_test_suite! {
    mod single_dft,
    backend = super::Gpu,
    params = TestParams { size: 1 << 8, base2k: 50, n: 8 },
    tests = {
        test_vec_znx_idft_normalize_derived => poulpy_hal::test_suite::derived::test_vec_znx_idft_normalize_derived,
        test_vec_znx_dft_automorphism_add_with_plan_derived => poulpy_hal::test_suite::derived::test_vec_znx_dft_automorphism_add_with_plan_derived,
        test_vec_znx_dft_automorphism_derived => poulpy_hal::test_suite::derived::test_vec_znx_dft_automorphism_derived,
        test_vec_znx_dft_rejects_zero_step => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_rejects_zero_step,
    }
}

// The largest transform done by a single launch, in three sub-passes.
cross_backend_test_suite! {
    mod ntt_n4096,
    backend_ref = poulpy_cpu_portable::NTT4x30Portable,
    backend_test = super::Gpu,
    params = TestParams { size: 1 << 12, base2k: 50, n: 8 },
    tests = {
        test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
        test_vec_znx_dft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_apply,
        test_vec_znx_dft_automorphism => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_automorphism,
        test_vec_znx_idft_normalize => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_normalize,
    }
}

// The smallest transform done by two launches.
cross_backend_test_suite! {
    mod ntt_n8192,
    backend_ref = poulpy_cpu_portable::NTT4x30Portable,
    backend_test = super::Gpu,
    params = TestParams { size: 1 << 13, base2k: 50, n: 8 },
    tests = {
        test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
        test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
        test_vec_znx_dft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_apply,
        test_vec_znx_dft_automorphism => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_automorphism,
        test_vec_znx_idft_normalize => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_normalize,
    }
}

// Two launches of two sub-passes each.
cross_backend_test_suite! {
    mod ntt_n65536,
    backend_ref = poulpy_cpu_portable::NTT4x30Portable,
    backend_test = super::Gpu,
    params = TestParams { size: 1 << 16, base2k: 50, n: 8 },
    tests = {
        test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
    }
}

// The largest degree: a launch of three sub-passes and one of two. The
// consuming inverse at this degree is covered by the roundtrips below.
cross_backend_test_suite! {
    mod ntt_n131072,
    backend_ref = poulpy_cpu_portable::NTT4x30Portable,
    backend_test = super::Gpu,
    params = TestParams { size: 1 << 17, base2k: 50, n: 8 },
    tests = {
        test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
    }
}

cross_backend_test_suite! {
    mod ci_vec_znx,
    backend_ref = poulpy_cpu_portable::NTT4x30CIPortable,
    backend_test = super::GpuCI,
    params = TestParams { size: 1 << 8, base2k: 50, n: 8 },
    tests = {
        test_vec_znx_add_matches_reference => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_matches_reference,
        test_vec_znx_add_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_assign,
        test_vec_znx_add_scalar_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_scalar_assign,
        test_vec_znx_sub => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub,
        test_vec_znx_sub_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub_assign,
        test_vec_znx_sub_negate_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub_negate_assign,
        test_vec_znx_rsh => poulpy_hal::test_suite::vec_znx::test_vec_znx_rsh,
        test_vec_znx_rsh_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_rsh_assign,
        test_vec_znx_lsh => poulpy_hal::test_suite::vec_znx::test_vec_znx_lsh,
        test_vec_znx_lsh_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_lsh_assign,
        test_vec_znx_negate => poulpy_hal::test_suite::vec_znx::test_vec_znx_negate,
        test_vec_znx_negate_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_negate_assign,
        test_vec_znx_automorphism => poulpy_hal::test_suite::vec_znx::test_vec_znx_automorphism,
        test_vec_znx_automorphism_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_automorphism_assign,
        test_vec_znx_normalize => poulpy_hal::test_suite::vec_znx::test_vec_znx_normalize,
        test_vec_znx_normalize_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_normalize_assign,
        test_vec_znx_switch_ring => poulpy_hal::test_suite::vec_znx::test_vec_znx_switch_ring,
        test_vec_znx_copy => poulpy_hal::test_suite::vec_znx::test_vec_znx_copy,
    }
}

cross_backend_test_suite! {
    mod ci_vec_znx_big,
    backend_ref = poulpy_cpu_portable::NTT4x30CIPortable,
    backend_test = super::GpuCI,
    params = TestParams { size: 1 << 8, base2k: 50, n: 8 },
    tests = {
        test_vec_znx_big_add => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add,
        test_vec_znx_big_add_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add_assign,
        test_vec_znx_big_add_small => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add_small,
        test_vec_znx_big_add_small_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add_small_assign,
        test_vec_znx_big_automorphism => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_automorphism,
        test_vec_znx_big_automorphism_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_automorphism_assign,
        test_vec_znx_big_negate => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_negate,
        test_vec_znx_big_negate_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_negate_assign,
        test_vec_znx_big_normalize => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_normalize,
        test_vec_znx_big_sub => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub,
        test_vec_znx_big_sub_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_assign,
        test_vec_znx_big_sub_negate_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_negate_assign,
        test_vec_znx_big_sub_small_a => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_small_a,
        test_vec_znx_big_sub_small_b => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_small_b,
        test_vec_znx_big_sub_small_a_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_small_a_assign,
        test_vec_znx_big_sub_small_b_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_small_b_assign,
        test_vec_znx_big_from_small => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_from_small,
        test_vec_znx_big_inner_sum => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_inner_sum,
        test_vec_znx_big_col_weighted_sum => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_col_weighted_sum,
        test_vec_znx_scalar_product => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_scalar_product,
    }
}

cross_backend_test_suite! {
    mod ci_vec_znx_dft,
    backend_ref = poulpy_cpu_portable::NTT4x30CIPortable,
    backend_test = super::GpuCI,
    params = TestParams { size: 1 << 8, base2k: 50, n: 8 },
    tests = {
        test_vec_znx_dft_copy => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_copy,
        test_vec_znx_dft_add => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_add,
        test_vec_znx_dft_add_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_add_assign,
        test_vec_znx_dft_sub => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub,
        test_vec_znx_dft_sub_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub_assign,
        test_vec_znx_dft_sub_negate_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub_negate_assign,
        test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
        test_vec_znx_dft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_apply,
        test_vec_znx_dft_zero => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_zero,
        test_vec_znx_dft_automorphism => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_automorphism,
        test_vec_znx_dft_automorphism_add => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_automorphism_add,
        test_vec_znx_idft_normalize => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_normalize,
    }
}

// The conjugate-invariant basis change inside a two-launch transform.
cross_backend_test_suite! {
    mod ci_ntt_n8192,
    backend_ref = poulpy_cpu_portable::NTT4x30CIPortable,
    backend_test = super::GpuCI,
    params = TestParams { size: 1 << 13, base2k: 50, n: 8 },
    tests = {
        test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
        test_vec_znx_dft_automorphism => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_automorphism,
        test_vec_znx_idft_normalize => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_normalize,
    }
}

/// Exact transform roundtrip at the limits of the `i64` range, which the
/// upstream suites (50-bit limbs) do not reach.
fn roundtrip_at_word_limits<R: poulpy_hal::layouts::Ring>(n: usize) {
    use poulpy_hal::{
        api::{ScratchOwnedAlloc, VecZnxBigAlloc, VecZnxDftAlloc, VecZnxDftApply, VecZnxIdftApply, VecZnxIdftApplyTmpBytes},
        layouts::{
            Backend, DataView, HostBytesBackend, Module, ScratchOwned, VecZnxBigToBackendMut, VecZnxDftToBackendMut,
            VecZnxDftToBackendRef, ZnxView, ZnxViewMut,
        },
        test_suite::{upload_vec_znx, vec_znx_backend_ref},
    };

    type BE<R> = crate::NTT4x30Wgpu<R>;
    let module = Module::<BE<R>>::new(n as u64);
    let mut host = Module::<HostBytesBackend>::new(n as u64).vec_znx_alloc(n, 1, 2);
    for limb in 0..2 {
        for (i, x) in host.at_mut(0, limb).iter_mut().enumerate() {
            *x = match i % 4 {
                0 => i64::MIN,
                1 => i64::MAX,
                _ => (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(limb as u32) as i64,
            };
        }
    }
    let input = upload_vec_znx::<BE<R>>(&host);
    let mut dft = module.vec_znx_dft_alloc(n, 1, 2);
    let mut big = module.vec_znx_big_alloc(n, 1, 2);
    module.vec_znx_dft_apply(1, 0, &mut dft.to_backend_mut(), 0, &vec_znx_backend_ref::<BE<R>>(&input), 0);
    let mut scratch = ScratchOwned::<BE<R>>::alloc(module.vec_znx_idft_apply_tmp_bytes());
    module.vec_znx_idft_apply(&mut big.to_backend_mut(), 0, &dft.to_backend_ref(), 0, &mut scratch.arena());
    let got: Vec<i128> = bytemuck::pod_collect_to_vec(&<BE<R> as Backend>::to_host_bytes(big.data()));
    for limb in 0..2 {
        for (i, &want) in host.at(0, limb).iter().enumerate() {
            assert_eq!(got[limb * n + i], want as i128, "roundtrip n={n} limb={limb} i={i}");
        }
    }
}

#[test]
fn roundtrip_at_word_limits_standard() {
    for n in [8, 1 << 12, 1 << 13, 1 << 14, 1 << 15, 1 << 17] {
        roundtrip_at_word_limits::<poulpy_hal::layouts::Standard>(n);
    }
}

#[test]
fn roundtrip_at_word_limits_conjugate_invariant() {
    for n in [8, 1 << 12, 1 << 13, 1 << 14, 1 << 15, 1 << 17] {
        roundtrip_at_word_limits::<poulpy_hal::layouts::ConjugateInvariant>(n);
    }
}

cross_backend_test_suite! {
    mod svp,
    backend_ref = poulpy_cpu_portable::NTT4x30Portable,
    backend_test = super::Gpu,
    params = TestParams { size: 1 << 8, base2k: 50, n: 8 },
    tests = {
        test_svp_apply_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft,
        test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
        test_svp_apply_dft_to_dft_assign => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft_assign,
    }
}

cross_backend_test_suite! {
    mod vmp,
    backend_ref = poulpy_cpu_portable::NTT4x30Portable,
    backend_test = super::Gpu,
    params = TestParams { size: 1 << 8, base2k: 50, n: 8 },
    tests = {
        test_vmp_apply_dft => poulpy_hal::test_suite::vmp::test_vmp_apply_dft,
        test_vmp_apply_dft_to_dft => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft,
        test_vmp_extract_selected_rows => poulpy_hal::test_suite::vmp::test_vmp_extract_selected_rows,
        test_vmp_apply_dft_to_dft_add => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft_add,
        test_vmp_zero => poulpy_hal::test_suite::vmp::test_vmp_zero,
    }
}

backend_test_suite! {
    mod derived,
    backend = super::Gpu,
    params = TestParams { size: 1 << 8, base2k: 50, n: 8 },
    tests = {
        test_vmp_apply_dft_derived => poulpy_hal::test_suite::derived::test_vmp_apply_dft_derived,
        test_vmp_apply_dft_to_dft_add_derived => poulpy_hal::test_suite::derived::test_vmp_apply_dft_to_dft_add_derived,
        test_vec_znx_lsh_derived => poulpy_hal::test_suite::derived::test_vec_znx_lsh_derived,
        test_vec_znx_rsh_derived => poulpy_hal::test_suite::derived::test_vec_znx_rsh_derived,
        test_vec_znx_lsh_add_derived => poulpy_hal::test_suite::derived::test_vec_znx_lsh_add_derived,
        test_vec_znx_lsh_sub_derived => poulpy_hal::test_suite::derived::test_vec_znx_lsh_sub_derived,
        test_vec_znx_rsh_add_derived => poulpy_hal::test_suite::derived::test_vec_znx_rsh_add_derived,
        test_vec_znx_rsh_sub_derived => poulpy_hal::test_suite::derived::test_vec_znx_rsh_sub_derived,
        test_vec_znx_lsh_assign_derived => poulpy_hal::test_suite::derived::test_vec_znx_lsh_assign_derived,
        test_vec_znx_rsh_assign_derived => poulpy_hal::test_suite::derived::test_vec_znx_rsh_assign_derived,
        test_vec_znx_mul_xp_minus_one_derived => poulpy_hal::test_suite::derived::test_vec_znx_mul_xp_minus_one_derived,
        test_vec_znx_mul_xp_minus_one_assign_derived => poulpy_hal::test_suite::derived::test_vec_znx_mul_xp_minus_one_assign_derived,
        test_vec_znx_fill_uniform_source_all_derived => poulpy_hal::test_suite::derived::test_vec_znx_fill_uniform_source_all_derived,
        test_vec_znx_add_scalar_assign_derived => poulpy_hal::test_suite::derived::test_vec_znx_add_scalar_assign_derived,
        test_vec_znx_big_add_small_derived => poulpy_hal::test_suite::derived::test_vec_znx_big_add_small_derived,
        test_vec_znx_big_sub_small_a_derived => poulpy_hal::test_suite::derived::test_vec_znx_big_sub_small_a_derived,
        test_vec_znx_big_sub_small_b_derived => poulpy_hal::test_suite::derived::test_vec_znx_big_sub_small_b_derived,
        test_svp_apply_dft_derived => poulpy_hal::test_suite::derived::test_svp_apply_dft_derived,
        test_cnv_prepare_self_derived => poulpy_hal::test_suite::derived::test_cnv_prepare_self_derived,
        test_cnv_apply_dft_add_derived => poulpy_hal::test_suite::derived::test_cnv_apply_dft_add_derived,
        test_cnv_apply_dft_sum_derived => poulpy_hal::test_suite::derived::test_cnv_apply_dft_sum_derived,
        test_cnv_pairwise_apply_dft_derived => poulpy_hal::test_suite::derived::test_cnv_pairwise_apply_dft_derived,
        test_cnv_by_const_apply_add_derived => poulpy_hal::test_suite::derived::test_cnv_by_const_apply_add_derived,
    }
}

backend_test_suite! {
    mod window,
    backend = super::Gpu,
    params = TestParams { size: 1 << 8, base2k: 50, n: 8 },
    tests = {
        test_vec_znx_window_rejected_by_ring_ops => poulpy_hal::test_suite::window::test_vec_znx_window_rejected_by_ring_ops,
        test_cnv_prepare_rejects_mismatched_shape => poulpy_hal::test_suite::convolution::test_cnv_prepare_rejects_mismatched_shape,
        test_vmp_apply_dft_to_dft_rejects_column_mismatch => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft_rejects_column_mismatch,
        test_convolution_embed => poulpy_hal::test_suite::convolution::test_convolution_embed,
        test_convolution_galois => poulpy_hal::test_suite::convolution::test_convolution_galois,
        test_cnv_by_const_rejects_degree_mismatch => poulpy_hal::test_suite::convolution::test_cnv_by_const_rejects_degree_mismatch,
    }
}

#[test]
fn test_convolution_direct() {
    use poulpy_hal::{
        layouts::{Backend, Module},
        test_suite::convolution::{
            test_convolution, test_convolution_add, test_convolution_by_const, test_convolution_by_const_add,
            test_convolution_pairwise, test_convolution_sum,
        },
    };
    let module = Module::<Gpu>::new(1 << 8);
    const FLOOR: usize = <Gpu as Backend>::MIN_DEGREE;
    for n in [FLOOR, module.n()] {
        test_convolution(&module, n, 50);
        test_convolution_add(&module, n, 50);
        test_convolution_by_const(&module, n, 50);
        test_convolution_by_const_add(&module, n, 50);
        test_convolution_pairwise(&module, n, 50);
        test_convolution_sum(&module, n, 50);
    }
    // Sparse right operands one and two halvings below the module degree.
    for gap in [1, 2] {
        test_convolution(&module, module.n() >> gap, 50);
        test_convolution_pairwise(&module, module.n() >> gap, 50);
        test_convolution_sum(&module, module.n() >> gap, 50);
    }
}

#[test]
fn test_transform_domain_packed_byte_sizes() {
    use poulpy_hal::layouts::{Backend, PrepareHint};
    let (n, cols, size) = (256, 3, 5);
    let packed_bytes = n * cols * size * 4 * size_of::<u32>();
    assert_eq!(<Gpu as Backend>::bytes_of_vec_znx_dft(n, cols, size), packed_bytes);
    assert_eq!(
        <Gpu as Backend>::bytes_of_cnv_pvec_left(n, cols, size, PrepareHint::Reuse),
        packed_bytes
    );
    assert_eq!(
        <Gpu as Backend>::bytes_of_cnv_pvec_right(n, cols, size, PrepareHint::Reuse),
        packed_bytes
    );
    assert_eq!(
        <Gpu as Backend>::bytes_of_svp_ppol(n, cols, PrepareHint::Reuse),
        n * cols * 4 * size_of::<u32>()
    );
    let (rows, cols_in, cols_out) = (3, 2, 4);
    assert_eq!(
        <Gpu as Backend>::bytes_of_vmp_pmat(n, rows, cols_in, cols_out, size, PrepareHint::Reuse),
        n * rows * cols_in * cols_out * size * 4 * size_of::<u32>()
    );
}

// The products of the conjugate-invariant ring, as `conjugate_invariant_test_suite!`
// of `poulpy-cpu-portable` checks them. That macro itself writes operands through
// host slices of backend allocations, which a device backend does not have.
#[test]
fn ci_products_parity() {
    use poulpy_hal::{layouts::Module, test_suite::TestParams};
    let module = Module::<GpuCI>::new(256);
    let reference = Module::<poulpy_cpu_portable::NTT4x30CIPortable>::new(256);
    for n in [8, 256] {
        let params = TestParams {
            size: 256,
            n,
            base2k: 10,
        };
        poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft(&params, &reference, &module);
        poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft_assign(&params, &reference, &module);
        poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft(&params, &reference, &module);
        poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft_add(&params, &reference, &module);
    }
}

#[test]
fn ci_embed_convolution() {
    use poulpy_hal::{layouts::Module, test_suite::TestParams};
    let module = Module::<GpuCI>::new(256);
    for n in [8, 128, 256] {
        let params = TestParams {
            size: 256,
            n,
            base2k: 10,
        };
        poulpy_hal::test_suite::convolution::test_convolution_embed(&params, &module);
        poulpy_hal::test_suite::convolution::test_convolution_galois(&params, &module);
        poulpy_hal::test_suite::embed::test_vec_znx_embed_add_sub(&params, &module);
        poulpy_hal::test_suite::embed::test_vec_znx_big_embed_add_sub(&params, &module);
    }
}

backend_test_suite! {
    mod ci_derived,
    backend = super::GpuCI,
    params = TestParams { size: 1 << 8, base2k: 50, n: 8 },
    tests = {
        test_vmp_apply_dft_derived => poulpy_hal::test_suite::derived::test_vmp_apply_dft_derived,
        test_vmp_apply_dft_to_dft_add_derived => poulpy_hal::test_suite::derived::test_vmp_apply_dft_to_dft_add_derived,
        test_vec_znx_idft_normalize_derived => poulpy_hal::test_suite::derived::test_vec_znx_idft_normalize_derived,
        test_vec_znx_dft_automorphism_add_with_plan_derived => poulpy_hal::test_suite::derived::test_vec_znx_dft_automorphism_add_with_plan_derived,
        test_vec_znx_dft_automorphism_derived => poulpy_hal::test_suite::derived::test_vec_znx_dft_automorphism_derived,
        test_svp_apply_dft_derived => poulpy_hal::test_suite::derived::test_svp_apply_dft_derived,
        test_cnv_prepare_self_derived => poulpy_hal::test_suite::derived::test_cnv_prepare_self_derived,
        test_cnv_apply_dft_add_derived => poulpy_hal::test_suite::derived::test_cnv_apply_dft_add_derived,
        test_cnv_apply_dft_sum_derived => poulpy_hal::test_suite::derived::test_cnv_apply_dft_sum_derived,
        test_cnv_pairwise_apply_dft_derived => poulpy_hal::test_suite::derived::test_cnv_pairwise_apply_dft_derived,
        test_cnv_by_const_apply_add_derived => poulpy_hal::test_suite::derived::test_cnv_by_const_apply_add_derived,
    }
}
