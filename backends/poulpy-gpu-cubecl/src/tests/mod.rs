//! The upstream HAL suites, with `NTT4x30Portable` as the comparison backend.

#![cfg(any(feature = "enable-metal", feature = "enable-vulkan", feature = "enable-wgpu"))]

use poulpy_hal::{backend_test_suite, cross_backend_test_suite};

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
    for n in [8, 1 << 12, 1 << 13, 1 << 17] {
        roundtrip_at_word_limits::<poulpy_hal::layouts::Standard>(n);
    }
}

#[test]
fn roundtrip_at_word_limits_conjugate_invariant() {
    for n in [8, 1 << 12, 1 << 13, 1 << 17] {
        roundtrip_at_word_limits::<poulpy_hal::layouts::ConjugateInvariant>(n);
    }
}
