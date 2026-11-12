use poulpy_ckks::{
    oep::CKKSShapedRoundingImpl,
    shaping::{ShapingProfile, ShapingUnit},
};
use poulpy_hal::layouts::{
    Backend, ConjugateInvariant, Module, Ring, Standard, VecZnx, vec_znx_backend_mut, vec_znx_backend_ref,
};

use super::Runtime;

type BE = crate::NTT4x30Gpu<Runtime>;

fn correction<R: Ring>(module: &Module<BE>, profile: &ShapingProfile, lambda: usize, replay: bool) {
    let n = module.n();
    let values: Vec<i64> = (0..n)
        .map(|i| {
            let word = (i as u64 + 3).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(i as u32);
            (word as i64) >> (64 - lambda)
        })
        .collect();
    let mut h = vec![0; n];
    let mut gh = vec![0; n];
    profile.shaped_multiple_host(lambda, &values, &mut h, &mut gh);
    let carry: Vec<i64> = values
        .iter()
        .zip(gh)
        .map(|(&c, gh)| (c - gh) >> lambda)
        .chain(values.iter().map(|&c| -c))
        .collect();
    let low = VecZnx::<_, i64, R>::from_data(BE::from_host_bytes(bytemuck::cast_slice(&values)), n, 1, 1);
    let mut got_h = VecZnx::<_, i64, R>::from_data(BE::from_host_bytes(&vec![0xa5; n * 8]), n, 1, 1);
    let mut got_carry = VecZnx::<_, i64, R>::from_data(BE::from_host_bytes(&vec![0xa5; n * 16]), n, 1, 2);
    let mut apply = || {
        <BE as CKKSShapedRoundingImpl<R>>::ckks_shaped_round_correction(
            module,
            &mut vec_znx_backend_mut::<BE, R>(&mut got_carry),
            &mut vec_znx_backend_mut::<BE, R>(&mut got_h),
            &vec_znx_backend_ref::<BE, R>(&low),
            profile,
            lambda,
        )
    };
    apply();
    if replay {
        let mut recording = crate::Replay::<Runtime>::record(apply);
        let events = recording.host_events();
        assert_eq!((events.reads, events.writes, events.syncs, events.allocations), (0, 0, 0, 0));
        recording.replay();
        recording.check();
    }
    assert_eq!(
        BE::to_host_bytes(got_h.data()),
        bytemuck::cast_slice::<i64, u8>(&h),
        "preimage, lambda={lambda}"
    );
    assert_eq!(
        BE::to_host_bytes(got_carry.data()),
        bytemuck::cast_slice::<i64, u8>(&carry),
        "correction, lambda={lambda}"
    );
    assert_eq!(BE::to_host_bytes(low.data()), bytemuck::cast_slice::<i64, u8>(&values));
}

fn profiles<R: Ring>() {
    for n in [64, 128] {
        let module = Module::<BE>::new(n);
        for unit in ShapingUnit::ALL.into_iter().chain([ShapingUnit {
            log_r: 1,
            target_bits: 0,
            coeffs: &[1, 0, 0, 0],
        }]) {
            let profile = ShapingProfile::for_ring::<R>(&unit.layout(1).unwrap()).unwrap();
            for lambda in [1, 32.min(profile.max_pass_bits()), profile.max_pass_bits()] {
                correction::<R>(&module, &profile, lambda, false);
            }
        }
    }
}

#[test]
fn shaped_correction_matches_reference() {
    profiles::<Standard>();
    profiles::<ConjugateInvariant>();
}

#[test]
fn shaped_correction_replays_without_host_transfers() {
    let module = Module::<BE>::new(64);
    let profile = ShapingProfile::for_ring::<ConjugateInvariant>(&ShapingUnit::U16_40.layout(1).unwrap()).unwrap();
    correction::<ConjugateInvariant>(&module, &profile, 17, true);
}

/// One module serves both rings: a unit without a conjugate-invariant shift has the same
/// coefficients on both, and each ring must still read its own table, whichever fills the
/// cache first.
#[test]
fn shaped_correction_shares_one_module_across_rings() {
    for invariant_first in [false, true] {
        let module = Module::<BE>::new(64);
        for unit in ShapingUnit::ALL {
            let layout = unit.layout(1).unwrap();
            let standard = ShapingProfile::for_ring::<Standard>(&layout).unwrap();
            let invariant = ShapingProfile::for_ring::<ConjugateInvariant>(&layout).unwrap();
            if invariant_first {
                correction::<ConjugateInvariant>(&module, &invariant, invariant.max_pass_bits(), false);
            }
            correction::<Standard>(&module, &standard, standard.max_pass_bits(), false);
            correction::<ConjugateInvariant>(&module, &invariant, invariant.max_pass_bits(), false);
        }
    }
}
