use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use cubecl::prelude::*;
use poulpy_ckks::shaping::{ShapingProfile, ShapingRing};
use poulpy_hal::layouts::{Module, Ring, VecZnxBackendMut, VecZnxBackendRef};

use crate::{
    NTT4x30Gpu,
    hal_impl::{column_mut, column_ref},
    module::handle,
    runtime::{GpuBuf, GpuRuntime, Label, access, launch},
};

struct ShapingTables<RT: GpuRuntime>(Mutex<HashMap<Vec<i16>, Arc<GpuBuf<RT>>>>);

#[cube]
#[allow(clippy::too_many_arguments)]
fn product(a: &[u64], p: &[u64], i: u32, n: u32, a_base: u32, p_base: u32, terms: u32, #[comptime] invariant: bool) -> u64 {
    let stride = n / terms;
    let mut sum = 0u64;
    for t in 0..terms {
        let coeff = p[(p_base + t) as usize];
        let shift = t * stride;
        if invariant {
            let mut index = shift - i;
            if i >= shift {
                index = i - shift;
            }
            sum += coeff * a[(a_base + index) as usize];
            if t != 0u32 {
                let other = i + shift;
                if other < n {
                    sum += coeff * a[(a_base + other) as usize];
                } else if other > n {
                    sum -= coeff * a[(a_base + 2u32 * n - other) as usize];
                }
            }
        } else if i >= shift {
            sum += coeff * a[(a_base + i - shift) as usize];
        } else {
            sum -= coeff * a[(a_base + n + i - shift) as usize];
        }
    }
    sum
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn preimage_kernel(
    h: &mut [u64],
    low: &[u64],
    table: &[u64],
    n: u32,
    h_base: u32,
    low_base: u32,
    table_base: u32,
    terms: u32,
    lambda: u32,
    #[comptime] invariant: bool,
) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        let modulus = 1u64 << lambda as u64;
        let mask = modulus - 1u64;
        let mut value = product(low, table, i, n, low_base, table_base, terms, invariant) & mask;
        if value > mask >> 1u64 {
            value -= modulus;
        }
        h[(h_base + i) as usize] = value;
    }
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn correction_kernel(
    carry: &mut [u64],
    h: &[u64],
    low: &[u64],
    table: &[u64],
    n: u32,
    carry_base: u32,
    carry_stride: u32,
    h_base: u32,
    low_base: u32,
    table_base: u32,
    terms: u32,
    lambda: u32,
    #[comptime] invariant: bool,
) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        let c = low[(low_base + i) as usize];
        let gh = product(h, table, i, n, h_base, table_base + terms, terms, invariant);
        let difference = c - gh;
        let sign = 0u64 - (difference >> 63u64);
        carry[(carry_base + i) as usize] = (difference >> lambda as u64) | (sign << (64u64 - lambda as u64));
        carry[(carry_base + carry_stride + i) as usize] = 0u64 - c;
    }
}

unsafe impl<RT: GpuRuntime, R: Ring> poulpy_ckks::oep::CKKSShapedRoundingImpl for NTT4x30Gpu<RT, R> {
    fn ckks_shaped_round_correction_impl(
        module: &Module<Self>,
        carry: &mut VecZnxBackendMut<'_, Self>,
        h: &mut VecZnxBackendMut<'_, Self>,
        low: &VecZnxBackendRef<'_, Self>,
        profile: &ShapingProfile,
        lambda: usize,
    ) {
        assert!(low.n() == h.n() && low.n() == carry.n(), "shaping degrees differ");
        assert!(
            low.cols() == 1 && low.size() == 1 && h.cols() == 1 && h.size() == 1 && carry.cols() == 1 && carry.size() == 2,
            "invalid shaping buffers"
        );
        assert!(
            low.n().is_multiple_of(profile.period()) && lambda > 0 && lambda <= profile.max_pass_bits(),
            "invalid shaping depth or degree"
        );
        let _access = access::shared();
        let table = handle(module)
            .plan_cache()
            .with_or_create::<ShapingTables<RT>, _, _>(
                || Ok(ShapingTables(Mutex::new(HashMap::new()))),
                |tables: &ShapingTables<RT>| {
                    let mut tables = tables.0.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                    tables
                        .entry(profile.coefficients().to_vec())
                        .or_insert_with(|| {
                            let mut words = profile.inverse()[..profile.period()].to_vec();
                            words.extend(profile.coefficients()[..profile.period()].iter().map(|&c| c as i64 as u64));
                            Arc::new(GpuBuf::from_host(bytemuck::cast_slice(&words)))
                        })
                        .clone()
                },
            )
            .expect("shaping table cache");
        let table = table.view().binding().expect("shaping table storage");
        let low = column_ref(low, 0).expect("shaping input storage");
        let h = column_mut(h, 0).expect("shaping preimage storage");
        let carry = column_mut(carry, 0).expect("shaping correction storage");
        let invariant = profile.ring() == ShapingRing::ConjugateInvariant;
        let n = low.n;
        let terms = profile.period() as u32;
        launch!(
            RT, Label::ShapingPreimage, n * (profile.period() + 1) * 8, preimage_kernel, (n, RT::GROUP, RT::GROUP);
            [h.binding => u64, low.binding => u64, table => u64];
            [n as u32, h.base, low.base, table.base::<u64>(), terms, lambda as u32];
            [invariant]
        );
        launch!(
            RT, Label::ShapingCorrection, n * (profile.period() + 3) * 8, correction_kernel, (n, RT::GROUP, RT::GROUP);
            [carry.binding => u64, h.binding => u64, low.binding => u64, table => u64];
            [n as u32, carry.base, carry.stride, h.base, low.base, table.base::<u64>(), terms, lambda as u32];
            [invariant]
        );
    }
}
