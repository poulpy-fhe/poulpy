//! Signed permutations of the coefficients: automorphisms, rotations, and the
//! maps between the conjugate-invariant ring and the standard one.

use cubecl::prelude::*;

use super::{Column, DIM};
use crate::runtime::{GpuRuntime, cubes, launched};

/// A signed permutation, as the gather each output coefficient performs.
#[derive(Clone, Copy)]
pub(crate) enum Map {
    /// `X -> X^p` on `Z[X]/(X^n + 1)`.
    Automorphism { p: i64 },
    /// Multiplication by `X^p` on `Z[X]/(X^n + 1)`.
    Rotation { p: i64 },
    /// `X -> X^p` on the conjugate-invariant ring of degree `n`.
    AutomorphismCI { p: i64 },
}

/// Inverse of the odd `p` modulo the power of two `modulus`.
fn inverse_odd(p: i64, modulus: u64) -> u32 {
    assert!(p & 1 == 1, "p must be odd, got {p}");
    let p = (p as u64) & (modulus - 1);
    // Newton iteration doubles the number of correct low bits each step.
    let mut inv = p;
    for _ in 0..6 {
        inv = inv.wrapping_mul(2u64.wrapping_sub(p.wrapping_mul(inv)));
    }
    (inv & (modulus - 1)) as u32
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn permute_kernel(
    res: &mut [i64],
    a: &[i64],
    n: u32,
    limbs: u32,
    res_base: u32,
    res_stride: u32,
    a_base: u32,
    a_stride: u32,
    a_size: u32,
    mul: u32,
    add: u32,
    invariant: u32,
) {
    let pos = ABSOLUTE_POS as u32;
    if pos < n * limbs {
        let limb = pos / n;
        let t = pos - limb * n;
        let mut value = 0i64;
        if limb < a_size {
            let src = a_base + limb * a_stride;
            if invariant == 0u32 {
                // Output `t` reads input `s mod n`, negated when `s` wraps past `n`.
                let s = (t * mul + add) & (2u32 * n - 1u32);
                let index = s & (n - 1u32);
                value = a[(src + index) as usize];
                if s >= n {
                    value = 0i64 - value;
                }
            } else {
                // The same gather on the ambient vector
                // `[a_0, .., a_{n-1}, 0, -a_{n-1}, .., -a_1]` of degree `2n`.
                let s = (t * mul) & (4u32 * n - 1u32);
                let index = s & (2u32 * n - 1u32);
                if index < n {
                    value = a[(src + index) as usize];
                } else if index > n {
                    value = 0i64 - a[(src + 2u32 * n - index) as usize];
                }
                if s >= 2u32 * n {
                    value = 0i64 - value;
                }
            }
        }
        res[(res_base + limb * res_stride + t) as usize] = value;
    }
}

/// Writes every limb of `res` with the image of `a` under `map`; limbs past
/// `a`'s are zero. `res` and `a` must not overlap.
pub(crate) fn permute<RT: GpuRuntime>(res: &Column, a: &Column, map: Map) {
    assert_eq!(res.n, a.n, "permutation between different degrees");
    if res.size == 0 || res.n == 0 {
        return;
    }
    let n = res.n as u64;
    let (mul, add, invariant) = match map {
        Map::Automorphism { p } => (inverse_odd(p, 2 * n), 0u32, 0u32),
        Map::Rotation { p } => (1u32, ((-(p as i128)).rem_euclid(2 * n as i128)) as u32, 0u32),
        Map::AutomorphismCI { p } => (inverse_odd(p, 4 * n), 0u32, 1u32),
    };
    let client = RT::client();
    let units = res.n * res.size;
    unsafe {
        permute_kernel::launch_unchecked(
            client,
            cubes(client, units, DIM),
            CubeDim::new_1d(DIM),
            res.binding.arg::<i64>(),
            a.binding.arg::<i64>(),
            res.n as u32,
            res.size as u32,
            res.base,
            res.stride,
            a.base,
            a.stride,
            a.size as u32,
            mul,
            add,
            invariant,
        );
    }
    launched(units * 16);
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn ci_kernel(
    res: &mut [i64],
    a: &[i64],
    res_n: u32,
    limbs: u32,
    res_base: u32,
    res_stride: u32,
    a_base: u32,
    a_stride: u32,
    a_size: u32,
    trace: u32,
) {
    let pos = ABSOLUTE_POS as u32;
    if pos < res_n * limbs {
        let limb = pos / res_n;
        let t = pos - limb * res_n;
        let mut value = 0i64;
        if limb < a_size {
            let src = a_base + limb * a_stride;
            if trace == 0u32 {
                // `res` has degree `2n`: `[a_0, .., a_{n-1}, 0, -a_{n-1}, .., -a_1]`.
                let n = res_n / 2u32;
                if t < n {
                    value = a[(src + t) as usize];
                } else if t > n {
                    value = 0i64 - a[(src + res_n - t) as usize];
                }
            } else {
                // `a` has degree `2n`: `res_0 = 2 a_0`, `res_t = a_t - a_{2n-t}`.
                if t == 0u32 {
                    value = a[src as usize] * 2i64;
                } else {
                    value = a[(src + t) as usize] - a[(src + 2u32 * res_n - t) as usize];
                }
            }
        }
        res[(res_base + limb * res_stride + t) as usize] = value;
    }
}

fn ci_map<RT: GpuRuntime>(res: &Column, a: &Column, trace: u32) {
    if res.size == 0 || res.n == 0 {
        return;
    }
    let client = RT::client();
    let units = res.n * res.size;
    unsafe {
        ci_kernel::launch_unchecked(
            client,
            cubes(client, units, DIM),
            CubeDim::new_1d(DIM),
            res.binding.arg::<i64>(),
            a.binding.arg::<i64>(),
            res.n as u32,
            res.size as u32,
            res.base,
            res.stride,
            a.base,
            a.stride,
            a.size as u32,
            trace,
        );
    }
    launched(units * 16);
}

/// Embeds a conjugate-invariant `a` of degree `n` into the standard ring of degree `2n`.
pub(crate) fn ci_embed<RT: GpuRuntime>(res: &Column, a: &Column) {
    assert_eq!(res.n, 2 * a.n, "vec_znx_ci_embed: res degree must be twice the degree of a");
    ci_map::<RT>(res, a, 0);
}

/// Traces a standard `a` of degree `2n` down to the conjugate-invariant ring of degree `n`.
pub(crate) fn ci_trace<RT: GpuRuntime>(res: &Column, a: &Column) {
    assert_eq!(a.n, 2 * res.n, "vec_znx_ci_trace: a degree must be twice the degree of res");
    ci_map::<RT>(res, a, 1);
}
