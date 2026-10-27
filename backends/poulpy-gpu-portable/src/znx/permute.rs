//! Signed permutations of the coefficients: automorphisms, rotations, and the
//! maps between the conjugate-invariant ring and the standard one.

use cubecl::prelude::*;

use super::Column;
use crate::runtime::{GpuRuntime, Label, launch};

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
    assert!(p & 1 == 1, "p must be odd");
    let p = (p as u64) & (modulus - 1);
    // Newton iteration doubles the number of correct low bits each step.
    let mut inv = p;
    for _ in 0..6 {
        inv = inv.wrapping_mul(2u64.wrapping_sub(p.wrapping_mul(inv)));
    }
    (inv & (modulus - 1)) as u32
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments, unused_assignments)]
fn permute_kernel(
    res: &mut [u64],
    a: &[u64],
    n: u32,
    limbs: u32,
    words: u32,
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
        // `index` is the source coefficient, `negate` its sign, `zero` a hole.
        let mut index = 0u32;
        let mut negate = false;
        let mut zero = limb >= a_size;
        if invariant == 0u32 {
            // Output `t` reads input `s mod n`, negated when `s` wraps past `n`.
            let s = (t * mul + add) & (2u32 * n - 1u32);
            index = s & (n - 1u32);
            negate = s >= n;
        } else {
            // The same gather on the ambient vector
            // `[a_0, .., a_{n-1}, 0, -a_{n-1}, .., -a_1]` of degree `2n`.
            let s = (t * mul) & (4u32 * n - 1u32);
            let ambient = s & (2u32 * n - 1u32);
            negate = s >= 2u32 * n;
            if ambient < n {
                index = ambient;
            } else if ambient > n {
                index = 2u32 * n - ambient;
                negate = !negate;
            } else {
                zero = true;
            }
        }
        let mut lo = 0u64;
        let mut hi = 0u64;
        if !zero {
            let src = a_base + limb * a_stride + index * words;
            lo = a[src as usize];
            if words == 2u32 {
                hi = a[(src + 1u32) as usize];
            }
            if negate {
                hi = 0u64 - hi;
                if lo != 0u64 {
                    hi -= 1u64;
                }
                lo = 0u64 - lo;
            }
        }
        let dst = res_base + limb * res_stride + t * words;
        res[dst as usize] = lo;
        if words == 2u32 {
            res[(dst + 1u32) as usize] = hi;
        }
    }
}

/// Writes every limb of `res` with the image of `a` under `map`; limbs past
/// `a`'s are zero. `res` and `a` must not overlap.
pub(crate) fn permute<RT: GpuRuntime>(res: &Column, a: &Column, map: Map) {
    assert!(res.n == a.n, "permutation between different degrees");
    assert!(res.words == a.words, "permutation between different word widths");
    if res.size == 0 || res.n == 0 {
        return;
    }
    let n = res.n as u64;
    let (mul, add, invariant) = match map {
        Map::Automorphism { p } => (inverse_odd(p, 2 * n), 0u32, 0u32),
        Map::Rotation { p } => (1u32, ((-(p as i128)).rem_euclid(2 * n as i128)) as u32, 0u32),
        Map::AutomorphismCI { p } => (inverse_odd(p, 4 * n), 0u32, 1u32),
    };
    let units = res.n * res.size;
    launch!(
        RT, if res.words == 2 { Label::BigPermute } else { Label::ZnxPermute }, units * 16 * res.words as usize, permute_kernel, (units, RT::GROUP, RT::GROUP);
        [
            res.binding => u64,
            a.binding => u64,
        ];
        [
            res.n as u32,
            res.size as u32,
            res.words,
            res.base,
            res.stride,
            a.base,
            a.stride,
            a.size as u32,
            mul,
            add,
            invariant,
        ];
        []
    );
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
    assert!(res.words == 1 && a.words == 1, "the ring maps act on `i64` limbs");
    if res.size == 0 || res.n == 0 {
        return;
    }
    let units = res.n * res.size;
    launch!(
        RT, Label::ZnxRingMap, units * 16, ci_kernel, (units, RT::GROUP, RT::GROUP);
        [
            res.binding => i64,
            a.binding => i64,
        ];
        [
            res.n as u32,
            res.size as u32,
            res.base,
            res.stride,
            a.base,
            a.stride,
            a.size as u32,
            trace,
        ];
        []
    );
}

/// Embeds a conjugate-invariant `a` of degree `n` into the standard ring of degree `2n`.
pub(crate) fn ci_embed<RT: GpuRuntime>(res: &Column, a: &Column) {
    assert!(res.n == 2 * a.n, "vec_znx_ci_embed: res degree must be twice the degree of a");
    ci_map::<RT>(res, a, 0);
}

/// Traces a standard `a` of degree `2n` down to the conjugate-invariant ring of degree `n`.
pub(crate) fn ci_trace<RT: GpuRuntime>(res: &Column, a: &Column) {
    assert!(a.n == 2 * res.n, "vec_znx_ci_trace: a degree must be twice the degree of res");
    ci_map::<RT>(res, a, 1);
}
