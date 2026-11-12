//! Normalization: `res = canon(a * 2^offset, res_base2k, res_k, res.size())`.
//!
//! The source limbs are `i64` or `i128`, or the output of an inverse transform:
//! four residues per coefficient, reconstructed in registers and summed with
//! an optional `i64` addend. Each thread owns one coefficient, or, reading
//! residues over the conjugate-invariant ring, a coefficient and its
//! reflection, whose basis change shares its products, and follows for each
//! the exact definition: the limbs of `a` are read from the least significant one
//! as a stream of bits (carrying across unnormalized limbs), the bits below
//! the output precision are dropped with one rounding bit and a sticky bit,
//! rounding ties to even, and the output digits are emitted from the least
//! significant one, each centered by the carry of the previous.

use cubecl::prelude::*;

use super::{Column, products::const_product};
use crate::{
    ntt4x30::{Consts, DftColumn, RingTables, crt_at, crt_pair_at},
    runtime::{GpuRuntime, Label, launch, read_at, wipe_host, write_at},
};

/// `(lo, hi) >> a_base2k` for the 128-bit two's complement value `(lo, hi)`,
/// with `ab = a_base2k` and `ab_up = 64 - a_base2k`.
#[cube]
fn shift_right(lo: u64, hi: u64, ab: u64, ab_up: u64) -> (u64, u64) {
    let sign = 0u64 - (hi >> 63u64);
    ((lo >> ab) | (hi << ab_up), (hi >> ab) | (sign << ab_up))
}

/// `(lo, hi) + y` for the `i64` `y`.
#[cube]
fn add_word(lo: u64, hi: u64, y: u64) -> (u64, u64) {
    let sum = lo + y;
    let mut high = hi + (0u64 - (y >> 63u64));
    if sum < lo {
        high += 1u64;
    }
    (sum, high)
}

/// The carry and the pending bits of a stream once limb `x` joins it:
/// `low = (x mod 2^a_base2k) + (carry mod 2^a_base2k)`, the carry becoming
/// `(x >> a_base2k) + shifted + (low >> a_base2k)`, `shifted = carry >> a_base2k`,
/// and the pending bits `low mod 2^a_base2k`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn absorb(
    x_lo: u64,
    x_hi: u64,
    carry_lo: u64,
    shifted_lo: u64,
    shifted_hi: u64,
    ab: u64,
    ab_up: u64,
    mask: u64,
) -> (u64, u64, u64) {
    let low = (x_lo & mask) + (carry_lo & mask);
    let (xs_lo, xs_hi) = shift_right(x_lo, x_hi, ab, ab_up);
    let sum_lo = xs_lo + shifted_lo;
    let mut sum_hi = xs_hi + shifted_hi;
    if sum_lo < xs_lo {
        sum_hi += 1u64;
    }
    let next_lo = sum_lo + (low >> ab);
    let mut next_hi = sum_hi;
    if next_lo < sum_lo {
        next_hi += 1u64;
    }
    (next_lo, next_hi, low & mask)
}

/// Whether the 128-bit two's complement value `(lo, hi)` is 0 or -1.
#[cube]
fn sign_only(lo: u64, hi: u64) -> bool {
    let ones = 0xFFFF_FFFF_FFFF_FFFFu64;
    (lo == 0u64 && hi == 0u64) || (lo == ones && hi == ones)
}

/// The digit of field `field >= 2` and the carry into the next: its bits
/// `result` above `zeroes` zero bits plus `rounding`, the carry of the previous
/// field or, into the first digit, the rounding bit, rounding ties to even.
#[cube]
fn emit(result: u64, rounding: u64, sticky: u64, field: u32, width: u32, zeroes: u32, precision_padding: u32) -> (u64, u64) {
    let w = width as u64;
    let mut value = result << (zeroes as u64);
    if field == 2u32 {
        // Ties to even: up past a half, and at a half onto an odd digit.
        let odd = value & 1u64;
        let mut up = 0u64;
        if sticky != 0u64 || odd != 0u64 {
            up = rounding;
        }
        value += up;
    } else {
        value += rounding;
    }
    let carry = (value + (1u64 << (w - 1u64))) >> w;
    let mut digit = value - (carry << w);
    if field == 2u32 && precision_padding != 0u32 {
        digit <<= precision_padding as u64;
    }
    (digit, carry)
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments, unused_assignments)]
fn normalize_kernel(
    res: &mut [u64],
    // The source limbs, or the addend of a source held in `dft`.
    a: &[u64],
    dft: &[u32],
    basis: &[u32],
    // The operands of a source that is a sum of constant products.
    ta0: &[u64],
    ta1: &[u64],
    ta2: &[u64],
    ta3: &[u64],
    tb0: &[u64],
    tb1: &[u64],
    tb2: &[u64],
    tb3: &[u64],
    n: u32,
    // `n`, or `n / 2 + 1` reading residues over the conjugate-invariant ring.
    units: u32,
    res_base: u32,
    res_stride: u32,
    res_size: u32,
    a_base: u32,
    a_stride: u32,
    a_size: u32,
    // 64-bit words per source coefficient: 1 for `i64`, 2 for `i128`.
    a_words: u32,
    a_base2k: u32,
    res_base2k: u32,
    // Limbs that carry precision, and the unused low bits of the last of them.
    active_size: u32,
    precision_padding: u32,
    // Bits of `a` below the output precision, and zero bits to insert
    // below `a`: one of the two is zero.
    drop: u32,
    insert: u32,
    dft_base: u32,
    dft_stride: u32,
    dft_size: u32,
    // Per product: `a` base, stride and size, `b` base, stride and size, offset.
    t0_a_base: u32,
    t0_a_stride: u32,
    t0_a_size: u32,
    t0_b_base: u32,
    t0_b_stride: u32,
    t0_b_size: u32,
    t0_offset: u32,
    t1_a_base: u32,
    t1_a_stride: u32,
    t1_a_size: u32,
    t1_b_base: u32,
    t1_b_stride: u32,
    t1_b_size: u32,
    t1_offset: u32,
    t2_a_base: u32,
    t2_a_stride: u32,
    t2_a_size: u32,
    t2_b_base: u32,
    t2_b_stride: u32,
    t2_b_size: u32,
    t2_offset: u32,
    t3_a_base: u32,
    t3_a_stride: u32,
    t3_a_size: u32,
    t3_b_base: u32,
    t3_b_stride: u32,
    t3_b_size: u32,
    t3_offset: u32,
    #[comptime] c: Consts,
    // The source is `dft`, the output of an inverse transform.
    #[comptime] residues: bool,
    #[comptime] invariant: bool,
    // The source limbs are sums of this many constant products, `a_size` of them.
    #[comptime] products: u32,
) {
    // A unit then also streams the reflection `j = n - i` (the `_j` values) in lockstep with `i`.
    let pair = comptime!(residues && invariant);
    let i = ABSOLUTE_POS as u32;
    if i < units {
        let j = (n - i) & (n - 1u32);
        // Every value is two's complement in unsigned registers, with the
        // arithmetic shifts and sign extensions written out: the runtimes do
        // not agree on signed 64-bit shifts.
        let ab = a_base2k as u64;
        let ab_up = 64u64 - ab;
        let mask = (1u64 << ab) - 1u64;
        let ones = 0xFFFF_FFFF_FFFF_FFFFu64;

        // Bit stream over the limbs of `a`, least significant first. The
        // carry between limbs is a 128-bit two's complement value.
        let mut left = a_size;
        if residues {
            left = dft_size;
        }
        let mut carry_lo = 0u64;
        let mut carry_hi = 0u64;
        let mut pending = 0u64;
        let mut carry_lo_j = 0u64;
        let mut carry_hi_j = 0u64;
        let mut pending_j = 0u64;
        let mut available = 0u32;

        let mut rounding = 0u64;
        let mut rounding_j = 0u64;
        // Any one among the bits below the rounding bit.
        let mut sticky = 0u64;
        let mut sticky_j = 0u64;
        let mut padding = insert;

        // Field 0 discards `drop - 1` bits, field 1 reads the rounding bit, and
        // each later field reads one output digit, least significant first.
        let fields = active_size + 2u32;
        for field in 0..fields {
            let mut count = 0u32;
            let mut width = 0u32;
            let mut zeroes = 0u32;
            let discard = field == 0u32;
            if field == 0u32 {
                if drop > 1u32 {
                    count = drop - 1u32;
                }
            } else if field == 1u32 {
                if drop > 0u32 {
                    count = 1u32;
                }
            } else {
                width = res_base2k;
                if field == 2u32 && precision_padding != 0u32 {
                    width = res_base2k - precision_padding;
                }
                if padding > 0u32 {
                    zeroes = min(padding, width);
                    padding -= zeroes;
                }
                count = width - zeroes;
            }

            let mut result = 0u64;
            let mut result_j = 0u64;
            let mut written = 0u32;
            while count != 0u32 {
                if available == 0u32 {
                    // `carry >> a_base2k`, shared by both refills.
                    let shifted = shift_right(carry_lo, carry_hi, ab, ab_up);
                    let shifted_j = shift_right(carry_lo_j, carry_hi_j, ab, ab_up);
                    if left != 0u32 {
                        left -= 1u32;
                        let mut x_lo = 0u64;
                        let mut x_hi = 0u64;
                        let mut y_lo = 0u64;
                        let mut y_hi = 0u64;
                        if comptime!(products > 0) {
                            #[unroll]
                            for term in 0..products {
                                let mut p_lo = 0u64;
                                let mut p_hi = 0u64;
                                if comptime!(term == 0) {
                                    let product = const_product(
                                        ta0,
                                        tb0,
                                        i,
                                        left + t0_offset,
                                        t0_a_base,
                                        t0_a_stride,
                                        t0_a_size,
                                        t0_b_base,
                                        t0_b_stride,
                                        t0_b_size,
                                    );
                                    p_lo = product.0;
                                    p_hi = product.1;
                                } else if comptime!(term == 1) {
                                    let product = const_product(
                                        ta1,
                                        tb1,
                                        i,
                                        left + t1_offset,
                                        t1_a_base,
                                        t1_a_stride,
                                        t1_a_size,
                                        t1_b_base,
                                        t1_b_stride,
                                        t1_b_size,
                                    );
                                    p_lo = product.0;
                                    p_hi = product.1;
                                } else if comptime!(term == 2) {
                                    let product = const_product(
                                        ta2,
                                        tb2,
                                        i,
                                        left + t2_offset,
                                        t2_a_base,
                                        t2_a_stride,
                                        t2_a_size,
                                        t2_b_base,
                                        t2_b_stride,
                                        t2_b_size,
                                    );
                                    p_lo = product.0;
                                    p_hi = product.1;
                                } else {
                                    let product = const_product(
                                        ta3,
                                        tb3,
                                        i,
                                        left + t3_offset,
                                        t3_a_base,
                                        t3_a_stride,
                                        t3_a_size,
                                        t3_b_base,
                                        t3_b_stride,
                                        t3_b_size,
                                    );
                                    p_lo = product.0;
                                    p_hi = product.1;
                                }
                                let sum = x_lo + p_lo;
                                x_hi += p_hi;
                                if sum < x_lo {
                                    x_hi += 1u64;
                                }
                                x_lo = sum;
                            }
                        } else if residues {
                            let block = dft_base + left * dft_stride;
                            if pair {
                                let (value, value_j) = crt_pair_at(dft, basis, block, i, n, c);
                                x_lo = value.0;
                                x_hi = value.1;
                                y_lo = value_j.0;
                                y_hi = value_j.1;
                            } else {
                                let value = crt_at(dft, block, i, n, c);
                                x_lo = value.0;
                                x_hi = value.1;
                            }
                            if left < a_size {
                                let row = a_base + left * a_stride;
                                let sum = add_word(x_lo, x_hi, a[(row + i) as usize]);
                                x_lo = sum.0;
                                x_hi = sum.1;
                                if pair {
                                    let sum_j = add_word(y_lo, y_hi, a[(row + j) as usize]);
                                    y_lo = sum_j.0;
                                    y_hi = sum_j.1;
                                }
                            }
                        } else {
                            let index = a_base + left * a_stride + i * a_words;
                            x_lo = a[index as usize];
                            if a_words == 2u32 {
                                x_hi = a[(index + 1u32) as usize];
                            } else {
                                x_hi = 0u64 - (x_lo >> 63u64);
                            }
                        }
                        let next = absorb(x_lo, x_hi, carry_lo, shifted.0, shifted.1, ab, ab_up, mask);
                        carry_lo = next.0;
                        carry_hi = next.1;
                        pending = next.2;
                        if pair {
                            let next_j = absorb(y_lo, y_hi, carry_lo_j, shifted_j.0, shifted_j.1, ab, ab_up, mask);
                            carry_lo_j = next_j.0;
                            carry_hi_j = next_j.1;
                            pending_j = next_j.2;
                        }
                        available = a_base2k;
                    } else {
                        let mut settled = discard && sign_only(carry_lo, carry_hi);
                        if pair {
                            settled = settled && sign_only(carry_lo_j, carry_hi_j);
                        }
                        if settled {
                            // Only the sign extension is left to discard: ones below a negative value.
                            if carry_lo == ones {
                                sticky = 1u64;
                            }
                            if pair {
                                sticky_j = select(carry_lo_j == ones, 1u64, sticky_j);
                            }
                            count = 0u32;
                        } else {
                            pending = carry_lo & mask;
                            carry_lo = shifted.0;
                            carry_hi = shifted.1;
                            if pair {
                                pending_j = carry_lo_j & mask;
                                carry_lo_j = shifted_j.0;
                                carry_hi_j = shifted_j.1;
                            }
                            available = a_base2k;
                        }
                    }
                }
                if count != 0u32 {
                    let mut take = count;
                    if available < take {
                        take = available;
                    }
                    let keep = (1u64 << (take as u64)) - 1u64;
                    if discard {
                        sticky |= pending & keep;
                        if pair {
                            sticky_j |= pending_j & keep;
                        }
                    } else {
                        result |= (pending & keep) << (written as u64);
                        if pair {
                            result_j |= (pending_j & keep) << (written as u64);
                        }
                        written += take;
                    }
                    pending >>= take as u64;
                    if pair {
                        pending_j >>= take as u64;
                    }
                    available -= take;
                    count -= take;
                }
            }

            if field == 1u32 {
                rounding = result;
                if pair {
                    rounding_j = result_j;
                }
            } else if field >= 2u32 {
                let row = res_base + (active_size + 1u32 - field) * res_stride;
                let (digit, carry) = emit(result, rounding, sticky, field, width, zeroes, precision_padding);
                rounding = carry;
                res[(row + i) as usize] = digit;
                if pair {
                    let (digit_j, carry_j) = emit(result_j, rounding_j, sticky_j, field, width, zeroes, precision_padding);
                    rounding_j = carry_j;
                    res[(row + j) as usize] = digit_j;
                }
            }
        }

        for limb in active_size..res_size {
            let row = res_base + limb * res_stride;
            res[(row + i) as usize] = 0u64;
            if pair {
                res[(row + j) as usize] = 0u64;
            }
        }
    }
}

/// What a normalization reads.
enum Source<'a, RT: GpuRuntime> {
    /// `i64` or `i128` limbs.
    Limbs(&'a Column),
    /// The output of an inverse transform, plus the first limbs of an `i64` column.
    Residues {
        dft: &'a DftColumn,
        addend: Option<&'a Column>,
        tables: RingTables<'a, RT>,
    },
    /// The sum of constant products `(a, b, coefficient of b, offset)`, over `size` limbs.
    Products {
        terms: &'a [(Column, Column, usize, usize)],
        size: usize,
    },
}

/// Normalizes every limb of `res` from `a`. `res` may be `a` itself when
/// `offset <= 0` and both radices are equal: a digit is only written once the
/// limbs it depends on have been read.
pub(crate) fn normalize<RT: GpuRuntime>(res: &Column, res_base2k: usize, res_k: usize, offset: i64, a: &Column, a_base2k: usize) {
    assert!(res.n == a.n, "normalization between different degrees");
    if a.words == 2 && (res_base2k > KERNEL_RADIX || a_base2k > KERNEL_RADIX) {
        return normalize_big_on_host::<RT>(res, res_base2k, res_k, offset, a, a_base2k);
    }
    launch::<RT>(res, res_base2k, res_k, offset, Source::Limbs(a), a_base2k);
}

/// Widest radix the kernel handles: it keeps its bit stream and its digits
/// in 64-bit registers.
pub(crate) const KERNEL_RADIX: usize = 62;

/// [`normalize`] of an `i128` source at the radices the kernel does not
/// handle (a source radix up to 127, a destination radix up to 64), which
/// the HAL admits and no parameter set of this backend reaches: the limbs go
/// through the host and the portable routine.
fn normalize_big_on_host<RT: GpuRuntime>(
    res: &Column,
    res_base2k: usize,
    res_k: usize,
    offset: i64,
    a: &Column,
    a_base2k: usize,
) {
    use poulpy_cpu_portable::{NTT4x30Portable, kernels::vec_znx_big_normalize::vec_znx_big_normalize_portable};
    use poulpy_hal::{
        AlignedBuf,
        layouts::{DataViewMut, Standard, VecZnx, VecZnxBig, VecZnxBigToBackendRef, VecZnxToBackendMut},
    };
    assert!(res.words == 1, "normalization writes `i64` limbs");
    if res.size == 0 || res.n == 0 {
        return;
    }
    let n = res.n;
    // Both columns as dense host vectors of one column.
    let limb = |column: &Column, limb: usize| (column.base as usize + limb * column.stride as usize) * 8;
    let mut source = AlignedBuf::from(vec![0u8; n * a.size * 16]);
    for j in 0..a.size {
        read_at::<RT>(&a.binding, limb(a, j), &mut source[j * n * 16..(j + 1) * n * 16]);
    }
    let mut source = VecZnxBig::<AlignedBuf, i128, NTT4x30Portable>::from_data(source, n, 1, a.size);
    let mut target = VecZnx::<AlignedBuf, i64>::from_data(AlignedBuf::from(vec![0u8; n * res.size * 8]), n, 1, res.size);
    vec_znx_big_normalize_portable::<NTT4x30Portable, Standard>(
        &mut VecZnxToBackendMut::<NTT4x30Portable>::to_backend_mut(&mut target),
        res_base2k,
        res_k,
        offset,
        0,
        &source.to_backend_ref(),
        a_base2k,
        0,
    );
    for j in 0..res.size {
        write_at::<RT>(&res.binding, limb(res, j), &target.data()[j * n * 8..(j + 1) * n * 8]);
    }
    // A decryption reaches this stage with `c1 * s` in the source.
    wipe_host(source.data_mut());
    wipe_host(target.data_mut());
}

/// Normalizes every limb of `res` from the `size` limbs
/// `sum_t sum_{u + v = j + offset_t} a_t[u] * b_t[v][coeff_t]`, the sum of
/// [`convolution_by_const`](super::convolution_by_const) over one to
/// [`CONST_TERMS`](super::CONST_TERMS) terms `(a_t, b_t, coeff_t, offset_t)`, without storing it. `res` overlaps no operand.
#[cfg_attr(not(feature = "enable-core"), allow(dead_code))]
pub(crate) fn normalize_products<RT: GpuRuntime>(
    res: &Column,
    res_base2k: usize,
    res_k: usize,
    terms: &[(Column, Column, usize, usize)],
    size: usize,
    a_base2k: usize,
) {
    assert!((1..=super::CONST_TERMS).contains(&terms.len()), "one to four products");
    for (a, b, b_coeff, _) in terms {
        assert!(
            a.words == 1 && b.words == 1 && a.n == res.n,
            "constant products take one-word operands of the destination degree"
        );
        assert!(*b_coeff < b.n || b.size == 0, "constant coefficient out of range");
    }
    launch::<RT>(res, res_base2k, res_k, 0, Source::Products { terms, size }, a_base2k);
}

/// Normalizes every limb of `res` from `dft + addend`, `dft` holding the
/// output of the inverse transform and `addend` contributing its first
/// `dft.size` limbs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn normalize_residues<RT: GpuRuntime>(
    tables: RingTables<'_, RT>,
    res: &Column,
    res_base2k: usize,
    res_k: usize,
    offset: i64,
    dft: &DftColumn,
    addend: Option<&Column>,
    a_base2k: usize,
) {
    assert!(res.n == dft.n, "normalization between different degrees");
    if let Some(addend) = addend {
        assert!(
            addend.n == res.n && addend.words == 1,
            "the addend is an `i64` column of the same degree"
        );
    }
    launch::<RT>(
        res,
        res_base2k,
        res_k,
        offset,
        Source::Residues { dft, addend, tables },
        a_base2k,
    );
}

fn launch<RT: GpuRuntime>(res: &Column, res_base2k: usize, res_k: usize, offset: i64, source: Source<'_, RT>, a_base2k: usize) {
    assert!(res.words == 1, "normalization writes `i64` limbs");
    assert!(
        (1..=KERNEL_RADIX).contains(&res_base2k) && (1..=KERNEL_RADIX).contains(&a_base2k),
        "radix widths must be in 1..={KERNEL_RADIX} (got {a_base2k} to {res_base2k})"
    );
    assert!(res_k <= res.size * res_base2k, "res_k exceeds the destination precision");
    if res.size == 0 || res.n == 0 {
        return;
    }
    let (active_size, precision_padding) = if res_k == res.size * res_base2k {
        (res.size, 0)
    } else {
        let active_size = res_k.div_ceil(res_base2k);
        (active_size, active_size * res_base2k - res_k)
    };
    // Operands a mode does not read are bound to a buffer of the launch.
    let terms = match source {
        Source::Products { terms, .. } => terms,
        _ => &[],
    };
    let (a, a_size, dft, basis, invariant, source_size) = match source {
        Source::Limbs(a) => (a, a.size, None, a.binding.clone(), false, a.size),
        Source::Products { size, .. } => (res, size, None, res.binding.clone(), false, size),
        Source::Residues { dft, addend, tables } => (
            addend.unwrap_or(res),
            addend.map_or(0, |addend| addend.size.min(dft.size)),
            Some(dft),
            tables
                .of(res.n)
                .basis
                .view()
                .binding()
                .expect("transform tables hold storage"),
            tables.invariant(),
            dft.size,
        ),
    };
    // Products beyond the last repeat it: the kernel does not read them.
    let term = |index: usize| terms.get(index.min(terms.len().saturating_sub(1)));
    let scalars = |index: usize| match term(index) {
        Some((a, b, b_coeff, offset)) => [
            a.base,
            a.stride,
            a.size as u32,
            b.base + *b_coeff as u32,
            b.stride,
            b.size as u32,
            (*offset).min(a.size + b.size) as u32,
        ],
        None => [0; 7],
    };
    let label = match (dft.is_some(), a.words) {
        _ if !terms.is_empty() => Label::NormalizeProducts,
        (true, _) => Label::NormalizeResidues,
        (false, 2) => Label::NormalizeBig,
        _ => Label::NormalizeLimbs,
    };
    // Past these bounds every further bit dropped is sign extension, and
    // every further bit inserted lies below the last digit.
    let source_bits = (source_size * a_base2k) as i128;
    let drop = (source_bits - res_k as i128 - offset as i128).clamp(-(res_k as i128) - 64, source_bits + 192);
    let (drop, insert) = (drop.max(0) as u32, (-drop).max(0) as u32);
    let source_bytes = match dft {
        _ if !terms.is_empty() => terms.iter().map(|(a, b, _, _)| 8 * a.size.min(a_size + b.size)).sum(),
        Some(dft) => dft.size * 16 + a_size * 8,
        None => a_size * 8 * a.words as usize,
    };
    // `invariant` holds for residues only, where the kernel pairs each coefficient with its reflection.
    let units = if invariant { res.n / 2 + 1 } else { res.n };
    launch!(
        RT, label, res.n * (res.size * 8 + source_bytes), normalize_kernel, (units, RT::GROUP, RT::GROUP);
        [
            res.binding => u64,
            a.binding => u64,
            *dft.map_or(&a.binding, |dft| &dft.binding) => u32,
            basis => u32,
            *term(0).map_or(&a.binding, |term| &term.0.binding) => u64,
            *term(1).map_or(&a.binding, |term| &term.0.binding) => u64,
            *term(2).map_or(&a.binding, |term| &term.0.binding) => u64,
            *term(3).map_or(&a.binding, |term| &term.0.binding) => u64,
            *term(0).map_or(&a.binding, |term| &term.1.binding) => u64,
            *term(1).map_or(&a.binding, |term| &term.1.binding) => u64,
            *term(2).map_or(&a.binding, |term| &term.1.binding) => u64,
            *term(3).map_or(&a.binding, |term| &term.1.binding) => u64,
        ];
        [
            res.n as u32,
            units as u32,
            res.base,
            res.stride,
            res.size as u32,
            a.base,
            a.stride,
            a_size as u32,
            a.words,
            a_base2k as u32,
            res_base2k as u32,
            active_size as u32,
            precision_padding as u32,
            drop,
            insert,
            dft.map_or(0, |dft| dft.base),
            dft.map_or(0, |dft| dft.stride),
            dft.map_or(0, |dft| dft.size as u32),
            scalars(0)[0],
            scalars(0)[1],
            scalars(0)[2],
            scalars(0)[3],
            scalars(0)[4],
            scalars(0)[5],
            scalars(0)[6],
            scalars(1)[0],
            scalars(1)[1],
            scalars(1)[2],
            scalars(1)[3],
            scalars(1)[4],
            scalars(1)[5],
            scalars(1)[6],
            scalars(2)[0],
            scalars(2)[1],
            scalars(2)[2],
            scalars(2)[3],
            scalars(2)[4],
            scalars(2)[5],
            scalars(2)[6],
            scalars(3)[0],
            scalars(3)[1],
            scalars(3)[2],
            scalars(3)[3],
            scalars(3)[4],
            scalars(3)[5],
            scalars(3)[6],
        ];
        [Consts::get(), dft.is_some(), invariant, terms.len() as u32]
    );
}
