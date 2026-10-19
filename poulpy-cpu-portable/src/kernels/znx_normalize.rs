//! One normalization kernel per backend: `res = canon(a * 2^offset, res_base2k, res_k, res.size)`
//! with ties to even. The driver walks blocks of coefficients through every limb with the
//! carry, sticky and half bits in lanes, so rounding sees the exact dropped part.

use std::marker::PhantomData;

use crate::layouts::{Backend, VecZnxShape};

/// Limb `j` of one column, over a coefficient range, starts at `ptr + j * stride`.
#[derive(Clone, Copy)]
pub struct Limbs<T> {
    pub ptr: *const T,
    pub stride: usize,
    pub size: usize,
}

#[derive(Clone, Copy)]
pub struct LimbsMut<T> {
    pub ptr: *mut T,
    pub stride: usize,
    pub size: usize,
}

fn column_stride(shape: VecZnxShape, col: usize) -> usize {
    if shape.size() < 2 {
        0
    } else {
        shape.scalar_offset(col, 1) - shape.scalar_offset(col, 0)
    }
}

impl<T> Limbs<T> {
    /// Column `col` of `shape` from coefficient `start`, `base` being the buffer base.
    pub fn new(base: *const T, shape: VecZnxShape, col: usize, start: usize) -> Self {
        Self {
            ptr: base.wrapping_add(shape.scalar_offset(col, 0) + start),
            stride: column_stride(shape, col),
            size: shape.size(),
        }
    }
}

impl<T> LimbsMut<T> {
    /// Column `col` of `shape` from coefficient `start`, `base` being the buffer base.
    pub fn new(base: *mut T, shape: VecZnxShape, col: usize, start: usize) -> Self {
        Self {
            ptr: base.wrapping_add(shape.scalar_offset(col, 0) + start),
            stride: column_stride(shape, col),
            size: shape.size(),
        }
    }
}

/// A source or carry word of the normalization kernel.
pub trait NormalizeWord: Copy + Send + Sync + 'static {
    const ZERO: Self;
    /// Low 64 bits.
    fn low(self) -> i64;
    fn sra(self, k: u32) -> Self;
    fn wrapping_add(self, other: Self) -> Self;
    /// `self + x` with `x` sign-extended.
    fn add_i64(self, x: i64) -> Self;
    fn to_i128(self) -> i128;
}

impl NormalizeWord for i64 {
    const ZERO: Self = 0;
    #[inline(always)]
    fn low(self) -> i64 {
        self
    }
    #[inline(always)]
    fn sra(self, k: u32) -> Self {
        self >> k
    }
    #[inline(always)]
    fn wrapping_add(self, other: Self) -> Self {
        i64::wrapping_add(self, other)
    }
    #[inline(always)]
    fn add_i64(self, x: i64) -> Self {
        i64::wrapping_add(self, x)
    }
    #[inline(always)]
    fn to_i128(self) -> i128 {
        self as i128
    }
}

impl NormalizeWord for i128 {
    const ZERO: Self = 0;
    #[inline(always)]
    fn low(self) -> i64 {
        self as i64
    }
    #[inline(always)]
    fn sra(self, k: u32) -> Self {
        self >> k
    }
    #[inline(always)]
    fn wrapping_add(self, other: Self) -> Self {
        i128::wrapping_add(self, other)
    }
    #[inline(always)]
    fn add_i64(self, x: i64) -> Self {
        i128::wrapping_add(self, x as i128)
    }
    #[inline(always)]
    fn to_i128(self) -> i128 {
        self
    }
}

/// `WIDTH` coefficients in registers: `N` holds i64 lanes, `W` holds source words and
/// carries. Every shift amount is uniform across lanes and below 64.
pub trait NormalizeLanes {
    const WIDTH: usize;
    type Word: NormalizeWord;
    type N: Copy;
    type W: Copy;

    /// # Safety
    /// `p` is valid for `WIDTH` reads.
    unsafe fn load(p: *const Self::Word) -> Self::W;
    /// # Safety
    /// `p` is valid for `WIDTH` writes.
    unsafe fn store(p: *mut i64, x: Self::N);
    fn zero() -> Self::W;
    fn splat(x: i64) -> Self::N;
    fn add(x: Self::N, y: Self::N) -> Self::N;
    fn sub(x: Self::N, y: Self::N) -> Self::N;
    fn and(x: Self::N, y: Self::N) -> Self::N;
    fn or(x: Self::N, y: Self::N) -> Self::N;
    fn shl(x: Self::N, k: u32) -> Self::N;
    fn srl(x: Self::N, k: u32) -> Self::N;
    /// Sign-filling right shift.
    fn sra(x: Self::N, k: u32) -> Self::N;
    /// 1 in every lane where `x != 0`, else 0.
    fn ne_zero(x: Self::N) -> Self::N;
    fn wide_add(x: Self::W, y: Self::W) -> Self::W;
    /// `x + y` with `y` sign-extended.
    fn wide_add_n(x: Self::W, y: Self::N) -> Self::W;
    fn wide_sra(x: Self::W, k: u32) -> Self::W;
    /// Low 64 bits of every lane.
    fn wide_low(x: Self::W) -> Self::N;
}

/// Portable lanes: `W` words in arrays.
pub struct PortableLanes<S, const W: usize>(PhantomData<S>);

impl<S: NormalizeWord, const W: usize> NormalizeLanes for PortableLanes<S, W> {
    const WIDTH: usize = W;
    type Word = S;
    type N = [i64; W];
    type W = [S; W];

    #[inline(always)]
    unsafe fn load(p: *const S) -> [S; W] {
        unsafe { p.cast::<[S; W]>().read_unaligned() }
    }
    #[inline(always)]
    unsafe fn store(p: *mut i64, x: [i64; W]) {
        unsafe { p.cast::<[i64; W]>().write_unaligned(x) }
    }
    #[inline(always)]
    fn zero() -> [S; W] {
        [S::ZERO; W]
    }
    #[inline(always)]
    fn splat(x: i64) -> [i64; W] {
        [x; W]
    }
    #[inline(always)]
    fn add(x: [i64; W], y: [i64; W]) -> [i64; W] {
        std::array::from_fn(|i| x[i].wrapping_add(y[i]))
    }
    #[inline(always)]
    fn sub(x: [i64; W], y: [i64; W]) -> [i64; W] {
        std::array::from_fn(|i| x[i].wrapping_sub(y[i]))
    }
    #[inline(always)]
    fn and(x: [i64; W], y: [i64; W]) -> [i64; W] {
        std::array::from_fn(|i| x[i] & y[i])
    }
    #[inline(always)]
    fn or(x: [i64; W], y: [i64; W]) -> [i64; W] {
        std::array::from_fn(|i| x[i] | y[i])
    }
    #[inline(always)]
    fn shl(x: [i64; W], k: u32) -> [i64; W] {
        std::array::from_fn(|i| x[i] << k)
    }
    #[inline(always)]
    fn srl(x: [i64; W], k: u32) -> [i64; W] {
        std::array::from_fn(|i| ((x[i] as u64) >> k) as i64)
    }
    #[inline(always)]
    fn sra(x: [i64; W], k: u32) -> [i64; W] {
        std::array::from_fn(|i| x[i] >> k)
    }
    #[inline(always)]
    fn ne_zero(x: [i64; W]) -> [i64; W] {
        std::array::from_fn(|i| (x[i] != 0) as i64)
    }
    #[inline(always)]
    fn wide_add(x: [S; W], y: [S; W]) -> [S; W] {
        std::array::from_fn(|i| x[i].wrapping_add(y[i]))
    }
    #[inline(always)]
    fn wide_add_n(x: [S; W], y: [i64; W]) -> [S; W] {
        std::array::from_fn(|i| x[i].add_i64(y[i]))
    }
    #[inline(always)]
    fn wide_sra(x: [S; W], k: u32) -> [S; W] {
        std::array::from_fn(|i| x[i].sra(k))
    }
    #[inline(always)]
    fn wide_low(x: [S; W]) -> [i64; W] {
        std::array::from_fn(|i| x[i].low())
    }
}

/// The normalization kernel of a backend. Every CPU backend word is `i64`; its big word is
/// `i64` or `i128`.
pub trait ZnxNormalize: Backend<ZnxWord = i64, BigWord: NormalizeWord> {
    /// `res = canon(a * 2^offset, res_base2k, res_k, res.size)` over `len` coefficients.
    ///
    /// # Safety
    /// See [`normalize_blocked`].
    #[allow(clippy::too_many_arguments)]
    unsafe fn znx_normalize(
        res: LimbsMut<Self::ZnxWord>,
        res_base2k: usize,
        res_k: usize,
        offset: i64,
        a: Limbs<Self::ZnxWord>,
        a_base2k: usize,
        len: usize,
    ) {
        unsafe {
            normalize_blocked::<PortableLanes<i64, 8>, PortableLanes<i64, 1>>(res, res_base2k, res_k, offset, a, a_base2k, len)
        }
    }

    /// [`Self::znx_normalize`] from big-word limbs.
    ///
    /// # Safety
    /// See [`normalize_blocked`].
    #[allow(clippy::too_many_arguments)]
    unsafe fn znx_normalize_big(
        res: LimbsMut<Self::ZnxWord>,
        res_base2k: usize,
        res_k: usize,
        offset: i64,
        a: Limbs<Self::BigWord>,
        a_base2k: usize,
        len: usize,
    ) {
        unsafe {
            normalize_blocked::<PortableLanes<Self::BigWord, 8>, PortableLanes<Self::BigWord, 1>>(
                res, res_base2k, res_k, offset, a, a_base2k, len,
            )
        }
    }
}

/// Source positions past the source a call may walk before the scalar path is cheaper.
const MAX_EXTRA_LIMBS: i128 = 128;

#[derive(Clone, Copy)]
struct SameBase {
    base: u32,
    lsh: u32,
    padding: u32,
    active: usize,
    /// Destination limb `d` reads source limb `d + limb_offset`.
    limb_offset: i64,
    /// Source position of the limb holding `res_k`.
    boundary: i64,
    /// Lowest walked source position.
    low: i64,
}

impl SameBase {
    /// With `|a| <= 2^62`, `a + carry` and `t + 2^(b-1)` fit an i64 when `2 <= b <= 62`: `|carry| < 2^61`, `2^(b-1) <= 2^61`.
    fn whole(&self) -> bool {
        self.lsh == 0 && (2..=62).contains(&self.base)
    }
}

#[derive(Clone, Copy)]
struct Cross {
    a_base: u32,
    res_base: u32,
    padding: u32,
    active: usize,
    /// Dropped low bits of the source integer; a negative value enters zero bits below it.
    drop: i64,
}

enum Plan {
    Zero,
    Same(SameBase),
    Cross(Cross),
    Exact,
}

fn plan(res_base2k: usize, res_k: usize, offset: i64, a_size: usize, a_base2k: usize) -> Plan {
    if res_k == 0 {
        return Plan::Zero;
    }
    if !(1..=63).contains(&res_base2k) || !(1..=63).contains(&a_base2k) {
        return Plan::Exact;
    }
    let active = res_k.div_ceil(res_base2k);
    let padding = (active * res_base2k - res_k) as u32;
    if a_base2k == res_base2k {
        let b = res_base2k as i64;
        let limb_offset = offset.div_euclid(b);
        let Some(boundary) = (active as i64 - 1).checked_add(limb_offset) else {
            return Plan::Exact;
        };
        let low = boundary.max(a_size as i64 - 1);
        if low as i128 - boundary as i128 > a_size as i128 + MAX_EXTRA_LIMBS {
            return Plan::Exact;
        }
        return Plan::Same(SameBase {
            base: b as u32,
            lsh: offset.rem_euclid(b) as u32,
            padding,
            active,
            limb_offset,
            boundary,
            low,
        });
    }
    let drop = (a_size * a_base2k) as i128 - res_k as i128 - offset as i128;
    if -drop >= res_k as i128 {
        return Plan::Zero;
    }
    if drop > (a_size as i128 + MAX_EXTRA_LIMBS) * a_base2k as i128 {
        return Plan::Exact;
    }
    Plan::Cross(Cross {
        a_base: a_base2k as u32,
        res_base: res_base2k as u32,
        padding,
        active,
        drop: drop as i64,
    })
}

#[inline(always)]
fn low_mask(bits: u32) -> i64 {
    ((1u64 << bits) - 1) as i64
}

/// `(t mod 2^b, floor(t / 2^b))` of `t = a * 2^lsh + carry`; only `WHOLE` (`lsh == 0`, `t` fits) forms `t`.
#[inline(always)]
fn split<L: NormalizeLanes, const WHOLE: bool>(a: L::W, carry: L::W, b: u32, lsh: u32) -> (L::N, L::W) {
    if WHOLE {
        let t = L::wide_add(a, carry);
        return (L::and(L::wide_low(t), L::splat(low_mask(b))), L::wide_sra(t, b));
    }
    let low = L::add(
        L::shl(L::and(L::wide_low(a), L::splat(low_mask(b - lsh))), lsh),
        L::and(L::wide_low(carry), L::splat(low_mask(b))),
    );
    let high = L::wide_add_n(L::wide_add(L::wide_sra(a, b - lsh), L::wide_sra(carry, b)), L::srl(low, b));
    (L::and(low, L::splat(low_mask(b))), high)
}

/// Centered digit of `x` on `w` bits and the unit it carries; `0 <= x <= 2^w`.
#[inline(always)]
fn center<L: NormalizeLanes>(x: L::N, w: u32) -> (L::N, L::N) {
    let half = L::splat(1i64 << (w - 1));
    let digit = L::sub(L::and(L::add(x, half), L::splat(low_mask(w))), half);
    (digit, L::srl(L::sub(x, digit), w))
}

/// `kept` plus the tie-to-even increment of its dropped part `half`, `sticky`.
#[inline(always)]
fn round_even<L: NormalizeLanes>(kept: L::N, half: L::N, sticky: L::N) -> L::N {
    L::add(kept, L::and(half, L::or(sticky, L::and(kept, L::splat(1)))))
}

#[inline(always)]
unsafe fn load_limb<L: NormalizeLanes>(a: Limbs<L::Word>, s: i64, i: usize) -> L::W {
    if (0..a.size as i64).contains(&s) {
        unsafe { L::load(a.ptr.add(s as usize * a.stride + i)) }
    } else {
        L::zero()
    }
}

#[inline(always)]
unsafe fn same_base_block<L: NormalizeLanes, const WHOLE: bool>(g: &SameBase, res: LimbsMut<i64>, a: Limbs<L::Word>, i: usize) {
    let b = g.base;
    let mut carry = L::zero();
    // Nothing below the boundary: nothing to round, it is a kept limb.
    let top = if WHOLE && g.padding == 0 && g.low == g.boundary {
        g.active
    } else {
        let mut sticky = L::splat(0);
        let mut half = L::splat(0);
        let mut s = g.low;
        while s > g.boundary {
            let (low, high) = split::<L, WHOLE>(unsafe { load_limb::<L>(a, s, i) }, carry, b, g.lsh);
            carry = high;
            if g.padding == 0 && s == g.boundary + 1 {
                half = L::srl(low, b - 1);
                sticky = L::or(sticky, L::ne_zero(L::and(low, L::splat(low_mask(b - 1)))));
            } else {
                sticky = L::or(sticky, L::ne_zero(low));
            }
            s -= 1;
        }
        let (low, high) = split::<L, WHOLE>(unsafe { load_limb::<L>(a, s, i) }, carry, b, g.lsh);
        let kept = if g.padding > 0 {
            let p = g.padding;
            half = L::and(L::srl(low, p - 1), L::splat(1));
            sticky = L::or(sticky, L::ne_zero(L::and(low, L::splat(low_mask(p - 1)))));
            L::srl(low, p)
        } else {
            low
        };
        let (digit, up) = center::<L>(round_even::<L>(kept, half, sticky), b - g.padding);
        carry = L::wide_add_n(high, up);
        unsafe { L::store(res.ptr.add((g.active - 1) * res.stride + i), L::shl(digit, g.padding)) };
        g.active - 1
    };
    for d in (0..top).rev() {
        let x = unsafe { load_limb::<L>(a, d as i64 + g.limb_offset, i) };
        let digit = if WHOLE {
            let t = L::wide_add(x, carry);
            carry = L::wide_sra(L::wide_add_n(t, L::splat(1 << (b - 1))), b);
            L::sra(L::shl(L::wide_low(t), 64 - b), 64 - b)
        } else {
            let (low, high) = split::<L, false>(x, carry, b, g.lsh);
            let (digit, up) = center::<L>(low, b);
            carry = L::wide_add_n(high, up);
            digit
        };
        unsafe { L::store(res.ptr.add(d * res.stride + i), digit) };
    }
    for d in g.active..res.size {
        unsafe { L::store(res.ptr.add(d * res.stride + i), L::splat(0)) };
    }
}

unsafe fn same_base<L, T, const WHOLE: bool>(g: &SameBase, res: LimbsMut<i64>, a: Limbs<L::Word>, len: usize)
where
    L: NormalizeLanes,
    T: NormalizeLanes<Word = L::Word>,
{
    let full = len - len % L::WIDTH;
    for i in (0..full).step_by(L::WIDTH) {
        unsafe { same_base_block::<L, WHOLE>(g, res, a, i) };
    }
    for i in full..len {
        unsafe { same_base_block::<T, WHOLE>(g, res, a, i) };
    }
}

/// The source integer read as unsigned `base`-bit digits, lowest first.
struct Bits<L: NormalizeLanes> {
    a: Limbs<L::Word>,
    i: usize,
    base: u32,
    carry: L::W,
    buf: L::N,
    avail: u32,
    next: usize,
}

impl<L: NormalizeLanes> Bits<L> {
    #[inline(always)]
    unsafe fn refill(&mut self) {
        let s = self.a.size as i64 - 1 - self.next as i64;
        let (digit, carry) = split::<L, false>(unsafe { load_limb::<L>(self.a, s, self.i) }, self.carry, self.base, 0);
        self.carry = carry;
        self.buf = digit;
        self.avail = self.base;
        self.next += 1;
    }

    /// The next `n` bits, `1 <= n <= 63`.
    #[inline(always)]
    unsafe fn take(&mut self, n: u32) -> L::N {
        let mut value = L::splat(0);
        let mut got = 0;
        while got < n {
            if self.avail == 0 {
                unsafe { self.refill() };
            }
            let t = (n - got).min(self.avail);
            value = L::or(value, L::shl(L::and(self.buf, L::splat(low_mask(t))), got));
            self.buf = L::srl(self.buf, t);
            self.avail -= t;
            got += t;
        }
        value
    }

    /// Drops `n` bits; 1 in every lane where one of them was set.
    #[inline(always)]
    unsafe fn skip(&mut self, mut n: u64) -> L::N {
        let mut sticky = L::splat(0);
        while n > 0 {
            if self.avail == 0 {
                unsafe { self.refill() };
            }
            let t = n.min(self.avail as u64) as u32;
            sticky = L::or(sticky, L::ne_zero(L::and(self.buf, L::splat(low_mask(t)))));
            self.buf = L::srl(self.buf, t);
            self.avail -= t;
            n -= t as u64;
        }
        sticky
    }
}

/// The next `width` output bits: zero bits entering below the source first, then source bits.
#[inline(always)]
unsafe fn window<L: NormalizeLanes>(zeros: &mut u64, width: u32, bits: &mut Bits<L>) -> L::N {
    let z = (*zeros).min(width as u64) as u32;
    *zeros -= z as u64;
    if z < width {
        L::shl(unsafe { bits.take(width - z) }, z)
    } else {
        L::splat(0)
    }
}

#[inline(always)]
unsafe fn cross_block<L: NormalizeLanes>(g: &Cross, res: LimbsMut<i64>, a: Limbs<L::Word>, i: usize) {
    let mut bits = Bits::<L> {
        a,
        i,
        base: g.a_base,
        carry: L::zero(),
        buf: L::splat(0),
        avail: 0,
        next: 0,
    };
    let (mut sticky, mut half, mut zeros) = (L::splat(0), L::splat(0), 0u64);
    if g.drop > 0 {
        sticky = unsafe { bits.skip(g.drop as u64 - 1) };
        half = unsafe { bits.take(1) };
    } else {
        zeros = g.drop.unsigned_abs();
    }
    let w = g.res_base - g.padding;
    let kept = unsafe { window(&mut zeros, w, &mut bits) };
    let (digit, mut up) = center::<L>(round_even::<L>(kept, half, sticky), w);
    unsafe { L::store(res.ptr.add((g.active - 1) * res.stride + i), L::shl(digit, g.padding)) };
    for d in (0..g.active - 1).rev() {
        let x = L::add(unsafe { window(&mut zeros, g.res_base, &mut bits) }, up);
        let (digit, carry) = center::<L>(x, g.res_base);
        up = carry;
        unsafe { L::store(res.ptr.add(d * res.stride + i), digit) };
    }
    for d in g.active..res.size {
        unsafe { L::store(res.ptr.add(d * res.stride + i), L::splat(0)) };
    }
}

/// Writes `canon(a * 2^offset, res_base2k, res_k, res.size)` over `len` coefficients, ties
/// to even, with lanes `L` and the last `len % L::WIDTH` coefficients with `T`.
/// Every source coefficient is bounded by `|a| <= 2^62` on i64 words and `|a| <= 2^126` on i128 words.
///
/// # Safety
/// Every limb of `res` and `a` is valid for `len` coefficients. `res` overlaps `a` only as the
/// same column (same `ptr`, `stride` and `size`) with `a_base2k == res_base2k` and
/// `offset < res_base2k`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn normalize_blocked<L, T>(
    res: LimbsMut<i64>,
    res_base2k: usize,
    res_k: usize,
    offset: i64,
    a: Limbs<L::Word>,
    a_base2k: usize,
    len: usize,
) where
    L: NormalizeLanes,
    T: NormalizeLanes<Word = L::Word>,
{
    const { assert!(T::WIDTH == 1) };
    assert!(res_k <= res.size * res_base2k, "normalize: precision exceeds the destination");
    let full = len - len % L::WIDTH;
    match plan(res_base2k, res_k, offset, a.size, a_base2k) {
        Plan::Zero => {
            for j in 0..res.size {
                unsafe { std::ptr::write_bytes(res.ptr.add(j * res.stride), 0, len) };
            }
        }
        Plan::Same(g) if g.whole() => unsafe { same_base::<L, T, true>(&g, res, a, len) },
        Plan::Same(g) => unsafe { same_base::<L, T, false>(&g, res, a, len) },
        Plan::Cross(g) => {
            for i in (0..full).step_by(L::WIDTH) {
                unsafe { cross_block::<L>(&g, res, a, i) };
            }
            for i in full..len {
                unsafe { cross_block::<T>(&g, res, a, i) };
            }
        }
        Plan::Exact => {
            for i in 0..len {
                let source = |j: usize| unsafe { a.ptr.add(j * a.stride + i).read() }.to_i128();
                let output = |j: usize, digit: i64| unsafe { res.ptr.add(j * res.stride + i).write(digit) };
                if size_of::<L::Word>() == size_of::<i64>() {
                    normalize_exact::<true, _, _>(source, a.size, a_base2k, res.size, res_base2k, res_k, offset, output);
                } else {
                    normalize_exact::<false, _, _>(source, a.size, a_base2k, res.size, res_base2k, res_k, offset, output);
                }
            }
        }
    }
}

/// Reads the two's-complement bits of the exact radix-weighted input integer.
pub(crate) struct NormalizationBits<F, const NARROW: bool> {
    source: F,
    left: usize,
    base2k: usize,
    carry: i128,
    digit: u64,
    available: usize,
}

impl<F: FnMut(usize) -> i128, const NARROW: bool> NormalizationBits<F, NARROW> {
    pub(crate) fn new(source: F, size: usize, base2k: usize) -> Self {
        Self {
            source,
            left: size,
            base2k,
            carry: 0,
            digit: 0,
            available: 0,
        }
    }

    fn refill(&mut self) {
        let mask = (1u64 << self.base2k) - 1;
        if self.left != 0 {
            self.left -= 1;
            let a = (self.source)(self.left);
            let low = (a as u64 & mask) + (self.carry as u64 & mask);
            self.carry = if NARROW {
                ((a as i64 >> self.base2k) + (self.carry as i64 >> self.base2k) + (low >> self.base2k) as i64) as i128
            } else {
                (a >> self.base2k) + (self.carry >> self.base2k) + (low >> self.base2k) as i128
            };
            self.digit = low & mask;
        } else {
            self.digit = self.carry as u64 & mask;
            self.carry = if NARROW {
                (self.carry as i64 >> self.base2k) as i128
            } else {
                self.carry >> self.base2k
            };
        }
        self.available = self.base2k;
    }

    fn read(&mut self, mut count: usize) -> u64 {
        let mut value = 0;
        let mut written = 0;
        while count != 0 {
            if self.available == 0 {
                self.refill();
            }
            let take = count.min(self.available);
            value |= (self.digit & ((1u64 << take) - 1)) << written;
            self.digit >>= take;
            self.available -= take;
            count -= take;
            written += take;
        }
        value
    }

    fn skip(&mut self, mut count: i128) -> bool {
        let mut sticky = false;
        while count != 0 {
            if self.available == 0 {
                if self.left == 0 && (self.carry == 0 || self.carry == -1) {
                    // Past the source every bit repeats the sign.
                    return sticky || self.carry == -1;
                }
                self.refill();
            }
            let take = count.min(self.available as i128) as usize;
            sticky |= self.digit & ((1u64 << take) - 1) != 0;
            self.digit >>= take;
            self.available -= take;
            count -= take as i128;
        }
        sticky
    }
}

/// Quantizes once at `res_k`, then emits centered digits with zero precision padding.
/// `NARROW` requires every source coefficient to fit in i64.
#[allow(clippy::too_many_arguments)]
pub(crate) fn normalize_exact<const NARROW: bool, F: FnMut(usize) -> i128, G: FnMut(usize, i64)>(
    source: F,
    a_size: usize,
    a_base2k: usize,
    res_size: usize,
    res_base2k: usize,
    res_k: usize,
    offset: i64,
    mut output: G,
) {
    if a_base2k > 63 || res_base2k > 63 {
        normalize_exact_wide(source, a_size, a_base2k, res_size, res_base2k, res_k, offset, output);
        return;
    }
    let (active_size, precision_padding) = if res_k == res_size * res_base2k {
        (res_size, 0)
    } else {
        let active_size = res_k.div_ceil(res_base2k);
        (active_size, active_size * res_base2k - res_k)
    };
    let mut bits = NormalizationBits::<_, NARROW>::new(source, a_size, a_base2k);
    let drop = a_size as i128 * a_base2k as i128 - res_k as i128 - offset as i128;
    let (mut half, mut sticky) = (0u64, false);
    let mut padding = 0i128;
    if drop > 0 {
        sticky = bits.skip(drop - 1);
        half = bits.read(1);
    } else {
        padding = -drop;
    }
    let mut carry = 0u64;
    // Only the lowest output window holds the rounding position.
    let mut first = true;
    let mut add_round = |raw: u64, carry: u64| -> u64 {
        if std::mem::take(&mut first) {
            raw + (half & (sticky as u64 | (raw & 1)))
        } else {
            raw + carry
        }
    };
    let mut full_size = active_size;
    if precision_padding != 0 {
        full_size -= 1;
        let width = res_base2k - precision_padding;
        let zeroes = padding.min(width as i128) as usize;
        padding -= zeroes as i128;
        let raw = bits.read(width - zeroes) << zeroes;
        let value = add_round(raw, carry);
        carry = (value + (1u64 << (width - 1))) >> width;
        output(
            full_size,
            ((value as i128 - ((carry as i128) << width)) << precision_padding) as i64,
        );
    }
    let half = 1u64 << (res_base2k - 1);
    for j in (0..full_size).rev() {
        let zeroes = padding.min(res_base2k as i128) as usize;
        padding -= zeroes as i128;
        let raw = bits.read(res_base2k - zeroes) << zeroes;
        let value = add_round(raw, carry);
        carry = (value + half) >> res_base2k;
        output(j, (value as i128 - ((carry as i128) << res_base2k)) as i64);
    }
    for j in active_size..res_size {
        output(j, 0);
    }
}

// Preserves the wider source radices and base-64 output supported by NTT.
#[allow(clippy::too_many_arguments)]
fn normalize_exact_wide<F: FnMut(usize) -> i128, G: FnMut(usize, i64)>(
    mut source: F,
    a_size: usize,
    a_base2k: usize,
    res_size: usize,
    res_base2k: usize,
    res_k: usize,
    offset: i64,
    mut output: G,
) {
    assert!((1..=127).contains(&a_base2k));
    assert!((1..=64).contains(&res_base2k));
    let (active_size, precision_padding) = if res_k == res_size * res_base2k {
        (res_size, 0)
    } else {
        let active_size = res_k.div_ceil(res_base2k);
        (active_size, active_size * res_base2k - res_k)
    };
    let mask = (1u128 << a_base2k) - 1;
    let mut left = a_size;
    let mut carry = 0i128;
    let mut pending = 0u128;
    let mut available = 0usize;
    // A discarding read returns 1 when one of the dropped bits was set.
    let mut read = |mut count: i128, discard: bool| {
        let mut result = 0u128;
        let mut written = 0usize;
        while count != 0 {
            if available == 0 {
                if discard && left == 0 && (carry == 0 || carry == -1) {
                    result |= (carry == -1) as u128;
                    break;
                }
                if left != 0 {
                    left -= 1;
                    let a = source(left);
                    let low = (a as u128 & mask) + (carry as u128 & mask);
                    carry = (a >> a_base2k) + (carry >> a_base2k) + (low >> a_base2k) as i128;
                    pending = low & mask;
                } else {
                    pending = carry as u128 & mask;
                    carry >>= a_base2k;
                }
                available = a_base2k;
            }
            let take = count.min(available as i128) as usize;
            if discard {
                result |= (pending & ((1u128 << take) - 1) != 0) as u128;
            } else {
                result |= (pending & ((1u128 << take) - 1)) << written;
                written += take;
            }
            pending >>= take;
            available -= take;
            count -= take as i128;
        }
        result
    };
    let drop = a_size as i128 * a_base2k as i128 - res_k as i128 - offset as i128;
    let (mut half, mut sticky) = (0u128, 0u128);
    let mut padding = 0i128;
    if drop > 0 {
        sticky = read(drop - 1, true);
        half = read(1, false);
    } else {
        padding = -drop;
    }
    let mut rounding = 0u128;
    let mut first = true;
    let mut add_round = |raw: u128, rounding: u128| -> u128 {
        if std::mem::take(&mut first) {
            raw + (half & (sticky | (raw & 1)))
        } else {
            raw + rounding
        }
    };
    let mut full_size = active_size;
    if precision_padding != 0 {
        full_size -= 1;
        let width = res_base2k - precision_padding;
        let zeroes = padding.min(width as i128) as usize;
        padding -= zeroes as i128;
        let value = add_round(read((width - zeroes) as i128, false) << zeroes, rounding);
        rounding = (value + (1u128 << (width - 1))) >> width;
        output(
            full_size,
            ((value as i128 - ((rounding as i128) << width)) << precision_padding) as i64,
        );
    }
    let half = 1u128 << (res_base2k - 1);
    for j in (0..full_size).rev() {
        let zeroes = padding.min(res_base2k as i128) as usize;
        padding -= zeroes as i128;
        let value = add_round(read((res_base2k - zeroes) as i128, false) << zeroes, rounding);
        rounding = (value + half) >> res_base2k;
        output(j, (value as i128 - ((rounding as i128) << res_base2k)) as i64);
    }
    for j in active_size..res_size {
        output(j, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_suite::znx_normalize::oracle;

    /// `f(res, a, n)` on `a` laid out limb by limb; returns the `size` output limbs per coefficient.
    fn layout<S: NormalizeWord>(a: &[Vec<S>], size: usize, f: impl FnOnce(LimbsMut<i64>, Limbs<S>, usize)) -> Vec<Vec<i64>> {
        let n = a.first().map_or(1, Vec::len);
        let src: Vec<S> = a.iter().flatten().copied().collect();
        let mut dst = vec![0x5a5a_i64; size * n];
        f(
            LimbsMut {
                ptr: dst.as_mut_ptr(),
                stride: n,
                size,
            },
            Limbs {
                ptr: src.as_ptr(),
                stride: n,
                size: a.len(),
            },
            n,
        );
        (0..n).map(|i| (0..size).map(|j| dst[j * n + i]).collect()).collect()
    }

    /// The portable driver over 11 coefficients (a full 8-lane block and a 3-coefficient tail).
    fn run<S: NormalizeWord>(a: &[Vec<S>], ka: usize, kr: usize, k: usize, size: usize, offset: i64) -> Vec<Vec<i64>> {
        layout(a, size, |res, src, n| unsafe {
            normalize_blocked::<PortableLanes<S, 8>, PortableLanes<S, 1>>(res, kr, k, offset, src, ka, n)
        })
    }

    fn check(a: &[Vec<i128>], ka: usize, kr: usize, k: usize, size: usize, offset: i64) {
        let column = |i: usize| a.iter().map(|limb| limb[i]).collect::<Vec<_>>();
        let have = run::<i128>(a, ka, kr, k, size, offset);
        for (i, got) in have.iter().enumerate() {
            assert_eq!(
                *got,
                oracle(&column(i), ka, kr, k, size, offset),
                "i128 a={:?} ka={ka} kr={kr} k={k} size={size} offset={offset}",
                column(i)
            );
        }
        if a.iter().flatten().all(|&x| i64::try_from(x).is_ok()) {
            let a64: Vec<Vec<i64>> = a.iter().map(|l| l.iter().map(|&x| x as i64).collect()).collect();
            let have = run::<i64>(&a64, ka, kr, k, size, offset);
            for (i, got) in have.iter().enumerate() {
                assert_eq!(
                    *got,
                    oracle(&column(i), ka, kr, k, size, offset),
                    "i64 a={:?} ka={ka} kr={kr} k={k} size={size} offset={offset}",
                    column(i)
                );
            }
        }
    }

    #[test]
    fn ties_go_to_even_at_every_dropped_width() {
        for b in [3usize, 4, 17, 50, 62] {
            for a_size in 1..=3usize {
                for drop in 1..(a_size * b).min(b + 4) {
                    let k = a_size * b - drop;
                    for q in [-3i128, -2, -1, 0, 1, 2, 3, 6, 7] {
                        // Below, at and above the tie `q + 1/2` at `drop` dropped bits.
                        let values: Vec<i128> = (0..11i128).map(|t| (q << drop) + (1i128 << (drop - 1)) + t % 3 - 1).collect();
                        let limbs: Vec<Vec<i128>> = (0..a_size)
                            .map(|j| {
                                let shift = (a_size - 1 - j) * b;
                                values
                                    .iter()
                                    .map(|&v| {
                                        if j == 0 {
                                            v >> shift
                                        } else {
                                            (v >> shift) & ((1i128 << b) - 1)
                                        }
                                    })
                                    .collect()
                            })
                            .collect();
                        for kr in [b, (b + 2).min(63), b.max(2) - 1] {
                            for size in [k.div_ceil(kr), k.div_ceil(kr) + 1] {
                                check(&limbs, b, kr, k, size, 0);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn matches_oracle_on_mixed_inputs() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for (ka, kr) in [
            (17, 17),
            (50, 50),
            (62, 62),
            (1, 1),
            (17, 19),
            (51, 50),
            (62, 1),
            (1, 62),
            (19, 17),
        ] {
            for a_size in [1usize, 3] {
                let limbs: Vec<Vec<i128>> = (0..a_size)
                    .map(|_| {
                        (0..11)
                            .map(|t| match t % 4 {
                                0 => -(1i128 << 62),
                                1 => 1i128 << 62,
                                _ => (next() as i64 >> 1) as i128,
                            })
                            .collect()
                    })
                    .collect();
                for size in [1usize, 3] {
                    let mut ks = vec![0, 1, kr - 1, kr, size * kr / 2, size * kr - 1, size * kr];
                    ks.sort_unstable();
                    ks.dedup();
                    for k in ks {
                        for offset in [
                            -(ka as i64) - 1,
                            -(ka as i64),
                            -1,
                            0,
                            1,
                            ka as i64,
                            -300,
                            300,
                            i64::MIN,
                            i64::MAX,
                        ] {
                            check(&limbs, ka, kr, k, size, offset);
                        }
                    }
                }
            }
        }
    }

    /// The driver against the forced split path, same base, offset a whole number of limbs.
    fn pin_whole_limbs<S: NormalizeWord>(a: &[Vec<S>], b: usize, k: usize, size: usize, offset: i64) {
        let Plan::Same(g) = plan(b, k, offset, a.len(), b) else {
            panic!("b={b} k={k} offset={offset}: not a same-base plan")
        };
        assert_eq!(g.whole(), (2..=62).contains(&b), "b={b} offset={offset}");
        let general = layout(a, size, |res, src, n| unsafe {
            same_base::<PortableLanes<S, 8>, PortableLanes<S, 1>, false>(&g, res, src, n)
        });
        assert_eq!(
            run(a, b, b, k, size, offset),
            general,
            "b={b} k={k} size={size} offset={offset}"
        );
    }

    #[test]
    fn whole_limb_fast_path_matches_split_path() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        for b in [1usize, 2, 62, 63] {
            let limbs: Vec<Vec<i64>> = (0..3)
                .map(|_| {
                    (0..11)
                        .map(|t| match t % 4 {
                            0 => -(1i64 << 62),
                            1 => 1i64 << 62,
                            _ => {
                                state ^= state << 13;
                                state ^= state >> 7;
                                state ^= state << 17;
                                state as i64 >> 1
                            }
                        })
                        .collect()
                })
                .collect();
            let wide: Vec<Vec<i128>> = limbs
                .iter()
                .map(|l| l.iter().map(|&x| ((x as i128) << 63) + x as i128).collect())
                .collect();
            for offset in [0, -(b as i64)] {
                for k in [3 * b, 3 * b - 1, b + 1] {
                    for size in [3, 4] {
                        pin_whole_limbs(&limbs, b, k, size, offset);
                        pin_whole_limbs(&wide, b, k, size, offset);
                    }
                }
            }
        }
    }

    #[test]
    fn wide_sources_match_oracle() {
        for (ka, kr) in [(50, 50), (62, 62), (52, 50)] {
            let limbs: Vec<Vec<i128>> = vec![
                (0..11).map(|t| (1i128 << 120) - 7 * t).collect(),
                (0..11).map(|t| -(1i128 << 100) + t).collect(),
            ];
            for k in [1, kr, 2 * kr - 1, 2 * kr] {
                for offset in [-(ka as i64), 0, 3] {
                    check(&limbs, ka, kr, k, 2, offset);
                }
            }
        }
    }
}
