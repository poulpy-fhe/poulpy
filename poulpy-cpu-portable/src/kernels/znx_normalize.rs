//! One normalization kernel per backend: `res = canon(a * 2^offset, res_base2k, res_k, res.size)`
//! with ties to even. The driver walks blocks of coefficients through every limb with the
//! carry, sticky and half bits in lanes, so rounding sees the exact dropped part.

use std::{
    hint::Locality,
    marker::PhantomData,
    simd::{
        Simd, Swizzle,
        cmp::{SimdPartialEq, SimdPartialOrd},
        num::{SimdInt, SimdUint},
    },
};

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
    /// Block lanes of the tiled walk in [`normalize_portable`].
    type Lanes: NormalizeLanes<Word = Self>;
    /// Block lanes of the blocked walk in [`normalize_portable`], which amortize its per-block driver work.
    type WideLanes: NormalizeLanes<Word = Self>;
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
    type Lanes = SimdLanes<8>;
    #[cfg(target_feature = "avx2")]
    type WideLanes = SimdLanes<16>;
    // Sixteen lanes spill SSE2's sixteen registers: the cross walk loses 6-8% (Zen 4 bench).
    #[cfg(not(target_feature = "avx2"))]
    type WideLanes = SimdLanes<8>;
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
    // AVX-512 lanes fill zmm registers: each half of a word is one zmm at width 8.
    #[cfg(target_feature = "avx512f")]
    type Lanes = SimdI128Lanes<8>;
    #[cfg(not(target_feature = "avx512f"))]
    type Lanes = SimdI128Lanes<4>;
    #[cfg(target_feature = "avx512f")]
    type WideLanes = SimdI128Lanes<16>;
    #[cfg(not(target_feature = "avx512f"))]
    type WideLanes = SimdI128Lanes<8>;
    #[inline(always)]
    fn low(self) -> i64 {
        self as i64
    }
    #[inline(always)]
    fn sra(self, k: u32) -> Self {
        // The driver shifts by less than 64, so the k >= 64 select drops out.
        self >> (k & 63)
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
    /// 1 in every lane where `x != 0`, else 0.
    fn ne_zero(x: Self::N) -> Self::N;
    fn wide_add(x: Self::W, y: Self::W) -> Self::W;
    /// `x + y` with `y` sign-extended.
    fn wide_add_n(x: Self::W, y: Self::N) -> Self::W;
    /// Arithmetic shift right by `1 <= k <= 63`.
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

/// i64 words in `std::simd` vectors of `W` lanes.
pub struct SimdLanes<const W: usize>;

impl<const W: usize> NormalizeLanes for SimdLanes<W> {
    const WIDTH: usize = W;
    type Word = i64;
    type N = Simd<i64, W>;
    type W = Simd<i64, W>;

    #[inline(always)]
    unsafe fn load(p: *const i64) -> Self::W {
        // SAFETY: the caller guarantees `W` readable i64 at `p`; the read is unaligned.
        Simd::from_array(unsafe { p.cast::<[i64; W]>().read_unaligned() })
    }
    #[inline(always)]
    unsafe fn store(p: *mut i64, x: Self::N) {
        // SAFETY: the caller guarantees `W` writable i64 at `p`; the write is unaligned.
        unsafe { p.cast::<[i64; W]>().write_unaligned(x.to_array()) }
    }
    #[inline(always)]
    fn zero() -> Self::W {
        Simd::splat(0)
    }
    #[inline(always)]
    fn splat(x: i64) -> Self::N {
        Simd::splat(x)
    }
    #[inline(always)]
    fn add(x: Self::N, y: Self::N) -> Self::N {
        x + y
    }
    #[inline(always)]
    fn sub(x: Self::N, y: Self::N) -> Self::N {
        x - y
    }
    #[inline(always)]
    fn and(x: Self::N, y: Self::N) -> Self::N {
        x & y
    }
    #[inline(always)]
    fn or(x: Self::N, y: Self::N) -> Self::N {
        x | y
    }
    #[inline(always)]
    fn shl(x: Self::N, k: u32) -> Self::N {
        x << k as i64
    }
    #[inline(always)]
    fn srl(x: Self::N, k: u32) -> Self::N {
        (x.cast::<u64>() >> k as u64).cast::<i64>()
    }
    #[inline(always)]
    fn ne_zero(x: Self::N) -> Self::N {
        -x.simd_ne(Simd::splat(0)).to_simd()
    }
    #[inline(always)]
    fn wide_add(x: Self::W, y: Self::W) -> Self::W {
        x + y
    }
    #[inline(always)]
    fn wide_add_n(x: Self::W, y: Self::N) -> Self::W {
        x + y
    }
    #[inline(always)]
    fn wide_sra(x: Self::W, k: u32) -> Self::W {
        x >> k as i64
    }
    #[inline(always)]
    fn wide_low(x: Self::W) -> Self::N {
        x
    }
}

/// i128 words as `(lo, hi)` halves in `std::simd` vectors of `W` lanes, each 4-coefficient chunk
/// in lane order [0, 2, 1, 3], coefficient order on AVX-512.
pub struct SimdI128Lanes<const W: usize>;

/// In-lane unpacks yield [0, 2, 1, 3] on 256-bit registers; AVX-512 gathers coefficient order and stores it unpermuted.
#[cfg(not(target_feature = "avx512f"))]
const CHUNK: [usize; 4] = [0, 2, 1, 3];
#[cfg(target_feature = "avx512f")]
const CHUNK: [usize; 4] = [0, 1, 2, 3];

/// The source half holding the low 64 bits of an i128.
const LO: usize = cfg!(target_endian = "big") as usize;

/// Lane `j` reads element `scale * c + half`, `c` the coefficient lane `j` holds.
const fn chunked<const W: usize>(scale: usize, half: usize) -> [usize; W] {
    let mut idx = [0; W];
    let mut j = 0;
    while j < W {
        idx[j] = scale * (4 * (j / 4) + CHUNK[j % 4]) + half;
        j += 1;
    }
    idx
}

struct Chunked<const SCALE: usize, const HALF: usize>;

impl<const W: usize, const SCALE: usize, const HALF: usize> Swizzle<W> for Chunked<SCALE, HALF> {
    const INDEX: [usize; W] = chunked::<W>(SCALE, HALF);
}

impl<const W: usize> NormalizeLanes for SimdI128Lanes<W> {
    const WIDTH: usize = W;
    type Word = i128;
    type N = Simd<i64, W>;
    type W = (Simd<u64, W>, Simd<i64, W>);

    #[inline(always)]
    unsafe fn load(p: *const i128) -> Self::W {
        let p = p.cast::<[u64; W]>();
        // SAFETY: the caller guarantees `W` readable i128 at `p`, two `[u64; W]`; the reads are unaligned.
        let (a, b) = unsafe {
            (
                Simd::from_array(p.read_unaligned()),
                Simd::from_array(p.add(1).read_unaligned()),
            )
        };
        (
            Chunked::<2, LO>::concat_swizzle(a, b),
            Chunked::<2, { 1 - LO }>::concat_swizzle(a, b).cast::<i64>(),
        )
    }
    #[inline(always)]
    unsafe fn store(p: *mut i64, x: Self::N) {
        // Both orders are their own inverse.
        let x = Chunked::<1, 0>::swizzle(x);
        // SAFETY: the caller guarantees `W` writable i64 at `p`.
        unsafe { SimdLanes::<W>::store(p, x) }
    }
    #[inline(always)]
    fn zero() -> Self::W {
        (Simd::splat(0), Simd::splat(0))
    }
    #[inline(always)]
    fn splat(x: i64) -> Self::N {
        Simd::splat(x)
    }
    #[inline(always)]
    fn add(x: Self::N, y: Self::N) -> Self::N {
        SimdLanes::<W>::add(x, y)
    }
    #[inline(always)]
    fn sub(x: Self::N, y: Self::N) -> Self::N {
        SimdLanes::<W>::sub(x, y)
    }
    #[inline(always)]
    fn and(x: Self::N, y: Self::N) -> Self::N {
        SimdLanes::<W>::and(x, y)
    }
    #[inline(always)]
    fn or(x: Self::N, y: Self::N) -> Self::N {
        SimdLanes::<W>::or(x, y)
    }
    #[inline(always)]
    fn shl(x: Self::N, k: u32) -> Self::N {
        SimdLanes::<W>::shl(x, k)
    }
    #[inline(always)]
    fn srl(x: Self::N, k: u32) -> Self::N {
        SimdLanes::<W>::srl(x, k)
    }
    #[inline(always)]
    fn ne_zero(x: Self::N) -> Self::N {
        SimdLanes::<W>::ne_zero(x)
    }
    #[inline(always)]
    fn wide_add(x: Self::W, y: Self::W) -> Self::W {
        let lo = x.0 + y.0;
        // The low half wrapped iff it carries; a true mask lane is -1.
        (lo, x.1 + y.1 - lo.simd_lt(x.0).to_simd())
    }
    #[inline(always)]
    fn wide_add_n(x: Self::W, y: Self::N) -> Self::W {
        let lo = x.0 + y.cast::<u64>();
        (lo, x.1 + y.simd_lt(Simd::splat(0)).to_simd() - lo.simd_lt(x.0).to_simd())
    }
    #[inline(always)]
    fn wide_sra(x: Self::W, k: u32) -> Self::W {
        let k = k as u64;
        ((x.0 >> k) | (x.1.cast::<u64>() << (64 - k)), x.1 >> k as i64)
    }
    #[inline(always)]
    fn wide_low(x: Self::W) -> Self::N {
        x.0.cast::<i64>()
    }
}

/// The normalization kernel of a backend. Every CPU backend word is `i64`; its big word is
/// `i64` or `i128`.
pub trait ZnxNormalize: Backend<ZnxWord = i64, BigWord: NormalizeWord> {
    /// `res = canon(a * 2^offset, res_base2k, res_k, res.size)` over `len` coefficients.
    ///
    /// # Safety
    /// See [`normalize_blocked_portable`].
    unsafe fn znx_normalize(
        res: LimbsMut<Self::ZnxWord>,
        res_base2k: usize,
        res_k: usize,
        offset: i64,
        a: Limbs<Self::ZnxWord>,
        a_base2k: usize,
        len: usize,
    ) {
        unsafe { normalize_portable(res, res_base2k, res_k, offset, a, a_base2k, len) }
    }

    /// [`Self::znx_normalize`] from big-word limbs.
    ///
    /// # Safety
    /// See [`normalize_blocked_portable`].
    unsafe fn znx_normalize_big(
        res: LimbsMut<Self::ZnxWord>,
        res_base2k: usize,
        res_k: usize,
        offset: i64,
        a: Limbs<Self::BigWord>,
        a_base2k: usize,
        len: usize,
    ) {
        unsafe { normalize_portable(res, res_base2k, res_k, offset, a, a_base2k, len) }
    }
}

/// [`normalize_blocked_portable`] with [`NormalizeWord::WideLanes`] where the walk is blocked
/// (cross radix, or at most `TILE_MIN_LINES` lines per block), else with [`NormalizeWord::Lanes`].
///
/// # Safety
/// See [`normalize_blocked_portable`].
pub unsafe fn normalize_portable<S: NormalizeWord>(
    res: LimbsMut<i64>,
    res_base2k: usize,
    res_k: usize,
    offset: i64,
    a: Limbs<S>,
    a_base2k: usize,
    len: usize,
) {
    unsafe {
        if a_base2k != res_base2k || a.size + res.size <= TILE_MIN_LINES {
            normalize_blocked_portable::<S::WideLanes, PortableLanes<S, 1>>(res, res_base2k, res_k, offset, a, a_base2k, len)
        } else {
            normalize_blocked_portable::<S::Lanes, PortableLanes<S, 1>>(res, res_base2k, res_k, offset, a, a_base2k, len)
        }
    }
}

/// Source limbs a plan may walk past the source; any farther and the source is below half a
/// destination unit, so the result is zero.
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
    /// With `|a| <= 2^62` and `2 <= b <= 62`, `t = a + carry` plus the rounding increment (at most
    /// `2^(b-2)`) and `2^(b-1)` stays at most `1.75 * 2^62 + 2`: `|carry| <= 2^62/(2^b - 1) + 1`.
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
            return Plan::Zero;
        };
        let low = boundary.max(a_size as i64 - 1);
        if low as i128 - boundary as i128 > a_size as i128 + MAX_EXTRA_LIMBS {
            return Plan::Zero;
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
        return Plan::Zero;
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

/// Blocks a tile walks limb by limb: 32 keeps the tile state within about 6 KiB and runs long per limb (Zen 4 bench).
const TILE: usize = 32;

/// Source plus destination lines per block past which the walk tiles: the 8 ways of an L1 set (crossover between 7 and 16 on Zen 4).
const TILE_MIN_LINES: usize = 8;

/// `K` blocks from coefficient `i`, limb by limb; `K > 1` prefetches the limb it reads next.
#[inline(always)]
unsafe fn same_base_tile<L: NormalizeLanes, const WHOLE: bool, const K: usize>(
    g: &SameBase,
    res: LimbsMut<i64>,
    a: Limbs<L::Word>,
    i: usize,
) {
    let b = g.base;
    let at = |k: usize| i + k * L::WIDTH;
    let load = |s: i64, j: usize| {
        if K > 1 && (1..=a.size as i64).contains(&s) {
            std::hint::prefetch_read(a.ptr.wrapping_add((s as usize - 1) * a.stride + j), Locality::L1);
        }
        unsafe { load_limb::<L>(a, s, j) }
    };
    let mut carry = [L::zero(); K];
    // Nothing below the boundary: nothing to round, it is a kept limb.
    let top = if WHOLE && g.padding == 0 && g.low == g.boundary {
        g.active
    } else {
        let mut sticky = [L::splat(0); K];
        let mut half = [L::splat(0); K];
        let mut s = g.low;
        while s > g.boundary {
            for k in 0..K {
                let (low, high) = split::<L, WHOLE>(load(s, at(k)), carry[k], b, g.lsh);
                carry[k] = high;
                if g.padding == 0 && s == g.boundary + 1 {
                    if WHOLE {
                        // The whole dropped limb; rounded at the boundary below.
                        half[k] = low;
                    } else {
                        half[k] = L::srl(low, b - 1);
                        sticky[k] = L::or(sticky[k], L::ne_zero(L::and(low, L::splat(low_mask(b - 1)))));
                    }
                } else {
                    sticky[k] = L::or(sticky[k], L::ne_zero(low));
                }
            }
            s -= 1;
        }
        for k in 0..K {
            let (half, sticky) = (half[k], sticky[k]);
            if WHOLE {
                let p = g.padding;
                let mut t = L::wide_add(load(s, at(k)), carry[k]);
                // The `e` dropped bits (`p`, else `b`) plus `2^(e-1) - 1 + (odd | sticky)` carry out iff ties to even rounds up.
                if p > 0 {
                    let odd = L::and(L::srl(L::wide_low(t), p), L::splat(1));
                    t = L::wide_add_n(t, L::add(L::or(odd, sticky), L::splat(low_mask(p - 1))));
                } else {
                    let odd = L::and(L::wide_low(t), L::splat(1));
                    let up = L::srl(L::add(half, L::add(L::or(odd, sticky), L::splat(low_mask(b - 1)))), b);
                    t = L::wide_add_n(t, up);
                }
                let u = L::wide_add_n(t, L::splat(1 << (b - 1)));
                carry[k] = L::wide_sra(u, b);
                let digit = L::sub(
                    L::and(L::wide_low(u), L::splat(low_mask(b) & !low_mask(p))),
                    L::splat(1 << (b - 1)),
                );
                unsafe { L::store(res.ptr.add((g.active - 1) * res.stride + at(k)), digit) };
            } else {
                let (low, high) = split::<L, false>(load(s, at(k)), carry[k], b, g.lsh);
                let (kept, half, sticky) = if g.padding > 0 {
                    let p = g.padding;
                    (
                        L::srl(low, p),
                        L::and(L::srl(low, p - 1), L::splat(1)),
                        L::or(sticky, L::ne_zero(L::and(low, L::splat(low_mask(p - 1))))),
                    )
                } else {
                    (low, half, sticky)
                };
                let (digit, up) = center::<L>(round_even::<L>(kept, half, sticky), b - g.padding);
                carry[k] = L::wide_add_n(high, up);
                unsafe { L::store(res.ptr.add((g.active - 1) * res.stride + at(k)), L::shl(digit, g.padding)) };
            }
        }
        g.active - 1
    };
    for d in (0..top).rev() {
        for (k, carry) in carry.iter_mut().enumerate() {
            let x = load(d as i64 + g.limb_offset, at(k));
            let digit = if WHOLE {
                let u = L::wide_add_n(L::wide_add(x, *carry), L::splat(1 << (b - 1)));
                *carry = L::wide_sra(u, b);
                L::sub(L::and(L::wide_low(u), L::splat(low_mask(b))), L::splat(1 << (b - 1)))
            } else {
                let (low, high) = split::<L, false>(x, *carry, b, g.lsh);
                let (digit, up) = center::<L>(low, b);
                *carry = L::wide_add_n(high, up);
                digit
            };
            unsafe { L::store(res.ptr.add(d * res.stride + at(k)), digit) };
        }
    }
    for d in g.active..res.size {
        for k in 0..K {
            unsafe { L::store(res.ptr.add(d * res.stride + at(k)), L::splat(0)) };
        }
    }
}

// Inlined, the tile bodies cost the cross walk beside them 1-3%.
#[inline(never)]
unsafe fn same_base<L, T, const WHOLE: bool>(g: &SameBase, res: LimbsMut<i64>, a: Limbs<L::Word>, len: usize)
where
    L: NormalizeLanes,
    T: NormalizeLanes<Word = L::Word>,
{
    let full = len - len % L::WIDTH;
    // A block's limb lines, a power-of-two stride apart, share one L1 set: past its 8 ways, walk tiles.
    let tiled = if a.size + res.size > TILE_MIN_LINES {
        full - full % (TILE * L::WIDTH)
    } else {
        0
    };
    for i in (0..tiled).step_by(TILE * L::WIDTH) {
        unsafe { same_base_tile::<L, WHOLE, TILE>(g, res, a, i) };
    }
    for i in (tiled..full).step_by(L::WIDTH) {
        unsafe { same_base_tile::<L, WHOLE, 1>(g, res, a, i) };
    }
    for i in full..len {
        unsafe { same_base_tile::<T, WHOLE, 1>(g, res, a, i) };
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
pub unsafe fn normalize_blocked_portable<L, T>(
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
    assert!((1..=64).contains(&res_base2k), "normalize: destination radix out of range");
    assert!((1..=127).contains(&a_base2k), "normalize: source radix out of range");
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
                normalize_exact(source, a.size, a_base2k, res.size, res_base2k, res_k, offset, output);
            }
        }
    }
}

/// Scalar path for radixes past 63: quantizes once at `res_k`, then emits centered digits with
/// zero precision padding.
#[allow(clippy::too_many_arguments)]
fn normalize_exact<F: FnMut(usize) -> i128, G: FnMut(usize, i64)>(
    mut source: F,
    a_size: usize,
    a_base2k: usize,
    res_size: usize,
    res_base2k: usize,
    res_k: usize,
    offset: i64,
    mut output: G,
) {
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

    /// The portable driver over the coefficients of the given limbs.
    fn run<S: NormalizeWord>(a: &[Vec<S>], ka: usize, kr: usize, k: usize, size: usize, offset: i64) -> Vec<Vec<i64>> {
        layout(a, size, |res, src, n| unsafe {
            normalize_portable(res, kr, k, offset, src, ka, n)
        })
    }

    /// The driver against the forced split path, same base, offset a whole number of limbs.
    fn pin_whole_limbs<S: NormalizeWord>(a: &[Vec<S>], b: usize, k: usize, size: usize, offset: i64) {
        let Plan::Same(g) = plan(b, k, offset, a.len(), b) else {
            panic!("b={b} k={k} offset={offset}: not a same-base plan")
        };
        assert_eq!(g.whole(), (2..=62).contains(&b), "b={b} offset={offset}");
        let general = layout(a, size, |res, src, n| unsafe {
            same_base::<S::Lanes, PortableLanes<S, 1>, false>(&g, res, src, n)
        });
        assert_eq!(
            run(a, b, b, k, size, offset),
            general,
            "b={b} k={k} size={size} offset={offset}"
        );
    }

    /// `size` limbs of `n` coefficients: the `±2^62` input bounds and random words between.
    fn sample(size: usize, n: usize, state: &mut u64) -> Vec<Vec<i64>> {
        (0..size)
            .map(|_| {
                (0..n)
                    .map(|t| match t % 4 {
                        0 => -(1i64 << 62),
                        1 => 1i64 << 62,
                        _ => {
                            *state ^= *state << 13;
                            *state ^= *state >> 7;
                            *state ^= *state << 17;
                            *state as i64 >> 1
                        }
                    })
                    .collect()
            })
            .collect()
    }

    fn widen(limbs: &[Vec<i64>]) -> Vec<Vec<i128>> {
        limbs
            .iter()
            .map(|l| l.iter().map(|&x| ((x as i128) << 63) + x as i128).collect())
            .collect()
    }

    #[test]
    fn whole_limb_fast_path_matches_split_path() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        for b in [1usize, 2, 62, 63] {
            let limbs = sample(3, 11, &mut state);
            let wide = widen(&limbs);
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

    /// Tiles, leftover blocks and the tail against the driver one coefficient at a time, and in place.
    #[test]
    fn tiled_walk_matches_single_coefficients() {
        fn single<S: NormalizeWord>(a: &[Vec<S>], size: usize, b: usize, k: usize, offset: i64) -> Vec<Vec<i64>> {
            layout(a, size, |res, src, n| {
                for i in 0..n {
                    let res = LimbsMut {
                        ptr: res.ptr.wrapping_add(i),
                        ..res
                    };
                    let src = Limbs {
                        ptr: src.ptr.wrapping_add(i),
                        ..src
                    };
                    unsafe { normalize_blocked_portable::<S::Lanes, PortableLanes<S, 1>>(res, b, k, offset, src, b, 1) };
                }
            })
        }
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        // At least two tiles, three leftover blocks and a tail at every lane width.
        let n = 2 * TILE * 8 + 3 * 8 + 5;
        let size = 6;
        let limbs = sample(size, n, &mut state);
        let wide = widen(&limbs);
        for b in [17usize, 50, 62] {
            for k in [size * b, size * b - 5, 2 * b + 1] {
                for offset in [0, 1, b as i64, -(b as i64), -(b as i64) - 1] {
                    let want = single(&limbs, size, b, k, offset);
                    assert_eq!(run(&limbs, b, b, k, size, offset), want, "b={b} k={k} offset={offset}");
                    assert_eq!(
                        run(&wide, b, b, k, size, offset),
                        single(&wide, size, b, k, offset),
                        "wide b={b} k={k} offset={offset}"
                    );
                    // In place needs `offset < b`.
                    if offset >= b as i64 {
                        continue;
                    }
                    let mut buf = limbs.concat();
                    let ptr = buf.as_mut_ptr();
                    unsafe {
                        normalize_blocked_portable::<<i64 as NormalizeWord>::Lanes, PortableLanes<i64, 1>>(
                            LimbsMut { ptr, stride: n, size },
                            b,
                            k,
                            offset,
                            Limbs {
                                ptr: ptr.cast_const(),
                                stride: n,
                                size,
                            },
                            b,
                            n,
                        )
                    };
                    let in_place: Vec<Vec<i64>> = (0..n).map(|i| (0..size).map(|j| buf[j * n + i]).collect()).collect();
                    assert_eq!(in_place, want, "in place b={b} k={k} offset={offset}");
                }
            }
            // Fewer destination limbs than source limbs, still past the line threshold.
            let k = 4 * b;
            assert_eq!(
                run(&limbs, b, b, k, 4, b as i64),
                single(&limbs, 4, b, k, b as i64),
                "size 4 b={b}"
            );
        }
    }
}
