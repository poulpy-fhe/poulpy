//! The negacyclic NTT over the four primes, on lazy 32-bit lanes.
//!
//! Forward: Cooley-Tukey butterflies `(x, y) -> (x + w y, x - w y)` on lanes in
//! `[0, 4q)`, natural-order input, bit-reversed-order output
//! `out[i] = a(psi^(2 bitrev(i) + 1))`. Inverse: the stages in the opposite
//! order with Gentleman-Sande butterflies `(u, v) -> (u + v, (u - v) w^-1)` on
//! lanes in `[0, 2q)`, `n^-1` folded into stage 0. A product by a twiddle is
//! a Shoup product in `[0, 2q)` for any 32-bit multiplicand. Buffers hold
//! canonical residues outside a transform.
//!
//! A launch runs `mt` consecutive stages. One workgroup owns `2^log_w` tiles of
//! `2^mt` lanes that the stages close over, and runs them as up to three
//! register sub-passes that hand their lanes over through workgroup shared
//! memory, one barrier apart. A transform is one launch over whole polynomials
//! up to degree `2^SINGLE_LAUNCH_LOG_N`, and two launches above: stages
//! `0..m` on strided columns and stages `m..log_n` on contiguous blocks.
//!
//! A launch transforms a batch of limbs. Polynomial `p` of the batch is prime
//! `p % 4` of limb `p / 4`; its plane starts `limb * stride + (p % 4) * n`
//! words after the base of its buffer.

use cubecl::prelude::*;

use super::{
    DftColumn,
    primes::{Consts, mul_lazy, sel4},
    tables::Tables,
};
use crate::{
    runtime::{Binding, GpuBuf, GpuRuntime, Label, launch},
    znx::Column,
};

/// Largest degree transformed by a single launch: a tile must fit the
/// shared memory and the thread count of one workgroup.
const SINGLE_LAUNCH_LOG_N: u32 = 12;
/// Most stages of one launch.
const MAX_TILE_STAGES: u32 = 12;
const MAX_SUBS: usize = 3;

/// Lanes stay lazy between the launches of a transform.
const STORE_LAZY: u32 = 0;
/// The last forward launch: from `[0, 4q)` to `[0, q)`.
const STORE_FORWARD: u32 = 1;
/// The last inverse launch: from the `[0, 2q)` of the scaling stage to `[0, q)`.
const STORE_INVERSE: u32 = 2;

/// The first sub-pass reads lanes of the transform.
const SOURCE_RESIDUES: u32 = 0;
/// The first sub-pass reads `i64` coefficients and reduces them modulo the plane's prime.
const SOURCE_LIMBS: u32 = 1;
/// [`SOURCE_LIMBS`] of the conjugate-invariant ring: the coefficients go
/// through the change to the negacyclic basis, `b_j = alpha^-j (c_j - i c_{n-j})`.
const SOURCE_LIMBS_INVARIANT: u32 = 2;

/// Compile-time geometry of a launch.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct Tile {
    log_n: u32,
    /// A tile has `2^mt` lanes and closes over `mt` stages.
    mt: u32,
    /// `true`: a strided column (lanes `2^(log_n - mt)` apart, stages `0..mt`);
    /// `false`: a contiguous block (stages `log_n - mt..log_n`).
    strided: bool,
    /// log2 of the tiles one workgroup owns.
    log_w: u32,
    /// log2 of the threads per tile.
    log_tpt: u32,
    nsubs: u32,
    /// Stages of each sub-pass, in processing order.
    sub_k: [u32; MAX_SUBS],
    /// First in-tile stage of each sub-pass, in processing order.
    sub_a: [u32; MAX_SUBS],
}

impl Tile {
    /// `mt` stages as `ceil(mt / 4)` sub-passes of near-equal radix, with as
    /// many tiles per workgroup as reach 128 threads (8 tiles at least) within
    /// 16 KiB of shared memory and 1024 threads.
    fn new(log_n: u32, mt: u32, strided: bool, inverse: bool) -> Self {
        assert!((1..=MAX_TILE_STAGES).contains(&mt) && mt <= log_n);
        let nsubs = mt.div_ceil(4);
        let radix = |i: u32| mt / nsubs + (i < mt % nsubs) as u32;
        let log_tpt = mt - radix(nsubs - 1);
        let mut log_w = 7u32.saturating_sub(log_tpt).max(3);
        while log_w > 0 && (mt + log_w > 12 || log_tpt + log_w > 10) {
            log_w -= 1;
        }
        assert!(mt + log_w <= 12 && log_tpt + log_w <= 10);
        let (mut sub_k, mut sub_a) = ([0; MAX_SUBS], [0; MAX_SUBS]);
        for step in 0..nsubs {
            // Sub-passes in stage order are 0, 1, ..; the inverse runs them backwards.
            let i = if inverse { nsubs - 1 - step } else { step };
            sub_k[step as usize] = radix(i);
            sub_a[step as usize] = (0..i).map(radix).sum();
        }
        Self {
            log_n,
            mt,
            strided,
            log_w,
            log_tpt,
            nsubs,
            sub_k,
            sub_a,
        }
    }
}

/// `x mod q` in `[0, q)` for the `i64` of words `(lo, hi)`.
///
/// `x + 2^63 = lo + hb 2^32` with `hb = hi ^ 2^31`, so `x = lo + hb r + c63`
/// modulo `q`; that sum is below `2^60` and one Barrett step with
/// `mu = floor(2^61 / q)` leaves it in `[0, 2q)`.
#[cube]
fn load_limb(lo: u32, hi: u32, q: u32, r: u32, c63: u32, mu: u32) -> u32 {
    let hb = hi ^ 0x8000_0000u32;
    let t = (lo as u64) + (hb as u64) * (r as u64) + (c63 as u64);
    let quotient = ((t >> 28u64) * (mu as u64)) >> 33u64;
    let d = u32::cast_from(t - quotient * (q as u64));
    min(d, d - q)
}

/// Forward butterfly on lanes in `[0, 4q)`; `wq` is the twiddle's companion.
#[cube]
fn ct(x: u32, y: u32, wq: u32, w: u32, q: u32) -> (u32, u32) {
    let q2 = q + q;
    let t = y * w - y.mul_hi(wq) * q;
    let x1 = min(x, x - q2);
    (x1 + t, x1 + q2 - t)
}

/// Inverse butterfly on lanes in `[0, 2q)`.
#[cube]
fn gs(u: u32, v: u32, wq: u32, w: u32, q: u32) -> (u32, u32) {
    let q2 = q + q;
    let s = u + v;
    let d = u - v + q2;
    (min(s, s - q2), d * w - d.mul_hi(wq) * q)
}

/// Stage 0 of the inverse: the sum is multiplied by `n^-1` (`nq`, `nw`), the
/// difference by the twiddle that has `n^-1` folded in. Both leave in `[0, 2q)`.
#[cube]
fn gs_scaled(u: u32, v: u32, nq: u32, nw: u32, wq: u32, w: u32, q: u32) -> (u32, u32) {
    let s = u + v;
    let d = u - v + q + q;
    (s * nw - s.mul_hi(nq) * q, d * w - d.mul_hi(wq) * q)
}

/// `k` forward stages over the `2^k` lanes of `x`. Step `d` pairs lanes
/// `2^(k-1-d)` apart in `2^d` blocks; `tb` is the twiddle entry of the thread's
/// block at step 0, so block `b` of step `d` uses entry `tb * 2^d + b` of the
/// plane starting at `toff`.
#[cube]
fn ct_net(x: &mut Sequence<u32>, tw: &[u32], toff: usize, tb: usize, q: u32, #[comptime] k: u32) {
    #[unroll]
    for d in 0..k {
        let half = 1usize << (k - 1 - d);
        let mult = 1usize << d;
        #[unroll]
        for b in 0..mult {
            let wq = tw[toff + tb * mult + b];
            let w = wq.mul_hi(q) + 1u32;
            #[unroll]
            for j in 0..half {
                let i0 = b * 2 * half + j;
                let i1 = i0 + half;
                let r = ct(*x.index(i0), *x.index(i1), wq, w, q);
                x.insert(i0, r.0);
                x.insert(i1, r.1);
            }
        }
    }
}

/// The same `k` stages in the opposite order, with inverse butterflies. With
/// `scale` the last one is stage 0 of the transform (`tb` is then 1).
#[cube]
fn gs_net(x: &mut Sequence<u32>, tw: &[u32], toff: usize, tb: usize, q: u32, #[comptime] k: u32, #[comptime] scale: bool) {
    #[unroll]
    for e in 0..k {
        let d = k - 1 - e;
        let half = 1usize << e;
        let mult = 1usize << d;
        #[unroll]
        for b in 0..mult {
            let wq = tw[toff + tb * mult + b];
            let w = wq.mul_hi(q) + 1u32;
            if scale && d == 0u32 {
                let nq = tw[toff];
                let nw = nq.mul_hi(q) + 1u32;
                #[unroll]
                for j in 0..half {
                    let i0 = b * 2 * half + j;
                    let i1 = i0 + half;
                    let r = gs_scaled(*x.index(i0), *x.index(i1), nq, nw, wq, w, q);
                    x.insert(i0, r.0);
                    x.insert(i1, r.1);
                }
            } else {
                #[unroll]
                for j in 0..half {
                    let i0 = b * 2 * half + j;
                    let i1 = i0 + half;
                    let r = gs(*x.index(i0), *x.index(i1), wq, w, q);
                    x.insert(i0, r.0);
                    x.insert(i1, r.1);
                }
            }
        }
    }
}

/// One launch of a transform; see the module documentation. Thread `u` of
/// workgroup `grp` serves tile `grp * 2^log_w + w` of the batch, tiles being
/// numbered polynomial after polynomial. A sub-pass of `kj` stages keeps
/// `2^(mt - kj)` threads per tile busy; each lane of a tile is read and
/// written by exactly one of them, so the barrier is the only synchronization.
/// Threads past the batch take the barriers and touch no memory.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments, unused_assignments)]
fn tile_pass(
    dst: &mut [u32],
    src: &[u32],
    tw: &[u32],
    polys: u32,
    dst_base: u32,
    dst_stride: u32,
    src_base: u32,
    src_stride: u32,
    #[comptime] c: Consts,
    #[comptime] g: Tile,
    #[comptime] inverse: bool,
    #[comptime] scale: bool,
    #[comptime] mode: u32,
    #[comptime] source: u32,
) {
    let mt = g.mt as usize;
    let ln = g.log_n as usize;
    let rest = ln - mt;
    let lw = g.log_w as usize;
    let ltp = g.log_tpt as usize;
    let mut sh = Shared::<[u32]>::new_slice(1usize << (mt + lw));
    let u = UNIT_POS as usize;
    // Threads of adjacent addresses are adjacent: tile-major for contiguous
    // tiles, tile-minor for strided ones.
    let mut w = u >> ltp;
    let mut r = u - (w << ltp);
    if g.strided {
        r = u >> lw;
        w = u - (r << lw);
    }
    let tile = (CUBE_POS << lw) + w;
    let p = tile >> rest;
    let ti = tile - (p << rest);
    let active = p < polys as usize;
    let limb = p >> 2usize;
    let pi = u32::cast_from(p) & 3u32;
    let q = sel4(pi, c.q);
    let plane = usize::cast_from(pi) << ln;
    // First lane of the tile in its plane, log2 of the distance between two
    // lanes, and the twiddle entry of the tile at its first stage.
    let mut first = ti << mt;
    let mut lgs = 0usize;
    let mut tile_tb = (1usize << rest) + ti;
    if g.strided {
        first = ti;
        lgs = rest;
        tile_tb = 1usize;
    }
    let dbase = dst_base as usize + limb * (dst_stride as usize) + plane + first;
    let sbase = src_base as usize + limb * (src_stride as usize);
    let mut toff = plane;
    if inverse {
        toff += 4usize << ln;
    }
    let mut load_r = q;
    let mut load_c63 = q;
    let mut load_mu = q;
    let mut unit = q;
    let mut unit_q = q;
    if source != SOURCE_RESIDUES {
        load_r = sel4(pi, c.r);
        load_c63 = sel4(pi, c.c63);
        load_mu = sel4(pi, c.mu61);
    }
    if source == SOURCE_LIMBS_INVARIANT {
        unit = sel4(pi, c.i);
        unit_q = sel4(pi, c.i_shoup);
    }
    let shared = w << mt;
    #[unroll]
    for step in 0..g.nsubs {
        let kj = g.sub_k[step as usize];
        let a = g.sub_a[step as usize] as usize;
        // log2 of the smallest distance of this sub-pass, in tile coordinates.
        let ltl = mt - a - kj as usize;
        let lanes = 1usize << kj;
        if step > 0u32 {
            sync_cube();
        }
        if active && r < (1usize << (mt - kj as usize)) {
            let i0 = r >> ltl;
            let e0 = (i0 << (ltl + kj as usize)) + (r - (i0 << ltl));
            let tb = tile_tb * (1usize << a) + i0;
            let mut x = Sequence::<u32>::new();
            #[unroll]
            for l in 0..lanes {
                let e = e0 + (l << ltl);
                if step > 0u32 {
                    x.push(sh[shared + e]);
                } else if source == SOURCE_RESIDUES {
                    x.push(src[sbase + plane + first + (e << lgs)]);
                } else {
                    let j = first + (e << lgs);
                    let direct = load_limb(src[sbase + 2 * j], src[sbase + 2 * j + 1], q, load_r, load_c63, load_mu);
                    if source == SOURCE_LIMBS_INVARIANT {
                        let n = 1usize << ln;
                        let other = (n - j) & (n - 1);
                        let reflected = load_limb(
                            src[sbase + 2 * other],
                            src[sbase + 2 * other + 1],
                            q,
                            load_r,
                            load_c63,
                            load_mu,
                        );
                        // `i * reflected` is in `[0, 2q)`, the difference in `(0, 3q)`.
                        let turned = reflected * unit - reflected.mul_hi(unit_q) * q;
                        let mixed = mul_lazy(direct + q + q - turned, tw[(8usize << ln) + plane + j], q);
                        x.push(select(j == 0usize, direct, mixed));
                    } else {
                        x.push(direct);
                    }
                }
            }
            if inverse {
                gs_net(&mut x, tw, toff, tb, q, kj, scale && step == g.nsubs - 1u32);
            } else {
                ct_net(&mut x, tw, toff, tb, q, kj);
            }
            #[unroll]
            for l in 0..lanes {
                let e = e0 + (l << ltl);
                let lane = *x.index(l);
                if step < g.nsubs - 1u32 {
                    sh[shared + e] = lane;
                } else if mode == STORE_FORWARD {
                    let q2 = q + q;
                    let reduced = min(lane, lane - q2);
                    dst[dbase + (e << lgs)] = min(reduced, reduced - q);
                } else if mode == STORE_INVERSE {
                    dst[dbase + (e << lgs)] = min(lane, lane - q);
                } else {
                    dst[dbase + (e << lgs)] = lane;
                }
            }
        }
    }
}

/// Where the planes of a batch are in a buffer.
struct Span<'a> {
    binding: &'a Binding,
    base: u32,
    stride: u32,
}

#[allow(clippy::too_many_arguments)]
fn launch<RT: GpuRuntime>(
    table: &GpuBuf<RT>,
    tile: Tile,
    inverse: bool,
    scale: bool,
    mode: u32,
    source: u32,
    limbs: usize,
    dst: Span<'_>,
    src: Span<'_>,
) {
    let polys = 4 * limbs;
    let tiles = polys << (tile.log_n - tile.mt);
    let table = table.view().binding().expect("transform tables hold storage");
    let label = match (inverse, tile.mt == tile.log_n, tile.strided) {
        (false, true, _) => Label::NttForwardSingle,
        (false, false, true) => Label::NttForwardStrided,
        (false, false, false) => Label::NttForwardContiguous,
        (true, true, _) => Label::NttInverseSingle,
        (true, false, true) => Label::NttInverseStrided,
        (true, false, false) => Label::NttInverseContiguous,
    };
    launch!(
        RT, label, (polys << tile.log_n) * 8, tile_pass, (tiles, 1 << tile.log_w, 1 << (tile.log_w + tile.log_tpt));
        [
            dst.binding => u32,
            src.binding => u32,
            table => u32,
        ];
        [
            u32::try_from(polys).expect("batch beyond the 32-bit polynomial range"),
            dst.base,
            dst.stride,
            src.base,
            src.stride,
        ];
        [Consts::get(), tile, inverse, scale, mode, source]
    );
}

/// The launches of a transform of degree `2^log_n`, in order.
fn passes(log_n: u32, inverse: bool) -> Vec<Tile> {
    if log_n <= SINGLE_LAUNCH_LOG_N {
        return vec![Tile::new(log_n, log_n, false, inverse)];
    }
    let m = log_n.div_ceil(2);
    if inverse {
        vec![Tile::new(log_n, log_n - m, false, true), Tile::new(log_n, m, true, true)]
    } else {
        vec![Tile::new(log_n, m, true, false), Tile::new(log_n, log_n - m, false, false)]
    }
}

/// `res[j] = dft(a[j])` for the first `limbs` limbs: the `i64` coefficients of
/// `a` are reduced by the first launch.
pub(crate) fn forward<RT: GpuRuntime>(tables: &Tables<RT>, res: &DftColumn, a: &Column, limbs: usize) {
    assert_eq!(res.n, a.n, "transform between different degrees");
    assert!(a.words == 1 && limbs <= res.size.min(a.size));
    if limbs == 0 {
        return;
    }
    let log_n = res.n.ilog2();
    let table = tables.of(res.n);
    let passes = passes(log_n, false);
    let load = if tables.invariant() {
        SOURCE_LIMBS_INVARIANT
    } else {
        SOURCE_LIMBS
    };
    for (index, &tile) in passes.iter().enumerate() {
        let last = index + 1 == passes.len();
        let dst = Span {
            binding: &res.binding,
            base: res.base,
            stride: res.stride,
        };
        let (src, source) = if index == 0 {
            let base = a.base.checked_mul(2).expect("source beyond the 32-bit word range");
            let stride = a.stride.checked_mul(2).expect("source beyond the 32-bit word range");
            (
                Span {
                    binding: &a.binding,
                    base,
                    stride,
                },
                load,
            )
        } else {
            (
                Span {
                    binding: &res.binding,
                    base: res.base,
                    stride: res.stride,
                },
                SOURCE_RESIDUES,
            )
        };
        let mode = if last { STORE_FORWARD } else { STORE_LAZY };
        launch::<RT>(table, tile, false, false, mode, source, limbs, dst, src);
    }
}

/// `dst[j] = idft(src[j])` in the negacyclic basis for the first `limbs`
/// limbs. `dst` may be `src` itself; otherwise `src` is left unchanged.
pub(crate) fn inverse<RT: GpuRuntime>(tables: &Tables<RT>, dst: &DftColumn, src: &DftColumn, limbs: usize) {
    assert_eq!(dst.n, src.n, "transform between different degrees");
    assert!(limbs <= dst.size.min(src.size));
    if limbs == 0 {
        return;
    }
    let table = tables.of(dst.n);
    let passes = passes(dst.n.ilog2(), true);
    for (index, &tile) in passes.iter().enumerate() {
        let last = index + 1 == passes.len();
        let from = if index == 0 { src } else { dst };
        launch::<RT>(
            table,
            tile,
            true,
            last,
            if last { STORE_INVERSE } else { STORE_LAZY },
            SOURCE_RESIDUES,
            limbs,
            Span {
                binding: &dst.binding,
                base: dst.base,
                stride: dst.stride,
            },
            Span {
                binding: &from.binding,
                base: from.base,
                stride: from.stride,
            },
        );
    }
}
