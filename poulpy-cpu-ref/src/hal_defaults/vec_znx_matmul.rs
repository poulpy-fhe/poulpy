//! Reference implementation of the packed coefficient-matrix product
//! `res[res_col+j][out] = sum_in U[out,in] * a[a_col+j][in]`.
//!
//! Restructured as a GEMM: `U` and the `i64` digits of `A` are decomposed into
//! `W`-wide balanced pieces (`W` = the compile-time `CoeffMatrix` entry bound,
//! 16/32), the `cols` right-hand-side columns run in parallel threads, and each
//! output column is base-`2^base2k` normalized once. The per-piece `W x W`
//! dot is the only backend-specialized piece (the [`GemmKernel`] trait); this
//! file holds the portable scalar kernels, AVX2/AVX-512 live in the backend
//! crates and reuse [`matmul_gemm`].

use poulpy_hal::{
    api::{
        ModuleN, ScratchArenaTakeBasic, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddAssignBackend, VecZnxBigBytesOf,
        VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxNormalizeAssignBackend, VecZnxNormalizeTmpBytes,
    },
    layouts::{
        Backend, HostDataMut, HostDataRef, Module, ScratchArena, ScratchOwned, VecZnx, VecZnxBackendMut, VecZnxBackendRef,
        VecZnxBigToBackendRef, VecZnxToBackendMut, VecZnxToBackendRef, ZnxView, ZnxViewMut,
    },
};

/// Number of `W`-wide pieces needed to hold a `bits`-wide signed integer.
#[inline]
pub fn n_pieces(bits: u32, w: u32) -> usize {
    (bits.div_ceil(w)).max(1) as usize
}

/// Per-piece dot kernel: `U`/`A` digits are balanced base-`2^W` decompositions,
/// `Elt` is the `W`-wide prepared piece type, `Acc` the accumulator
/// (`BE::ScalarBig`). Backends provide SIMD `dot`; `prep` is shared scalar.
pub trait GemmKernel {
    type Acc: Copy + Default;
    type Elt: Copy + Default + Send + Sync + 'static;
    /// Piece width in bits.
    const W: u32;
    /// `U`-digit piece count baked in at compile time. Selects the kernel
    /// specialization: kernels with `UP = 1` skip the outer `i` loop entirely.
    const UP: usize;

    /// `A`-digit piece count for a `base2k`-bit digit.
    fn a_pieces(base2k: usize) -> usize {
        n_pieces(base2k as u32, Self::W)
    }
    /// Balanced base-`2^W` decomposition of `x` into `n` pieces.
    fn prep(x: i64, n: usize, out: &mut [Self::Elt]);
    /// `sum_{i<UP, j<ap} (sum_in u[i][in]*a[j][in]) << (W*(i+j))`.
    /// `u`/`a` are piece-major: piece `p` is the slice `[p*rows_in .. ][..rows_in]`.
    fn dot(u: &[Self::Elt], a: &[Self::Elt], ap: usize, rows_in: usize) -> Self::Acc;
}

/// Generates a scalar [`GemmKernel`] with a compile-time `UP`. SIMD backends
/// provide their own `dot`. `$up = 1` produces a single-U-piece kernel (no
/// outer `i` loop); `$up = 2` does an unrolled two-piece body.
macro_rules! ref_gemm_kernel {
    ($name:ident, $elt:ty, $acc:ty, $w:expr, $up:expr) => {
        pub struct $name;
        impl GemmKernel for $name {
            type Acc = $acc;
            type Elt = $elt;
            const W: u32 = $w;
            const UP: usize = $up;

            #[inline]
            fn prep(x: i64, n: usize, out: &mut [$elt]) {
                let mut r: i64 = x;
                for k in 0..n {
                    if k + 1 == n {
                        out[k] = r as $elt;
                    } else {
                        let hi: i64 = (r + (1i64 << ($w - 1))) >> $w;
                        out[k] = (r - (hi << $w)) as $elt;
                        r = hi;
                    }
                }
            }

            #[inline]
            fn dot(u: &[$elt], a: &[$elt], ap: usize, rows_in: usize) -> $acc {
                let mut acc: $acc = 0;
                // `for i in 0..$up` is a literal const so LLVM unrolls trivially.
                for i in 0..$up {
                    let us = &u[i * rows_in..i * rows_in + rows_in];
                    for j in 0..ap {
                        let as_ = &a[j * rows_in..j * rows_in + rows_in];
                        let mut p: $acc = 0;
                        for k in 0..rows_in {
                            p = p.wrapping_add((us[k] as $acc).wrapping_mul(as_[k] as $acc));
                        }
                        acc = acc.wrapping_add(p << ($w as u32 * (i + j) as u32));
                    }
                }
                acc
            }
        }
    };
}

ref_gemm_kernel!(RefK16I64, i16, i64, 16, 1);
ref_gemm_kernel!(RefK16I128, i16, i128, 16, 1);
ref_gemm_kernel!(RefK32I64, i32, i64, 32, 1);
ref_gemm_kernel!(RefK32I128S, i32, i128, 32, 1);
ref_gemm_kernel!(RefK32I128D, i32, i128, 32, 2);

/// Selects the scalar kernels and the per-backend max piece width for a
/// `ScalarBig` accumulator. FFT64 (`i64`) caps `W` at 32 — its entries always
/// fit `i32` by FFT precision; NTT (`i128`) allows 64-bit `U`.
///
/// `K32D` is the two-`U`-piece variant, used by NTT when `BU = i64`. FFT
/// never reaches it (its `w` is clamped to 32 -> `U` fits one piece), so its
/// `K32D` alias is identical to `K32S` and is dead code on fft64.
pub trait ScalarKernels: Copy + Default {
    type K16: GemmKernel<Acc = Self>;
    type K32S: GemmKernel<Acc = Self>;
    type K32D: GemmKernel<Acc = Self>;
    const MAX_W: u32;
}
impl ScalarKernels for i64 {
    type K16 = RefK16I64;
    type K32S = RefK32I64;
    type K32D = RefK32I64;
    const MAX_W: u32 = 32;
}
impl ScalarKernels for i128 {
    type K16 = RefK16I128;
    type K32S = RefK32I128S;
    type K32D = RefK32I128D;
    const MAX_W: u32 = 64;
}

#[doc(hidden)]
pub trait VecZnxMatMulDefault<BE: Backend>: Backend
where
    BE::OwnedBuf: HostDataMut,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    fn vec_znx_matmul_tmp_bytes_default(
        module: &Module<BE>,
        _rows_in: usize,
        _rows_out: usize,
        _cols: usize,
        res_size: usize,
        _u_size: usize,
        a_size: usize,
    ) -> usize
    where
        Module<BE>: ModuleN + VecZnxBigBytesOf + VecZnxBigNormalizeTmpBytes + VecZnxNormalizeTmpBytes,
    {
        let n = module.n();
        let big = module.bytes_of_vec_znx_big(1, a_size);
        let tmp = VecZnx::<Vec<u8>>::bytes_of(n, 1, res_size);
        let row_acc = VecZnx::<Vec<u8>>::bytes_of(n, 1, res_size);
        let op = module
            .vec_znx_big_normalize_tmp_bytes()
            .max(module.vec_znx_normalize_tmp_bytes());
        big + tmp + row_acc + op + 4 * (BE::SCRATCH_ALIGN - 1)
    }

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_matmul_default(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        res_base2k: usize,
        u: &VecZnxBackendRef<'_, BE>,
        u_base2k: usize,
        u_bound_bits: u32,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        cols: usize,
        a_base2k: usize,
        rows_in: usize,
        rows_out: usize,
        _scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>: ModuleN
            + VecZnxBigBytesOf
            + VecZnxBigNormalizeTmpBytes
            + VecZnxBigNormalize<BE>
            + VecZnxAddAssignBackend<BE>
            + VecZnxNormalizeAssignBackend<BE>
            + VecZnxNormalizeTmpBytes,
        ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
        BE::ScalarBig: ScalarKernels,
    {
        // Portable scalar kernels; dispatch on the (clamped) U-bound width.
        let w = u_bound_bits.min(<BE::ScalarBig as ScalarKernels>::MAX_W);
        if w <= 16 {
            matmul_gemm::<BE, <BE::ScalarBig as ScalarKernels>::K16>(
                module, res, res_col, res_base2k, u, u_base2k, a, a_col, cols, a_base2k, rows_in, rows_out,
            );
        } else if w <= 32 {
            matmul_gemm::<BE, <BE::ScalarBig as ScalarKernels>::K32S>(
                module, res, res_col, res_base2k, u, u_base2k, a, a_col, cols, a_base2k, rows_in, rows_out,
            );
        } else {
            matmul_gemm::<BE, <BE::ScalarBig as ScalarKernels>::K32D>(
                module, res, res_col, res_base2k, u, u_base2k, a, a_col, cols, a_base2k, rows_in, rows_out,
            );
        }
    }
}

impl<BE: Backend> VecZnxMatMulDefault<BE> for BE
where
    BE::OwnedBuf: HostDataMut,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
}

/// Shared GEMM orchestration: decomposes `U`/`A` into `K::W`-wide pieces,
/// threads over RHS columns, normalizes each output column once. Backend
/// crates call this with their own [`GemmKernel`] (scalar / AVX2 / AVX-512).
#[allow(clippy::too_many_arguments)]
pub fn matmul_gemm<BE, K>(
    module: &Module<BE>,
    res: &mut VecZnxBackendMut<'_, BE>,
    res_col: usize,
    res_base2k: usize,
    u: &VecZnxBackendRef<'_, BE>,
    u_base2k: usize,
    a: &VecZnxBackendRef<'_, BE>,
    a_col: usize,
    cols: usize,
    a_base2k: usize,
    rows_in: usize,
    rows_out: usize,
) where
    BE: Backend,
    BE::OwnedBuf: HostDataMut,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
    Module<BE>: ModuleN
        + VecZnxBigBytesOf
        + VecZnxBigNormalizeTmpBytes
        + VecZnxBigNormalize<BE>
        + VecZnxAddAssignBackend<BE>
        + VecZnxNormalizeAssignBackend<BE>
        + VecZnxNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    K: GemmKernel<Acc = BE::ScalarBig>,
{
    assert!(rows_in <= module.n(), "vec_znx_matmul: rows_in exceeds ring degree");
    assert!(rows_out <= module.n(), "vec_znx_matmul: rows_out exceeds ring degree");
    assert!(rows_in <= u.n(), "vec_znx_matmul: rows_in exceeds U ring degree");
    assert!(rows_in <= a.n(), "vec_znx_matmul: rows_in exceeds input ring degree");
    assert!(rows_out <= u.cols(), "vec_znx_matmul: rows_out exceeds U columns");
    assert!(rows_out <= res.n(), "vec_znx_matmul: rows_out exceeds result ring degree");
    assert!(
        res_col + cols <= res.cols(),
        "vec_znx_matmul: result column range out of bounds"
    );
    assert!(a_col + cols <= a.cols(), "vec_znx_matmul: input column range out of bounds");

    let u_size = u.size();
    let a_size = a.size();
    let res_size = res.size();
    let n = module.n();

    let up = K::UP;
    let ap = K::a_pieces(a_base2k);
    debug_assert!(up <= 4 && ap <= 4);

    // Prepare U once: uprep[((u_limb*rows_out + out) * up + piece) * rows_in + in].
    let u_stride = rows_in * up;
    let mut uprep: Vec<K::Elt> = vec![K::Elt::default(); u_size * rows_out * u_stride];
    {
        let mut tmp = [K::Elt::default(); 4];
        for u_limb in 0..u_size {
            for out in 0..rows_out {
                let src = &u.at(out, u_limb)[..rows_in];
                let base = (u_limb * rows_out + out) * u_stride;
                for (in_, &x) in src.iter().enumerate() {
                    K::prep(x, up, &mut tmp[..up]);
                    for piece in 0..up {
                        uprep[base + piece * rows_in + in_] = tmp[piece];
                    }
                }
            }
        }
    }
    let uprep: &[K::Elt] = &uprep;

    let araw: &[i64] = a.raw();
    let a_n = a.n();
    let a_cols = a.cols();
    let a_stride = rows_in * ap;

    let stride_c = res_size * rows_out;
    let mut out_buf: Vec<i64> = vec![0i64; cols * stride_c];

    let per_thread_bytes = <BE as VecZnxMatMulDefault<BE>>::vec_znx_matmul_tmp_bytes_default(
        module, rows_in, rows_out, cols, res_size, u_size, a_size,
    );

    // One normalized RHS column -> its slice of `out`. `aprep` is per-thread
    // scratch of `a_size * a_stride` holding the column's piece-major digits.
    let column = move |module: &Module<BE>, j: usize, out: &mut [i64], aprep: &mut [K::Elt], so: &mut ScratchOwned<BE>| {
        let scratch = so.borrow();
        let (mut big, scratch_1) = scratch.take_vec_znx_big_scratch(module, 1, a_size);
        let (mut tmp, scratch_2) = scratch_1.take_vec_znx_scratch(n, 1, res_size);
        let (mut row_acc, mut scratch_3) = scratch_2.take_vec_znx_scratch(n, 1, res_size);

        // Decompose this RHS column's digits once (reused over all u_limb).
        {
            let mut piece_buf = [K::Elt::default(); 4];
            for a_limb in 0..a_size {
                let base = a_limb * a_stride;
                for in_ in 0..rows_in {
                    let x = araw[a_n * (a_limb * a_cols + a_col + j) + in_];
                    K::prep(x, ap, &mut piece_buf[..ap]);
                    for piece in 0..ap {
                        aprep[base + piece * rows_in + in_] = piece_buf[piece];
                    }
                }
            }
        }

        row_acc.raw_mut().fill(0);

        for u_limb in 0..u_size {
            for a_limb in 0..a_size {
                let ac: &[K::Elt] = &aprep[a_limb * a_stride..a_limb * a_stride + a_stride];
                let dst = big.at_mut(0, a_limb);
                for (out_row, slot) in dst.iter_mut().enumerate().take(rows_out) {
                    let ub = (u_limb * rows_out + out_row) * u_stride;
                    *slot = K::dot(&uprep[ub..ub + u_stride], ac, ap, rows_in);
                }
            }

            {
                let big_ref = big.to_backend_ref();
                let mut tmp_mut = tmp.to_backend_mut();
                module.vec_znx_big_normalize(
                    &mut tmp_mut,
                    res_base2k,
                    (u_limb * u_base2k) as i64,
                    0,
                    &big_ref,
                    a_base2k,
                    0,
                    &mut scratch_3.borrow(),
                );
            }

            {
                let tmp_ref = tmp.to_backend_ref();
                let mut row_acc_mut = row_acc.to_backend_mut();
                module.vec_znx_add_assign_backend(&mut row_acc_mut, 0, &tmp_ref, 0);
            }
        }

        {
            let mut row_acc_mut = row_acc.to_backend_mut();
            module.vec_znx_normalize_assign_backend(res_base2k, &mut row_acc_mut, 0, &mut scratch_3.borrow());
        }

        for limb in 0..res_size {
            out[limb * rows_out..limb * rows_out + rows_out].copy_from_slice(&row_acc.at(0, limb)[..rows_out]);
        }
    };

    let max_threads = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1);
    let work = (rows_in as u128) * (rows_out as u128) * (cols as u128);
    let nthreads = if work < (1u128 << 22) { 1 } else { max_threads.clamp(1, cols) };

    if nthreads <= 1 {
        let mut so = <ScratchOwned<BE> as ScratchOwnedAlloc<BE>>::alloc(per_thread_bytes);
        let mut aprep: Vec<K::Elt> = vec![K::Elt::default(); a_size * a_stride];
        for j in 0..cols {
            let s0 = j * stride_c;
            column(module, j, &mut out_buf[s0..s0 + stride_c], &mut aprep, &mut so);
        }
    } else {
        let base = cols / nthreads;
        let rem = cols % nthreads;
        let mut tasks: Vec<(usize, usize, &mut [i64])> = Vec::with_capacity(nthreads);
        let mut rest: &mut [i64] = out_buf.as_mut_slice();
        let mut c0 = 0;
        for t in 0..nthreads {
            let len = base + usize::from(t < rem);
            if len == 0 {
                continue;
            }
            let (head, tail) = rest.split_at_mut(len * stride_c);
            tasks.push((c0, len, head));
            rest = tail;
            c0 += len;
        }

        let column_ref = &column;
        std::thread::scope(|scope| {
            for (start, len, slab) in tasks {
                scope.spawn(move || {
                    let mut so = <ScratchOwned<BE> as ScratchOwnedAlloc<BE>>::alloc(per_thread_bytes);
                    let mut aprep: Vec<K::Elt> = vec![K::Elt::default(); a_size * a_stride];
                    for jj in 0..len {
                        let j = start + jj;
                        let s0 = jj * stride_c;
                        column_ref(module, j, &mut slab[s0..s0 + stride_c], &mut aprep, &mut so);
                    }
                });
            }
        });
    }

    for j in 0..cols {
        let s0 = j * stride_c;
        for limb in 0..res_size {
            let src = &out_buf[s0 + limb * rows_out..s0 + limb * rows_out + rows_out];
            res.at_mut(res_col + j, limb)[..rows_out].copy_from_slice(src);
        }
    }
}
