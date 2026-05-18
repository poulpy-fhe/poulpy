//! Reference (scalar) implementation of the packed coefficient-matrix product
//! `res[res_col+j][out] = sum_in U[out,in] * a[a_col+j][in]`.
//!
//! Restructured as a GEMM: `U` is prepared once into a backend-specific
//! per-digit representation, the `cols` right-hand-side columns are processed
//! in parallel (disjoint threads), and each output column is base-`2^base2k`
//! normalized once (not once per scalar). The per-digit dot product is the
//! only backend-specialized piece, exposed via [`GemmScalar`]; this file
//! provides the portable scalar implementation, AVX2/AVX-512 live in the
//! respective backend crates.

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

/// Backend accumulator with a prepared per-digit input representation.
///
/// - `i64` (FFT64): digits fit `i32` (FFT mantissa keeps `base2k` small), so
///   the prepared form is a single `i32` and the exact product is `i32 x i32
///   -> i64`. `MAX_BASE2K = 30`.
/// - `i128` (NTT): `base2k` can be large (e.g. 50), so each digit is split
///   into balanced signed halves `[lo, hi]` (`v = hi*2^s + lo`, both `i32`)
///   and the product is reassembled in `i128`. `MAX_BASE2K = 52`.
///
/// The scalar `dot` is the portable reference; backend crates override the
/// kernel for SIMD.
pub trait GemmScalar: Copy {
    type Prep: Copy + Default + Send + Sync + 'static;
    const MAX_BASE2K: usize;

    fn split_shift(base2k: usize) -> u32;
    fn prep(x: i64, s: u32) -> Self::Prep;
    fn dot(u: &[Self::Prep], a: &[Self::Prep], s: u32) -> Self;
}

impl GemmScalar for i64 {
    type Prep = i32;
    const MAX_BASE2K: usize = 30;

    #[inline]
    fn split_shift(_base2k: usize) -> u32 {
        0
    }

    #[inline]
    fn prep(x: i64, _s: u32) -> i32 {
        x as i32
    }

    #[inline]
    fn dot(u: &[i32], a: &[i32], _s: u32) -> i64 {
        let mut acc: i64 = 0;
        for (&ui, &ai) in u.iter().zip(a.iter()) {
            acc = acc.wrapping_add((ui as i64).wrapping_mul(ai as i64));
        }
        acc
    }
}

impl GemmScalar for i128 {
    type Prep = [i32; 2];
    const MAX_BASE2K: usize = 52;

    #[inline]
    fn split_shift(base2k: usize) -> u32 {
        let s = base2k.div_ceil(2).max(base2k.saturating_sub(31));
        s.clamp(1, 31) as u32
    }

    #[inline]
    fn prep(x: i64, s: u32) -> [i32; 2] {
        let hi: i64 = (x + (1i64 << (s - 1))) >> s;
        let lo: i64 = x - (hi << s);
        [lo as i32, hi as i32]
    }

    #[inline]
    fn dot(u: &[[i32; 2]], a: &[[i32; 2]], s: u32) -> i128 {
        let mut acc: i128 = 0;
        for (&ud, &ad) in u.iter().zip(a.iter()) {
            let uv: i128 = (ud[1] as i128) * (1i128 << s) + ud[0] as i128;
            let av: i128 = (ad[1] as i128) * (1i128 << s) + ad[0] as i128;
            acc += uv * av;
        }
        acc
    }
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
        // Per-thread requirement: one big accumulator column + tmp + row_acc +
        // op scratch. Threads allocate their own arenas internally; this is the
        // size the caller must still satisfy for the API contract.
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
        BE::ScalarBig: GemmScalar,
    {
        // Reference path: portable scalar kernel.
        matmul_gemm::<BE, _>(
            module,
            res,
            res_col,
            res_base2k,
            u,
            u_base2k,
            a,
            a_col,
            cols,
            a_base2k,
            rows_in,
            rows_out,
            <BE::ScalarBig as GemmScalar>::dot,
        );
    }
}

impl<BE: Backend> VecZnxMatMulDefault<BE> for BE
where
    BE::OwnedBuf: HostDataMut,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
}

/// Shared GEMM orchestration: prepares `U`, threads over RHS columns, and
/// normalizes each output column once. Backend crates call this with their own
/// `dot` kernel (scalar / AVX2 / AVX-512). `prep`/`split_shift` come from
/// [`GemmScalar`] and are backend-agnostic scalar conversions.
#[allow(clippy::too_many_arguments)]
pub fn matmul_gemm<BE, D>(
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
    dot: D,
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
    BE::ScalarBig: GemmScalar,
    D: Fn(&[<BE::ScalarBig as GemmScalar>::Prep], &[<BE::ScalarBig as GemmScalar>::Prep], u32) -> BE::ScalarBig + Sync,
{
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

        let max_base2k = u_base2k.max(a_base2k);
        assert!(
            max_base2k <= <BE::ScalarBig as GemmScalar>::MAX_BASE2K,
            "vec_znx_matmul: base2k {} exceeds GEMM limit {} for this backend",
            max_base2k,
            <BE::ScalarBig as GemmScalar>::MAX_BASE2K
        );
        let s: u32 = <BE::ScalarBig as GemmScalar>::split_shift(max_base2k);

        let u_size = u.size();
        let a_size = a.size();
        let res_size = res.size();
        let n = module.n();

        type Prep<BE> = <<BE as Backend>::ScalarBig as GemmScalar>::Prep;

        // Prepare U once: uprep[(u_limb*rows_out + out) * rows_in + in].
        let mut uprep: Vec<Prep<BE>> = vec![<BE::ScalarBig as GemmScalar>::prep(0, s); u_size * rows_out * rows_in];
        for u_limb in 0..u_size {
            for out in 0..rows_out {
                let src = &u.at(out, u_limb)[..rows_in];
                let base = (u_limb * rows_out + out) * rows_in;
                for (d, &x) in uprep[base..base + rows_in].iter_mut().zip(src.iter()) {
                    *d = <BE::ScalarBig as GemmScalar>::prep(x, s);
                }
            }
        }
        let uprep: &[Prep<BE>] = &uprep;

        // Plain integer view of A (Send + Sync) so threads can read it.
        let araw: &[i64] = a.raw();
        let a_n = a.n();
        let a_cols = a.cols();

        let stride_c = res_size * rows_out;
        let mut out_buf: Vec<i64> = vec![0i64; cols * stride_c];

        let per_thread_bytes = <BE as VecZnxMatMulDefault<BE>>::vec_znx_matmul_tmp_bytes_default(
            module, rows_in, rows_out, cols, res_size, u_size, a_size,
        );

        // One normalized RHS column -> its slice of `out`.
        let column = move |module: &Module<BE>, j: usize, out: &mut [i64], aprep: &mut [Prep<BE>], so: &mut ScratchOwned<BE>| {
            let scratch = so.borrow();
            let (mut big, scratch_1) = scratch.take_vec_znx_big_scratch(module, 1, a_size);
            let (mut tmp, scratch_2) = scratch_1.take_vec_znx_scratch(n, 1, res_size);
            let (mut row_acc, mut scratch_3) = scratch_2.take_vec_znx_scratch(n, 1, res_size);

            // Prepare this RHS column's digits once (reused over all u_limb).
            for a_limb in 0..a_size {
                let abase = a_n * (a_limb * a_cols + a_col + j);
                let src = &araw[abase..abase + rows_in];
                let dst = &mut aprep[a_limb * rows_in..a_limb * rows_in + rows_in];
                for (d, &x) in dst.iter_mut().zip(src.iter()) {
                    *d = <BE::ScalarBig as GemmScalar>::prep(x, s);
                }
            }

            row_acc.raw_mut().fill(0);

            for u_limb in 0..u_size {
                for a_limb in 0..a_size {
                    let ac: &[Prep<BE>] = &aprep[a_limb * rows_in..a_limb * rows_in + rows_in];
                    let dst = big.at_mut(0, a_limb);
                    for (out_row, slot) in dst.iter_mut().enumerate().take(rows_out) {
                        let base = (u_limb * rows_out + out_row) * rows_in;
                        *slot = dot(&uprep[base..base + rows_in], ac, s);
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
        let nthreads = if work < (1u128 << 22) {
            1
        } else {
            max_threads.clamp(1, cols)
        };

        if nthreads <= 1 {
            let mut so = <ScratchOwned<BE> as ScratchOwnedAlloc<BE>>::alloc(per_thread_bytes);
            let mut aprep: Vec<Prep<BE>> = vec![<BE::ScalarBig as GemmScalar>::prep(0, s); a_size * rows_in];
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
                        let mut aprep: Vec<Prep<BE>> = vec![<BE::ScalarBig as GemmScalar>::prep(0, s); a_size * rows_in];
                        for jj in 0..len {
                            let j = start + jj;
                            let s0 = jj * stride_c;
                            column_ref(module, j, &mut slab[s0..s0 + stride_c], &mut aprep, &mut so);
                        }
                    });
                }
            });
        }

        // Scatter normalized columns into the result.
        for j in 0..cols {
            let s0 = j * stride_c;
            for limb in 0..res_size {
                let src = &out_buf[s0 + limb * rows_out..s0 + limb * rows_out + rows_out];
                res.at_mut(res_col + j, limb)[..rows_out].copy_from_slice(src);
            }
        }
    }
}
