//! NTT-domain vector polynomial operations for the NTT120 backend.
//!
//! This module provides:
//!
//! - The [`NttModuleHandle`] trait, which exposes precomputed NTT/iNTT
//!   tables and multiply–accumulate metadata from a module handle.
//! - Forward (`ntt120_vec_znx_dft_apply`) and inverse
//!   (`ntt120_vec_znx_idft_apply`, `ntt120_vec_znx_idft_apply_tmpa`) DFT
//!   operations.
//! - Component-wise DFT-domain arithmetic (add, sub, negate, copy, zero).
//!
//! # Scalar layout
//!
//! `VecZnxDft<_, NTT120Ref>` stores [`Q120bScalar`] values (32 bytes each).
//! Each `Q120bScalar` holds four `u64` CRT residues for one ring coefficient.
//! A `bytemuck::cast_slice` converts a `&[Q120bScalar]` limb slice to
//! `&[u64]` for use with the primitive NTT arithmetic functions.
//!
//! # Prime set
//!
//! All arithmetic is hardcoded to [`Primes30`] (the spqlios-arithmetic
//! default, Q ≈ 2^120).  Generalisation to `Primes29` / `Primes31`
//! is future work.

use bytemuck::{cast_slice, cast_slice_mut};

use crate::{
    layouts::{
        Backend, Data, Module, VecZnxBig, VecZnxBigToMut, VecZnxDft, VecZnxDftToMut, VecZnxDftToRef, VecZnxToRef, ZnxInfos,
        ZnxView, ZnxViewMut,
    },
    reference::ntt120::{
        NttAdd, NttAddAssign, NttCopy, NttDFTExecute, NttFromZnx64, NttNegate, NttNegateAssign, NttSub, NttSubAssign,
        NttSubNegateAssign, NttToZnx128, NttZero,
        mat_vec::{BbbMeta, BbcMeta},
        ntt::{NttTable, NttTableInv, intt_ref},
        primes::{PrimeSet, Primes30},
        types::Q120bScalar,
    },
};

// ──────────────────────────────────────────────────────────────────────────────
// NttModuleHandle trait + NttHandleProvider blanket impl
// ──────────────────────────────────────────────────────────────────────────────

// TODO(ntt120): Associate PrimeSet with NttModuleHandle (add associated type)
//               to enable Primes29/Primes31 dispatch through the public API.

/// Access to the precomputed NTT/iNTT tables and lazy-accumulation metadata
/// stored inside a `Module<B>` handle.
///
/// Automatically implemented for any `Module<B>` whose `B::Handle` implements
/// [`NttHandleProvider`].  Backend crates (e.g. `poulpy-cpu-ref`) implement
/// `NttHandleProvider` for their concrete handle type; they do *not* implement
/// this trait directly (which would violate the orphan rule).
///
/// <!-- DOCUMENTED EXCEPTION: Primes30 hardcoded for spqlios compatibility.
///   Generalisation path: add `type PrimeSet: PrimeSet` as an associated type here,
///   then parameterise NttTable/NttTableInv/BbcMeta accordingly. -->
pub trait NttModuleHandle {
    /// Precomputed forward NTT twiddle table (Primes30, size `n`).
    fn get_ntt_table(&self) -> &NttTable<Primes30>;
    /// Precomputed inverse NTT twiddle table (Primes30, size `n`).
    fn get_intt_table(&self) -> &NttTableInv<Primes30>;
    /// Precomputed metadata for `q120b × q120c` lazy multiply–accumulate.
    fn get_bbc_meta(&self) -> &BbcMeta<Primes30>;
    /// Precomputed metadata for `q120b × q120b` lazy multiply–accumulate.
    fn get_bbb_meta(&self) -> &BbbMeta<Primes30>;
}

/// Implemented by backend `Handle` types that store NTT/iNTT tables and BBC
/// metadata.
///
/// Implement this trait for your concrete handle struct (e.g. `NTT120RefHandle`)
/// in the backend crate.  A blanket `impl NttModuleHandle for Module<B>` is
/// provided here in `poulpy-hal`, so no orphan-rule violation occurs.
///
/// # Safety
///
/// Implementors must ensure the returned references are valid for the lifetime
/// of `&self` and that the tables were fully initialised before first use.
///
/// The blanket `impl<B> NttModuleHandle for Module<B>` assumes the handle is
/// fully initialised before `Module::new()` returns.  This invariant is
/// established by the module defaults (or a backend override).  There is no
/// runtime check in release builds.
pub unsafe trait NttHandleProvider {
    /// Returns a reference to the forward NTT twiddle table.
    fn get_ntt_table(&self) -> &NttTable<Primes30>;
    /// Returns a reference to the inverse NTT twiddle table.
    fn get_intt_table(&self) -> &NttTableInv<Primes30>;
    /// Returns a reference to the `q120b × q120c` lazy multiply–accumulate metadata.
    fn get_bbc_meta(&self) -> &BbcMeta<Primes30>;
    /// Returns a reference to the `q120b × q120b` lazy multiply–accumulate metadata.
    fn get_bbb_meta(&self) -> &BbbMeta<Primes30>;
}

/// Construct NTT120 backend handles for [`Module::new`](crate::api::ModuleNew::new).
///
/// # Safety
///
/// Implementors must return a fully initialized handle for the requested `n`.
/// The handle is boxed and stored inside the `Module`, so it must be safe to
/// drop via [`Backend::destroy`](crate::layouts::Backend::destroy).
pub unsafe trait NttHandleFactory: Sized {
    /// Builds a fully initialized handle for ring dimension `n`.
    fn create_ntt_handle(n: usize) -> Self;

    /// Optional runtime capability check (default: no-op).
    fn assert_ntt_runtime_support() {}
}

/// Blanket impl: any `Module<B>` whose handle implements `NttHandleProvider`
/// automatically satisfies `NttModuleHandle`.
impl<B> NttModuleHandle for Module<B>
where
    B: Backend,
    B::Handle: NttHandleProvider,
{
    fn get_ntt_table(&self) -> &NttTable<Primes30> {
        // SAFETY: `ptr()` returns a valid, non-null pointer to `B::Handle`
        // that was initialised by the module defaults and is kept alive by
        // the `Module`.
        unsafe { (&*self.ptr()).get_ntt_table() }
    }

    fn get_intt_table(&self) -> &NttTableInv<Primes30> {
        unsafe { (&*self.ptr()).get_intt_table() }
    }

    fn get_bbc_meta(&self) -> &BbcMeta<Primes30> {
        unsafe { (&*self.ptr()).get_bbc_meta() }
    }

    fn get_bbb_meta(&self) -> &BbbMeta<Primes30> {
        unsafe { (&*self.ptr()).get_bbb_meta() }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Helper: cast VecZnxDft limb to &[u64]
// ──────────────────────────────────────────────────────────────────────────────

/// Returns the q120b u64 slice for limb `(col, limb)` of a VecZnxDft.
///
/// `at(col, limb)` returns `&[Q120bScalar]` of length `n`; we cast to
/// `&[u64]` of length `4*n`.
#[inline(always)]
fn limb_u64<D: crate::layouts::DataRef, BE: Backend<ScalarPrep = Q120bScalar>>(
    v: &VecZnxDft<D, BE>,
    col: usize,
    limb: usize,
) -> &[u64] {
    cast_slice(v.at(col, limb))
}

#[inline(always)]
fn limb_u64_mut<D: crate::layouts::DataMut, BE: Backend<ScalarPrep = Q120bScalar>>(
    v: &mut VecZnxDft<D, BE>,
    col: usize,
    limb: usize,
) -> &mut [u64] {
    cast_slice_mut(v.at_mut(col, limb))
}

// ──────────────────────────────────────────────────────────────────────────────
// Forward DFT
// ──────────────────────────────────────────────────────────────────────────────

/// Forward NTT: encode `a[a_col]` into `res[res_col]`.
///
/// For each output limb `j`:
/// - Input limb index `= offset + j * step` from `a[a_col]`.
/// - Converts i64 coefficients to q120b with [`NttFromZnx64`],
///   then applies the forward NTT in-place via [`NttDFTExecute`].
/// - Missing input limbs (out of range) are zeroed in `res`.
pub fn ntt120_vec_znx_dft_apply<R, A, BE>(
    module: &impl NttModuleHandle,
    step: usize,
    offset: usize,
    res: &mut R,
    res_col: usize,
    a: &A,
    a_col: usize,
) where
    BE: Backend<ScalarPrep = Q120bScalar> + NttDFTExecute<NttTable<Primes30>> + NttFromZnx64 + NttZero,
    R: VecZnxDftToMut<BE>,
    A: VecZnxToRef,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a = a.to_ref();

    let a_size = a.size();
    let res_size = res.size();

    let table = module.get_ntt_table();

    let steps = a_size.div_ceil(step);
    let min_steps = res_size.min(steps);

    for j in 0..min_steps {
        let limb = offset + j * step;
        if limb < a_size {
            let res_slice: &mut [u64] = limb_u64_mut(&mut res, res_col, j);
            BE::ntt_from_znx64(res_slice, a.at(a_col, limb));
            BE::ntt_dft_execute(table, res_slice);
        } else {
            BE::ntt_zero(limb_u64_mut(&mut res, res_col, j));
        }
    }

    for j in min_steps..res_size {
        BE::ntt_zero(limb_u64_mut(&mut res, res_col, j));
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Inverse DFT
// ──────────────────────────────────────────────────────────────────────────────

/// Returns the scratch space (in bytes) for [`ntt120_vec_znx_idft_apply`].
///
/// Requires one q120b buffer of length `n` (4 u64 per coefficient).
pub fn ntt120_vec_znx_idft_apply_tmp_bytes(n: usize) -> usize {
    4 * n * size_of::<u64>()
}

/// Inverse NTT (non-destructive): decode `a[a_col]` into `res[res_col]`.
///
/// For each output limb `j`:
/// 1. Copies `a.at(a_col, j)` into `tmp` via [`NttCopy`].
/// 2. Applies the inverse NTT to `tmp` in place via [`NttDFTExecute`].
/// 3. CRT-reconstructs the `i128` coefficients via [`NttToZnx128`].
///
/// `tmp` must hold at least `4 * n` `u64` values.
pub fn ntt120_vec_znx_idft_apply<R, A, BE>(
    module: &impl NttModuleHandle,
    res: &mut R,
    res_col: usize,
    a: &A,
    a_col: usize,
    tmp: &mut [u64],
) where
    BE: Backend<ScalarPrep = Q120bScalar, ScalarBig = i128> + NttDFTExecute<NttTableInv<Primes30>> + NttToZnx128 + NttCopy,
    R: VecZnxBigToMut<BE>,
    A: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxBig<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();

    let n = res.n();
    let res_size = res.size();
    let min_size = res_size.min(a.size());

    let table = module.get_intt_table();

    for j in 0..min_size {
        let a_slice: &[u64] = limb_u64(&a, a_col, j);
        let tmp_n: &mut [u64] = &mut tmp[..4 * n];
        BE::ntt_copy(tmp_n, a_slice);
        BE::ntt_dft_execute(table, tmp_n);
        BE::ntt_to_znx128(res.at_mut(res_col, j), n, tmp_n);
    }

    for j in min_size..res_size {
        res.at_mut(res_col, j).fill(0i128);
    }
}

/// Inverse NTT (destructive): decode `a[a_col]` into `res[res_col]`.
///
/// Like [`ntt120_vec_znx_idft_apply`] but applies the inverse NTT
/// **in place** to `a`, modifying it.  Requires no scratch space.
pub fn ntt120_vec_znx_idft_apply_tmpa<R, A, BE>(
    module: &impl NttModuleHandle,
    res: &mut R,
    res_col: usize,
    a: &mut A,
    a_col: usize,
) where
    BE: Backend<ScalarPrep = Q120bScalar, ScalarBig = i128> + NttDFTExecute<NttTableInv<Primes30>> + NttToZnx128,
    R: VecZnxBigToMut<BE>,
    A: VecZnxDftToMut<BE>,
{
    let mut res: VecZnxBig<&mut [u8], BE> = res.to_mut();
    let mut a: VecZnxDft<&mut [u8], BE> = a.to_mut();

    let n = res.n();
    let res_size = res.size();
    let min_size = res_size.min(a.size());

    let table = module.get_intt_table();

    for j in 0..min_size {
        BE::ntt_dft_execute(table, limb_u64_mut(&mut a, a_col, j));
        let a_slice: &[u64] = limb_u64(&a, a_col, j);
        BE::ntt_to_znx128(res.at_mut(res_col, j), n, a_slice);
    }

    for j in min_size..res_size {
        res.at_mut(res_col, j).fill(0i128);
    }
}

#[inline(always)]
fn barrett_u61(x: u64, q: u64, mu: u64) -> u64 {
    let q_approx = ((x as u128 * mu as u128) >> 61) as u64;
    let r = x - q_approx * q;
    let r = if r >= q { r - q } else { r };
    if r >= q { r - q } else { r }
}

#[inline(always)]
fn reduce_q120b_crt(x: u64, q: u64, mu: u64, pow32_crt: u64, pow16_crt: u64, crt: u64) -> u64 {
    let x_hi = x >> 32;
    let x_hi_r = if x_hi >= q { x_hi - q } else { x_hi };
    let x_lo = x & 0xFFFF_FFFF;
    let x_lo_hi = x_lo >> 16;
    let x_lo_lo = x_lo & 0xFFFF;
    let tmp = x_hi_r
        .wrapping_mul(pow32_crt)
        .wrapping_add(x_lo_hi.wrapping_mul(pow16_crt))
        .wrapping_add(x_lo_lo.wrapping_mul(crt));
    barrett_u61(tmp, q, mu)
}

unsafe fn compact_all_blocks_scalar(n: usize, n_blocks: usize, u64_ptr: *mut u64, table: &NttTableInv<Primes30>) {
    let q_u64: [u64; 4] = Primes30::Q.map(|qi| qi as u64);
    let mu: [u64; 4] = q_u64.map(|qi| (1u64 << 61) / qi);
    let crt: [u64; 4] = Primes30::CRT_CST.map(|c| c as u64);

    let pow32_crt: [u64; 4] = std::array::from_fn(|k| {
        let pow32 = ((1u128 << 32) % q_u64[k] as u128) as u64;
        barrett_u61(pow32 * crt[k], q_u64[k], mu[k])
    });
    let pow16_crt: [u64; 4] = std::array::from_fn(|k| barrett_u61((1u64 << 16) * crt[k], q_u64[k], mu[k]));

    let q: [u128; 4] = q_u64.map(|qi| qi as u128);
    let total_q: u128 = q[0] * q[1] * q[2] * q[3];
    let qm: [u128; 4] = [q[1] * q[2] * q[3], q[0] * q[2] * q[3], q[0] * q[1] * q[3], q[0] * q[1] * q[2]];
    let half_q: u128 = total_q.div_ceil(2);
    let total_q_mult: [u128; 4] = [0, total_q, total_q * 2, total_q * 3];

    for k in 0..n_blocks {
        let src_start = 4 * n * k;
        let dst_start = 2 * n * k;

        {
            let blk: &mut [u64] = unsafe { std::slice::from_raw_parts_mut(u64_ptr.add(src_start), 4 * n) };
            intt_ref::<Primes30>(table, blk);
        }

        for c in 0..n {
            let (x0, x1, x2, x3) = unsafe {
                (
                    *u64_ptr.add(src_start + 4 * c),
                    *u64_ptr.add(src_start + 4 * c + 1),
                    *u64_ptr.add(src_start + 4 * c + 2),
                    *u64_ptr.add(src_start + 4 * c + 3),
                )
            };

            let t0 = reduce_q120b_crt(x0, q_u64[0], mu[0], pow32_crt[0], pow16_crt[0], crt[0]);
            let t1 = reduce_q120b_crt(x1, q_u64[1], mu[1], pow32_crt[1], pow16_crt[1], crt[1]);
            let t2 = reduce_q120b_crt(x2, q_u64[2], mu[2], pow32_crt[2], pow16_crt[2], crt[2]);
            let t3 = reduce_q120b_crt(x3, q_u64[3], mu[3], pow32_crt[3], pow16_crt[3], crt[3]);

            let mut v: u128 = t0 as u128 * qm[0] + t1 as u128 * qm[1] + t2 as u128 * qm[2] + t3 as u128 * qm[3];

            let q_approx = (v >> 120) as usize;
            v -= total_q_mult[q_approx];
            if v >= total_q {
                v -= total_q;
            }

            let val: i128 = if v >= half_q { v as i128 - total_q as i128 } else { v as i128 };

            unsafe { (u64_ptr.add(dst_start + 2 * c) as *mut i128).write_unaligned(val) };
        }
    }
}

/// Inverse NTT consuming the input and compacting the q120b layout in place.
///
/// This applies the inverse NTT block by block, then CRT-compacts the owned
/// `VecZnxDft` buffer from q120b (32 bytes/coeff) to the `VecZnxBig<i128>`
/// layout (16 bytes/coeff) without allocating a new buffer.
pub fn ntt120_vec_znx_idft_apply_consume<D: Data, BE>(module: &impl NttModuleHandle, mut a: VecZnxDft<D, BE>) -> VecZnxBig<D, BE>
where
    BE: Backend<ScalarPrep = Q120bScalar, ScalarBig = i128>,
    VecZnxDft<D, BE>: VecZnxDftToMut<BE>,
{
    let table = module.get_intt_table();

    let (n, n_blocks, u64_ptr) = {
        let mut a_mut: VecZnxDft<&mut [u8], BE> = a.to_mut();
        let n = a_mut.n();
        let n_blocks = a_mut.cols() * a_mut.size();
        let ptr: *mut u64 = {
            let s = a_mut.raw_mut();
            cast_slice_mut::<_, u64>(s).as_mut_ptr()
        };
        (n, n_blocks, ptr)
    };

    unsafe { compact_all_blocks_scalar(n, n_blocks, u64_ptr, table) };

    a.into_big()
}

// ──────────────────────────────────────────────────────────────────────────────
// DFT-domain arithmetic
// ──────────────────────────────────────────────────────────────────────────────

/// DFT-domain add: `res[res_col] = a[a_col] + b[b_col]`.
///
/// Uses lazy q120b addition; out-of-range limbs are copied or zeroed.
pub fn ntt120_vec_znx_dft_add_into<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<ScalarPrep = Q120bScalar> + NttAdd + NttCopy + NttZero,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
    B: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();
    let b: VecZnxDft<&[u8], BE> = b.to_ref();

    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();

    if a_size <= b_size {
        let sum_size = a_size.min(res_size);
        let cpy_size = b_size.min(res_size);
        for j in 0..sum_size {
            BE::ntt_add(
                limb_u64_mut(&mut res, res_col, j),
                limb_u64(&a, a_col, j),
                limb_u64(&b, b_col, j),
            );
        }
        for j in sum_size..cpy_size {
            BE::ntt_copy(limb_u64_mut(&mut res, res_col, j), limb_u64(&b, b_col, j));
        }
        for j in cpy_size..res_size {
            BE::ntt_zero(limb_u64_mut(&mut res, res_col, j));
        }
    } else {
        let sum_size = b_size.min(res_size);
        let cpy_size = a_size.min(res_size);
        for j in 0..sum_size {
            BE::ntt_add(
                limb_u64_mut(&mut res, res_col, j),
                limb_u64(&a, a_col, j),
                limb_u64(&b, b_col, j),
            );
        }
        for j in sum_size..cpy_size {
            BE::ntt_copy(limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, j));
        }
        for j in cpy_size..res_size {
            BE::ntt_zero(limb_u64_mut(&mut res, res_col, j));
        }
    }
}

/// DFT-domain in-place add: `res[res_col] += a[a_col]`.
pub fn ntt120_vec_znx_dft_add_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarPrep = Q120bScalar> + NttAddAssign,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();

    let sum_size = res.size().min(a.size());
    for j in 0..sum_size {
        BE::ntt_add_assign(limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, j));
    }
}

/// DFT-domain scaled in-place add: `res[res_col] += a[a_col] >> (a_scale * base2k)`.
///
/// `a_scale > 0` shifts `a` down by `a_scale` limbs (drops low limbs);
/// `a_scale < 0` shifts `a` up by `|a_scale|` limbs (adds into higher limbs).
pub fn ntt120_vec_znx_dft_add_scaled_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, a_scale: i64)
where
    BE: Backend<ScalarPrep = Q120bScalar> + NttAddAssign,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();

    let res_size = res.size();
    let a_size = a.size();

    if a_scale > 0 {
        let shift = (a_scale as usize).min(a_size);
        let sum_size = a_size.min(res_size).saturating_sub(shift);
        for j in 0..sum_size {
            BE::ntt_add_assign(limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, j + shift));
        }
    } else if a_scale < 0 {
        let shift = (a_scale.unsigned_abs() as usize).min(res_size);
        let sum_size = a_size.min(res_size.saturating_sub(shift));
        for j in 0..sum_size {
            BE::ntt_add_assign(limb_u64_mut(&mut res, res_col, j + shift), limb_u64(&a, a_col, j));
        }
    } else {
        let sum_size = a_size.min(res_size);
        for j in 0..sum_size {
            BE::ntt_add_assign(limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, j));
        }
    }
}

/// DFT-domain sub: `res[res_col] = a[a_col] - b[b_col]`.
pub fn ntt120_vec_znx_dft_sub<R, A, B, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize, b: &B, b_col: usize)
where
    BE: Backend<ScalarPrep = Q120bScalar> + NttSub + NttNegate + NttCopy + NttZero,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
    B: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();
    let b: VecZnxDft<&[u8], BE> = b.to_ref();

    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();

    if a_size <= b_size {
        let sum_size = a_size.min(res_size);
        let cpy_size = b_size.min(res_size);
        for j in 0..sum_size {
            BE::ntt_sub(
                limb_u64_mut(&mut res, res_col, j),
                limb_u64(&a, a_col, j),
                limb_u64(&b, b_col, j),
            );
        }
        for j in sum_size..cpy_size {
            BE::ntt_negate(limb_u64_mut(&mut res, res_col, j), limb_u64(&b, b_col, j));
        }
        for j in cpy_size..res_size {
            BE::ntt_zero(limb_u64_mut(&mut res, res_col, j));
        }
    } else {
        let sum_size = b_size.min(res_size);
        let cpy_size = a_size.min(res_size);
        for j in 0..sum_size {
            BE::ntt_sub(
                limb_u64_mut(&mut res, res_col, j),
                limb_u64(&a, a_col, j),
                limb_u64(&b, b_col, j),
            );
        }
        for j in sum_size..cpy_size {
            BE::ntt_copy(limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, j));
        }
        for j in cpy_size..res_size {
            BE::ntt_zero(limb_u64_mut(&mut res, res_col, j));
        }
    }
}

/// DFT-domain in-place sub: `res[res_col] -= a[a_col]`.
pub fn ntt120_vec_znx_dft_sub_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarPrep = Q120bScalar> + NttSubAssign,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();

    let sum_size = res.size().min(a.size());
    for j in 0..sum_size {
        BE::ntt_sub_assign(limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, j));
    }
}

/// DFT-domain in-place swap-sub: `res[res_col] = a[a_col] - res[res_col]`.
///
/// Extra `res` limbs beyond `a.size()` are negated.
pub fn ntt120_vec_znx_dft_sub_negate_assign<R, A, BE>(res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarPrep = Q120bScalar> + NttSubNegateAssign + NttNegateAssign,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();

    let res_size = res.size();
    let sum_size = res_size.min(a.size());
    for j in 0..sum_size {
        BE::ntt_sub_negate_assign(limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, j));
    }
    for j in sum_size..res_size {
        BE::ntt_negate_assign(limb_u64_mut(&mut res, res_col, j));
    }
}

/// DFT-domain copy with stride: `res[res_col][j] = a[a_col][offset + j*step]`.
///
/// Mirrors `vec_znx_dft_copy` from the FFT64 backend.
pub fn ntt120_vec_znx_dft_copy<R, A, BE>(step: usize, offset: usize, res: &mut R, res_col: usize, a: &A, a_col: usize)
where
    BE: Backend<ScalarPrep = Q120bScalar> + NttCopy + NttZero,
    R: VecZnxDftToMut<BE>,
    A: VecZnxDftToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: VecZnxDft<&[u8], BE> = a.to_ref();

    #[cfg(debug_assertions)]
    {
        assert_eq!(res.n(), a.n())
    }

    let steps: usize = a.size().div_ceil(step);
    let min_steps: usize = res.size().min(steps);

    for j in 0..min_steps {
        let limb = offset + j * step;
        if limb < a.size() {
            BE::ntt_copy(limb_u64_mut(&mut res, res_col, j), limb_u64(&a, a_col, limb));
        } else {
            BE::ntt_zero(limb_u64_mut(&mut res, res_col, j));
        }
    }
    for j in min_steps..res.size() {
        BE::ntt_zero(limb_u64_mut(&mut res, res_col, j));
    }
}

/// Zero all limbs of `res[res_col]`.
pub fn ntt120_vec_znx_dft_zero<R, BE>(res: &mut R, res_col: usize)
where
    BE: Backend<ScalarPrep = Q120bScalar> + NttZero,
    R: VecZnxDftToMut<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    for j in 0..res.size() {
        BE::ntt_zero(limb_u64_mut(&mut res, res_col, j));
    }
}
