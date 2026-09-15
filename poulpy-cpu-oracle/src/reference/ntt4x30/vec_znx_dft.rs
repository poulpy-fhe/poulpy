//! NTT-domain vector operations using four canonical residues per coefficient.

use bytemuck::{cast_slice, cast_slice_mut};

use crate::{
    layouts::{
        Backend, HostDataMut, HostDataRef, Module, VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDft, VecZnxDftBackendMut,
        VecZnxDftBackendRef, ZnxView, ZnxViewMut,
    },
    reference::ntt4x30::{
        NttAdd, NttAddAssign, NttCopy, NttDFTExecute, NttFromZnx64, NttNegate, NttNegateAssign, NttSub, NttSubAssign,
        NttSubNegateAssign, NttToZnx128, NttZero,
        ntt::{NttTable, NttTableInv},
        primes::{PrimeSetCrt4, Primes30},
        types::Q120bScalar,
    },
};

/// Forward and inverse NTT tables for one ring degree.
pub struct NttPlan<P: PrimeSetCrt4> {
    ntt: NttTable<P>,
    intt: NttTableInv<P>,
}

impl<P: PrimeSetCrt4> NttPlan<P> {
    pub fn new(n: usize) -> Self {
        Self {
            ntt: NttTable::new(n),
            intt: NttTableInv::new(n),
        }
    }

    pub fn ntt(&self) -> &NttTable<P> {
        &self.ntt
    }

    pub fn intt(&self) -> &NttTableInv<P> {
        &self.intt
    }
}

/// Complete geometric family of NTT plans up to a maximum ring degree.
pub struct NttPlanSet<P: PrimeSetCrt4> {
    plans: Vec<NttPlan<P>>,
    max_n: usize,
}

impl<P: PrimeSetCrt4> NttPlanSet<P> {
    pub fn new(max_n: usize) -> Self {
        assert!(
            max_n.is_power_of_two(),
            "maximum ring degree must be a power of two, got {max_n}"
        );
        let plans = (0..=max_n.ilog2() as usize)
            .map(|log_n| NttPlan::new(1usize << log_n))
            .collect();
        Self { plans, max_n }
    }

    pub fn for_ring(&self, n: usize) -> &NttPlan<P> {
        assert!(
            n.is_power_of_two() && n <= self.max_n,
            "unsupported ring degree {n}; maximum is {}",
            self.max_n
        );
        &self.plans[n.ilog2() as usize]
    }
}

/// Access to the forward and inverse NTT tables.
pub trait NttModuleHandle: poulpy_hal::api::ModuleN {
    /// Combined NTT plan for an explicit ring degree.
    fn get_ntt_plan(&self, n: usize) -> &NttPlan<Primes30>;
    /// Precomputed forward NTT twiddle table (Primes30, size `n`).
    fn get_ntt_table_for(&self, n: usize) -> &NttTable<Primes30> {
        self.get_ntt_plan(n).ntt()
    }
    /// Precomputed inverse NTT twiddle table (Primes30, size `n`).
    fn get_intt_table_for(&self, n: usize) -> &NttTableInv<Primes30> {
        self.get_ntt_plan(n).intt()
    }
    fn get_ntt_table(&self) -> &NttTable<Primes30> {
        self.get_ntt_table_for(self.n())
    }
    fn get_intt_table(&self) -> &NttTableInv<Primes30> {
        self.get_intt_table_for(self.n())
    }
}

/// Access to the module's initialized transform plans.
///
/// # Safety
/// The plans must remain valid for the lifetime of the handle.
pub unsafe trait NttHandleProvider {
    /// Returns the combined NTT plan for `n`.
    fn get_ntt_plan(&self, n: usize) -> &NttPlan<Primes30>;
}

/// Construct NTT4x30 backend handles for [`Module::new`](poulpy_hal::api::ModuleNew::new).
///
/// # Safety
///
/// Implementors must return a fully initialized handle for the requested `n`.
/// The handle is boxed and stored inside the `Module`, so it must be safe to
/// drop via [`crate::layouts::Backend::destroy`].
pub unsafe trait NttHandleFactory: Sized {
    /// Builds a fully initialized handle for ring dimension `n`.
    fn create_ntt_handle(n: usize) -> Self;
}

/// Blanket impl: any `Module<B>` whose handle implements `NttHandleProvider`
/// automatically satisfies `NttModuleHandle`.
impl<B> NttModuleHandle for Module<B>
where
    B: Backend<ZnxWord = i64>,
    B::Handle: NttHandleProvider,
{
    fn get_ntt_plan(&self, n: usize) -> &NttPlan<Primes30> {
        // SAFETY: `ptr()` returns a valid, non-null pointer to `B::Handle`
        // that was initialised by the module defaults and is kept alive by
        // the `Module`.
        unsafe { (&*self.ptr()).get_ntt_plan(n) }
    }
}

/// Returns the q120b u64 slice for limb `(col, limb)` of a VecZnxDft.
///
/// `at(col, limb)` returns `&[Q120bScalar]` of length `n`; we cast to
/// `&[u64]` of length `4*n`.
fn limb_u64<D: crate::layouts::HostDataRef, BE: Backend<DftWord = Q120bScalar, ZnxWord = i64>>(
    v: &VecZnxDft<D, BE::DftWord, BE>,
    col: usize,
    limb: usize,
) -> &[u64] {
    cast_slice(v.at(col, limb))
}

fn limb_u64_mut<D: crate::layouts::HostDataMut, BE: Backend<DftWord = Q120bScalar, ZnxWord = i64>>(
    v: &mut VecZnxDft<D, BE::DftWord, BE>,
    col: usize,
    limb: usize,
) -> &mut [u64] {
    cast_slice_mut(v.at_mut(col, limb))
}

/// Forward NTT: encode `a[a_col]` into `res[res_col]`.
///
/// For each output limb `j`:
/// - Input limb index `= offset + j * step` from `a[a_col]`.
/// - Converts i64 coefficients to q120b with [`NttFromZnx64`],
///   then applies the forward NTT in-place via [`NttDFTExecute`].
/// - Missing input limbs (out of range) are zeroed in `res`.
pub fn ntt4x30_vec_znx_dft_apply<BE>(
    module: &impl NttModuleHandle,
    step: usize,
    offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, BE>,
    a_col: usize,
) where
    BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttDFTExecute<NttTable<Primes30>> + NttFromZnx64 + NttZero + 'static,
    for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8], ZnxWord = i64>,
{
    poulpy_hal::layouts::assert_dense(a, "ntt4x30_vec_znx_dft_apply");
    assert!(step >= 1, "ntt4x30_vec_znx_dft_apply: step must be >= 1");
    let a_size = a.size();
    let res_size = res.size();

    let table = module.get_ntt_table();

    let steps = a_size.div_ceil(step);
    let min_steps = res_size.min(steps);

    for j in 0..min_steps {
        let limb = offset + j * step;
        if limb < a_size {
            let res_slice: &mut [u64] = limb_u64_mut::<_, BE>(res, res_col, j);
            BE::ntt_from_znx64(res_slice, a.at(a_col, limb));
            BE::ntt_dft_execute(table, res_slice);
        } else {
            BE::ntt_zero(limb_u64_mut::<_, BE>(res, res_col, j));
        }
    }

    for j in min_steps..res_size {
        BE::ntt_zero(limb_u64_mut::<_, BE>(res, res_col, j));
    }
}

/// Returns the scratch space (in bytes) for [`ntt4x30_vec_znx_idft_apply`].
///
/// Requires one q120b buffer of length `n` (4 u64 per coefficient).
pub fn ntt4x30_vec_znx_idft_apply_tmp_bytes(n: usize) -> usize {
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
pub fn ntt4x30_vec_znx_idft_apply<BE>(
    module: &impl NttModuleHandle,
    res: &mut VecZnxBigBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, BE>,
    a_col: usize,
    tmp: &mut [u64],
) where
    BE: Backend<DftWord = Q120bScalar, BigWord = i128, ZnxWord = i64>
        + NttDFTExecute<NttTableInv<Primes30>>
        + NttToZnx128
        + NttCopy,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
{
    poulpy_hal::layouts::assert_dense(res, "ntt4x30_vec_znx_idft_apply");
    let n = res.n();
    let res_size = res.size();
    let min_size = res_size.min(a.size());

    let table = module.get_intt_table();

    for j in 0..min_size {
        let a_slice: &[u64] = limb_u64::<_, BE>(a, a_col, j);
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
/// Like [`ntt4x30_vec_znx_idft_apply`] but applies the inverse NTT
/// **in place** to `a`, modifying it.  Requires no scratch space.
pub fn ntt4x30_vec_znx_idft_apply_tmpa<BE>(
    module: &impl NttModuleHandle,
    res: &mut VecZnxBigBackendMut<'_, BE>,
    res_col: usize,
    a: &mut VecZnxDftBackendMut<'_, BE>,
    a_col: usize,
) where
    BE: Backend<DftWord = Q120bScalar, BigWord = i128, ZnxWord = i64> + NttDFTExecute<NttTableInv<Primes30>> + NttToZnx128,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
{
    poulpy_hal::layouts::assert_dense(res, "ntt4x30_vec_znx_idft_apply_tmpa");
    let n = res.n();
    let res_size = res.size();
    let min_size = res_size.min(a.size());

    let table = module.get_intt_table();

    for j in 0..min_size {
        BE::ntt_dft_execute(table, limb_u64_mut::<_, BE>(a, a_col, j));
        let a_slice: &[u64] = limb_u64::<_, BE>(a, a_col, j);
        BE::ntt_to_znx128(res.at_mut(res_col, j), n, a_slice);
    }

    for j in min_size..res_size {
        res.at_mut(res_col, j).fill(0i128);
    }
}

/// DFT-domain add: `res[res_col] = a[a_col] + b[b_col]`.
///
/// Uses modular addition; out-of-range limbs are copied or zeroed.
pub fn ntt4x30_vec_znx_dft_add<BE>(
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, BE>,
    a_col: usize,
    b: &VecZnxDftBackendRef<'_, BE>,
    b_col: usize,
) where
    BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttAdd + NttCopy + NttZero,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
{
    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();

    if a_size <= b_size {
        let sum_size = a_size.min(res_size);
        let cpy_size = b_size.min(res_size);
        for j in 0..sum_size {
            BE::ntt_add(
                limb_u64_mut::<_, BE>(res, res_col, j),
                limb_u64::<_, BE>(a, a_col, j),
                limb_u64::<_, BE>(b, b_col, j),
            );
        }
        for j in sum_size..cpy_size {
            BE::ntt_copy(limb_u64_mut::<_, BE>(res, res_col, j), limb_u64::<_, BE>(b, b_col, j));
        }
        for j in cpy_size..res_size {
            BE::ntt_zero(limb_u64_mut::<_, BE>(res, res_col, j));
        }
    } else {
        let sum_size = b_size.min(res_size);
        let cpy_size = a_size.min(res_size);
        for j in 0..sum_size {
            BE::ntt_add(
                limb_u64_mut::<_, BE>(res, res_col, j),
                limb_u64::<_, BE>(a, a_col, j),
                limb_u64::<_, BE>(b, b_col, j),
            );
        }
        for j in sum_size..cpy_size {
            BE::ntt_copy(limb_u64_mut::<_, BE>(res, res_col, j), limb_u64::<_, BE>(a, a_col, j));
        }
        for j in cpy_size..res_size {
            BE::ntt_zero(limb_u64_mut::<_, BE>(res, res_col, j));
        }
    }
}

/// DFT-domain in-place add: `res[res_col] += a[a_col]`.
pub fn ntt4x30_vec_znx_dft_add_assign<BE>(
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, BE>,
    a_col: usize,
) where
    BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttAddAssign,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
{
    let sum_size = res.size().min(a.size());
    for j in 0..sum_size {
        BE::ntt_add_assign(limb_u64_mut::<_, BE>(res, res_col, j), limb_u64::<_, BE>(a, a_col, j));
    }
}

/// DFT-domain sub: `res[res_col] = a[a_col] - b[b_col]`.
pub fn ntt4x30_vec_znx_dft_sub<BE>(
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, BE>,
    a_col: usize,
    b: &VecZnxDftBackendRef<'_, BE>,
    b_col: usize,
) where
    BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttSub + NttNegate + NttCopy + NttZero,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
{
    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();

    if a_size <= b_size {
        let sum_size = a_size.min(res_size);
        let cpy_size = b_size.min(res_size);
        for j in 0..sum_size {
            BE::ntt_sub(
                limb_u64_mut::<_, BE>(res, res_col, j),
                limb_u64::<_, BE>(a, a_col, j),
                limb_u64::<_, BE>(b, b_col, j),
            );
        }
        for j in sum_size..cpy_size {
            BE::ntt_negate(limb_u64_mut::<_, BE>(res, res_col, j), limb_u64::<_, BE>(b, b_col, j));
        }
        for j in cpy_size..res_size {
            BE::ntt_zero(limb_u64_mut::<_, BE>(res, res_col, j));
        }
    } else {
        let sum_size = b_size.min(res_size);
        let cpy_size = a_size.min(res_size);
        for j in 0..sum_size {
            BE::ntt_sub(
                limb_u64_mut::<_, BE>(res, res_col, j),
                limb_u64::<_, BE>(a, a_col, j),
                limb_u64::<_, BE>(b, b_col, j),
            );
        }
        for j in sum_size..cpy_size {
            BE::ntt_copy(limb_u64_mut::<_, BE>(res, res_col, j), limb_u64::<_, BE>(a, a_col, j));
        }
        for j in cpy_size..res_size {
            BE::ntt_zero(limb_u64_mut::<_, BE>(res, res_col, j));
        }
    }
}

/// DFT-domain in-place sub: `res[res_col] -= a[a_col]`.
pub fn ntt4x30_vec_znx_dft_sub_assign<BE>(
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, BE>,
    a_col: usize,
) where
    BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttSubAssign,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
{
    let sum_size = res.size().min(a.size());
    for j in 0..sum_size {
        BE::ntt_sub_assign(limb_u64_mut::<_, BE>(res, res_col, j), limb_u64::<_, BE>(a, a_col, j));
    }
}

/// DFT-domain in-place swap-sub: `res[res_col] = a[a_col] - res[res_col]`.
///
/// Extra `res` limbs beyond `a.size()` are negated.
pub fn ntt4x30_vec_znx_dft_sub_negate_assign<BE>(
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, BE>,
    a_col: usize,
) where
    BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttSubNegateAssign + NttNegateAssign,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
{
    let res_size = res.size();
    let sum_size = res_size.min(a.size());
    for j in 0..sum_size {
        BE::ntt_sub_negate_assign(limb_u64_mut::<_, BE>(res, res_col, j), limb_u64::<_, BE>(a, a_col, j));
    }
    for j in sum_size..res_size {
        BE::ntt_negate_assign(limb_u64_mut::<_, BE>(res, res_col, j));
    }
}

/// DFT-domain copy with stride: `res[res_col][j] = a[a_col][offset + j*step]`.
///
/// Mirrors `vec_znx_dft_copy` from the FFT64 backend.
pub fn ntt4x30_vec_znx_dft_copy<BE>(
    step: usize,
    offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, BE>,
    a_col: usize,
) where
    BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttCopy + NttZero,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
{
    {
        assert_eq!(res.n(), a.n());
        assert!(step >= 1, "ntt4x30_vec_znx_dft_copy: step must be >= 1");
    }

    let steps: usize = a.size().div_ceil(step);
    let min_steps: usize = res.size().min(steps);

    for j in 0..min_steps {
        let limb = offset + j * step;
        if limb < a.size() {
            BE::ntt_copy(limb_u64_mut::<_, BE>(res, res_col, j), limb_u64::<_, BE>(a, a_col, limb));
        } else {
            BE::ntt_zero(limb_u64_mut::<_, BE>(res, res_col, j));
        }
    }
    for j in min_steps..res.size() {
        BE::ntt_zero(limb_u64_mut::<_, BE>(res, res_col, j));
    }
}

/// Zero all limbs of `res[res_col]`.
pub fn ntt4x30_vec_znx_dft_zero<BE>(res: &mut VecZnxDftBackendMut<'_, BE>, res_col: usize)
where
    BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttZero,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
{
    for j in 0..res.size() {
        BE::ntt_zero(limb_u64_mut::<_, BE>(res, res_col, j));
    }
}

/// Precomputed permutation for `VecZnxDft` automorphism `tau_p : X -> X^p`
/// in the NTT4x30 layout.
///
/// `perm[i]` is the source slot (in q120b units of 4 u64) that supplies
/// output slot `i`. NTT4x30 stores all `n` evaluations, so the action is a
/// pure permutation with no sign / conjugate flag.
#[derive(Clone, Debug)]
pub struct NttAutomorphismPlan {
    pub p: i64,
    pub perm: Vec<u32>,
}

/// Builds the [`NttAutomorphismPlan`] for ring dimension `n` and odd `p`.
///
/// The DIF NTT places output slot `i` at the evaluation point
/// `omega^{2 * bitrev(i) + 1}` mod `2n`, where `bitrev` is the bit-reversal
/// of `i` over `log2(n)` bits and `omega` is a primitive `2n`-th root.
/// The set `{1, 3, …, 2n - 1}` is closed under multiplication by any odd
/// `p`, so the action is a pure permutation — no closure trick or
/// conjugation flag is needed.
pub fn build_ntt4x30_automorphism_plan(n: usize, p: i64) -> NttAutomorphismPlan {
    assert!(n.is_power_of_two(), "n must be a power of two, got {n}");
    assert!(p & 1 == 1, "p must be odd for an R/(X^N+1) automorphism, got {p}");

    let mask = (2 * n - 1) as i64;
    let p_mod_2n = p & mask;
    let log_n = n.trailing_zeros();
    let ir = |i: u32| -> u32 { i.reverse_bits() >> (32 - log_n) };

    let mut perm: Vec<u32> = vec![0u32; n];
    for (i, mi) in perm.iter_mut().enumerate().take(n) {
        let e_out: i64 = 2 * ir(i as u32) as i64 + 1;
        let e_src: i64 = (p_mod_2n * e_out) & mask;
        let src: u32 = ((e_src - 1) >> 1) as u32;
        *mi = ir(src);
    }
    NttAutomorphismPlan { p, perm }
}

/// Applies a precomputed NTT4x30 automorphism plan to `a`, writing the
/// result into `res` (out-of-place).
///
/// Per output slot, one 4-u64 q120b copy from the indexed source slot.
/// No modular arithmetic, no negation — full-spectrum NTT layout makes
/// the action a pure permutation.
pub fn ntt4x30_vec_znx_dft_automorphism<BE>(
    plan: &NttAutomorphismPlan,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, BE>,
    a_col: usize,
) where
    BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttZero,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
{
    {
        assert_eq!(a.n(), res.n());
        assert_eq!(plan.perm.len(), res.n());
    }

    let n: usize = res.n();
    let res_size: usize = res.size();
    let a_size: usize = a.size();
    let min_size: usize = res_size.min(a_size);
    let perm: &[u32] = &plan.perm;

    for limb in 0..min_size {
        let a_slice: &[u64] = limb_u64::<_, BE>(a, a_col, limb);
        let res_slice: &mut [u64] = limb_u64_mut::<_, BE>(res, res_col, limb);

        for i in 0..n {
            let s = perm[i] as usize;
            // 4-u64 q120b slot copy. The destination is sequential, the
            // source is gathered through `perm`.
            res_slice[4 * i..4 * i + 4].copy_from_slice(&a_slice[4 * s..4 * s + 4]);
        }
    }

    for limb in min_size..res_size {
        BE::ntt_zero(limb_u64_mut::<_, BE>(res, res_col, limb));
    }
}
