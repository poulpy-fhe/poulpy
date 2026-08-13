//! Scalar-vector product (SVP) operations for [`NTT3x42Ifma`].
//!
//! Every public kernel takes its prepared operand as the concrete layout type,
//! once per tier: the `ppol` set consumes [`SvpPPol`](poulpy_hal::layouts::SvpPPol),
//! the `tpol` set [`SvpTPol`](poulpy_hal::layouts::SvpTPol). Callers therefore
//! never lose the container, and the two tiers can diverge independently.
//!
//! `NTT3x42Ifma` currently builds both tiers with the same IFMA NTT plus
//! q120b-to-q120c conversion, so each pair of public kernels forwards to one
//! private `*_inner` routine rather than repeating the body. The inner routines
//! take the prepared coefficient row (`a.at(a_col, 0)`) because that is the only
//! thing the arithmetic reads; they are private precisely so that this sharing
//! stays an implementation detail and never reaches a signature. Giving this
//! backend a genuinely cheaper hot-prep form means splitting a pair and pointing
//! the `tpol` kernel at its own body, with no change above this module.

use bytemuck::{cast_slice, cast_slice_mut};
use poulpy_hal::layouts::PrimeSet;

use poulpy_hal::{
    api::{VecZnxDftAlloc, VecZnxDftApply},
    layouts::{
        Backend, HostDataMut, HostDataRef, Module, ScalarZnxBackendRef, SvpPPolBackendMut, SvpPPolBackendRef, SvpTPolBackendMut,
        SvpTPolBackendRef, SvpTPolOwned, SvpTPolToBackendMut, SvpTPolToBackendRef, VecZnxBackendRef, VecZnxDftBackendMut,
        VecZnxDftBackendRef, VecZnxDftReborrowBackendRef, VecZnxDftToBackendMut, ZnxView, ZnxViewMut,
    },
};

use crate::NTT3x42Ifma;
use crate::ntt3x42_ifma::{
    kernels::ntt_avx512,
    module::handle,
    primes::Primes42,
    tables::{harvey_modmul, harvey_quotient},
    traits::{Ntt3x42IfmaCFromB, Ntt3x42IfmaFromZnx64, Ntt3x42IfmaZero},
    types::Q126Scalar,
};

#[inline(always)]
fn mul_mod_lazy(a: u64, b: u64, prime: usize) -> u64 {
    let q = Primes42::Q[prime];
    let b = if b >= q { b - q } else { b };
    harvey_modmul(a, b, harvey_quotient(b, q), q)
}

/// Shared body of [`svp_prepare_ppol`] and [`svp_prepare_tpol`].
///
/// Maps the i64 coefficients to q120b, runs the forward NTT leaving the result
/// lazy in `[0, 4q)`, then converts to q120c, which re-reduces. `res` is the
/// destination coefficient row of either prepared container.
fn prepare_inner(module: &Module<NTT3x42Ifma>, n: usize, res: &mut [Q126Scalar], a: &[i64]) {
    let mut tmp = vec![0u64; 3 * n];
    NTT3x42Ifma::ntt3x42_ifma_from_znx64(&mut tmp, a);
    // Lazy [0, 4q): consumed only by c_from_b (re-reduces).
    unsafe { ntt_avx512::<Primes42>(&handle(module).table_ntt, &mut tmp, true) };

    NTT3x42Ifma::ntt3x42_ifma_c_from_b(n, cast_slice_mut(res), &tmp);
}

/// Shared body of the two `*_small_to_dft` kernels.
///
/// Lifts `b` to the DFT domain via the forward NTT into an owned temporary,
/// then defers to [`dft_to_dft_inner`]. `pol` is the prepared coefficient row.
fn small_to_dft_inner(
    module: &Module<NTT3x42Ifma>,
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    pol: &[Q126Scalar],
    b: &VecZnxBackendRef<'_, NTT3x42Ifma>,
    b_col: usize,
) {
    let mut b_dft_owned = module.vec_znx_dft_alloc(1, b.size());
    let mut b_dft = b_dft_owned.to_backend_mut();
    <Module<NTT3x42Ifma> as VecZnxDftApply<NTT3x42Ifma>>::vec_znx_dft_apply(module, 1, 0, &mut b_dft, 0, b, b_col);
    let b_dft_ref = b_dft.reborrow_backend_ref();
    dft_to_dft_inner(res, res_col, pol, &b_dft_ref, 0);
}

/// Shared body of the two `*_dft_to_dft` kernels.
///
/// Pointwise `res = pol * b` over the three primes, zeroing limbs past
/// `b.size()`. `pol` is the prepared coefficient row, constant across limbs.
fn dft_to_dft_inner(
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    pol: &[Q126Scalar],
    b: &VecZnxDftBackendRef<'_, NTT3x42Ifma>,
    b_col: usize,
) {
    let n = res.n();
    let res_size = res.size();
    let min_size = res_size.min(b.size());

    let a_u64: &[u64] = cast_slice(pol);

    for j in 0..min_size {
        let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(res_col, j));
        let b_u64: &[u64] = cast_slice(b.at(b_col, j));
        for prime in 0..3 {
            let base = prime * n;
            for i in 0..n {
                res_u64[base + i] = mul_mod_lazy(b_u64[base + i], a_u64[base + i], prime);
            }
        }
    }

    for j in min_size..res_size {
        NTT3x42Ifma::ntt3x42_ifma_zero(cast_slice_mut(res.at_mut(res_col, j)));
    }
}

/// Shared body of the two `*_dft_to_dft_assign` kernels.
///
/// Pointwise `res = pol * res` over the three primes. Every limb of `res` is
/// touched, so there is no zero-fill tail.
fn dft_to_dft_assign_inner(res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>, res_col: usize, pol: &[Q126Scalar]) {
    let n = res.n();
    let res_size = res.size();

    let a_u64: &[u64] = cast_slice(pol);

    for j in 0..res_size {
        let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(res_col, j));
        for prime in 0..3 {
            let base = prime * n;
            for i in 0..n {
                res_u64[base + i] = mul_mod_lazy(res_u64[base + i], a_u64[base + i], prime);
            }
        }
    }
}

/// Encodes `a[a_col]` into the packed cold-prep column `res[res_col]`.
pub(crate) fn svp_prepare_ppol(
    module: &Module<NTT3x42Ifma>,
    res: &mut SvpPPolBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &ScalarZnxBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
) {
    let n = res.n();
    prepare_inner(module, n, res.at_mut(res_col, 0), a.at(a_col, 0));
}

/// Encodes `a[a_col]` into the transformed hot-prep column `res[res_col]`.
pub(crate) fn svp_prepare_tpol(
    module: &Module<NTT3x42Ifma>,
    res: &mut SvpTPolBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &ScalarZnxBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
) {
    let n = res.n();
    prepare_inner(module, n, res.at_mut(res_col, 0), a.at(a_col, 0));
}

/// `res = a * NTT(b)`, with `a` cold-prepared and `b` in coefficient domain.
pub(crate) fn svp_apply_ppol_small_to_dft(
    module: &Module<NTT3x42Ifma>,
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &SvpPPolBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
    b: &VecZnxBackendRef<'_, NTT3x42Ifma>,
    b_col: usize,
) {
    small_to_dft_inner(module, res, res_col, a.at(a_col, 0), b, b_col);
}

/// `res = a * NTT(b)`, with `a` hot-prepared and `b` in coefficient domain.
pub(crate) fn svp_apply_tpol_small_to_dft(
    module: &Module<NTT3x42Ifma>,
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &SvpTPolBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
    b: &VecZnxBackendRef<'_, NTT3x42Ifma>,
    b_col: usize,
) {
    small_to_dft_inner(module, res, res_col, a.at(a_col, 0), b, b_col);
}

/// `res = a * b`, with `a` cold-prepared and `b` in DFT domain.
pub(crate) fn svp_apply_ppol_dft_to_dft(
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &SvpPPolBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
    b: &VecZnxDftBackendRef<'_, NTT3x42Ifma>,
    b_col: usize,
) {
    dft_to_dft_inner(res, res_col, a.at(a_col, 0), b, b_col);
}

/// `res = a * b`, with `a` hot-prepared and `b` in DFT domain.
pub(crate) fn svp_apply_tpol_dft_to_dft(
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &SvpTPolBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
    b: &VecZnxDftBackendRef<'_, NTT3x42Ifma>,
    b_col: usize,
) {
    dft_to_dft_inner(res, res_col, a.at(a_col, 0), b, b_col);
}

/// `res = a * res`, with `a` cold-prepared.
pub(crate) fn svp_apply_ppol_dft_to_dft_assign(
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &SvpPPolBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
) {
    dft_to_dft_assign_inner(res, res_col, a.at(a_col, 0));
}

/// `res = a * res`, with `a` hot-prepared.
pub(crate) fn svp_apply_tpol_dft_to_dft_assign(
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &SvpTPolBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
) {
    dft_to_dft_assign_inner(res, res_col, a.at(a_col, 0));
}

/// Flavor trait wiring the IFMA SVP kernels into `hal_impl_svp!`.
///
/// The eight required methods are the backend kernels, one per (tier, shape)
/// pair. The rest are shared forwarders: the two column copies, and the three
/// `small`-scalar variants, which prepare into a temporary [`SvpTPol`] and then
/// run the corresponding `tpol` kernel.
#[doc(hidden)]
pub trait NTT3x42IfmaSvpDefault<BE: Backend<ZnxWord = i64>>: Backend
where
    BE::OwnedBuf: HostDataMut,
{
    fn svp_prepare_ppol_default(
        module: &Module<BE>,
        res: &mut SvpPPolBackendMut<'_, BE>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
    );

    fn svp_prepare_tpol_default(
        module: &Module<BE>,
        res: &mut SvpTPolBackendMut<'_, BE>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
    );

    #[allow(clippy::too_many_arguments)]
    fn svp_apply_ppol_small_to_dft_default(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE>,
        b_col: usize,
    );

    #[allow(clippy::too_many_arguments)]
    fn svp_apply_tpol_small_to_dft_default(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &SvpTPolBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE>,
        b_col: usize,
    );

    #[allow(clippy::too_many_arguments)]
    fn svp_apply_ppol_dft_to_dft_default(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, BE>,
        b_col: usize,
    );

    #[allow(clippy::too_many_arguments)]
    fn svp_apply_tpol_dft_to_dft_default(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &SvpTPolBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, BE>,
        b_col: usize,
    );

    fn svp_apply_ppol_dft_to_dft_assign_default(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, BE>,
        a_col: usize,
    );

    fn svp_apply_tpol_dft_to_dft_assign_default(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &SvpTPolBackendRef<'_, BE>,
        a_col: usize,
    );

    fn svp_ppol_copy_backend_default(
        _module: &Module<BE>,
        res: &mut SvpPPolBackendMut<'_, BE>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, BE>,
        a_col: usize,
    ) where
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        res.at_mut(res_col, 0).copy_from_slice(a.at(a_col, 0));
    }

    fn svp_tpol_copy_backend_default(
        _module: &Module<BE>,
        res: &mut SvpTPolBackendMut<'_, BE>,
        res_col: usize,
        a: &SvpTPolBackendRef<'_, BE>,
        a_col: usize,
    ) where
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        res.at_mut(res_col, 0).copy_from_slice(a.at(a_col, 0));
    }

    #[allow(clippy::too_many_arguments)]
    fn svp_apply_small_small_to_dft_default(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE>,
        b_col: usize,
    ) where
        BE::OwnedBuf: HostDataMut,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        let mut tpol = SvpTPolOwned::<BE>::alloc(module.n(), 1);
        Self::svp_prepare_tpol_default(module, &mut tpol.to_backend_mut(), 0, a, a_col);
        Self::svp_apply_tpol_small_to_dft_default(module, res, res_col, &tpol.to_backend_ref(), 0, b, b_col);
    }

    #[allow(clippy::too_many_arguments)]
    fn svp_apply_small_dft_to_dft_default(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, BE>,
        b_col: usize,
    ) where
        BE::OwnedBuf: HostDataMut,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        let mut tpol = SvpTPolOwned::<BE>::alloc(module.n(), 1);
        Self::svp_prepare_tpol_default(module, &mut tpol.to_backend_mut(), 0, a, a_col);
        Self::svp_apply_tpol_dft_to_dft_default(module, res, res_col, &tpol.to_backend_ref(), 0, b, b_col);
    }

    fn svp_apply_small_dft_to_dft_assign_default(
        module: &Module<BE>,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
    ) where
        BE::OwnedBuf: HostDataMut,
        for<'x> BE::BufMut<'x>: HostDataMut,
        for<'x> BE::BufRef<'x>: HostDataRef,
    {
        let mut tpol = SvpTPolOwned::<BE>::alloc(module.n(), 1);
        Self::svp_prepare_tpol_default(module, &mut tpol.to_backend_mut(), 0, a, a_col);
        Self::svp_apply_tpol_dft_to_dft_assign_default(module, res, res_col, &tpol.to_backend_ref(), 0);
    }
}

impl NTT3x42IfmaSvpDefault<NTT3x42Ifma> for NTT3x42Ifma {
    fn svp_prepare_ppol_default(
        module: &Module<NTT3x42Ifma>,
        res: &mut SvpPPolBackendMut<'_, NTT3x42Ifma>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, NTT3x42Ifma>,
        a_col: usize,
    ) {
        svp_prepare_ppol(module, res, res_col, a, a_col);
    }

    fn svp_prepare_tpol_default(
        module: &Module<NTT3x42Ifma>,
        res: &mut SvpTPolBackendMut<'_, NTT3x42Ifma>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, NTT3x42Ifma>,
        a_col: usize,
    ) {
        svp_prepare_tpol(module, res, res_col, a, a_col);
    }

    fn svp_apply_ppol_small_to_dft_default(
        module: &Module<NTT3x42Ifma>,
        res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, NTT3x42Ifma>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, NTT3x42Ifma>,
        b_col: usize,
    ) {
        svp_apply_ppol_small_to_dft(module, res, res_col, a, a_col, b, b_col);
    }

    fn svp_apply_tpol_small_to_dft_default(
        module: &Module<NTT3x42Ifma>,
        res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
        res_col: usize,
        a: &SvpTPolBackendRef<'_, NTT3x42Ifma>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, NTT3x42Ifma>,
        b_col: usize,
    ) {
        svp_apply_tpol_small_to_dft(module, res, res_col, a, a_col, b, b_col);
    }

    fn svp_apply_ppol_dft_to_dft_default(
        _module: &Module<NTT3x42Ifma>,
        res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, NTT3x42Ifma>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, NTT3x42Ifma>,
        b_col: usize,
    ) {
        svp_apply_ppol_dft_to_dft(res, res_col, a, a_col, b, b_col);
    }

    fn svp_apply_tpol_dft_to_dft_default(
        _module: &Module<NTT3x42Ifma>,
        res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
        res_col: usize,
        a: &SvpTPolBackendRef<'_, NTT3x42Ifma>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, NTT3x42Ifma>,
        b_col: usize,
    ) {
        svp_apply_tpol_dft_to_dft(res, res_col, a, a_col, b, b_col);
    }

    fn svp_apply_ppol_dft_to_dft_assign_default(
        _module: &Module<NTT3x42Ifma>,
        res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
        res_col: usize,
        a: &SvpPPolBackendRef<'_, NTT3x42Ifma>,
        a_col: usize,
    ) {
        svp_apply_ppol_dft_to_dft_assign(res, res_col, a, a_col);
    }

    fn svp_apply_tpol_dft_to_dft_assign_default(
        _module: &Module<NTT3x42Ifma>,
        res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
        res_col: usize,
        a: &SvpTPolBackendRef<'_, NTT3x42Ifma>,
        a_col: usize,
    ) {
        svp_apply_tpol_dft_to_dft_assign(res, res_col, a, a_col);
    }
}
