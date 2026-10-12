use poulpy_cpu_portable::kernels::ntt4x30::primes::Primes30;
use poulpy_hal::{
    layouts::{
        MaxBase2k, Module, PrimeSet, Ring, ScratchArena, VecZnxBackendMut, VecZnxBackendRef, VecZnxBigBackendMut,
        VecZnxDftBackendMut, VecZnxDftBackendRef, assert_dense, check_degree,
    },
    oep::HalVecZnxDftImpl,
};

use super::operand::{big_column_mut, column_mut, column_ref, dft_column_mut, dft_column_ref, empty_dft_like};
use crate::{
    NTT4x30Gpu,
    module::handle,
    ntt4x30::{AutomorphismPlan, DftColumn, DftTerm, automorphism, combine, forward, inverse, to_big, to_big_clearing},
    runtime::GpuRuntime,
    znx::{Column, KERNEL_RADIX, Term, lincomb, normalize, normalize_residues},
};

impl<RT: GpuRuntime, R: Ring> MaxBase2k for NTT4x30Gpu<RT, R> {
    fn max_base2k(n: usize, products: usize, failure_bits: usize, squaring: bool) -> usize {
        poulpy_hal::layouts::max_base2k_ntt::<Self>(<Primes30 as PrimeSet>::LOG_Q_PRODUCT, n, products, failure_bits, squaring)
    }

    fn max_base2k_squares(n: usize, products: usize, squares: usize, failure_bits: usize) -> usize {
        poulpy_hal::layouts::max_base2k_ntt_squares::<Self>(
            <Primes30 as PrimeSet>::LOG_Q_PRODUCT,
            n,
            products,
            squares,
            failure_bits,
        )
    }
}

/// How many of the first `res_size` selected limbs `offset, offset + step, ..` exist in a vector of `a_size` limbs.
fn selected_limbs(res_size: usize, a_size: usize, step: usize, offset: usize) -> usize {
    if offset >= a_size {
        0
    } else {
        res_size.min((a_size - offset).div_ceil(step))
    }
}

/// Zeroes the limbs of `res` from `first` on.
fn zero_from<RT: GpuRuntime>(res: &DftColumn, first: usize) {
    let tail = res.limbs(first, 1, res.size - first);
    combine::<RT>(&tail, tail.size, DftTerm::zero(&tail), DftTerm::zero(&tail));
}

/// `vec_znx_idft_normalize` of a column whose first inverse launch
/// has already run (`gglwe_product_digits_strided_inverse`): the remaining
/// launches, then the normalization.
#[cfg_attr(not(feature = "enable-core"), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn idft_tail_normalize<RT: GpuRuntime, R: Ring>(
    module: &Module<NTT4x30Gpu<RT, R>>,
    res: &mut VecZnxBackendMut<'_, NTT4x30Gpu<RT, R>>,
    res_base2k: usize,
    res_k: usize,
    res_offset: i64,
    res_col: usize,
    a: &mut VecZnxDftBackendMut<'_, NTT4x30Gpu<RT, R>>,
    a_col: usize,
    a_base2k: usize,
    addend: Option<(&VecZnxBackendRef<'_, NTT4x30Gpu<RT, R>>, usize)>,
) {
    assert_eq!(res.n(), a.n(), "idft_tail_normalize: res.n() != a.n()");
    let Some(res) = column_mut(res, res_col) else { return };
    let a = dft_column_mut(a, a_col).expect("idft_tail_normalize: a holds no storage");
    let addend = addend.and_then(|(addend, col)| {
        assert_eq!(addend.n(), a.n, "idft_tail_normalize: addend.n() != a.n()");
        column_ref(addend, col)
    });
    let tables = handle(module).tables();
    crate::ntt4x30::inverse_tail::<RT>(tables, &a, a.size);
    normalize_residues::<RT>(tables, &res, res_base2k, res_k, res_offset, &a, addend.as_ref(), a_base2k);
}

unsafe impl<RT: GpuRuntime, R: Ring> HalVecZnxDftImpl for NTT4x30Gpu<RT, R> {
    fn vec_znx_dft_apply(
        module: &Module<Self>,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, Self>,
        a_col: usize,
    ) {
        assert_dense(a, "vec_znx_dft_apply");
        assert!(step >= 1, "vec_znx_dft_apply: step must be >= 1");
        check_degree::<Self>(module.n(), res.n());
        assert!(a.n() == res.n(), "vec_znx_dft_apply: a.n() != res.n()");
        let Some(res) = dft_column_mut(res, res_col) else { return };
        let a = column_ref(a, a_col);
        let limbs = a.as_ref().map_or(0, |a| selected_limbs(res.size, a.size, step, offset));
        if let Some(a) = a.filter(|_| limbs != 0) {
            let narrow = |words: usize| u32::try_from(words).expect("source beyond the 32-bit word range");
            let selected = Column {
                base: narrow(a.base as usize + offset * a.stride as usize),
                stride: narrow(step * a.stride as usize),
                size: limbs,
                ..a
            };
            forward::<RT>(handle(module).tables(), &res, &selected, limbs);
        }
        zero_from::<RT>(&res, limbs);
    }

    fn vec_znx_idft_apply_tmp_bytes(_module: &Module<Self>) -> usize {
        0
    }

    fn vec_znx_idft_apply(
        module: &Module<Self>,
        res: &mut VecZnxBigBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        assert_dense(res, "vec_znx_idft_apply");
        check_degree::<Self>(module.n(), res.n());
        assert!(a.n() == res.n(), "vec_znx_idft_apply: a.n() != res.n()");
        let Some(res) = big_column_mut(res, res_col) else { return };
        let limbs = res.size.min(a.size());
        let Some(a) = dft_column_ref(a, a_col).filter(|_| limbs != 0) else {
            lincomb::<RT>(&res, res.size, Term::zero(&res), Term::zero(&res));
            return;
        };
        // `a` is left unchanged: the transform lands in the module's temporary.
        let handle = handle(module);
        let block = 4 * a.n;
        handle.with_temp(limbs * block * size_of::<u32>(), |temp| {
            let tmp = DftColumn {
                binding: temp.view().binding().expect("temporary holds storage"),
                base: 0,
                stride: block as u32,
                n: a.n,
                size: limbs,
            };
            inverse::<RT>(handle.tables(), &tmp, &a, limbs);
            to_big_clearing::<RT>(handle.tables(), &res, &tmp);
        });
    }

    fn vec_znx_idft_normalize_tmp_bytes(_module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
        0
    }

    fn vec_znx_idft_normalize(
        module: &Module<Self>,
        res: &mut VecZnxBackendMut<'_, Self>,
        res_base2k: usize,
        res_k: usize,
        res_offset: i64,
        res_col: usize,
        a: &mut VecZnxDftBackendMut<'_, Self>,
        a_col: usize,
        a_base2k: usize,
        addend: Option<(&VecZnxBackendRef<'_, Self>, usize)>,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        check_degree::<Self>(module.n(), a.n());
        assert!(res.n() == a.n(), "vec_znx_idft_normalize: res.n() != a.n()");
        let Some(res) = column_mut(res, res_col) else { return };
        let a = dft_column_mut(a, a_col).expect("vec_znx_idft_normalize: a holds no storage");
        let addend = addend.and_then(|(addend, col)| {
            assert!(addend.n() == a.n, "vec_znx_idft_normalize: addend.n() != a.n()");
            column_ref(addend, col)
        });
        let handle = handle(module);
        let tables = handle.tables();
        inverse::<RT>(tables, &a, &a, a.size);
        if res_base2k <= KERNEL_RADIX && a_base2k <= KERNEL_RADIX {
            normalize_residues::<RT>(tables, &res, res_base2k, res_k, res_offset, &a, addend.as_ref(), a_base2k);
            return;
        }
        // The radices the kernel does not handle: the reconstructed `i128`
        // values take the host stage of `normalize`.
        handle.with_temp(a.size * a.n * size_of::<i128>(), |temp| {
            let big = Column {
                binding: temp.view().binding().expect("temporary holds storage"),
                base: 0,
                stride: 2 * a.n as u32,
                words: 2,
                n: a.n,
                size: a.size,
            };
            to_big::<RT>(tables, &big, &a);
            if let Some(addend) = &addend {
                lincomb::<RT>(&big, big.size.min(addend.size), Term::plus(&big), Term::plus(addend));
            }
            normalize::<RT>(&res, res_base2k, res_k, res_offset, &big, a_base2k);
            // A host stage, at radices no kernel serves: a zeroing launch of its own.
            temp.wipe();
        });
    }

    fn vec_znx_dft_add(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, Self>,
        b_col: usize,
    ) {
        let Some(res) = dft_column_mut(res, res_col) else { return };
        let a = dft_column_ref(a, a_col).unwrap_or_else(|| empty_dft_like(&res));
        let b = dft_column_ref(b, b_col).unwrap_or_else(|| empty_dft_like(&res));
        combine::<RT>(&res, res.size, DftTerm::plus(&a), DftTerm::plus(&b));
    }

    fn vec_znx_dft_add_assign(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        let Some(res) = dft_column_mut(res, res_col) else { return };
        let Some(a) = dft_column_ref(a, a_col) else { return };
        combine::<RT>(&res, res.size.min(a.size), DftTerm::plus(&res), DftTerm::plus(&a));
    }

    fn vec_znx_dft_sub(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        b: &VecZnxDftBackendRef<'_, Self>,
        b_col: usize,
    ) {
        let Some(res) = dft_column_mut(res, res_col) else { return };
        let a = dft_column_ref(a, a_col).unwrap_or_else(|| empty_dft_like(&res));
        let b = dft_column_ref(b, b_col).unwrap_or_else(|| empty_dft_like(&res));
        combine::<RT>(&res, res.size, DftTerm::plus(&a), DftTerm::minus(&b));
    }

    fn vec_znx_dft_sub_assign(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        let Some(res) = dft_column_mut(res, res_col) else { return };
        let Some(a) = dft_column_ref(a, a_col) else { return };
        combine::<RT>(&res, res.size.min(a.size), DftTerm::plus(&res), DftTerm::minus(&a));
    }

    fn vec_znx_dft_sub_negate_assign(
        _module: &Module<Self>,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        let Some(res) = dft_column_mut(res, res_col) else { return };
        let a = dft_column_ref(a, a_col).unwrap_or_else(|| empty_dft_like(&res));
        combine::<RT>(&res, res.size, DftTerm::plus(&a), DftTerm::minus(&res));
    }

    fn vec_znx_dft_copy(
        _module: &Module<Self>,
        step: usize,
        offset: usize,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        assert!(res.n() == a.n(), "vec_znx_dft_copy: res.n() != a.n()");
        assert!(step >= 1, "vec_znx_dft_copy: step must be >= 1");
        let Some(res) = dft_column_mut(res, res_col) else { return };
        let a = dft_column_ref(a, a_col).unwrap_or_else(|| empty_dft_like(&res));
        let selected = a.limbs(offset, step, selected_limbs(res.size, a.size, step, offset));
        combine::<RT>(&res, res.size, DftTerm::plus(&selected), DftTerm::zero(&selected));
    }

    fn vec_znx_dft_zero(_module: &Module<Self>, res: &mut VecZnxDftBackendMut<'_, Self>, res_col: usize) {
        let Some(res) = dft_column_mut(res, res_col) else { return };
        zero_from::<RT>(&res, 0);
    }

    type AutomorphismPlan = AutomorphismPlan;

    fn vec_znx_dft_automorphism_plan(module: &Module<Self>, n: usize, p: i64) -> Self::AutomorphismPlan {
        check_degree::<Self>(module.n(), n);
        AutomorphismPlan::new(n, p, handle(module).tables().invariant())
    }

    fn vec_znx_dft_automorphism_with_plan(
        module: &Module<Self>,
        plan: &Self::AutomorphismPlan,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
    ) {
        check_degree::<Self>(module.n(), res.n());
        assert!(a.n() == res.n(), "vec_znx_dft_automorphism: a.n() != res.n()");
        let Some(res) = dft_column_mut(res, res_col) else { return };
        let a = dft_column_ref(a, a_col).unwrap_or_else(|| empty_dft_like(&res));
        automorphism::<RT>(plan, &res, &a, false);
    }

    fn vec_znx_dft_automorphism_add_with_plan_tmp_bytes(_module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
        0
    }

    fn vec_znx_dft_automorphism_add_with_plan(
        module: &Module<Self>,
        plan: &Self::AutomorphismPlan,
        res: &mut VecZnxDftBackendMut<'_, Self>,
        res_col: usize,
        a: &VecZnxDftBackendRef<'_, Self>,
        a_col: usize,
        _scratch: &mut ScratchArena<'_, Self>,
    ) {
        check_degree::<Self>(module.n(), res.n());
        assert!(a.n() == res.n(), "vec_znx_dft_automorphism_add: a.n() != res.n()");
        let Some(res) = dft_column_mut(res, res_col) else { return };
        let Some(a) = dft_column_ref(a, a_col) else { return };
        automorphism::<RT>(plan, &res, &a, true);
    }
}
