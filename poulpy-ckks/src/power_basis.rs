use std::collections::HashMap;

use anyhow::{Result, anyhow, ensure};
use poulpy_core::layouts::{
    GGLWEInfos, GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, prepared::GLWETensorKeyPreparedToBackendRef,
};
use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSInfos, SetCKKSInfos,
    api::{CKKSAddOps, CKKSMulOps, CKKSSubOps, PowerBasisHelper},
    checked_mul_ct_log_budget,
    layouts::{CKKSCiphertext, CKKSModuleAlloc, ScratchArenaTakeCKKS},
};

// Re-export so callers can use `polynomial::Basis`/`Parity` without reaching into `api`.
pub use crate::api::{Basis, Parity};

// ── PowerBasis ────────────────────────────────────────────────────────────────

/// Stores pre-computed powers of a ciphertext for BSGS polynomial evaluation.
///
/// `values[n]` = X^n (monomial basis) or Tₙ(X) (Chebyshev basis).
/// `values[1]` must be provided at construction time.
///
/// Implements [`PowerBasisHelper`] so it can be passed directly to
/// `ckks_eval_poly_real_const_coeffs_from_power_basis`.
pub struct PowerBasis<A> {
    basis: Basis,
    values: HashMap<usize, A>,
}

impl<A> PowerBasis<A> {
    /// Creates a power basis with `x` treated as X (or T₁(X) for Chebyshev).
    pub fn new(basis: Basis, x: A) -> Self {
        let mut values = HashMap::new();
        values.insert(1, x);
        Self { basis, values }
    }

    /// Returns the polynomial basis represented by the stored powers.
    pub fn basis(&self) -> Basis {
        self.basis
    }

    /// Returns a reference to the stored power at degree `n`, if computed.
    pub fn get_stored(&self, n: usize) -> Option<&A> {
        self.values.get(&n)
    }
}

impl<BE: Backend, A> PowerBasisHelper<BE, A> for PowerBasis<A>
where
    A: GLWEToBackendRef<BE>,
{
    fn basis(&self) -> Basis {
        self.basis
    }

    fn has_power(&self, power: usize) -> bool {
        self.values.contains_key(&power)
    }

    fn get(&self, power: usize) -> Result<&A> {
        self.values
            .get(&power)
            .ok_or_else(|| anyhow!("PowerBasis: X^{power} not computed; call gen_power or populate first"))
    }
}

impl<D: Data> PowerBasis<CKKSCiphertext<D>> {
    /// Inserts a caller-provided pre-computed ciphertext power.
    ///
    /// This checks that the ciphertext storage is compatible with the
    /// degree-one power created at construction time. The caller is still
    /// responsible for the semantic invariant: `value` must encrypt `X^n` for a
    /// monomial basis or `T_n(X)` for a Chebyshev basis, derived from the same
    /// input ciphertext and compatible CKKS metadata.
    pub fn insert(&mut self, n: usize, value: CKKSCiphertext<D>) -> Result<()> {
        ensure!(
            n >= 2,
            "PowerBasis::insert: power must be at least 2; power 1 is set at construction"
        );
        ensure!(
            !self.values.contains_key(&n),
            "PowerBasis::insert: power {n} is already present"
        );

        let (expected_n, expected_base2k, expected_rank) = {
            let x = self
                .values
                .get(&1)
                .expect("PowerBasis::new always stores the degree-one power");
            (x.n(), x.base2k(), x.rank())
        };
        ensure!(
            value.n() == expected_n,
            "PowerBasis::insert: power {n} ring degree {} does not match degree-one ring degree {}",
            value.n(),
            expected_n
        );
        ensure!(
            value.base2k() == expected_base2k,
            "PowerBasis::insert: power {n} base2k {} does not match degree-one base2k {}",
            value.base2k(),
            expected_base2k
        );
        ensure!(
            value.rank() == expected_rank,
            "PowerBasis::insert: power {n} rank {} does not match degree-one rank {}",
            value.rank(),
            expected_rank
        );

        self.values.insert(n, value);
        Ok(())
    }

    /// Recursively computes and stores X^`n` using `split_degree` to choose the
    /// multiplication tree: X^n = X^a · X^b where `split_degree(n) = (a, b)`.
    pub fn gen_power<BE, T>(&mut self, n: usize, module: &Module<BE>, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        BE: Backend<OwnedBuf = D>,
        Module<BE>: CKKSMulOps<BE> + CKKSModuleAlloc<BE>,
        CKKSCiphertext<D>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        ensure!(
            self.basis == Basis::Monomial,
            "PowerBasis::gen_power only supports the monomial basis; use gen_power_chebyshev for Chebyshev"
        );

        if self.values.contains_key(&n) {
            return Ok(());
        }

        ensure!(n >= 2, "gen_power: n={n} < 2; X^1 must be provided at construction");

        let (a, b) = split_degree(n);
        self.gen_power(a, module, tsk, scratch)?;
        self.gen_power(b, module, tsk, scratch)?;

        // Hold immutable borrows only inside this block; insert afterwards.
        let result = {
            let a_val = self.values.get(&a).expect("gen_power(a) just succeeded");
            let b_val = self.values.get(&b).expect("gen_power(b) just succeeded");
            let k = mul_ct_effective_k(a_val, b_val)?;
            let mut r = module.ckks_ciphertext_alloc(a_val.base2k(), k.into());
            module.ckks_mul_into(&mut r, a_val, b_val, tsk, scratch)?;
            r
        };
        self.values.insert(n, result);
        Ok(())
    }

    /// Recursively computes and stores `T_n(X)` for the Chebyshev basis.
    ///
    /// Generates the plaintext `T_0 = 1` term on demand for
    /// `T_{a+b}(X) = 2*T_a(X)*T_b(X) - T_{|a-b|}(X)`.
    pub fn gen_power_chebyshev<BE, T>(
        &mut self,
        n: usize,
        module: &Module<BE>,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: Backend<OwnedBuf = D>,
        Module<BE>: CKKSAddOps<BE> + CKKSMulOps<BE> + CKKSSubOps<BE> + CKKSModuleAlloc<BE>,
        CKKSCiphertext<D>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        ensure!(
            self.basis == Basis::Chebyshev,
            "gen_power_chebyshev requires a Chebyshev PowerBasis"
        );

        if self.values.contains_key(&n) {
            return Ok(());
        }

        ensure!(n >= 2, "gen_power_chebyshev: n={n} < 2; T_1 must be provided at construction");

        let (a, b) = split_degree(n);
        self.gen_power_chebyshev(a, module, tsk, scratch)?;
        self.gen_power_chebyshev(b, module, tsk, scratch)?;

        let c = a.abs_diff(b);
        if c != 0 {
            self.gen_power_chebyshev(c, module, tsk, scratch)?;
        }

        let result = scratch.scope(|scratch| -> Result<CKKSCiphertext<D>> {
            let a_val = self.values.get(&a).expect("gen_power_chebyshev(a) just succeeded");
            let b_val = self.values.get(&b).expect("gen_power_chebyshev(b) just succeeded");
            let k = mul_ct_effective_k(a_val, b_val)?;
            let product_layout = GLWELayout {
                n: a_val.n(),
                base2k: a_val.base2k(),
                k: k.into(),
                rank: a_val.rank(),
            };
            let (mut product, mut scratch) = scratch.take_ckks_ciphertext_scratch(&product_layout, a_val.meta());
            module.ckks_mul_into(&mut product, a_val, b_val, tsk, &mut scratch)?;

            let mut doubled = module.ckks_ciphertext_alloc(product.base2k(), product.effective_k().into());
            module.ckks_add_into(&mut doubled, &product, &product, &mut scratch)?;

            if c == 0 {
                module.ckks_sub_one_assign(&mut doubled, &mut scratch)?;
            } else {
                let c_val = self.values.get(&c).expect("gen_power_chebyshev(c) just succeeded");
                module.ckks_sub_assign(&mut doubled, c_val, &mut scratch)?;
            }

            Ok(doubled)
        })?;

        self.values.insert(n, result);
        Ok(())
    }

    /// Pre-computes all powers required to evaluate a polynomial of the given
    /// `degree` using BSGS, for the basis stored in `self`.
    ///
    /// `log_split` is the baby-step split (`base = 2^log_split`); read it off
    /// the BSGSPolynomial the caller will evaluate via
    /// `bsgs.log_split()`.
    ///
    /// `parity` should match the polynomial to be evaluated:
    /// - [`Parity::Even`]: only even baby-step powers are needed (skip 3, 5, 7, …).
    /// - [`Parity::Odd`]:  only odd baby-step powers are needed (skip 4, 6, 8, …).
    /// - [`Parity::Full`]: all baby-step powers from 3 through
    ///   `min(degree, 2^log_split − 1)`.
    ///
    /// Giant-step powers of two up to `2^(⌈log₂ degree⌉−1)` are always computed.
    pub fn populate<BE, T>(
        &mut self,
        degree: usize,
        log_split: usize,
        parity: Parity,
        module: &Module<BE>,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: Backend<OwnedBuf = D>,
        Module<BE>: CKKSAddOps<BE> + CKKSMulOps<BE> + CKKSSubOps<BE> + CKKSModuleAlloc<BE>,
        CKKSCiphertext<D>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        ensure!(degree >= 1, "populate: degree must be ≥ 1");

        let log_degree = (usize::BITS - degree.leading_zeros()) as usize;
        // Giant-step powers of two (also computes all smaller powers of two recursively).
        let largest_pow2 = 1usize << (log_degree - 1);
        let base = 1usize << log_split;

        macro_rules! gen_pow {
            ($n:expr) => {
                match self.basis {
                    Basis::Monomial => self.gen_power($n, module, tsk, scratch),
                    Basis::Chebyshev => self.gen_power_chebyshev($n, module, tsk, scratch),
                }
            };
        }

        if largest_pow2 >= 2 {
            gen_pow!(largest_pow2)?;
        }

        // Baby-step intermediate powers: skip parities not used by the evaluator.
        // Power 2 is always computed transitively (gen_power(3) or gen_power(largest_pow2)).
        let baby_limit = base.min(degree + 1);
        match parity {
            Parity::Even => {
                for i in (4..baby_limit).step_by(2) {
                    gen_pow!(i)?;
                }
            }
            Parity::Odd => {
                for i in (3..baby_limit).step_by(2) {
                    gen_pow!(i)?;
                }
            }
            Parity::Full => {
                for i in (3..baby_limit).rev() {
                    gen_pow!(i)?;
                }
            }
        }

        Ok(())
    }
}

fn mul_ct_effective_k<A, B>(a: &A, b: &B) -> Result<usize>
where
    A: GLWEInfos + CKKSInfos,
    B: GLWEInfos + CKKSInfos,
{
    let log_budget = checked_mul_ct_log_budget("power_basis", a.log_budget(), b.log_budget(), a.log_delta(), b.log_delta())?;
    Ok(log_budget + a.log_delta().min(b.log_delta()))
}

/// Splits `n` into `(a, b)` with `n = a + b` and `|a – b|` minimised.
///
/// When `n` is a power of two `a = b = n/2`; otherwise uses the
/// Lee et al. (2020) strategy that maximises the number of odd-degree
/// Chebyshev terms.
pub(crate) fn split_degree(n: usize) -> (usize, usize) {
    assert!(n > 1);
    if n.is_power_of_two() {
        (n / 2, n / 2)
    } else {
        let k = (usize::BITS - (n - 1).leading_zeros()) as usize - 1;
        let a = (1usize << k) - 1;
        let b = n + 1 - (1usize << k);
        (a, b)
    }
}
