use anyhow::{Result, ensure};
use poulpy_core::layouts::{
    GGLWEInfos, GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToBackendRef, GLWETensorKeyPrepared, LWEInfos, split_degree,
    prepared::GLWETensorKeyPreparedToBackendRef,
};
use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSInfos, SetCKKSInfos,
    api::{CKKSAddOps, CKKSMulOps, CKKSSubOps},
    checked_mul_ct_log_budget,
    layouts::{CKKSCiphertext, CKKSModuleAlloc, ScratchArenaTakeCKKS},
};

pub use crate::api::{Basis, Parity};
pub use poulpy_core::layouts::{PowerBasis, PowerBasisHelper};

/// CKKS computation of the power basis entries used by BSGS evaluation.
pub trait PowerBasisGen<D: Data> {
    /// Inserts a caller-provided pre-computed ciphertext power.
    fn insert(&mut self, n: usize, value: CKKSCiphertext<D>) -> Result<()>;

    /// Recursively computes and stores X^`n` (monomial basis).
    fn gen_power<BE>(
        &mut self,
        n: usize,
        module: &Module<BE>,
        tsk: &GLWETensorKeyPrepared<D, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: Backend<OwnedBuf = D>,
        Module<BE>: CKKSMulOps<BE> + CKKSModuleAlloc<BE>,
        CKKSCiphertext<D>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        GLWETensorKeyPrepared<D, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos;

    /// Recursively computes and stores `T_n(X)` (Chebyshev basis).
    fn gen_power_chebyshev<BE>(
        &mut self,
        n: usize,
        module: &Module<BE>,
        tsk: &GLWETensorKeyPrepared<D, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: Backend<OwnedBuf = D>,
        Module<BE>: CKKSAddOps<BE> + CKKSMulOps<BE> + CKKSSubOps<BE> + CKKSModuleAlloc<BE>,
        CKKSCiphertext<D>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        GLWETensorKeyPrepared<D, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos;

    /// Pre-computes all powers required to evaluate a polynomial of the given
    /// `degree` using BSGS, for the basis stored in `self`.
    fn populate<BE>(
        &mut self,
        degree: usize,
        log_split: usize,
        parity: Parity,
        module: &Module<BE>,
        tsk: &GLWETensorKeyPrepared<D, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: Backend<OwnedBuf = D>,
        Module<BE>: CKKSAddOps<BE> + CKKSMulOps<BE> + CKKSSubOps<BE> + CKKSModuleAlloc<BE>,
        CKKSCiphertext<D>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        GLWETensorKeyPrepared<D, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos;
}

impl<D: Data> PowerBasisGen<D> for PowerBasis<CKKSCiphertext<D>> {
    fn insert(&mut self, n: usize, value: CKKSCiphertext<D>) -> Result<()> {
        ensure!(
            n >= 2,
            "PowerBasis::insert: power must be at least 2; power 1 is set at construction"
        );
        ensure!(
            !self.contains_power(n),
            "PowerBasis::insert: power {n} is already present"
        );

        let (expected_n, expected_base2k, expected_rank) = {
            let x = self
                .get_stored(1)
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

        self.set_power(n, value);
        Ok(())
    }

    fn gen_power<BE>(
        &mut self,
        n: usize,
        module: &Module<BE>,
        tsk: &GLWETensorKeyPrepared<D, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: Backend<OwnedBuf = D>,
        Module<BE>: CKKSMulOps<BE> + CKKSModuleAlloc<BE>,
        CKKSCiphertext<D>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        GLWETensorKeyPrepared<D, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        ensure!(
            self.basis() == Basis::Monomial,
            "PowerBasis::gen_power only supports the monomial basis; use gen_power_chebyshev for Chebyshev"
        );

        if self.contains_power(n) {
            return Ok(());
        }

        ensure!(n >= 2, "gen_power: n={n} < 2; X^1 must be provided at construction");

        let (a, b) = split_degree(n);
        self.gen_power(a, module, tsk, scratch)?;
        self.gen_power(b, module, tsk, scratch)?;

        // Hold immutable borrows only inside this block; insert afterwards.
        let result = {
            let a_val = self.get_stored(a).expect("gen_power(a) just succeeded");
            let b_val = self.get_stored(b).expect("gen_power(b) just succeeded");
            let k = mul_ct_effective_k(a_val, b_val)?;
            let mut r = module.ckks_ciphertext_alloc(a_val.base2k(), k.into());
            module.ckks_mul_into(&mut r, a_val, b_val, tsk, scratch)?;
            r
        };
        self.set_power(n, result);
        Ok(())
    }

    fn gen_power_chebyshev<BE>(
        &mut self,
        n: usize,
        module: &Module<BE>,
        tsk: &GLWETensorKeyPrepared<D, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: Backend<OwnedBuf = D>,
        Module<BE>: CKKSAddOps<BE> + CKKSMulOps<BE> + CKKSSubOps<BE> + CKKSModuleAlloc<BE>,
        CKKSCiphertext<D>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        GLWETensorKeyPrepared<D, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        ensure!(
            self.basis() == Basis::Chebyshev,
            "gen_power_chebyshev requires a Chebyshev PowerBasis"
        );

        if self.contains_power(n) {
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
            let a_val = self.get_stored(a).expect("gen_power_chebyshev(a) just succeeded");
            let b_val = self.get_stored(b).expect("gen_power_chebyshev(b) just succeeded");
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
                let c_val = self.get_stored(c).expect("gen_power_chebyshev(c) just succeeded");
                module.ckks_sub_assign(&mut doubled, c_val, &mut scratch)?;
            }

            Ok(doubled)
        })?;

        self.set_power(n, result);
        Ok(())
    }

    fn populate<BE>(
        &mut self,
        degree: usize,
        log_split: usize,
        parity: Parity,
        module: &Module<BE>,
        tsk: &GLWETensorKeyPrepared<D, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: Backend<OwnedBuf = D>,
        Module<BE>: CKKSAddOps<BE> + CKKSMulOps<BE> + CKKSSubOps<BE> + CKKSModuleAlloc<BE>,
        CKKSCiphertext<D>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        GLWETensorKeyPrepared<D, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        ensure!(degree >= 1, "populate: degree must be ≥ 1");

        let log_degree = (usize::BITS - degree.leading_zeros()) as usize;
        let largest_pow2 = 1usize << (log_degree - 1);
        let base = 1usize << log_split;

        macro_rules! gen_pow {
            ($n:expr) => {
                match self.basis() {
                    Basis::Monomial => self.gen_power($n, module, tsk, scratch),
                    Basis::Chebyshev => self.gen_power_chebyshev($n, module, tsk, scratch),
                }
            };
        }

        if largest_pow2 >= 2 {
            gen_pow!(largest_pow2)?;
        }

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
