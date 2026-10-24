//! Thin arithmetic functions.

use anyhow::Result;
use poulpy_core::layouts::{
    GGLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef,
    prepared::GLWETensorKeyPreparedToBackendRef,
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use poulpy_ckks::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSCopyOps, CKKSMulAddOps, CKKSPow2Ops},
    layouts::CKKSCiphertext,
};

/// Homomorphic arithmetic matching simple `libm` operations.
pub trait CKKSArithmeticOps<BE: Backend> {
    /// Scratch bytes for [`Self::ckks_fma`].
    fn ckks_fma_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        T: GGLWEInfos;

    /// Computes `a * b + c` through Poulpy's multiply-add path.
    fn ckks_fma(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        a: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        b: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        c: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;

    /// Scratch bytes for [`Self::ckks_scalbn`] and [`Self::ckks_ldexp`].
    fn ckks_scalbn_tmp_bytes(&self) -> usize;

    /// Computes `input * 2^exponent` for a public integer exponent.
    fn ckks_scalbn(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        input: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        exponent: i32,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;

    /// Alias of [`Self::ckks_scalbn`].
    fn ckks_ldexp(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        input: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        exponent: i32,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;
}

impl<BE: Backend> CKKSArithmeticOps<BE> for Module<BE>
where
    Module<BE>: CKKSCopyOps<BE> + CKKSMulAddOps<BE> + CKKSPow2Ops<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
{
    fn ckks_fma_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        T: GGLWEInfos,
    {
        self.ckks_copy_tmp_bytes()
            .max(self.ckks_mul_add_ct_tmp_bytes(res, res, res, tsk))
    }

    fn ckks_fma(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        a: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        b: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        c: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()> {
        self.ckks_copy(res, c, scratch)?;
        self.ckks_mul_add_ct_into(res, a, b, tsk, scratch)?;
        Ok(())
    }

    fn ckks_scalbn_tmp_bytes(&self) -> usize {
        self.ckks_mul_pow2_tmp_bytes()
            .max(self.ckks_div_pow2_tmp_bytes())
    }

    fn ckks_scalbn(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        input: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        exponent: i32,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()> {
        let bits = exponent.unsigned_abs() as usize;
        if exponent >= 0 {
            self.ckks_mul_pow2_into(res, input, bits, scratch)?;
        } else {
            self.ckks_div_pow2_into(res, input, bits, scratch)?;
        }
        Ok(())
    }

    fn ckks_ldexp(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        input: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        exponent: i32,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()> {
        self.ckks_scalbn(res, input, exponent, scratch)
    }
}
