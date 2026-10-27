//! Thin arithmetic functions.

use anyhow::Result;
use poulpy_core::{
    GLWEBytesOf,
    layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetTensorKey},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use poulpy_ckks::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSCopyOps, CKKSMulAddOps, CKKSPow2Ops},
    layouts::ScratchArenaTakeCKKS,
};

/// Homomorphic arithmetic matching simple `libm` operations.
pub trait CKKSArithmeticOps<BE: Backend> {
    /// Scratch bytes for [`Self::ckks_fma`].
    fn ckks_fma_tmp_bytes<R, A, B, C, T>(&self, res: &R, a: &A, b: &B, c: &C, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        C: CKKSCtBounds,
        T: GGLWEInfos;

    /// Computes `a * b + c` through Poulpy's multiply-add path.
    ///
    /// # Metadata
    ///
    /// As `ckks_mul_add_ct_into` on an accumulator holding `c` at the layout of `res`.
    ///
    /// # Errors
    ///
    /// The errors of `ckks_copy` and `ckks_mul_add_ct_into`. The product accumulates in scratch, so `res` is untouched on
    /// `Err`.
    fn ckks_fma<R, A, B, C, H>(
        &self,
        res: &mut R,
        a: &A,
        b: &B,
        c: &C,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        C: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetTensorKey<BE>;

    /// Scratch bytes for [`Self::ckks_scalbn`] and [`Self::ckks_ldexp`].
    fn ckks_scalbn_tmp_bytes<R>(&self, res: &R) -> usize
    where
        R: CKKSCtBounds;

    /// Computes `input * 2^exponent` for a public integer exponent.
    ///
    /// # Metadata
    ///
    /// As `ckks_mul_pow2_into` for a non-negative exponent, as `ckks_div_pow2_into` for a negative one.
    ///
    /// # Errors
    ///
    /// The errors of that call, with `res` untouched.
    fn ckks_scalbn<R, I>(&self, exponent: i32, res: &mut R, input: &I, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        I: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Alias of [`Self::ckks_scalbn`].
    fn ckks_ldexp<R, I>(&self, exponent: i32, res: &mut R, input: &I, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        I: GLWEToBackendRef<BE> + CKKSCtBounds;
}

impl<BE: Backend> CKKSArithmeticOps<BE> for Module<BE>
where
    Module<BE>: CKKSCopyOps<BE> + CKKSMulAddOps<BE> + CKKSPow2Ops<BE> + GLWEBytesOf<BE>,
{
    fn ckks_fma_tmp_bytes<R, A, B, C, T>(&self, res: &R, a: &A, b: &B, c: &C, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        C: CKKSCtBounds,
        T: GGLWEInfos,
    {
        // The accumulator has the layout of `res`.
        BE::scratch_aligned(self.glwe_bytes_of_from_infos(res))
            + self
                .ckks_copy_tmp_bytes(res, c)
                .max(self.ckks_mul_add_ct_tmp_bytes(res, a, b, tsk))
                .max(self.ckks_copy_tmp_bytes(res, res))
    }

    fn ckks_fma<R, A, B, C, H>(&self, res: &mut R, a: &A, b: &B, c: &C, tsk: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        C: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetTensorKey<BE>,
    {
        scratch.scope(|scratch_local| {
            let (mut acc, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(&*res);
            self.ckks_copy(&mut acc, c, &mut scratch_local)?;
            self.ckks_mul_add_ct_into(&mut acc, a, b, tsk, &mut scratch_local)?;
            self.ckks_copy(res, &acc, &mut scratch_local)?;
            Ok(())
        })
    }

    fn ckks_scalbn_tmp_bytes<R>(&self, res: &R) -> usize
    where
        R: CKKSCtBounds,
    {
        self.ckks_mul_pow2_tmp_bytes(res.size())
            .max(self.ckks_div_pow2_tmp_bytes(res.size()))
    }

    fn ckks_scalbn<R, I>(&self, exponent: i32, res: &mut R, input: &I, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        let bits = exponent.unsigned_abs() as usize;
        if exponent >= 0 {
            self.ckks_mul_pow2_into(bits, res, input, scratch)?;
        } else {
            self.ckks_div_pow2_into(bits, res, input, scratch)?;
        }
        Ok(())
    }

    fn ckks_ldexp<R, I>(&self, exponent: i32, res: &mut R, input: &I, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        self.ckks_scalbn(exponent, res, input, scratch)
    }
}
