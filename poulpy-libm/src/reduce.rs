//! Slot reductions.

use anyhow::{Result, ensure};
use poulpy_core::layouts::{
    BSGSMeta, GGLWEInfos, GGLWEPreparedToBackendRef, GLWE, GLWEAutomorphismKeyHelper,
    GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement, SetBSGSMeta,
    prepared::{GLWEAutomorphismKeyPreparedToBackendRef, GLWETensorKeyPreparedToBackendRef},
};
use poulpy_hal::layouts::{Backend, HostStaged, Module, ScratchArena};

use poulpy_ckks::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSAddOps, CKKSCopyOps, CKKSRotateOps},
    layouts::{CKKSCiphertext, CKKSModuleAlloc, ScratchArenaTakeCKKS},
};

use crate::comparison::{CKKSComparisonOps, SignComposite};

/// Full-slot sum and extrema.
pub trait CKKSReductionOps<BE: Backend> {
    /// Scratch bytes for [`Self::ckks_sum_slots`].
    fn ckks_sum_slots_tmp_bytes<R, A>(&self, ct: &R, rotation_key: &A) -> usize
    where
        R: CKKSCtBounds,
        A: GGLWEInfos;

    /// Sums all dense slots and broadcasts the result.
    fn ckks_sum_slots<R, I, H, K>(
        &self,
        res: &mut R,
        input: &I,
        rotation_keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE>
            + GGLWEPreparedToBackendRef<BE>
            + GetGaloisElement
            + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    /// Scratch bytes for slot extrema reductions.
    fn ckks_fmax_slots_tmp_bytes<R, T, A, P>(
        &self,
        ct: &R,
        tsk: &T,
        automorphism_key: &A,
        coeffs: &P,
    ) -> usize
    where
        R: CKKSCtBounds,
        T: GGLWEInfos,
        A: GGLWEInfos,
        P: poulpy_ckks::CKKSInfos + poulpy_core::layouts::LWEInfos;

    /// Reduces `fmax` over all dense slots and broadcasts the result.
    #[allow(clippy::too_many_arguments)]
    fn ckks_fmax_slots<F, P, H, RK, CK>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        input: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        composite: &SignComposite<F, P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        rotation_keys: &H,
        conj_key: &CK,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: HostStaged,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
        RK: GLWEAutomorphismKeyPreparedToBackendRef<BE>
            + GGLWEPreparedToBackendRef<BE>
            + GetGaloisElement
            + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<RK, BE>,
        CK: GLWEAutomorphismKeyPreparedToBackendRef<BE>
            + GGLWEPreparedToBackendRef<BE>
            + GetGaloisElement
            + GGLWEInfos;

    /// Reduces `fmin` over all dense slots and broadcasts the result.
    #[allow(clippy::too_many_arguments)]
    fn ckks_fmin_slots<F, P, H, RK, CK>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        input: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        composite: &SignComposite<F, P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        rotation_keys: &H,
        conj_key: &CK,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: HostStaged,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
        RK: GLWEAutomorphismKeyPreparedToBackendRef<BE>
            + GGLWEPreparedToBackendRef<BE>
            + GetGaloisElement
            + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<RK, BE>,
        CK: GLWEAutomorphismKeyPreparedToBackendRef<BE>
            + GGLWEPreparedToBackendRef<BE>
            + GetGaloisElement
            + GGLWEInfos;
}

impl<BE: Backend> CKKSReductionOps<BE> for Module<BE>
where
    Module<BE>: CKKSAddOps<BE>
        + CKKSComparisonOps<BE>
        + CKKSCopyOps<BE>
        + CKKSModuleAlloc<BE>
        + CKKSRotateOps<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
{
    fn ckks_sum_slots_tmp_bytes<R, A>(&self, ct: &R, rotation_key: &A) -> usize
    where
        R: CKKSCtBounds,
        A: GGLWEInfos,
    {
        GLWE::<Vec<u8>, BE::ZnxWord>::bytes_of_from_infos(ct)
            + self
                .ckks_rotate_tmp_bytes(ct, rotation_key)
                .max(self.ckks_add_tmp_bytes())
                .max(self.ckks_copy_tmp_bytes())
    }

    fn ckks_sum_slots<R, I, H, K>(
        &self,
        res: &mut R,
        input: &I,
        rotation_keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE>
            + GGLWEPreparedToBackendRef<BE>
            + GetGaloisElement
            + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        ensure_dense("ckks_sum_slots", input)?;
        self.ckks_copy(res, input, scratch)?;
        scratch.scope(|scratch_local| {
            let (mut rotated, mut scratch_local) =
                scratch_local.take_ckks_ciphertext_like_scratch(input);
            for shift in slot_shifts(input) {
                self.ckks_rotate_into(
                    &mut rotated,
                    &*res,
                    shift,
                    rotation_keys,
                    &mut scratch_local,
                )?;
                self.ckks_add_assign(res, &rotated, &mut scratch_local)?;
            }
            Ok(())
        })
    }

    fn ckks_fmax_slots_tmp_bytes<R, T, A, P>(
        &self,
        ct: &R,
        tsk: &T,
        automorphism_key: &A,
        coeffs: &P,
    ) -> usize
    where
        R: CKKSCtBounds,
        T: GGLWEInfos,
        A: GGLWEInfos,
        P: poulpy_ckks::CKKSInfos + poulpy_core::layouts::LWEInfos,
    {
        2 * GLWE::<Vec<u8>, BE::ZnxWord>::bytes_of_from_infos(ct)
            + self
                .ckks_rotate_tmp_bytes(ct, automorphism_key)
                .max(self.ckks_comparison_tmp_bytes(ct, tsk, automorphism_key, coeffs))
                .max(self.ckks_copy_tmp_bytes())
    }

    fn ckks_fmax_slots<F, P, H, RK, CK>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        input: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        composite: &SignComposite<F, P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        rotation_keys: &H,
        conj_key: &CK,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: HostStaged,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
        RK: GLWEAutomorphismKeyPreparedToBackendRef<BE>
            + GGLWEPreparedToBackendRef<BE>
            + GetGaloisElement
            + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<RK, BE>,
        CK: GLWEAutomorphismKeyPreparedToBackendRef<BE>
            + GGLWEPreparedToBackendRef<BE>
            + GetGaloisElement
            + GGLWEInfos,
    {
        extrema_slots(
            self,
            res,
            input,
            composite,
            tsk,
            rotation_keys,
            conj_key,
            true,
            scratch,
        )
    }

    fn ckks_fmin_slots<F, P, H, RK, CK>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        input: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        composite: &SignComposite<F, P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        rotation_keys: &H,
        conj_key: &CK,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: HostStaged,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
        RK: GLWEAutomorphismKeyPreparedToBackendRef<BE>
            + GGLWEPreparedToBackendRef<BE>
            + GetGaloisElement
            + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<RK, BE>,
        CK: GLWEAutomorphismKeyPreparedToBackendRef<BE>
            + GGLWEPreparedToBackendRef<BE>
            + GetGaloisElement
            + GGLWEInfos,
    {
        extrema_slots(
            self,
            res,
            input,
            composite,
            tsk,
            rotation_keys,
            conj_key,
            false,
            scratch,
        )
    }
}

fn ensure_dense(op: &'static str, input: &impl CKKSCtBounds) -> Result<()> {
    ensure!(
        input.log_sparsity() == 0,
        "{op}: sparse packing is not supported"
    );
    Ok(())
}

fn slot_shifts(input: &impl CKKSCtBounds) -> impl Iterator<Item = i64> {
    let log_slots = input.log_n() - 1;
    (0..log_slots).map(|i| 1i64 << i)
}

#[allow(clippy::too_many_arguments)]
fn extrema_slots<BE, F, P, H, RK, CK>(
    module: &Module<BE>,
    res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
    input: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
    composite: &SignComposite<F, P>,
    tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
    rotation_keys: &H,
    conj_key: &CK,
    max: bool,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + HostStaged,
    Module<BE>: CKKSComparisonOps<BE> + CKKSCopyOps<BE> + CKKSRotateOps<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
    RK: GLWEAutomorphismKeyPreparedToBackendRef<BE>
        + GGLWEPreparedToBackendRef<BE>
        + GetGaloisElement
        + GGLWEInfos,
    H: GLWEAutomorphismKeyHelper<RK, BE>,
    CK: GLWEAutomorphismKeyPreparedToBackendRef<BE>
        + GGLWEPreparedToBackendRef<BE>
        + GetGaloisElement
        + GGLWEInfos,
{
    ensure_dense(
        if max {
            "ckks_fmax_slots"
        } else {
            "ckks_fmin_slots"
        },
        input,
    )?;
    module.ckks_copy(res, input, scratch)?;
    scratch.scope(|scratch_local| {
        let (mut rotated, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(input);
        let (mut next, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(input);
        for shift in slot_shifts(input) {
            module.ckks_rotate_into(
                &mut rotated,
                &*res,
                shift,
                rotation_keys,
                &mut scratch_local,
            )?;
            if max {
                module.ckks_fmax(
                    &mut next,
                    &*res,
                    &rotated,
                    composite,
                    tsk,
                    conj_key,
                    &mut scratch_local,
                )?;
            } else {
                module.ckks_fmin(
                    &mut next,
                    &*res,
                    &rotated,
                    composite,
                    tsk,
                    conj_key,
                    &mut scratch_local,
                )?;
            }
            module.ckks_copy(res, &next, &mut scratch_local)?;
        }
        Ok(())
    })
}
