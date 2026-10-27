//! Reference body of the add-many family: the selected additions and copy, with poulpy-core's normalization at the
//! caller's cadence.
use poulpy_core::{
    GLWENormalize,
    layouts::{GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSResult as Result, SetCKKSInfos,
    api::{CKKSAddOps, CKKSCopyOps},
    ckks_ensure,
};

/// Reference bodies of [`CKKSAddManyImpl`](crate::oep::CKKSAddManyImpl), forwarded by
/// `impl_ckks_add_many_reference_full!`.
pub trait CKKSAddManyReference<BE: Backend> {
    /// The largest of the selected `ckks_add_tmp_bytes` at `res.max_size()`, poulpy-core's `glwe_normalize_tmp_bytes`
    /// and the selected `ckks_copy_tmp_bytes` from `a` to `res`, the copy serving a single term.
    fn ckks_add_many_tmp_bytes_reference<R, A>(&self, res: &R, a: &A) -> usize
    where
        Self: CKKSAddOps<BE> + CKKSCopyOps<BE> + GLWENormalize<BE>,
        R: CKKSCtBounds,
        A: CKKSCtBounds,
    {
        self.ckks_add_tmp_bytes(res.max_size())
            .max(self.glwe_normalize_tmp_bytes())
            .max(self.ckks_copy_tmp_bytes(res, a))
    }

    /// A zero `normalize_after`, then an empty `a`, then a term of another degree, radix or rank than `res`, is a
    /// refusal, then an arena below [`Self::ckks_add_many_tmp_bytes_reference`]. One term is the selected `ckks_copy`. Two or more are `ckks_add_into`
    /// of the first two terms, then `ckks_add_assign` of each later one, poulpy-core's `glwe_normalize_assign` of `res`
    /// running before every addition that follows a multiple of `normalize_after` additions. The noise record is
    /// cleared in every case.
    fn ckks_add_many_reference<R, A>(
        &self,
        normalize_after: usize,
        res: &mut R,
        a: &[&A],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSAddOps<BE> + CKKSCopyOps<BE> + GLWENormalize<BE>,
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        const OP: &str = "ckks_add_many";
        let invalid = |reason: &str| -> Result<()> {
            Err(CKKSCompositionError::InvalidOperand {
                op: OP,
                reason: reason.into(),
            }
            .into())
        };
        if normalize_after == 0 {
            return invalid("normalize_after must be at least 1");
        }
        let [first, rest @ ..] = a else {
            return invalid("a must contain at least one ciphertext");
        };
        // Every addition panics in core on a mismatched term, so all are checked before the first write.
        for (i, term) in a.iter().enumerate() {
            if term.n() != res.n() {
                return invalid(&format!("term {i} has degree {} where res has {}", term.n(), res.n()));
            }
            if term.base2k() != res.base2k() {
                return invalid(&format!(
                    "term {i} has radix {} where res has {}",
                    term.base2k(),
                    res.base2k()
                ));
            }
            if term.rank() != res.rank() {
                return invalid(&format!("term {i} has rank {} where res has {}", term.rank(), res.rank()));
            }
        }
        ckks_ensure!(
            scratch.available() >= self.ckks_add_many_tmp_bytes_reference(&*res, *first),
            "{OP}: scratch below its query"
        );
        match rest {
            [] => self.ckks_copy(res, *first, scratch)?,
            [second, rest @ ..] => {
                self.ckks_add_into(res, *first, *second, scratch)?;
                for (additions, term) in (1..).zip(rest) {
                    if additions % normalize_after == 0 {
                        self.glwe_normalize_assign(res, scratch);
                    }
                    self.ckks_add_assign(res, *term, scratch)?;
                }
            }
        }
        res.set_noise(None);
        Ok(())
    }
}

impl<BE: Backend> CKKSAddManyReference<BE> for Module<BE> {}
