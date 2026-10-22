//! Read-only views of the first digits of a key or a GGSW.

use poulpy_hal::layouts::Backend;

use crate::{
    error::{CoreError, Result},
    layouts::{
        Base2K, Degree, Dnum, GGLWEBackendRef, GGLWEInfos, GGLWEToBackendRef, GGSWBackendRef, GGSWInfos, GGSWToBackendRef,
        GLWEInfos, GetGaloisElement, LWEInfos, Rank, TorusPrecision,
    },
};

pub(crate) fn check_dnum(op: &'static str, dnum: Dnum, stored: Dnum) -> Result<()> {
    if dnum.0 == 0 || dnum.0 > stored.0 {
        return Err(CoreError::GGLWEKeyUse {
            op,
            detail: format!("dnum={} is not in 1..={}", dnum.0, stored.0),
        });
    }
    Ok(())
}

/// The first `dnum` digits of a key, from [`GGLWEViewWithDigits`]: rows
/// `0..dnum` over the key's buffer, read at the precision
/// `k_offset + dnum * k_digit + k_aux` of a key of `dnum` digits and the same
/// guard. No row is copied, and its backend view is
/// [`GGLWEBackendRef::with_dnum`].
///
/// A switch decomposes the rows of the key it switches exactly, guard
/// included, so a key never covers a key of its own layout, but covers a view
/// of fewer digits; the result goes into a key of that many digits.
///
/// The rows of a view of fewer digits keep the key's bits below its
/// precision: it reports them non-canonical, so an operation reading them
/// normalizes each row at the view's width, and it has no noise estimate.
/// Preparation panics on such a view and serialization refuses it: normalize
/// its rows into a key of that layout first.
pub struct GGLWEDigits<'a, K> {
    key: &'a K,
    dnum: Dnum,
}

/// `key.view_with_digits(dnum)` reads a key as a key of its first `dnum`
/// digits, see [`GGLWEDigits`].
pub trait GGLWEViewWithDigits: GGLWEInfos + Sized {
    /// Fails unless `1 <= dnum <= self.dnum()`.
    fn view_with_digits(&self, dnum: Dnum) -> Result<GGLWEDigits<'_, Self>> {
        check_dnum("view_with_digits", dnum, self.dnum())?;
        Ok(GGLWEDigits { key: self, dnum })
    }
}

impl<K: GGLWEInfos> GGLWEViewWithDigits for K {}

impl<K: GGLWEInfos> LWEInfos for GGLWEDigits<'_, K> {
    fn noise(&self) -> Option<crate::ComponentNoise> {
        (self.dnum == self.key.dnum()).then(|| self.key.noise()).flatten()
    }

    fn n(&self) -> Degree {
        self.key.n()
    }

    fn k(&self) -> TorusPrecision {
        crate::layouts::offset_key_k(self.key.k_offset(), self.dnum, self.key.k_digit(), self.key.k_aux())
    }

    fn base2k(&self) -> Base2K {
        self.key.base2k()
    }

    fn max_size(&self) -> usize {
        self.key.max_size()
    }
}

impl<K: GGLWEInfos> GLWEInfos for GGLWEDigits<'_, K> {
    fn rank(&self) -> Rank {
        self.key.rank()
    }
}

impl<K: GGLWEInfos> GGLWEInfos for GGLWEDigits<'_, K> {
    fn k_aux(&self) -> TorusPrecision {
        self.key.k_aux()
    }

    fn dnum(&self) -> Dnum {
        self.dnum
    }

    fn k_digit(&self) -> TorusPrecision {
        self.key.k_digit()
    }

    fn rank_in(&self) -> Rank {
        self.key.rank_in()
    }

    fn rank_out(&self) -> Rank {
        self.key.rank_out()
    }

    fn k_offset(&self) -> TorusPrecision {
        self.key.k_offset()
    }
}

impl<BE: Backend, K: GGLWEToBackendRef<BE>> GGLWEToBackendRef<BE> for GGLWEDigits<'_, K> {
    fn to_backend_ref(&self) -> GGLWEBackendRef<'_, BE> {
        self.key
            .to_backend_ref()
            .with_dnum(self.dnum)
            .expect("checked by view_with_digits")
    }
}

impl<K: GetGaloisElement> GetGaloisElement for GGLWEDigits<'_, K> {
    fn p(&self) -> i64 {
        self.key.p()
    }
}

/// The first `dnum` digits of a GGSW, from [`GGSWViewWithDigits`]: rows
/// `0..dnum` read at `dnum * k_digit + k_aux` bits, as [`GGLWEDigits`] reads a
/// key.
pub struct GGSWDigits<'a, G> {
    ggsw: &'a G,
    dnum: Dnum,
}

/// `ggsw.view_with_digits(dnum)` reads a GGSW as a GGSW of its first `dnum`
/// digits, see [`GGSWDigits`].
pub trait GGSWViewWithDigits: GGSWInfos + Sized {
    /// Fails unless `1 <= dnum <= self.dnum()`.
    fn view_with_digits(&self, dnum: Dnum) -> Result<GGSWDigits<'_, Self>> {
        check_dnum("view_with_digits", dnum, self.dnum())?;
        Ok(GGSWDigits { ggsw: self, dnum })
    }
}

impl<G: GGSWInfos> GGSWViewWithDigits for G {}

impl<G: GGSWInfos> LWEInfos for GGSWDigits<'_, G> {
    fn noise(&self) -> Option<crate::ComponentNoise> {
        (self.dnum == self.ggsw.dnum()).then(|| self.ggsw.noise()).flatten()
    }

    fn n(&self) -> Degree {
        self.ggsw.n()
    }

    fn k(&self) -> TorusPrecision {
        crate::layouts::key_k(self.dnum, self.ggsw.k_digit(), self.ggsw.k_aux())
    }

    fn base2k(&self) -> Base2K {
        self.ggsw.base2k()
    }

    fn max_size(&self) -> usize {
        self.ggsw.max_size()
    }
}

impl<G: GGSWInfos> GLWEInfos for GGSWDigits<'_, G> {
    fn rank(&self) -> Rank {
        self.ggsw.rank()
    }
}

impl<G: GGSWInfos> GGSWInfos for GGSWDigits<'_, G> {
    fn k_aux(&self) -> TorusPrecision {
        self.ggsw.k_aux()
    }

    fn dnum(&self) -> Dnum {
        self.dnum
    }

    fn k_digit(&self) -> TorusPrecision {
        self.ggsw.k_digit()
    }
}

impl<BE: Backend, G: GGSWToBackendRef<BE>> GGSWToBackendRef<BE> for GGSWDigits<'_, G> {
    fn to_backend_ref(&self) -> GGSWBackendRef<'_, BE> {
        self.ggsw
            .to_backend_ref()
            .with_dnum(self.dnum)
            .expect("checked by view_with_digits")
    }
}
