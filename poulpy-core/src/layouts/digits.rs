//! Views of the first digits of a key or a GGSW.
//!
//! A switch decomposes the rows of the key it switches exactly, guard
//! included, so a key cannot switch a key of its own layout. Read through its
//! first digits, the switched key has rows of `dnum * k_digit + k_aux` bits,
//! which a key of more digits covers. The view copies nothing.

use poulpy_hal::layouts::Backend;

use crate::{
    error::{CoreError, Result},
    layouts::{
        Base2K, Degree, Dnum, GGLWEBackendMut, GGLWEBackendRef, GGLWEInfos, GGLWEToBackendMut, GGLWEToBackendRef, GGSWBackendMut,
        GGSWBackendRef, GGSWInfos, GGSWToBackendMut, GGSWToBackendRef, GLWEInfos, GetGaloisElement, LWEInfos, Rank,
        SetGaloisElement, TorusPrecision,
    },
};

fn check_dnum(op: &'static str, dnum: Dnum, stored: Dnum) -> Result<()> {
    if dnum.0 == 0 || dnum.0 > stored.0 {
        return Err(CoreError::GGLWEKeyUse {
            op,
            detail: format!("dnum={} is not in 1..={}", dnum.0, stored.0),
        });
    }
    Ok(())
}

/// The first `dnum` digits of the key `key` refers to, `&K` or `&mut K`: its
/// backend views are [`GGLWEBackendRef::with_dnum`] and
/// [`GGLWEBackendMut::with_dnum`], and its layout that of a key of `dnum`
/// digits and the same guard. A view of fewer digits than the key has no
/// noise estimate.
pub struct GGLWEDigits<K> {
    key: K,
    dnum: Dnum,
}

impl<K: GGLWEInfos> GGLWEDigits<K> {
    /// Fails unless `1 <= dnum <= key.dnum()`.
    pub fn new(key: K, dnum: Dnum) -> Result<Self> {
        check_dnum("GGLWEDigits::new", dnum, key.dnum())?;
        Ok(Self { key, dnum })
    }
}

impl<K: GGLWEInfos> LWEInfos for GGLWEDigits<K> {
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

impl<K: GGLWEInfos> GLWEInfos for GGLWEDigits<K> {
    fn rank(&self) -> Rank {
        self.key.rank()
    }
}

impl<K: GGLWEInfos> GGLWEInfos for GGLWEDigits<K> {
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

impl<BE: Backend, K: GGLWEToBackendRef<BE>> GGLWEToBackendRef<BE> for GGLWEDigits<&K> {
    fn to_backend_ref(&self) -> GGLWEBackendRef<'_, BE> {
        self.key
            .to_backend_ref()
            .with_dnum(self.dnum)
            .expect("checked by GGLWEDigits::new")
    }
}

impl<BE: Backend, K: GGLWEToBackendRef<BE>> GGLWEToBackendRef<BE> for GGLWEDigits<&mut K> {
    fn to_backend_ref(&self) -> GGLWEBackendRef<'_, BE> {
        self.key
            .to_backend_ref()
            .with_dnum(self.dnum)
            .expect("checked by GGLWEDigits::new")
    }
}

impl<BE: Backend, K: GGLWEToBackendMut<BE>> GGLWEToBackendMut<BE> for GGLWEDigits<&mut K> {
    /// The estimate is of the view's layout, not the key's: the key keeps none.
    fn set_noise(&mut self, _metadata: Option<crate::ComponentNoise>) {
        self.key.set_noise(None);
    }

    fn to_backend_mut(&mut self) -> GGLWEBackendMut<'_, BE> {
        self.key
            .to_backend_mut()
            .with_dnum(self.dnum)
            .expect("checked by GGLWEDigits::new")
    }
}

impl<K: GetGaloisElement> GetGaloisElement for GGLWEDigits<&K> {
    fn p(&self) -> i64 {
        self.key.p()
    }
}

impl<K: GetGaloisElement> GetGaloisElement for GGLWEDigits<&mut K> {
    fn p(&self) -> i64 {
        self.key.p()
    }
}

impl<K: SetGaloisElement> SetGaloisElement for GGLWEDigits<&mut K> {
    fn set_p(&mut self, p: i64) {
        self.key.set_p(p)
    }
}

/// The first `dnum` digits of the GGSW `ggsw` refers to, as [`GGLWEDigits`]:
/// rows `0..dnum` at `dnum * k_digit + k_aux` bits.
pub struct GGSWDigits<G> {
    ggsw: G,
    dnum: Dnum,
}

impl<G: GGSWInfos> GGSWDigits<G> {
    /// Fails unless `1 <= dnum <= ggsw.dnum()`.
    pub fn new(ggsw: G, dnum: Dnum) -> Result<Self> {
        check_dnum("GGSWDigits::new", dnum, ggsw.dnum())?;
        Ok(Self { ggsw, dnum })
    }
}

impl<G: GGSWInfos> LWEInfos for GGSWDigits<G> {
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

impl<G: GGSWInfos> GLWEInfos for GGSWDigits<G> {
    fn rank(&self) -> Rank {
        self.ggsw.rank()
    }
}

impl<G: GGSWInfos> GGSWInfos for GGSWDigits<G> {
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

impl<BE: Backend, G: GGSWToBackendRef<BE>> GGSWToBackendRef<BE> for GGSWDigits<&G> {
    fn to_backend_ref(&self) -> GGSWBackendRef<'_, BE> {
        self.ggsw
            .to_backend_ref()
            .with_dnum(self.dnum)
            .expect("checked by GGSWDigits::new")
    }
}

impl<BE: Backend, G: GGSWToBackendRef<BE>> GGSWToBackendRef<BE> for GGSWDigits<&mut G> {
    fn to_backend_ref(&self) -> GGSWBackendRef<'_, BE> {
        self.ggsw
            .to_backend_ref()
            .with_dnum(self.dnum)
            .expect("checked by GGSWDigits::new")
    }
}

impl<BE: Backend, G: GGSWToBackendMut<BE>> GGSWToBackendMut<BE> for GGSWDigits<&mut G> {
    /// The estimate is of the view's layout, not the GGSW's: the GGSW keeps none.
    fn set_noise(&mut self, _metadata: Option<crate::ComponentNoise>) {
        self.ggsw.set_noise(None);
    }

    fn to_backend_mut(&mut self) -> GGSWBackendMut<'_, BE> {
        self.ggsw
            .to_backend_mut()
            .with_dnum(self.dnum)
            .expect("checked by GGSWDigits::new")
    }
}
