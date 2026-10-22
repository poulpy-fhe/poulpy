use poulpy_hal::layouts::{
    Backend, Data, HostDataMut, HostDataRef, MatZnx, MatZnxAtBackendMut, MatZnxAtBackendRef, MatZnxToBackendMut,
    MatZnxToBackendRef, ReaderFrom, WriterTo,
};
use poulpy_hal::{AlignedBuf, alloc_aligned};
use std::{
    fmt,
    ops::{Deref, DerefMut},
};

use crate::layouts::{Base2K, Degree, Dnum, Dsize, GLWE, GLWEInfos, GLWEViewMut, GLWEViewRef, LWEInfos, Rank, TorusPrecision};
use poulpy_hal::layouts::ZnxWord;

/// Trait providing the parameter accessors for a GGSW (Gadget GSW) ciphertext.
///
/// A GGSW ciphertext is a matrix of [`GLWE`] ciphertexts with `rank_in = rank + 1`
/// input columns and `rank_out = rank + 1` output columns. It is used as the
/// left operand of external products.
/// Extends [`GLWEInfos`] with gadget decomposition parameters.
pub trait GGSWInfos
where
    Self: GLWEInfos,
{
    /// Auxiliary guard precision (in bits) stored below the gadget region and
    /// used for noise management during key operations. Free value, need not be
    /// a multiple of `base2k`.
    fn k_aux(&self) -> TorusPrecision;
    /// Returns the number of gadget-decomposition rows.
    fn dnum(&self) -> Dnum;
    /// Width in bits of one gadget digit, see [`crate::layouts::GGLWEInfos::k_digit`].
    fn k_digit(&self) -> TorusPrecision;
    /// Limbs one digit spans at the key radix, `ceil(k_digit / base2k)`.
    fn dsize(&self) -> Dsize {
        Dsize(self.k_digit().0.div_ceil(self.base2k().0))
    }
    /// Whether the digits are whole limbs (see [`crate::layouts::digits_aligned`]).
    fn digits_aligned(&self) -> bool {
        crate::layouts::digits_aligned(self.base2k(), self.k_digit())
    }
    /// Returns a plain-data [`GGSWLayout`] snapshot of the current parameters.
    fn ggsw_layout(&self) -> GGSWLayout {
        GGSWLayout {
            n: self.n(),
            base2k: self.base2k(),
            dnum: self.dnum(),
            k_aux: self.k_aux(),
            rank: self.rank(),
            k_digit: self.k_digit(),
        }
    }
}

impl<T: GGSWInfos + ?Sized> GGSWInfos for &T {
    fn k_aux(&self) -> TorusPrecision {
        (**self).k_aux()
    }

    fn dnum(&self) -> Dnum {
        (**self).dnum()
    }

    fn k_digit(&self) -> TorusPrecision {
        (**self).k_digit()
    }
}

impl<T: GGSWInfos + ?Sized> GGSWInfos for &mut T {
    fn k_aux(&self) -> TorusPrecision {
        (**self).k_aux()
    }

    fn dnum(&self) -> Dnum {
        (**self).dnum()
    }

    fn k_digit(&self) -> TorusPrecision {
        (**self).k_digit()
    }
}

/// Plain-data snapshot of the parameters that describe a [`GGSW`] ciphertext.
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GGSWLayout {
    /// Ring degree.
    pub n: Degree,
    /// Base-2-log of the limb width.
    pub base2k: Base2K,
    /// Number of gadget-decomposition rows.
    pub dnum: Dnum,
    /// Auxiliary guard precision (torus bits) below the gadget region.
    pub k_aux: TorusPrecision,
    /// GLWE rank (number of mask polynomials per row).
    pub rank: Rank,
    /// Width in bits of one gadget digit.
    pub k_digit: TorusPrecision,
}

impl LWEInfos for GGSWLayout {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn n(&self) -> Degree {
        self.n
    }

    fn max_size(&self) -> usize {
        crate::layouts::key_size(self.base2k, self.dnum, self.k_digit, self.k_aux)
    }

    fn k(&self) -> TorusPrecision {
        crate::layouts::key_k(self.dnum, self.k_digit, self.k_aux)
    }
}
impl GLWEInfos for GGSWLayout {
    fn rank(&self) -> Rank {
        self.rank
    }
}

impl GGSWInfos for GGSWLayout {
    fn k_aux(&self) -> TorusPrecision {
        self.k_aux
    }

    fn dnum(&self) -> Dnum {
        self.dnum
    }

    fn k_digit(&self) -> TorusPrecision {
        self.k_digit
    }
}

/// A GGSW (Gadget GSW) ciphertext.
///
/// Stored as a [`MatZnx`] matrix of [`GLWE`] ciphertexts with
/// `rank_in = rank + 1` input columns and `rank_out = rank + 1` output columns.
/// Used as the left operand of external products.
///
/// `D: Data` is the storage backend (e.g. `AlignedBuf`, `&[u8]`, `&mut [u8]`).
///
/// Its rows are canonical at `k()` (see [`GLWE`]) but on a view of its first
/// digits ([`Self::with_dnum`]), whose rows hold its bits below that
/// precision. Equality ignores the flag.
#[derive(Clone)]
pub struct GGSW<D: Data, W: ZnxWord> {
    pub(crate) noise: Option<crate::ComponentNoise>,
    pub(crate) data: MatZnx<D, W>,
    pub(crate) k_aux: TorusPrecision,
    pub(crate) canonical: bool,
    pub(crate) base2k: Base2K,
    pub(crate) k_digit: TorusPrecision,
}

impl<D: Data, W: ZnxWord> PartialEq for GGSW<D, W>
where
    MatZnx<D, W>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.noise == other.noise
            && self.data == other.data
            && self.k_aux == other.k_aux
            && self.base2k == other.base2k
            && self.k_digit == other.k_digit
    }
}

impl<D: Data, W: ZnxWord> Eq for GGSW<D, W> where MatZnx<D, W>: Eq {}

impl<D: Data, W: ZnxWord> GGSW<D, W> {
    /// Its first `dnum` digits, over the same buffer: rows `0..dnum`, at the
    /// precision `dnum * k_digit + k_aux`, as
    /// [`GGLWE::with_dnum`](crate::layouts::GGLWE::with_dnum).
    ///
    /// Fails unless `1 <= dnum <= self.dnum()`.
    pub fn with_dnum(self, dnum: Dnum) -> crate::Result<Self> {
        let stored: Dnum = self.dnum();
        if dnum.0 == 0 || dnum.0 > stored.0 {
            return Err(crate::CoreError::GGLWEKeyUse {
                op: "GGSW::with_dnum",
                detail: format!("dnum={} is not in 1..={}", dnum.0, stored.0),
            });
        }
        if dnum == stored {
            return Ok(self);
        }
        Ok(GGSW {
            noise: None,
            data: self.data.with_rows(dnum.as_usize()),
            canonical: false,
            ..self
        })
    }
}

pub struct GGSWBackendRef<'a, BE: Backend + 'a> {
    inner: GGSW<BE::BufRef<'a>, BE::ZnxWord>,
}

impl<'a, BE: Backend + 'a> GGSWBackendRef<'a, BE> {
    pub fn from_inner(inner: GGSW<BE::BufRef<'a>, BE::ZnxWord>) -> Self {
        Self { inner }
    }

    /// The view of its first `dnum` digits ([`GGSW::with_dnum`]).
    pub fn with_dnum(self, dnum: Dnum) -> crate::Result<Self> {
        Ok(Self::from_inner(self.inner.with_dnum(dnum)?))
    }

    pub fn into_inner(self) -> GGSW<BE::BufRef<'a>, BE::ZnxWord> {
        self.inner
    }

    pub fn at_view(&self, row: usize, col: usize) -> GLWEViewRef<'_, BE> {
        GLWEViewRef::from_inner(ggsw_at_backend_ref_from_ref::<BE>(&self.inner, row, col))
    }
}

impl<'a, BE: Backend + 'a> Deref for GGSWBackendRef<'a, BE> {
    type Target = GGSW<BE::BufRef<'a>, BE::ZnxWord>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub struct GGSWBackendMut<'a, BE: Backend + 'a> {
    inner: GGSW<BE::BufMut<'a>, BE::ZnxWord>,
}

impl<'a, BE: Backend + 'a> GGSWBackendMut<'a, BE> {
    pub fn from_inner(inner: GGSW<BE::BufMut<'a>, BE::ZnxWord>) -> Self {
        Self { inner }
    }

    /// The view of its first `dnum` digits ([`GGSW::with_dnum`]).
    pub fn with_dnum(self, dnum: Dnum) -> crate::Result<Self> {
        Ok(Self::from_inner(self.inner.with_dnum(dnum)?))
    }

    pub fn into_inner(self) -> GGSW<BE::BufMut<'a>, BE::ZnxWord> {
        self.inner
    }

    pub fn at_view(&self, row: usize, col: usize) -> GLWEViewRef<'_, BE> {
        GLWEViewRef::from_inner(ggsw_at_backend_ref_from_mut::<BE>(&self.inner, row, col))
    }

    /// Clears the owner's component noise metadata.
    pub fn at_view_mut(&mut self, row: usize, col: usize) -> GLWEViewMut<'_, BE> {
        GLWEViewMut::from_inner(ggsw_at_backend_mut_from_mut::<BE>(&mut self.inner, row, col))
    }
}

impl<'a, BE: Backend + 'a> Deref for GGSWBackendMut<'a, BE> {
    type Target = GGSW<BE::BufMut<'a>, BE::ZnxWord>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<BE: Backend> DerefMut for GGSWBackendMut<'_, BE> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<BE: Backend> LWEInfos for GGSWBackendRef<'_, BE> {
    fn noise(&self) -> Option<crate::ComponentNoise> {
        self.inner.noise()
    }

    fn base2k(&self) -> Base2K {
        self.inner.base2k()
    }

    fn n(&self) -> Degree {
        self.inner.n()
    }

    fn max_size(&self) -> usize {
        self.inner.max_size()
    }

    fn k(&self) -> TorusPrecision {
        self.inner.k()
    }
}

impl<BE: Backend> GLWEInfos for GGSWBackendRef<'_, BE> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

impl<BE: Backend> GGSWInfos for GGSWBackendRef<'_, BE> {
    fn k_aux(&self) -> TorusPrecision {
        self.inner.k_aux()
    }

    fn dnum(&self) -> Dnum {
        self.inner.dnum()
    }

    fn k_digit(&self) -> TorusPrecision {
        self.inner.k_digit()
    }
}

impl<BE: Backend> LWEInfos for GGSWBackendMut<'_, BE> {
    fn noise(&self) -> Option<crate::ComponentNoise> {
        self.inner.noise()
    }

    fn base2k(&self) -> Base2K {
        self.inner.base2k()
    }

    fn n(&self) -> Degree {
        self.inner.n()
    }

    fn max_size(&self) -> usize {
        self.inner.max_size()
    }

    fn k(&self) -> TorusPrecision {
        self.inner.k()
    }
}

impl<BE: Backend> GLWEInfos for GGSWBackendMut<'_, BE> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

impl<BE: Backend> GGSWInfos for GGSWBackendMut<'_, BE> {
    fn k_aux(&self) -> TorusPrecision {
        self.inner.k_aux()
    }

    fn dnum(&self) -> Dnum {
        self.inner.dnum()
    }

    fn k_digit(&self) -> TorusPrecision {
        self.inner.k_digit()
    }
}

impl<BE: Backend> GGSWToBackendRef<BE> for GGSWBackendRef<'_, BE> {
    fn to_backend_ref(&self) -> GGSWBackendRef<'_, BE> {
        GGSWBackendRef::from_inner(GGSW {
            noise: crate::layouts::LWEInfos::noise(&self.inner),
            k_digit: self.inner.k_digit(),
            base2k: self.inner.base2k(),
            k_aux: self.inner.k_aux(),
            canonical: self.inner.canonical,
            data: poulpy_hal::layouts::mat_znx_backend_ref_from_ref::<BE>(&self.inner.data),
        })
    }
}

impl<BE: Backend> GGSWToBackendRef<BE> for GGSWBackendMut<'_, BE> {
    fn to_backend_ref(&self) -> GGSWBackendRef<'_, BE> {
        GGSWBackendRef::from_inner(GGSW {
            noise: crate::layouts::LWEInfos::noise(&self.inner),
            k_digit: self.inner.k_digit,
            base2k: self.inner.base2k,
            k_aux: self.inner.k_aux,
            canonical: self.inner.canonical,
            data: poulpy_hal::layouts::mat_znx_backend_ref_from_mut::<BE>(&self.inner.data),
        })
    }
}

impl<BE: Backend> GGSWToBackendMut<BE> for GGSWBackendMut<'_, BE> {
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        self.inner.noise = crate::layouts::checked_noise(metadata, crate::layouts::GLWEInfos::rank(self).as_usize() + 1);
    }

    fn to_backend_mut(&mut self) -> GGSWBackendMut<'_, BE> {
        self.inner.noise = None;
        GGSWBackendMut::from_inner(GGSW {
            noise: None,
            k_digit: self.inner.k_digit,
            base2k: self.inner.base2k,
            k_aux: self.inner.k_aux,
            canonical: self.inner.canonical,
            data: poulpy_hal::layouts::mat_znx_backend_mut_from_mut::<BE>(&mut self.inner.data),
        })
    }
}

impl<BE: Backend> GGSWAtViewRef<BE> for GGSWBackendRef<'_, BE> {
    fn at_view(&self, row: usize, col: usize) -> GLWEViewRef<'_, BE> {
        GGSWBackendRef::at_view(self, row, col)
    }
}

impl<BE: Backend> GGSWAtViewRef<BE> for GGSWBackendMut<'_, BE> {
    fn at_view(&self, row: usize, col: usize) -> GLWEViewRef<'_, BE> {
        GGSWBackendMut::at_view(self, row, col)
    }
}

impl<BE: Backend> GGSWAtViewRef<BE> for &GGSWBackendRef<'_, BE> {
    fn at_view(&self, row: usize, col: usize) -> GLWEViewRef<'_, BE> {
        GGSWBackendRef::at_view(self, row, col)
    }
}

impl<D: Data, W: ZnxWord> LWEInfos for GGSW<D, W> {
    fn noise(&self) -> Option<crate::ComponentNoise> {
        self.noise.clone()
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn max_size(&self) -> usize {
        self.data.size()
    }

    fn k(&self) -> TorusPrecision {
        crate::layouts::key_k(self.dnum(), self.k_digit, self.k_aux)
    }
}

impl<D: Data, W: ZnxWord> GLWEInfos for GGSW<D, W> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols_out() as u32 - 1)
    }
}

impl<D: Data, W: ZnxWord> GGSWInfos for GGSW<D, W> {
    fn k_aux(&self) -> TorusPrecision {
        self.k_aux
    }

    fn k_digit(&self) -> TorusPrecision {
        self.k_digit
    }

    fn dnum(&self) -> Dnum {
        Dnum(self.data.rows() as u32)
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Debug for GGSW<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.data)
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Display for GGSW<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(GGSW: k: {} base2k: {} dsize: {}) {}",
            self.k().0,
            self.base2k().0,
            self.dsize().0,
            self.data
        )
    }
}

impl<D: HostDataRef, W: ZnxWord> GGSW<D, W> {
    pub fn at(&self, row: usize, col: usize) -> GLWE<&[u8], W> {
        let data = self.data.at(row, col);
        GLWE {
            noise: crate::layouts::LWEInfos::noise(&self),
            base2k: self.base2k,
            k: self.k(),
            canonical: self.canonical,
            plaintext_meta: None,
            data,
        }
    }
}

/// Backend-native shared view of one GLWE row.
pub trait GGSWAtBackendRef<BE: Backend> {
    fn at_backend(&self, row: usize, col: usize) -> GLWE<BE::BufRef<'_>, BE::ZnxWord>;
}

impl<BE: Backend> GGSWAtBackendRef<BE> for GGSW<BE::OwnedBuf, BE::ZnxWord> {
    fn at_backend(&self, row: usize, col: usize) -> GLWE<BE::BufRef<'_>, BE::ZnxWord> {
        let data = <MatZnx<BE::OwnedBuf, BE::ZnxWord> as MatZnxAtBackendRef<BE>>::at_backend(&self.data, row, col);
        GLWE {
            noise: crate::layouts::LWEInfos::noise(&self),
            base2k: self.base2k,
            k: self.k(),
            canonical: self.canonical,
            plaintext_meta: None,
            data,
        }
    }
}

pub(crate) fn ggsw_at_backend_ref_from_ref<'a, 'b, BE: Backend>(
    ggsw: &'a GGSW<BE::BufRef<'b>, BE::ZnxWord>,
    row: usize,
    col: usize,
) -> GLWE<BE::BufRef<'a>, BE::ZnxWord> {
    let data = poulpy_hal::layouts::mat_znx_at_backend_ref_from_ref::<BE>(&ggsw.data, row, col);
    GLWE {
        noise: crate::layouts::LWEInfos::noise(&ggsw),
        base2k: ggsw.base2k,
        k: ggsw.k(),
        canonical: ggsw.canonical,
        plaintext_meta: None,
        data,
    }
}

pub trait GGSWAtViewRef<BE: Backend> {
    fn at_view(&self, row: usize, col: usize) -> GLWEViewRef<'_, BE>;
}

impl<BE: Backend> GGSWAtViewRef<BE> for GGSW<BE::OwnedBuf, BE::ZnxWord> {
    fn at_view(&self, row: usize, col: usize) -> GLWEViewRef<'_, BE> {
        GLWEViewRef::from_inner(<GGSW<BE::OwnedBuf, BE::ZnxWord> as GGSWAtBackendRef<BE>>::at_backend(
            self, row, col,
        ))
    }
}

pub(crate) fn ggsw_at_backend_ref_from_mut<'a, 'b, BE: Backend>(
    ggsw: &'a GGSW<BE::BufMut<'b>, BE::ZnxWord>,
    row: usize,
    col: usize,
) -> GLWE<BE::BufRef<'a>, BE::ZnxWord> {
    let data = poulpy_hal::layouts::mat_znx_at_backend_ref_from_mut::<BE>(&ggsw.data, row, col);
    GLWE {
        noise: crate::layouts::LWEInfos::noise(&ggsw),
        base2k: ggsw.base2k,
        k: ggsw.k(),
        canonical: ggsw.canonical,
        plaintext_meta: None,
        data,
    }
}

impl<D: HostDataMut, W: ZnxWord> GGSW<D, W> {
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWE<&mut [u8], W> {
        let base2k = self.base2k;
        let k = self.k();
        let canonical = self.canonical;
        let data = self.data.at_mut(row, col);
        self.noise = None;
        GLWE {
            noise: None,
            base2k,
            k,
            canonical,
            plaintext_meta: None,
            data,
        }
    }
}

/// Backend-native mutable view of one GLWE row.
pub trait GGSWAtBackendMut<BE: Backend> {
    /// Clears the owner's component noise metadata.
    fn at_backend_mut(&mut self, row: usize, col: usize) -> GLWE<BE::BufMut<'_>, BE::ZnxWord>;
}

impl<BE: Backend> GGSWAtBackendMut<BE> for GGSW<BE::OwnedBuf, BE::ZnxWord> {
    /// Clears the owner's component noise metadata.
    fn at_backend_mut(&mut self, row: usize, col: usize) -> GLWE<BE::BufMut<'_>, BE::ZnxWord> {
        let base2k = self.base2k;
        let k = self.k();
        let canonical = self.canonical;
        let data = <MatZnx<BE::OwnedBuf, BE::ZnxWord> as MatZnxAtBackendMut<BE>>::at_backend_mut(&mut self.data, row, col);
        self.noise = None;
        GLWE {
            noise: None,
            base2k,
            k,
            canonical,
            plaintext_meta: None,
            data,
        }
    }
}

pub(crate) fn ggsw_at_backend_mut_from_mut<'a, 'b, BE: Backend>(
    ggsw: &'a mut GGSW<BE::BufMut<'b>, BE::ZnxWord>,
    row: usize,
    col: usize,
) -> GLWE<BE::BufMut<'a>, BE::ZnxWord> {
    let (base2k, k, canonical) = (ggsw.base2k, ggsw.k(), ggsw.canonical);
    let data = poulpy_hal::layouts::mat_znx_at_backend_mut_from_mut::<BE>(&mut ggsw.data, row, col);
    ggsw.noise = None;
    GLWE {
        noise: None,
        base2k,
        k,
        canonical,
        plaintext_meta: None,
        data,
    }
}

pub trait GGSWAtViewMut<BE: Backend> {
    fn at_view_mut(&mut self, row: usize, col: usize) -> GLWEViewMut<'_, BE>;
}

impl<BE: Backend> GGSWAtViewMut<BE> for GGSW<BE::OwnedBuf, BE::ZnxWord> {
    fn at_view_mut(&mut self, row: usize, col: usize) -> GLWEViewMut<'_, BE> {
        GLWEViewMut::from_inner(<GGSW<BE::OwnedBuf, BE::ZnxWord> as GGSWAtBackendMut<BE>>::at_backend_mut(
            self, row, col,
        ))
    }
}

impl<BE: Backend> GGSWAtViewMut<BE> for GGSWBackendMut<'_, BE> {
    fn at_view_mut(&mut self, row: usize, col: usize) -> GLWEViewMut<'_, BE> {
        GGSWBackendMut::at_view_mut(self, row, col)
    }
}

impl<D: Data, W: ZnxWord> GGSW<D, W> {
    /// Zero-cost rename when both backends share the same `OwnedBuf`.
    pub fn reinterpret<To>(self) -> GGSW<To::OwnedBuf, To::ZnxWord>
    where
        To: Backend<OwnedBuf = D, ZnxWord = W>,
    {
        let (n, rows, cols_in, cols_out, size) = (
            self.data.n(),
            self.data.rows(),
            self.data.cols_in(),
            self.data.cols_out(),
            self.data.size(),
        );
        GGSW {
            noise: crate::layouts::LWEInfos::noise(&self),
            data: MatZnx::from_data(self.data.into_data(), n, rows, cols_in, cols_out, size),
            k_aux: self.k_aux,
            canonical: self.canonical,
            base2k: self.base2k,
            k_digit: self.k_digit,
        }
    }
}

#[expect(
    dead_code,
    reason = "host-owned constructors are kept for serialization and host-only staging"
)]
impl<W: ZnxWord> GGSW<AlignedBuf, W> {
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GGSWInfos,
    {
        Self::alloc(
            infos.n(),
            infos.base2k(),
            infos.dnum(),
            infos.k_digit(),
            infos.k_aux(),
            infos.rank(),
        )
    }

    pub(crate) fn alloc(
        n: Degree,
        base2k: Base2K,
        dnum: Dnum,
        k_digit: TorusPrecision,
        k_aux: TorusPrecision,
        rank: Rank,
    ) -> Self {
        let size: usize = crate::layouts::key_size(base2k, dnum, k_digit, k_aux);

        GGSW {
            noise: None,
            data: MatZnx::from_data(
                alloc_aligned::<u8>(MatZnx::<AlignedBuf, W>::bytes_of(
                    n.into(),
                    dnum.into(),
                    (rank + 1).into(),
                    (rank + 1).into(),
                    size,
                )),
                n.into(),
                dnum.into(),
                (rank + 1).into(),
                (rank + 1).into(),
                size,
            ),
            k_aux,
            canonical: true,
            base2k,
            k_digit,
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        Self::bytes_of(
            infos.n(),
            infos.base2k(),
            infos.dnum(),
            infos.k_digit(),
            infos.k_aux(),
            infos.rank(),
        )
    }

    pub fn bytes_of(n: Degree, base2k: Base2K, dnum: Dnum, k_digit: TorusPrecision, k_aux: TorusPrecision, rank: Rank) -> usize {
        let size: usize = crate::layouts::key_size(base2k, dnum, k_digit, k_aux);

        MatZnx::<AlignedBuf, W>::bytes_of(n.into(), dnum.into(), (rank + 1).into(), (rank + 1).into(), size)
    }
}

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for GGSW<D, W> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.noise = None;
        let components = self.data.cols_out();
        let noise = crate::ComponentNoise::read_optional(reader, components)?;
        let base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        let k_digit = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        let k_aux = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        crate::layouts::read_mat_znx_with_shape(&mut self.data, reader, None, Some(components), components)?;
        crate::layouts::validate_noise_components(noise.as_ref(), self.data.cols_out())?;
        self.base2k = base2k;
        self.k_digit = k_digit;
        self.k_aux = k_aux;
        self.canonical = true;
        self.noise = noise;
        Ok(())
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for GGSW<D, W> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        if !self.canonical {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "GGSW is a view of its first digits: copy it into a GGSW of that layout before serializing",
            ));
        }
        crate::layouts::validate_noise_components(self.noise.as_ref(), self.data.cols_out())?;
        crate::ComponentNoise::write_optional(self.noise.as_ref(), writer)?;
        writer.write_u32::<LittleEndian>(self.base2k.into())?;
        writer.write_u32::<LittleEndian>(self.k_digit.into())?;
        writer.write_u32::<LittleEndian>(self.k_aux.into())?;
        self.data.write_to(writer)
    }
}

pub trait GGSWToBackendMut<BE: Backend>: GGSWToBackendRef<BE> {
    /// Backend hook for recording or propagating component noise metadata.
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>);

    /// Borrows coefficients mutably and clears the owner's component noise metadata.
    fn to_backend_mut(&mut self) -> GGSWBackendMut<'_, BE>;
}

impl<BE: Backend, D: Data> GGSWToBackendMut<BE> for GGSW<D, BE::ZnxWord>
where
    MatZnx<D, BE::ZnxWord>: MatZnxToBackendRef<BE> + MatZnxToBackendMut<BE>,
{
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        self.noise = crate::layouts::checked_noise(metadata, crate::layouts::GLWEInfos::rank(self).as_usize() + 1);
    }

    fn to_backend_mut(&mut self) -> GGSWBackendMut<'_, BE> {
        self.noise = None;
        GGSWBackendMut::from_inner(GGSW {
            noise: None,
            k_digit: self.k_digit,
            base2k: self.base2k,
            k_aux: self.k_aux,
            canonical: self.canonical,
            data: self.data.to_backend_mut(),
        })
    }
}

impl<BE: Backend> GGSWToBackendRef<BE> for &mut GGSW<BE::BufMut<'_>, BE::ZnxWord> {
    fn to_backend_ref(&self) -> GGSWBackendRef<'_, BE> {
        GGSWBackendRef::from_inner(GGSW {
            noise: crate::layouts::LWEInfos::noise(&self),
            k_digit: self.k_digit,
            base2k: self.base2k,
            k_aux: self.k_aux,
            canonical: self.canonical,
            data: poulpy_hal::layouts::mat_znx_backend_ref_from_mut::<BE>(&self.data),
        })
    }
}

impl<BE: Backend> GGSWToBackendMut<BE> for &mut GGSW<BE::BufMut<'_>, BE::ZnxWord> {
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        self.noise = crate::layouts::checked_noise(metadata, crate::layouts::GLWEInfos::rank(self).as_usize() + 1);
    }

    fn to_backend_mut(&mut self) -> GGSWBackendMut<'_, BE> {
        ggsw_backend_mut_from_mut::<BE>(self)
    }
}

pub fn ggsw_backend_mut_from_mut<'a, 'b, BE: Backend>(ggsw: &'a mut GGSW<BE::BufMut<'b>, BE::ZnxWord>) -> GGSWBackendMut<'a, BE> {
    ggsw.noise = None;
    GGSWBackendMut::from_inner(GGSW {
        noise: None,
        k_digit: ggsw.k_digit,
        base2k: ggsw.base2k,
        k_aux: ggsw.k_aux,
        canonical: ggsw.canonical,
        data: poulpy_hal::layouts::mat_znx_backend_mut_from_mut::<BE>(&mut ggsw.data),
    })
}

/// Row-view adapter that lets a `GGSWToBackendMut` type satisfy both `GGSWAtViewRef` and
/// `GGSWAtViewMut` simultaneously, which is required by several reference algorithms that need
/// to read and write individual GLWE rows through the trait interface.
pub struct GGSWBackendRowViewMut<'a, BE: Backend + 'a> {
    inner: GGSWBackendMut<'a, BE>,
}

impl<'a, BE: Backend + 'a> GGSWBackendRowViewMut<'a, BE> {
    pub fn from_inner(inner: GGSWBackendMut<'a, BE>) -> Self {
        Self { inner }
    }
}

impl<BE: Backend> LWEInfos for GGSWBackendRowViewMut<'_, BE> {
    fn noise(&self) -> Option<crate::ComponentNoise> {
        self.inner.noise()
    }

    fn base2k(&self) -> Base2K {
        self.inner.base2k()
    }
    fn n(&self) -> Degree {
        self.inner.n()
    }
    fn max_size(&self) -> usize {
        self.inner.max_size()
    }
    fn k(&self) -> TorusPrecision {
        self.inner.k()
    }
}

impl<BE: Backend> GLWEInfos for GGSWBackendRowViewMut<'_, BE> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

impl<BE: Backend> GGSWInfos for GGSWBackendRowViewMut<'_, BE> {
    fn k_aux(&self) -> TorusPrecision {
        self.inner.k_aux()
    }
    fn dnum(&self) -> Dnum {
        self.inner.dnum()
    }
    fn k_digit(&self) -> TorusPrecision {
        self.inner.k_digit()
    }
}

impl<BE: Backend> GGSWToBackendRef<BE> for GGSWBackendRowViewMut<'_, BE> {
    fn to_backend_ref(&self) -> GGSWBackendRef<'_, BE> {
        self.inner.to_backend_ref()
    }
}

impl<BE: Backend> GGSWToBackendMut<BE> for GGSWBackendRowViewMut<'_, BE> {
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        self.inner.noise = crate::layouts::checked_noise(metadata, crate::layouts::GLWEInfos::rank(self).as_usize() + 1);
    }

    fn to_backend_mut(&mut self) -> GGSWBackendMut<'_, BE> {
        self.inner.noise = None;
        GGSWBackendMut::from_inner(GGSW {
            noise: None,
            k_digit: self.inner.inner.k_digit,
            base2k: self.inner.inner.base2k,
            k_aux: self.inner.inner.k_aux,
            canonical: self.inner.inner.canonical,
            data: poulpy_hal::layouts::mat_znx_backend_mut_from_mut::<BE>(&mut self.inner.inner.data),
        })
    }
}

impl<BE: Backend> GGSWAtViewRef<BE> for GGSWBackendRowViewMut<'_, BE> {
    fn at_view(&self, row: usize, col: usize) -> GLWEViewRef<'_, BE> {
        self.inner.at_view(row, col)
    }
}

impl<BE: Backend> GGSWAtViewMut<BE> for GGSWBackendRowViewMut<'_, BE> {
    fn at_view_mut(&mut self, row: usize, col: usize) -> GLWEViewMut<'_, BE> {
        self.inner.at_view_mut(row, col)
    }
}

pub trait GGSWToBackendRef<BE: Backend> {
    fn to_backend_ref(&self) -> GGSWBackendRef<'_, BE>;
}

impl<BE: Backend, D: Data> GGSWToBackendRef<BE> for GGSW<D, BE::ZnxWord>
where
    MatZnx<D, BE::ZnxWord>: MatZnxToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GGSWBackendRef<'_, BE> {
        GGSWBackendRef::from_inner(GGSW {
            noise: crate::layouts::LWEInfos::noise(&self),
            k_digit: self.k_digit,
            base2k: self.base2k,
            k_aux: self.k_aux,
            canonical: self.canonical,
            data: self.data.to_backend_ref(),
        })
    }
}
