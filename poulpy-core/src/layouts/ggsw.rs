use poulpy_hal::{
    layouts::{
        Backend, Data, FillUniform, HostDataMut, HostDataRef, MatZnx, MatZnxAtBackendMut, MatZnxAtBackendRef, MatZnxToBackendMut,
        MatZnxToBackendRef, Module, ReaderFrom, TransferFrom, WriterTo,
    },
    source::Source,
};
use std::{
    fmt,
    ops::{Deref, DerefMut},
};

use crate::api::ModuleTransfer;
use crate::layouts::{Base2K, Degree, Dnum, Dsize, GLWE, GLWEInfos, GLWEViewMut, GLWEViewRef, LWEInfos, Rank, TorusPrecision};

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
    /// Returns the number of decomposition rows.
    fn dnum(&self) -> Dnum;
    /// Returns the decomposition digit size.
    fn dsize(&self) -> Dsize;
    /// Returns a plain-data [`GGSWLayout`] snapshot of the current parameters.
    fn ggsw_layout(&self) -> GGSWLayout {
        GGSWLayout {
            n: self.n(),
            base2k: self.base2k(),
            k: self.max_k(),
            rank: self.rank(),
            dnum: self.dnum(),
            dsize: self.dsize(),
        }
    }
}

/// Plain-data snapshot of the parameters that describe a [`GGSW`] ciphertext.
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GGSWLayout {
    /// Ring degree.
    pub n: Degree,
    /// Base-2-log of the limb width.
    pub base2k: Base2K,
    /// Torus precision.
    pub k: TorusPrecision,
    /// GLWE rank (number of mask polynomials per row).
    pub rank: Rank,
    /// Number of decomposition rows.
    pub dnum: Dnum,
    /// Decomposition digit size.
    pub dsize: Dsize,
}

impl LWEInfos for GGSWLayout {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn n(&self) -> Degree {
        self.n
    }

    fn size(&self) -> usize {
        self.k.as_usize().div_ceil(self.base2k.as_usize())
    }
}
impl GLWEInfos for GGSWLayout {
    fn rank(&self) -> Rank {
        self.rank
    }
}

impl GGSWInfos for GGSWLayout {
    fn dsize(&self) -> Dsize {
        self.dsize
    }

    fn dnum(&self) -> Dnum {
        self.dnum
    }
}

/// A GGSW (Gadget GSW) ciphertext.
///
/// Stored as a [`MatZnx`] matrix of [`GLWE`] ciphertexts with
/// `rank_in = rank + 1` input columns and `rank_out = rank + 1` output columns.
/// Used as the left operand of external products.
///
/// `D: Data` is the storage backend (e.g. `Vec<u8>`, `&[u8]`, `&mut [u8]`).
#[derive(PartialEq, Eq, Clone)]
pub struct GGSW<D: Data> {
    pub(crate) data: MatZnx<D>,
    pub(crate) base2k: Base2K,
    pub(crate) dsize: Dsize,
}

pub struct GGSWBackendRef<'a, BE: Backend + 'a> {
    inner: GGSW<BE::BufRef<'a>>,
}

impl<'a, BE: Backend + 'a> GGSWBackendRef<'a, BE> {
    pub fn from_inner(inner: GGSW<BE::BufRef<'a>>) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> GGSW<BE::BufRef<'a>> {
        self.inner
    }

    pub fn at_view(&self, row: usize, col: usize) -> GLWEViewRef<'_, BE> {
        GLWEViewRef::from_inner(ggsw_at_backend_ref_from_ref::<BE>(&self.inner, row, col))
    }
}

impl<'a, BE: Backend + 'a> Deref for GGSWBackendRef<'a, BE> {
    type Target = GGSW<BE::BufRef<'a>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub struct GGSWBackendMut<'a, BE: Backend + 'a> {
    inner: GGSW<BE::BufMut<'a>>,
}

impl<'a, BE: Backend + 'a> GGSWBackendMut<'a, BE> {
    pub fn from_inner(inner: GGSW<BE::BufMut<'a>>) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> GGSW<BE::BufMut<'a>> {
        self.inner
    }

    pub fn at_view(&self, row: usize, col: usize) -> GLWEViewRef<'_, BE> {
        GLWEViewRef::from_inner(ggsw_at_backend_ref_from_mut::<BE>(&self.inner, row, col))
    }

    pub fn at_view_mut(&mut self, row: usize, col: usize) -> GLWEViewMut<'_, BE> {
        GLWEViewMut::from_inner(ggsw_at_backend_mut_from_mut::<BE>(&mut self.inner, row, col))
    }
}

impl<'a, BE: Backend + 'a> Deref for GGSWBackendMut<'a, BE> {
    type Target = GGSW<BE::BufMut<'a>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'a, BE: Backend + 'a> DerefMut for GGSWBackendMut<'a, BE> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<'a, BE: Backend + 'a> LWEInfos for GGSWBackendRef<'a, BE> {
    fn base2k(&self) -> Base2K {
        self.inner.base2k()
    }

    fn n(&self) -> Degree {
        self.inner.n()
    }

    fn size(&self) -> usize {
        self.inner.size()
    }
}

impl<'a, BE: Backend + 'a> GLWEInfos for GGSWBackendRef<'a, BE> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

impl<'a, BE: Backend + 'a> GGSWInfos for GGSWBackendRef<'a, BE> {
    fn dnum(&self) -> Dnum {
        self.inner.dnum()
    }

    fn dsize(&self) -> Dsize {
        self.inner.dsize()
    }
}

impl<'a, BE: Backend + 'a> LWEInfos for GGSWBackendMut<'a, BE> {
    fn base2k(&self) -> Base2K {
        self.inner.base2k()
    }

    fn n(&self) -> Degree {
        self.inner.n()
    }

    fn size(&self) -> usize {
        self.inner.size()
    }
}

impl<'a, BE: Backend + 'a> GLWEInfos for GGSWBackendMut<'a, BE> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

impl<'a, BE: Backend + 'a> GGSWInfos for GGSWBackendMut<'a, BE> {
    fn dnum(&self) -> Dnum {
        self.inner.dnum()
    }

    fn dsize(&self) -> Dsize {
        self.inner.dsize()
    }
}

impl<'a, BE: Backend + 'a> GGSWToBackendRef<BE> for GGSWBackendRef<'a, BE> {
    fn to_backend_ref(&self) -> GGSWBackendRef<'_, BE> {
        GGSWBackendRef::from_inner(GGSW {
            dsize: self.inner.dsize,
            base2k: self.inner.base2k,
            data: poulpy_hal::layouts::mat_znx_backend_ref_from_ref::<BE>(&self.inner.data),
        })
    }
}

impl<'a, BE: Backend + 'a> GGSWToBackendRef<BE> for GGSWBackendMut<'a, BE> {
    fn to_backend_ref(&self) -> GGSWBackendRef<'_, BE> {
        GGSWBackendRef::from_inner(GGSW {
            dsize: self.inner.dsize,
            base2k: self.inner.base2k,
            data: poulpy_hal::layouts::mat_znx_backend_ref_from_mut::<BE>(&self.inner.data),
        })
    }
}

impl<'a, BE: Backend + 'a> GGSWToBackendMut<BE> for GGSWBackendMut<'a, BE> {
    fn to_backend_mut(&mut self) -> GGSWBackendMut<'_, BE> {
        GGSWBackendMut::from_inner(GGSW {
            dsize: self.inner.dsize,
            base2k: self.inner.base2k,
            data: poulpy_hal::layouts::mat_znx_backend_mut_from_mut::<BE>(&mut self.inner.data),
        })
    }
}

impl<'a, BE: Backend + 'a> GGSWAtViewRef<BE> for GGSWBackendRef<'a, BE> {
    fn at_view(&self, row: usize, col: usize) -> GLWEViewRef<'_, BE> {
        GGSWBackendRef::at_view(self, row, col)
    }
}

impl<'a, BE: Backend + 'a> GGSWAtViewRef<BE> for GGSWBackendMut<'a, BE> {
    fn at_view(&self, row: usize, col: usize) -> GLWEViewRef<'_, BE> {
        GGSWBackendMut::at_view(self, row, col)
    }
}

impl<'a, BE: Backend + 'a> GGSWAtViewRef<BE> for &GGSWBackendRef<'a, BE> {
    fn at_view(&self, row: usize, col: usize) -> GLWEViewRef<'_, BE> {
        GGSWBackendRef::at_view(self, row, col)
    }
}

impl<D: Data> LWEInfos for GGSW<D> {
    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn size(&self) -> usize {
        self.data.size()
    }
}

impl<D: Data> LWEInfos for &GGSW<D> {
    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn size(&self) -> usize {
        self.data.size()
    }
}

impl<D: Data> GLWEInfos for GGSW<D> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols_out() as u32 - 1)
    }
}

impl<D: Data> GLWEInfos for &GGSW<D> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols_out() as u32 - 1)
    }
}

impl<D: Data> GGSWInfos for GGSW<D> {
    fn dsize(&self) -> Dsize {
        self.dsize
    }

    fn dnum(&self) -> Dnum {
        Dnum(self.data.rows() as u32)
    }
}

impl<D: Data> GGSWInfos for &GGSW<D> {
    fn dsize(&self) -> Dsize {
        self.dsize
    }

    fn dnum(&self) -> Dnum {
        Dnum(self.data.rows() as u32)
    }
}

impl<D: HostDataRef> fmt::Debug for GGSW<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.data)
    }
}

impl<D: HostDataRef> fmt::Display for GGSW<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(GGSW: k: {} base2k: {} dsize: {}) {}",
            self.max_k().0,
            self.base2k().0,
            self.dsize().0,
            self.data
        )
    }
}

impl<D: HostDataMut> FillUniform for GGSW<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

impl<D: HostDataRef> GGSW<D> {
    pub fn at(&self, row: usize, col: usize) -> GLWE<&[u8]> {
        let data = self.data.at(row, col);
        GLWE {
            base2k: self.base2k,
            data,
        }
    }
}

pub(crate) trait GGSWAtBackendRef<BE: Backend> {
    fn at_backend(&self, row: usize, col: usize) -> GLWE<BE::BufRef<'_>>;
}

impl<BE: Backend> GGSWAtBackendRef<BE> for GGSW<BE::OwnedBuf> {
    fn at_backend(&self, row: usize, col: usize) -> GLWE<BE::BufRef<'_>> {
        let data = <MatZnx<BE::OwnedBuf> as MatZnxAtBackendRef<BE>>::at_backend(&self.data, row, col);
        GLWE {
            base2k: self.base2k,
            data,
        }
    }
}

pub(crate) fn ggsw_at_backend_ref_from_ref<'a, 'b, BE: Backend>(
    ggsw: &'a GGSW<BE::BufRef<'b>>,
    row: usize,
    col: usize,
) -> GLWE<BE::BufRef<'a>> {
    let data = poulpy_hal::layouts::mat_znx_at_backend_ref_from_ref::<BE>(&ggsw.data, row, col);
    GLWE {
        base2k: ggsw.base2k,
        data,
    }
}

pub trait GGSWAtViewRef<BE: Backend> {
    fn at_view(&self, row: usize, col: usize) -> GLWEViewRef<'_, BE>;
}

impl<BE: Backend> GGSWAtViewRef<BE> for GGSW<BE::OwnedBuf> {
    fn at_view(&self, row: usize, col: usize) -> GLWEViewRef<'_, BE> {
        GLWEViewRef::from_inner(<GGSW<BE::OwnedBuf> as GGSWAtBackendRef<BE>>::at_backend(self, row, col))
    }
}

impl<'b, BE: Backend + 'b> GGSWAtViewRef<BE> for &GGSW<BE::BufRef<'b>> {
    fn at_view(&self, row: usize, col: usize) -> GLWEViewRef<'_, BE> {
        GLWEViewRef::from_inner(ggsw_at_backend_ref_from_ref::<BE>(self, row, col))
    }
}

pub(crate) fn ggsw_at_backend_ref_from_mut<'a, 'b, BE: Backend>(
    ggsw: &'a GGSW<BE::BufMut<'b>>,
    row: usize,
    col: usize,
) -> GLWE<BE::BufRef<'a>> {
    let data = poulpy_hal::layouts::mat_znx_at_backend_ref_from_mut::<BE>(&ggsw.data, row, col);
    GLWE {
        base2k: ggsw.base2k,
        data,
    }
}

impl<D: HostDataMut> GGSW<D> {
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWE<&mut [u8]> {
        let base2k = self.base2k;
        let data = self.data.at_mut(row, col);
        GLWE { base2k, data }
    }
}

pub(crate) trait GGSWAtBackendMut<BE: Backend> {
    fn at_backend_mut(&mut self, row: usize, col: usize) -> GLWE<BE::BufMut<'_>>;
}

impl<BE: Backend> GGSWAtBackendMut<BE> for GGSW<BE::OwnedBuf> {
    fn at_backend_mut(&mut self, row: usize, col: usize) -> GLWE<BE::BufMut<'_>> {
        let base2k = self.base2k;
        let data = <MatZnx<BE::OwnedBuf> as MatZnxAtBackendMut<BE>>::at_backend_mut(&mut self.data, row, col);
        GLWE { base2k, data }
    }
}

pub(crate) fn ggsw_at_backend_mut_from_mut<'a, 'b, BE: Backend>(
    ggsw: &'a mut GGSW<BE::BufMut<'b>>,
    row: usize,
    col: usize,
) -> GLWE<BE::BufMut<'a>> {
    let base2k = ggsw.base2k;
    let data = poulpy_hal::layouts::mat_znx_at_backend_mut_from_mut::<BE>(&mut ggsw.data, row, col);
    GLWE { base2k, data }
}

pub trait GGSWAtViewMut<BE: Backend> {
    fn at_view_mut(&mut self, row: usize, col: usize) -> GLWEViewMut<'_, BE>;
}

impl<BE: Backend> GGSWAtViewMut<BE> for GGSW<BE::OwnedBuf> {
    fn at_view_mut(&mut self, row: usize, col: usize) -> GLWEViewMut<'_, BE> {
        GLWEViewMut::from_inner(<GGSW<BE::OwnedBuf> as GGSWAtBackendMut<BE>>::at_backend_mut(self, row, col))
    }
}

impl<'b, BE: Backend + 'b> GGSWAtViewMut<BE> for &mut GGSW<BE::BufMut<'b>> {
    fn at_view_mut(&mut self, row: usize, col: usize) -> GLWEViewMut<'_, BE> {
        GLWEViewMut::from_inner(ggsw_at_backend_mut_from_mut::<BE>(*self, row, col))
    }
}

impl<'a, BE: Backend + 'a> GGSWAtViewMut<BE> for GGSWBackendMut<'a, BE> {
    fn at_view_mut(&mut self, row: usize, col: usize) -> GLWEViewMut<'_, BE> {
        GGSWBackendMut::at_view_mut(self, row, col)
    }
}

impl<D: HostDataRef> GGSW<D> {
    /// Copies this ciphertext's backing bytes into an owned buffer of
    /// backend `To`, routing via host bytes.
    pub fn to_backend<BE, To>(&self, dst: &Module<To>) -> GGSW<To::OwnedBuf>
    where
        BE: Backend<OwnedBuf = D>,
        To: Backend,
        To: TransferFrom<BE>,
    {
        dst.upload_ggsw(self)
    }
}

impl<D: Data> GGSW<D> {
    /// Zero-cost rename when both backends share the same `OwnedBuf`.
    pub fn reinterpret<To>(self) -> GGSW<To::OwnedBuf>
    where
        To: Backend<OwnedBuf = D>,
    {
        let (n, rows, cols_in, cols_out, size) = (
            self.data.n(),
            self.data.rows(),
            self.data.cols_in(),
            self.data.cols_out(),
            self.data.size(),
        );
        GGSW {
            data: MatZnx::from_data(self.data.into_data(), n, rows, cols_in, cols_out, size),
            base2k: self.base2k,
            dsize: self.dsize,
        }
    }
}

#[expect(
    dead_code,
    reason = "host-owned constructors are kept for serialization and host-only staging"
)]
impl GGSW<Vec<u8>> {
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GGSWInfos,
    {
        Self::alloc(
            infos.n(),
            infos.base2k(),
            infos.max_k(),
            infos.rank(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    pub(crate) fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> Self {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        assert!(
            size as u32 > dsize.0,
            "invalid ggsw: ceil(k/base2k): {size} <= dsize: {}",
            dsize.0
        );

        assert!(
            dnum.0 * dsize.0 <= size as u32,
            "invalid ggsw: dnum: {} * dsize:{} > ceil(k/base2k): {size}",
            dnum.0,
            dsize.0,
        );

        GGSW {
            data: MatZnx::from_data(
                poulpy_hal::layouts::HostBytesBackend::alloc_bytes(MatZnx::<Vec<u8>>::bytes_of(
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
            base2k,
            dsize,
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        Self::bytes_of(
            infos.n(),
            infos.base2k(),
            infos.max_k(),
            infos.rank(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    pub fn bytes_of(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank, dnum: Dnum, dsize: Dsize) -> usize {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        assert!(
            size as u32 > dsize.0,
            "invalid ggsw: ceil(k/base2k): {size} <= dsize: {}",
            dsize.0
        );

        assert!(
            dnum.0 * dsize.0 <= size as u32,
            "invalid ggsw: dnum: {} * dsize:{} > ceil(k/base2k): {size}",
            dnum.0,
            dsize.0,
        );

        MatZnx::bytes_of(
            n.into(),
            dnum.into(),
            (rank + 1).into(),
            (rank + 1).into(),
            k.0.div_ceil(base2k.0) as usize,
        )
    }
}

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

impl<D: HostDataMut> ReaderFrom for GGSW<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        self.dsize = Dsize(reader.read_u32::<LittleEndian>()?);
        self.data.read_from(reader)
    }
}

impl<D: HostDataRef> WriterTo for GGSW<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.base2k.into())?;
        writer.write_u32::<LittleEndian>(self.dsize.into())?;
        self.data.write_to(writer)
    }
}

pub trait GGSWToBackendMut<BE: Backend>: GGSWToBackendRef<BE> {
    fn to_backend_mut(&mut self) -> GGSWBackendMut<'_, BE>;
}

impl<BE: Backend, D: Data> GGSWToBackendMut<BE> for GGSW<D>
where
    MatZnx<D>: MatZnxToBackendRef<BE> + MatZnxToBackendMut<BE>,
{
    fn to_backend_mut(&mut self) -> GGSWBackendMut<'_, BE> {
        GGSWBackendMut::from_inner(GGSW {
            dsize: self.dsize,
            base2k: self.base2k,
            data: self.data.to_backend_mut(),
        })
    }
}

impl<'b, BE: Backend + 'b> GGSWToBackendRef<BE> for &mut GGSW<BE::BufMut<'b>> {
    fn to_backend_ref(&self) -> GGSWBackendRef<'_, BE> {
        GGSWBackendRef::from_inner(GGSW {
            dsize: self.dsize,
            base2k: self.base2k,
            data: poulpy_hal::layouts::mat_znx_backend_ref_from_mut::<BE>(&self.data),
        })
    }
}

impl<'b, BE: Backend + 'b> GGSWToBackendMut<BE> for &mut GGSW<BE::BufMut<'b>> {
    fn to_backend_mut(&mut self) -> GGSWBackendMut<'_, BE> {
        ggsw_backend_mut_from_mut::<BE>(self)
    }
}

pub fn ggsw_backend_mut_from_mut<'a, 'b, BE: Backend>(ggsw: &'a mut GGSW<BE::BufMut<'b>>) -> GGSWBackendMut<'a, BE> {
    GGSWBackendMut::from_inner(GGSW {
        dsize: ggsw.dsize,
        base2k: ggsw.base2k,
        data: poulpy_hal::layouts::mat_znx_backend_mut_from_mut::<BE>(&mut ggsw.data),
    })
}

/// Row-view adapter that lets a `GGSWToBackendMut` type satisfy both `GGSWAtViewRef` and
/// `GGSWAtViewMut` simultaneously, which is required by several default algorithms that need
/// to read and write individual GLWE rows through the trait interface.
pub struct GGSWBackendRowViewMut<'a, BE: Backend + 'a> {
    inner: GGSWBackendMut<'a, BE>,
}

impl<'a, BE: Backend + 'a> GGSWBackendRowViewMut<'a, BE> {
    pub fn from_inner(inner: GGSWBackendMut<'a, BE>) -> Self {
        Self { inner }
    }
}

impl<'a, BE: Backend + 'a> LWEInfos for GGSWBackendRowViewMut<'a, BE> {
    fn base2k(&self) -> Base2K {
        self.inner.base2k()
    }
    fn n(&self) -> Degree {
        self.inner.n()
    }
    fn size(&self) -> usize {
        self.inner.size()
    }
}

impl<'a, BE: Backend + 'a> GLWEInfos for GGSWBackendRowViewMut<'a, BE> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

impl<'a, BE: Backend + 'a> GGSWInfos for GGSWBackendRowViewMut<'a, BE> {
    fn dnum(&self) -> Dnum {
        self.inner.dnum()
    }
    fn dsize(&self) -> Dsize {
        self.inner.dsize()
    }
}

impl<'a, BE: Backend + 'a> GGSWToBackendRef<BE> for GGSWBackendRowViewMut<'a, BE> {
    fn to_backend_ref(&self) -> GGSWBackendRef<'_, BE> {
        self.inner.to_backend_ref()
    }
}

impl<'a, BE: Backend + 'a> GGSWToBackendMut<BE> for GGSWBackendRowViewMut<'a, BE> {
    fn to_backend_mut(&mut self) -> GGSWBackendMut<'_, BE> {
        GGSWBackendMut::from_inner(GGSW {
            dsize: self.inner.inner.dsize,
            base2k: self.inner.inner.base2k,
            data: poulpy_hal::layouts::mat_znx_backend_mut_from_mut::<BE>(&mut self.inner.inner.data),
        })
    }
}

impl<'a, BE: Backend + 'a> GGSWAtViewRef<BE> for GGSWBackendRowViewMut<'a, BE> {
    fn at_view(&self, row: usize, col: usize) -> GLWEViewRef<'_, BE> {
        self.inner.at_view(row, col)
    }
}

impl<'a, BE: Backend + 'a> GGSWAtViewMut<BE> for GGSWBackendRowViewMut<'a, BE> {
    fn at_view_mut(&mut self, row: usize, col: usize) -> GLWEViewMut<'_, BE> {
        self.inner.at_view_mut(row, col)
    }
}

pub trait GGSWToBackendRef<BE: Backend> {
    fn to_backend_ref(&self) -> GGSWBackendRef<'_, BE>;
}

impl<BE: Backend, D: Data> GGSWToBackendRef<BE> for GGSW<D>
where
    MatZnx<D>: MatZnxToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GGSWBackendRef<'_, BE> {
        GGSWBackendRef::from_inner(GGSW {
            dsize: self.dsize,
            base2k: self.base2k,
            data: self.data.to_backend_ref(),
        })
    }
}
