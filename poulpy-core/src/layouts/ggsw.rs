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
    /// Number of key limbs an operation should process when applied to an input
    /// ciphertext of precision `input_k`, clamped to the key's allocated width.
    /// The input region is rounded to whole `dsize`-limb gadget digits before
    /// the auxiliary limbs are added; see [`crate::layouts::key_work_size`].
    fn work_size(&self, input_k: TorusPrecision) -> usize {
        self.size().min(crate::layouts::key_work_size(
            self.base2k(),
            input_k,
            self.dsize(),
            self.k_aux(),
        ))
    }
    /// Returns the decomposition digit size.
    fn dsize(&self) -> Dsize;
    /// Returns a plain-data [`GGSWLayout`] snapshot of the current parameters.
    fn ggsw_layout(&self) -> GGSWLayout {
        GGSWLayout {
            n: self.n(),
            base2k: self.base2k(),
            dnum: self.dnum(),
            k_aux: self.k_aux(),
            rank: self.rank(),
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
    /// Number of gadget-decomposition rows.
    pub dnum: Dnum,
    /// Auxiliary guard precision (torus bits) below the gadget region.
    pub k_aux: TorusPrecision,
    /// GLWE rank (number of mask polynomials per row).
    pub rank: Rank,
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

    fn max_size(&self) -> usize {
        crate::layouts::key_size(self.base2k, self.dnum, self.dsize, self.k_aux)
    }

    fn k(&self) -> TorusPrecision {
        crate::layouts::key_k(self.base2k, self.dnum, self.dsize, self.k_aux)
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

    fn dsize(&self) -> Dsize {
        self.dsize
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
pub struct GGSW<D: Data, W: ZnxWord> {
    pub(crate) data: MatZnx<D, W>,
    pub(crate) k_aux: TorusPrecision,
    pub(crate) base2k: Base2K,
    pub(crate) dsize: Dsize,
}

pub struct GGSWBackendRef<'a, BE: Backend + 'a> {
    inner: GGSW<BE::BufRef<'a>, BE::ZnxWord>,
}

impl<'a, BE: Backend + 'a> GGSWBackendRef<'a, BE> {
    pub fn from_inner(inner: GGSW<BE::BufRef<'a>, BE::ZnxWord>) -> Self {
        Self { inner }
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

    pub fn into_inner(self) -> GGSW<BE::BufMut<'a>, BE::ZnxWord> {
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
    type Target = GGSW<BE::BufMut<'a>, BE::ZnxWord>;

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

    fn max_size(&self) -> usize {
        self.inner.max_size()
    }

    fn k(&self) -> TorusPrecision {
        self.inner.k()
    }
}

impl<'a, BE: Backend + 'a> GLWEInfos for GGSWBackendRef<'a, BE> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

impl<'a, BE: Backend + 'a> GGSWInfos for GGSWBackendRef<'a, BE> {
    fn k_aux(&self) -> TorusPrecision {
        self.inner.k_aux()
    }

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

    fn max_size(&self) -> usize {
        self.inner.max_size()
    }

    fn k(&self) -> TorusPrecision {
        self.inner.k()
    }
}

impl<'a, BE: Backend + 'a> GLWEInfos for GGSWBackendMut<'a, BE> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

impl<'a, BE: Backend + 'a> GGSWInfos for GGSWBackendMut<'a, BE> {
    fn k_aux(&self) -> TorusPrecision {
        self.inner.k_aux()
    }

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
            dsize: self.inner.dsize(),
            base2k: self.inner.base2k(),
            k_aux: self.inner.k_aux(),
            data: poulpy_hal::layouts::mat_znx_backend_ref_from_ref::<BE>(&self.inner.data),
        })
    }
}

impl<'a, BE: Backend + 'a> GGSWToBackendRef<BE> for GGSWBackendMut<'a, BE> {
    fn to_backend_ref(&self) -> GGSWBackendRef<'_, BE> {
        GGSWBackendRef::from_inner(GGSW {
            dsize: self.inner.dsize,
            base2k: self.inner.base2k,
            k_aux: self.inner.k_aux,
            data: poulpy_hal::layouts::mat_znx_backend_ref_from_mut::<BE>(&self.inner.data),
        })
    }
}

impl<'a, BE: Backend + 'a> GGSWToBackendMut<BE> for GGSWBackendMut<'a, BE> {
    fn to_backend_mut(&mut self) -> GGSWBackendMut<'_, BE> {
        GGSWBackendMut::from_inner(GGSW {
            dsize: self.inner.dsize,
            base2k: self.inner.base2k,
            k_aux: self.inner.k_aux,
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

impl<D: Data, W: ZnxWord> LWEInfos for GGSW<D, W> {
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
        crate::layouts::key_k(self.base2k, self.dnum(), self.dsize, self.k_aux)
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

    fn dsize(&self) -> Dsize {
        self.dsize
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

impl<D: HostDataMut, W: ZnxWord> FillUniform for GGSW<D, W> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

impl<D: HostDataRef, W: ZnxWord> GGSW<D, W> {
    pub fn at(&self, row: usize, col: usize) -> GLWE<&[u8], W> {
        let data = self.data.at(row, col);
        GLWE {
            base2k: self.base2k,
            k: self.k(),
            data,
        }
    }
}

pub(crate) trait GGSWAtBackendRef<BE: Backend> {
    fn at_backend(&self, row: usize, col: usize) -> GLWE<BE::BufRef<'_>, BE::ZnxWord>;
}

impl<BE: Backend> GGSWAtBackendRef<BE> for GGSW<BE::OwnedBuf, BE::ZnxWord> {
    fn at_backend(&self, row: usize, col: usize) -> GLWE<BE::BufRef<'_>, BE::ZnxWord> {
        let data = <MatZnx<BE::OwnedBuf, BE::ZnxWord> as MatZnxAtBackendRef<BE>>::at_backend(&self.data, row, col);
        GLWE {
            base2k: self.base2k,
            k: self.k(),
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
        base2k: ggsw.base2k,
        k: ggsw.k(),
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
        base2k: ggsw.base2k,
        k: ggsw.k(),
        data,
    }
}

impl<D: HostDataMut, W: ZnxWord> GGSW<D, W> {
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWE<&mut [u8], W> {
        let base2k = self.base2k;
        let k = self.k();
        let data = self.data.at_mut(row, col);
        GLWE { base2k, k, data }
    }
}

pub(crate) trait GGSWAtBackendMut<BE: Backend> {
    fn at_backend_mut(&mut self, row: usize, col: usize) -> GLWE<BE::BufMut<'_>, BE::ZnxWord>;
}

impl<BE: Backend> GGSWAtBackendMut<BE> for GGSW<BE::OwnedBuf, BE::ZnxWord> {
    fn at_backend_mut(&mut self, row: usize, col: usize) -> GLWE<BE::BufMut<'_>, BE::ZnxWord> {
        let base2k = self.base2k;
        let k = self.k();
        let data = <MatZnx<BE::OwnedBuf, BE::ZnxWord> as MatZnxAtBackendMut<BE>>::at_backend_mut(&mut self.data, row, col);
        GLWE { base2k, k, data }
    }
}

pub(crate) fn ggsw_at_backend_mut_from_mut<'a, 'b, BE: Backend>(
    ggsw: &'a mut GGSW<BE::BufMut<'b>, BE::ZnxWord>,
    row: usize,
    col: usize,
) -> GLWE<BE::BufMut<'a>, BE::ZnxWord> {
    let base2k = ggsw.base2k;
    let k = ggsw.k();
    let data = poulpy_hal::layouts::mat_znx_at_backend_mut_from_mut::<BE>(&mut ggsw.data, row, col);
    GLWE { base2k, k, data }
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

impl<'a, BE: Backend + 'a> GGSWAtViewMut<BE> for GGSWBackendMut<'a, BE> {
    fn at_view_mut(&mut self, row: usize, col: usize) -> GLWEViewMut<'_, BE> {
        GGSWBackendMut::at_view_mut(self, row, col)
    }
}

impl<D: HostDataRef, W: ZnxWord> GGSW<D, W> {
    /// Copies this ciphertext's backing bytes into an owned buffer of
    /// backend `To`, routing via host bytes.
    pub fn to_backend<BE, To>(&self, dst: &Module<To>) -> GGSW<To::OwnedBuf, To::ZnxWord>
    where
        BE: Backend<OwnedBuf = D, ZnxWord = W>,
        To: Backend<ZnxWord = W>,
        To: TransferFrom<BE>,
    {
        dst.upload_ggsw(self)
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
            data: MatZnx::from_data(self.data.into_data(), n, rows, cols_in, cols_out, size),
            k_aux: self.k_aux,
            base2k: self.base2k,
            dsize: self.dsize,
        }
    }
}

#[expect(
    dead_code,
    reason = "host-owned constructors are kept for serialization and host-only staging"
)]
impl<W: ZnxWord> GGSW<Vec<u8>, W> {
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GGSWInfos,
    {
        Self::alloc(
            infos.n(),
            infos.base2k(),
            infos.dnum(),
            infos.dsize(),
            infos.k_aux(),
            infos.rank(),
        )
    }

    pub(crate) fn alloc(n: Degree, base2k: Base2K, dnum: Dnum, dsize: Dsize, k_aux: TorusPrecision, rank: Rank) -> Self {
        let size: usize = crate::layouts::key_size(base2k, dnum, dsize, k_aux);

        GGSW {
            data: MatZnx::from_data(
                poulpy_hal::layouts::HostBytesBackend::alloc_bytes(MatZnx::<Vec<u8>, W>::bytes_of(
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
            infos.dnum(),
            infos.dsize(),
            infos.k_aux(),
            infos.rank(),
        )
    }

    pub fn bytes_of(n: Degree, base2k: Base2K, dnum: Dnum, dsize: Dsize, k_aux: TorusPrecision, rank: Rank) -> usize {
        let size: usize = crate::layouts::key_size(base2k, dnum, dsize, k_aux);

        MatZnx::<Vec<u8>, W>::bytes_of(n.into(), dnum.into(), (rank + 1).into(), (rank + 1).into(), size)
    }
}

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for GGSW<D, W> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        self.dsize = Dsize(reader.read_u32::<LittleEndian>()?);
        self.k_aux = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        self.data.read_from(reader)
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for GGSW<D, W> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.base2k.into())?;
        writer.write_u32::<LittleEndian>(self.dsize.into())?;
        writer.write_u32::<LittleEndian>(self.k_aux.into())?;
        self.data.write_to(writer)
    }
}

pub trait GGSWToBackendMut<BE: Backend>: GGSWToBackendRef<BE> {
    fn to_backend_mut(&mut self) -> GGSWBackendMut<'_, BE>;
}

impl<BE: Backend, D: Data> GGSWToBackendMut<BE> for GGSW<D, BE::ZnxWord>
where
    MatZnx<D, BE::ZnxWord>: MatZnxToBackendRef<BE> + MatZnxToBackendMut<BE>,
{
    fn to_backend_mut(&mut self) -> GGSWBackendMut<'_, BE> {
        GGSWBackendMut::from_inner(GGSW {
            dsize: self.dsize,
            base2k: self.base2k,
            k_aux: self.k_aux,
            data: self.data.to_backend_mut(),
        })
    }
}

impl<'b, BE: Backend + 'b> GGSWToBackendRef<BE> for &mut GGSW<BE::BufMut<'b>, BE::ZnxWord> {
    fn to_backend_ref(&self) -> GGSWBackendRef<'_, BE> {
        GGSWBackendRef::from_inner(GGSW {
            dsize: self.dsize,
            base2k: self.base2k,
            k_aux: self.k_aux,
            data: poulpy_hal::layouts::mat_znx_backend_ref_from_mut::<BE>(&self.data),
        })
    }
}

impl<'b, BE: Backend + 'b> GGSWToBackendMut<BE> for &mut GGSW<BE::BufMut<'b>, BE::ZnxWord> {
    fn to_backend_mut(&mut self) -> GGSWBackendMut<'_, BE> {
        ggsw_backend_mut_from_mut::<BE>(self)
    }
}

pub fn ggsw_backend_mut_from_mut<'a, 'b, BE: Backend>(ggsw: &'a mut GGSW<BE::BufMut<'b>, BE::ZnxWord>) -> GGSWBackendMut<'a, BE> {
    GGSWBackendMut::from_inner(GGSW {
        dsize: ggsw.dsize,
        base2k: ggsw.base2k,
        k_aux: ggsw.k_aux,
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
    fn max_size(&self) -> usize {
        self.inner.max_size()
    }
    fn k(&self) -> TorusPrecision {
        self.inner.k()
    }
}

impl<'a, BE: Backend + 'a> GLWEInfos for GGSWBackendRowViewMut<'a, BE> {
    fn rank(&self) -> Rank {
        self.inner.rank()
    }
}

impl<'a, BE: Backend + 'a> GGSWInfos for GGSWBackendRowViewMut<'a, BE> {
    fn k_aux(&self) -> TorusPrecision {
        self.inner.k_aux()
    }
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
            k_aux: self.inner.inner.k_aux,
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

impl<BE: Backend, D: Data> GGSWToBackendRef<BE> for GGSW<D, BE::ZnxWord>
where
    MatZnx<D, BE::ZnxWord>: MatZnxToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GGSWBackendRef<'_, BE> {
        GGSWBackendRef::from_inner(GGSW {
            dsize: self.dsize,
            base2k: self.base2k,
            k_aux: self.k_aux,
            data: self.data.to_backend_ref(),
        })
    }
}
