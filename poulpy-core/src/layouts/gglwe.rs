use poulpy_hal::{
    layouts::{
        Backend, Data, FillUniform, HostDataMut, HostDataRef, MatZnx, MatZnxAtBackendMut, MatZnxAtBackendRef, MatZnxToBackendMut,
        MatZnxToBackendRef, Module, ReaderFrom, TransferFrom, WriterTo,
    },
    source::Source,
};

use crate::api::ModuleTransfer;
use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGSWAtViewRef, GLWE, GLWEInfos, GLWEViewMut, GLWEViewRef, LWEInfos, Rank, TorusPrecision,
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use std::{
    fmt,
    ops::{Deref, DerefMut},
};

pub trait GGLWEInfos
where
    Self: GLWEInfos,
{
    fn dnum(&self) -> Dnum;
    fn dsize(&self) -> Dsize;
    fn rank_in(&self) -> Rank;
    fn rank_out(&self) -> Rank;
    fn gglwe_layout(&self) -> GGLWELayout {
        GGLWELayout {
            n: self.n(),
            base2k: self.base2k(),
            k: self.max_k(),
            rank_in: self.rank_in(),
            rank_out: self.rank_out(),
            dsize: self.dsize(),
            dnum: self.dnum(),
        }
    }
}

pub trait SetGGLWEInfos {
    fn set_dsize(&mut self, dsize: usize);
}

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GGLWELayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
    pub rank_in: Rank,
    pub rank_out: Rank,
    pub dnum: Dnum,
    pub dsize: Dsize,
}

impl LWEInfos for GGLWELayout {
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

impl GLWEInfos for GGLWELayout {
    fn rank(&self) -> Rank {
        self.rank_out
    }
}

impl GGLWEInfos for GGLWELayout {
    fn rank_in(&self) -> Rank {
        self.rank_in
    }

    fn dsize(&self) -> Dsize {
        self.dsize
    }

    fn rank_out(&self) -> Rank {
        self.rank_out
    }

    fn dnum(&self) -> Dnum {
        self.dnum
    }
}

#[derive(PartialEq, Eq, Clone)]
pub struct GGLWE<D: Data> {
    pub(crate) data: MatZnx<D>,
    pub(crate) base2k: Base2K,
    pub(crate) dsize: Dsize,
}

pub struct GGLWEBackendRef<'a, BE: Backend + 'a> {
    inner: GGLWE<BE::BufRef<'a>>,
}

impl<'a, BE: Backend + 'a> GGLWEBackendRef<'a, BE> {
    pub fn from_inner(inner: GGLWE<BE::BufRef<'a>>) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> GGLWE<BE::BufRef<'a>> {
        self.inner
    }

    pub fn at_view(&self, row: usize, col: usize) -> GLWEViewRef<'_, BE> {
        GLWEViewRef::from_inner(gglwe_at_backend_ref_from_ref::<BE>(&self.inner, row, col))
    }
}

impl<'a, BE: Backend + 'a> Deref for GGLWEBackendRef<'a, BE> {
    type Target = GGLWE<BE::BufRef<'a>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub struct GGLWEBackendMut<'a, BE: Backend + 'a> {
    inner: GGLWE<BE::BufMut<'a>>,
}

impl<'a, BE: Backend + 'a> GGLWEBackendMut<'a, BE> {
    pub fn from_inner(inner: GGLWE<BE::BufMut<'a>>) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> GGLWE<BE::BufMut<'a>> {
        self.inner
    }

    pub fn at_view(&self, row: usize, col: usize) -> GLWEViewRef<'_, BE> {
        GLWEViewRef::from_inner(gglwe_at_backend_ref_from_mut::<BE>(&self.inner, row, col))
    }

    pub fn at_view_mut(&mut self, row: usize, col: usize) -> GLWEViewMut<'_, BE> {
        GLWEViewMut::from_inner(gglwe_at_backend_mut_from_mut::<BE>(&mut self.inner, row, col))
    }
}

impl<'a, BE: Backend + 'a> Deref for GGLWEBackendMut<'a, BE> {
    type Target = GGLWE<BE::BufMut<'a>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'a, BE: Backend + 'a> DerefMut for GGLWEBackendMut<'a, BE> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl_gglwe_infos_for_inner!(GGLWEBackendRef<'a, BE>, ['a, BE: Backend + 'a]; inner);
impl_gglwe_infos_for_inner!(GGLWEBackendMut<'a, BE>, ['a, BE: Backend + 'a]; inner);

impl<'a, BE: Backend + 'a> GGLWEToBackendRef<BE> for GGLWEBackendRef<'a, BE> {
    fn to_backend_ref(&self) -> GGLWEBackendRef<'_, BE> {
        GGLWEBackendRef::from_inner(GGLWE {
            base2k: self.inner.base2k,
            dsize: self.inner.dsize,
            data: poulpy_hal::layouts::mat_znx_backend_ref_from_ref::<BE>(&self.inner.data),
        })
    }
}

impl<'a, BE: Backend + 'a> GGSWAtViewRef<BE> for GGLWEBackendRef<'a, BE> {
    fn at_view(&self, row: usize, col: usize) -> GLWEViewRef<'_, BE> {
        GGLWEBackendRef::at_view(self, row, col)
    }
}

impl<'a, BE: Backend + 'a> GGLWEToBackendRef<BE> for GGLWEBackendMut<'a, BE> {
    fn to_backend_ref(&self) -> GGLWEBackendRef<'_, BE> {
        GGLWEBackendRef::from_inner(GGLWE {
            base2k: self.inner.base2k,
            dsize: self.inner.dsize,
            data: poulpy_hal::layouts::mat_znx_backend_ref_from_mut::<BE>(&self.inner.data),
        })
    }
}

impl<'a, BE: Backend + 'a> GGLWEToBackendMut<BE> for GGLWEBackendMut<'a, BE> {
    fn to_backend_mut(&mut self) -> GGLWEBackendMut<'_, BE> {
        GGLWEBackendMut::from_inner(GGLWE {
            base2k: self.inner.base2k,
            dsize: self.inner.dsize,
            data: poulpy_hal::layouts::mat_znx_backend_mut_from_mut::<BE>(&mut self.inner.data),
        })
    }
}

impl<D: Data> LWEInfos for GGLWE<D> {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn size(&self) -> usize {
        self.data.size()
    }
}

impl<D: Data> LWEInfos for &GGLWE<D> {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn size(&self) -> usize {
        self.data.size()
    }
}

impl<D: Data> GLWEInfos for GGLWE<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data> GLWEInfos for &GGLWE<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data> GGLWEInfos for GGLWE<D> {
    fn rank_in(&self) -> Rank {
        Rank(self.data.cols_in() as u32)
    }

    fn rank_out(&self) -> Rank {
        Rank(self.data.cols_out() as u32 - 1)
    }

    fn dsize(&self) -> Dsize {
        self.dsize
    }

    fn dnum(&self) -> Dnum {
        Dnum(self.data.rows() as u32)
    }
}

impl<D: Data> GGLWEInfos for &GGLWE<D> {
    fn rank_in(&self) -> Rank {
        Rank(self.data.cols_in() as u32)
    }

    fn rank_out(&self) -> Rank {
        Rank(self.data.cols_out() as u32 - 1)
    }

    fn dsize(&self) -> Dsize {
        self.dsize
    }

    fn dnum(&self) -> Dnum {
        Dnum(self.data.rows() as u32)
    }
}

impl<D: Data> LWEInfos for &mut GGLWE<D> {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn size(&self) -> usize {
        self.data.size()
    }
}

impl<D: Data> GLWEInfos for &mut GGLWE<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data> GGLWEInfos for &mut GGLWE<D> {
    fn rank_in(&self) -> Rank {
        Rank(self.data.cols_in() as u32)
    }

    fn rank_out(&self) -> Rank {
        Rank(self.data.cols_out() as u32 - 1)
    }

    fn dsize(&self) -> Dsize {
        self.dsize
    }

    fn dnum(&self) -> Dnum {
        Dnum(self.data.rows() as u32)
    }
}

impl<D: HostDataRef> GGLWE<D> {
    pub fn data(&self) -> &MatZnx<D> {
        &self.data
    }
}

pub(crate) trait GGLWEAtBackendRef<BE: Backend> {
    fn at_backend(&self, row: usize, col: usize) -> GLWE<BE::BufRef<'_>>;
}

impl<BE: Backend> GGLWEAtBackendRef<BE> for GGLWE<BE::OwnedBuf> {
    fn at_backend(&self, row: usize, col: usize) -> GLWE<BE::BufRef<'_>> {
        let data = <MatZnx<BE::OwnedBuf> as MatZnxAtBackendRef<BE>>::at_backend(&self.data, row, col);
        GLWE {
            base2k: self.base2k,
            data,
        }
    }
}

pub(crate) fn gglwe_at_backend_ref_from_ref<'a, 'b, BE: Backend>(
    gglwe: &'a GGLWE<BE::BufRef<'b>>,
    row: usize,
    col: usize,
) -> GLWE<BE::BufRef<'a>> {
    let data = poulpy_hal::layouts::mat_znx_at_backend_ref_from_ref::<BE>(&gglwe.data, row, col);
    GLWE {
        base2k: gglwe.base2k,
        data,
    }
}

pub trait GGLWEAtViewRef<BE: Backend> {
    fn at_view(&self, row: usize, col: usize) -> GLWEViewRef<'_, BE>;
}

impl<BE: Backend> GGLWEAtViewRef<BE> for GGLWE<BE::OwnedBuf> {
    fn at_view(&self, row: usize, col: usize) -> GLWEViewRef<'_, BE> {
        GLWEViewRef::from_inner(<GGLWE<BE::OwnedBuf> as GGLWEAtBackendRef<BE>>::at_backend(self, row, col))
    }
}

impl<'b, BE: Backend + 'b> GGLWEAtViewRef<BE> for &GGLWE<BE::BufRef<'b>> {
    fn at_view(&self, row: usize, col: usize) -> GLWEViewRef<'_, BE> {
        GLWEViewRef::from_inner(gglwe_at_backend_ref_from_ref::<BE>(self, row, col))
    }
}

pub(crate) fn gglwe_at_backend_ref_from_mut<'a, 'b, BE: Backend>(
    gglwe: &'a GGLWE<BE::BufMut<'b>>,
    row: usize,
    col: usize,
) -> GLWE<BE::BufRef<'a>> {
    let data = poulpy_hal::layouts::mat_znx_at_backend_ref_from_mut::<BE>(&gglwe.data, row, col);
    GLWE {
        base2k: gglwe.base2k,
        data,
    }
}

impl<D: HostDataMut> GGLWE<D> {
    pub fn data_mut(&mut self) -> &mut MatZnx<D> {
        &mut self.data
    }
}

pub(crate) trait GGLWEAtBackendMut<BE: Backend> {
    fn at_backend_mut(&mut self, row: usize, col: usize) -> GLWE<BE::BufMut<'_>>;
}

impl<BE: Backend> GGLWEAtBackendMut<BE> for GGLWE<BE::OwnedBuf> {
    fn at_backend_mut(&mut self, row: usize, col: usize) -> GLWE<BE::BufMut<'_>> {
        let base2k = self.base2k;
        let data = <MatZnx<BE::OwnedBuf> as MatZnxAtBackendMut<BE>>::at_backend_mut(&mut self.data, row, col);
        GLWE { base2k, data }
    }
}

pub(crate) fn gglwe_at_backend_mut_from_mut<'a, 'b, BE: Backend>(
    gglwe: &'a mut GGLWE<BE::BufMut<'b>>,
    row: usize,
    col: usize,
) -> GLWE<BE::BufMut<'a>> {
    let base2k = gglwe.base2k;
    let data = poulpy_hal::layouts::mat_znx_at_backend_mut_from_mut::<BE>(&mut gglwe.data, row, col);
    GLWE { base2k, data }
}

pub trait GGLWEAtViewMut<BE: Backend> {
    fn at_view_mut(&mut self, row: usize, col: usize) -> GLWEViewMut<'_, BE>;
}

impl<BE: Backend> GGLWEAtViewMut<BE> for GGLWE<BE::OwnedBuf> {
    fn at_view_mut(&mut self, row: usize, col: usize) -> GLWEViewMut<'_, BE> {
        GLWEViewMut::from_inner(<GGLWE<BE::OwnedBuf> as GGLWEAtBackendMut<BE>>::at_backend_mut(self, row, col))
    }
}

impl<'b, BE: Backend + 'b> GGLWEAtViewMut<BE> for &mut GGLWE<BE::BufMut<'b>> {
    fn at_view_mut(&mut self, row: usize, col: usize) -> GLWEViewMut<'_, BE> {
        GLWEViewMut::from_inner(gglwe_at_backend_mut_from_mut::<BE>(*self, row, col))
    }
}

impl<D: HostDataRef> fmt::Debug for GGLWE<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataMut> FillUniform for GGLWE<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

impl<D: HostDataRef> fmt::Display for GGLWE<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(GGLWE: k={} base2k={} dsize={}) {}",
            self.max_k().0,
            self.base2k().0,
            self.dsize().0,
            self.data
        )
    }
}

impl<D: HostDataRef> GGLWE<D> {
    pub fn at(&self, row: usize, col: usize) -> GLWE<&[u8]> {
        let data = self.data.at(row, col);
        GLWE {
            base2k: self.base2k,
            data,
        }
    }
}

impl<D: HostDataMut> GGLWE<D> {
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWE<&mut [u8]> {
        let base2k = self.base2k;
        let data = self.data.at_mut(row, col);
        GLWE { base2k, data }
    }
}

impl<D: HostDataRef> GGLWE<D> {
    /// Copies this ciphertext's backing bytes into an owned buffer of
    /// backend `To`, routing via host bytes.
    pub fn to_backend<BE, To>(&self, dst: &Module<To>) -> GGLWE<To::OwnedBuf>
    where
        BE: Backend<OwnedBuf = D>,
        To: Backend,
        To: TransferFrom<BE>,
    {
        dst.upload_gglwe(self)
    }
}

impl<D: Data> GGLWE<D> {
    /// Zero-cost rename when both backends share the same `OwnedBuf`.
    pub fn reinterpret<To>(self) -> GGLWE<To::OwnedBuf>
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
        GGLWE {
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
impl GGLWE<Vec<u8>> {
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        Self::alloc(
            infos.n(),
            infos.base2k(),
            infos.max_k(),
            infos.rank_in(),
            infos.rank_out(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    pub(crate) fn alloc(
        n: Degree,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> Self {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        assert!(
            size as u32 > dsize.0,
            "invalid gglwe: ceil(k/base2k): {size} <= dsize: {}",
            dsize.0
        );

        assert!(
            dnum.0 * dsize.0 <= size as u32,
            "invalid gglwe: dnum: {} * dsize:{} > ceil(k/base2k): {size}",
            dnum.0,
            dsize.0,
        );

        GGLWE {
            data: MatZnx::from_data(
                poulpy_hal::layouts::HostBytesBackend::alloc_bytes(MatZnx::<Vec<u8>>::bytes_of(
                    n.into(),
                    dnum.into(),
                    rank_in.into(),
                    (rank_out + 1).into(),
                    size,
                )),
                n.into(),
                dnum.into(),
                rank_in.into(),
                (rank_out + 1).into(),
                size,
            ),
            base2k,
            dsize,
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        Self::bytes_of(
            infos.n(),
            infos.base2k(),
            infos.max_k(),
            infos.rank_in(),
            infos.rank_out(),
            infos.dnum(),
            infos.dsize(),
        )
    }

    pub fn bytes_of(
        n: Degree,
        base2k: Base2K,
        k: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
        dnum: Dnum,
        dsize: Dsize,
    ) -> usize {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        assert!(
            size as u32 > dsize.0,
            "invalid gglwe: ceil(k/base2k): {size} <= dsize: {}",
            dsize.0
        );

        assert!(
            dnum.0 * dsize.0 <= size as u32,
            "invalid gglwe: dnum: {} * dsize:{} > ceil(k/base2k): {size}",
            dnum.0,
            dsize.0,
        );

        MatZnx::bytes_of(
            n.into(),
            dnum.into(),
            rank_in.into(),
            (rank_out + 1).into(),
            k.0.div_ceil(base2k.0) as usize,
        )
    }
}

pub trait GGLWEToBackendMut<BE: Backend>: GGLWEToBackendRef<BE> {
    fn to_backend_mut(&mut self) -> GGLWEBackendMut<'_, BE>;
}

impl<BE: Backend, D: Data> GGLWEToBackendMut<BE> for GGLWE<D>
where
    MatZnx<D>: MatZnxToBackendRef<BE> + MatZnxToBackendMut<BE>,
{
    fn to_backend_mut(&mut self) -> GGLWEBackendMut<'_, BE> {
        GGLWEBackendMut::from_inner(GGLWE {
            base2k: self.base2k(),
            dsize: self.dsize(),
            data: self.data.to_backend_mut(),
        })
    }
}

impl<'b, BE: Backend + 'b> GGLWEToBackendRef<BE> for &mut GGLWE<BE::BufMut<'b>> {
    fn to_backend_ref(&self) -> GGLWEBackendRef<'_, BE> {
        GGLWEBackendRef::from_inner(GGLWE {
            base2k: self.base2k(),
            dsize: self.dsize(),
            data: poulpy_hal::layouts::mat_znx_backend_ref_from_mut::<BE>(&self.data),
        })
    }
}

impl<'b, BE: Backend + 'b> GGLWEToBackendMut<BE> for &mut GGLWE<BE::BufMut<'b>> {
    fn to_backend_mut(&mut self) -> GGLWEBackendMut<'_, BE> {
        GGLWEBackendMut::from_inner(GGLWE {
            base2k: self.base2k(),
            dsize: self.dsize(),
            data: poulpy_hal::layouts::mat_znx_backend_mut_from_mut::<BE>(&mut self.data),
        })
    }
}

pub trait GGLWEToBackendRef<BE: Backend> {
    fn to_backend_ref(&self) -> GGLWEBackendRef<'_, BE>;
}

impl<BE: Backend, D: Data> GGLWEToBackendRef<BE> for GGLWE<D>
where
    MatZnx<D>: MatZnxToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GGLWEBackendRef<'_, BE> {
        GGLWEBackendRef::from_inner(GGLWE {
            base2k: self.base2k(),
            dsize: self.dsize(),
            data: self.data.to_backend_ref(),
        })
    }
}

impl<D: HostDataMut> ReaderFrom for GGLWE<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        self.dsize = Dsize(reader.read_u32::<LittleEndian>()?);
        self.data.read_from(reader)
    }
}

impl<D: HostDataRef> WriterTo for GGLWE<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.base2k.0)?;
        writer.write_u32::<LittleEndian>(self.dsize.0)?;
        self.data.write_to(writer)
    }
}
