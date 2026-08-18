use poulpy_hal::{
    layouts::{
        Backend, Data, FillUniform, HostDataMut, HostDataRef, MatZnx, MatZnxAtBackendMut, MatZnxAtBackendRef, MatZnxToBackendMut,
        MatZnxToBackendRef, ReaderFrom, WriterTo,
    },
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGSWAtViewRef, GLWE, GLWEInfos, GLWEViewMut, GLWEViewRef, LWEInfos, Rank, TorusPrecision,
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use poulpy_hal::layouts::ZnxWord;

use std::{
    fmt,
    ops::{Deref, DerefMut},
};

pub trait GGLWEInfos
where
    Self: GLWEInfos,
{
    /// Auxiliary guard precision (in bits) stored below the gadget region and
    /// used for noise management during key operations. Free value, need not be
    /// a multiple of `base2k`. In practice `dsize*base2k + logN`.
    fn k_aux(&self) -> TorusPrecision;
    /// Number of gadget-decomposition rows.
    fn dnum(&self) -> Dnum;
    fn dsize(&self) -> Dsize;
    fn rank_in(&self) -> Rank;
    fn rank_out(&self) -> Rank;
    fn gglwe_layout(&self) -> GGLWELayout {
        GGLWELayout {
            n: self.n(),
            base2k: self.base2k(),
            dnum: self.dnum(),
            k_aux: self.k_aux(),
            rank_in: self.rank_in(),
            rank_out: self.rank_out(),
            dsize: self.dsize(),
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
    pub dnum: Dnum,
    pub k_aux: TorusPrecision,
    pub rank_in: Rank,
    pub rank_out: Rank,
    pub dsize: Dsize,
}

impl LWEInfos for GGLWELayout {
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

impl GLWEInfos for GGLWELayout {
    fn rank(&self) -> Rank {
        self.rank_out
    }
}

impl GGLWEInfos for GGLWELayout {
    fn k_aux(&self) -> TorusPrecision {
        self.k_aux
    }

    fn dnum(&self) -> Dnum {
        self.dnum
    }

    fn rank_in(&self) -> Rank {
        self.rank_in
    }

    fn dsize(&self) -> Dsize {
        self.dsize
    }

    fn rank_out(&self) -> Rank {
        self.rank_out
    }
}

#[derive(PartialEq, Eq, Clone)]
pub struct GGLWE<D: Data, W: ZnxWord> {
    pub(crate) data: MatZnx<D, W>,
    pub(crate) k_aux: TorusPrecision,
    pub(crate) base2k: Base2K,
    pub(crate) dsize: Dsize,
}

pub struct GGLWEBackendRef<'a, BE: Backend + 'a> {
    inner: GGLWE<BE::BufRef<'a>, BE::ZnxWord>,
}

impl<'a, BE: Backend + 'a> GGLWEBackendRef<'a, BE> {
    pub fn from_inner(inner: GGLWE<BE::BufRef<'a>, BE::ZnxWord>) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> GGLWE<BE::BufRef<'a>, BE::ZnxWord> {
        self.inner
    }

    pub fn at_view(&self, row: usize, col: usize) -> GLWEViewRef<'_, BE> {
        GLWEViewRef::from_inner(gglwe_at_backend_ref_from_ref::<BE>(&self.inner, row, col))
    }
}

impl<'a, BE: Backend + 'a> Deref for GGLWEBackendRef<'a, BE> {
    type Target = GGLWE<BE::BufRef<'a>, BE::ZnxWord>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub struct GGLWEBackendMut<'a, BE: Backend + 'a> {
    inner: GGLWE<BE::BufMut<'a>, BE::ZnxWord>,
}

impl<'a, BE: Backend + 'a> GGLWEBackendMut<'a, BE> {
    pub fn from_inner(inner: GGLWE<BE::BufMut<'a>, BE::ZnxWord>) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> GGLWE<BE::BufMut<'a>, BE::ZnxWord> {
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
    type Target = GGLWE<BE::BufMut<'a>, BE::ZnxWord>;

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
            k_aux: self.inner.k_aux,
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
            k_aux: self.inner.k_aux,
            dsize: self.inner.dsize,
            data: poulpy_hal::layouts::mat_znx_backend_ref_from_mut::<BE>(&self.inner.data),
        })
    }
}

impl<'a, BE: Backend + 'a> GGLWEToBackendMut<BE> for GGLWEBackendMut<'a, BE> {
    fn to_backend_mut(&mut self) -> GGLWEBackendMut<'_, BE> {
        GGLWEBackendMut::from_inner(GGLWE {
            base2k: self.inner.base2k,
            k_aux: self.inner.k_aux,
            dsize: self.inner.dsize,
            data: poulpy_hal::layouts::mat_znx_backend_mut_from_mut::<BE>(&mut self.inner.data),
        })
    }
}

impl<D: Data, W: ZnxWord> LWEInfos for GGLWE<D, W> {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn max_size(&self) -> usize {
        self.data.size()
    }

    fn k(&self) -> TorusPrecision {
        crate::layouts::key_k(self.base2k, self.dnum(), self.dsize, self.k_aux)
    }
}

impl<D: Data, W: ZnxWord> GLWEInfos for GGLWE<D, W> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, W: ZnxWord> GGLWEInfos for GGLWE<D, W> {
    fn k_aux(&self) -> TorusPrecision {
        self.k_aux
    }

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

impl<D: Data, W: ZnxWord> GGLWE<D, W> {
    /// Returns a shared reference to the underlying [`MatZnx`].
    pub fn data(&self) -> &MatZnx<D, W> {
        &self.data
    }
}

/// Backend-native shared view of one GLWE row.
pub trait GGLWEAtBackendRef<BE: Backend> {
    fn at_backend(&self, row: usize, col: usize) -> GLWE<BE::BufRef<'_>, BE::ZnxWord>;
}

impl<BE: Backend> GGLWEAtBackendRef<BE> for GGLWE<BE::OwnedBuf, BE::ZnxWord> {
    fn at_backend(&self, row: usize, col: usize) -> GLWE<BE::BufRef<'_>, BE::ZnxWord> {
        let data = <MatZnx<BE::OwnedBuf, BE::ZnxWord> as MatZnxAtBackendRef<BE>>::at_backend(&self.data, row, col);
        GLWE {
            base2k: self.base2k,
            k: self.k(),
            data,
        }
    }
}

pub(crate) fn gglwe_at_backend_ref_from_ref<'a, 'b, BE: Backend>(
    gglwe: &'a GGLWE<BE::BufRef<'b>, BE::ZnxWord>,
    row: usize,
    col: usize,
) -> GLWE<BE::BufRef<'a>, BE::ZnxWord> {
    let data = poulpy_hal::layouts::mat_znx_at_backend_ref_from_ref::<BE>(&gglwe.data, row, col);
    GLWE {
        base2k: gglwe.base2k,
        k: gglwe.k(),
        data,
    }
}

pub trait GGLWEAtViewRef<BE: Backend> {
    fn at_view(&self, row: usize, col: usize) -> GLWEViewRef<'_, BE>;
}

impl<BE: Backend> GGLWEAtViewRef<BE> for GGLWE<BE::OwnedBuf, BE::ZnxWord> {
    fn at_view(&self, row: usize, col: usize) -> GLWEViewRef<'_, BE> {
        GLWEViewRef::from_inner(<GGLWE<BE::OwnedBuf, BE::ZnxWord> as GGLWEAtBackendRef<BE>>::at_backend(
            self, row, col,
        ))
    }
}

pub(crate) fn gglwe_at_backend_ref_from_mut<'a, 'b, BE: Backend>(
    gglwe: &'a GGLWE<BE::BufMut<'b>, BE::ZnxWord>,
    row: usize,
    col: usize,
) -> GLWE<BE::BufRef<'a>, BE::ZnxWord> {
    let data = poulpy_hal::layouts::mat_znx_at_backend_ref_from_mut::<BE>(&gglwe.data, row, col);
    GLWE {
        base2k: gglwe.base2k,
        k: gglwe.k(),
        data,
    }
}

impl<D: Data, W: ZnxWord> GGLWE<D, W> {
    /// Returns a mutable reference to the underlying [`MatZnx`].
    pub fn data_mut(&mut self) -> &mut MatZnx<D, W> {
        &mut self.data
    }
}

/// Backend-native mutable view of one GLWE row.
pub trait GGLWEAtBackendMut<BE: Backend> {
    fn at_backend_mut(&mut self, row: usize, col: usize) -> GLWE<BE::BufMut<'_>, BE::ZnxWord>;
}

impl<BE: Backend> GGLWEAtBackendMut<BE> for GGLWE<BE::OwnedBuf, BE::ZnxWord> {
    fn at_backend_mut(&mut self, row: usize, col: usize) -> GLWE<BE::BufMut<'_>, BE::ZnxWord> {
        let base2k = self.base2k;
        let k = self.k();
        let data = <MatZnx<BE::OwnedBuf, BE::ZnxWord> as MatZnxAtBackendMut<BE>>::at_backend_mut(&mut self.data, row, col);
        GLWE { base2k, k, data }
    }
}

pub(crate) fn gglwe_at_backend_mut_from_mut<'a, 'b, BE: Backend>(
    gglwe: &'a mut GGLWE<BE::BufMut<'b>, BE::ZnxWord>,
    row: usize,
    col: usize,
) -> GLWE<BE::BufMut<'a>, BE::ZnxWord> {
    let base2k = gglwe.base2k;
    let k = gglwe.k();
    let data = poulpy_hal::layouts::mat_znx_at_backend_mut_from_mut::<BE>(&mut gglwe.data, row, col);
    GLWE { base2k, k, data }
}

pub trait GGLWEAtViewMut<BE: Backend> {
    fn at_view_mut(&mut self, row: usize, col: usize) -> GLWEViewMut<'_, BE>;
}

impl<BE: Backend> GGLWEAtViewMut<BE> for GGLWE<BE::OwnedBuf, BE::ZnxWord> {
    fn at_view_mut(&mut self, row: usize, col: usize) -> GLWEViewMut<'_, BE> {
        GLWEViewMut::from_inner(<GGLWE<BE::OwnedBuf, BE::ZnxWord> as GGLWEAtBackendMut<BE>>::at_backend_mut(
            self, row, col,
        ))
    }
}

impl<'b, BE: Backend + 'b> GGLWEAtViewMut<BE> for &mut GGLWE<BE::BufMut<'b>, BE::ZnxWord> {
    fn at_view_mut(&mut self, row: usize, col: usize) -> GLWEViewMut<'_, BE> {
        GLWEViewMut::from_inner(gglwe_at_backend_mut_from_mut::<BE>(*self, row, col))
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Debug for GGLWE<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataMut, W: ZnxWord> FillUniform for GGLWE<D, W> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Display for GGLWE<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(GGLWE: k={} base2k={} dsize={}) {}",
            self.k().0,
            self.base2k().0,
            self.dsize().0,
            self.data
        )
    }
}

impl<D: HostDataRef, W: ZnxWord> GGLWE<D, W> {
    pub fn at(&self, row: usize, col: usize) -> GLWE<&[u8], W> {
        let data = self.data.at(row, col);
        GLWE {
            base2k: self.base2k,
            k: self.k(),
            data,
        }
    }
}

impl<D: HostDataMut, W: ZnxWord> GGLWE<D, W> {
    pub fn at_mut(&mut self, row: usize, col: usize) -> GLWE<&mut [u8], W> {
        let base2k = self.base2k;
        let k = self.k();
        let data = self.data.at_mut(row, col);
        GLWE { base2k, k, data }
    }
}

impl<D: Data, W: ZnxWord> GGLWE<D, W> {
    /// Zero-cost rename when both backends share the same `OwnedBuf`.
    pub fn reinterpret<To>(self) -> GGLWE<To::OwnedBuf, To::ZnxWord>
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
        GGLWE {
            data: MatZnx::from_data(self.data.into_data(), n, rows, cols_in, cols_out, size),
            base2k: self.base2k,
            dsize: self.dsize,
            k_aux: self.k_aux,
        }
    }
}

#[expect(
    dead_code,
    reason = "host-owned constructors are kept for serialization and host-only staging"
)]
impl<W: ZnxWord> GGLWE<Vec<u8>, W> {
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        Self::alloc(
            infos.n(),
            infos.base2k(),
            infos.dnum(),
            infos.dsize(),
            infos.k_aux(),
            infos.rank_in(),
            infos.rank_out(),
        )
    }

    pub(crate) fn alloc(
        n: Degree,
        base2k: Base2K,
        dnum: Dnum,
        dsize: Dsize,
        k_aux: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
    ) -> Self {
        let size: usize = crate::layouts::key_size(base2k, dnum, dsize, k_aux);

        GGLWE {
            data: MatZnx::from_data(
                poulpy_hal::layouts::HostBytesBackend::alloc_bytes(MatZnx::<Vec<u8>, W>::bytes_of(
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
            k_aux,
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        Self::bytes_of(
            infos.n(),
            infos.base2k(),
            infos.dnum(),
            infos.dsize(),
            infos.k_aux(),
            infos.rank_in(),
            infos.rank_out(),
        )
    }

    pub fn bytes_of(
        n: Degree,
        base2k: Base2K,
        dnum: Dnum,
        dsize: Dsize,
        k_aux: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
    ) -> usize {
        let size: usize = crate::layouts::key_size(base2k, dnum, dsize, k_aux);

        MatZnx::<Vec<u8>, W>::bytes_of(n.into(), dnum.into(), rank_in.into(), (rank_out + 1).into(), size)
    }
}

pub trait GGLWEToBackendMut<BE: Backend>: GGLWEToBackendRef<BE> {
    fn to_backend_mut(&mut self) -> GGLWEBackendMut<'_, BE>;
}

impl<BE: Backend, D: Data> GGLWEToBackendMut<BE> for GGLWE<D, BE::ZnxWord>
where
    MatZnx<D, BE::ZnxWord>: MatZnxToBackendRef<BE> + MatZnxToBackendMut<BE>,
{
    fn to_backend_mut(&mut self) -> GGLWEBackendMut<'_, BE> {
        GGLWEBackendMut::from_inner(GGLWE {
            base2k: self.base2k(),
            dsize: self.dsize(),
            k_aux: self.k_aux(),
            data: self.data.to_backend_mut(),
        })
    }
}

impl<'b, BE: Backend + 'b> GGLWEToBackendRef<BE> for &mut GGLWE<BE::BufMut<'b>, BE::ZnxWord> {
    fn to_backend_ref(&self) -> GGLWEBackendRef<'_, BE> {
        GGLWEBackendRef::from_inner(GGLWE {
            base2k: self.base2k(),
            dsize: self.dsize(),
            k_aux: self.k_aux(),
            data: poulpy_hal::layouts::mat_znx_backend_ref_from_mut::<BE>(&self.data),
        })
    }
}

impl<'b, BE: Backend + 'b> GGLWEToBackendMut<BE> for &mut GGLWE<BE::BufMut<'b>, BE::ZnxWord> {
    fn to_backend_mut(&mut self) -> GGLWEBackendMut<'_, BE> {
        GGLWEBackendMut::from_inner(GGLWE {
            base2k: self.base2k(),
            dsize: self.dsize(),
            k_aux: self.k_aux(),
            data: poulpy_hal::layouts::mat_znx_backend_mut_from_mut::<BE>(&mut self.data),
        })
    }
}

pub trait GGLWEToBackendRef<BE: Backend> {
    fn to_backend_ref(&self) -> GGLWEBackendRef<'_, BE>;
}

impl<BE: Backend, D: Data> GGLWEToBackendRef<BE> for GGLWE<D, BE::ZnxWord>
where
    MatZnx<D, BE::ZnxWord>: MatZnxToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GGLWEBackendRef<'_, BE> {
        GGLWEBackendRef::from_inner(GGLWE {
            base2k: self.base2k(),
            dsize: self.dsize(),
            k_aux: self.k_aux(),
            data: self.data.to_backend_ref(),
        })
    }
}

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for GGLWE<D, W> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        self.dsize = Dsize(reader.read_u32::<LittleEndian>()?);
        self.k_aux = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        self.data.read_from(reader)
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for GGLWE<D, W> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.base2k.0)?;
        writer.write_u32::<LittleEndian>(self.dsize.0)?;
        writer.write_u32::<LittleEndian>(self.k_aux.0)?;
        self.data.write_to(writer)
    }
}
