use poulpy_hal::{
    layouts::{
        Backend, Data, FillUniform, HostDataMut, HostDataRef, Module, ReaderFrom, ToOwnedDeep, TransferFrom, VecZnx,
        VecZnxToBackendMut, VecZnxToBackendRef, WriterTo,
    },
    source::Source,
};

use crate::api::ModuleTransfer;
use crate::layouts::{Base2K, Degree, LWEInfos, Rank, SetLWEInfos, TorusPrecision};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fmt;

/// Trait providing the parameter accessors for a GLWE (Generalised LWE) ciphertext.
///
/// A GLWE ciphertext is a polynomial-ring LWE ciphertext consisting of
/// a body polynomial and `rank` mask polynomials, all defined over `Z[X]/(X^n + 1)`.
/// Extends [`LWEInfos`] with the GLWE rank.
pub trait GLWEInfos
where
    Self: LWEInfos,
{
    /// Returns the GLWE rank (number of mask polynomials).
    fn rank(&self) -> Rank;
    /// Returns a plain-data [`GLWELayout`] snapshot of the current parameters.
    fn glwe_layout(&self) -> GLWELayout {
        GLWELayout {
            n: self.n(),
            base2k: self.base2k(),
            k: self.max_k(),
            rank: self.rank(),
        }
    }
}

/// Plain-data snapshot of the parameters that describe a [`GLWE`] ciphertext.
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GLWELayout {
    /// Ring degree.
    pub n: Degree,
    /// Base-2-log of the limb width.
    pub base2k: Base2K,
    /// Torus precision.
    pub k: TorusPrecision,
    /// Number of mask polynomials.
    pub rank: Rank,
}

impl LWEInfos for GLWELayout {
    fn n(&self) -> Degree {
        self.n
    }

    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn size(&self) -> usize {
        self.k.as_usize().div_ceil(self.base2k.as_usize())
    }
}

impl GLWEInfos for GLWELayout {
    fn rank(&self) -> Rank {
        self.rank
    }
}

/// A GLWE (Generalised LWE) ciphertext over the polynomial ring `Z[X]/(X^n + 1)`.
///
/// Wraps a [`VecZnx`] with `rank + 1` columns: the first column is the body
/// polynomial, and the remaining `rank` columns are the mask polynomials.
///
/// `D: Data` is the storage backend (e.g. `Vec<u8>`, `&[u8]`, `&mut [u8]`).
#[derive(PartialEq, Eq, Clone)]
pub struct GLWE<D: Data> {
    pub(crate) data: VecZnx<D>,
    pub(crate) base2k: Base2K,
}

pub type GLWEBackendRef<'a, BE> = GLWE<<BE as Backend>::BufRef<'a>>;
pub type GLWEBackendMut<'a, BE> = GLWE<<BE as Backend>::BufMut<'a>>;

impl<D: Data> SetLWEInfos for GLWE<D> {
    fn set_base2k(&mut self, base2k: Base2K) {
        self.base2k = base2k
    }
}

impl<D: Data> SetLWEInfos for &mut GLWE<D> {
    fn set_base2k(&mut self, base2k: Base2K) {
        self.base2k = base2k
    }
}

impl<D: Data> GLWE<D> {
    /// Returns a shared reference to the underlying [`VecZnx`].
    pub fn data(&self) -> &VecZnx<D> {
        &self.data
    }

    /// Returns the allocated limb capacity, which can exceed the active `size()`
    /// after a precision-consuming rescale.
    pub fn max_size(&self) -> usize {
        self.data.max_size()
    }
}

impl<D: Data> GLWE<D> {
    /// Returns a mutable reference to the underlying [`VecZnx`].
    pub fn data_mut(&mut self) -> &mut VecZnx<D> {
        &mut self.data
    }
}

impl<D: Data> LWEInfos for GLWE<D> {
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

impl<D: Data> LWEInfos for &GLWE<D> {
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

impl<D: Data> LWEInfos for &mut GLWE<D> {
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

impl<D: Data> GLWEInfos for GLWE<D> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols() as u32 - 1)
    }
}

impl<D: Data> GLWEInfos for &GLWE<D> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols() as u32 - 1)
    }
}

impl<D: Data> GLWEInfos for &mut GLWE<D> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols() as u32 - 1)
    }
}

impl<D: HostDataRef> ToOwnedDeep for GLWE<D> {
    type Owned = GLWE<Vec<u8>>;
    fn to_owned_deep(&self) -> Self::Owned {
        GLWE {
            data: self.data.to_owned_deep(),
            base2k: self.base2k,
        }
    }
}

impl<D: HostDataRef> GLWE<D> {
    /// Copies this ciphertext's backing bytes into an owned buffer of
    /// backend `To`, routing via host bytes.
    ///
    /// `BE` is the backend that produced `self`; `To` is the destination.
    pub fn to_backend<BE, To>(&self, dst: &Module<To>) -> GLWE<To::OwnedBuf>
    where
        BE: Backend<OwnedBuf = D>,
        To: Backend,
        To: TransferFrom<BE>,
    {
        dst.upload_glwe(self)
    }
}

impl<D: Data> GLWE<D> {
    /// Rebuilds this backend-owned ciphertext as a host-owned [`GLWE<Vec<u8>>`].
    pub fn to_host_owned<BE>(&self) -> GLWE<Vec<u8>>
    where
        BE: Backend<OwnedBuf = D>,
    {
        GLWE {
            data: self.data.to_host_owned::<BE>(),
            base2k: self.base2k,
        }
    }

    /// Formats this backend-owned ciphertext through the existing host [`fmt::Display`] implementation.
    pub fn display_host<BE>(&self) -> String
    where
        BE: Backend<OwnedBuf = D>,
    {
        self.to_host_owned::<BE>().to_string()
    }
}

impl<D: Data> GLWE<D> {
    /// Zero-cost rename when both backends share the same `OwnedBuf`.
    pub fn reinterpret<To>(self) -> GLWE<To::OwnedBuf>
    where
        To: Backend<OwnedBuf = D>,
    {
        let shape = self.data.shape();
        let data = self.data.data;
        GLWE {
            data: VecZnx::from_data_with_max_size(data, shape.n(), shape.cols(), shape.size(), shape.max_size()),
            base2k: self.base2k,
        }
    }
}

impl<D: HostDataRef> fmt::Debug for GLWE<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataRef> fmt::Display for GLWE<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GLWE: base2k={} k={}: {}", self.base2k().0, self.max_k().0, self.data)
    }
}

impl<D: HostDataMut> FillUniform for GLWE<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

impl GLWE<Vec<u8>> {
    /// Allocates a new [`GLWE`] with the given parameters.
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        Self::alloc(infos.n(), infos.base2k(), infos.max_k(), infos.rank())
    }

    /// Allocates a new [`GLWE`] with the given parameters.
    ///
    /// * `n` -- ring degree.
    /// * `base2k` -- base-2-log of the limb width.
    /// * `k` -- torus precision.
    /// * `rank` -- number of mask polynomials.
    pub(crate) fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> Self {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        GLWE {
            data: VecZnx::from_data(
                poulpy_hal::layouts::HostBytesBackend::alloc_bytes(VecZnx::<Vec<u8>>::bytes_of(
                    n.into(),
                    (rank + 1).into(),
                    size,
                )),
                n.into(),
                (rank + 1).into(),
                size,
            ),
            base2k,
        }
    }

    /// Returns the byte count required for a [`GLWE`] with the given parameters.
    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        Self::bytes_of(infos.n(), infos.base2k(), infos.max_k(), infos.rank())
    }

    /// Returns the byte count required for a [`GLWE`] with the given parameters.
    ///
    /// * `n` -- ring degree.
    /// * `base2k` -- base-2-log of the limb width.
    /// * `k` -- torus precision.
    /// * `rank` -- number of mask polynomials.
    pub fn bytes_of(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> usize {
        VecZnx::bytes_of(n.into(), (rank + 1).into(), k.0.div_ceil(base2k.0) as usize)
    }

    /// Reallocates the backing buffer so capacity matches `size` limb count.
    pub fn reallocate_limbs(&mut self, size: usize) {
        self.data.reallocate_limbs(size);
    }
}

impl<D: HostDataMut> ReaderFrom for GLWE<D> {
    /// Deserialises a [`GLWE`] in little-endian binary format.
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        self.data.read_from(reader)?;
        Ok(())
    }
}

impl<D: HostDataRef> WriterTo for GLWE<D> {
    /// Serialises the [`GLWE`] in little-endian binary format.
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.base2k.0)?;
        self.data.write_to(writer)
    }
}

pub trait GLWEToBackendRef<BE: Backend>: Sized {
    fn to_backend_ref(&self) -> GLWEBackendRef<'_, BE>;
}

impl<BE: Backend, D: Data> GLWEToBackendRef<BE> for GLWE<D>
where
    VecZnx<D>: VecZnxToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GLWEBackendRef<'_, BE> {
        GLWE {
            base2k: self.base2k,
            data: self.data.to_backend_ref(),
        }
    }
}

pub fn glwe_backend_ref_from_ref<'a, 'b, BE: Backend>(glwe: &'a GLWE<BE::BufRef<'b>>) -> GLWEBackendRef<'a, BE> {
    GLWE {
        base2k: glwe.base2k,
        data: poulpy_hal::layouts::vec_znx_backend_ref_from_ref::<BE>(&glwe.data),
    }
}

impl<'b, BE: Backend + 'b> GLWEToBackendRef<BE> for &GLWE<BE::BufRef<'b>> {
    fn to_backend_ref(&self) -> GLWEBackendRef<'_, BE> {
        glwe_backend_ref_from_ref::<BE>(self)
    }
}

pub fn glwe_backend_ref_from_mut<'a, 'b, BE: Backend>(glwe: &'a GLWE<BE::BufMut<'b>>) -> GLWEBackendRef<'a, BE> {
    GLWE {
        base2k: glwe.base2k,
        data: poulpy_hal::layouts::vec_znx_backend_ref_from_mut::<BE>(&glwe.data),
    }
}

pub trait GLWEToBackendMut<BE: Backend>: GLWEToBackendRef<BE> {
    fn to_backend_mut(&mut self) -> GLWEBackendMut<'_, BE>;
}

impl<BE: Backend, D: Data> GLWEToBackendMut<BE> for GLWE<D>
where
    VecZnx<D>: VecZnxToBackendRef<BE> + VecZnxToBackendMut<BE>,
{
    fn to_backend_mut(&mut self) -> GLWEBackendMut<'_, BE> {
        GLWE {
            base2k: self.base2k,
            data: self.data.to_backend_mut(),
        }
    }
}

impl<'b, BE: Backend + 'b> GLWEToBackendRef<BE> for &mut GLWE<BE::BufMut<'b>> {
    fn to_backend_ref(&self) -> GLWEBackendRef<'_, BE> {
        glwe_backend_ref_from_mut::<BE>(self)
    }
}

impl<'b, BE: Backend + 'b> GLWEToBackendMut<BE> for &mut GLWE<BE::BufMut<'b>> {
    fn to_backend_mut(&mut self) -> GLWEBackendMut<'_, BE> {
        glwe_backend_mut_from_mut::<BE>(self)
    }
}

pub fn glwe_backend_mut_from_mut<'a, 'b, BE: Backend>(glwe: &'a mut GLWE<BE::BufMut<'b>>) -> GLWEBackendMut<'a, BE> {
    GLWE {
        base2k: glwe.base2k,
        data: poulpy_hal::layouts::vec_znx_backend_mut_from_mut::<BE>(&mut glwe.data),
    }
}
