use std::fmt;

use poulpy_hal::{
    layouts::{
        Backend, Data, FillUniform, HostDataMut, HostDataRef, Module, ReaderFrom, TransferFrom, VecZnx, VecZnxToBackendMut,
        VecZnxToBackendRef, WriterTo,
    },
    source::Source,
};

use crate::api::ModuleTransfer;
use crate::layouts::{Base2K, Degree, TorusPrecision};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

/// Trait providing the parameter accessors for an LWE ciphertext.
///
/// An LWE ciphertext is a scalar (non-polynomial) ciphertext consisting of
/// a body `b` and a mask `(a_1, ..., a_n)`.
pub trait LWEInfos {
    /// Returns the LWE dimension, i.e. the number of mask elements (= GLWE ring degree N).
    fn n(&self) -> Degree;
    /// Returns `log2(n)`.
    fn log_n(&self) -> usize {
        self.n().log2()
    }
    /// Returns the maximum torus precision representable with the current limb decomposition.
    fn max_k(&self) -> TorusPrecision {
        TorusPrecision(self.size() as u32 * self.base2k().as_u32())
    }

    /// Returns the base-2-log of the limb width used for the RNS/CRT representation.
    fn base2k(&self) -> Base2K;
    /// Returns the number of limbs, i.e. `ceil(k / base2k)`.
    fn size(&self) -> usize;

    /// Returns a plain-data [`LWELayout`] snapshot of the current parameters.
    fn lwe_layout(&self) -> LWELayout {
        LWELayout {
            n: self.n(),
            k: self.max_k(),
            base2k: self.base2k(),
        }
    }
}

/// Trait for mutating LWE parameters in place.
pub trait SetLWEInfos {
    /// Sets the limb width `base2k`.
    fn set_base2k(&mut self, base2k: Base2K);
}

/// Plain-data snapshot of the parameters that describe an [`LWE`] ciphertext.
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct LWELayout {
    /// Ring degree (LWE dimension).
    pub n: Degree,
    /// Torus precision.
    pub k: TorusPrecision,
    /// Base-2-log of the limb width.
    pub base2k: Base2K,
}

impl LWEInfos for LWELayout {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn n(&self) -> Degree {
        self.n
    }

    fn size(&self) -> usize {
        self.k.as_usize().div_ceil(self.base2k.into())
    }
}
/// A scalar (non-polynomial) LWE ciphertext.
///
/// Stored as a single-column [`VecZnx`] of dimension `n + 1` where
/// the body `b` and the mask `(a_1, ..., a_n)` are packed together.
/// When `rank = 0` the mask column is embedded in the single [`VecZnx`] column.
///
/// `D: Data` is the storage backend (e.g. `Vec<u8>`, `&[u8]`, `&mut [u8]`).
#[derive(PartialEq, Eq, Clone)]
pub struct LWE<D: Data> {
    pub(crate) data: VecZnx<D>,
    pub(crate) base2k: Base2K,
}

pub type LWEBackendRef<'a, BE> = LWE<<BE as Backend>::BufRef<'a>>;
pub type LWEBackendMut<'a, BE> = LWE<<BE as Backend>::BufMut<'a>>;

impl<D: Data> LWEInfos for LWE<D> {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32 - 1)
    }

    fn size(&self) -> usize {
        self.data.size()
    }
}

impl<D: Data> SetLWEInfos for LWE<D> {
    fn set_base2k(&mut self, base2k: Base2K) {
        self.base2k = base2k
    }
}

impl<D: HostDataRef> LWE<D> {
    /// Returns a shared reference to the underlying [`VecZnx`].
    pub fn data(&self) -> &VecZnx<D> {
        &self.data
    }
}

impl<D: HostDataMut> LWE<D> {
    /// Returns a mutable reference to the underlying [`VecZnx`].
    pub fn data_mut(&mut self) -> &mut VecZnx<D> {
        &mut self.data
    }
}

impl<D: HostDataRef> LWE<D> {
    /// Copies this ciphertext's backing bytes into an owned buffer of
    /// backend `To`, routing via host bytes.
    pub fn to_backend<BE, To>(&self, dst: &Module<To>) -> LWE<To::OwnedBuf>
    where
        BE: Backend<OwnedBuf = D>,
        To: Backend,
        To: TransferFrom<BE>,
    {
        dst.upload_lwe(self)
    }
}

impl<D: Data> LWE<D> {
    /// Zero-cost rename when both backends share the same `OwnedBuf`.
    pub fn reinterpret<To>(self) -> LWE<To::OwnedBuf>
    where
        To: Backend<OwnedBuf = D>,
    {
        let shape = self.data.shape();
        let data = self.data.data;
        LWE {
            data: VecZnx::from_data_with_max_size(data, shape.n(), shape.cols(), shape.size(), shape.max_size()),
            base2k: self.base2k,
        }
    }
}

impl<D: HostDataRef> fmt::Debug for LWE<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataRef> fmt::Display for LWE<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LWE: base2k={} k={}: {}", self.base2k().0, self.max_k().0, self.data)
    }
}

impl<D: HostDataMut> FillUniform for LWE<D>
where
    VecZnx<D>: FillUniform,
{
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

#[expect(
    dead_code,
    reason = "host-owned constructors are kept for serialization and host-only staging"
)]
impl LWE<Vec<u8>> {
    /// Allocates a new [`LWE`] with the given parameters.
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: LWEInfos,
    {
        Self::alloc(infos.n(), infos.base2k(), infos.max_k())
    }

    /// Allocates a new [`LWE`] with the given parameters.
    ///
    /// * `n` -- ring degree (LWE dimension).
    /// * `base2k` -- base-2-log of the limb width.
    /// * `k` -- torus precision.
    pub(crate) fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision) -> Self {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        LWE {
            data: VecZnx::from_data(
                poulpy_hal::layouts::HostBytesBackend::alloc_bytes(VecZnx::<Vec<u8>>::bytes_of((n + 1).into(), 1, size)),
                (n + 1).into(),
                1,
                size,
            ),
            base2k,
        }
    }

    /// Returns the byte count required for an [`LWE`] with the given parameters.
    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: LWEInfos,
    {
        Self::bytes_of(infos.n(), infos.base2k(), infos.max_k())
    }

    /// Returns the byte count required for an [`LWE`] with the given parameters.
    ///
    /// * `n` -- ring degree (LWE dimension).
    /// * `base2k` -- base-2-log of the limb width.
    /// * `k` -- torus precision.
    pub fn bytes_of(n: Degree, base2k: Base2K, k: TorusPrecision) -> usize {
        VecZnx::bytes_of((n + 1).into(), 1, k.0.div_ceil(base2k.0) as usize)
    }
}

pub trait LWEToBackendRef<BE: Backend> {
    fn to_backend_ref(&self) -> LWEBackendRef<'_, BE>;
}

impl<BE: Backend, D: Data> LWEToBackendRef<BE> for LWE<D>
where
    VecZnx<D>: VecZnxToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> LWEBackendRef<'_, BE> {
        LWE {
            base2k: self.base2k,
            data: self.data.to_backend_ref(),
        }
    }
}

pub trait LWEToBackendMut<BE: Backend>: LWEToBackendRef<BE> {
    fn to_backend_mut(&mut self) -> LWEBackendMut<'_, BE>;
}

impl<BE: Backend, D: Data> LWEToBackendMut<BE> for LWE<D>
where
    VecZnx<D>: VecZnxToBackendRef<BE> + VecZnxToBackendMut<BE>,
{
    fn to_backend_mut(&mut self) -> LWEBackendMut<'_, BE> {
        LWE {
            base2k: self.base2k,
            data: self.data.to_backend_mut(),
        }
    }
}

impl<'b, BE: Backend + 'b> LWEToBackendRef<BE> for &mut LWE<BE::BufMut<'b>> {
    fn to_backend_ref(&self) -> LWEBackendRef<'_, BE> {
        LWE {
            base2k: self.base2k,
            data: poulpy_hal::layouts::vec_znx_backend_ref_from_mut::<BE>(&self.data),
        }
    }
}

impl<'b, BE: Backend + 'b> LWEToBackendMut<BE> for &mut LWE<BE::BufMut<'b>> {
    fn to_backend_mut(&mut self) -> LWEBackendMut<'_, BE> {
        LWE {
            base2k: self.base2k,
            data: poulpy_hal::layouts::vec_znx_backend_mut_from_mut::<BE>(&mut self.data),
        }
    }
}

impl<D: HostDataMut> ReaderFrom for LWE<D> {
    /// Deserialises an [`LWE`] in little-endian binary format.
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        self.data.read_from(reader)
    }
}

impl<D: HostDataRef> WriterTo for LWE<D> {
    /// Serialises the [`LWE`] in little-endian binary format.
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.base2k.into())?;
        self.data.write_to(writer)
    }
}
