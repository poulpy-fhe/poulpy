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
    /// Returns the maximum Torus precision representable by the object.
    fn max_k(&self) -> TorusPrecision {
        TorusPrecision(self.max_size() as u32 * self.base2k().as_u32())
    }

    /// Returns the Torus precision used by the object.
    fn k(&self) -> TorusPrecision;

    /// Returns the limb width generic core operations should process.
    fn size(&self) -> usize {
        self.k().div_ceil(self.base2k()) as usize
    }

    /// Returns the base-2-log of the limb width used for the RNS/CRT representation.
    fn base2k(&self) -> Base2K;

    /// Returns the maximum limb width this object can expose through [`Self::size`].
    fn max_size(&self) -> usize;

    /// Returns a plain-data [`LWELayout`] snapshot of the current parameters.
    fn lwe_layout(&self) -> LWELayout {
        LWELayout {
            n: self.n(),
            k: self.k(),
            base2k: self.base2k(),
        }
    }
}

impl<T: LWEInfos + ?Sized> LWEInfos for &T {
    fn n(&self) -> Degree {
        (**self).n()
    }

    fn k(&self) -> TorusPrecision {
        (**self).k()
    }

    fn base2k(&self) -> Base2K {
        (**self).base2k()
    }

    fn max_size(&self) -> usize {
        (**self).max_size()
    }
}

impl<T: LWEInfos + ?Sized> LWEInfos for &mut T {
    fn n(&self) -> Degree {
        (**self).n()
    }

    fn k(&self) -> TorusPrecision {
        (**self).k()
    }

    fn base2k(&self) -> Base2K {
        (**self).base2k()
    }

    fn max_size(&self) -> usize {
        (**self).max_size()
    }
}

/// Trait for mutating LWE parameters in place.
pub trait SetBase2k {
    /// Sets the limb width `base2k`.
    fn set_base2k(&mut self, base2k: Base2K);
}

/// Trait for mutating the Torus precision `k` in place.
///
/// `k` is a metadata label (the effective torus width); it is independent of the
/// underlying buffer's active size, which `compact()`/`set_size` manage on use.
pub trait SetK {
    /// Sets the Torus precision `k`.
    fn set_k(&mut self, k: TorusPrecision);
}

pub trait SetSize {
    fn set_size(&mut self, size: usize);
}

pub trait Compact
where
    Self: LWEInfos + SetK + SetSize,
{
    fn compact(&mut self) {
        let limbs = (self.k().as_usize() + self.log_n())
            .div_ceil(self.base2k().as_usize())
            .max(1)
            .min(self.max_size());
        self.set_size(limbs);
    }
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

    fn max_size(&self) -> usize {
        self.k.div_ceil(self.base2k) as usize
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }
}

/// A scalar (non-polynomial) LWE ciphertext.
///
/// Stored as two separate [`VecZnx`] buffers:
/// - `body`: degree-0 polynomial (n = 1) holding the scalar body `b`.
/// - `mask`: degree-n polynomial (n = lwe_dim) holding the mask `(a_1, ..., a_n)`.
///
/// `D: Data` is the storage backend (e.g. `Vec<u8>`, `&[u8]`, `&mut [u8]`).
#[derive(PartialEq, Eq, Clone)]
pub struct LWE<D: Data> {
    pub(crate) body: VecZnx<D>,
    pub(crate) mask: VecZnx<D>,
    pub(crate) k: TorusPrecision,
    pub(crate) base2k: Base2K,
}

pub type LWEBackendRef<'a, BE> = LWE<<BE as Backend>::BufRef<'a>>;
pub type LWEBackendMut<'a, BE> = LWE<<BE as Backend>::BufMut<'a>>;

impl<D: Data> LWEInfos for LWE<D> {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn n(&self) -> Degree {
        Degree(self.mask.n() as u32)
    }

    fn max_size(&self) -> usize {
        self.mask.size()
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }
}

impl<D: Data> SetBase2k for LWE<D> {
    fn set_base2k(&mut self, base2k: Base2K) {
        self.base2k = base2k
    }
}

impl<D: Data> LWE<D> {
    /// Returns a shared reference to the body [`VecZnx`] (n = 1).
    pub fn body(&self) -> &VecZnx<D> {
        &self.body
    }

    /// Returns a mutable reference to the body [`VecZnx`] (n = 1).
    pub fn body_mut(&mut self) -> &mut VecZnx<D> {
        &mut self.body
    }

    /// Returns a shared reference to the mask [`VecZnx`] (n = lwe_dim).
    pub fn mask(&self) -> &VecZnx<D> {
        &self.mask
    }

    /// Returns a mutable reference to the mask [`VecZnx`] (n = lwe_dim).
    pub fn mask_mut(&mut self) -> &mut VecZnx<D> {
        &mut self.mask
    }

    fn validate_shape(&self) -> std::io::Result<()> {
        if self.base2k.as_u32() == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "LWE base2k must be non-zero",
            ));
        }
        if self.body.n() != 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("LWE body degree must be 1, got {}", self.body.n()),
            ));
        }
        if self.body.cols() != 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("LWE body cols must be 1, got {}", self.body.cols()),
            ));
        }
        if self.mask.cols() != 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("LWE mask cols must be 1, got {}", self.mask.cols()),
            ));
        }
        if self.body.size() != self.mask.size() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "LWE body and mask sizes must match, got body.size={} mask.size={}",
                    self.body.size(),
                    self.mask.size()
                ),
            ));
        }
        if self.body.size() > self.body.max_size() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "LWE body size must not exceed max_size, got size={} max_size={}",
                    self.body.size(),
                    self.body.max_size()
                ),
            ));
        }
        if self.mask.size() > self.mask.max_size() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "LWE mask size must not exceed max_size, got size={} max_size={}",
                    self.mask.size(),
                    self.mask.max_size()
                ),
            ));
        }
        Ok(())
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
        let body_shape = self.body.shape();
        let body_data = self.body.data;
        let mask_shape = self.mask.shape();
        let mask_data = self.mask.data;
        LWE {
            body: VecZnx::from_data_with_max_size(
                body_data,
                body_shape.n(),
                body_shape.cols(),
                body_shape.size(),
                body_shape.size(),
            ),
            mask: VecZnx::from_data_with_max_size(
                mask_data,
                mask_shape.n(),
                mask_shape.cols(),
                mask_shape.size(),
                mask_shape.size(),
            ),
            base2k: self.base2k,
            k: self.k,
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
        write!(
            f,
            "LWE: base2k={} k={}: body={} mask={}",
            self.base2k().0,
            self.k().0,
            self.body,
            self.mask
        )
    }
}

impl<D: HostDataMut> FillUniform for LWE<D>
where
    VecZnx<D>: FillUniform,
{
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.mask.fill_uniform(log_bound, source);
    }
}

impl LWE<Vec<u8>> {
    /// Allocates a new [`LWE`] with the given parameters.
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: LWEInfos,
    {
        Self::alloc(infos.n(), infos.base2k(), infos.k())
    }

    /// Allocates a new [`LWE`] with the given parameters.
    ///
    /// * `n` -- LWE dimension (mask length).
    /// * `base2k` -- base-2-log of the limb width.
    /// * `k` -- torus precision.
    pub(crate) fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision) -> Self {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        LWE {
            body: VecZnx::from_data(
                poulpy_hal::layouts::HostBytesBackend::alloc_bytes(VecZnx::<Vec<u8>>::bytes_of(1, 1, size)),
                1,
                1,
                size,
            ),
            mask: VecZnx::from_data(
                poulpy_hal::layouts::HostBytesBackend::alloc_bytes(VecZnx::<Vec<u8>>::bytes_of(n.as_usize(), 1, size)),
                n.as_usize(),
                1,
                size,
            ),
            base2k,
            k,
        }
    }

    /// Returns the byte count required for an [`LWE`] with the given parameters.
    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: LWEInfos,
    {
        Self::bytes_of(infos.n(), infos.base2k(), infos.k())
    }

    /// Returns the byte count required for an [`LWE`] with the given parameters.
    ///
    /// * `n` -- LWE dimension (mask length).
    /// * `base2k` -- base-2-log of the limb width.
    /// * `k` -- torus precision.
    pub fn bytes_of(n: Degree, base2k: Base2K, k: TorusPrecision) -> usize {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        VecZnx::<Vec<u8>>::bytes_of(1, 1, size) + VecZnx::<Vec<u8>>::bytes_of(n.as_usize(), 1, size)
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
            k: self.k,
            body: self.body.to_backend_ref(),
            mask: self.mask.to_backend_ref(),
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
            k: self.k,
            body: self.body.to_backend_mut(),
            mask: self.mask.to_backend_mut(),
        }
    }
}

impl<'b, BE: Backend + 'b> LWEToBackendRef<BE> for &mut LWE<BE::BufMut<'b>> {
    fn to_backend_ref(&self) -> LWEBackendRef<'_, BE> {
        LWE {
            base2k: self.base2k,
            k: self.k,
            body: poulpy_hal::layouts::vec_znx_backend_ref_from_mut::<BE>(&self.body),
            mask: poulpy_hal::layouts::vec_znx_backend_ref_from_mut::<BE>(&self.mask),
        }
    }
}

impl<'b, BE: Backend + 'b> LWEToBackendMut<BE> for &mut LWE<BE::BufMut<'b>> {
    fn to_backend_mut(&mut self) -> LWEBackendMut<'_, BE> {
        LWE {
            base2k: self.base2k,
            k: self.k,
            body: poulpy_hal::layouts::vec_znx_backend_mut_from_mut::<BE>(&mut self.body),
            mask: poulpy_hal::layouts::vec_znx_backend_mut_from_mut::<BE>(&mut self.mask),
        }
    }
}

impl<D: HostDataMut> ReaderFrom for LWE<D> {
    /// Deserialises an [`LWE`] in little-endian binary format.
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        self.body.read_from(reader)?;
        self.mask.read_from(reader)?;
        self.validate_shape()
    }
}

impl<D: HostDataRef> WriterTo for LWE<D> {
    /// Serialises the [`LWE`] in little-endian binary format.
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.base2k.into())?;
        self.body.write_to(writer)?;
        self.mask.write_to(writer)
    }
}
