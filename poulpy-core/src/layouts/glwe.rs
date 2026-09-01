use poulpy_hal::{
    layouts::{
        Backend, Data, FillUniform, FitsIn, HostDataMut, HostDataRef, NormalizationState, Normalized, ReaderFrom, ScratchArena,
        ToOwnedDeep, Unnormalized, VecZnx, VecZnxToBackendMut, VecZnxToBackendRef, WriterTo,
    },
    source::Source,
};

use crate::{
    GLWENormalize,
    layouts::{Base2K, Degree, LWEInfos, Rank, SetBase2k, SetK, TorusPrecision},
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use poulpy_hal::layouts::ZnxWord;
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
            k: self.k(),
            rank: self.rank(),
        }
    }
}

impl<T: GLWEInfos + ?Sized> GLWEInfos for &T {
    fn rank(&self) -> Rank {
        (**self).rank()
    }
}

impl<T: GLWEInfos + ?Sized> GLWEInfos for &mut T {
    fn rank(&self) -> Rank {
        (**self).rank()
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

    fn max_size(&self) -> usize {
        self.k.div_ceil(self.base2k) as usize
    }

    fn k(&self) -> TorusPrecision {
        self.k
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
pub struct GLWE<D: Data, W: ZnxWord, S: NormalizationState = Normalized> {
    pub(crate) data: VecZnx<D, W, S>,
    pub(crate) k: TorusPrecision,
    pub(crate) base2k: Base2K,
}

pub type GLWEBackendRef<'a, BE, S = Normalized> = GLWE<<BE as Backend>::BufRef<'a>, <BE as Backend>::ZnxWord, S>;
pub type GLWEBackendMut<'a, BE, S = Normalized> = GLWE<<BE as Backend>::BufMut<'a>, <BE as Backend>::ZnxWord, S>;

/// A GLWE whose limb digits may hold un-propagated carries; see [`NormalizationState`].
pub type UnnormalizedGLWE<D, W> = GLWE<D, W, Unnormalized>;

impl<D: Data, W: ZnxWord, S: NormalizationState> GLWE<D, W, S> {
    /// Relabels this ciphertext as [`Unnormalized`] (free; see [`VecZnx::into_unnormalized`]).
    pub fn into_unnormalized(self) -> GLWE<D, W, Unnormalized> {
        GLWE {
            data: self.data.into_unnormalized(),
            k: self.k,
            base2k: self.base2k,
        }
    }

    /// Relabels this ciphertext into any state its current state [`FitsIn`].
    pub fn into_state<T: NormalizationState>(self) -> GLWE<D, W, T>
    where
        S: FitsIn<T>,
    {
        GLWE {
            data: self.data.into_state(),
            k: self.k,
            base2k: self.base2k,
        }
    }

    /// Relabels as [`Normalized`] without normalizing; see [`VecZnx::assume_normalized`].
    pub fn assume_normalized(self) -> GLWE<D, W, Normalized> {
        GLWE {
            data: self.data.assume_normalized(),
            k: self.k,
            base2k: self.base2k,
        }
    }
}

impl<D: Data, W: ZnxWord> GLWE<D, W, Unnormalized> {
    /// Propagates carries through every column and returns the ciphertext relabelled as [`Normalized`].
    pub fn normalize<M, BE>(self, module: &M, scratch: &mut ScratchArena<'_, BE>) -> GLWE<D, W, Normalized>
    where
        BE: Backend<ZnxWord = W>,
        M: GLWENormalize<BE> + ?Sized,
        Self: GLWEToBackendMut<BE>,
    {
        let mut me = self;
        module.glwe_normalize_assign(&mut me, scratch);
        me.assume_normalized()
    }
}

impl<D: Data, W: ZnxWord, S: NormalizationState> SetBase2k for GLWE<D, W, S> {
    fn set_base2k(&mut self, base2k: Base2K) {
        self.base2k = base2k
    }
}

impl<D: Data, W: ZnxWord, S: NormalizationState> SetBase2k for &mut GLWE<D, W, S> {
    fn set_base2k(&mut self, base2k: Base2K) {
        self.base2k = base2k
    }
}

impl<D: Data, W: ZnxWord, S: NormalizationState> SetK for GLWE<D, W, S> {
    fn set_k(&mut self, k: TorusPrecision) {
        self.k = k
    }
}

impl<D: Data, W: ZnxWord, S: NormalizationState> SetK for &mut GLWE<D, W, S> {
    fn set_k(&mut self, k: TorusPrecision) {
        self.k = k
    }
}

impl<D: Data, W: ZnxWord, S: NormalizationState> GLWE<D, W, S> {
    /// Returns a shared reference to the underlying [`VecZnx`].
    pub fn data(&self) -> &VecZnx<D, W, S> {
        &self.data
    }
}

impl<D: Data, W: ZnxWord, S: NormalizationState> GLWE<D, W, S> {
    /// Returns a mutable reference to the underlying [`VecZnx`].
    pub fn data_mut(&mut self) -> &mut VecZnx<D, W, S> {
        &mut self.data
    }
}

impl<D: Data, W: ZnxWord, S: NormalizationState> LWEInfos for GLWE<D, W, S> {
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
        self.k
    }
}

impl<D: Data, W: ZnxWord, S: NormalizationState> GLWEInfos for GLWE<D, W, S> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols() as u32 - 1)
    }
}

impl<D: HostDataRef, W: ZnxWord, S: NormalizationState> ToOwnedDeep for GLWE<D, W, S> {
    type Owned = GLWE<Vec<u8>, W, S>;
    fn to_owned_deep(&self) -> Self::Owned {
        GLWE {
            data: self.data.to_owned_deep(),
            base2k: self.base2k,
            k: self.k,
        }
    }
}

impl<D: Data, W: ZnxWord, S: NormalizationState> GLWE<D, W, S> {
    /// Rebuilds this backend-owned ciphertext as a host-owned [`GLWE<Vec<u8>, W>`].
    pub fn to_host_owned<BE>(&self) -> GLWE<Vec<u8>, W, S>
    where
        BE: Backend<OwnedBuf = D, ZnxWord = W>,
    {
        GLWE {
            data: self.data.to_host_owned::<BE>(),
            base2k: self.base2k,
            k: self.k,
        }
    }

    /// Formats this backend-owned ciphertext through the existing host [`fmt::Display`] implementation.
    pub fn display_host<BE>(&self) -> String
    where
        BE: Backend<OwnedBuf = D, ZnxWord = W>,
    {
        self.to_host_owned::<BE>().to_string()
    }
}

impl<D: Data, W: ZnxWord, S: NormalizationState> GLWE<D, W, S> {
    /// Zero-cost rename when both backends share the same `OwnedBuf`.
    pub fn reinterpret<To>(self) -> GLWE<To::OwnedBuf, To::ZnxWord, S>
    where
        To: Backend<OwnedBuf = D, ZnxWord = W>,
    {
        let shape = self.data.shape();
        let data = self.data.data;
        GLWE {
            data: VecZnx::from_data(data, shape.n(), shape.cols(), shape.size()),
            base2k: self.base2k,
            k: self.k,
        }
    }
}

impl<D: HostDataRef, W: ZnxWord, S: NormalizationState> fmt::Debug for GLWE<D, W, S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataRef, W: ZnxWord, S: NormalizationState> fmt::Display for GLWE<D, W, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GLWE: base2k={} k={}: {}", self.base2k().0, self.k().0, self.data)
    }
}

impl<D: HostDataMut, W: ZnxWord, S: NormalizationState> FillUniform for GLWE<D, W, S> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

#[expect(
    dead_code,
    reason = "host-owned constructors are kept for serialization and host-only staging"
)]
impl<W: ZnxWord> GLWE<Vec<u8>, W> {
    /// Allocates a new [`GLWE`] with the given parameters.
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        Self::alloc(infos.n(), infos.base2k(), infos.k(), infos.rank())
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
                poulpy_hal::layouts::HostBytesBackend::alloc_bytes(VecZnx::<Vec<u8>, W>::bytes_of(
                    n.into(),
                    (rank + 1).into(),
                    size,
                )),
                n.into(),
                (rank + 1).into(),
                size,
            ),
            base2k,
            k,
        }
    }

    /// Returns the byte count required for a [`GLWE`] with the given parameters.
    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        Self::bytes_of(infos.n(), infos.base2k(), infos.k(), infos.rank())
    }

    /// Returns the byte count required for a [`GLWE`] with the given parameters.
    ///
    /// * `n` -- ring degree.
    /// * `base2k` -- base-2-log of the limb width.
    /// * `k` -- torus precision.
    /// * `rank` -- number of mask polynomials.
    pub fn bytes_of(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> usize {
        VecZnx::<Vec<u8>, W>::bytes_of(n.into(), (rank + 1).into(), k.0.div_ceil(base2k.0) as usize)
    }
}

impl<D: HostDataMut, W: ZnxWord, S: NormalizationState> ReaderFrom for GLWE<D, W, S> {
    /// Deserialises a [`GLWE`] in little-endian binary format.
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        self.data.read_from(reader)?;
        Ok(())
    }
}

impl<D: HostDataRef, W: ZnxWord, S: NormalizationState> WriterTo for GLWE<D, W, S> {
    /// Serialises the [`GLWE`] in little-endian binary format.
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        writer.write_u32::<LittleEndian>(self.base2k.0)?;
        self.data.write_to(writer)
    }
}

/// Borrow a GLWE using the backend's native view type.
///
/// [`Self::State`] carries the [`NormalizationState`] of the storage: an op
/// that consumes DFT-domain inputs bounds its parameters with
/// `A: GLWEToBackendRef<BE, State = Normalized>`, while a carry-producing op
/// bounds its destination with `R: GLWEToBackendMut<BE, State = Unnormalized>`.
pub trait GLWEToBackendRef<BE: Backend>: Sized {
    type State: NormalizationState;
    fn to_backend_ref(&self) -> GLWEBackendRef<'_, BE, Self::State>;
}

impl<BE: Backend, D: Data, S: NormalizationState> GLWEToBackendRef<BE> for GLWE<D, BE::ZnxWord, S>
where
    VecZnx<D, BE::ZnxWord, S>: VecZnxToBackendRef<BE, State = S>,
{
    type State = S;
    fn to_backend_ref(&self) -> GLWEBackendRef<'_, BE, S> {
        GLWE {
            base2k: self.base2k,
            k: self.k,
            data: self.data.to_backend_ref(),
        }
    }
}

pub fn glwe_backend_ref_from_ref<'a, 'b, BE: Backend, S: NormalizationState>(
    glwe: &'a GLWE<BE::BufRef<'b>, BE::ZnxWord, S>,
) -> GLWEBackendRef<'a, BE, S> {
    GLWE {
        base2k: glwe.base2k,
        k: glwe.k,
        data: poulpy_hal::layouts::vec_znx_backend_ref_from_ref::<BE, S>(&glwe.data),
    }
}

impl<'b, BE: Backend + 'b, S: NormalizationState> GLWEToBackendRef<BE> for &GLWE<BE::BufRef<'b>, BE::ZnxWord, S> {
    type State = S;
    fn to_backend_ref(&self) -> GLWEBackendRef<'_, BE, S> {
        glwe_backend_ref_from_ref::<BE, S>(self)
    }
}

pub fn glwe_backend_ref_from_mut<'a, 'b, BE: Backend, S: NormalizationState>(
    glwe: &'a GLWE<BE::BufMut<'b>, BE::ZnxWord, S>,
) -> GLWEBackendRef<'a, BE, S> {
    GLWE {
        base2k: glwe.base2k,
        k: glwe.k,
        data: poulpy_hal::layouts::vec_znx_backend_ref_from_mut::<BE, S>(&glwe.data),
    }
}

/// Mutably borrow a GLWE using the backend's native view type; see [`GLWEToBackendRef`].
pub trait GLWEToBackendMut<BE: Backend>: GLWEToBackendRef<BE> {
    fn to_backend_mut(&mut self) -> GLWEBackendMut<'_, BE, Self::State>;
}

impl<BE: Backend, D: Data, S: NormalizationState> GLWEToBackendMut<BE> for GLWE<D, BE::ZnxWord, S>
where
    VecZnx<D, BE::ZnxWord, S>: VecZnxToBackendRef<BE, State = S> + VecZnxToBackendMut<BE, State = S>,
{
    fn to_backend_mut(&mut self) -> GLWEBackendMut<'_, BE, S> {
        GLWE {
            base2k: self.base2k,
            k: self.k,
            data: self.data.to_backend_mut(),
        }
    }
}

impl<'b, BE: Backend + 'b, S: NormalizationState> GLWEToBackendRef<BE> for &mut GLWE<BE::BufMut<'b>, BE::ZnxWord, S> {
    type State = S;
    fn to_backend_ref(&self) -> GLWEBackendRef<'_, BE, S> {
        glwe_backend_ref_from_mut::<BE, S>(self)
    }
}

impl<'b, BE: Backend + 'b, S: NormalizationState> GLWEToBackendMut<BE> for &mut GLWE<BE::BufMut<'b>, BE::ZnxWord, S> {
    fn to_backend_mut(&mut self) -> GLWEBackendMut<'_, BE, S> {
        glwe_backend_mut_from_mut::<BE, S>(self)
    }
}

pub fn glwe_backend_mut_from_mut<'a, 'b, BE: Backend, S: NormalizationState>(
    glwe: &'a mut GLWE<BE::BufMut<'b>, BE::ZnxWord, S>,
) -> GLWEBackendMut<'a, BE, S> {
    GLWE {
        base2k: glwe.base2k,
        k: glwe.k,
        data: poulpy_hal::layouts::vec_znx_backend_mut_from_mut::<BE, S>(&mut glwe.data),
    }
}
