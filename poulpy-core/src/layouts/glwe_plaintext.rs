use std::fmt;

use poulpy_hal::layouts::{
    Backend, CoeffNormalized, Data, FillUniform, HostDataMut, HostDataRef, VecZnx, VecZnxReborrowBackendMut,
    VecZnxReborrowBackendRef, VecZnxToBackendMut, VecZnxToBackendRef, ZnxWord,
};
use poulpy_hal::source::Source;

use crate::layouts::{
    Base2K, Degree, GLWE, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, Rank, SetBase2k, SetK, TorusPrecision,
};

/// Width vocabulary for **integer-polynomial** (non-Torus) operands.
///
/// A ciphertext is a Torus element: MSB-anchored, `k()` measures precision
/// from the top, and limbs below it are droppable noise. A plaintext operand
/// fed to a convolution (`GLWEMulPlain`, linear-transformation diagonals,
/// codec buffers) is **not** a Torus polynomial — it is an LSB-anchored
/// integer polynomial in which every encoded limb carries data. Its consuming
/// width is therefore neither `k()` (claimed precision, a label for budget
/// arithmetic) nor `max_size()`/`max_k()` (the allocation, never consumed by
/// compute), but the **encoded width** declared here.
///
/// Ops that consume a plaintext operand bound on this trait, so a type that
/// cannot state its encoded width cannot reach the convolution.
pub trait IntPolyInfos: LWEInfos {
    /// Bit-width of the encoded integer polynomial: every limb up to this
    /// width carries data and is consumed by the convolution.
    fn encoded_k(&self) -> TorusPrecision;
}

impl<T: IntPolyInfos + ?Sized> IntPolyInfos for &T {
    fn encoded_k(&self) -> TorusPrecision {
        (**self).encoded_k()
    }
}

impl<T: IntPolyInfos + ?Sized> IntPolyInfos for &mut T {
    fn encoded_k(&self) -> TorusPrecision {
        (**self).encoded_k()
    }
}

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GLWEPlaintextLayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
}

impl LWEInfos for GLWEPlaintextLayout {
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

impl GLWEInfos for GLWEPlaintextLayout {
    fn rank(&self) -> Rank {
        Rank(0)
    }
}

pub struct GLWEPlaintext<D: Data, W: ZnxWord> {
    pub(crate) data: VecZnx<D, W>,
    pub(crate) k: TorusPrecision,
    pub(crate) base2k: Base2K,
}

pub type GLWEPlaintextBackendRef<'a, BE> = GLWEPlaintext<<BE as Backend>::BufRef<'a>, <BE as Backend>::ZnxWord>;
pub type GLWEPlaintextBackendMut<'a, BE> = GLWEPlaintext<<BE as Backend>::BufMut<'a>, <BE as Backend>::ZnxWord>;

impl<D: Data, W: ZnxWord> SetBase2k for GLWEPlaintext<D, W> {
    fn set_base2k(&mut self, base2k: Base2K) {
        self.base2k = base2k
    }
}

impl<D: Data, W: ZnxWord> SetBase2k for &mut GLWEPlaintext<D, W> {
    fn set_base2k(&mut self, base2k: Base2K) {
        self.base2k = base2k
    }
}

impl<D: Data, W: ZnxWord> SetK for GLWEPlaintext<D, W> {
    fn set_k(&mut self, k: TorusPrecision) {
        self.k = k
    }
}

impl<D: Data, W: ZnxWord> SetK for &mut GLWEPlaintext<D, W> {
    fn set_k(&mut self, k: TorusPrecision) {
        self.k = k
    }
}

impl<D: Data, W: ZnxWord> LWEInfos for GLWEPlaintext<D, W> {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn max_size(&self) -> usize {
        self.data.size()
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }
}

impl<D: Data, W: ZnxWord> GLWEInfos for GLWEPlaintext<D, W> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols() as u32 - 1)
    }
}

impl<D: Data, W: ZnxWord> IntPolyInfos for GLWEPlaintext<D, W> {
    /// Plaintexts are encoded across their whole allocation today, so the
    /// encoded width equals the allocated width. This equality is a property
    /// of the encoders, not of the vocabulary: an encoder writing narrower
    /// than the allocation would report the narrower width here.
    fn encoded_k(&self) -> TorusPrecision {
        self.max_k()
    }
}

impl IntPolyInfos for GLWEPlaintextLayout {
    fn encoded_k(&self) -> TorusPrecision {
        self.max_k()
    }
}

impl<D: Data, W: ZnxWord> GLWEPlaintext<D, W> {
    /// Replaces this plaintext's backing storage with host bytes uploaded into
    /// backend `BE`. The shape and metadata are preserved.
    pub fn copy_from_host_bytes<BE>(&mut self, bytes: &[u8])
    where
        BE: Backend<OwnedBuf = D, ZnxWord = W>,
    {
        assert_eq!(bytes.len(), BE::len_bytes(&self.data.data));
        BE::copy_from_host(&mut self.data.data, bytes);
    }

    /// Rebuilds this backend-owned plaintext as a host-owned [`GLWEPlaintext<Vec<u8>, W>`].
    pub fn to_host_owned<BE>(&self) -> GLWEPlaintext<Vec<u8>, W>
    where
        BE: Backend<OwnedBuf = D, ZnxWord = W>,
    {
        GLWEPlaintext {
            data: self.data.to_host_owned::<BE>(),
            base2k: self.base2k,
            k: self.k,
        }
    }

    /// Formats this backend-owned plaintext through the existing host [`fmt::Display`] implementation.
    pub fn display_host<BE>(&self) -> String
    where
        BE: Backend<OwnedBuf = D, ZnxWord = W>,
    {
        self.to_host_owned::<BE>().to_string()
    }
}

impl<D: Data, W: ZnxWord> GLWEPlaintext<D, W> {
    /// Zero-cost rename when both backends share the same `OwnedBuf`.
    pub fn reinterpret<To>(self) -> GLWEPlaintext<To::OwnedBuf, To::ZnxWord>
    where
        To: Backend<OwnedBuf = D, ZnxWord = W>,
    {
        let shape = self.data.shape();
        let data = self.data.data;
        GLWEPlaintext {
            data: VecZnx::from_data(data, shape.n(), shape.cols(), shape.size()),
            base2k: self.base2k,
            k: self.k,
        }
    }
}

impl<D: HostDataMut, W: ZnxWord> FillUniform for GLWEPlaintext<D, W> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.data.fill_uniform(log_bound, source);
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Display for GLWEPlaintext<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GLWEPlaintext: base2k={} k={}: {}", self.base2k().0, self.k().0, self.data)
    }
}

#[expect(
    dead_code,
    reason = "host-owned constructors are kept for serialization and host-only staging"
)]
impl<W: ZnxWord> GLWEPlaintext<Vec<u8>, W> {
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        // Size to `infos.size()` (not `ceil(k/base2k)`) so that a plaintext
        // allocated from a *key* info (whose `size()` includes the auxiliary
        // limbs above the gadget precision `k`) matches the width reserved by
        // `take_glwe_plaintext_scratch`. For non-key infos the two coincide.
        let n: Degree = infos.n();
        let size: usize = infos.size();
        GLWEPlaintext {
            data: VecZnx::from_data(
                poulpy_hal::layouts::HostBytesBackend::alloc_bytes(VecZnx::<Vec<u8>, W>::bytes_of(n.into(), 1, size)),
                n.into(),
                1,
                size,
            ),
            base2k: infos.base2k(),
            k: infos.k(),
        }
    }

    pub(crate) fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision) -> Self {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        GLWEPlaintext {
            data: VecZnx::from_data(
                poulpy_hal::layouts::HostBytesBackend::alloc_bytes(VecZnx::<Vec<u8>, W>::bytes_of(n.into(), 1, size)),
                n.into(),
                1,
                size,
            ),
            base2k,
            k,
        }
    }
}

impl<W: ZnxWord> GLWEPlaintext<Vec<u8>, W> {
    pub fn alloc_with_meta(n: Degree, base2k: Base2K, k: TorusPrecision) -> Self {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        GLWEPlaintext {
            data: VecZnx::from_data(
                poulpy_hal::layouts::HostBytesBackend::alloc_bytes(VecZnx::<Vec<u8>, W>::bytes_of(n.into(), 1, size)),
                n.into(),
                1,
                size,
            ),
            base2k,
            k,
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        // Mirror `alloc_from_infos` / `take_glwe_plaintext_scratch`: size to
        // `infos.size()` so key infos (with auxiliary limbs) reserve the full
        // width. For non-key infos `size() == ceil(k/base2k)`.
        VecZnx::<Vec<u8>, W>::bytes_of(infos.n().into(), 1, infos.size())
    }

    pub fn bytes_of(n: Degree, base2k: Base2K, k: TorusPrecision) -> usize {
        VecZnx::<Vec<u8>, W>::bytes_of(n.into(), 1, k.0.div_ceil(base2k.0) as usize)
    }
}

impl<BE: Backend, D: Data> GLWEToBackendRef<BE> for GLWEPlaintext<D, BE::ZnxWord>
where
    VecZnx<D, BE::ZnxWord>: VecZnxToBackendRef<BE, State = CoeffNormalized>,
{
    type State = CoeffNormalized;
    fn to_backend_ref(&self) -> GLWE<BE::BufRef<'_>, BE::ZnxWord> {
        GLWE {
            base2k: self.base2k,
            k: self.k,
            data: self.data.to_backend_ref(),
        }
    }
}

impl<BE: Backend, D: Data> GLWEToBackendMut<BE> for GLWEPlaintext<D, BE::ZnxWord>
where
    VecZnx<D, BE::ZnxWord>: VecZnxToBackendRef<BE, State = CoeffNormalized> + VecZnxToBackendMut<BE, State = CoeffNormalized>,
{
    fn to_backend_mut(&mut self) -> GLWE<BE::BufMut<'_>, BE::ZnxWord> {
        GLWE {
            base2k: self.base2k,
            k: self.k,
            data: self.data.to_backend_mut(),
        }
    }
}

/// Reborrows a mutable-view-backed plaintext as a shared backend view.
pub trait GLWEPlaintextReborrowBackendRef<BE: Backend> {
    fn reborrow_backend_ref(&self) -> GLWE<BE::BufRef<'_>, BE::ZnxWord>;
}

impl<'b, BE: Backend + 'b> GLWEPlaintextReborrowBackendRef<BE> for GLWEPlaintext<BE::BufMut<'b>, BE::ZnxWord> {
    fn reborrow_backend_ref(&self) -> GLWE<BE::BufRef<'_>, BE::ZnxWord> {
        GLWE {
            base2k: self.base2k,
            k: self.k,
            data: <VecZnx<BE::BufMut<'b>, BE::ZnxWord> as VecZnxReborrowBackendRef<BE>>::reborrow_backend_ref(&self.data),
        }
    }
}

/// Reborrows a mutable-view-backed plaintext as a mutable backend view.
pub trait GLWEPlaintextReborrowBackendMut<BE: Backend>: GLWEPlaintextReborrowBackendRef<BE> {
    fn reborrow_backend_mut(&mut self) -> GLWE<BE::BufMut<'_>, BE::ZnxWord>;
}

impl<'b, BE: Backend + 'b> GLWEPlaintextReborrowBackendMut<BE> for GLWEPlaintext<BE::BufMut<'b>, BE::ZnxWord> {
    fn reborrow_backend_mut(&mut self) -> GLWE<BE::BufMut<'_>, BE::ZnxWord> {
        GLWE {
            base2k: self.base2k,
            k: self.k,
            data: <VecZnx<BE::BufMut<'b>, BE::ZnxWord> as VecZnxReborrowBackendMut<BE>>::reborrow_backend_mut(&mut self.data),
        }
    }
}

impl<'b, BE: Backend + 'b> GLWEToBackendRef<BE> for &mut GLWEPlaintext<BE::BufMut<'b>, BE::ZnxWord> {
    type State = CoeffNormalized;
    fn to_backend_ref(&self) -> GLWE<BE::BufRef<'_>, BE::ZnxWord> {
        <GLWEPlaintext<BE::BufMut<'b>, BE::ZnxWord> as GLWEPlaintextReborrowBackendRef<BE>>::reborrow_backend_ref(*self)
    }
}

impl<'b, BE: Backend + 'b> GLWEToBackendMut<BE> for &mut GLWEPlaintext<BE::BufMut<'b>, BE::ZnxWord> {
    fn to_backend_mut(&mut self) -> GLWE<BE::BufMut<'_>, BE::ZnxWord> {
        <GLWEPlaintext<BE::BufMut<'b>, BE::ZnxWord> as GLWEPlaintextReborrowBackendMut<BE>>::reborrow_backend_mut(*self)
    }
}

impl<D: Data, W: ZnxWord> GLWEPlaintext<D, W> {
    pub fn data_mut(&mut self) -> &mut VecZnx<D, W> {
        &mut self.data
    }
}

impl<D: Data, W: ZnxWord> GLWEPlaintext<D, W> {
    pub fn data(&self) -> &VecZnx<D, W> {
        &self.data
    }
}
