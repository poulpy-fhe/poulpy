use std::fmt;

use poulpy_hal::layouts::{
    Backend, Data, HostDataMut, HostDataRef, Module, TransferFrom, VecZnx, VecZnxReborrowBackendMut, VecZnxReborrowBackendRef,
    VecZnxToBackendMut, VecZnxToBackendRef,
};

use crate::api::ModuleTransfer;
use crate::layouts::{
    Base2K, Degree, GLWE, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, Rank, SetLWEInfos, TorusPrecision,
};

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

    fn size(&self) -> usize {
        self.k.as_usize().div_ceil(self.base2k.as_usize())
    }
}

impl GLWEInfos for GLWEPlaintextLayout {
    fn rank(&self) -> Rank {
        Rank(0)
    }
}

pub struct GLWEPlaintext<D: Data> {
    pub data: VecZnx<D>,
    pub base2k: Base2K,
}

pub type GLWEPlaintextBackendRef<'a, BE> = GLWEPlaintext<<BE as Backend>::BufRef<'a>>;
pub type GLWEPlaintextBackendMut<'a, BE> = GLWEPlaintext<<BE as Backend>::BufMut<'a>>;

impl<D: Data> SetLWEInfos for GLWEPlaintext<D> {
    fn set_base2k(&mut self, base2k: Base2K) {
        self.base2k = base2k
    }
}

impl<D: Data> SetLWEInfos for &mut GLWEPlaintext<D> {
    fn set_base2k(&mut self, base2k: Base2K) {
        self.base2k = base2k
    }
}

impl<D: Data> LWEInfos for GLWEPlaintext<D> {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn size(&self) -> usize {
        self.data.size()
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }
}

impl<D: Data> LWEInfos for &mut GLWEPlaintext<D> {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn size(&self) -> usize {
        self.data.size()
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }
}

impl<D: Data> GLWEInfos for GLWEPlaintext<D> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols() as u32 - 1)
    }
}

impl<D: Data> GLWEInfos for &mut GLWEPlaintext<D> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols() as u32 - 1)
    }
}

impl<D: HostDataRef> GLWEPlaintext<D> {
    /// Copies this plaintext's backing bytes into an owned buffer of
    /// backend `To`, routing via host bytes.
    pub fn to_backend<BE, To>(&self, dst: &Module<To>) -> GLWEPlaintext<To::OwnedBuf>
    where
        BE: Backend<OwnedBuf = D>,
        To: Backend,
        To: TransferFrom<BE>,
    {
        dst.upload_glwe_plaintext(self)
    }
}

impl<D: Data> GLWEPlaintext<D> {
    /// Rebuilds this backend-owned plaintext as a host-owned [`GLWEPlaintext<Vec<u8>>`].
    pub fn to_host_owned<BE>(&self) -> GLWEPlaintext<Vec<u8>>
    where
        BE: Backend<OwnedBuf = D>,
    {
        GLWEPlaintext {
            data: self.data.to_host_owned::<BE>(),
            base2k: self.base2k,
        }
    }

    /// Formats this backend-owned plaintext through the existing host [`fmt::Display`] implementation.
    pub fn display_host<BE>(&self) -> String
    where
        BE: Backend<OwnedBuf = D>,
    {
        self.to_host_owned::<BE>().to_string()
    }
}

impl<D: Data> GLWEPlaintext<D> {
    /// Zero-cost rename when both backends share the same `OwnedBuf`.
    pub fn reinterpret<To>(self) -> GLWEPlaintext<To::OwnedBuf>
    where
        To: Backend<OwnedBuf = D>,
    {
        let shape = self.data.shape();
        let data = self.data.data;
        GLWEPlaintext {
            data: VecZnx::from_data_with_max_size(data, shape.n(), shape.cols(), shape.size(), shape.max_size()),
            base2k: self.base2k,
        }
    }
}

impl<D: HostDataRef> fmt::Display for GLWEPlaintext<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GLWEPlaintext: base2k={} k={}: {}",
            self.base2k().0,
            self.max_k().0,
            self.data
        )
    }
}

#[expect(
    dead_code,
    reason = "host-owned constructors are kept for serialization and host-only staging"
)]
impl GLWEPlaintext<Vec<u8>> {
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        Self::alloc(infos.n(), infos.base2k(), infos.max_k())
    }

    pub(crate) fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision) -> Self {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        GLWEPlaintext {
            data: VecZnx::from_data(
                poulpy_hal::layouts::HostBytesBackend::alloc_bytes(VecZnx::<Vec<u8>>::bytes_of(n.into(), 1, size)),
                n.into(),
                1,
                size,
            ),
            base2k,
        }
    }
}

impl GLWEPlaintext<Vec<u8>> {
    pub fn alloc_with_meta(n: Degree, base2k: Base2K, k: TorusPrecision) -> Self {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        GLWEPlaintext {
            data: VecZnx::from_data(
                poulpy_hal::layouts::HostBytesBackend::alloc_bytes(VecZnx::<Vec<u8>>::bytes_of(n.into(), 1, size)),
                n.into(),
                1,
                size,
            ),
            base2k,
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        Self::bytes_of(infos.n(), infos.base2k(), infos.max_k())
    }

    pub fn bytes_of(n: Degree, base2k: Base2K, k: TorusPrecision) -> usize {
        VecZnx::bytes_of(n.into(), 1, k.0.div_ceil(base2k.0) as usize)
    }
}

impl<BE: Backend, D: Data> GLWEToBackendRef<BE> for GLWEPlaintext<D>
where
    VecZnx<D>: VecZnxToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GLWE<BE::BufRef<'_>> {
        GLWE {
            base2k: self.base2k,
            data: self.data.to_backend_ref(),
        }
    }
}

impl<BE: Backend, D: Data> GLWEToBackendMut<BE> for GLWEPlaintext<D>
where
    VecZnx<D>: VecZnxToBackendRef<BE> + VecZnxToBackendMut<BE>,
{
    fn to_backend_mut(&mut self) -> GLWE<BE::BufMut<'_>> {
        GLWE {
            base2k: self.base2k,
            data: self.data.to_backend_mut(),
        }
    }
}

impl<'b, BE: Backend + 'b> GLWEToBackendRef<BE> for &mut GLWEPlaintext<BE::BufMut<'b>> {
    fn to_backend_ref(&self) -> GLWE<BE::BufRef<'_>> {
        GLWE {
            base2k: self.base2k,
            data: <VecZnx<BE::BufMut<'b>> as VecZnxReborrowBackendRef<BE>>::reborrow_backend_ref(&self.data),
        }
    }
}

impl<'b, BE: Backend + 'b> GLWEToBackendMut<BE> for &mut GLWEPlaintext<BE::BufMut<'b>> {
    fn to_backend_mut(&mut self) -> GLWE<BE::BufMut<'_>> {
        GLWE {
            base2k: self.base2k,
            data: <VecZnx<BE::BufMut<'b>> as VecZnxReborrowBackendMut<BE>>::reborrow_backend_mut(&mut self.data),
        }
    }
}

impl<D: HostDataMut> GLWEPlaintext<D> {
    pub fn data_mut(&mut self) -> &mut VecZnx<D> {
        &mut self.data
    }
}

impl<D: HostDataRef> GLWEPlaintext<D> {
    pub fn data(&self) -> &VecZnx<D> {
        &self.data
    }
}
