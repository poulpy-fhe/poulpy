use std::fmt;

use poulpy_hal::layouts::{
    Backend, Data, HostDataMut, HostDataRef, Module, TransferFrom, VecZnx, VecZnxToBackendMut, VecZnxToBackendRef,
};

use crate::api::ModuleTransfer;
use crate::layouts::{Base2K, Degree, LWEInfos, SetLWEInfos, TorusPrecision};

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct LWEPlaintextLayout {
    k: TorusPrecision,
    base2k: Base2K,
}

impl LWEInfos for LWEPlaintextLayout {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn n(&self) -> Degree {
        Degree(0)
    }

    fn size(&self) -> usize {
        self.k.0.div_ceil(self.base2k.0) as usize
    }
}

pub struct LWEPlaintext<D: Data> {
    pub(crate) data: VecZnx<D>,
    pub(crate) base2k: Base2K,
}

pub type LWEPlaintextBackendRef<'a, BE> = LWEPlaintext<<BE as Backend>::BufRef<'a>>;
pub type LWEPlaintextBackendMut<'a, BE> = LWEPlaintext<<BE as Backend>::BufMut<'a>>;

impl<D: HostDataMut> SetLWEInfos for LWEPlaintext<D> {
    fn set_base2k(&mut self, base2k: Base2K) {
        self.base2k = base2k
    }
}

impl<D: Data> LWEInfos for LWEPlaintext<D> {
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

impl<D: HostDataRef> LWEPlaintext<D> {
    /// Copies this plaintext's backing bytes into an owned buffer of
    /// backend `To`, routing via host bytes.
    pub fn to_backend<BE, To>(&self, dst: &Module<To>) -> LWEPlaintext<To::OwnedBuf>
    where
        BE: Backend<OwnedBuf = D>,
        To: Backend,
        To: TransferFrom<BE>,
    {
        dst.upload_lwe_plaintext(self)
    }
}

impl<D: Data> LWEPlaintext<D> {
    /// Zero-cost rename when both backends share the same `OwnedBuf`.
    pub fn reinterpret<To>(self) -> LWEPlaintext<To::OwnedBuf>
    where
        To: Backend<OwnedBuf = D>,
    {
        let shape = self.data.shape();
        let data = self.data.data;
        LWEPlaintext {
            data: VecZnx::from_data_with_max_size(data, shape.n(), shape.cols(), shape.size(), shape.max_size()),
            base2k: self.base2k,
        }
    }
}

#[expect(
    dead_code,
    reason = "host-owned constructors are kept for serialization and host-only staging"
)]
impl LWEPlaintext<Vec<u8>> {
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: LWEInfos,
    {
        Self::alloc(infos.base2k(), infos.max_k())
    }

    pub(crate) fn alloc(base2k: Base2K, k: TorusPrecision) -> Self {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        LWEPlaintext {
            data: VecZnx::from_data(
                poulpy_hal::layouts::HostBytesBackend::alloc_bytes(VecZnx::<Vec<u8>>::bytes_of(1, 1, size)),
                1,
                1,
                size,
            ),
            base2k,
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: LWEInfos,
    {
        Self::bytes_of(infos.size())
    }

    pub fn bytes_of(size: usize) -> usize {
        VecZnx::bytes_of(1, 1, size)
    }
}

impl<D: HostDataRef> fmt::Display for LWEPlaintext<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LWEPlaintext: base2k={} k={}: {}",
            self.base2k().0,
            self.max_k().0,
            self.data
        )
    }
}

pub trait LWEPlaintextToBackendRef<BE: Backend> {
    fn to_backend_ref(&self) -> LWEPlaintextBackendRef<'_, BE>;
}

impl<BE: Backend> LWEPlaintextToBackendRef<BE> for LWEPlaintext<BE::OwnedBuf> {
    fn to_backend_ref(&self) -> LWEPlaintextBackendRef<'_, BE> {
        LWEPlaintext {
            data: <VecZnx<BE::OwnedBuf> as VecZnxToBackendRef<BE>>::to_backend_ref(&self.data),
            base2k: self.base2k,
        }
    }
}

impl<'b, BE: Backend + 'b> LWEPlaintextToBackendRef<BE> for &LWEPlaintext<BE::BufRef<'b>> {
    fn to_backend_ref(&self) -> LWEPlaintextBackendRef<'_, BE> {
        LWEPlaintext {
            data: VecZnx::from_data_with_max_size(
                BE::view_ref(&self.data.data),
                self.data.n(),
                self.data.cols(),
                self.data.size(),
                self.data.max_size(),
            ),
            base2k: self.base2k,
        }
    }
}

impl<'b, BE: Backend + 'b> LWEPlaintextToBackendRef<BE> for &mut LWEPlaintext<BE::BufMut<'b>> {
    fn to_backend_ref(&self) -> LWEPlaintextBackendRef<'_, BE> {
        LWEPlaintext {
            data: VecZnx::from_data_with_max_size(
                BE::view_ref_mut(&self.data.data),
                self.data.n(),
                self.data.cols(),
                self.data.size(),
                self.data.max_size(),
            ),
            base2k: self.base2k,
        }
    }
}

pub trait LWEPlaintextToBackendMut<BE: Backend>: LWEPlaintextToBackendRef<BE> {
    fn to_backend_mut(&mut self) -> LWEPlaintextBackendMut<'_, BE>;
}

impl<BE: Backend> LWEPlaintextToBackendMut<BE> for LWEPlaintext<BE::OwnedBuf> {
    fn to_backend_mut(&mut self) -> LWEPlaintextBackendMut<'_, BE> {
        LWEPlaintext {
            data: <VecZnx<BE::OwnedBuf> as VecZnxToBackendMut<BE>>::to_backend_mut(&mut self.data),
            base2k: self.base2k,
        }
    }
}

impl<'b, BE: Backend + 'b> LWEPlaintextToBackendMut<BE> for &mut LWEPlaintext<BE::BufMut<'b>> {
    fn to_backend_mut(&mut self) -> LWEPlaintextBackendMut<'_, BE> {
        let shape = self.data.shape();
        LWEPlaintext {
            data: VecZnx::from_data_with_max_size(
                BE::view_mut_ref(&mut self.data.data),
                shape.n(),
                shape.cols(),
                shape.size(),
                shape.max_size(),
            ),
            base2k: self.base2k,
        }
    }
}

impl<D: HostDataRef> LWEPlaintext<D> {
    pub fn data(&self) -> &VecZnx<D> {
        &self.data
    }
}

impl<D: HostDataMut> LWEPlaintext<D> {
    pub fn data_mut(&mut self) -> &mut VecZnx<D> {
        &mut self.data
    }
}
