use std::fmt;

use poulpy_hal::layouts::{
    Backend, Data, HostDataMut, HostDataRef, Module, TransferFrom, VecZnx, VecZnxToBackendMut, VecZnxToBackendRef, ZnxWord,
};

use crate::api::ModuleTransfer;
use crate::layouts::{Base2K, Degree, LWEInfos, SetBase2k, TorusPrecision};

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

    fn max_size(&self) -> usize {
        self.k.div_ceil(self.base2k) as usize
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }
}

pub struct LWEPlaintext<D: Data, W: ZnxWord> {
    pub(crate) data: VecZnx<D, W>,
    pub(crate) k: TorusPrecision,
    pub(crate) base2k: Base2K,
}

pub type LWEPlaintextBackendRef<'a, BE> = LWEPlaintext<<BE as Backend>::BufRef<'a>, <BE as Backend>::ZnxWord>;
pub type LWEPlaintextBackendMut<'a, BE> = LWEPlaintext<<BE as Backend>::BufMut<'a>, <BE as Backend>::ZnxWord>;

impl<D: HostDataMut, W: ZnxWord> SetBase2k for LWEPlaintext<D, W> {
    fn set_base2k(&mut self, base2k: Base2K) {
        self.base2k = base2k
    }
}

impl<D: Data, W: ZnxWord> LWEInfos for LWEPlaintext<D, W> {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32 - 1)
    }

    fn max_size(&self) -> usize {
        self.data.size()
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }
}

impl<D: Data, W: ZnxWord> crate::layouts::IntPolyInfos for LWEPlaintext<D, W> {
    fn encoded_k(&self) -> crate::layouts::TorusPrecision {
        self.max_k()
    }
}

impl crate::layouts::IntPolyInfos for LWEPlaintextLayout {
    fn encoded_k(&self) -> crate::layouts::TorusPrecision {
        self.max_k()
    }
}

impl<D: HostDataRef, W: ZnxWord> LWEPlaintext<D, W> {
    /// Copies this plaintext's backing bytes into an owned buffer of
    /// backend `To`, routing via host bytes.
    pub fn to_backend<BE, To>(&self, dst: &Module<To>) -> LWEPlaintext<To::OwnedBuf, To::ZnxWord>
    where
        BE: Backend<OwnedBuf = D, ZnxWord = W>,
        To: Backend<ZnxWord = W>,
        To: TransferFrom<BE>,
    {
        dst.upload_lwe_plaintext(self)
    }
}

impl<D: Data, W: ZnxWord> LWEPlaintext<D, W> {
    /// Zero-cost rename when both backends share the same `OwnedBuf`.
    pub fn reinterpret<To>(self) -> LWEPlaintext<To::OwnedBuf, To::ZnxWord>
    where
        To: Backend<OwnedBuf = D, ZnxWord = W>,
    {
        let shape = self.data.shape();
        let data = self.data.data;
        LWEPlaintext {
            data: VecZnx::from_data_with_max_size(data, shape.n(), shape.cols(), shape.size(), shape.size()),
            base2k: self.base2k,
            k: self.k,
        }
    }
}

#[expect(
    dead_code,
    reason = "host-owned constructors are kept for serialization and host-only staging"
)]
impl<W: ZnxWord> LWEPlaintext<Vec<u8>, W> {
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: LWEInfos,
    {
        Self::alloc(infos.base2k(), infos.k())
    }

    pub(crate) fn alloc(base2k: Base2K, k: TorusPrecision) -> Self {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        LWEPlaintext {
            data: VecZnx::from_data(
                poulpy_hal::layouts::HostBytesBackend::alloc_bytes(VecZnx::<Vec<u8>, W>::bytes_of(1, 1, size)),
                1,
                1,
                size,
            ),
            base2k,
            k,
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: LWEInfos,
    {
        Self::bytes_of(infos.size())
    }

    pub fn bytes_of(size: usize) -> usize {
        VecZnx::<Vec<u8>, W>::bytes_of(1, 1, size)
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Display for LWEPlaintext<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LWEPlaintext: base2k={} k={}: {}", self.base2k().0, self.k().0, self.data)
    }
}

pub trait LWEPlaintextToBackendRef<BE: Backend> {
    fn to_backend_ref(&self) -> LWEPlaintextBackendRef<'_, BE>;
}

impl<BE: Backend> LWEPlaintextToBackendRef<BE> for LWEPlaintext<BE::OwnedBuf, BE::ZnxWord> {
    fn to_backend_ref(&self) -> LWEPlaintextBackendRef<'_, BE> {
        LWEPlaintext {
            data: <VecZnx<BE::OwnedBuf, BE::ZnxWord> as VecZnxToBackendRef<BE>>::to_backend_ref(&self.data),
            base2k: self.base2k,
            k: self.k,
        }
    }
}

impl<'b, BE: Backend + 'b> LWEPlaintextToBackendRef<BE> for &LWEPlaintext<BE::BufRef<'b>, BE::ZnxWord> {
    fn to_backend_ref(&self) -> LWEPlaintextBackendRef<'_, BE> {
        LWEPlaintext {
            data: VecZnx::from_data_with_max_size(
                BE::view_ref(&self.data.data),
                self.data.n(),
                self.data.cols(),
                self.data.size(),
                self.data.size(),
            ),
            base2k: self.base2k,
            k: self.k,
        }
    }
}

impl<'b, BE: Backend + 'b> LWEPlaintextToBackendRef<BE> for &mut LWEPlaintext<BE::BufMut<'b>, BE::ZnxWord> {
    fn to_backend_ref(&self) -> LWEPlaintextBackendRef<'_, BE> {
        LWEPlaintext {
            data: VecZnx::from_data_with_max_size(
                BE::view_ref_mut(&self.data.data),
                self.data.n(),
                self.data.cols(),
                self.data.size(),
                self.data.size(),
            ),
            base2k: self.base2k,
            k: self.k,
        }
    }
}

pub trait LWEPlaintextToBackendMut<BE: Backend>: LWEPlaintextToBackendRef<BE> {
    fn to_backend_mut(&mut self) -> LWEPlaintextBackendMut<'_, BE>;
}

impl<BE: Backend> LWEPlaintextToBackendMut<BE> for LWEPlaintext<BE::OwnedBuf, BE::ZnxWord> {
    fn to_backend_mut(&mut self) -> LWEPlaintextBackendMut<'_, BE> {
        LWEPlaintext {
            data: <VecZnx<BE::OwnedBuf, BE::ZnxWord> as VecZnxToBackendMut<BE>>::to_backend_mut(&mut self.data),
            base2k: self.base2k,
            k: self.k,
        }
    }
}

impl<'b, BE: Backend + 'b> LWEPlaintextToBackendMut<BE> for &mut LWEPlaintext<BE::BufMut<'b>, BE::ZnxWord> {
    fn to_backend_mut(&mut self) -> LWEPlaintextBackendMut<'_, BE> {
        let shape = self.data.shape();
        LWEPlaintext {
            data: VecZnx::from_data_with_max_size(
                BE::view_mut_ref(&mut self.data.data),
                shape.n(),
                shape.cols(),
                shape.size(),
                shape.size(),
            ),
            base2k: self.base2k,
            k: self.k,
        }
    }
}

impl<D: HostDataRef, W: ZnxWord> LWEPlaintext<D, W> {
    pub fn data(&self) -> &VecZnx<D, W> {
        &self.data
    }
}

impl<D: HostDataMut, W: ZnxWord> LWEPlaintext<D, W> {
    pub fn data_mut(&mut self) -> &mut VecZnx<D, W> {
        &mut self.data
    }
}
