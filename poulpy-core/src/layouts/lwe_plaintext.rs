use poulpy_hal::{AlignedBuf, alloc_aligned};
use std::fmt;

use poulpy_hal::layouts::{Backend, Data, HostDataRef, Ring, Standard, VecZnx, VecZnxToBackendMut, VecZnxToBackendRef, ZnxWord};

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

pub struct LWEPlaintext<D: Data, W: ZnxWord, Rg: Ring = Standard> {
    pub(crate) data: VecZnx<D, W, Rg>,
    pub(crate) k: TorusPrecision,
    pub(crate) base2k: Base2K,
}

pub type LWEPlaintextBackendRef<'a, BE, Rg = Standard> = LWEPlaintext<<BE as Backend>::BufRef<'a>, <BE as Backend>::ZnxWord, Rg>;
pub type LWEPlaintextBackendMut<'a, BE, Rg = Standard> = LWEPlaintext<<BE as Backend>::BufMut<'a>, <BE as Backend>::ZnxWord, Rg>;

impl<D: Data, W: ZnxWord, Rg: Ring> SetBase2k for LWEPlaintext<D, W, Rg> {
    fn set_base2k(&mut self, base2k: Base2K) {
        self.base2k = base2k
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> LWEInfos for LWEPlaintext<D, W, Rg> {
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

impl<D: Data, W: ZnxWord, Rg: Ring> crate::layouts::IntPolyInfos for LWEPlaintext<D, W, Rg> {
    fn encoded_k(&self) -> crate::layouts::TorusPrecision {
        self.max_k()
    }
}

impl crate::layouts::IntPolyInfos for LWEPlaintextLayout {
    fn encoded_k(&self) -> crate::layouts::TorusPrecision {
        self.max_k()
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> LWEPlaintext<D, W, Rg> {
    /// Zero-cost rename when both backends share the same `OwnedBuf`.
    pub fn reinterpret<To>(self) -> LWEPlaintext<To::OwnedBuf, To::ZnxWord, Rg>
    where
        To: Backend<OwnedBuf = D, ZnxWord = W>,
    {
        let shape = self.data.shape();
        let data = self.data.into_data();
        LWEPlaintext {
            data: VecZnx::from_shape(data, shape),
            base2k: self.base2k,
            k: self.k,
        }
    }
}

#[expect(
    dead_code,
    reason = "host-owned constructors are kept for serialization and host-only staging"
)]
impl<W: ZnxWord, Rg: Ring> LWEPlaintext<AlignedBuf, W, Rg> {
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
                alloc_aligned::<u8>(VecZnx::<AlignedBuf, W, Rg>::bytes_of(1, 1, size)),
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
        VecZnx::<AlignedBuf, W, Rg>::bytes_of(1, 1, size)
    }
}

impl<D: HostDataRef, W: ZnxWord, Rg: Ring> fmt::Display for LWEPlaintext<D, W, Rg> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LWEPlaintext: base2k={} k={}: {}", self.base2k().0, self.k().0, self.data)
    }
}

pub trait LWEPlaintextToBackendRef<BE: Backend, Rg: Ring = Standard> {
    fn to_backend_ref(&self) -> LWEPlaintextBackendRef<'_, BE, Rg>;
}

impl<BE: Backend, Rg: Ring> LWEPlaintextToBackendRef<BE, Rg> for LWEPlaintext<BE::OwnedBuf, BE::ZnxWord, Rg> {
    fn to_backend_ref(&self) -> LWEPlaintextBackendRef<'_, BE, Rg> {
        LWEPlaintext {
            data: <VecZnx<BE::OwnedBuf, BE::ZnxWord, Rg> as VecZnxToBackendRef<BE, Rg>>::to_backend_ref(&self.data),
            base2k: self.base2k,
            k: self.k,
        }
    }
}

impl<BE: Backend, Rg: Ring> LWEPlaintextToBackendRef<BE, Rg> for &LWEPlaintext<BE::BufRef<'_>, BE::ZnxWord, Rg> {
    fn to_backend_ref(&self) -> LWEPlaintextBackendRef<'_, BE, Rg> {
        LWEPlaintext {
            data: VecZnx::from_shape(BE::view_ref(self.data.data()), self.data.shape()),
            base2k: self.base2k,
            k: self.k,
        }
    }
}

impl<BE: Backend, Rg: Ring> LWEPlaintextToBackendRef<BE, Rg> for &mut LWEPlaintext<BE::BufMut<'_>, BE::ZnxWord, Rg> {
    fn to_backend_ref(&self) -> LWEPlaintextBackendRef<'_, BE, Rg> {
        LWEPlaintext {
            data: VecZnx::from_shape(BE::view_ref_mut(self.data.data()), self.data.shape()),
            base2k: self.base2k,
            k: self.k,
        }
    }
}

pub trait LWEPlaintextToBackendMut<BE: Backend, Rg: Ring = Standard>: LWEPlaintextToBackendRef<BE, Rg> {
    fn to_backend_mut(&mut self) -> LWEPlaintextBackendMut<'_, BE, Rg>;
}

impl<BE: Backend, Rg: Ring> LWEPlaintextToBackendMut<BE, Rg> for LWEPlaintext<BE::OwnedBuf, BE::ZnxWord, Rg> {
    fn to_backend_mut(&mut self) -> LWEPlaintextBackendMut<'_, BE, Rg> {
        LWEPlaintext {
            data: <VecZnx<BE::OwnedBuf, BE::ZnxWord, Rg> as VecZnxToBackendMut<BE, Rg>>::to_backend_mut(&mut self.data),
            base2k: self.base2k,
            k: self.k,
        }
    }
}

impl<BE: Backend, Rg: Ring> LWEPlaintextToBackendMut<BE, Rg> for &mut LWEPlaintext<BE::BufMut<'_>, BE::ZnxWord, Rg> {
    fn to_backend_mut(&mut self) -> LWEPlaintextBackendMut<'_, BE, Rg> {
        LWEPlaintext {
            data: poulpy_hal::layouts::vec_znx_backend_mut_from_mut::<BE, Rg>(&mut self.data),
            base2k: self.base2k,
            k: self.k,
        }
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> LWEPlaintext<D, W, Rg> {
    pub fn data(&self) -> &VecZnx<D, W, Rg> {
        &self.data
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> LWEPlaintext<D, W, Rg> {
    pub fn data_mut(&mut self) -> &mut VecZnx<D, W, Rg> {
        &mut self.data
    }
}
