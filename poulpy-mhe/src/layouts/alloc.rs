use poulpy_core::layouts::{Base2K, GLWEInfos, ModuleCoreAlloc, ModuleCoreCompressedAlloc, Rank, TorusPrecision};
use poulpy_hal::layouts::{Backend, Module};

use crate::layouts::{GLWEPatCompressed, GLWEPatCompressedOwned};

pub trait MHEModuleAlloc<BE: Backend>:
    ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
    + ModuleCoreCompressedAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
{
    fn glwe_pat_compressed_alloc_from_infos<A: GLWEInfos>(&self, infos: &A) -> GLWEPatCompressedOwned<BE> {
        GLWEPatCompressed {
            inner: self.glwe_compressed_alloc_from_infos(infos),
            canonical: true,
        }
    }

    fn glwe_pat_compressed_alloc(&self, base2k: Base2K, k: TorusPrecision, rank: Rank) -> GLWEPatCompressedOwned<BE> {
        GLWEPatCompressed {
            inner: self.glwe_compressed_alloc(base2k, k, rank),
            canonical: true,
        }
    }
}

impl<BE: Backend> MHEModuleAlloc<BE> for Module<BE> where
    Self: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + ModuleCoreCompressedAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
{
}
