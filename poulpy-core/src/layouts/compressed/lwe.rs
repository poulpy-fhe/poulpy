use poulpy_hal::AlignedBuf;
use poulpy_hal::layouts::ZnxWord;
use std::fmt;

use poulpy_hal::{
    api::VecZnxCopy,
    layouts::{
        Backend, Data, HostDataMut, HostDataRef, Module, ReaderFrom, VecZnx, VecZnxToBackendMut, VecZnxToBackendRef, WriterTo,
        vec_znx_alloc_zeroed, vec_znx_backend_mut_from_mut, vec_znx_backend_ref_from_mut, vec_znx_backend_ref_from_ref,
    },
};

use crate::{
    api::LWEFillMask,
    layouts::{Base2K, Degree, LWEInfos, LWEToBackendMut, SetBase2k, TorusPrecision},
};

/// Seed-compressed LWE ciphertext layout.
///
/// Stores only the body (constant term) of an [`LWE`](crate::layouts::LWE) ciphertext; the
/// mask coefficients are regenerated deterministically from a 32-byte
/// PRNG seed during decompression. This layout has no secret dimension and
/// carries no component-noise metadata.
#[derive(PartialEq, Eq, Clone)]
pub struct LWECompressed<D: Data, W: ZnxWord> {
    pub(crate) data: VecZnx<D, W>,
    pub(crate) k: TorusPrecision,
    pub(crate) base2k: Base2K,
    pub(crate) seed: [u8; 32],
}

pub type LWECompressedBackendRef<'a, BE> = LWECompressed<<BE as Backend>::BufRef<'a>, <BE as Backend>::ZnxWord>;
pub type LWECompressedBackendMut<'a, BE> = LWECompressed<<BE as Backend>::BufMut<'a>, <BE as Backend>::ZnxWord>;

impl<D: Data, W: ZnxWord> LWEInfos for LWECompressed<D, W> {
    fn noise(&self) -> Option<crate::ComponentNoise> {
        None
    }

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

impl<D: HostDataRef, W: ZnxWord> fmt::Debug for LWECompressed<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Display for LWECompressed<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LWECompressed: base2k={} k={} seed={:?}: {}",
            self.base2k(),
            self.k(),
            self.seed,
            self.data
        )
    }
}

impl<D: Data, W: ZnxWord> LWECompressed<D, W> {
    /// Allocates a new compressed LWE by copying parameters from an existing info provider.
    pub(crate) fn alloc_from_infos<B: Backend<OwnedBuf = D, ZnxWord = W>, A>(infos: &A) -> Self
    where
        A: LWEInfos,
    {
        Self::alloc::<B>(infos.base2k(), infos.k())
    }

    /// Allocates a new compressed LWE with the given parameters.
    ///
    /// The ring degree is fixed to 1 (scalar LWE). The number of limbs
    /// is `ceil(k / base2k)`.
    pub(crate) fn alloc<B: Backend<OwnedBuf = D, ZnxWord = W>>(base2k: Base2K, k: TorusPrecision) -> Self {
        let size: usize = k.0.div_ceil(base2k.0) as usize;
        LWECompressed {
            data: vec_znx_alloc_zeroed::<B>(1, 1, size),
            k,
            base2k,
            seed: [0u8; 32],
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: LWEInfos,
    {
        Self::bytes_of(infos.base2k(), infos.k())
    }

    pub fn bytes_of(base2k: Base2K, k: TorusPrecision) -> usize {
        VecZnx::<AlignedBuf, W>::bytes_of(1, 1, k.0.div_ceil(base2k.0) as usize)
    }
}

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for LWECompressed<D, W> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        crate::ComponentNoise::read_optional(reader, 0)?;
        let k = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        let base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        let mut seed = [0u8; 32];
        reader.read_exact(&mut seed)?;
        crate::layouts::read_vec_znx_with_shape(&mut self.data, reader, Some(1), 1)?;
        self.k = k;
        self.base2k = base2k;
        self.seed = seed;
        Ok(())
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for LWECompressed<D, W> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        crate::ComponentNoise::write_optional(None, writer)?;
        writer.write_u32::<LittleEndian>(self.k.into())?;
        writer.write_u32::<LittleEndian>(self.base2k.into())?;
        writer.write_all(&self.seed)?;
        self.data.write_to(writer)
    }
}

pub trait LWEDecompress
where
    Self: LWEFillMask<Self::Backend> + VecZnxCopy<Self::Backend>,
{
    type Backend: Backend;

    fn decompress_lwe<R, O>(&self, res: &mut R, other: &O)
    where
        R: LWEToBackendMut<Self::Backend> + LWEInfos + SetBase2k,
        O: LWECompressedToBackendRef<Self::Backend>,
    {
        let other = other.to_backend_ref();

        {
            let mut res = res.to_backend_mut();
            assert_eq!(res.base2k(), other.base2k(), "decompress_lwe: base2k mismatch");
            assert_eq!(res.size(), other.size(), "decompress_lwe: limb count mismatch");
            self.vec_znx_copy(&mut res.body, 0, &other.data, 0);
        }
        self.fill_lwe_mask_from_seed(other.base2k().into(), res, other.seed);
        res.set_base2k(other.base2k());
        res.set_noise(None);
    }
}

impl<B: Backend> LWEDecompress for Module<B>
where
    Self: LWEFillMask<B> + VecZnxCopy<B>,
{
    type Backend = B;
}

// module-only API: decompression is provided by `LWEDecompress` on `Module`.

pub trait LWECompressedToBackendRef<BE: Backend> {
    fn to_backend_ref(&self) -> LWECompressedBackendRef<'_, BE>;
}

impl<BE: Backend> LWECompressedToBackendRef<BE> for LWECompressed<BE::OwnedBuf, BE::ZnxWord> {
    fn to_backend_ref(&self) -> LWECompressedBackendRef<'_, BE> {
        LWECompressed {
            k: self.k,
            base2k: self.base2k,
            seed: self.seed,
            data: <VecZnx<BE::OwnedBuf, BE::ZnxWord> as VecZnxToBackendRef<BE>>::to_backend_ref(&self.data),
        }
    }
}

impl<BE: Backend> LWECompressedToBackendRef<BE> for &LWECompressed<BE::BufRef<'_>, BE::ZnxWord> {
    fn to_backend_ref(&self) -> LWECompressedBackendRef<'_, BE> {
        LWECompressed {
            k: self.k,
            base2k: self.base2k,
            seed: self.seed,
            data: vec_znx_backend_ref_from_ref::<BE>(&self.data),
        }
    }
}

impl<BE: Backend> LWECompressedToBackendRef<BE> for &mut LWECompressed<BE::BufMut<'_>, BE::ZnxWord> {
    fn to_backend_ref(&self) -> LWECompressedBackendRef<'_, BE> {
        LWECompressed {
            k: self.k,
            base2k: self.base2k,
            seed: self.seed,
            data: vec_znx_backend_ref_from_mut::<BE>(&self.data),
        }
    }
}

pub trait LWECompressedToBackendMut<BE: Backend>: LWECompressedToBackendRef<BE> {
    /// Only `None` is supported because this layout has no secret dimension.
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>);

    /// Borrows coefficients and copies the current layout.
    fn to_backend_mut(&mut self) -> LWECompressedBackendMut<'_, BE>;
}

impl<BE: Backend> LWECompressedToBackendMut<BE> for LWECompressed<BE::OwnedBuf, BE::ZnxWord> {
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        assert!(metadata.is_none(), "compressed LWE does not carry component metadata");
    }

    fn to_backend_mut(&mut self) -> LWECompressedBackendMut<'_, BE> {
        LWECompressed {
            k: self.k,
            base2k: self.base2k,
            seed: self.seed,
            data: <VecZnx<BE::OwnedBuf, BE::ZnxWord> as VecZnxToBackendMut<BE>>::to_backend_mut(&mut self.data),
        }
    }
}

impl<BE: Backend> LWECompressedToBackendMut<BE> for &mut LWECompressed<BE::BufMut<'_>, BE::ZnxWord> {
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        assert!(metadata.is_none(), "compressed LWE does not carry component metadata");
    }

    fn to_backend_mut(&mut self) -> LWECompressedBackendMut<'_, BE> {
        LWECompressed {
            k: self.k,
            base2k: self.base2k,
            seed: self.seed,
            data: vec_znx_backend_mut_from_mut::<BE>(&mut self.data),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ComponentNoise, Distribution};
    use poulpy_hal::layouts::HostBytesBackend;

    #[test]
    fn metadata_is_rejected_without_a_secret_dimension() {
        let mut ciphertext = LWECompressed::<AlignedBuf, i64>::alloc::<HostBytesBackend>(Base2K(12), TorusPrecision(35));
        let mut bytes = Vec::new();
        ComponentNoise::write_optional(
            Some(&ComponentNoise::from_secret(Distribution::TernaryProb(0.3), 64)),
            &mut bytes,
        )
        .unwrap();
        assert!(
            ciphertext
                .read_from(&mut bytes.as_slice())
                .unwrap_err()
                .to_string()
                .contains("count")
        );
        assert!(ciphertext.noise().is_none());
        ciphertext.write_to(&mut Vec::new()).unwrap();
    }
}
