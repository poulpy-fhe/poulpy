use std::{
    fmt,
    ops::{Deref, DerefMut},
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use poulpy_hal::AlignedBuf;
use poulpy_hal::layouts::{
    Backend, Data, HostDataMut, HostDataRef, MatZnx, MatZnxAtBackendMut, MatZnxAtBackendRef, MatZnxToBackendMut,
    MatZnxToBackendRef, ReaderFrom, VecZnx, WriterTo, ZnxWord, mat_znx_at_backend_mut_from_mut, mat_znx_at_backend_ref_from_mut,
    mat_znx_at_backend_ref_from_ref, mat_znx_backend_mut_from_mut, mat_znx_backend_ref_from_mut, mat_znx_backend_ref_from_ref,
};

use crate::{
    GetDistribution, GetDistributionMut,
    dist::Distribution,
    layouts::{Base2K, Degree, GLWE, GLWEInfos, GLWEViewMut, GLWEViewRef, LWEInfos, Rank, TorusPrecision},
};

/// GLWE public key of rank `r`: `r` encryptions of zero under the secret `s`,
/// `pk_l = (b_l, a_{l,1}, .., a_{l,r})` with `b_l = -Sum_j a_{l,j} s_j + e_l`,
/// one per ephemeral of a public-key encryption.
///
/// Stored as a GGLWE is: a matrix of one row, `r` input and `r + 1` output
/// columns, entry `l` at input column `l`. Its entries are canonical: a writer
/// through a mutable view or `data_mut` must leave them so.
pub struct GLWEPublicKey<D: Data, W: ZnxWord> {
    pub(crate) metadata: Option<crate::EncryptionMetadata>,
    pub(crate) data: MatZnx<D, W>,
    pub(crate) base2k: Base2K,
    pub(crate) k: TorusPrecision,
    pub(crate) dist: Distribution,
}

impl<D: Data, W: ZnxWord> PartialEq for GLWEPublicKey<D, W>
where
    MatZnx<D, W>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.metadata == other.metadata
            && self.data == other.data
            && self.base2k == other.base2k
            && self.k == other.k
            && self.dist == other.dist
    }
}

impl<D: Data, W: ZnxWord> Eq for GLWEPublicKey<D, W> where MatZnx<D, W>: Eq {}

fn entry<D: Data, W: ZnxWord>(
    data: VecZnx<D, W>,
    base2k: Base2K,
    k: TorusPrecision,
    metadata: Option<crate::EncryptionMetadata>,
) -> GLWE<D, W> {
    GLWE {
        metadata,
        data,
        base2k,
        k,
        canonical: true,
    }
}

impl<D: Data, W: ZnxWord> GLWEPublicKey<D, W> {
    pub fn data(&self) -> &MatZnx<D, W> {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut MatZnx<D, W> {
        &mut self.data
    }
}

impl<D: HostDataRef, W: ZnxWord> GLWEPublicKey<D, W> {
    /// Entry `l` is the encryption of zero paired with the ephemeral `u_l`.
    pub fn at(&self, l: usize) -> GLWE<&[u8], W> {
        entry(self.data.at(0, l), self.base2k, self.k, self.metadata)
    }
}

impl<D: HostDataMut, W: ZnxWord> GLWEPublicKey<D, W> {
    pub fn at_mut(&mut self, l: usize) -> GLWE<&mut [u8], W> {
        entry(self.data.at_mut(0, l), self.base2k, self.k, self.metadata)
    }
}

/// Backend view of entry `l` of a public key.
pub trait GLWEPublicKeyAtViewRef<BE: Backend> {
    fn at_view(&self, l: usize) -> GLWEViewRef<'_, BE>;
}

/// Mutable backend view of entry `l` of a public key.
///
/// A backend generation override reaches the entries of the key it receives:
///
/// ```
/// use poulpy_core::layouts::{GLWEInfos, GLWEPublicKeyAtViewMut, GLWEPublicKeyToBackendMut};
/// use poulpy_hal::layouts::Backend;
///
/// fn write_entries<BE: Backend, R: GLWEPublicKeyToBackendMut<BE> + GLWEInfos>(res: &mut R) {
///     let rank = res.rank().as_usize();
///     let mut pk = res.to_backend_mut();
///     for l in 0..rank {
///         let _entry = pk.at_view_mut(l);
///     }
/// }
/// ```
pub trait GLWEPublicKeyAtViewMut<BE: Backend> {
    fn at_view_mut(&mut self, l: usize) -> GLWEViewMut<'_, BE>;
}

impl<BE: Backend> GLWEPublicKeyAtViewRef<BE> for GLWEPublicKey<BE::OwnedBuf, BE::ZnxWord> {
    fn at_view(&self, l: usize) -> GLWEViewRef<'_, BE> {
        GLWEViewRef::from_inner(entry(
            MatZnxAtBackendRef::<BE>::at_backend(&self.data, 0, l),
            self.base2k,
            self.k,
            self.metadata,
        ))
    }
}

impl<BE: Backend> GLWEPublicKeyAtViewMut<BE> for GLWEPublicKey<BE::OwnedBuf, BE::ZnxWord> {
    fn at_view_mut(&mut self, l: usize) -> GLWEViewMut<'_, BE> {
        GLWEViewMut::from_inner(entry(
            MatZnxAtBackendMut::<BE>::at_backend_mut(&mut self.data, 0, l),
            self.base2k,
            self.k,
            self.metadata,
        ))
    }
}

impl<BE: Backend> GLWEPublicKeyAtViewRef<BE> for GLWEPublicKeyBackendRef<'_, BE> {
    fn at_view(&self, l: usize) -> GLWEViewRef<'_, BE> {
        let pk = &self.inner;
        GLWEViewRef::from_inner(entry(
            mat_znx_at_backend_ref_from_ref::<BE>(&pk.data, 0, l),
            pk.base2k,
            pk.k,
            pk.metadata,
        ))
    }
}

impl<BE: Backend> GLWEPublicKeyAtViewRef<BE> for GLWEPublicKeyBackendMut<'_, BE> {
    fn at_view(&self, l: usize) -> GLWEViewRef<'_, BE> {
        let pk = &self.inner;
        GLWEViewRef::from_inner(entry(
            mat_znx_at_backend_ref_from_mut::<BE>(&pk.data, 0, l),
            pk.base2k,
            pk.k,
            pk.metadata,
        ))
    }
}

impl<BE: Backend> GLWEPublicKeyAtViewMut<BE> for GLWEPublicKeyBackendMut<'_, BE> {
    fn at_view_mut(&mut self, l: usize) -> GLWEViewMut<'_, BE> {
        let pk = &mut self.inner;
        GLWEViewMut::from_inner(entry(
            mat_znx_at_backend_mut_from_mut::<BE>(&mut pk.data, 0, l),
            pk.base2k,
            pk.k,
            pk.metadata,
        ))
    }
}

impl<D: Data, W: ZnxWord> GetDistributionMut for GLWEPublicKey<D, W> {
    fn dist_mut(&mut self) -> &mut Distribution {
        &mut self.dist
    }
}

impl<D: Data, W: ZnxWord> GetDistribution for GLWEPublicKey<D, W> {
    fn dist(&self) -> &Distribution {
        &self.dist
    }
}

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GLWEPublicKeyLayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
    pub rank: Rank,
}

impl<D: Data, W: ZnxWord> LWEInfos for GLWEPublicKey<D, W> {
    fn encryption_metadata(&self) -> Option<crate::EncryptionMetadata> {
        self.metadata
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

impl<D: Data, W: ZnxWord> GLWEInfos for GLWEPublicKey<D, W> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols_in() as u32)
    }
}

impl LWEInfos for GLWEPublicKeyLayout {
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

impl GLWEInfos for GLWEPublicKeyLayout {
    fn rank(&self) -> Rank {
        self.rank
    }
}

#[expect(
    dead_code,
    reason = "host-owned constructors are kept for serialization and host-only staging"
)]
impl<W: ZnxWord> GLWEPublicKey<AlignedBuf, W> {
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        Self::alloc(infos.n(), infos.base2k(), infos.k(), infos.rank())
    }

    pub(crate) fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> Self {
        assert!(rank.as_usize() >= 1, "invalid public key: rank must be at least 1");
        let (rows, cols_in, cols_out, size) = (1, rank.as_usize(), (rank + 1).as_usize(), k.0.div_ceil(base2k.0) as usize);
        GLWEPublicKey {
            metadata: None,
            data: MatZnx::from_data(
                <poulpy_hal::layouts::HostBytesBackend>::alloc_bytes(MatZnx::<AlignedBuf, W>::bytes_of(
                    n.into(),
                    rows,
                    cols_in,
                    cols_out,
                    size,
                )),
                n.into(),
                rows,
                cols_in,
                cols_out,
                size,
            ),
            base2k,
            k,
            dist: Distribution::NONE,
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        Self::bytes_of(infos.n(), infos.base2k(), infos.k(), infos.rank())
    }

    pub fn bytes_of(n: Degree, base2k: Base2K, k: TorusPrecision, rank: Rank) -> usize {
        MatZnx::<AlignedBuf, W>::bytes_of(n.into(), 1, rank.into(), (rank + 1).into(), k.0.div_ceil(base2k.0) as usize)
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Debug for GLWEPublicKey<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GLWEPublicKey")
            .field("base2k", &self.base2k)
            .field("k", &self.k)
            .field("dist", &self.dist)
            .field("data", &self.data)
            .finish()
    }
}

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for GLWEPublicKey<D, W> {
    /// Fails with [`std::io::ErrorKind::InvalidData`], leaving the key's layout
    /// unchanged, on a stream that is not one row of `r >= 1` encryptions of
    /// zero of a nonzero degree at its precision.
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        let metadata = crate::EncryptionMetadata::read_optional(reader)?;
        let dist = Distribution::read_from(reader)?;
        let base2k = Base2K(reader.read_u32::<LittleEndian>()?);
        let k = TorusPrecision(reader.read_u32::<LittleEndian>()?);
        // The matrix header (n, size, rows, cols_in, cols_out) is checked before the matrix overwrites the key.
        let mut header = [0u8; 40];
        reader.read_exact(&mut header)?;
        let field = |i: usize| u64::from_le_bytes(header[8 * i..8 * i + 8].try_into().unwrap());
        let (n, size, rows, cols_in, cols_out) = (field(0), field(1), field(2), field(3), field(4));
        if base2k.0 == 0
            || n == 0
            || rows != 1
            || cols_in == 0
            || cols_in.checked_add(1) != Some(cols_out)
            || size != u64::from(k.0.div_ceil(base2k.0))
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid public key: not one row of rank encryptions of zero at its precision",
            ));
        }
        self.data.read_from(&mut std::io::Read::chain(header.as_slice(), reader))?;
        self.metadata = metadata;
        self.dist = dist;
        self.base2k = base2k;
        self.k = k;
        Ok(())
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for GLWEPublicKey<D, W> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        crate::EncryptionMetadata::write_optional(self.metadata, writer)?;
        self.dist.write_to(writer)?;
        writer.write_u32::<LittleEndian>(self.base2k.0)?;
        writer.write_u32::<LittleEndian>(self.k.0)?;
        self.data.write_to(writer)
    }
}

pub struct GLWEPublicKeyBackendRef<'a, BE: Backend + 'a> {
    inner: GLWEPublicKey<BE::BufRef<'a>, BE::ZnxWord>,
}

impl<'a, BE: Backend + 'a> GLWEPublicKeyBackendRef<'a, BE> {
    pub fn from_inner(inner: GLWEPublicKey<BE::BufRef<'a>, BE::ZnxWord>) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> GLWEPublicKey<BE::BufRef<'a>, BE::ZnxWord> {
        self.inner
    }
}

impl<'a, BE: Backend + 'a> Deref for GLWEPublicKeyBackendRef<'a, BE> {
    type Target = GLWEPublicKey<BE::BufRef<'a>, BE::ZnxWord>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub struct GLWEPublicKeyBackendMut<'a, BE: Backend + 'a> {
    inner: GLWEPublicKey<BE::BufMut<'a>, BE::ZnxWord>,
}

impl<'a, BE: Backend + 'a> GLWEPublicKeyBackendMut<'a, BE> {
    pub fn from_inner(inner: GLWEPublicKey<BE::BufMut<'a>, BE::ZnxWord>) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> GLWEPublicKey<BE::BufMut<'a>, BE::ZnxWord> {
        self.inner
    }
}

impl<'a, BE: Backend + 'a> Deref for GLWEPublicKeyBackendMut<'a, BE> {
    type Target = GLWEPublicKey<BE::BufMut<'a>, BE::ZnxWord>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<BE: Backend> DerefMut for GLWEPublicKeyBackendMut<'_, BE> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

macro_rules! impl_public_key_infos_for_inner {
    ($ty:ident) => {
        impl<BE: Backend> LWEInfos for $ty<'_, BE> {
            fn encryption_metadata(&self) -> Option<crate::EncryptionMetadata> {
                self.inner.metadata
            }

            fn base2k(&self) -> Base2K {
                self.inner.base2k()
            }

            fn n(&self) -> Degree {
                self.inner.n()
            }

            fn max_size(&self) -> usize {
                self.inner.max_size()
            }

            fn k(&self) -> TorusPrecision {
                self.inner.k()
            }
        }

        impl<BE: Backend> GLWEInfos for $ty<'_, BE> {
            fn rank(&self) -> Rank {
                self.inner.rank()
            }
        }

        impl<BE: Backend> GetDistribution for $ty<'_, BE> {
            fn dist(&self) -> &Distribution {
                self.inner.dist()
            }
        }
    };
}

impl_public_key_infos_for_inner!(GLWEPublicKeyBackendRef);
impl_public_key_infos_for_inner!(GLWEPublicKeyBackendMut);

impl<BE: Backend> GetDistributionMut for GLWEPublicKeyBackendMut<'_, BE> {
    fn dist_mut(&mut self) -> &mut Distribution {
        self.inner.dist_mut()
    }
}

impl<BE: Backend> GLWEPublicKeyToBackendRef<BE> for GLWEPublicKeyBackendRef<'_, BE> {
    fn to_backend_ref(&self) -> GLWEPublicKeyBackendRef<'_, BE> {
        GLWEPublicKeyBackendRef::from_inner(GLWEPublicKey {
            metadata: self.inner.metadata,
            data: mat_znx_backend_ref_from_ref::<BE>(&self.inner.data),
            base2k: self.inner.base2k,
            k: self.inner.k,
            dist: self.inner.dist,
        })
    }
}

impl<BE: Backend> GLWEPublicKeyToBackendRef<BE> for GLWEPublicKeyBackendMut<'_, BE> {
    fn to_backend_ref(&self) -> GLWEPublicKeyBackendRef<'_, BE> {
        GLWEPublicKeyBackendRef::from_inner(GLWEPublicKey {
            metadata: self.inner.metadata,
            data: mat_znx_backend_ref_from_mut::<BE>(&self.inner.data),
            base2k: self.inner.base2k,
            k: self.inner.k,
            dist: self.inner.dist,
        })
    }
}

impl<BE: Backend> GLWEPublicKeyToBackendMut<BE> for GLWEPublicKeyBackendMut<'_, BE> {
    fn set_encryption_metadata(&mut self, metadata: Option<crate::EncryptionMetadata>) {
        self.inner.metadata = metadata;
    }

    fn to_backend_mut(&mut self) -> GLWEPublicKeyBackendMut<'_, BE> {
        GLWEPublicKeyBackendMut::from_inner(GLWEPublicKey {
            metadata: self.inner.metadata,
            data: mat_znx_backend_mut_from_mut::<BE>(&mut self.inner.data),
            base2k: self.inner.base2k,
            k: self.inner.k,
            dist: self.inner.dist,
        })
    }
}

pub trait GLWEPublicKeyToBackendRef<BE: Backend> {
    fn to_backend_ref(&self) -> GLWEPublicKeyBackendRef<'_, BE>;
}

impl<BE: Backend, D: Data> GLWEPublicKeyToBackendRef<BE> for GLWEPublicKey<D, BE::ZnxWord>
where
    MatZnx<D, BE::ZnxWord>: MatZnxToBackendRef<BE>,
{
    fn to_backend_ref(&self) -> GLWEPublicKeyBackendRef<'_, BE> {
        GLWEPublicKeyBackendRef::from_inner(GLWEPublicKey {
            metadata: self.metadata,
            data: self.data.to_backend_ref(),
            base2k: self.base2k,
            k: self.k,
            dist: self.dist,
        })
    }
}

pub trait GLWEPublicKeyToBackendMut<BE: Backend> {
    /// Records derived encryption provenance on this key.
    fn set_encryption_metadata(&mut self, metadata: Option<crate::EncryptionMetadata>);

    /// Borrows coefficients and copies the current layout and provenance metadata.
    /// Metadata changed on the returned view is local to that view. Operations
    /// that update the owner must call its `set_encryption_metadata` hook.
    fn to_backend_mut(&mut self) -> GLWEPublicKeyBackendMut<'_, BE>;
}

impl<BE: Backend, D: Data> GLWEPublicKeyToBackendMut<BE> for GLWEPublicKey<D, BE::ZnxWord>
where
    MatZnx<D, BE::ZnxWord>: MatZnxToBackendMut<BE>,
{
    fn set_encryption_metadata(&mut self, metadata: Option<crate::EncryptionMetadata>) {
        self.metadata = metadata;
    }

    fn to_backend_mut(&mut self) -> GLWEPublicKeyBackendMut<'_, BE> {
        GLWEPublicKeyBackendMut::from_inner(GLWEPublicKey {
            metadata: self.metadata,
            data: self.data.to_backend_mut(),
            base2k: self.base2k,
            k: self.k,
            dist: self.dist,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn borrows_forward_the_key_traits() {
        // Deref does not satisfy trait bounds: each borrow implements them itself.
        fn key<BE: Backend, T: GLWEInfos + GetDistribution + GLWEPublicKeyToBackendRef<BE> + GLWEPublicKeyAtViewRef<BE>>() {}
        fn key_mut<BE: Backend, T: GetDistributionMut + GLWEPublicKeyToBackendMut<BE> + GLWEPublicKeyAtViewMut<BE>>() {}
        fn assert_for<'a, BE: Backend + 'a>() {
            key::<BE, GLWEPublicKeyBackendRef<'a, BE>>();
            key::<BE, GLWEPublicKeyBackendMut<'a, BE>>();
            key_mut::<BE, GLWEPublicKeyBackendMut<'a, BE>>();
        }
        assert_for::<poulpy_hal::layouts::HostBytesBackend>();
    }
}
