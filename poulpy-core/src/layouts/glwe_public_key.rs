use std::{
    fmt,
    ops::{Deref, DerefMut},
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use poulpy_hal::AlignedBuf;
use poulpy_hal::layouts::{
    Backend, Data, HostDataMut, HostDataRef, MatZnx, MatZnxAtBackendMut, MatZnxAtBackendRef, MatZnxToBackendMut,
    MatZnxToBackendRef, ReaderFrom, Ring, Standard, VecZnx, WriterTo, ZnxWord, mat_znx_at_backend_mut_from_mut,
    mat_znx_at_backend_ref_from_mut, mat_znx_at_backend_ref_from_ref, mat_znx_backend_mut_from_mut, mat_znx_backend_ref_from_mut,
    mat_znx_backend_ref_from_ref,
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
pub struct GLWEPublicKey<D: Data, W: ZnxWord, Rg: Ring = Standard> {
    pub(crate) noise: Option<crate::ComponentNoise>,
    pub(crate) data: MatZnx<D, W, Rg>,
    pub(crate) base2k: Base2K,
    pub(crate) k: TorusPrecision,
    pub(crate) dist: Distribution,
}

impl<D: Data, W: ZnxWord, Rg: Ring> PartialEq for GLWEPublicKey<D, W, Rg>
where
    MatZnx<D, W, Rg>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.noise == other.noise
            && self.data == other.data
            && self.base2k == other.base2k
            && self.k == other.k
            && self.dist == other.dist
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> Eq for GLWEPublicKey<D, W, Rg> where MatZnx<D, W, Rg>: Eq {}

fn entry<D: Data, W: ZnxWord, Rg: Ring>(
    data: VecZnx<D, W, Rg>,
    base2k: Base2K,
    k: TorusPrecision,
    metadata: Option<crate::ComponentNoise>,
) -> GLWE<D, W, Rg> {
    GLWE {
        noise: metadata,
        data,
        base2k,
        k,
        canonical: true,
        plaintext_meta: None,
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GLWEPublicKey<D, W, Rg> {
    pub fn data(&self) -> &MatZnx<D, W, Rg> {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut MatZnx<D, W, Rg> {
        self.noise = None;
        &mut self.data
    }
}

impl<D: HostDataRef, W: ZnxWord, Rg: Ring> GLWEPublicKey<D, W, Rg> {
    /// Entry `l` is the encryption of zero paired with the ephemeral `u_l`.
    pub fn at(&self, l: usize) -> GLWE<&[u8], W, Rg> {
        entry(self.data.at(0, l), self.base2k, self.k, self.noise.clone())
    }
}

impl<D: HostDataMut, W: ZnxWord, Rg: Ring> GLWEPublicKey<D, W, Rg> {
    pub fn at_mut(&mut self, l: usize) -> GLWE<&mut [u8], W, Rg> {
        self.noise = None;
        entry(self.data.at_mut(0, l), self.base2k, self.k, None)
    }
}

/// Backend view of entry `l` of a public key.
pub trait GLWEPublicKeyAtViewRef<BE: Backend, Rg: Ring = Standard> {
    fn at_view(&self, l: usize) -> GLWEViewRef<'_, BE, Rg>;
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
pub trait GLWEPublicKeyAtViewMut<BE: Backend, Rg: Ring = Standard> {
    fn at_view_mut(&mut self, l: usize) -> GLWEViewMut<'_, BE, Rg>;
}

impl<BE: Backend, Rg: Ring> GLWEPublicKeyAtViewRef<BE, Rg> for GLWEPublicKey<BE::OwnedBuf, BE::ZnxWord, Rg> {
    fn at_view(&self, l: usize) -> GLWEViewRef<'_, BE, Rg> {
        GLWEViewRef::from_inner(entry(
            MatZnxAtBackendRef::<BE, Rg>::at_backend(&self.data, 0, l),
            self.base2k,
            self.k,
            self.noise.clone(),
        ))
    }
}

impl<BE: Backend, Rg: Ring> GLWEPublicKeyAtViewMut<BE, Rg> for GLWEPublicKey<BE::OwnedBuf, BE::ZnxWord, Rg> {
    fn at_view_mut(&mut self, l: usize) -> GLWEViewMut<'_, BE, Rg> {
        self.noise = None;
        GLWEViewMut::from_inner(entry(
            MatZnxAtBackendMut::<BE, Rg>::at_backend_mut(&mut self.data, 0, l),
            self.base2k,
            self.k,
            None,
        ))
    }
}

impl<BE: Backend, Rg: Ring> GLWEPublicKeyAtViewRef<BE, Rg> for GLWEPublicKeyBackendRef<'_, BE, Rg> {
    fn at_view(&self, l: usize) -> GLWEViewRef<'_, BE, Rg> {
        let pk = &self.inner;
        GLWEViewRef::from_inner(entry(
            mat_znx_at_backend_ref_from_ref::<BE, Rg>(&pk.data, 0, l),
            pk.base2k,
            pk.k,
            pk.noise.clone(),
        ))
    }
}

impl<BE: Backend, Rg: Ring> GLWEPublicKeyAtViewRef<BE, Rg> for GLWEPublicKeyBackendMut<'_, BE, Rg> {
    fn at_view(&self, l: usize) -> GLWEViewRef<'_, BE, Rg> {
        let pk = &self.inner;
        GLWEViewRef::from_inner(entry(
            mat_znx_at_backend_ref_from_mut::<BE, Rg>(&pk.data, 0, l),
            pk.base2k,
            pk.k,
            pk.noise.clone(),
        ))
    }
}

impl<BE: Backend, Rg: Ring> GLWEPublicKeyAtViewMut<BE, Rg> for GLWEPublicKeyBackendMut<'_, BE, Rg> {
    fn at_view_mut(&mut self, l: usize) -> GLWEViewMut<'_, BE, Rg> {
        let pk = &mut self.inner;
        pk.noise = None;
        GLWEViewMut::from_inner(entry(
            mat_znx_at_backend_mut_from_mut::<BE, Rg>(&mut pk.data, 0, l),
            pk.base2k,
            pk.k,
            None,
        ))
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GetDistributionMut for GLWEPublicKey<D, W, Rg> {
    fn dist_mut(&mut self) -> &mut Distribution {
        &mut self.dist
    }
}

impl<D: Data, W: ZnxWord, Rg: Ring> GetDistribution for GLWEPublicKey<D, W, Rg> {
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

impl<D: Data, W: ZnxWord, Rg: Ring> LWEInfos for GLWEPublicKey<D, W, Rg> {
    fn noise(&self) -> Option<crate::ComponentNoise> {
        self.noise.clone()
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

impl<D: Data, W: ZnxWord, Rg: Ring> GLWEInfos for GLWEPublicKey<D, W, Rg> {
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
impl<W: ZnxWord, Rg: Ring> GLWEPublicKey<AlignedBuf, W, Rg> {
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
            noise: None,
            data: MatZnx::from_data(
                <poulpy_hal::layouts::HostBytesBackend>::alloc_bytes(MatZnx::<AlignedBuf, W, Rg>::bytes_of(
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
        MatZnx::<AlignedBuf, W, Rg>::bytes_of(n.into(), 1, rank.into(), (rank + 1).into(), k.0.div_ceil(base2k.0) as usize)
    }
}

impl<D: HostDataRef, W: ZnxWord, Rg: Ring> fmt::Debug for GLWEPublicKey<D, W, Rg> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GLWEPublicKey")
            .field("base2k", &self.base2k)
            .field("k", &self.k)
            .field("dist", &self.dist)
            .field("data", &self.data)
            .finish()
    }
}

impl<D: HostDataMut, W: ZnxWord, Rg: Ring> ReaderFrom for GLWEPublicKey<D, W, Rg> {
    /// Rejects an invalid shape with [`std::io::ErrorKind::InvalidData`] before
    /// changing the layout. Every read attempt clears the previous metadata.
    /// The stream must contain one row of `r >= 1` encryptions of zero at the
    /// destination's rank, with a nonzero degree at its precision.
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.noise = None;
        let metadata = crate::ComponentNoise::read_optional(reader, self.data.cols_out())?;
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
            || cols_out != self.data.cols_out() as u64
            || size != u64::from(k.0.div_ceil(base2k.0))
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid public key: not one row of rank encryptions of zero at its precision",
            ));
        }
        let components = usize::try_from(cols_out)
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "noise component count is too large"))?;
        crate::layouts::validate_noise_components(metadata.as_ref(), components)?;
        let components = self.data.cols_out();
        crate::layouts::read_mat_znx_with_shape(
            &mut self.data,
            &mut std::io::Read::chain(header.as_slice(), reader),
            Some(1),
            Some(components - 1),
            components,
        )?;
        self.noise = metadata;
        self.dist = dist;
        self.base2k = base2k;
        self.k = k;
        Ok(())
    }
}

impl<D: HostDataRef, W: ZnxWord, Rg: Ring> WriterTo for GLWEPublicKey<D, W, Rg> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        self.dist.validate_wire()?;
        crate::layouts::validate_noise_components(self.noise.as_ref(), self.data.cols_out())?;
        crate::ComponentNoise::write_optional(self.noise.as_ref(), writer)?;
        self.dist.write_to(writer)?;
        writer.write_u32::<LittleEndian>(self.base2k.0)?;
        writer.write_u32::<LittleEndian>(self.k.0)?;
        self.data.write_to(writer)
    }
}

pub struct GLWEPublicKeyBackendRef<'a, BE: Backend + 'a, Rg: Ring = Standard> {
    inner: GLWEPublicKey<BE::BufRef<'a>, BE::ZnxWord, Rg>,
}

impl<'a, BE: Backend + 'a, Rg: Ring> GLWEPublicKeyBackendRef<'a, BE, Rg> {
    pub fn from_inner(inner: GLWEPublicKey<BE::BufRef<'a>, BE::ZnxWord, Rg>) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> GLWEPublicKey<BE::BufRef<'a>, BE::ZnxWord, Rg> {
        self.inner
    }
}

impl<'a, BE: Backend + 'a, Rg: Ring> Deref for GLWEPublicKeyBackendRef<'a, BE, Rg> {
    type Target = GLWEPublicKey<BE::BufRef<'a>, BE::ZnxWord, Rg>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub struct GLWEPublicKeyBackendMut<'a, BE: Backend + 'a, Rg: Ring = Standard> {
    inner: GLWEPublicKey<BE::BufMut<'a>, BE::ZnxWord, Rg>,
}

impl<'a, BE: Backend + 'a, Rg: Ring> GLWEPublicKeyBackendMut<'a, BE, Rg> {
    pub fn from_inner(inner: GLWEPublicKey<BE::BufMut<'a>, BE::ZnxWord, Rg>) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> GLWEPublicKey<BE::BufMut<'a>, BE::ZnxWord, Rg> {
        self.inner
    }
}

impl<'a, BE: Backend + 'a, Rg: Ring> Deref for GLWEPublicKeyBackendMut<'a, BE, Rg> {
    type Target = GLWEPublicKey<BE::BufMut<'a>, BE::ZnxWord, Rg>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<BE: Backend, Rg: Ring> DerefMut for GLWEPublicKeyBackendMut<'_, BE, Rg> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

macro_rules! impl_public_key_infos_for_inner {
    ($ty:ident) => {
        impl<BE: Backend, Rg: Ring> LWEInfos for $ty<'_, BE, Rg> {
            fn noise(&self) -> Option<crate::ComponentNoise> {
                self.inner.noise.clone()
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

        impl<BE: Backend, Rg: Ring> GLWEInfos for $ty<'_, BE, Rg> {
            fn rank(&self) -> Rank {
                self.inner.rank()
            }
        }

        impl<BE: Backend, Rg: Ring> GetDistribution for $ty<'_, BE, Rg> {
            fn dist(&self) -> &Distribution {
                self.inner.dist()
            }
        }
    };
}

impl_public_key_infos_for_inner!(GLWEPublicKeyBackendRef);
impl_public_key_infos_for_inner!(GLWEPublicKeyBackendMut);

impl<BE: Backend, Rg: Ring> GetDistributionMut for GLWEPublicKeyBackendMut<'_, BE, Rg> {
    fn dist_mut(&mut self) -> &mut Distribution {
        self.inner.dist_mut()
    }
}

impl<BE: Backend, Rg: Ring> GLWEPublicKeyToBackendRef<BE, Rg> for GLWEPublicKeyBackendRef<'_, BE, Rg> {
    fn to_backend_ref(&self) -> GLWEPublicKeyBackendRef<'_, BE, Rg> {
        GLWEPublicKeyBackendRef::from_inner(GLWEPublicKey {
            noise: self.inner.noise.clone(),
            data: mat_znx_backend_ref_from_ref::<BE, Rg>(&self.inner.data),
            base2k: self.inner.base2k,
            k: self.inner.k,
            dist: self.inner.dist,
        })
    }
}

impl<BE: Backend, Rg: Ring> GLWEPublicKeyToBackendRef<BE, Rg> for GLWEPublicKeyBackendMut<'_, BE, Rg> {
    fn to_backend_ref(&self) -> GLWEPublicKeyBackendRef<'_, BE, Rg> {
        GLWEPublicKeyBackendRef::from_inner(GLWEPublicKey {
            noise: self.inner.noise.clone(),
            data: mat_znx_backend_ref_from_mut::<BE, Rg>(&self.inner.data),
            base2k: self.inner.base2k,
            k: self.inner.k,
            dist: self.inner.dist,
        })
    }
}

impl<BE: Backend, Rg: Ring> GLWEPublicKeyToBackendMut<BE, Rg> for GLWEPublicKeyBackendMut<'_, BE, Rg> {
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        self.inner.noise = crate::layouts::checked_noise(metadata, crate::layouts::GLWEInfos::rank(self).as_usize() + 1);
    }

    fn to_backend_mut(&mut self) -> GLWEPublicKeyBackendMut<'_, BE, Rg> {
        self.inner.noise = None;
        GLWEPublicKeyBackendMut::from_inner(GLWEPublicKey {
            noise: None,
            data: mat_znx_backend_mut_from_mut::<BE, Rg>(&mut self.inner.data),
            base2k: self.inner.base2k,
            k: self.inner.k,
            dist: self.inner.dist,
        })
    }
}

pub trait GLWEPublicKeyToBackendRef<BE: Backend, Rg: Ring = Standard> {
    fn to_backend_ref(&self) -> GLWEPublicKeyBackendRef<'_, BE, Rg>;
}

impl<BE: Backend, D: Data, Rg: Ring> GLWEPublicKeyToBackendRef<BE, Rg> for GLWEPublicKey<D, BE::ZnxWord, Rg>
where
    MatZnx<D, BE::ZnxWord, Rg>: MatZnxToBackendRef<BE, Rg>,
{
    fn to_backend_ref(&self) -> GLWEPublicKeyBackendRef<'_, BE, Rg> {
        GLWEPublicKeyBackendRef::from_inner(GLWEPublicKey {
            noise: self.noise.clone(),
            data: self.data.to_backend_ref(),
            base2k: self.base2k,
            k: self.k,
            dist: self.dist,
        })
    }
}

pub trait GLWEPublicKeyToBackendMut<BE: Backend, Rg: Ring = Standard> {
    /// Records component noise metadata on this key.
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>);

    /// Borrows coefficients mutably and clears the owner's component noise metadata.
    fn to_backend_mut(&mut self) -> GLWEPublicKeyBackendMut<'_, BE, Rg>;
}

impl<BE: Backend, D: Data, Rg: Ring> GLWEPublicKeyToBackendMut<BE, Rg> for GLWEPublicKey<D, BE::ZnxWord, Rg>
where
    MatZnx<D, BE::ZnxWord, Rg>: MatZnxToBackendMut<BE, Rg>,
{
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        self.noise = crate::layouts::checked_noise(metadata, crate::layouts::GLWEInfos::rank(self).as_usize() + 1);
    }

    fn to_backend_mut(&mut self) -> GLWEPublicKeyBackendMut<'_, BE, Rg> {
        self.noise = None;
        GLWEPublicKeyBackendMut::from_inner(GLWEPublicKey {
            noise: None,
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
        fn key<
            BE: Backend,
            T: GLWEInfos + GetDistribution + GLWEPublicKeyToBackendRef<BE, Rg> + GLWEPublicKeyAtViewRef<BE, Rg>,
            Rg: Ring,
        >() {
        }
        fn key_mut<
            BE: Backend,
            T: GetDistributionMut + GLWEPublicKeyToBackendMut<BE, Rg> + GLWEPublicKeyAtViewMut<BE, Rg>,
            Rg: Ring,
        >() {
        }
        fn assert_for<'a, BE: Backend + 'a>() {
            key::<BE, GLWEPublicKeyBackendRef<'a, BE>, Standard>();
            key::<BE, GLWEPublicKeyBackendMut<'a, BE>, Standard>();
            key_mut::<BE, GLWEPublicKeyBackendMut<'a, BE>, Standard>();
        }
        assert_for::<poulpy_hal::layouts::HostBytesBackend>();
    }
}
