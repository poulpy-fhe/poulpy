use poulpy_hal::{
    api::{VmpPMatAlloc, VmpPMatBytesOf, VmpPrepare, VmpPrepareTmpBytes},
    layouts::{Backend, Data, Module, PrepareHint, ScratchArena, VmpPMat, VmpPMatToBackendMut, VmpPMatToBackendRef},
};

use crate::layouts::{GLWELayout, operand_degree};
use crate::{
    GetDistribution, GetDistributionMut,
    dist::Distribution,
    layouts::{Base2K, Degree, GLWEInfos, GLWEPublicKeyToBackendRef, GetDegree, LWEInfos, Rank, TorusPrecision},
};

/// DFT-domain (prepared) variant of a [`GLWEPublicKey`](crate::layouts::GLWEPublicKey):
/// its `r` entries as one prepared matrix of one row, entry `l` at input
/// column `l`, and the distribution public-key encryption draws its
/// ephemerals from. Tied to a specific backend via `B: Backend`.
#[derive(PartialEq)]
pub struct GLWEPublicKeyPrepared<D: Data, B: Backend> {
    pub(crate) metadata: Option<crate::EncryptionMetadata>,
    pub(crate) data: VmpPMat<D, B::DftWord, B>,
    pub(crate) base2k: Base2K,
    pub(crate) k: TorusPrecision,
    pub(crate) dist: Distribution,
}

impl<D: Data, B: Backend> GLWEPublicKeyPrepared<D, B> {
    pub fn data(&self) -> &VmpPMat<D, B::DftWord, B> {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut VmpPMat<D, B::DftWord, B> {
        &mut self.data
    }
}

impl<D: Data, BE: Backend> GetDistribution for GLWEPublicKeyPrepared<D, BE> {
    fn dist(&self) -> &Distribution {
        &self.dist
    }
}

impl<D: Data, BE: Backend> GetDistributionMut for GLWEPublicKeyPrepared<D, BE> {
    fn dist_mut(&mut self) -> &mut Distribution {
        &mut self.dist
    }
}

impl<D: Data, B: Backend> LWEInfos for GLWEPublicKeyPrepared<D, B> {
    fn encryption_metadata(&self) -> Option<crate::EncryptionMetadata> {
        self.metadata
    }

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

impl<D: Data, B: Backend> GLWEInfos for GLWEPublicKeyPrepared<D, B> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols_in() as u32)
    }
}

pub trait GLWEPublicKeyPreparedFactory<B: Backend>
where
    Self: GetDegree + VmpPMatAlloc<B> + VmpPMatBytesOf + VmpPrepare<B> + VmpPrepareTmpBytes,
{
    fn glwe_public_key_prepared_alloc(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank: Rank,
    ) -> GLWEPublicKeyPrepared<B::OwnedBuf, B> {
        self.glwe_public_key_prepared_alloc_from_infos(&GLWELayout {
            n: self.ring_degree(),
            base2k,
            k,
            rank,
        })
    }

    fn glwe_public_key_prepared_alloc_from_infos<A>(&self, infos: &A) -> GLWEPublicKeyPrepared<B::OwnedBuf, B>
    where
        A: GLWEInfos,
    {
        let rank: Rank = infos.rank();
        assert!(rank.as_usize() >= 1, "invalid public key: rank must be at least 1");
        let n: usize = operand_degree(self.ring_degree().as_usize(), &[infos.n()]);
        GLWEPublicKeyPrepared {
            metadata: None,
            data: self.vmp_pmat_alloc(n, 1, rank.into(), (rank + 1).into(), infos.size(), PrepareHint::Reuse),
            base2k: infos.base2k(),
            k: infos.k(),
            dist: Distribution::NONE,
        }
    }

    fn glwe_public_key_prepared_bytes_of(&self, base2k: Base2K, k: TorusPrecision, rank: Rank) -> usize {
        self.glwe_public_key_prepared_bytes_of_from_infos(&GLWELayout {
            n: self.ring_degree(),
            base2k,
            k,
            rank,
        })
    }

    fn glwe_public_key_prepared_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        let rank: usize = infos.rank().into();
        let n: usize = operand_degree(self.ring_degree().as_usize(), &[infos.n()]);
        self.bytes_of_vmp_pmat(n, 1, rank, rank + 1, infos.size(), PrepareHint::Reuse)
    }

    fn glwe_public_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        let rank: usize = infos.rank().into();
        self.vmp_prepare_tmp_bytes(1, rank, rank + 1, infos.size())
    }

    fn glwe_public_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, B>)
    where
        R: GLWEPublicKeyPreparedToBackendMut<B> + GetDistributionMut,
        O: GLWEPublicKeyToBackendRef<B> + GetDistribution,
    {
        {
            let mut res = res.to_backend_mut();
            let other = other.to_backend_ref();
            assert!(
                res.data.cols_in() == other.data.cols_in(),
                "public key and prepared public key have different entry counts"
            );
            operand_degree(self.ring_degree().as_usize(), &[res.n(), other.n()]);
            assert_eq!(res.base2k(), other.base2k());
            assert_eq!(res.k(), other.k());
            assert_eq!(res.size(), other.size());

            self.vmp_prepare(&mut res.data, &other.data, scratch);
        }
        res.set_encryption_metadata(other.to_backend_ref().encryption_metadata());
        *res.dist_mut() = *other.dist();
    }
}

impl<B: Backend> GLWEPublicKeyPreparedFactory<B> for Module<B> where
    Self: GetDegree + VmpPMatAlloc<B> + VmpPMatBytesOf + VmpPrepare<B> + VmpPrepareTmpBytes
{
}

// module-only API: allocation, sizing, and preparation are provided by
// `GLWEPublicKeyPreparedFactory` on `Module`.

pub type GLWEPublicKeyPreparedBackendRef<'a, B> = GLWEPublicKeyPrepared<<B as Backend>::BufRef<'a>, B>;
pub type GLWEPublicKeyPreparedBackendMut<'a, B> = GLWEPublicKeyPrepared<<B as Backend>::BufMut<'a>, B>;

pub trait GLWEPublicKeyPreparedToBackendRef<B: Backend> {
    fn to_backend_ref(&self) -> GLWEPublicKeyPreparedBackendRef<'_, B>;
}

impl<D: Data, B: Backend> GLWEPublicKeyPreparedToBackendRef<B> for GLWEPublicKeyPrepared<D, B>
where
    VmpPMat<D, B::DftWord, B>: VmpPMatToBackendRef<B>,
{
    fn to_backend_ref(&self) -> GLWEPublicKeyPreparedBackendRef<'_, B> {
        GLWEPublicKeyPrepared {
            metadata: self.metadata,
            data: self.data.to_backend_ref(),
            base2k: self.base2k,
            k: self.k,
            dist: self.dist,
        }
    }
}

pub trait GLWEPublicKeyPreparedToBackendMut<B: Backend> {
    /// Records derived encryption provenance on this key.
    fn set_encryption_metadata(&mut self, metadata: Option<crate::EncryptionMetadata>);

    /// Borrows coefficients and copies the current layout and provenance metadata.
    /// Metadata changed on the returned view is local to that view. Operations
    /// that update the owner must call its `set_encryption_metadata` hook.
    fn to_backend_mut(&mut self) -> GLWEPublicKeyPreparedBackendMut<'_, B>;
}

impl<D: Data, B: Backend> GLWEPublicKeyPreparedToBackendMut<B> for GLWEPublicKeyPrepared<D, B>
where
    VmpPMat<D, B::DftWord, B>: VmpPMatToBackendMut<B>,
{
    fn set_encryption_metadata(&mut self, metadata: Option<crate::EncryptionMetadata>) {
        self.metadata = metadata;
    }

    fn to_backend_mut(&mut self) -> GLWEPublicKeyPreparedBackendMut<'_, B> {
        GLWEPublicKeyPrepared {
            metadata: self.metadata,
            data: self.data.to_backend_mut(),
            base2k: self.base2k,
            k: self.k,
            dist: self.dist,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layouts::prepared::GLWESecretTensorPrepared;

    #[test]
    fn distribution_metadata_accepts_opaque_storage() {
        // `()` satisfies Data but exposes no host memory. The generic witness
        // prevents metadata access from accidentally acquiring HostData bounds.
        fn assert_metadata<T: GetDistribution + GetDistributionMut>() {}
        fn assert_for_any_storage<D: Data, B: Backend>() {
            assert_metadata::<GLWEPublicKeyPrepared<D, B>>();
            assert_metadata::<GLWESecretTensorPrepared<D, B>>();
        }
        fn assert_compressed_metadata<D: Data>() {
            use crate::layouts::*;
            fn seed<T: GLWECompressedSeed + GLWECompressedSeedMut>() {}
            fn gadget_seed<T: GGLWECompressedSeed + GGLWECompressedSeedMut>() {}
            fn ggsw_seed<T: GGSWCompressedSeed + GGSWCompressedSeedMut>() {}
            fn automorphism<T: GetGaloisElement + SetGaloisElement>() {}
            fn degrees<T: GLWESwitchingKeyDegrees + GLWESwitchingKeyDegreesMut>() {}
            seed::<GLWECompressed<D, i64>>();
            gadget_seed::<GGLWECompressed<D, i64>>();
            gadget_seed::<GLWEAutomorphismKeyCompressed<D, i64>>();
            ggsw_seed::<GGSWCompressed<D, i64>>();
            automorphism::<GLWEAutomorphismKeyCompressed<D, i64>>();
            degrees::<GLWESwitchingKeyCompressed<D, i64>>();
        }
        assert_compressed_metadata::<()>();
        assert_for_any_storage::<(), poulpy_hal::layouts::HostBytesBackend>();
    }
}
