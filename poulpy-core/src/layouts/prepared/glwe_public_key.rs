use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena};

use crate::{
    GetDistribution, GetDistributionMut,
    dist::Distribution,
    layouts::{
        Base2K, Degree, GLWEInfos, GLWEPrepared, GLWEPreparedFactory, GLWEPreparedToBackendMut, GLWEPreparedToBackendRef,
        GLWEPublicKeyToBackendRef, GetDegree, LWEInfos, Rank, TorusPrecision,
    },
};

/// DFT-domain (prepared) variant of a [`GLWEPublicKey`](crate::layouts::GLWEPublicKey):
/// one [`GLWEPrepared`] per entry and the distribution public-key encryption
/// draws its ephemerals from. Tied to a specific backend via `B: Backend`.
#[derive(PartialEq)]
pub struct GLWEPublicKeyPrepared<D: Data, B: Backend> {
    pub(crate) keys: Vec<GLWEPrepared<D, B>>,
    pub(crate) dist: Distribution,
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
    fn base2k(&self) -> Base2K {
        self.keys[0].base2k()
    }

    fn max_size(&self) -> usize {
        self.keys[0].max_size()
    }

    fn n(&self) -> Degree {
        self.keys[0].n()
    }

    fn k(&self) -> TorusPrecision {
        self.keys[0].k()
    }
}

impl<D: Data, B: Backend> GLWEInfos for GLWEPublicKeyPrepared<D, B> {
    fn rank(&self) -> Rank {
        self.keys[0].rank()
    }
}

pub trait GLWEPublicKeyPreparedFactory<B: Backend>
where
    Self: GetDegree + GLWEPreparedFactory<B>,
{
    fn glwe_public_key_prepared_alloc(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank: Rank,
    ) -> GLWEPublicKeyPrepared<B::OwnedBuf, B> {
        assert!(rank.as_usize() >= 1, "invalid public key: rank must be at least 1");
        GLWEPublicKeyPrepared {
            keys: (0..rank.as_usize())
                .map(|_| self.glwe_prepared_alloc(base2k, k, rank))
                .collect(),
            dist: Distribution::NONE,
        }
    }

    fn glwe_public_key_prepared_alloc_from_infos<A>(&self, infos: &A) -> GLWEPublicKeyPrepared<B::OwnedBuf, B>
    where
        A: GLWEInfos,
    {
        self.glwe_public_key_prepared_alloc(infos.base2k(), infos.k(), infos.rank())
    }

    fn glwe_public_key_prepared_bytes_of(&self, base2k: Base2K, k: TorusPrecision, rank: Rank) -> usize {
        rank.as_usize() * self.glwe_prepared_bytes_of(base2k, k, rank)
    }

    fn glwe_public_key_prepared_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        self.glwe_public_key_prepared_bytes_of(infos.base2k(), infos.k(), infos.rank())
    }

    fn glwe_public_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        self.glwe_prepare_tmp_bytes(infos)
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
                res.keys.len() == other.keys.len(),
                "public key and prepared public key have different entry counts"
            );
            for (mut res_key, other_key) in res.keys.iter_mut().zip(other.keys.iter()) {
                self.glwe_prepare(&mut res_key, &other_key, scratch);
            }
        }
        *res.dist_mut() = *other.dist();
    }
}

impl<B: Backend> GLWEPublicKeyPreparedFactory<B> for Module<B> where Self: GLWEPreparedFactory<B> {}

// module-only API: allocation, sizing, and preparation are provided by
// `GLWEPublicKeyPreparedFactory` on `Module`.

pub type GLWEPublicKeyPreparedBackendRef<'a, B> = GLWEPublicKeyPrepared<<B as Backend>::BufRef<'a>, B>;
pub type GLWEPublicKeyPreparedBackendMut<'a, B> = GLWEPublicKeyPrepared<<B as Backend>::BufMut<'a>, B>;

pub trait GLWEPublicKeyPreparedToBackendRef<B: Backend> {
    fn to_backend_ref(&self) -> GLWEPublicKeyPreparedBackendRef<'_, B>;
}

impl<D: Data, B: Backend> GLWEPublicKeyPreparedToBackendRef<B> for GLWEPublicKeyPrepared<D, B>
where
    GLWEPrepared<D, B>: GLWEPreparedToBackendRef<B>,
{
    fn to_backend_ref(&self) -> GLWEPublicKeyPreparedBackendRef<'_, B> {
        GLWEPublicKeyPrepared {
            keys: self.keys.iter().map(|key| key.to_backend_ref()).collect(),
            dist: self.dist,
        }
    }
}

pub trait GLWEPublicKeyPreparedToBackendMut<B: Backend> {
    fn to_backend_mut(&mut self) -> GLWEPublicKeyPreparedBackendMut<'_, B>;
}

impl<D: Data, B: Backend> GLWEPublicKeyPreparedToBackendMut<B> for GLWEPublicKeyPrepared<D, B>
where
    GLWEPrepared<D, B>: GLWEPreparedToBackendMut<B>,
{
    fn to_backend_mut(&mut self) -> GLWEPublicKeyPreparedBackendMut<'_, B> {
        GLWEPublicKeyPrepared {
            keys: self.keys.iter_mut().map(|key| key.to_backend_mut()).collect(),
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
