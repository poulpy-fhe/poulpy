use poulpy_hal::layouts::{Backend, Data, Ring, Standard, VmpPMat, VmpPMatToBackendMut, VmpPMatToBackendRef};

use crate::{
    GetDistribution, GetDistributionMut,
    dist::Distribution,
    layouts::{Base2K, Degree, GLWEInfos, LWEInfos, Rank, TorusPrecision},
};

/// DFT-domain (prepared) variant of a [`GLWEPublicKey`](crate::layouts::GLWEPublicKey):
/// its `r` entries as one prepared matrix of one row, entry `l` at input
/// column `l`, and the distribution public-key encryption draws its
/// ephemerals from. Tied to a specific backend via `B: Backend`.
#[derive(PartialEq)]
pub struct GLWEPublicKeyPrepared<D: Data, B: Backend, Rg: Ring = Standard> {
    pub(crate) noise: Option<crate::ComponentNoise>,
    pub(crate) data: VmpPMat<D, B::DftWord, B, Rg>,
    pub(crate) base2k: Base2K,
    pub(crate) k: TorusPrecision,
    pub(crate) dist: Distribution,
}

impl<D: Data, B: Backend, Rg: Ring> GLWEPublicKeyPrepared<D, B, Rg> {
    pub fn data(&self) -> &VmpPMat<D, B::DftWord, B, Rg> {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut VmpPMat<D, B::DftWord, B, Rg> {
        self.noise = None;
        &mut self.data
    }
}

impl<D: Data, BE: Backend, Rg: Ring> GetDistribution for GLWEPublicKeyPrepared<D, BE, Rg> {
    fn dist(&self) -> &Distribution {
        &self.dist
    }
}

impl<D: Data, BE: Backend, Rg: Ring> GetDistributionMut for GLWEPublicKeyPrepared<D, BE, Rg> {
    /// Changing this law requires matching noise provenance. Public-key encryption
    /// rejects a mismatch when metadata is present; untagged keys skip that check.
    fn dist_mut(&mut self) -> &mut Distribution {
        &mut self.dist
    }
}

impl<D: Data, B: Backend, Rg: Ring> LWEInfos for GLWEPublicKeyPrepared<D, B, Rg> {
    fn noise(&self) -> Option<crate::ComponentNoise> {
        self.noise.clone()
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

impl<D: Data, B: Backend, Rg: Ring> GLWEInfos for GLWEPublicKeyPrepared<D, B, Rg> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols_in() as u32)
    }
}

pub type GLWEPublicKeyPreparedBackendRef<'a, B, Rg = Standard> = GLWEPublicKeyPrepared<<B as Backend>::BufRef<'a>, B, Rg>;
pub type GLWEPublicKeyPreparedBackendMut<'a, B, Rg = Standard> = GLWEPublicKeyPrepared<<B as Backend>::BufMut<'a>, B, Rg>;

pub trait GLWEPublicKeyPreparedToBackendRef<B: Backend, Rg: Ring = Standard> {
    fn to_backend_ref(&self) -> GLWEPublicKeyPreparedBackendRef<'_, B, Rg>;
}

impl<D: Data, B: Backend, Rg: Ring> GLWEPublicKeyPreparedToBackendRef<B, Rg> for GLWEPublicKeyPrepared<D, B, Rg>
where
    VmpPMat<D, B::DftWord, B, Rg>: VmpPMatToBackendRef<B, Rg>,
{
    fn to_backend_ref(&self) -> GLWEPublicKeyPreparedBackendRef<'_, B, Rg> {
        GLWEPublicKeyPrepared {
            noise: self.noise.clone(),
            data: self.data.to_backend_ref(),
            base2k: self.base2k,
            k: self.k,
            dist: self.dist,
        }
    }
}

pub trait GLWEPublicKeyPreparedToBackendMut<B: Backend, Rg: Ring = Standard> {
    /// Records component noise metadata on this key.
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>);

    /// Borrows coefficients mutably and clears the owner's component noise metadata.
    fn to_backend_mut(&mut self) -> GLWEPublicKeyPreparedBackendMut<'_, B, Rg>;
}

impl<D: Data, B: Backend, Rg: Ring> GLWEPublicKeyPreparedToBackendMut<B, Rg> for GLWEPublicKeyPrepared<D, B, Rg>
where
    VmpPMat<D, B::DftWord, B, Rg>: VmpPMatToBackendMut<B, Rg>,
{
    fn set_noise(&mut self, metadata: Option<crate::ComponentNoise>) {
        self.noise = crate::layouts::checked_noise(metadata, crate::layouts::GLWEInfos::rank(self).as_usize() + 1);
    }

    fn to_backend_mut(&mut self) -> GLWEPublicKeyPreparedBackendMut<'_, B, Rg> {
        self.noise = None;
        GLWEPublicKeyPrepared {
            noise: None,
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
