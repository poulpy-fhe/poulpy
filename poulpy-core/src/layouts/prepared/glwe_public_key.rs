use poulpy_hal::{
    api::{ScratchArenaTakeBasic, VecZnxCopy, VmpPMatAlloc, VmpPMatBytesOf, VmpPrepare, VmpPrepareTmpBytes},
    layouts::{
        Backend, Data, MatZnxToBackendRef, Module, PrepareHint, ScratchArena, VmpPMat, VmpPMatToBackendMut, VmpPMatToBackendRef,
        mat_znx_at_backend_mut_from_mut,
    },
};

use crate::{
    GetDistribution, GetDistributionMut,
    dist::Distribution,
    layouts::{
        Base2K, Degree, GLWE, GLWEInfos, GLWEPreparedFactory, GLWEPublicKeyToBackendRef, GetDegree, LWEInfos, Rank,
        TorusPrecision,
    },
};

/// DFT-domain (prepared) variant of a [`GLWEPublicKey`](crate::layouts::GLWEPublicKey):
/// its `r` entries as one prepared matrix of one row, entry `l` at input
/// column `l`, and the distribution public-key encryption draws its
/// ephemerals from. Tied to a specific backend via `B: Backend`.
#[derive(PartialEq)]
pub struct GLWEPublicKeyPrepared<D: Data, B: Backend> {
    pub(crate) data: VmpPMat<D, B::DftWord, B>,
    pub(crate) base2k: Base2K,
    pub(crate) k: TorusPrecision,
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
    Self: GetDegree
        + GLWEPreparedFactory<B>
        + VmpPMatAlloc<B>
        + VmpPMatBytesOf
        + VmpPrepare<B>
        + VmpPrepareTmpBytes
        + VecZnxCopy<B>,
{
    fn glwe_public_key_prepared_alloc(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank: Rank,
    ) -> GLWEPublicKeyPrepared<B::OwnedBuf, B> {
        assert!(rank.as_usize() >= 1, "invalid public key: rank must be at least 1");
        GLWEPublicKeyPrepared {
            data: self.vmp_pmat_alloc(
                self.ring_degree().into(),
                1,
                rank.into(),
                (rank + 1).into(),
                k.0.div_ceil(base2k.0) as usize,
                PrepareHint::Reuse,
            ),
            base2k,
            k,
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
        self.bytes_of_vmp_pmat(
            self.ring_degree().into(),
            1,
            rank.into(),
            (rank + 1).into(),
            k.0.div_ceil(base2k.0) as usize,
            PrepareHint::Reuse,
        )
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
        let (rank, size): (usize, usize) = (infos.rank().into(), infos.size());
        let lvl_0: usize = B::bytes_of_mat_znx(self.ring_degree().into(), 1, rank, rank + 1, size);
        let lvl_1: usize = self
            .glwe_normalize_tmp_bytes()
            .max(self.vmp_prepare_tmp_bytes(1, rank, rank + 1, size));
        B::scratch_aligned(lvl_0) + lvl_1
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
                res.data.cols_in() == other.keys.len(),
                "public key and prepared public key have different entry counts"
            );
            assert_eq!(res.n(), self.ring_degree());
            assert_eq!(other.n(), self.ring_degree());
            assert_eq!(res.base2k(), other.base2k());
            assert_eq!(res.k(), other.k());
            assert_eq!(res.size(), other.size());

            let (rank, size): (usize, usize) = (other.keys.len(), other.size());
            let (mut mat, mut scratch_1) =
                scratch
                    .borrow()
                    .take_mat_znx_scratch(self.ring_degree().into(), 1, rank, rank + 1, size);
            for (l, key) in other.keys.iter().enumerate() {
                let mut entry = GLWE {
                    data: mat_znx_at_backend_mut_from_mut::<B>(&mut mat, 0, l),
                    k: key.k(),
                    base2k: key.base2k(),
                    canonical: true,
                };
                if key.is_canonical() {
                    for i in 0..rank + 1 {
                        self.vec_znx_copy(&mut entry.data, i, &key.data, i);
                    }
                } else {
                    self.glwe_normalize(&mut &mut entry, &key, &mut scratch_1.borrow());
                }
            }
            self.vmp_prepare(&mut res.data, &mat.to_backend_ref(), &mut scratch_1);
        }
        *res.dist_mut() = *other.dist();
    }
}

impl<B: Backend> GLWEPublicKeyPreparedFactory<B> for Module<B> where
    Self: GetDegree
        + GLWEPreparedFactory<B>
        + VmpPMatAlloc<B>
        + VmpPMatBytesOf
        + VmpPrepare<B>
        + VmpPrepareTmpBytes
        + VecZnxCopy<B>
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
            data: self.data.to_backend_ref(),
            base2k: self.base2k,
            k: self.k,
            dist: self.dist,
        }
    }
}

pub trait GLWEPublicKeyPreparedToBackendMut<B: Backend> {
    fn to_backend_mut(&mut self) -> GLWEPublicKeyPreparedBackendMut<'_, B>;
}

impl<D: Data, B: Backend> GLWEPublicKeyPreparedToBackendMut<B> for GLWEPublicKeyPrepared<D, B>
where
    VmpPMat<D, B::DftWord, B>: VmpPMatToBackendMut<B>,
{
    fn to_backend_mut(&mut self) -> GLWEPublicKeyPreparedBackendMut<'_, B> {
        GLWEPublicKeyPrepared {
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
