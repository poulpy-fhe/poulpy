use poulpy_hal::{
    api::{ModuleN, ScratchArenaTakeScalarZnx, VecZnxCopy},
    layouts::{
        Backend, Module, ScratchArena, scalar_znx_as_vec_znx_backend_mut_from_mut, scalar_znx_as_vec_znx_backend_ref_from_mut,
    },
    source::Source,
};

use crate::api::{GLWESecretBytesOf, GLWESecretTensorBytesOf};
use crate::layouts::operand_degree;
use crate::{
    GGLWECompressedEncryptSk, GetDistribution, ScratchArenaTakeGLWESecretPrepared, ScratchArenaTakeGLWESecretTensor,
    api::{GLWESecretPreparedFactory, GLWESecretTensorFromGLWESecret},
    layouts::{GGLWEInfos, GGLWEToGGSWKeyCompressedToBackendMut, GLWEInfos, GLWESecretToBackendRef},
};

/// Portable implementation using HAL operations.
///
/// Backend implementations may call this helper without changing their override selection.
pub trait GGLWEToGGSWKeyCompressedEncryptSkReference<BE: Backend> {
    fn gglwe_to_ggsw_key_compressed_encrypt_sk_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_to_ggsw_key_compressed_encrypt_sk_reference<R, S>(
        &self,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToGGSWKeyCompressedToBackendMut<BE> + GGLWEInfos,
        S: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos;
}

impl<BE: Backend> GGLWEToGGSWKeyCompressedEncryptSkReference<BE> for Module<BE>
where
    Self: ModuleN
        + GGLWECompressedEncryptSk<BE>
        + GLWESecretTensorFromGLWESecret<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxCopy<BE>,
{
    fn gglwe_to_ggsw_key_compressed_encrypt_sk_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        let n: usize = operand_degree(self.n(), &[infos.n()]);

        let sk_prepared: usize = self.glwe_secret_prepared_bytes_of(infos.rank());
        let sk_tensor: usize = self.glwe_secret_tensor_bytes_of_from_infos(infos);
        let sk_ij: usize = self.glwe_secret_bytes_of(n.into(), infos.rank());
        let lvl_0: usize = sk_prepared;
        let lvl_1: usize = sk_tensor;
        let lvl_2: usize = sk_ij;
        let lvl_3_prepare: usize = self.glwe_secret_tensor_from_glwe_secret_tmp_bytes(infos.rank());
        let lvl_3_encrypt: usize = self.gglwe_compressed_encrypt_sk_tmp_bytes(infos);

        BE::scratch_aligned(lvl_0) + BE::scratch_aligned(lvl_1) + BE::scratch_aligned(lvl_2) + lvl_3_prepare.max(lvl_3_encrypt)
    }

    fn gglwe_to_ggsw_key_compressed_encrypt_sk_reference<R, S>(
        &self,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToGGSWKeyCompressedToBackendMut<BE> + GGLWEInfos,
        S: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
    {
        let metadata = crate::component_noise::fresh_sk_noise(sk.to_backend_ref().dist, &res.to_backend_ref());
        {
            assert!(
                res.rank() == sk.rank(),
                "gglwe_to_ggsw_key_compressed_encrypt_sk: the key rank differs from the secret's rank"
            );
            operand_degree(self.n(), &[res.n(), sk.n()]);
            let tmp_bytes: usize = self.gglwe_to_ggsw_key_compressed_encrypt_sk_tmp_bytes_reference(res);
            crate::layouts::assert_secret_set(sk.dist());
            assert!(
                scratch.available() >= tmp_bytes,
                "insufficient scratch for compressed GGLWE-to-GGSW key encryption"
            );
            {
                let mut res = res.to_backend_mut();
                let rank: usize = res.rank_out().as_usize();

                let scratch = scratch.borrow();
                let (mut sk_prepared, scratch_1) = scratch.take_glwe_secret_prepared_scratch(sk.n(), res.rank());
                let (mut sk_tensor, scratch_2) = scratch_1.take_glwe_secret_tensor_scratch(sk.n().as_usize().into(), res.rank());
                let (mut sk_ij, mut scratch_3) = scratch_2.take_scalar_znx_scratch(sk.n().as_usize(), rank);
                self.glwe_secret_prepare(&mut sk_prepared, sk);
                self.glwe_secret_tensor_from_glwe_secret(&mut sk_tensor, sk, &mut scratch_3.borrow());

                let sk_tensor_backend = scalar_znx_as_vec_znx_backend_ref_from_mut::<BE>(sk_tensor.data());

                let mut source_xa = Source::new(seed_xa);

                for i in 0..rank {
                    {
                        let mut sk_ij_backend = scalar_znx_as_vec_znx_backend_mut_from_mut::<BE>(&mut sk_ij);
                        for j in 0..rank {
                            let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
                            let idx: usize = lo * rank + hi - (lo * (lo + 1) / 2);
                            self.vec_znx_copy(&mut sk_ij_backend, j, &sk_tensor_backend, idx);
                        }
                    }

                    let (seed_xa_tmp, _) = source_xa.branch();

                    let mut ct = res.at_view_mut(i);

                    self.gglwe_compressed_encrypt_sk(&mut ct, &sk_ij, &sk_prepared, seed_xa_tmp, source_xe, &mut scratch_3);
                }
            }
            scratch.wipe(tmp_bytes);
        }
        res.set_noise(metadata);
    }
}
