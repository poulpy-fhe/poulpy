#![allow(clippy::too_many_arguments)]

use poulpy_hal::{
    api::{ModuleN, VecZnxCopy},
    layouts::{Backend, Module, ScalarZnxToBackendRef, ScratchArena},
    oep::HalVecZnxImpl,
    source::Source,
};

use crate::layouts::operand_degree;
use crate::{
    ScratchArenaTakeGLWE, ScratchArenaTakeGLWEPlaintext,
    encryption::{GGSWEncryptSk, GLWEEncryptSkInternal, GLWEFill},
    layouts::{
        GGSWCompressedSeedMut, GGSWInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, compressed::GGSWCompressedToBackendMut,
        prepared::GLWESecretPreparedToBackendRef,
    },
};

/// Portable implementation using HAL operations.
///
/// Backend implementations may call this helper without changing their override selection.
pub trait GGSWCompressedEncryptSkReference<BE: Backend> {
    fn ggsw_compressed_encrypt_sk_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos;

    fn ggsw_compressed_encrypt_sk_reference<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWCompressedToBackendMut<BE> + GGSWCompressedSeedMut + GGSWInfos,
        P: ScalarZnxToBackendRef<BE>,
        S: GLWESecretPreparedToBackendRef<BE>;
}

impl<BE: Backend + HalVecZnxImpl> GGSWCompressedEncryptSkReference<BE> for Module<BE>
where
    Self: ModuleN + GLWEEncryptSkInternal<BE> + GGSWEncryptSk<BE> + GLWEFill<BE> + VecZnxCopy<BE>,
{
    fn ggsw_compressed_encrypt_sk_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        let n: usize = operand_degree(self.n(), &[infos.n()]);
        let full_ct = self.vec_znx_bytes_of(n, infos.rank().as_usize() + 1, infos.size());
        self.ggsw_encrypt_sk_tmp_bytes(infos) + full_ct
    }

    fn ggsw_compressed_encrypt_sk_reference<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWCompressedToBackendMut<BE> + GGSWCompressedSeedMut + GGSWInfos,
        P: ScalarZnxToBackendRef<BE>,
        S: GLWESecretPreparedToBackendRef<BE>,
    {
        let metadata = crate::component_noise::fresh_sk_noise(sk.to_backend_ref().dist, &res.to_backend_ref());
        {
            let base2k: usize = res.base2k().into();
            let rank: usize = res.rank().into();
            let cols: usize = rank + 1;
            let k_digit: usize = res.k_digit().as_usize();

            let sk_ref = sk.to_backend_ref();
            let pt_backend = pt.to_backend_ref();

            assert!(
                res.rank() == sk_ref.rank(),
                "the ciphertext rank differs from the secret's rank"
            );
            operand_degree(self.n(), &[pt_backend.n().into(), res.n(), sk_ref.n()]);
            crate::layouts::assert_secret_set(&sk_ref.dist);
            assert!(
                scratch.available() >= self.ggsw_compressed_encrypt_sk_tmp_bytes_reference(res),
                "scratch.available(): {} < GGSWCompressedEncryptSk::ggsw_compressed_encrypt_sk_tmp_bytes: {}",
                scratch.available(),
                self.ggsw_compressed_encrypt_sk_tmp_bytes_reference(res)
            );

            let mut seeds: Vec<[u8; 32]> = vec![[0u8; 32]; res.dnum().as_usize() * (res.rank().as_usize() + 1)];

            {
                let mut res = res.to_backend_mut();

                let scratch = scratch.borrow();
                let (mut tmp_pt, mut scratch_1) = scratch.take_glwe_plaintext_scratch(&res);
                let tmp_pt_k = tmp_pt.k().as_usize();

                let mut source = Source::new(seed_xa);

                for row_i in 0..res.dnum().into() {
                    scratch_1 = scratch_1.apply_mut(|scratch| {
                        let mut tmp_pt_backend = tmp_pt.to_backend_mut();
                        crate::gadget::gglwe_gadget_row_plaintext(
                            self,
                            &mut tmp_pt_backend.data,
                            0,
                            base2k,
                            tmp_pt_k,
                            row_i,
                            k_digit,
                            0,
                            &pt_backend,
                            0,
                            scratch,
                        )
                    });

                    for col_j in 0..rank + 1 {
                        // GLWE encrypt of vec_znx_pt into vec_znx_ct

                        let (seed, _) = source.branch();

                        seeds[row_i * cols + col_j] = seed;

                        let tmp_pt_backend = tmp_pt.to_backend_ref();
                        let base2k = res.base2k().into();
                        let scratch_full = scratch_1.borrow();
                        let (mut full_ct, mut scratch_2) = scratch_full.take_glwe_scratch(&res);
                        self.glwe_fill_mask_from_seed(&mut full_ct, seed);
                        self.glwe_encrypt_sk_internal(
                            base2k,
                            full_ct.k().as_usize(),
                            &mut full_ct.data,
                            Some((tmp_pt_backend, col_j)),
                            sk,
                            source_xe,
                            &mut scratch_2,
                        );
                        let full_ct_ref = full_ct.to_backend_ref();
                        let mut ct = res.at_view_mut(row_i, col_j);
                        self.vec_znx_copy(&mut ct.data, 0, &full_ct_ref.data, 0);
                    }
                }
            };

            res.seed_mut().copy_from_slice(&seeds);
            scratch.wipe(self.ggsw_compressed_encrypt_sk_tmp_bytes_reference(res));
        }
        res.set_noise(metadata);
    }
}
