use poulpy_hal::{
    api::{ModuleN, ScratchArenaTakeScalarZnx, VecZnxSwitchRing},
    layouts::{
        Backend, Module, Ring, ScratchArena, Standard, scalar_znx_as_vec_znx_backend_mut_from_mut,
        scalar_znx_as_vec_znx_backend_ref_from_ref,
    },
    source::Source,
};

pub use crate::api::GLWESwitchingKeyEncryptSk;
use crate::layouts::operand_degree;
use crate::{
    Distribution, GetDistribution, GetDistributionMut, ScratchArenaTakeGLWESecret, ScratchArenaTakeGLWESecretPrepared,
    api::{GLWESecretPreparedFactory, GLWESecretSampling},
    encryption::gglwe::GGLWEEncryptSk,
    layouts::{
        GGLWEInfos, GGLWEToBackendMut, GLWEInfos, GLWESecretToBackendMut, GLWESecretToBackendRef, GLWESwitchingKeyDegreesMut,
        LWEInfos,
    },
};

/// Portable implementation using HAL operations.
///
/// Backend implementations may call this helper without changing their override selection.
pub trait GLWESwitchingKeyEncryptSkReference<BE: Backend, Rg: Ring = Standard> {
    fn glwe_switching_key_encrypt_sk_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_switching_key_encrypt_sk_reference<R, S1, S2>(
        &self,
        res: &mut R,
        sk_in: &S1,
        sk_out: &S2,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE, Rg> + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        S1: GLWESecretToBackendRef<BE, Rg> + GLWEInfos,
        S2: GLWESecretToBackendRef<BE, Rg> + GetDistribution + GLWEInfos;

    /// Scratch of [`Self::glwe_switching_key_encrypt_sk_encapsulation_reference`] at every gap: the larger of the
    /// subring secret of the smallest gap, half the key's degree, and the key's encryption.
    fn glwe_switching_key_encrypt_sk_encapsulation_tmp_bytes_reference<A>(&self, dense_to_sparse_infos: &A) -> usize
    where
        A: GGLWEInfos,
        Self: GLWESwitchingKeyEncryptSk<BE, Rg>;

    /// Implements
    /// [`EncryptionImpl::glwe_switching_key_encrypt_sk_encapsulation`](crate::oep::EncryptionImpl::glwe_switching_key_encrypt_sk_encapsulation),
    /// which `impl_encryption_reference_full!` forwards here: checks every operand, draws `sk_sparse` on the subring
    /// of the gap with `glwe_secret_fill_ternary_hw`, embeds it with `vec_znx_switch_ring`, tags it encapsulated,
    /// encrypts `dense_to_sparse` with `glwe_switching_key_encrypt_sk` and wipes its budget.
    #[allow(clippy::too_many_arguments)]
    fn glwe_switching_key_encrypt_sk_encapsulation_reference<R, S1, S2>(
        &self,
        dense_to_sparse: &mut R,
        sk_sparse: &mut S2,
        sk_dense: &S1,
        log_gap: usize,
        hamming_weight: usize,
        source_xs: &mut Source,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE, Rg> + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        S1: GLWESecretToBackendRef<BE, Rg> + GetDistribution + GLWEInfos,
        S2: GLWESecretToBackendMut<BE, Rg> + GetDistribution + GetDistributionMut + GLWEInfos,
        Self: GLWESwitchingKeyEncryptSk<BE, Rg> + GLWESecretSampling<BE, Rg>;
}

impl<BE: Backend, Rg: Ring> GLWESwitchingKeyEncryptSkReference<BE, Rg> for Module<BE>
where
    Self: ModuleN + GGLWEEncryptSk<BE, Rg> + GLWESecretPreparedFactory<BE, Rg> + VecZnxSwitchRing<BE, Rg>,
{
    fn glwe_switching_key_encrypt_sk_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        let n: usize = operand_degree(self.n(), &[infos.n()]);

        let lvl_0: usize = BE::scalar_znx_bytes_of(n, infos.rank_in().into());
        let lvl_1: usize = BE::scalar_znx_bytes_of(n, infos.rank_out().into());
        let lvl_2: usize = self.glwe_secret_prepared_bytes_of_from_infos(infos);
        let lvl_3_encrypt: usize = self.gglwe_encrypt_sk_tmp_bytes(infos);
        lvl_0 + lvl_1 + lvl_2 + lvl_3_encrypt
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_switching_key_encrypt_sk_reference<R, S1, S2>(
        &self,
        res: &mut R,
        sk_in: &S1,
        sk_out: &S2,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE, Rg> + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        S1: GLWESecretToBackendRef<BE, Rg> + GLWEInfos,
        S2: GLWESecretToBackendRef<BE, Rg> + GetDistribution + GLWEInfos,
    {
        let sk_in = sk_in.to_backend_ref();
        let sk_out_ref = sk_out.to_backend_ref();

        assert!(sk_in.n().0 <= res.n().0);
        assert!(sk_out_ref.n().0 <= res.n().0);
        crate::layouts::assert_secret_set(sk_out.dist());
        assert!(
            scratch.available()
                >= GLWESwitchingKeyEncryptSkReference::glwe_switching_key_encrypt_sk_tmp_bytes_reference(self, res),
            "scratch.available() < GLWESwitchingKeyEncryptSk::glwe_switching_key_encrypt_sk_tmp_bytes"
        );
        let tmp_bytes: usize = GLWESwitchingKeyEncryptSkReference::glwe_switching_key_encrypt_sk_tmp_bytes_reference(self, res);
        {
            let (mut sk_in_lifted, scratch_1) = scratch
                .borrow()
                .take_scalar_znx_scratch(res.n().as_usize(), sk_in.rank().into());
            let sk_in_backend_vec = scalar_znx_as_vec_znx_backend_ref_from_ref::<BE, Rg>(sk_in.data());
            for i in 0..sk_in.rank().into() {
                let mut sk_in_lifted_backend_vec = scalar_znx_as_vec_znx_backend_mut_from_mut::<BE, Rg>(&mut sk_in_lifted);
                self.vec_znx_switch_ring(&mut sk_in_lifted_backend_vec, i, &sk_in_backend_vec, i);
            }

            let (mut sk_out_lifted, scratch_2) = scratch_1.take_glwe_secret_scratch(res.n().as_usize().into(), sk_out_ref.rank());
            sk_out_lifted.dist = *sk_out.dist();
            let sk_out_backend_vec = scalar_znx_as_vec_znx_backend_ref_from_ref::<BE, Rg>(sk_out_ref.data());
            for i in 0..sk_out_ref.rank().into() {
                let mut sk_out_lifted_backend_vec =
                    scalar_znx_as_vec_znx_backend_mut_from_mut::<BE, Rg>(sk_out_lifted.data_mut());
                self.vec_znx_switch_ring(&mut sk_out_lifted_backend_vec, i, &sk_out_backend_vec, i);
            }

            let (mut sk_out_prepared, scratch_3) = scratch_2.take_glwe_secret_prepared_scratch(res.n(), sk_out_ref.rank());
            self.glwe_secret_prepare(&mut sk_out_prepared, &sk_out_lifted);

            let (mut enc_scratch, _scratch_4) = scratch_3.split_at(self.gglwe_encrypt_sk_tmp_bytes(res));
            self.gglwe_encrypt_sk(res, &sk_in_lifted, &sk_out_prepared, source_xe, source_xa, &mut enc_scratch);

            *res.input_degree() = sk_in.n();
            *res.output_degree() = sk_out_ref.n();
        }
        scratch.wipe(tmp_bytes);
    }

    fn glwe_switching_key_encrypt_sk_encapsulation_tmp_bytes_reference<A>(&self, dense_to_sparse_infos: &A) -> usize
    where
        A: GGLWEInfos,
        Self: GLWESwitchingKeyEncryptSk<BE, Rg>,
    {
        let n: usize = operand_degree(self.n(), &[dense_to_sparse_infos.n()]);
        let subring: usize = BE::scratch_aligned(BE::scalar_znx_bytes_of(n / 2, dense_to_sparse_infos.rank_out().into()));
        subring.max(self.glwe_switching_key_encrypt_sk_tmp_bytes(dense_to_sparse_infos))
    }

    fn glwe_switching_key_encrypt_sk_encapsulation_reference<R, S1, S2>(
        &self,
        dense_to_sparse: &mut R,
        sk_sparse: &mut S2,
        sk_dense: &S1,
        log_gap: usize,
        hamming_weight: usize,
        source_xs: &mut Source,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE, Rg> + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        S1: GLWESecretToBackendRef<BE, Rg> + GetDistribution + GLWEInfos,
        S2: GLWESecretToBackendMut<BE, Rg> + GetDistribution + GetDistributionMut + GLWEInfos,
        Self: GLWESwitchingKeyEncryptSk<BE, Rg> + GLWESecretSampling<BE, Rg>,
    {
        crate::layouts::assert_secret_set(sk_dense.dist());
        assert!(
            sk_sparse.n() == sk_dense.n() && sk_sparse.rank() == sk_dense.rank(),
            "the sparse secret's degree or rank differs from the dense secret's"
        );
        operand_degree(self.n(), &[dense_to_sparse.n()]);
        assert!(
            sk_dense.n() <= dense_to_sparse.n(),
            "the dense secret's degree exceeds the key's"
        );
        assert!(
            dense_to_sparse.rank_in() == sk_dense.rank() && dense_to_sparse.rank_out() == sk_dense.rank(),
            "the key's ranks differ from the secrets' rank"
        );
        let subring_n: usize = u32::try_from(log_gap)
            .ok()
            .and_then(|shift| sk_dense.n().as_usize().checked_shr(shift))
            .unwrap_or(0);
        assert!(subring_n >= 1, "the subring of the gap has no coefficient");
        assert!(hamming_weight <= subring_n, "the Hamming weight exceeds the subring's degree");
        let tmp_bytes: usize =
            GLWESwitchingKeyEncryptSkReference::glwe_switching_key_encrypt_sk_encapsulation_tmp_bytes_reference(
                self,
                dense_to_sparse,
            );
        assert!(
            scratch.available() >= tmp_bytes,
            "scratch.available() < GLWESwitchingKeyEncryptSk::glwe_switching_key_encrypt_sk_encapsulation_tmp_bytes"
        );
        if log_gap == 0 {
            self.glwe_secret_fill_ternary_hw(sk_sparse, hamming_weight, source_xs);
        } else {
            // The secret of the subring, `s(X^(2^log_gap))`: its weight is spent on the subring's coefficients.
            let (mut sk_subring, _) = scratch
                .borrow()
                .take_glwe_secret_scratch::<Rg>(subring_n.into(), sk_dense.rank());
            self.glwe_secret_fill_ternary_hw(&mut sk_subring, hamming_weight, source_xs);
            let sk_subring = GLWESecretToBackendRef::<BE, Rg>::to_backend_ref(&sk_subring);
            let a = scalar_znx_as_vec_znx_backend_ref_from_ref::<BE, Rg>(&sk_subring.data);
            let mut sk_sparse = sk_sparse.to_backend_mut();
            let mut res = scalar_znx_as_vec_znx_backend_mut_from_mut::<BE, Rg>(&mut sk_sparse.data);
            for col in 0..sk_dense.rank().as_usize() {
                self.vec_znx_switch_ring(&mut res, col, &a, col);
            }
        }
        *sk_sparse.dist_mut() = Distribution::ENCAPSULATED("sparse-encapsulation");
        self.glwe_switching_key_encrypt_sk(dense_to_sparse, sk_dense, &*sk_sparse, source_xe, source_xa, scratch);
        scratch.wipe(tmp_bytes);
    }
}
