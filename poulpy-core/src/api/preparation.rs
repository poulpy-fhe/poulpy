use std::marker::PhantomData;

use poulpy_hal::{
    api::{VecZnxDftAlloc, VecZnxDftBytesOf, VmpPMatAlloc, VmpPMatBytesOf},
    layouts::{Backend, Data, PrepareHint, ScratchArena},
};

use crate::{
    GetDistribution, GetDistributionMut,
    dist::Distribution,
    layouts::{
        Base2K, Degree, Dnum, GGLWEInfos, GGLWELayout, GGLWEPrepared, GGLWEPreparedToBackendMut, GGLWEToBackendRef,
        GGLWEToGGSWKeyPrepared, GGLWEToGGSWKeyPreparedToBackendMut, GGLWEToGGSWKeyToBackendRef, GGSWInfos, GGSWLayout,
        GGSWPrepared, GGSWPreparedToBackendMut, GGSWToBackendRef, GLWEAutomorphismKeyPrepared, GLWECIEmbedKeyPrepared, GLWECIKey,
        GLWECIKeyPrepared, GLWECITraceKeyPrepared, GLWEInfos, GLWELayout, GLWEPrepared, GLWEPreparedToBackendMut,
        GLWEPublicKeyPrepared, GLWEPublicKeyPreparedToBackendMut, GLWEPublicKeyToBackendRef, GLWESwitchingKey,
        GLWESwitchingKeyDegrees, GLWESwitchingKeyDegreesMut, GLWESwitchingKeyPrepared, GLWETensorKeyPrepared, GLWEToBackendRef,
        GLWEToLWEKeyPrepared, GetDegree, GetGaloisElement, LWESwitchingKeyPrepared, LWEToGLWEKeyPrepared, Rank, SetGaloisElement,
        TorusPrecision, operand_degree,
    },
};

/// Trait for allocating and preparing DFT-domain GLWE ciphertexts.
pub trait GLWEPreparedFactory<BE: Backend>
where
    Self: GetDegree + VecZnxDftAlloc<BE> + VecZnxDftBytesOf,
{
    /// Allocates a new prepared GLWE with the given parameters.
    fn glwe_prepared_alloc(&self, base2k: Base2K, k: TorusPrecision, rank: Rank) -> GLWEPrepared<BE::OwnedBuf, BE> {
        self.glwe_prepared_alloc_from_infos(&GLWELayout {
            n: self.ring_degree(),
            base2k,
            k,
            rank,
        })
    }

    fn glwe_prepared_alloc_from_infos<A>(&self, infos: &A) -> GLWEPrepared<BE::OwnedBuf, BE>
    where
        A: GLWEInfos,
    {
        let n: usize = operand_degree(self.ring_degree().as_usize(), &[infos.n()]);
        GLWEPrepared {
            noise: None,
            data: self.vec_znx_dft_alloc(n, (infos.rank() + 1).into(), infos.size()),
            base2k: infos.base2k(),
            k: infos.k(),
        }
    }

    fn glwe_prepared_bytes_of(&self, base2k: Base2K, k: TorusPrecision, rank: Rank) -> usize {
        self.glwe_prepared_bytes_of_from_infos(&GLWELayout {
            n: self.ring_degree(),
            base2k,
            k,
            rank,
        })
    }

    fn glwe_prepared_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        let n: usize = operand_degree(self.ring_degree().as_usize(), &[infos.n()]);
        self.bytes_of_vec_znx_dft(n, (infos.rank() + 1).into(), infos.size())
    }

    fn glwe_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEPreparedToBackendMut<BE>,
        O: GLWEToBackendRef<BE> + GLWEInfos;
}

/// Factory trait for allocating and preparing [`GGLWEPrepared`] instances.
pub trait GGLWEPreparedFactory<BE: Backend>
where
    Self: GetDegree + VmpPMatAlloc<BE> + VmpPMatBytesOf,
{
    /// Allocates a new [`GGLWEPrepared`] with the given parameters.
    fn gglwe_prepared_alloc(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        k_digit: TorusPrecision,
        k_aux: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
    ) -> GGLWEPrepared<BE::OwnedBuf, BE> {
        self.gglwe_prepared_alloc_from_infos(&GGLWELayout {
            n: self.ring_degree(),
            base2k,
            dnum,
            k_aux,
            rank_in,
            rank_out,
            k_digit,
            k_offset: TorusPrecision(0),
            stride: 1,
        })
    }

    /// Allocates a new [`GGLWEPrepared`] matching the parameters of `infos`.
    fn gglwe_prepared_alloc_from_infos<A>(&self, infos: &A) -> GGLWEPrepared<BE::OwnedBuf, BE>
    where
        A: GGLWEInfos,
    {
        let n: usize = operand_degree(self.ring_degree().as_usize(), &[infos.n()]);
        let size: usize =
            crate::layouts::offset_key_size(infos.base2k(), infos.k_offset(), infos.dnum(), infos.k_digit(), infos.k_aux());
        GGLWEPrepared {
            noise: None,
            data: self.vmp_pmat_alloc(
                n,
                infos.dnum().into(),
                infos.rank_in().into(),
                (infos.rank_out() + 1).into(),
                size,
                PrepareHint::Reuse,
            ),
            base2k: infos.base2k(),
            k_digit: infos.k_digit(),
            k_offset: infos.k_offset(),
            k_aux: infos.k_aux(),
            dnum: infos.dnum(),
            stride: 1,
        }
    }

    /// Returns the byte size required to store a [`GGLWEPrepared`] with the given parameters.
    fn gglwe_prepared_bytes_of(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        k_digit: TorusPrecision,
        k_aux: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
    ) -> usize {
        self.gglwe_prepared_bytes_of_from_infos(&GGLWELayout {
            n: self.ring_degree(),
            base2k,
            dnum,
            k_aux,
            rank_in,
            rank_out,
            k_digit,
            k_offset: TorusPrecision(0),
            stride: 1,
        })
    }

    /// Returns the byte size required to store a [`GGLWEPrepared`] matching `infos`.
    fn gglwe_prepared_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        let n: usize = operand_degree(self.ring_degree().as_usize(), &[infos.n()]);
        let size: usize =
            crate::layouts::offset_key_size(infos.base2k(), infos.k_offset(), infos.dnum(), infos.k_digit(), infos.k_aux());
        self.bytes_of_vmp_pmat(
            n,
            infos.dnum().into(),
            infos.rank_in().into(),
            (infos.rank_out() + 1).into(),
            size,
            PrepareHint::Reuse,
        )
    }

    /// Returns the scratch-space bytes needed by [`gglwe_prepare`](Self::gglwe_prepare).
    fn gglwe_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    /// Transforms a standard [`GGLWE`](crate::layouts::GGLWE) into the DFT domain, writing the result into `res`.
    ///
    /// Both `res` and `other` must share the same ring degree, base2k, precision, and digit width.
    fn gglwe_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEPreparedToBackendMut<BE>,
        O: GGLWEToBackendRef<BE>;
}

/// Trait for allocating and preparing DFT-domain GGSW ciphertexts.
pub trait GGSWPreparedFactory<BE: Backend>
where
    Self: GetDegree + VmpPMatAlloc<BE> + VmpPMatBytesOf,
{
    /// Allocates a new prepared GGSW with the given parameters.
    fn ggsw_prepared_alloc(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        k_digit: TorusPrecision,
        k_aux: TorusPrecision,
        rank: Rank,
    ) -> GGSWPrepared<BE::OwnedBuf, BE> {
        self.ggsw_prepared_alloc_from_infos(&GGSWLayout {
            n: self.ring_degree(),
            base2k,
            dnum,
            k_aux,
            rank,
            k_digit,
        })
    }

    fn ggsw_prepared_alloc_from_infos<A>(&self, infos: &A) -> GGSWPrepared<BE::OwnedBuf, BE>
    where
        A: GGSWInfos,
    {
        let n: usize = operand_degree(self.ring_degree().as_usize(), &[infos.n()]);
        let size: usize = crate::layouts::key_size(infos.base2k(), infos.dnum(), infos.k_digit(), infos.k_aux());
        GGSWPrepared {
            noise: None,
            data: self.vmp_pmat_alloc(
                n,
                infos.dnum().into(),
                (infos.rank() + 1).into(),
                (infos.rank() + 1).into(),
                size,
                PrepareHint::Reuse,
            ),
            base2k: infos.base2k(),
            k_digit: infos.k_digit(),
            k_aux: infos.k_aux(),
        }
    }

    fn ggsw_prepared_bytes_of(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        k_digit: TorusPrecision,
        k_aux: TorusPrecision,
        rank: Rank,
    ) -> usize {
        self.ggsw_prepared_bytes_of_from_infos(&GGSWLayout {
            n: self.ring_degree(),
            base2k,
            dnum,
            k_aux,
            rank,
            k_digit,
        })
    }

    fn ggsw_prepared_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        let n: usize = operand_degree(self.ring_degree().as_usize(), &[infos.n()]);
        let size: usize = crate::layouts::key_size(infos.base2k(), infos.dnum(), infos.k_digit(), infos.k_aux());
        self.bytes_of_vmp_pmat(
            n,
            infos.dnum().into(),
            (infos.rank() + 1).into(),
            (infos.rank() + 1).into(),
            size,
            PrepareHint::Reuse,
        )
    }

    fn ggsw_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos;

    fn ggsw_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGSWPreparedToBackendMut<BE>,
        O: GGSWToBackendRef<BE>;

    fn ggsw_zero<R>(&self, res: &mut R)
    where
        R: GGSWPreparedToBackendMut<BE>;
}

pub trait GLWEPublicKeyPreparedFactory<BE: Backend>
where
    Self: GetDegree + VmpPMatAlloc<BE> + VmpPMatBytesOf,
{
    fn glwe_public_key_prepared_alloc(
        &self,
        base2k: Base2K,
        k: TorusPrecision,
        rank: Rank,
    ) -> GLWEPublicKeyPrepared<BE::OwnedBuf, BE> {
        self.glwe_public_key_prepared_alloc_from_infos(&GLWELayout {
            n: self.ring_degree(),
            base2k,
            k,
            rank,
        })
    }

    fn glwe_public_key_prepared_alloc_from_infos<A>(&self, infos: &A) -> GLWEPublicKeyPrepared<BE::OwnedBuf, BE>
    where
        A: GLWEInfos,
    {
        let rank: Rank = infos.rank();
        assert!(rank.as_usize() >= 1, "invalid public key: rank must be at least 1");
        let n: usize = operand_degree(self.ring_degree().as_usize(), &[infos.n()]);
        GLWEPublicKeyPrepared {
            noise: None,
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
        A: GLWEInfos;

    fn glwe_public_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEPublicKeyPreparedToBackendMut<BE> + GetDistributionMut,
        O: GLWEPublicKeyToBackendRef<BE> + GetDistribution;
}

pub trait GLWEAutomorphismKeyPreparedFactory<BE: Backend>
where
    Self: GGLWEPreparedFactory<BE>,
{
    fn glwe_automorphism_key_prepared_alloc(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        k_digit: TorusPrecision,
        k_aux: TorusPrecision,
        rank: Rank,
    ) -> GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE> {
        GLWEAutomorphismKeyPrepared::<BE::OwnedBuf, BE> {
            key: self.gglwe_prepared_alloc(base2k, dnum, k_digit, k_aux, rank, rank),
            p: 0,
        }
    }

    fn glwe_automorphism_key_prepared_alloc_from_infos<A>(&self, infos: &A) -> GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for AutomorphismKeyPrepared"
        );
        GLWEAutomorphismKeyPrepared::<BE::OwnedBuf, BE> {
            key: self.gglwe_prepared_alloc_from_infos(infos),
            p: 0,
        }
    }

    fn glwe_automorphism_key_prepared_bytes_of(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        k_digit: TorusPrecision,
        k_aux: TorusPrecision,
        rank: Rank,
    ) -> usize {
        self.gglwe_prepared_bytes_of(base2k, dnum, k_digit, k_aux, rank, rank)
    }

    fn glwe_automorphism_key_prepared_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for AutomorphismKeyPrepared"
        );
        self.gglwe_prepared_bytes_of_from_infos(infos)
    }

    fn glwe_automorphism_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_automorphism_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEPreparedToBackendMut<BE> + SetGaloisElement,
        O: GGLWEToBackendRef<BE> + GetGaloisElement;
}

pub trait GLWESwitchingKeyPreparedFactory<BE: Backend>
where
    Self: GGLWEPreparedFactory<BE>,
{
    fn glwe_switching_key_prepared_alloc(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        k_digit: TorusPrecision,
        k_aux: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
    ) -> GLWESwitchingKeyPrepared<BE::OwnedBuf, BE> {
        GLWESwitchingKeyPrepared::<BE::OwnedBuf, BE> {
            key: self.gglwe_prepared_alloc(base2k, dnum, k_digit, k_aux, rank_in, rank_out),
            input_degree: Degree(0),
            output_degree: Degree(0),
        }
    }

    fn glwe_switching_key_prepared_alloc_from_infos<A>(&self, infos: &A) -> GLWESwitchingKeyPrepared<BE::OwnedBuf, BE>
    where
        A: GGLWEInfos,
    {
        GLWESwitchingKeyPrepared::<BE::OwnedBuf, BE> {
            key: self.gglwe_prepared_alloc_from_infos(infos),
            input_degree: Degree(0),
            output_degree: Degree(0),
        }
    }

    fn bytes_of_glwe_key_prepared(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        k_digit: TorusPrecision,
        k_aux: TorusPrecision,
        rank_in: Rank,
        rank_out: Rank,
    ) -> usize {
        self.gglwe_prepared_bytes_of(base2k, dnum, k_digit, k_aux, rank_in, rank_out)
    }

    fn glwe_switching_key_prepared_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        self.gglwe_prepared_bytes_of_from_infos(infos)
    }

    fn glwe_switching_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_switching_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEPreparedToBackendMut<BE> + GLWESwitchingKeyDegreesMut,
        O: GGLWEToBackendRef<BE> + GLWESwitchingKeyDegrees;
}

pub trait GLWETensorKeyPreparedFactory<BE: Backend>
where
    Self: GGLWEPreparedFactory<BE>,
{
    fn alloc_tensor_key_prepared(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        k_digit: TorusPrecision,
        k_aux: TorusPrecision,
        rank: Rank,
    ) -> GLWETensorKeyPrepared<BE::OwnedBuf, BE> {
        let pairs: u32 = (((rank.as_u32() + 1) * rank.as_u32()) >> 1).max(1);
        GLWETensorKeyPrepared(self.gglwe_prepared_alloc(base2k, dnum, k_digit, k_aux, Rank(pairs), rank))
    }

    fn alloc_tensor_key_prepared_from_infos<A>(&self, infos: &A) -> GLWETensorKeyPrepared<BE::OwnedBuf, BE>
    where
        A: GGLWEInfos,
    {
        GLWETensorKeyPrepared(self.gglwe_prepared_alloc_from_infos(&tensor_key_layout(infos)))
    }

    fn bytes_of_tensor_key_prepared(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        k_digit: TorusPrecision,
        k_aux: TorusPrecision,
        rank: Rank,
    ) -> usize {
        let pairs: u32 = (((rank.as_u32() + 1) * rank.as_u32()) >> 1).max(1);
        self.gglwe_prepared_bytes_of(base2k, dnum, k_digit, k_aux, Rank(pairs), rank)
    }

    fn bytes_of_tensor_key_prepared_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        self.gglwe_prepared_bytes_of_from_infos(&tensor_key_layout(infos))
    }

    fn glwe_tensor_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_tensor_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEPreparedToBackendMut<BE>,
        O: GGLWEToBackendRef<BE>;
}

/// The GGLWE layout of a tensor key like `infos`: one row per pair of secret columns.
fn tensor_key_layout<A: GGLWEInfos>(infos: &A) -> GGLWELayout {
    let rank = infos.rank_out();
    GGLWELayout {
        n: infos.n(),
        base2k: infos.base2k(),
        dnum: infos.dnum(),
        k_aux: infos.k_aux(),
        rank_in: Rank((((rank.as_u32() + 1) * rank.as_u32()) >> 1).max(1)),
        rank_out: rank,
        k_digit: infos.k_digit(),
        k_offset: infos.k_offset(),
        stride: 1,
    }
}

pub trait GLWEToLWEKeyPreparedFactory<BE: Backend>
where
    Self: GLWESwitchingKeyPreparedFactory<BE>,
{
    fn glwe_to_lwe_key_prepared_alloc(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        k_aux: TorusPrecision,
        rank_in: Rank,
    ) -> GLWEToLWEKeyPrepared<BE::OwnedBuf, BE> {
        GLWEToLWEKeyPrepared(self.glwe_switching_key_prepared_alloc(
            base2k,
            dnum,
            TorusPrecision::from(base2k),
            k_aux,
            rank_in,
            Rank(1),
        ))
    }
    fn glwe_to_lwe_key_prepared_alloc_from_infos<A>(&self, infos: &A) -> GLWEToLWEKeyPrepared<BE::OwnedBuf, BE>
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(
            infos.rank_out().0,
            1,
            "rank_out > 1 is not supported for GLWEToLWEKeyPrepared"
        );
        debug_assert_eq!(
            infos.k_digit().0,
            infos.base2k().0,
            "GLWEToLWEKeyPrepared uses one-limb digits: k_digit must equal base2k"
        );
        GLWEToLWEKeyPrepared(self.glwe_switching_key_prepared_alloc_from_infos(infos))
    }

    fn glwe_to_lwe_key_prepared_bytes_of(&self, base2k: Base2K, dnum: Dnum, k_aux: TorusPrecision, rank_in: Rank) -> usize {
        self.bytes_of_glwe_key_prepared(base2k, dnum, TorusPrecision::from(base2k), k_aux, rank_in, Rank(1))
    }

    fn glwe_to_lwe_key_prepared_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(
            infos.rank_out().0,
            1,
            "rank_out > 1 is not supported for GLWEToLWEKeyPrepared"
        );
        debug_assert_eq!(
            infos.k_digit().0,
            infos.base2k().0,
            "GLWEToLWEKeyPrepared uses one-limb digits: k_digit must equal base2k"
        );
        self.glwe_switching_key_prepared_bytes_of_from_infos(infos)
    }

    fn glwe_to_lwe_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_to_lwe_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEPreparedToBackendMut<BE> + GLWESwitchingKeyDegreesMut,
        O: GGLWEToBackendRef<BE> + GLWESwitchingKeyDegrees;
}

pub trait LWESwitchingKeyPreparedFactory<BE: Backend>
where
    Self: GLWESwitchingKeyPreparedFactory<BE>,
{
    fn lwe_switching_key_prepared_alloc(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        k_aux: TorusPrecision,
    ) -> LWESwitchingKeyPrepared<BE::OwnedBuf, BE> {
        LWESwitchingKeyPrepared(self.glwe_switching_key_prepared_alloc(
            base2k,
            dnum,
            TorusPrecision::from(base2k),
            k_aux,
            Rank(1),
            Rank(1),
        ))
    }

    fn lwe_switching_key_prepared_alloc_from_infos<A>(&self, infos: &A) -> LWESwitchingKeyPrepared<BE::OwnedBuf, BE>
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(
            infos.k_digit().0,
            infos.base2k().0,
            "LWESwitchingKey uses one-limb digits: k_digit must equal base2k"
        );
        debug_assert_eq!(infos.rank_in().0, 1, "rank_in > 1 is not supported for LWESwitchingKey");
        debug_assert_eq!(infos.rank_out().0, 1, "rank_out > 1 is not supported for LWESwitchingKey");
        LWESwitchingKeyPrepared(self.glwe_switching_key_prepared_alloc_from_infos(infos))
    }

    fn lwe_switching_key_prepared_bytes_of(&self, base2k: Base2K, dnum: Dnum, k_aux: TorusPrecision) -> usize {
        self.bytes_of_glwe_key_prepared(base2k, dnum, TorusPrecision::from(base2k), k_aux, Rank(1), Rank(1))
    }

    fn lwe_switching_key_prepared_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(
            infos.k_digit().0,
            infos.base2k().0,
            "LWESwitchingKey uses one-limb digits: k_digit must equal base2k"
        );
        debug_assert_eq!(infos.rank_in().0, 1, "rank_in > 1 is not supported for LWESwitchingKey");
        debug_assert_eq!(infos.rank_out().0, 1, "rank_out > 1 is not supported for LWESwitchingKey");
        self.glwe_switching_key_prepared_bytes_of_from_infos(infos)
    }

    fn lwe_switching_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn lwe_switching_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEPreparedToBackendMut<BE> + GLWESwitchingKeyDegreesMut,
        O: GGLWEToBackendRef<BE> + GLWESwitchingKeyDegrees;
}

pub trait LWEToGLWEKeyPreparedFactory<BE: Backend>
where
    Self: GLWESwitchingKeyPreparedFactory<BE>,
{
    fn lwe_to_glwe_key_prepared_alloc(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        k_aux: TorusPrecision,
        rank_out: Rank,
    ) -> LWEToGLWEKeyPrepared<BE::OwnedBuf, BE> {
        LWEToGLWEKeyPrepared(self.glwe_switching_key_prepared_alloc(
            base2k,
            dnum,
            TorusPrecision::from(base2k),
            k_aux,
            Rank(1),
            rank_out,
        ))
    }
    fn lwe_to_glwe_key_prepared_alloc_from_infos<A>(&self, infos: &A) -> LWEToGLWEKeyPrepared<BE::OwnedBuf, BE>
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(infos.rank_in().0, 1, "rank_in > 1 is not supported for LWEToGLWEKey");
        debug_assert_eq!(
            infos.k_digit().0,
            infos.base2k().0,
            "LWEToGLWEKey uses one-limb digits: k_digit must equal base2k"
        );
        LWEToGLWEKeyPrepared(self.glwe_switching_key_prepared_alloc_from_infos(infos))
    }

    fn lwe_to_glwe_key_prepared_bytes_of(&self, base2k: Base2K, dnum: Dnum, k_aux: TorusPrecision, rank_out: Rank) -> usize {
        self.bytes_of_glwe_key_prepared(base2k, dnum, TorusPrecision::from(base2k), k_aux, Rank(1), rank_out)
    }

    fn lwe_to_glwe_key_prepared_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(infos.rank_in().0, 1, "rank_in > 1 is not supported for LWEToGLWEKey");
        debug_assert_eq!(
            infos.k_digit().0,
            infos.base2k().0,
            "LWEToGLWEKey uses one-limb digits: k_digit must equal base2k"
        );
        self.glwe_switching_key_prepared_bytes_of_from_infos(infos)
    }

    fn lwe_to_glwe_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn lwe_to_glwe_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEPreparedToBackendMut<BE> + GLWESwitchingKeyDegreesMut,
        O: GGLWEToBackendRef<BE> + GLWESwitchingKeyDegrees;
}

/// Factory trait for allocating and preparing [`GGLWEToGGSWKeyPrepared`] instances.
pub trait GGLWEToGGSWKeyPreparedFactory<BE: Backend>
where
    Self: GGLWEPreparedFactory<BE>,
{
    /// Allocates a new [`GGLWEToGGSWKeyPrepared`] matching the parameters of `infos`.
    ///
    /// # Panics
    ///
    /// If `rank_in != rank_out`.
    fn gglwe_to_ggsw_key_prepared_alloc_from_infos<A>(&self, infos: &A) -> GGLWEToGGSWKeyPrepared<BE::OwnedBuf, BE>
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWEToGGSWKeyPrepared"
        );
        GGLWEToGGSWKeyPrepared {
            keys: (0..infos.rank().as_usize())
                .map(|_| self.gglwe_prepared_alloc_from_infos(infos))
                .collect(),
        }
    }

    /// Allocates a new [`GGLWEToGGSWKeyPrepared`] with explicit parameters.
    ///
    /// Creates `rank` prepared GGLWE matrices, one per secret-key component.
    fn gglwe_to_ggsw_key_prepared_alloc(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        k_digit: TorusPrecision,
        k_aux: TorusPrecision,
        rank: Rank,
    ) -> GGLWEToGGSWKeyPrepared<BE::OwnedBuf, BE> {
        GGLWEToGGSWKeyPrepared {
            keys: (0..rank.as_usize())
                .map(|_| self.gglwe_prepared_alloc(base2k, dnum, k_digit, k_aux, rank, rank))
                .collect(),
        }
    }

    /// Returns the byte size required to store a [`GGLWEToGGSWKeyPrepared`] matching `infos`.
    fn bytes_of_gglwe_to_ggsw_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(
            infos.rank_in(),
            infos.rank_out(),
            "rank_in != rank_out is not supported for GGLWEToGGSWKeyPrepared"
        );
        infos.rank().as_usize() * self.gglwe_prepared_bytes_of_from_infos(infos)
    }

    /// Returns the byte size required to store a [`GGLWEToGGSWKeyPrepared`] with explicit parameters.
    fn bytes_of_gglwe_to_ggsw(
        &self,
        base2k: Base2K,
        dnum: Dnum,
        k_digit: TorusPrecision,
        k_aux: TorusPrecision,
        rank: Rank,
    ) -> usize {
        rank.as_usize() * self.gglwe_prepared_bytes_of(base2k, dnum, k_digit, k_aux, rank, rank)
    }

    /// Returns the scratch-space bytes needed by [`gglwe_to_ggsw_key_prepare`](Self::gglwe_to_ggsw_key_prepare).
    fn gglwe_to_ggsw_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    /// Transforms a standard [`GGLWEToGGSWKey`](crate::layouts::GGLWEToGGSWKey) into the DFT domain, writing into `res`.
    ///
    /// Iterates over each key element and prepares it individually.
    fn gglwe_to_ggsw_key_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GGLWEToGGSWKeyPreparedToBackendMut<BE>,
        O: GGLWEToGGSWKeyToBackendRef<BE>;
}

pub trait GLWECIKeyPreparedFactory<BE: Backend>
where
    Self: GLWESwitchingKeyPreparedFactory<BE>,
{
    fn glwe_ci_embed_key_prepared_alloc_from_infos<A>(&self, infos: &A) -> GLWECIEmbedKeyPrepared<BE::OwnedBuf, BE>
    where
        A: GGLWEInfos,
    {
        GLWECIKeyPrepared(self.glwe_switching_key_prepared_alloc_from_infos(infos), PhantomData)
    }

    fn glwe_ci_trace_key_prepared_alloc_from_infos<A>(&self, infos: &A) -> GLWECITraceKeyPrepared<BE::OwnedBuf, BE>
    where
        A: GGLWEInfos,
    {
        GLWECIKeyPrepared(self.glwe_switching_key_prepared_alloc_from_infos(infos), PhantomData)
    }

    fn glwe_ci_key_prepared_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        self.glwe_switching_key_prepared_bytes_of_from_infos(infos)
    }

    fn glwe_ci_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    /// Prepares `other` into `res`; both carry the same direction.
    fn glwe_ci_key_prepare<DR, DO, M>(
        &self,
        res: &mut GLWECIKeyPrepared<DR, BE, M>,
        other: &GLWECIKey<DO, BE::ZnxWord, M>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        DR: Data,
        DO: Data,
        GLWESwitchingKeyPrepared<DR, BE>: GGLWEPreparedToBackendMut<BE>,
        GLWESwitchingKey<DO, BE::ZnxWord>: GGLWEToBackendRef<BE>;
}
