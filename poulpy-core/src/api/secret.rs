use poulpy_hal::{
    api::{SvpPPolAlloc, SvpPPolBytesOf},
    layouts::{Backend, PrepareHint, ScratchArena},
    source::Source,
};

use crate::{
    GetDistribution, GetDistributionMut,
    dist::Distribution,
    layouts::{
        Degree, GLWEInfos, GLWESecret, GLWESecretLayout, GLWESecretPrepared, GLWESecretPreparedToBackendMut,
        GLWESecretTensorPrepared, GLWESecretTensorPreparedToBackendMut, GLWESecretTensorToBackendMut,
        GLWESecretTensorToBackendRef, GLWESecretToBackendMut, GLWESecretToBackendRef, GetDegree, LWESecret,
        LWESecretToBackendMut, LWESecretToBackendRef, Rank, operand_degree, pairs,
    },
};

/// Secret-key sampling.
///
/// Each distribution is drawn in place by the backend, one seed per column,
/// through [`crate::ScalarZnxFillDistribution`].
/// A fixed `source` therefore gives a per-backend secret, not a cross-backend one.
///
/// Each entry point fills every one of the secret's `rank` polynomials and
/// tags it with the matching [`Distribution`]. See [`Distribution`] for what
/// that tag does and does not describe afterwards.
pub trait GLWESecretSampling<BE: Backend> {
    /// Ternary `{-1, 0, 1}` coefficients, each non-zero with probability `prob`.
    fn glwe_secret_fill_ternary_prob<S>(&self, sk: &mut S, prob: f64, source: &mut Source)
    where
        S: GLWESecretToBackendMut<BE> + GetDistributionMut + GLWEInfos;

    /// Ternary `{-1, 0, 1}` coefficients with exactly `hw` non-zero entries.
    fn glwe_secret_fill_ternary_hw<S>(&self, sk: &mut S, hw: usize, source: &mut Source)
    where
        S: GLWESecretToBackendMut<BE> + GetDistributionMut + GLWEInfos;

    /// Binary `{0, 1}` coefficients, each set with probability `prob`.
    fn glwe_secret_fill_binary_prob<S>(&self, sk: &mut S, prob: f64, source: &mut Source)
    where
        S: GLWESecretToBackendMut<BE> + GetDistributionMut + GLWEInfos;

    /// Binary `{0, 1}` coefficients with exactly `hw` ones.
    fn glwe_secret_fill_binary_hw<S>(&self, sk: &mut S, hw: usize, source: &mut Source)
    where
        S: GLWESecretToBackendMut<BE> + GetDistributionMut + GLWEInfos;

    /// Binary coefficients with at most one `1` per block of `block_size`.
    fn glwe_secret_fill_binary_block<S>(&self, sk: &mut S, block_size: usize, source: &mut Source)
    where
        S: GLWESecretToBackendMut<BE> + GetDistributionMut + GLWEInfos;

    /// All-zero secret, tagged [`Distribution::ZERO`] (debug / testing only).
    fn glwe_secret_fill_zero<S>(&self, sk: &mut S)
    where
        S: GLWESecretToBackendMut<BE> + GetDistributionMut + GLWEInfos;
}

/// Secret-key sampling.
///
/// The LWE counterpart of [`GLWESecretSampling`]:
/// each distribution is drawn in place by the backend through
/// [`crate::ScalarZnxFillDistribution`].
pub trait LWESecretSampling<BE: Backend> {
    /// Ternary `{-1, 0, 1}` coefficients, each non-zero with probability `prob`.
    fn lwe_secret_fill_ternary_prob<S>(&self, sk: &mut S, prob: f64, source: &mut Source)
    where
        S: LWESecretToBackendMut<BE> + GetDistributionMut;

    /// Ternary `{-1, 0, 1}` coefficients with exactly `hw` non-zero entries.
    fn lwe_secret_fill_ternary_hw<S>(&self, sk: &mut S, hw: usize, source: &mut Source)
    where
        S: LWESecretToBackendMut<BE> + GetDistributionMut;

    /// Binary `{0, 1}` coefficients, each set with probability `prob`.
    fn lwe_secret_fill_binary_prob<S>(&self, sk: &mut S, prob: f64, source: &mut Source)
    where
        S: LWESecretToBackendMut<BE> + GetDistributionMut;

    /// Binary `{0, 1}` coefficients with exactly `hw` ones.
    fn lwe_secret_fill_binary_hw<S>(&self, sk: &mut S, hw: usize, source: &mut Source)
    where
        S: LWESecretToBackendMut<BE> + GetDistributionMut;

    /// Binary coefficients with at most one `1` per block of `block_size`.
    fn lwe_secret_fill_binary_block<S>(&self, sk: &mut S, block_size: usize, source: &mut Source)
    where
        S: LWESecretToBackendMut<BE> + GetDistributionMut;

    /// All-zero secret, tagged [`Distribution::ZERO`] (debug / testing only).
    fn lwe_secret_fill_zero<S>(&self, sk: &mut S)
    where
        S: LWESecretToBackendMut<BE> + GetDistributionMut;
}

/// Allocation and preparation of DFT-domain GLWE secrets.
pub trait GLWESecretPreparedFactory<B: Backend>
where
    Self: GetDegree + SvpPPolBytesOf + SvpPPolAlloc<B>,
{
    fn glwe_secret_prepared_alloc(&self, rank: Rank) -> GLWESecretPrepared<B::OwnedBuf, B> {
        self.glwe_secret_prepared_alloc_from_infos(&GLWESecretLayout {
            n: self.ring_degree(),
            rank,
        })
    }
    fn glwe_secret_prepared_alloc_from_infos<A>(&self, infos: &A) -> GLWESecretPrepared<B::OwnedBuf, B>
    where
        A: GLWEInfos,
    {
        let n: usize = operand_degree(self.ring_degree().as_usize(), &[infos.n()]);
        GLWESecretPrepared {
            data: self.svp_ppol_alloc(n, infos.rank().into(), PrepareHint::Reuse),
            dist: Distribution::NONE,
        }
    }

    fn glwe_secret_prepared_bytes_of(&self, rank: Rank) -> usize {
        self.glwe_secret_prepared_bytes_of_from_infos(&GLWESecretLayout {
            n: self.ring_degree(),
            rank,
        })
    }
    fn glwe_secret_prepared_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        let n: usize = operand_degree(self.ring_degree().as_usize(), &[infos.n()]);
        self.bytes_of_svp_ppol(n, infos.rank().into(), PrepareHint::Reuse)
    }

    /// Moves the first `res.rank()` polynomials of `other` to the DFT domain and copies its
    /// [`Distribution`] tag onto `res`; it takes no scratch.
    fn glwe_secret_prepare<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWESecretPreparedToBackendMut<B> + GetDistributionMut,
        O: GLWESecretToBackendRef<B> + GetDistribution;
}

/// Allocation and preparation of DFT-domain GLWE secret tensors.
pub trait GLWESecretTensorPreparedFactory<B: Backend>
where
    Self: GetDegree + SvpPPolBytesOf + SvpPPolAlloc<B>,
{
    fn glwe_secret_tensor_prepared_alloc(&self, rank: Rank) -> GLWESecretTensorPrepared<B::OwnedBuf, B> {
        self.glwe_secret_tensor_prepared_alloc_from_infos(&GLWESecretLayout {
            n: self.ring_degree(),
            rank,
        })
    }
    fn glwe_secret_tensor_prepared_alloc_from_infos<A>(&self, infos: &A) -> GLWESecretTensorPrepared<B::OwnedBuf, B>
    where
        A: GLWEInfos,
    {
        let n: usize = operand_degree(self.ring_degree().as_usize(), &[infos.n()]);
        GLWESecretTensorPrepared {
            data: self.svp_ppol_alloc(n, pairs(infos.rank().into()), PrepareHint::Reuse),
            rank: infos.rank(),
            dist: Distribution::NONE,
        }
    }

    fn glwe_secret_tensor_prepared_bytes_of(&self, rank: Rank) -> usize {
        self.glwe_secret_tensor_prepared_bytes_of_from_infos(&GLWESecretLayout {
            n: self.ring_degree(),
            rank,
        })
    }
    fn glwe_secret_tensor_prepared_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        let n: usize = operand_degree(self.ring_degree().as_usize(), &[infos.n()]);
        self.bytes_of_svp_ppol(n, pairs(infos.rank().into()), PrepareHint::Reuse)
    }

    /// Moves a [`GLWESecretTensor`](crate::layouts::GLWESecretTensor) to the
    /// DFT domain and copies its [`Distribution`] tag onto `res`; it takes no scratch.
    ///
    /// Only a tensor secret is accepted: preparing a base
    /// [`GLWESecret`](crate::layouts::GLWESecret) is
    /// [`glwe_secret_prepare`](GLWESecretPreparedFactory::glwe_secret_prepare),
    /// and the two are not interchangeable (they do not even have the same
    /// number of columns).
    ///
    /// # Panics
    /// Before writing `res`, when the ranks or the column counts of `res` and `other` differ.
    fn glwe_secret_tensor_prepared_prepare<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWESecretTensorPreparedToBackendMut<B> + GetDistributionMut,
        O: GLWESecretTensorToBackendRef<B> + GetDistribution;
}

/// Tensoring of a GLWE secret with itself.
pub trait GLWESecretTensorFactory<BE: Backend> {
    /// Scratch bytes of [`Self::glwe_secret_tensor_prepare`] for a base secret of rank `rank`.
    fn glwe_secret_tensor_prepare_tmp_bytes(&self, rank: Rank) -> usize;

    /// Fills `res` with the pairwise products `s_i * s_j` of the base secret
    /// `other` and wipes the scratch it reports.
    ///
    /// `res` inherits `other`'s [`Distribution`] tag. That tag keeps
    /// describing the base secret: the products written into `res` follow a
    /// different (derived) distribution, which no tag variant encodes.
    ///
    /// # Panics
    /// Before writing `res`, when the ranks or degrees of `res` and `other` differ, a degree
    /// exceeds the module's, or the scratch is smaller than the query.
    fn glwe_secret_tensor_prepare<R, O>(&self, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWESecretTensorToBackendMut<BE> + GetDistributionMut + GLWEInfos,
        O: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos;
}

/// Conversions between LWE and GLWE secrets.
pub trait SecretConversion<B: Backend> {
    /// Derives the associated rank-1 `GLWESecret` from a `LWESecret` by applying
    /// the X → X⁻¹ automorphism (k = -1). The result is the GLWE polynomial key
    /// whose ring product with a mask decrypts LWE ciphertexts produced by
    /// `glwe_expand_lwe`.
    ///
    /// The source's [`Distribution`] tag is copied unchanged: the automorphism
    /// only permutes and negates coefficients, so the base secret it describes
    /// is the same one.
    ///
    /// # Panics
    /// When the degree of `src` differs from the module degree.
    fn glwe_secret_from_lwe_secret<S>(&self, src: &S) -> GLWESecret<B::OwnedBuf, B::ZnxWord>
    where
        S: LWESecretToBackendRef<B>;

    /// Scratch bytes of [`Self::lwe_secret_from_glwe_secret`] for a source GLWE secret of layout `infos`.
    fn lwe_secret_from_glwe_secret_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Derives an `LWESecret` of the requested flat dimension `lwe_n` from a
    /// `GLWESecret` by applying the X → X⁻¹ automorphism (k = -1) to each rank
    /// component and packing the results as consecutive `n`-coefficient blocks.
    ///
    /// `lwe_n` must satisfy `lwe_n ≤ rank * n` (the GLWE rank must cover the
    /// requested LWE dimension); the last block may be a partial slice when
    /// `lwe_n % n != 0`. The inverse relation `rank = ceil(lwe_n / n)` lets a
    /// caller size the GLWE just enough for a target LWE dimension.
    ///
    /// For `lwe_n == n` and `rank == 1`, this is the inverse of
    /// `glwe_secret_from_lwe_secret`: applying both conversions recovers the
    /// original key.
    ///
    /// The source's [`Distribution`] tag is copied unchanged: the automorphism
    /// only permutes and negates coefficients, and the packing only relocates
    /// them, so the base secret described by the tag is the same one. In
    /// particular, fixed-weight metadata denotes the advertised fixed-weight
    /// distribution of each polynomial component of the source key and is not
    /// multiplied by the rank during flattening.
    ///
    /// The rotated components are carved from `scratch`, which is wiped over
    /// [`Self::lwe_secret_from_glwe_secret_tmp_bytes`] before returning.
    ///
    /// # Panics
    /// When the degree of `src` differs from the module degree, `lwe_n > rank * n` or the scratch
    /// is smaller than the query.
    fn lwe_secret_from_glwe_secret<S>(
        &self,
        src: &S,
        lwe_n: Degree,
        scratch: &mut ScratchArena<'_, B>,
    ) -> LWESecret<B::OwnedBuf, B::ZnxWord>
    where
        S: GLWESecretToBackendRef<B>;
}
