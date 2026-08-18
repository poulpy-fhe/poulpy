use poulpy_hal::layouts::ZnxWord;
use poulpy_hal::{
    api::{
        ScalarZnxAutomorphismBackend, ScalarZnxFillBinaryBlockSourceBackend, ScalarZnxFillBinaryHwSourceBackend,
        ScalarZnxFillBinaryProbSourceBackend, ScalarZnxFillTernaryHwSourceBackend, ScalarZnxFillTernaryProbSourceBackend,
        VecZnxCopyRangeBackend, VecZnxZeroBackend,
    },
    layouts::{
        Backend, Data, HostDataMut, Module, ScalarZnx, ScalarZnxToBackendMut, ScalarZnxToBackendRef, ZnxViewMut,
        scalar_znx_as_vec_znx_backend_mut_from_mut, scalar_znx_as_vec_znx_backend_ref_from_mut,
    },
    oep::HalVecZnxImpl,
    source::Source,
};

use crate::{
    GetDistribution, GetDistributionMut,
    dist::Distribution,
    layouts::{Base2K, Degree, GLWEInfos, LWEInfos, LWESecretToBackendMut, Rank},
};

use super::{
    ModuleCoreAlloc,
    lwe_secret::{LWESecret, LWESecretToBackendRef},
};

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct GLWESecretLayout {
    pub n: Degree,
    pub rank: Rank,
}

impl LWEInfos for GLWESecretLayout {
    fn base2k(&self) -> Base2K {
        Base2K(0)
    }

    fn n(&self) -> Degree {
        self.n
    }

    fn max_size(&self) -> usize {
        unimplemented!("this method is not defined for secrets)")
    }

    fn k(&self) -> super::TorusPrecision {
        unimplemented!("this method is not defined for secrets")
    }
}
impl GLWEInfos for GLWESecretLayout {
    fn rank(&self) -> Rank {
        self.rank
    }
}

#[derive(PartialEq, Eq, Clone)]
pub struct GLWESecret<D: Data, W: ZnxWord> {
    pub(crate) data: ScalarZnx<D, W>,
    /// Distribution the base secret was sampled from, shared by all `rank`
    /// polynomial components. See [`Distribution`] for how this tag is
    /// propagated: it survives automorphisms, LWE/GLWE conversions, DFT
    /// preparation and backend transfers, and is *not* redefined for
    /// derived products such as [`GLWESecretTensor`](super::GLWESecretTensor).
    pub(crate) dist: Distribution,
}

pub type GLWESecretBackendRef<'a, BE> = GLWESecret<<BE as Backend>::BufRef<'a>, <BE as Backend>::ZnxWord>;
pub type GLWESecretBackendMut<'a, BE> = GLWESecret<<BE as Backend>::BufMut<'a>, <BE as Backend>::ZnxWord>;

impl<D: Data, W: ZnxWord> LWEInfos for GLWESecret<D, W> {
    fn base2k(&self) -> Base2K {
        Base2K(0)
    }

    fn n(&self) -> Degree {
        Degree(self.data.n() as u32)
    }

    fn max_size(&self) -> usize {
        1
    }

    fn k(&self) -> super::TorusPrecision {
        unimplemented!("this method is not defined for secrets")
    }
}

impl<D: Data, W: ZnxWord> GetDistribution for GLWESecret<D, W> {
    fn dist(&self) -> &Distribution {
        &self.dist
    }
}

impl<D: Data, W: ZnxWord> GetDistributionMut for GLWESecret<D, W> {
    fn dist_mut(&mut self) -> &mut Distribution {
        &mut self.dist
    }
}

impl<D: Data, W: ZnxWord> GLWEInfos for GLWESecret<D, W> {
    fn rank(&self) -> Rank {
        Rank(self.data.cols() as u32)
    }
}

impl<D: Data, W: ZnxWord> GLWESecret<D, W> {
    pub fn data(&self) -> &ScalarZnx<D, W> {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut ScalarZnx<D, W> {
        &mut self.data
    }

    /// Zero-cost rename when both backends share the same `OwnedBuf`.
    pub fn reinterpret<To>(self) -> GLWESecret<To::OwnedBuf, To::ZnxWord>
    where
        To: Backend<OwnedBuf = D, ZnxWord = W>,
    {
        let n = self.data.n();
        let cols = self.data.cols();
        let data = self.data.data;
        GLWESecret {
            data: ScalarZnx::from_data(data, n, cols),
            dist: self.dist,
        }
    }
}

#[expect(
    dead_code,
    reason = "host-owned constructors are kept for serialization and host-only staging"
)]
impl<W: ZnxWord> GLWESecret<Vec<u8>, W> {
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        Self::alloc(infos.n(), infos.rank())
    }

    pub(crate) fn alloc(n: Degree, rank: Rank) -> Self {
        GLWESecret {
            data: ScalarZnx::from_data(
                poulpy_hal::layouts::HostBytesBackend::alloc_bytes(ScalarZnx::<Vec<u8>, W>::bytes_of(n.into(), rank.into())),
                n.into(),
                rank.into(),
            ),
            dist: Distribution::NONE,
        }
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        Self::bytes_of(infos.n(), infos.rank())
    }

    pub fn bytes_of(n: Degree, rank: Rank) -> usize {
        ScalarZnx::<Vec<u8>, W>::bytes_of(n.into(), rank.into())
    }
}

impl<D: HostDataMut, W: ZnxWord> GLWESecret<D, W> {
    /// Sets column `col` to the caller-provided binary `{0, 1}` coefficient
    /// vector and tags the distribution as [`Distribution::BinaryFixed`] with
    /// the vector's Hamming weight. For structured binary secrets whose
    /// positions are chosen by the caller (e.g. the PaCo bootstrapping key).
    ///
    /// The distribution tag is shared by all columns; when filling several
    /// columns the last written weight wins.
    ///
    /// `coeffs` is taken as `i64` for caller convenience rather than as the
    /// layout's own word; the assert restricts the values to `{0, 1}`, so
    /// converting them into any [`ZnxWord`] is exact.
    pub fn fill_binary_coeffs(&mut self, col: usize, coeffs: &[i64]) {
        assert!(coeffs.iter().all(|&x| x == 0 || x == 1), "coefficients must be binary");
        let dst = self.data.at_mut(col, 0);
        assert_eq!(dst.len(), coeffs.len(), "coefficient length must equal the ring degree");
        dst.iter_mut().zip(coeffs.iter()).for_each(|(d, &c)| *d = W::from_i64(c));
        self.dist = Distribution::BinaryFixed(coeffs.iter().filter(|&&x| x != 0).count());
    }
}

/// Secret-key sampling, dispatched to the backend.
///
/// Sampling is a backend operation like any other: it is routed through the
/// `ScalarZnxFill*` extension points so a backend can substitute its own
/// implementation (device-side generation, a secure element, a hardware RNG),
/// rather than being a host-memory method on the layout.
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

impl<BE: Backend> GLWESecretSampling<BE> for Module<BE>
where
    Self: ScalarZnxFillTernaryProbSourceBackend<BE>
        + ScalarZnxFillTernaryHwSourceBackend<BE>
        + ScalarZnxFillBinaryProbSourceBackend<BE>
        + ScalarZnxFillBinaryHwSourceBackend<BE>
        + ScalarZnxFillBinaryBlockSourceBackend<BE>
        + VecZnxZeroBackend<BE>,
{
    fn glwe_secret_fill_ternary_prob<S>(&self, sk: &mut S, prob: f64, source: &mut Source)
    where
        S: GLWESecretToBackendMut<BE> + GetDistributionMut + GLWEInfos,
    {
        let rank: usize = sk.rank().into();
        {
            let mut sk_backend = sk.to_backend_mut();
            for i in 0..rank {
                self.scalar_znx_fill_ternary_prob_source_backend(&mut sk_backend.data, i, prob, source);
            }
        }
        *sk.dist_mut() = Distribution::TernaryProb(prob);
    }

    fn glwe_secret_fill_ternary_hw<S>(&self, sk: &mut S, hw: usize, source: &mut Source)
    where
        S: GLWESecretToBackendMut<BE> + GetDistributionMut + GLWEInfos,
    {
        let rank: usize = sk.rank().into();
        {
            let mut sk_backend = sk.to_backend_mut();
            for i in 0..rank {
                self.scalar_znx_fill_ternary_hw_source_backend(&mut sk_backend.data, i, hw, source);
            }
        }
        *sk.dist_mut() = Distribution::TernaryFixed(hw);
    }

    fn glwe_secret_fill_binary_prob<S>(&self, sk: &mut S, prob: f64, source: &mut Source)
    where
        S: GLWESecretToBackendMut<BE> + GetDistributionMut + GLWEInfos,
    {
        let rank: usize = sk.rank().into();
        {
            let mut sk_backend = sk.to_backend_mut();
            for i in 0..rank {
                self.scalar_znx_fill_binary_prob_source_backend(&mut sk_backend.data, i, prob, source);
            }
        }
        *sk.dist_mut() = Distribution::BinaryProb(prob);
    }

    fn glwe_secret_fill_binary_hw<S>(&self, sk: &mut S, hw: usize, source: &mut Source)
    where
        S: GLWESecretToBackendMut<BE> + GetDistributionMut + GLWEInfos,
    {
        let rank: usize = sk.rank().into();
        {
            let mut sk_backend = sk.to_backend_mut();
            for i in 0..rank {
                self.scalar_znx_fill_binary_hw_source_backend(&mut sk_backend.data, i, hw, source);
            }
        }
        *sk.dist_mut() = Distribution::BinaryFixed(hw);
    }

    fn glwe_secret_fill_binary_block<S>(&self, sk: &mut S, block_size: usize, source: &mut Source)
    where
        S: GLWESecretToBackendMut<BE> + GetDistributionMut + GLWEInfos,
    {
        let rank: usize = sk.rank().into();
        {
            let mut sk_backend = sk.to_backend_mut();
            for i in 0..rank {
                self.scalar_znx_fill_binary_block_source_backend(&mut sk_backend.data, i, block_size, source);
            }
        }
        *sk.dist_mut() = Distribution::BinaryBlock(block_size);
    }

    fn glwe_secret_fill_zero<S>(&self, sk: &mut S)
    where
        S: GLWESecretToBackendMut<BE> + GetDistributionMut + GLWEInfos,
    {
        let rank: usize = sk.rank().into();
        {
            let mut sk_backend = sk.to_backend_mut();
            let mut sk_vec = scalar_znx_as_vec_znx_backend_mut_from_mut::<BE>(&mut sk_backend.data);
            for i in 0..rank {
                self.vec_znx_zero_backend(&mut sk_vec, i);
            }
        }
        *sk.dist_mut() = Distribution::ZERO;
    }
}

pub trait GLWESecretToBackendMut<BE: Backend>: GLWESecretToBackendRef<BE> {
    fn to_backend_mut(&mut self) -> GLWESecretBackendMut<'_, BE>;
}

impl<BE: Backend> GLWESecretToBackendMut<BE> for GLWESecret<BE::OwnedBuf, BE::ZnxWord> {
    fn to_backend_mut(&mut self) -> GLWESecretBackendMut<'_, BE> {
        GLWESecret {
            dist: self.dist,
            data: <ScalarZnx<BE::OwnedBuf, BE::ZnxWord> as ScalarZnxToBackendMut<BE>>::to_backend_mut(&mut self.data),
        }
    }
}

pub trait GLWESecretToBackendRef<BE: Backend> {
    fn to_backend_ref(&self) -> GLWESecretBackendRef<'_, BE>;
}

impl<BE: Backend> GLWESecretToBackendRef<BE> for GLWESecret<BE::OwnedBuf, BE::ZnxWord> {
    fn to_backend_ref(&self) -> GLWESecretBackendRef<'_, BE> {
        GLWESecret {
            data: <ScalarZnx<BE::OwnedBuf, BE::ZnxWord> as ScalarZnxToBackendRef<BE>>::to_backend_ref(&self.data),
            dist: self.dist,
        }
    }
}

pub trait SecretConversion<B: Backend> {
    /// Derives the associated rank-1 `GLWESecret` from a `LWESecret` by applying
    /// the X → X⁻¹ automorphism (k = -1). The result is the GLWE polynomial key
    /// whose ring product with a mask decrypts LWE ciphertexts produced by
    /// `glwe_expand_lwe`.
    ///
    /// The source's [`Distribution`] tag is copied unchanged: the automorphism
    /// only permutes and negates coefficients, so the base secret it describes
    /// is the same one.
    fn glwe_secret_from_lwe_secret<S>(&self, src: &S) -> GLWESecret<B::OwnedBuf, B::ZnxWord>
    where
        S: LWESecretToBackendRef<B>;

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
    fn lwe_secret_from_glwe_secret<S>(&self, src: &S, lwe_n: Degree) -> LWESecret<B::OwnedBuf, B::ZnxWord>
    where
        S: GLWESecretToBackendRef<B>;
}

// Coefficient-word fence: `scalar_znx_automorphism_backend` is delegated by
// poulpy-hal for i64 backends only.
impl<B: Backend<ZnxWord = i64> + HalVecZnxImpl<B>> SecretConversion<B> for Module<B> {
    fn glwe_secret_from_lwe_secret<S>(&self, src: &S) -> GLWESecret<B::OwnedBuf, B::ZnxWord>
    where
        S: LWESecretToBackendRef<B>,
    {
        let src = src.to_backend_ref();
        assert_eq!(src.n().as_usize(), self.n(), "LWE secret degree must equal module degree");
        let mut res = self.glwe_secret_alloc(Rank(1));
        res.dist = src.dist;
        {
            let mut res_ref = GLWESecretToBackendMut::<B>::to_backend_mut(&mut res);
            self.scalar_znx_automorphism_backend(-1, res_ref.data_mut(), 0, src.data(), 0);
        }
        res
    }

    fn lwe_secret_from_glwe_secret<S>(&self, src: &S, lwe_n: Degree) -> LWESecret<B::OwnedBuf, B::ZnxWord>
    where
        S: GLWESecretToBackendRef<B>,
    {
        let src = src.to_backend_ref();
        let n: usize = self.n();
        let rank: usize = src.rank().into();
        let target: usize = lwe_n.as_usize();
        assert_eq!(src.n().as_usize(), n, "GLWE secret degree must equal module degree");
        assert!(
            target <= rank * n,
            "lwe_secret_from_glwe_secret: requested LWE dim {} > rank * N ({})",
            target,
            rank * n
        );

        // Scratch buffer shaped like the source (rank cols, ring N) so the
        // per-column automorphism dimensions match. The result is then packed
        // into the flat LWE secret via the backend copy op, truncating the
        // last block when `target % n != 0`.
        let mut tmp: GLWESecret<B::OwnedBuf, B::ZnxWord> = self.glwe_secret_alloc(src.rank());
        {
            let mut tmp_ref = GLWESecretToBackendMut::<B>::to_backend_mut(&mut tmp);
            for j in 0..rank {
                self.scalar_znx_automorphism_backend(-1, tmp_ref.data_mut(), j, src.data(), j);
            }
        }

        let mut res: LWESecret<B::OwnedBuf, B::ZnxWord> = self.lwe_secret_alloc(lwe_n);
        res.dist = src.dist;
        {
            let tmp_ref = GLWESecretToBackendMut::<B>::to_backend_mut(&mut tmp);
            let mut res_ref = LWESecretToBackendMut::<B>::to_backend_mut(&mut res);
            let tmp_vz = scalar_znx_as_vec_znx_backend_ref_from_mut::<B>(tmp_ref.data());
            let mut res_vz = scalar_znx_as_vec_znx_backend_mut_from_mut::<B>(res_ref.data_mut());

            let mut written: usize = 0;
            for j in 0..rank {
                if written >= target {
                    break;
                }
                let take: usize = (target - written).min(n);
                self.vec_znx_copy_range_backend(&mut res_vz, 0, 0, written, &tmp_vz, j, 0, 0, take);
                written += take;
            }
        }
        res
    }
}
