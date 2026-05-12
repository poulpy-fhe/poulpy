use crate::bdd_arithmetic::FheUintPreparedDebug;
use crate::circuit_bootstrapping::CircuitBootstrappingKeyInfos;
use crate::{
    bdd_arithmetic::{FheUint, UnsignedInteger},
    blind_rotation::BlindRotationAlgo,
    circuit_bootstrapping::{
        CircuitBootstrappingEncryptionInfos, CircuitBootstrappingKey, CircuitBootstrappingKeyEncryptSk,
        CircuitBootstrappingKeyLayout, CircuitBootstrappingKeyPrepared, CircuitBootstrappingKeyPreparedFactory,
    },
};

use anyhow::Result;
use byteorder::{ReadBytesExt, WriteBytesExt};
use poulpy_core::layouts::{
    GGLWEInfos, GLWEAutomorphismKeyHelper, GLWEAutomorphismKeyPrepared, GLWESecret, GLWESwitchingKey, GLWESwitchingKeyLayout,
    GLWESwitchingKeyPrepared, ModuleCoreAlloc,
};
use poulpy_core::{DEFAULT_BOUND_XE, DEFAULT_SIGMA_XE, GLWESwitchingKeyEncryptSk};
use poulpy_core::{
    GLWEToLWESwitchingKeyEncryptSk, GetDistribution, ScratchArenaTakeCore,
    layouts::{
        GLWEInfos, GLWESecretToBackendRef, GLWEToLWEKey, GLWEToLWEKeyLayout, GLWEToLWEKeyPreparedFactory, LWEInfos,
        LWESecretToBackendRef, prepared::GLWEToLWEKeyPrepared,
    },
};

use poulpy_hal::layouts::NoiseInfos;
use poulpy_hal::{
    layouts::{Backend, Data, HostBackend, Module, ReaderFrom, ScratchArena, WriterTo},
    source::Source,
};

/// Encryption noise parameters for all sub-keys of a BDD evaluation key bundle.
///
/// Created via [`BDDEncryptionInfos::from_default_sigma`] for the standard
/// Gaussian error distribution, or constructed manually for custom noise parameters.
pub struct BDDEncryptionInfos {
    /// Noise parameters for the circuit-bootstrapping key.
    pub cbt: CircuitBootstrappingEncryptionInfos,
    /// Noise parameters for the optional GLWE-to-GLWE switching key.
    pub ks_glwe: Option<NoiseInfos>,
    /// Noise parameters for the GLWE-to-LWE switching key.
    pub ks_lwe: NoiseInfos,
}

impl BDDEncryptionInfos {
    /// Constructs encryption infos using the default Gaussian sigma for all sub-keys.
    pub fn from_default_sigma(layout: &BDDKeyLayout) -> Result<Self> {
        Ok(Self {
            cbt: CircuitBootstrappingEncryptionInfos::from_default_sigma(&layout.cbt_layout)?,
            ks_glwe: match layout.ks_glwe_layout {
                Some(ref l) => Some(NoiseInfos::new(l.k.as_usize(), DEFAULT_SIGMA_XE, DEFAULT_BOUND_XE)?),
                None => None,
            },
            ks_lwe: NoiseInfos::new(layout.ks_lwe_layout.k.as_usize(), DEFAULT_SIGMA_XE, DEFAULT_BOUND_XE)?,
        })
    }
}

/// Dimension descriptor for a complete BDD evaluation key bundle.
///
/// Provides the layout parameters for the three constituent keys:
/// the circuit-bootstrapping key (`cbt`), the GLWE-to-LWE key-switching key
/// (`ks_lwe`), and the optional GLWE-to-GLWE key-switching key (`ks_glwe`).
///
/// `ks_glwe` is `Some` when the input ciphertext's GLWE rank differs from the
/// GLWE rank expected by the circuit-bootstrapping procedure, requiring an
/// intermediate rank reduction.
pub trait BDDKeyInfos {
    /// Layout of the circuit-bootstrapping key.
    fn cbt_infos(&self) -> CircuitBootstrappingKeyLayout;
    /// Layout of the GLWE-to-LWE key-switching key.
    fn ks_lwe_infos(&self) -> GLWEToLWEKeyLayout;
    /// Layout of the optional GLWE-to-GLWE key-switching key, or `None` if
    /// no intermediate rank reduction is needed.
    fn ks_glwe_infos(&self) -> Option<GLWESwitchingKeyLayout>;
}

/// Concrete dimension descriptor for a BDD evaluation key bundle.
///
/// Implements [`BDDKeyInfos`] and is suitable for use wherever a layout
/// descriptor is required (e.g. allocation, scratch-size queries).
#[derive(Debug, Clone, Copy)]
pub struct BDDKeyLayout {
    /// Layout of the circuit-bootstrapping key.
    pub cbt_layout: CircuitBootstrappingKeyLayout,
    /// Layout of the optional GLWE-to-GLWE key-switching key.
    pub ks_glwe_layout: Option<GLWESwitchingKeyLayout>,
    /// Layout of the GLWE-to-LWE key-switching key.
    pub ks_lwe_layout: GLWEToLWEKeyLayout,
}

impl BDDKeyInfos for BDDKeyLayout {
    fn cbt_infos(&self) -> CircuitBootstrappingKeyLayout {
        self.cbt_layout
    }

    fn ks_glwe_infos(&self) -> Option<GLWESwitchingKeyLayout> {
        self.ks_glwe_layout
    }

    fn ks_lwe_infos(&self) -> GLWEToLWEKeyLayout {
        self.ks_lwe_layout
    }
}

/// Raw BDD evaluation key bundle.
///
/// Contains the three sub-keys required to evaluate BDD circuits on encrypted
/// [`FheUint`] values:
///
/// - `cbt`: circuit-bootstrapping key (blind rotation + trace + key-switch).
/// - `ks_glwe`: optional GLWE-to-GLWE key-switching key for rank reduction before
///   LWE extraction.  Present when the input ciphertext's GLWE rank differs from
///   the bootstrapping GLWE rank.
/// - `ks_lwe`: GLWE-to-LWE key-switching key; applied after optional rank
///   reduction to produce LWE ciphertexts suitable for circuit bootstrapping.
///
/// ## Lifecycle
///
/// 1. Allocate with [`BDDKey::alloc_from_infos`].
/// 2. Fill with [`BDDKey::encrypt_sk`].
/// 3. Prepare into a [`BDDKeyPrepared`] before evaluation.
///
/// ## Thread Safety
///
/// `BDDKey` is `Sync`; multiple evaluation threads may hold shared references.
pub struct BDDKey<D, BRA>
where
    D: Data,
    BRA: BlindRotationAlgo,
{
    pub(crate) cbt: CircuitBootstrappingKey<D, BRA>,
    pub(crate) ks_glwe: Option<GLWESwitchingKey<D>>,
    pub(crate) ks_lwe: GLWEToLWEKey<D>,
}

impl<BRA: BlindRotationAlgo> BDDKey<Vec<u8>, BRA> {
    pub fn alloc_from_infos<M, A: BDDKeyInfos>(module: &M, infos: &A) -> Self
    where
        M: ModuleCoreAlloc<OwnedBuf = Vec<u8>> + poulpy_hal::api::ModuleN,
    {
        Self {
            cbt: CircuitBootstrappingKey::alloc_from_infos(module, &infos.cbt_infos()),
            ks_glwe: infos
                .ks_glwe_infos()
                .as_ref()
                .map(|infos| module.glwe_switching_key_alloc_from_infos(infos)),
            ks_lwe: module.glwe_to_lwe_key_alloc_from_infos(&infos.ks_lwe_infos()),
        }
    }
}

/// Backend-level factory for encrypting a [`BDDKey`] under a secret key.
///
/// Implemented for `Module<BE>` when the backend supports circuit-bootstrapping
/// and switching-key encryption.  Callers should prefer the convenience method
/// [`BDDKey::encrypt_sk`].
pub trait BDDKeyEncryptSk<BRA: BlindRotationAlgo, BE: Backend<OwnedBuf = Vec<u8>>> {
    /// Returns the minimum scratch-space size in bytes required by
    /// [`bdd_key_encrypt_sk`][Self::bdd_key_encrypt_sk].
    fn bdd_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: BDDKeyInfos;

    #[allow(clippy::too_many_arguments)]
    /// Fills `res` with key material encrypted under `sk_lwe` / `sk_glwe`.
    ///
    /// `source_xa` supplies mask randomness; `source_xe` supplies error
    /// randomness.  The scratch arena must be at least
    /// [`bdd_key_encrypt_sk_tmp_bytes`][Self::bdd_key_encrypt_sk_tmp_bytes]
    /// bytes.
    ///
    /// When `res.ks_glwe` is `Some`, a fresh intermediate GLWE key is sampled
    /// from `source_xe` and used as the bridging secret; `ks_lwe` is then
    /// encrypted under that intermediate key rather than `sk_glwe` directly.
    fn bdd_key_encrypt_sk<'s, S0, S1>(
        &self,
        res: &mut BDDKey<BE::OwnedBuf, BRA>,
        sk_lwe: &S0,
        sk_glwe: &S1,
        enc_infos: &BDDEncryptionInfos,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        S0: LWESecretToBackendRef<BE> + GetDistribution + LWEInfos,
        S1: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
        BE: 's;
}

impl<BE: Backend<OwnedBuf = Vec<u8>>, BRA: BlindRotationAlgo> BDDKeyEncryptSk<BRA, BE> for Module<BE>
where
    Self: CircuitBootstrappingKeyEncryptSk<BRA, BE> + GLWEToLWESwitchingKeyEncryptSk<BE> + GLWESwitchingKeyEncryptSk<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    fn bdd_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: BDDKeyInfos,
    {
        self.circuit_bootstrapping_key_encrypt_sk_tmp_bytes(&infos.cbt_infos())
            .max(self.glwe_to_lwe_key_encrypt_sk_tmp_bytes(&infos.ks_lwe_infos()))
    }

    #[allow(clippy::too_many_arguments)]
    fn bdd_key_encrypt_sk<'s, S0, S1>(
        &self,
        res: &mut BDDKey<BE::OwnedBuf, BRA>,
        sk_lwe: &S0,
        sk_glwe: &S1,
        enc_infos: &BDDEncryptionInfos,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        S0: LWESecretToBackendRef<BE> + GetDistribution + LWEInfos,
        S1: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
        BE: 's,
    {
        if let Some(key) = &mut res.ks_glwe {
            let ks_glwe_infos = enc_infos
                .ks_glwe
                .as_ref()
                .expect("ks_glwe enc_infos missing when ks_glwe key exists");
            let mut sk_out: GLWESecret<Vec<u8>> = self.glwe_secret_alloc(key.rank_out());
            sk_out.fill_ternary_prob(0.5, source_xe);
            self.glwe_switching_key_encrypt_sk(key, sk_glwe, &sk_out, ks_glwe_infos, source_xe, source_xa, scratch);
            self.glwe_to_lwe_key_encrypt_sk(
                &mut res.ks_lwe,
                sk_lwe,
                &sk_out,
                &enc_infos.ks_lwe,
                source_xe,
                source_xa,
                scratch,
            );
        } else {
            self.glwe_to_lwe_key_encrypt_sk(
                &mut res.ks_lwe,
                sk_lwe,
                sk_glwe,
                &enc_infos.ks_lwe,
                source_xe,
                source_xa,
                scratch,
            );
        }

        self.circuit_bootstrapping_key_encrypt_sk(&mut res.cbt, sk_lwe, sk_glwe, &enc_infos.cbt, source_xe, source_xa, scratch);
    }
}

impl<BRA: BlindRotationAlgo> BDDKey<Vec<u8>, BRA> {
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_sk<'s, S0, S1, M, BE: Backend<OwnedBuf = Vec<u8>> + HostBackend + 's>(
        &mut self,
        module: &M,
        sk_lwe: &S0,
        sk_glwe: &S1,
        enc_infos: &BDDEncryptionInfos,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        S0: LWESecretToBackendRef<BE> + GetDistribution + LWEInfos,
        S1: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
        M: BDDKeyEncryptSk<BRA, BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        module.bdd_key_encrypt_sk(self, sk_lwe, sk_glwe, enc_infos, source_xe, source_xa, scratch);
    }
}

impl<BRA: BlindRotationAlgo> ReaderFrom for BDDKey<Vec<u8>, BRA> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.cbt.read_from(reader)?;
        match reader.read_u8()? {
            0 => {
                if self.ks_glwe.is_some() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "self.ks_glwe.is_some()={} != expected false (ks_glwe tag=0)",
                            self.ks_glwe.is_some()
                        ),
                    ));
                }
            }
            1 => {
                let Some(ref mut ks_glwe) = self.ks_glwe else {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "self.ks_glwe.is_none()={} != expected false (ks_glwe tag=1)",
                            self.ks_glwe.is_none()
                        ),
                    ));
                };
                ks_glwe.read_from(reader)?;
            }
            tag => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("invalid ks_glwe tag={tag} (expected 0 or 1)"),
                ));
            }
        }
        self.ks_lwe.read_from(reader)?;
        Ok(())
    }
}

impl<BRA: BlindRotationAlgo> WriterTo for BDDKey<Vec<u8>, BRA> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.cbt.write_to(writer)?;
        match &self.ks_glwe {
            None => writer.write_u8(0)?,
            Some(k) => {
                writer.write_u8(1)?;
                k.write_to(writer)?;
            }
        }
        self.ks_lwe.write_to(writer)
    }
}

/// DFT-prepared BDD evaluation key bundle, ready for on-line evaluation.
///
/// Mirrors the structure of [`BDDKey`] but stores all sub-keys in their
/// DFT (frequency) domain representations for fast matrix-vector products
/// during circuit bootstrapping and key-switching.
///
/// ## Invariants
///
/// - `ks_glwe` is `Some` if and only if the corresponding [`BDDKey`]'s
///   `ks_glwe` was `Some`.
///
/// ## Thread Safety
///
/// `BDDKeyPrepared<&[u8], BRA, BE>` is `Sync`; evaluation threads may share
/// a single prepared key while each holding their own scratch arena.
pub struct BDDKeyPrepared<D, BRA, BE>
where
    D: Data,
    BRA: BlindRotationAlgo,
    BE: Backend,
{
    pub(crate) cbt: CircuitBootstrappingKeyPrepared<D, BRA, BE>,
    pub(crate) ks_glwe: Option<GLWESwitchingKeyPrepared<D, BE>>,
    pub(crate) ks_lwe: GLWEToLWEKeyPrepared<D, BE>,
}

impl<D: Data, BRA: BlindRotationAlgo, BE: Backend> BDDKeyInfos for BDDKeyPrepared<D, BRA, BE> {
    fn cbt_infos(&self) -> CircuitBootstrappingKeyLayout {
        CircuitBootstrappingKeyLayout {
            brk_layout: self.cbt.brk_infos(),
            atk_layout: self.cbt.atk_infos(),
            tsk_layout: self.cbt.tsk_infos(),
        }
    }
    fn ks_glwe_infos(&self) -> Option<GLWESwitchingKeyLayout> {
        self.ks_glwe.as_ref().map(|ks_glwe| GLWESwitchingKeyLayout {
            n: ks_glwe.n(),
            base2k: ks_glwe.base2k(),
            k: ks_glwe.max_k(),
            rank_in: ks_glwe.rank_in(),
            rank_out: ks_glwe.rank_out(),
            dnum: ks_glwe.dnum(),
            dsize: ks_glwe.dsize(),
        })
    }
    fn ks_lwe_infos(&self) -> GLWEToLWEKeyLayout {
        GLWEToLWEKeyLayout {
            n: self.ks_lwe.n(),
            base2k: self.ks_lwe.base2k(),
            k: self.ks_lwe.max_k(),
            rank_in: self.ks_lwe.rank_in(),
            dnum: self.ks_lwe.dnum(),
        }
    }
}

impl<BRA: BlindRotationAlgo, BE: Backend> GLWEAutomorphismKeyHelper<GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>, BE>
    for BDDKeyPrepared<BE::OwnedBuf, BRA, BE>
{
    fn automorphism_key_infos(&self) -> poulpy_core::layouts::GGLWELayout {
        self.cbt.automorphism_key_infos()
    }

    fn get_automorphism_key(&self, k: i64) -> Option<&GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>> {
        self.cbt.get_automorphism_key(k)
    }
}

/// Backend-level factory for allocating and preparing [`BDDKeyPrepared`] values.
///
/// Implemented for `Module<BE>` when the backend supports preparation of all
/// three constituent sub-keys.  Default method implementations delegate to
/// the corresponding sub-key factories.
pub trait BDDKeyPreparedFactory<BRA: BlindRotationAlgo, BE: Backend<OwnedBuf = Vec<u8>>>
where
    Self: Sized + CircuitBootstrappingKeyPreparedFactory<BRA, BE> + GLWEToLWEKeyPreparedFactory<BE>,
{
    fn alloc_bdd_key_from_infos<A>(&self, infos: &A) -> BDDKeyPrepared<BE::OwnedBuf, BRA, BE>
    where
        A: BDDKeyInfos,
    {
        let ks_glwe = if let Some(ks_glwe_infos) = &infos.ks_glwe_infos() {
            Some(self.glwe_switching_key_prepared_alloc_from_infos(ks_glwe_infos))
        } else {
            None
        };

        BDDKeyPrepared {
            cbt: CircuitBootstrappingKeyPrepared::alloc_from_infos(self, &infos.cbt_infos()),
            ks_glwe,
            ks_lwe: self.glwe_to_lwe_key_prepared_alloc_from_infos(&infos.ks_lwe_infos()),
        }
    }

    fn prepare_bdd_key_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: BDDKeyInfos,
    {
        self.circuit_bootstrapping_key_prepare_tmp_bytes(&infos.cbt_infos())
            .max(self.glwe_to_lwe_key_prepare_tmp_bytes(&infos.ks_lwe_infos()))
    }

    fn prepare_bdd_key<'s>(
        &self,
        res: &mut BDDKeyPrepared<BE::OwnedBuf, BRA, BE>,
        other: &BDDKey<BE::OwnedBuf, BRA>,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        BE: 's,
    {
        res.cbt.prepare(self, &other.cbt, scratch);

        if let Some(key_prep) = &mut res.ks_glwe {
            if let Some(other) = &other.ks_glwe {
                self.glwe_switching_key_prepare(key_prep, other, scratch);
            } else {
                panic!("incompatible keys: res has Some(ks_glwe) but other has none")
            }
        }

        self.glwe_to_lwe_key_prepare(&mut res.ks_lwe, &other.ks_lwe, scratch);
    }
}
impl<BRA: BlindRotationAlgo, BE: Backend<OwnedBuf = Vec<u8>>> BDDKeyPreparedFactory<BRA, BE> for Module<BE> where
    Self: Sized + CircuitBootstrappingKeyPreparedFactory<BRA, BE> + GLWEToLWEKeyPreparedFactory<BE>
{
}

impl<BRA: BlindRotationAlgo, BE: Backend<OwnedBuf = Vec<u8>>> BDDKeyPrepared<BE::OwnedBuf, BRA, BE> {
    pub fn alloc_from_infos<M, A>(module: &M, infos: &A) -> Self
    where
        M: BDDKeyPreparedFactory<BRA, BE>,
        A: BDDKeyInfos,
    {
        module.alloc_bdd_key_from_infos(infos)
    }
}

impl<D: Data, BRA: BlindRotationAlgo, BE: Backend> BDDKeyHelper<D, BRA, BE> for BDDKeyPrepared<D, BRA, BE> {
    fn get_cbt_key(
        &self,
    ) -> (
        &CircuitBootstrappingKeyPrepared<D, BRA, BE>,
        Option<&GLWESwitchingKeyPrepared<D, BE>>,
        &GLWEToLWEKeyPrepared<D, BE>,
    ) {
        (&self.cbt, self.ks_glwe.as_ref(), &self.ks_lwe)
    }
}

/// Accessor trait for the constituent sub-keys of a prepared BDD key bundle.
///
/// Implemented by [`BDDKeyPrepared`].  Evaluation routines are generic over
/// this trait so that callers can pass any type that exposes the three
/// constituent prepared keys.
pub trait BDDKeyHelper<D: Data, BRA: BlindRotationAlgo, BE: Backend> {
    /// Returns references to the three constituent prepared keys in order:
    /// the circuit-bootstrapping key, the optional GLWE switching key, and
    /// the GLWE-to-LWE switching key.
    #[allow(clippy::type_complexity)]
    fn get_cbt_key(
        &self,
    ) -> (
        &CircuitBootstrappingKeyPrepared<D, BRA, BE>,
        Option<&GLWESwitchingKeyPrepared<D, BE>>,
        &GLWEToLWEKeyPrepared<D, BE>,
    );
}

/// Backend-level factory for building [`FheUintPreparedDebug`] values.
///
/// Unlike `FheUintPrepare`, this variant stores the per-bit GGSW ciphertexts
/// in standard (non-DFT) form, enabling noise inspection via
/// [`FheUintPreparedDebug::noise`] without a forward DFT transform.
pub trait FheUintPrepareDebug<BRA: BlindRotationAlgo, T: UnsignedInteger, BE: Backend<OwnedBuf = Vec<u8>> + HostBackend> {
    /// Populates `res` by bootstrapping each bit of `bits` through `key`'s
    /// circuit-bootstrapping pipeline, storing the output GGSW in standard form.
    fn fhe_uint_debug_prepare(
        &self,
        res: &mut FheUintPreparedDebug<BE::OwnedBuf, T>,
        bits: &FheUint<BE::OwnedBuf, T>,
        key: &BDDKeyPrepared<BE::OwnedBuf, BRA, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>;
}
