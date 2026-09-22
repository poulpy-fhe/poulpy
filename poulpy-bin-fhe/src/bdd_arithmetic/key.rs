pub use crate::api::{BDDKeyEncryptSk, BDDKeyPreparedFactory};
use crate::bdd_arithmetic::FheUintPreparedDebug;
use crate::circuit_bootstrapping::CircuitBootstrappingKeyInfos;
use crate::{
    bdd_arithmetic::{FheUint, UnsignedInteger},
    blind_rotation::BlindRotationAlgo,
    circuit_bootstrapping::{
        CircuitBootstrappingEncryptionInfos, CircuitBootstrappingKey, CircuitBootstrappingKeyLayout,
        CircuitBootstrappingKeyPrepared,
    },
};
use poulpy_hal::AlignedBuf;

use anyhow::Result;
use byteorder::{ReadBytesExt, WriteBytesExt};
use poulpy_core::layouts::{
    GGLWEInfos, GLWESwitchingKey, GLWESwitchingKeyLayout, GLWESwitchingKeyPrepared, GetAutomorphismKey, ModuleCoreAlloc,
};
use poulpy_core::{DEFAULT_BOUND_XE, DEFAULT_SIGMA_XE, TransferInto};
use poulpy_core::{
    GetDistribution,
    layouts::{
        GLWEInfos, GLWESecretToBackendRef, GLWEToLWEKey, GLWEToLWEKeyLayout, LWEInfos, LWESecretToBackendRef,
        prepared::GLWEToLWEKeyPrepared,
    },
};

use poulpy_core::NoiseInfos;
use poulpy_hal::{
    layouts::{
        Backend, CopyFromHost, CopyToHost, Data, HostBackend, HostDataMut, HostDataRef, ReaderFrom, ScratchArena, WriterTo,
        ZnxWord,
    },
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
                Some(ref l) => Some(NoiseInfos::new(l.k().as_usize(), DEFAULT_SIGMA_XE, DEFAULT_BOUND_XE)?),
                None => None,
            },
            ks_lwe: NoiseInfos::new(layout.ks_lwe_layout.k().as_usize(), DEFAULT_SIGMA_XE, DEFAULT_BOUND_XE)?,
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
pub struct BDDKey<D, BRA, W = i64>
where
    D: Data,
    BRA: BlindRotationAlgo,
    W: ZnxWord,
{
    pub(crate) cbt: CircuitBootstrappingKey<D, BRA, W>,
    pub(crate) ks_glwe: Option<GLWESwitchingKey<D, W>>,
    pub(crate) ks_lwe: GLWEToLWEKey<D, W>,
}

impl<D: Data, BRA: BlindRotationAlgo, W: ZnxWord> BDDKey<D, BRA, W> {
    pub fn alloc_from_infos<M, A: BDDKeyInfos>(module: &M, infos: &A) -> Self
    where
        M: ModuleCoreAlloc<OwnedBuf = D, ZnxWord = W> + poulpy_hal::api::ModuleN,
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

impl<D1, D2, BRA, W> TransferInto<BDDKey<D2, BRA, W>> for BDDKey<D1, BRA, W>
where
    D1: Data + CopyToHost,
    D2: Data + CopyFromHost,
    BRA: BlindRotationAlgo,
    W: ZnxWord,
{
    fn transfer_into(&self, dst: &mut BDDKey<D2, BRA, W>) {
        self.cbt.transfer_into(&mut dst.cbt);
        self.ks_lwe.transfer_into(&mut dst.ks_lwe);
        match (&self.ks_glwe, &mut dst.ks_glwe) {
            (Some(src), Some(dst)) => src.transfer_into(dst),
            (None, None) => {}
            _ => panic!("transfer_into: GLWE switching key"),
        }
    }
}

impl<D: Data, BRA: BlindRotationAlgo> BDDKey<D, BRA, i64> {
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_sk<S0, S1, M, BE: Backend<OwnedBuf = D, ZnxWord = i64>>(
        &mut self,
        module: &M,
        sk_lwe: &S0,
        sk_glwe: &S1,
        enc_infos: &BDDEncryptionInfos,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S0: LWESecretToBackendRef<BE> + GetDistribution + LWEInfos,
        S1: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
        M: BDDKeyEncryptSk<BRA, BE>,
    {
        module.bdd_key_encrypt_sk(self, sk_lwe, sk_glwe, enc_infos, source_xe, source_xa, scratch);
    }
}

impl<BRA: BlindRotationAlgo> ReaderFrom for BDDKey<AlignedBuf, BRA, i64> {
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

impl<BRA: BlindRotationAlgo> WriterTo for BDDKey<AlignedBuf, BRA, i64> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
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
            dnum: ks_glwe.dnum(),
            rank_in: ks_glwe.rank_in(),
            rank_out: ks_glwe.rank_out(),
            k_aux: ks_glwe.k_aux(),
            dsize: ks_glwe.dsize(),
        })
    }
    fn ks_lwe_infos(&self) -> GLWEToLWEKeyLayout {
        GLWEToLWEKeyLayout {
            n: self.ks_lwe.n(),
            base2k: self.ks_lwe.base2k(),
            dnum: self.ks_lwe.dnum(),
            rank_in: self.ks_lwe.rank_in(),
            k_aux: self.ks_lwe.k_aux(),
        }
    }
}

impl<BRA: BlindRotationAlgo, BE: Backend> GetAutomorphismKey<BE> for BDDKeyPrepared<BE::OwnedBuf, BRA, BE> {
    fn lookup_automorphism_key(
        &self,
        p: i64,
        k: poulpy_core::layouts::TorusPrecision,
    ) -> poulpy_core::Result<poulpy_core::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, BE>> {
        self.cbt.get_automorphism_key(p, k)
    }
}

impl<BRA: BlindRotationAlgo, BE: Backend<ZnxWord = i64>> BDDKeyPrepared<BE::OwnedBuf, BRA, BE> {
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
pub trait FheUintPrepareDebug<
    BRA: BlindRotationAlgo,
    T: UnsignedInteger,
    BE: Backend<OwnedBuf: HostDataMut + HostDataRef> + HostBackend,
>
{
    /// Populates `res` by bootstrapping each bit of `bits` through `key`'s
    /// circuit-bootstrapping pipeline, storing the output GGSW in standard form.
    fn fhe_uint_debug_prepare(
        &self,
        res: &mut FheUintPreparedDebug<BE::OwnedBuf, T, BE::ZnxWord>,
        bits: &FheUint<BE::OwnedBuf, T, BE::ZnxWord>,
        key: &BDDKeyPrepared<BE::OwnedBuf, BRA, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    );
}

impl<D: Data, BRA: BlindRotationAlgo, W: ZnxWord> BDDKey<D, BRA, W> {
    /// Builds a bundle from its constituent keys for an independent implementation.
    pub fn from_parts(
        cbt: CircuitBootstrappingKey<D, BRA, W>,
        ks_glwe: Option<GLWESwitchingKey<D, W>>,
        ks_lwe: GLWEToLWEKey<D, W>,
    ) -> Self {
        Self { cbt, ks_glwe, ks_lwe }
    }
    /// Borrows the circuit, optional bridge and extraction keys.
    #[allow(clippy::type_complexity)]
    pub fn parts(
        &self,
    ) -> (
        &CircuitBootstrappingKey<D, BRA, W>,
        &Option<GLWESwitchingKey<D, W>>,
        &GLWEToLWEKey<D, W>,
    ) {
        (&self.cbt, &self.ks_glwe, &self.ks_lwe)
    }
    /// Mutably borrows the components for a selected key-encryption/preparation operation.
    #[allow(clippy::type_complexity)]
    pub fn parts_mut(
        &mut self,
    ) -> (
        &mut CircuitBootstrappingKey<D, BRA, W>,
        &mut Option<GLWESwitchingKey<D, W>>,
        &mut GLWEToLWEKey<D, W>,
    ) {
        (&mut self.cbt, &mut self.ks_glwe, &mut self.ks_lwe)
    }
}

impl<D: Data, BRA: BlindRotationAlgo, BE: Backend> BDDKeyPrepared<D, BRA, BE> {
    /// Builds a bundle from its constituent keys for an independent implementation.
    pub fn from_parts(
        cbt: CircuitBootstrappingKeyPrepared<D, BRA, BE>,
        ks_glwe: Option<GLWESwitchingKeyPrepared<D, BE>>,
        ks_lwe: GLWEToLWEKeyPrepared<D, BE>,
    ) -> Self {
        Self { cbt, ks_glwe, ks_lwe }
    }
    /// Borrows the circuit, optional bridge and extraction keys.
    #[allow(clippy::type_complexity)]
    pub fn parts(
        &self,
    ) -> (
        &CircuitBootstrappingKeyPrepared<D, BRA, BE>,
        &Option<GLWESwitchingKeyPrepared<D, BE>>,
        &GLWEToLWEKeyPrepared<D, BE>,
    ) {
        (&self.cbt, &self.ks_glwe, &self.ks_lwe)
    }
    /// Mutably borrows the components for a selected key-encryption/preparation operation.
    #[allow(clippy::type_complexity)]
    pub fn parts_mut(
        &mut self,
    ) -> (
        &mut CircuitBootstrappingKeyPrepared<D, BRA, BE>,
        &mut Option<GLWESwitchingKeyPrepared<D, BE>>,
        &mut GLWEToLWEKeyPrepared<D, BE>,
    ) {
        (&mut self.cbt, &mut self.ks_glwe, &mut self.ks_lwe)
    }
}
