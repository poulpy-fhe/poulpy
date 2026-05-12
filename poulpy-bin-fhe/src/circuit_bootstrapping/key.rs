use anyhow::Result;
use itertools::Itertools;
use poulpy_core::{
    DEFAULT_BOUND_XE, DEFAULT_SIGMA_XE, Distribution, GGLWEToGGSWKeyEncryptSk, GLWEAutomorphismKeyEncryptSk, GetDistribution,
    ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GGLWEToGGSWKey, GGLWEToGGSWKeyLayout, GGSWInfos, GLWEAutomorphismKey, GLWEAutomorphismKeyLayout, GLWEInfos,
        GLWESecretPreparedFactory, GLWESecretToBackendRef, LWEInfos, LWESecretToBackendRef, ModuleCoreAlloc,
        prepared::GLWESecretPrepared,
    },
};
use std::collections::HashMap;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use poulpy_hal::{
    api::{ModuleN, ScratchOwnedAlloc},
    layouts::{
        Backend, Data, HostBackend, HostDataMut, HostDataRef, Module, NoiseInfos, ReaderFrom, ScratchArena, ScratchOwned,
        WriterTo,
    },
    source::Source,
};

use crate::blind_rotation::{
    BlindRotationAlgo, BlindRotationKey, BlindRotationKeyEncryptSk, BlindRotationKeyInfos, BlindRotationKeyLayout,
};
use crate::circuit_bootstrapping::trace_galois_elements;

/// Encryption noise parameters for all three sub-keys of a circuit bootstrapping key bundle.
///
/// Created via [`CircuitBootstrappingEncryptionInfos::from_default_sigma`] for the
/// standard Gaussian error distribution, or constructed manually for custom noise parameters.
pub struct CircuitBootstrappingEncryptionInfos {
    /// Noise parameters for the blind rotation key.
    pub brk: NoiseInfos,
    /// Noise parameters for the automorphism (Galois) key.
    pub atk: NoiseInfos,
    /// Noise parameters for the tensor-switching key.
    pub tsk: NoiseInfos,
}

impl CircuitBootstrappingEncryptionInfos {
    /// Constructs encryption infos using the default Gaussian sigma for all sub-keys.
    pub fn from_default_sigma(layout: &CircuitBootstrappingKeyLayout) -> Result<Self> {
        Ok(Self {
            brk: NoiseInfos::new(layout.brk_layout.k.as_usize(), DEFAULT_SIGMA_XE, DEFAULT_BOUND_XE)?,
            atk: NoiseInfos::new(layout.atk_layout.k.as_usize(), DEFAULT_SIGMA_XE, DEFAULT_BOUND_XE)?,
            tsk: NoiseInfos::new(layout.tsk_layout.k.as_usize(), DEFAULT_SIGMA_XE, DEFAULT_BOUND_XE)?,
        })
    }
}

/// Accessor trait for the dimensional parameters of a circuit bootstrapping
/// key bundle.
///
/// Implemented by [`CircuitBootstrappingKeyLayout`], [`CircuitBootstrappingKey`],
/// and `CircuitBootstrappingKeyPrepared`.
pub trait CircuitBootstrappingKeyInfos {
    /// Number of LWE coefficients processed together in each BRK product step
    /// (1 for standard CGGI, > 1 for block-binary).
    fn block_size(&self) -> usize;
    /// Dimensional layout of the blind rotation key component.
    fn brk_infos(&self) -> BlindRotationKeyLayout;
    /// Dimensional layout of the automorphism (Galois) key component.
    fn atk_infos(&self) -> GLWEAutomorphismKeyLayout;
    /// Dimensional layout of the tensor-switching key component.
    fn tsk_infos(&self) -> GGLWEToGGSWKeyLayout;
}

/// Plain-old-data dimension descriptor for a circuit bootstrapping key bundle.
///
/// Contains the layouts for the three sub-keys:
/// - `brk_layout`: Blind rotation key.
/// - `atk_layout`: GLWE automorphism (trace / Galois) key.
/// - `tsk_layout`: GGLWE-to-GGSW tensor-switching key.
///
/// Note: [`CircuitBootstrappingKeyInfos::block_size`] is not representable
/// in this flat struct (it panics with `unimplemented!`); use an actual key
/// instance to query it.
#[derive(Debug, Clone, Copy)]
pub struct CircuitBootstrappingKeyLayout {
    pub brk_layout: BlindRotationKeyLayout,
    pub atk_layout: GLWEAutomorphismKeyLayout,
    pub tsk_layout: GGLWEToGGSWKeyLayout,
}

impl CircuitBootstrappingKeyInfos for CircuitBootstrappingKeyLayout {
    fn block_size(&self) -> usize {
        unimplemented!("unimplemented for CircuitBootstrappingKeyLayout")
    }

    fn atk_infos(&self) -> GLWEAutomorphismKeyLayout {
        self.atk_layout
    }

    fn brk_infos(&self) -> BlindRotationKeyLayout {
        self.brk_layout
    }

    fn tsk_infos(&self) -> GGLWEToGGSWKeyLayout {
        self.tsk_layout
    }
}

/// Backend-level trait for encrypting all sub-keys of a
/// [`CircuitBootstrappingKey`] at once.
///
/// Implemented for `Module<BE>` when the backend supports GGSW, automorphism,
/// and tensor-switching key encryption.  The module-level implementation
/// derives a fresh intermediate GLWE secret, prepares it, and delegates to
/// the individual sub-key encryption routines.
pub trait CircuitBootstrappingKeyEncryptSk<BRA, BE>
where
    BRA: BlindRotationAlgo,
    BE: Backend<OwnedBuf = Vec<u8>>,
{
    /// Returns the minimum scratch-space size (in bytes) required by
    /// [`circuit_bootstrapping_key_encrypt_sk`][Self::circuit_bootstrapping_key_encrypt_sk].
    fn circuit_bootstrapping_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: CircuitBootstrappingKeyInfos;

    /// Encrypts all sub-keys of a circuit bootstrapping key bundle.
    ///
    /// The three sub-key components are encrypted in order: ATK, BRK, TSK.
    /// Scratch space is reused across sub-key encryptions (peak is the maximum
    /// of the three individual requirements).
    #[allow(clippy::too_many_arguments)]
    fn circuit_bootstrapping_key_encrypt_sk<'s, S0, S1>(
        &self,
        res: &mut CircuitBootstrappingKey<BE::OwnedBuf, BRA>,
        sk_lwe: &S0,
        sk_glwe: &S1,
        enc_infos: &CircuitBootstrappingEncryptionInfos,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        S0: LWESecretToBackendRef<BE> + GetDistribution + LWEInfos,
        S1: GLWESecretToBackendRef<BE> + GLWEInfos + GetDistribution,
        BE: 's;
}

impl<D: Data, BRA: BlindRotationAlgo> CircuitBootstrappingKey<D, BRA> {
    pub fn alloc_from_infos<M, A: CircuitBootstrappingKeyInfos>(module: &M, infos: &A) -> Self
    where
        M: ModuleCoreAlloc<OwnedBuf = D> + ModuleN,
    {
        let atk_infos: &GLWEAutomorphismKeyLayout = &infos.atk_infos();
        let brk_infos: &BlindRotationKeyLayout = &infos.brk_infos();
        let trk_infos: &GGLWEToGGSWKeyLayout = &infos.tsk_infos();
        let gal_els: Vec<i64> = trace_galois_elements(atk_infos.log_n(), 2 * atk_infos.n().as_usize() as i64);

        assert!(
            !gal_els.is_empty(),
            "no Galois elements generated; log_n={} must be >= 1",
            atk_infos.log_n()
        );

        Self {
            brk: BRA::alloc_key(module, brk_infos),
            atk: gal_els
                .iter()
                .map(|&gal_el| {
                    let key = module.glwe_automorphism_key_alloc_from_infos(atk_infos);
                    (gal_el, key)
                })
                .collect(),
            tsk: module.gglwe_to_ggsw_key_alloc_from_infos(trk_infos),
        }
    }
}

/// Standard (un-prepared) circuit bootstrapping key bundle.
///
/// Bundles the three sub-keys required for a full circuit bootstrapping
/// evaluation:
///
/// - `brk`: The blind rotation key — one GGSW per LWE dimension.
/// - `atk`: A map of automorphism keys keyed by their Galois elements, used
///   for the trace and packing steps.  The set of Galois elements is the full
///   set of trace automorphisms of the cyclotomic ring (computed by
///   `trace_galois_elements`).
/// - `tsk`: The GGLWE-to-GGSW tensor-switching key, used in the final
///   promotion step.
///
/// ## Key Lifecycle
///
/// 1. Allocate with [`CircuitBootstrappingKey::alloc_from_infos`].
/// 2. Fill with [`CircuitBootstrappingKey::encrypt_sk`].
/// 3. Prepare with `CircuitBootstrappingKeyPrepared::prepare`.
pub struct CircuitBootstrappingKey<D: Data, BRA: BlindRotationAlgo> {
    pub(crate) brk: BlindRotationKey<D, BRA>,
    pub(crate) tsk: GGLWEToGGSWKey<D>,
    pub(crate) atk: HashMap<i64, GLWEAutomorphismKey<D>>,
}

impl<BRA: BlindRotationAlgo> CircuitBootstrappingKey<Vec<u8>, BRA> {
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_sk<'s, M, S0, S1, BE>(
        &mut self,
        module: &M,
        sk_lwe: &S0,
        sk_glwe: &S1,
        enc_infos: &CircuitBootstrappingEncryptionInfos,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        S0: LWESecretToBackendRef<BE> + GetDistribution + LWEInfos,
        S1: GLWESecretToBackendRef<BE> + GLWEInfos + GetDistribution,
        M: CircuitBootstrappingKeyEncryptSk<BRA, BE>,
        BE: Backend<OwnedBuf = Vec<u8>> + HostBackend + 's,
    {
        module.circuit_bootstrapping_key_encrypt_sk(self, sk_lwe, sk_glwe, enc_infos, source_xe, source_xa, scratch);
    }
}

impl<BRA: BlindRotationAlgo, BE: Backend<OwnedBuf = Vec<u8>>> CircuitBootstrappingKeyEncryptSk<BRA, BE> for Module<BE>
where
    Self: GGLWEToGGSWKeyEncryptSk<BE>
        + BlindRotationKeyEncryptSk<BRA, BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    BE::OwnedBuf: HostDataMut + HostDataRef,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    fn circuit_bootstrapping_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: CircuitBootstrappingKeyInfos,
    {
        self.glwe_automorphism_key_encrypt_sk_tmp_bytes(&infos.atk_infos())
            .max(self.blind_rotation_key_encrypt_sk_tmp_bytes(&infos.brk_infos()))
            .max(self.gglwe_to_ggsw_key_encrypt_sk_tmp_bytes(&infos.tsk_infos()))
    }

    fn circuit_bootstrapping_key_encrypt_sk<'s, S0, S1>(
        &self,
        res: &mut CircuitBootstrappingKey<BE::OwnedBuf, BRA>,
        sk_lwe: &S0,
        sk_glwe: &S1,
        enc_infos: &CircuitBootstrappingEncryptionInfos,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        S0: LWESecretToBackendRef<BE> + GetDistribution + LWEInfos,
        S1: GLWESecretToBackendRef<BE> + GLWEInfos + GetDistribution,
        BE: 's,
    {
        // TODO(device): this bundle encryptor is still effectively host-backed
        // through the current blind-rotation / automorphism / tensor key
        // encryptors. Keep the public trait backend-generic and move the host
        // assumptions down into the sub-key implementations until each path is
        // migrated.
        let brk_infos: &BlindRotationKeyLayout = &res.brk_infos();
        let atk_infos: &GLWEAutomorphismKeyLayout = &res.atk_infos();
        let tsk_infos: &GGLWEToGGSWKeyLayout = &res.tsk_infos();

        assert_eq!(sk_lwe.n(), brk_infos.n_lwe());
        assert_eq!(sk_glwe.n(), brk_infos.n_glwe());
        assert_eq!(sk_glwe.n(), atk_infos.n());
        assert_eq!(sk_glwe.n(), tsk_infos.n());

        assert!(sk_glwe.dist() != &Distribution::NONE);

        let gal_els: Vec<i64> = res.atk.keys().sorted().copied().collect();
        for p in gal_els {
            let atk = res.atk.get_mut(&p).unwrap();
            let mut atk_scratch: ScratchOwned<BE> = ScratchOwned::alloc(self.glwe_automorphism_key_encrypt_sk_tmp_bytes(atk));
            self.glwe_automorphism_key_encrypt_sk(
                atk,
                p,
                sk_glwe,
                &enc_infos.atk,
                source_xe,
                source_xa,
                &mut atk_scratch.arena(),
            );
        }

        let mut sk_glwe_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = self.glwe_secret_prepared_alloc(brk_infos.rank());
        self.glwe_secret_prepare(&mut sk_glwe_prepared, sk_glwe);

        self.blind_rotation_key_encrypt_sk(
            &mut res.brk,
            &sk_glwe_prepared,
            sk_lwe,
            &enc_infos.brk,
            source_xe,
            source_xa,
            scratch,
        );

        let mut tsk_scratch: ScratchOwned<BE> = ScratchOwned::alloc(self.gglwe_to_ggsw_key_encrypt_sk_tmp_bytes(&res.tsk));
        self.gglwe_to_ggsw_key_encrypt_sk(
            &mut res.tsk,
            sk_glwe,
            &enc_infos.tsk,
            source_xe,
            source_xa,
            &mut tsk_scratch.arena(),
        );
    }
}

impl<D: HostDataRef, BRA: BlindRotationAlgo> CircuitBootstrappingKeyInfos for CircuitBootstrappingKey<D, BRA> {
    fn block_size(&self) -> usize {
        self.brk.block_size()
    }

    fn atk_infos(&self) -> GLWEAutomorphismKeyLayout {
        let first_key = self.atk.keys().min().copied().expect("atk is empty");
        let atk = self.atk.get(&first_key).unwrap();
        GLWEAutomorphismKeyLayout {
            n: atk.n(),
            base2k: atk.base2k(),
            k: atk.max_k(),
            dnum: atk.dnum(),
            dsize: atk.dsize(),
            rank: atk.rank(),
        }
    }

    fn brk_infos(&self) -> BlindRotationKeyLayout {
        BlindRotationKeyLayout {
            n_glwe: self.brk.n_glwe(),
            n_lwe: self.brk.n_lwe(),
            base2k: self.brk.base2k(),
            k: self.brk.max_k(),
            dnum: self.brk.dnum(),
            rank: self.brk.rank(),
        }
    }

    fn tsk_infos(&self) -> GGLWEToGGSWKeyLayout {
        GGLWEToGGSWKeyLayout {
            n: self.tsk.n(),
            base2k: self.tsk.base2k(),
            k: self.tsk.max_k(),
            dnum: self.tsk.dnum(),
            dsize: self.tsk.dsize(),
            rank: self.tsk.rank(),
        }
    }
}

impl<D: HostDataMut, BRA: BlindRotationAlgo> ReaderFrom for CircuitBootstrappingKey<D, BRA> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.brk.read_from(reader)?;
        let n = reader.read_u64::<LittleEndian>()? as usize;
        if n != self.atk.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("self.atk.len()={} != read len={}", self.atk.len(), n),
            ));
        }
        for _ in 0..n {
            let gal_el = reader.read_i64::<LittleEndian>()?;
            let atk = self.atk.get_mut(&gal_el).ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, format!("self.atk.get(gal_el={gal_el})=None"))
            })?;
            atk.read_from(reader)?;
        }
        self.tsk.read_from(reader)
    }
}

impl<D: HostDataRef, BRA: BlindRotationAlgo> WriterTo for CircuitBootstrappingKey<D, BRA> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.brk.write_to(writer)?;
        writer.write_u64::<LittleEndian>(self.atk.len() as u64)?;
        // HashMap iteration order is undefined; sort for stable, canonical blobs.
        let mut keys: Vec<i64> = self.atk.keys().copied().collect();
        keys.sort_unstable();
        for k in keys {
            writer.write_i64::<LittleEndian>(k)?;
            self.atk[&k].write_to(writer)?;
        }
        self.tsk.write_to(writer)
    }
}
