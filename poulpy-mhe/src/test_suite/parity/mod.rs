//! Cross-backend parity of every multiparty protocol and PAT type.
//!
//! Fixtures (secrets, messages, ciphertexts, keys, PATs) are built once on the
//! comparison backend `BR` and transferred to the tested backend `BT`; each
//! backend prepares its own operands. Both run generation for every party from
//! the same private streams, then aggregation and finalization, and every
//! output is compared bit for bit: coefficients, layout, noise estimate,
//! canonical flag, seeds, distribution, degrees and Galois element, and the
//! next draw of every private stream. The pair must draw identical samples
//! from a seed, as backends sharing a sampler do, or the comparison backend
//! copies the tested backend's draws inside
//! [`with_backend_samples`](poulpy_core::test_suite::parity::controlled_sampling::with_backend_samples),
//! as [`mhe_parity_test_suite!`](crate::mhe_parity_test_suite) arranges.
//!
//! Every call runs on exactly its own query's bytes, poisoned with `0xA5`,
//! and every output starts poisoned. `BT` is bounded by API and transfer
//! traits only.

mod evaluation_key;
mod ggsw;
mod keyswitch;
mod pat;
mod public_key;
mod sharing;
mod tensor_key;

pub use evaluation_key::*;
pub use ggsw::*;
pub use keyswitch::*;
pub use pat::*;
pub use public_key::*;
pub use sharing::*;
pub use tensor_key::*;

use poulpy_core::{
    ComponentNoise, Distribution, GetDistribution, TransferInto,
    api::{GLWESecretPreparedFactory, GLWESecretSampling},
    layouts::{
        Degree, GGLWECompressedSeed, GGLWECompressedToBackendMut, GGLWECompressedToBackendRef, GGLWEInfos, GGLWELayout,
        GGLWEToBackendMut, GGLWEToBackendRef, GGSWInfos, GGSWLayout, GGSWToBackendMut, GGSWToBackendRef, GLWECompressedSeed,
        GLWECompressedToBackendMut, GLWECompressedToBackendRef, GLWEInfos, GLWELayout, GLWEPlaintextInfos, GLWEPlaintextMeta,
        GLWEPublicKeyCompressedSeed, GLWEPublicKeyCompressedToBackendRef, GLWEPublicKeyToBackendMut, GLWEPublicKeyToBackendRef,
        GLWESecret, GLWESecretPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, ModuleCoreAlloc, Rank,
    },
};
use poulpy_hal::{
    AlignedBuf,
    layouts::{Backend, CopyFromHost, CopyToHost, DataView, DataViewMut, Module, ScratchArena, ScratchOwned},
    source::Source,
};

use super::fixtures::{PARTIES, secret_from_seed_at};
use crate::layouts::GLWEPublicKeyShareOwned;

/// Storage the suite transfers fixtures into and reads outputs from, without host views.
pub trait ParityBackend: Backend<ZnxWord = i64, OwnedBuf: CopyToHost + CopyFromHost> {}
impl<BE: Backend<ZnxWord = i64, OwnedBuf: CopyToHost + CopyFromHost>> ParityBackend for BE {}

/// The comparison backend, which builds the fixtures on host-readable storage.
pub trait ReferenceBackend: Backend<ZnxWord = i64, OwnedBuf = AlignedBuf> {}
impl<BE: Backend<ZnxWord = i64, OwnedBuf = AlignedBuf>> ReferenceBackend for BE {}

#[derive(Debug, PartialEq, Eq)]
enum Meta {
    Glwe(GLWELayout),
    Gglwe(GGLWELayout),
    Ggsw(GGSWLayout),
    Canonical(bool),
    Plaintext(Option<GLWEPlaintextMeta>),
    Seeds(Vec<[u8; 32]>),
    Dist(Distribution),
    Degrees(Degree, Degree),
    GaloisElement(i64),
    Noise(Option<ComponentNoise>),
}

/// One output's coefficients and metadata; `Debug` prints the byte count only.
#[derive(PartialEq, Eq)]
struct Snapshot {
    label: &'static str,
    meta: Vec<Meta>,
    bytes: Vec<u8>,
}

impl std::fmt::Debug for Snapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Snapshot")
            .field("label", &self.label)
            .field("meta", &self.meta)
            .field("bytes", &self.bytes.len())
            .finish()
    }
}

impl Snapshot {
    fn with(mut self, meta: Meta) -> Self {
        self.meta.push(meta);
        self
    }

    /// Evaluation outputs carry no noise estimate, which equality between backends would miss.
    fn assert_untagged(&self) {
        assert!(
            self.meta.iter().all(|meta| !matches!(meta, Meta::Noise(Some(_)))),
            "{} kept a noise tag",
            self.label
        );
    }
}

fn view_bytes<BE: Backend>(buf: &BE::BufRef<'_>) -> Vec<u8> {
    let mut bytes = vec![0; BE::len_bytes_ref(buf)];
    BE::copy_view_to_host(buf, &mut bytes);
    bytes
}

fn poison_view<BE: Backend>(buf: &mut BE::BufMut<'_>) {
    let bytes = vec![0xA5; BE::len_bytes_mut(buf)];
    BE::copy_host_to_view(buf, &bytes);
}

fn snapshot_glwe<BE: Backend, A: GLWEToBackendRef<BE>>(label: &'static str, a: &A) -> Snapshot {
    let view = a.to_backend_ref();
    Snapshot {
        label,
        meta: vec![
            Meta::Glwe(view.glwe_layout()),
            Meta::Canonical(view.is_canonical()),
            Meta::Plaintext(view.plaintext_meta()),
            Meta::Noise(view.noise()),
        ],
        bytes: view_bytes::<BE>(view.data().data()),
    }
}

fn snapshot_glwe_compressed<BE, A>(label: &'static str, a: &A) -> Snapshot
where
    BE: Backend,
    A: GLWECompressedToBackendRef<BE> + GLWECompressedSeed,
{
    let view = a.to_backend_ref();
    Snapshot {
        label,
        meta: vec![
            Meta::Glwe(view.glwe_layout()),
            Meta::Seeds(vec![*a.seed()]),
            Meta::Plaintext(view.plaintext_meta()),
            Meta::Noise(view.noise()),
        ],
        bytes: view_bytes::<BE>(view.data().data()),
    }
}

fn snapshot_gglwe<BE: Backend, A: GGLWEToBackendRef<BE>>(label: &'static str, a: &A) -> Snapshot {
    let view = a.to_backend_ref();
    Snapshot {
        label,
        meta: vec![Meta::Gglwe(view.gglwe_layout()), Meta::Noise(view.noise())],
        bytes: view_bytes::<BE>(view.data().data()),
    }
}

fn snapshot_gglwe_compressed<BE, A>(label: &'static str, a: &A) -> Snapshot
where
    BE: Backend,
    A: GGLWECompressedToBackendRef<BE> + GGLWECompressedSeed,
{
    let view = a.to_backend_ref();
    Snapshot {
        label,
        meta: vec![
            Meta::Gglwe(view.gglwe_layout()),
            Meta::Seeds(a.seed().clone()),
            Meta::Noise(view.noise()),
        ],
        bytes: view_bytes::<BE>(view.data().data()),
    }
}

fn snapshot_ggsw<BE: Backend, A: GGSWToBackendRef<BE>>(label: &'static str, a: &A) -> Snapshot {
    let view = a.to_backend_ref();
    let mut bytes = Vec::new();
    for row in 0..view.dnum().as_usize() {
        for col in 0..=view.rank().as_usize() {
            let cell = view.at_view(row, col);
            bytes.extend(view_bytes::<BE>(cell.to_backend_ref().data().data()));
        }
    }
    Snapshot {
        label,
        meta: vec![Meta::Ggsw(view.ggsw_layout()), Meta::Noise(view.noise())],
        bytes,
    }
}

fn snapshot_public_key<BE, A>(label: &'static str, a: &A) -> Snapshot
where
    BE: Backend,
    A: GLWEPublicKeyToBackendRef<BE> + GetDistribution,
{
    let view = a.to_backend_ref();
    Snapshot {
        label,
        meta: vec![
            Meta::Glwe(view.glwe_layout()),
            Meta::Dist(*a.dist()),
            Meta::Noise(view.noise()),
        ],
        bytes: view_bytes::<BE>(view.data().data()),
    }
}

fn snapshot_public_key_share<BE: Backend>(label: &'static str, a: &GLWEPublicKeyShareOwned<BE>) -> Snapshot {
    let view = GLWEPublicKeyCompressedToBackendRef::<BE>::to_backend_ref(a);
    Snapshot {
        label,
        meta: vec![
            Meta::Glwe(view.glwe_layout()),
            Meta::Seeds(a.seed().to_vec()),
            Meta::Dist(*a.dist()),
            Meta::Noise(view.noise()),
        ],
        bytes: view_bytes::<BE>(view.data().data()),
    }
}

/// The next seed of a private stream, which equal draw counts leave equal.
fn snapshot_source(label: &'static str, source: &mut Source) -> Snapshot {
    Snapshot {
        label,
        meta: Vec::new(),
        bytes: source.new_seed().to_vec(),
    }
}

/// Fills a GLWE output with `0xA5` bytes and clears its canonical flag.
fn poison_glwe<BE: Backend, A: GLWEToBackendMut<BE>>(a: &mut A) {
    let mut view = a.to_backend_mut();
    poison_view::<BE>(view.data_mut().data_mut());
    view.set_canonical(false);
}

fn poison_glwe_compressed<BE: Backend, A: GLWECompressedToBackendMut<BE>>(a: &mut A) {
    poison_view::<BE>(a.to_backend_mut().data_mut().data_mut());
}

fn poison_gglwe<BE: Backend, A: GGLWEToBackendMut<BE>>(a: &mut A) {
    poison_view::<BE>(a.to_backend_mut().data_mut().data_mut());
}

fn poison_gglwe_compressed<BE: Backend, A: GGLWECompressedToBackendMut<BE>>(a: &mut A) {
    poison_view::<BE>(a.to_backend_mut().data_mut().data_mut());
}

fn poison_ggsw<BE: Backend, A: GGSWToBackendMut<BE>>(a: &mut A) {
    let mut view = a.to_backend_mut();
    for row in 0..view.dnum().as_usize() {
        for col in 0..=view.rank().as_usize() {
            poison_glwe::<BE, _>(&mut view.at_view_mut(row, col));
        }
    }
}

fn poison_public_key<BE: Backend, A: GLWEPublicKeyToBackendMut<BE>>(a: &mut A) {
    poison_view::<BE>(a.to_backend_mut().data_mut().data_mut());
}

/// Runs `call` on exactly `bytes` of scratch poisoned with `0xA5`, surrounded by guards.
pub(crate) fn with_scratch<BE: Backend, R>(bytes: usize, call: impl FnOnce(&mut ScratchArena<'_, BE>) -> R) -> R {
    let guard = BE::scratch_aligned(64);
    let mut scratch = ScratchOwned::<BE> {
        data: BE::from_host_bytes(&vec![0xA5; guard + bytes + guard]),
        _phantom: std::marker::PhantomData,
    };
    let result = {
        let (_, rest) = scratch.arena().split_at(guard);
        let (mut exact, _) = rest.split_at(bytes);
        assert_eq!(exact.available(), bytes);
        call(&mut exact)
    };
    let host = BE::to_host_bytes(&scratch.data);
    assert!(host[..guard].iter().all(|v| *v == 0xA5), "scratch prefix overwritten");
    assert!(
        host[guard + bytes..guard + bytes + guard].iter().all(|v| *v == 0xA5),
        "scratch suffix overwritten"
    );
    result
}

type Secrets<D> = Vec<GLWESecret<D, i64>>;

/// One secret per party at `rank`, from seeds `[base + i; 32]`, sampled on
/// `module_ref` and transferred to `module_test`.
fn fixture_secrets<BR, BT>(
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
    rank: Rank,
    base: u8,
) -> (Secrets<AlignedBuf>, Secrets<BT::OwnedBuf>)
where
    BR: ReferenceBackend,
    BT: ParityBackend,
    Module<BR>: GLWESecretSampling<BR> + GLWESecretPreparedFactory<BR>,
    Module<BT>: ModuleCoreAlloc<OwnedBuf = BT::OwnedBuf, ZnxWord = i64>,
{
    (0..PARTIES)
        .map(|i| {
            let (sk, _) = secret_from_seed_at(module_ref, rank, [base + i as u8; 32]);
            let mut copy = module_test.glwe_secret_alloc(rank);
            sk.transfer_into(&mut copy);
            (sk, copy)
        })
        .unzip()
}

/// The prepared forms of `secrets` on `module`.
fn prepare_secrets<BE: Backend>(
    module: &Module<BE>,
    secrets: &[GLWESecret<BE::OwnedBuf, BE::ZnxWord>],
) -> Vec<GLWESecretPrepared<BE::OwnedBuf, BE>>
where
    Module<BE>: GLWESecretPreparedFactory<BE>,
{
    secrets
        .iter()
        .map(|sk| {
            let mut prepared = module.glwe_secret_prepared_alloc(sk.rank());
            module.glwe_secret_prepare(&mut prepared, sk);
            prepared
        })
        .collect()
}

/// Declares the `poulpy-mhe` parity suite of `backend_test` against `backend_ref`.
///
/// Each test runs inside
/// [`with_backend_samples`](poulpy_core::test_suite::parity::controlled_sampling::with_backend_samples)
/// of a module of `backend_test`, so a controlled-sampling comparison backend
/// copies the tested backend's draws; a pair sharing a sampler compares
/// directly. Both modules have degree 256.
#[macro_export]
macro_rules! mhe_parity_test_suite {
    (mod $modname:ident, backend_ref = $backend_ref:ty, backend_test = $backend_test:ty $(,)?) => {
        $crate::mhe_parity_test_suite!(
            @tests $modname, $backend_ref, $backend_test,
            pat => test_pat_parity,
            glwe_public_key => test_glwe_public_key_parity,
            glwe_switching_key => test_glwe_switching_key_parity,
            glwe_automorphism_key => test_glwe_automorphism_key_parity,
            glwe_tensor_key => test_glwe_tensor_key_parity,
            ggsw => test_ggsw_parity,
            glwe_private_keyswitch => test_glwe_private_keyswitch_parity,
            glwe_public_keyswitch => test_glwe_public_keyswitch_parity,
            glwe_enc_to_share => test_glwe_enc_to_share_parity,
            glwe_share_to_enc => test_glwe_share_to_enc_parity,
        );
    };
    (@tests $modname:ident, $backend_ref:ty, $backend_test:ty, $($name:ident => $test:ident),+ $(,)?) => {
        mod $modname {
            use poulpy_hal::{api::ModuleNew, layouts::Module};

            $(
                #[test]
                fn $name() {
                    ::poulpy_core::test_suite::parity::controlled_sampling::with_backend_samples(
                        Module::<$backend_test>::new(256),
                        |module_test| $crate::test_suite::parity::$test(&Module::<$backend_ref>::new(256), module_test),
                    );
                }
            )+
        }
    };
}
