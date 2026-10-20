//! Cross-backend core contract tests.
//!
//! Operations run on a caller-selected comparison backend and a backend under
//! test, with identical logical inputs and independently prepared objects.
//! A validated backend can bootstrap another through transitive parity. Comparisons include integer coefficients, live precision and other
//! metadata; opaque transform bytes are never compared across backend layouts.
//!
//! Arithmetic fixtures use arbitrary canonical coefficients to stress carries,
//! widths and poisoned outputs. Mutation variants and each backend's advertised
//! scratch are checked independently. Encryption tests require identical realized
//! draws, using matching streams or an optional controlled-sampling adapter.
//! Same-seed byte equality is not required by the sampling contract; distribution
//! and source-consumption checks remain separate.
//!
//! The [`super::noise`] suite additionally checks scheme noise bounds. Backend
//! crates register these contract suites for their supported implementations.

/// `assert_eq!` on two evaluation outputs that also requires the reference to
/// be untagged; equality, which includes the tag, then covers the other.
macro_rules! assert_eval_eq {
    ($want:expr, $have:expr, $($msg:tt)+) => {{
        assert_eq!($want, $have, $($msg)+);
        assert!(crate::layouts::LWEInfos::noise(&$want).is_none(), "noise tag kept, {}", format_args!($($msg)+));
    }};
}

/// `assert_eval_eq!` on two GLWEs that also requires equal canonical flags,
/// which GLWE equality ignores.
macro_rules! assert_glwe_eq {
    ($want:expr, $have:expr, $($msg:tt)+) => {{
        assert_eq!($want, $have, $($msg)+);
        assert!(crate::layouts::LWEInfos::noise(&$want).is_none(), "noise tag kept, {}", format_args!($($msg)+));
        assert_eq!($want.is_canonical(), $have.is_canonical(), "canonical flag, {}", format_args!($($msg)+));
    }};
}

mod automorphism;
mod coarsened;
pub mod controlled_sampling;
mod conversion;
mod digits;
mod encryption;
mod encryption_keys;
mod external_product;
mod gadget;
mod keyswitch;
mod linear_transformation;
mod operations;
mod polynomial_evaluation;
mod preparation;
mod structure;

pub use automorphism::*;
pub use coarsened::*;
pub use conversion::*;
pub use digits::*;
pub use encryption::*;
pub use encryption_keys::*;
pub use external_product::*;
pub use gadget::*;
pub use keyswitch::*;
pub use linear_transformation::*;
pub use operations::*;
pub use polynomial_evaluation::*;
pub use preparation::*;
pub use structure::*;

use crate::oep::EncryptionImpl;
use poulpy_hal::layouts::{ZnxView, ZnxViewMut};
use poulpy_hal::oep::HalVecZnxImpl;
use poulpy_hal::{
    layouts::{Backend, CopyFromHost, CopyToHost, Module},
    source::Source,
};

use crate::{
    ComponentNoise, Distribution,
    api::{GLWEMaskFill, TransferInto},
    layouts::{
        BackendGGLWE, BackendGLWE, GGLWEInfos, GGLWEToBackendMut, GLWEInfos, GLWEToBackendMut, LWEInfos, ModuleCoreAlloc, Rank,
        SetK, TorusPrecision,
    },
    test_suite::keys::fill_by_digit,
};

/// Restricts the sweep to what a backend can actually serve.
///
/// [`Default`] is the full sweep. A backend with a narrower envelope (rank 1
/// only, a single `dsize`) narrows it here rather than giving up the suite:
/// coverage should degrade, not switch off.
#[derive(Clone, Debug)]
pub struct ParityShapes {
    /// Values swept on every rank axis, including `rank_in` and `rank_out`.
    pub ranks: Vec<usize>,
    /// Gadget digit sizes; `None` sweeps `1..=k.div_ceil(base2k)`.
    pub dsizes: Option<Vec<usize>>,
    /// Also sweeps digit widths off a limb boundary at rank 1.
    pub unaligned_digits: bool,
}

impl Default for ParityShapes {
    fn default() -> Self {
        Self {
            ranks: vec![1, 2],
            dsizes: None,
            unaligned_digits: true,
        }
    }
}

impl ParityShapes {
    /// The `dsize` sweep, resolving `None` against the operand precision.
    pub fn dsizes(&self, k: usize, base2k: usize) -> Vec<usize> {
        self.dsizes.clone().unwrap_or_else(|| (1..=k.div_ceil(base2k)).collect())
    }

    /// Digit widths in bits: `dsize * base2k` for every swept `dsize`, then,
    /// for the full sweep at rank 1, widths off a limb boundary (a sub-limb
    /// width and one bit past and short of a boundary), so the digit layout is
    /// compared too. `rank` is the largest rank of the case.
    pub fn digit_ks(&self, k: usize, base2k: usize, rank: usize) -> Vec<usize> {
        let mut widths: Vec<usize> = self.dsizes(k, base2k).into_iter().map(|d| d * base2k).collect();
        if self.dsizes.is_none() && self.unaligned_digits && rank == 1 {
            for width in [base2k / 2 + 1, base2k + 1, 2 * base2k - 1] {
                if (2..=k).contains(&width) && !widths.contains(&width) {
                    widths.push(width);
                }
            }
        }
        widths
    }
}

/// Coefficient word shared by every backend this suite compares.
pub trait ParityBackend: Backend<ZnxWord = i64, OwnedBuf: CopyToHost + CopyFromHost> + EncryptionImpl + HalVecZnxImpl {}

impl<BE: Backend<ZnxWord = i64, OwnedBuf: CopyToHost + CopyFromHost> + EncryptionImpl + HalVecZnxImpl> ParityBackend for BE {}

/// Builds precisely the advertised scratch capacity, poisoned before each operation.
/// No other operation or backend's budget can conceal an underestimate.
pub(crate) fn poisoned_scratch<B: Backend>(bytes: usize) -> poulpy_hal::layouts::ScratchOwned<B> {
    poulpy_hal::layouts::ScratchOwned {
        data: B::from_host_bytes(&vec![0xA5; bytes]),
        _phantom: std::marker::PhantomData,
    }
}

/// Allocates a GLWE on the reference module and fills it with uniform noise,
/// canonical at the `k` it reports. It carries a fresh noise estimate, so an
/// evaluation output that keeps a stale one fails the comparison.
///
/// The operations read an operand at exactly the width it reports, so a
/// layout whose `k` is not limb-aligned must carry nothing below it: the
/// padding bits of the last live limb and every limb past it are zero.
pub(crate) fn ref_glwe<BR, A>(module_ref: &Module<BR>, infos: &A, source: &mut Source) -> BackendGLWE<BR>
where
    BR: ParityBackend,
    A: GLWEInfos,
{
    let mut glwe = module_ref.glwe_alloc_from_infos(infos);
    module_ref.fill_glwe_from_source(&mut glwe, source);
    GLWEToBackendMut::<BR>::set_noise(&mut glwe, Some(fixture_noise(infos.k(), infos.rank())));
    glwe
}

/// Writes into `dst` the value of the canonical `a` with carries left in its
/// digits and bits below its `k`, flagged non-canonical.
pub(crate) fn unnormalized_twin<BS: ParityBackend, BD: ParityBackend>(a: &BackendGLWE<BS>, dst: &mut BackendGLWE<BD>) {
    let mut twin = a.to_host_owned::<BS>();
    let base2k: usize = twin.base2k().into();
    let k: usize = twin.k().as_usize();
    let live: usize = k.div_ceil(base2k);
    let pad: usize = live * base2k - k;
    let size: usize = twin.data.size();
    for col in 0..twin.data.cols() {
        // One unit of every limb moves into the limb below it: the value is
        // unchanged and the digits leave the canonical range.
        for limb in 1..live {
            twin.data.at_mut(col, limb - 1).iter_mut().for_each(|digit| *digit -= 1);
            twin.data.at_mut(col, limb).iter_mut().for_each(|digit| *digit += 1 << base2k);
        }
        // Bits below `k`, less than half its unit, round away.
        let (limb, bound) = if pad >= 2 {
            (live - 1, 1i64 << (pad - 1))
        } else if size > live && base2k >= 2 {
            (live, 1i64 << (base2k - 2))
        } else {
            continue;
        };
        for (i, digit) in twin.data.at_mut(col, limb).iter_mut().enumerate() {
            *digit += i as i64 % bound;
        }
    }
    twin.set_canonical(false);
    twin.transfer_into(dst);
}

/// Writes the canonical `a` into `dst`, a buffer of more limbs than `a`'s `k`
/// covers, at `a`'s precision: the limbs past it keep what `dst` held, so an
/// operation sized from `k` must read only the limbs `k` covers.
pub(crate) fn wide_twin<BS: ParityBackend, BD: ParityBackend>(a: &BackendGLWE<BS>, dst: &mut BackendGLWE<BD>) {
    let narrow = a.to_host_owned::<BS>();
    let mut wide = dst.to_host_owned::<BD>();
    assert_eq!(narrow.base2k(), wide.base2k());
    assert!(wide.data.size() > narrow.size(), "wide twin needs limbs past the precision");
    for col in 0..narrow.data.cols() {
        for limb in 0..narrow.size() {
            wide.data.at_mut(col, limb).copy_from_slice(narrow.data.at(col, limb));
        }
    }
    wide.set_k(narrow.k());
    wide.set_canonical(true);
    wide.noise = narrow.noise.clone();
    dst.set_k(narrow.k());
    wide.transfer_into(dst);
}

/// Allocates a GGLWE on the reference module and fills it with uniform noise,
/// tagged like [`ref_glwe`].
pub(crate) fn ref_gglwe<BR, A>(module_ref: &Module<BR>, infos: &A, source: &mut Source) -> BackendGGLWE<BR>
where
    BR: ParityBackend,
    A: GGLWEInfos,
{
    let mut gglwe = module_ref.gglwe_alloc_from_infos(infos);
    fill_by_digit(module_ref, &mut gglwe, 1, source);
    GGLWEToBackendMut::<BR>::set_noise(&mut gglwe, Some(fixture_noise(infos.k(), infos.rank_out())));
    gglwe
}

fn fixture_noise(k: TorusPrecision, rank: Rank) -> ComponentNoise {
    ComponentNoise::from_secret_at(Distribution::TernaryProb(0.5), k, rank.as_usize())
}

/// Declares a `poulpy-core` parity suite for a caller-selected backend pair.
///
/// `backend_ref` selects the comparison backend; `backend_test` selects the
/// backend under test. An already validated backend can serve as the comparison
/// backend for the same operations and parameter ranges.
///
/// Each test receives `(&TestParams, &ParityShapes, &Module<Ref>, &Module<Test>)`.
/// `shapes` is optional and defaults to the full sweep.
///
/// The two modules are `Lazy` statics shared by every test in the generated
/// module, and the test harness runs those tests in parallel. Their types must
/// be `Sync`, and the backend must support concurrent use of each module.
/// A backend that needs separate module state per test can call the test
/// helpers directly with locally constructed modules.
#[macro_export]
macro_rules! core_parity_test_suite {
    (
        mod $modname:ident,
        backend_ref = $backend_ref:ty,
        backend_test = $backend_test:ty,
        params = $params:expr,
        $(shapes = $shapes:expr,)?
        $(test_size = $test_size:expr,)?
        tests = {
            $( $(#[$attr:meta])* $test_name:ident => $impl:path ),+ $(,)?
        }
    ) => {
        mod $modname {
            use poulpy_hal::{api::ModuleNew, layouts::Module, test_suite::TestParams};

            use once_cell::sync::Lazy;

            static PARAMS: Lazy<TestParams> = Lazy::new(|| $params);
            static SHAPES: Lazy<$crate::test_suite::parity::ParityShapes> = Lazy::new(|| {
                #[allow(unused_mut)]
                let mut shapes = $crate::test_suite::parity::ParityShapes::default();
                $( shapes = $shapes; )?
                shapes
            });
            static MODULE_REF: Lazy<Module<$backend_ref>> = Lazy::new(|| Module::<$backend_ref>::new(PARAMS.size as u64));
            static MODULE_TEST: Lazy<Module<$backend_test>> = Lazy::new(|| {
                #[allow(unused_mut)]
                let mut size = PARAMS.size;
                $( size = $test_size; )?
                Module::<$backend_test>::new(size as u64)
            });

            $(
                $(#[$attr])*
                #[test]
                fn $test_name() {
                    ($impl)(&*PARAMS, &*SHAPES, &*MODULE_REF, &*MODULE_TEST);
                }
            )+
        }
    };
}
