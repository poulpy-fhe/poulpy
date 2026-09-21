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

use poulpy_hal::layouts::ZnxViewMut;
use poulpy_hal::{
    layouts::{Backend, CopyFromHost, CopyToHost, FillUniform, HostDataMut, Module},
    source::Source,
};

use crate::layouts::{BackendGGLWE, BackendGLWE, GGLWEInfos, GLWEInfos, ModuleCoreAlloc};

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
}

impl Default for ParityShapes {
    fn default() -> Self {
        Self {
            ranks: vec![1, 2],
            dsizes: None,
        }
    }
}

impl ParityShapes {
    /// The `dsize` sweep, resolving `None` against the operand precision.
    pub fn dsizes(&self, k: usize, base2k: usize) -> Vec<usize> {
        self.dsizes.clone().unwrap_or_else(|| (1..=k.div_ceil(base2k)).collect())
    }
}

/// Coefficient word shared by every backend this suite compares.
pub trait ParityBackend: Backend<ZnxWord = i64, OwnedBuf: CopyToHost + CopyFromHost> {}

impl<BE: Backend<ZnxWord = i64, OwnedBuf: CopyToHost + CopyFromHost>> ParityBackend for BE {}

/// Builds precisely the advertised scratch capacity, poisoned before each operation.
/// No other operation or backend's budget can conceal an underestimate.
pub(crate) fn poisoned_scratch<B: Backend>(bytes: usize) -> poulpy_hal::layouts::ScratchOwned<B> {
    poulpy_hal::layouts::ScratchOwned {
        data: B::from_host_bytes(&vec![0xA5; bytes]),
        _phantom: std::marker::PhantomData,
    }
}

/// Allocates a GLWE on the reference module and fills it with uniform noise,
/// canonical at the `k` it reports.
///
/// The operations read an operand at exactly the width it reports, so a
/// layout whose `k` is not limb-aligned must carry nothing below it: the
/// padding bits of the last live limb and every limb past it are zero.
pub(crate) fn ref_glwe<BR, A>(module_ref: &Module<BR>, infos: &A, source: &mut Source) -> BackendGLWE<BR>
where
    BR: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    A: GLWEInfos,
{
    let base2k: usize = infos.base2k().into();
    let mut glwe = module_ref.glwe_alloc_from_infos(infos);
    glwe.fill_uniform(base2k, source);
    let k: usize = infos.k().as_usize();
    let live: usize = k.div_ceil(base2k);
    let pad: usize = (base2k - k % base2k) % base2k;
    for col in 0..glwe.data.cols() {
        if pad != 0 && live > 0 {
            for digit in glwe.data.at_mut(col, live - 1) {
                *digit &= !0i64 << pad;
            }
        }
        for limb in live..glwe.data.size() {
            glwe.data.at_mut(col, limb).fill(0);
        }
    }
    glwe
}

/// Allocates a GGLWE on the reference module and fills it with uniform noise.
pub(crate) fn ref_gglwe<BR, A>(module_ref: &Module<BR>, infos: &A, source: &mut Source) -> BackendGGLWE<BR>
where
    BR: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    A: GGLWEInfos,
{
    let mut gglwe = module_ref.gglwe_alloc_from_infos(infos);
    gglwe.fill_uniform(infos.base2k().into(), source);
    gglwe
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
            static MODULE_TEST: Lazy<Module<$backend_test>> = Lazy::new(|| Module::<$backend_test>::new(PARAMS.size as u64));

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
