//! Cross-backend parity suite.
//!
//! Every test here runs one operation on a reference backend and on a backend
//! under test, over identical inputs, and asserts the outputs are equal
//! byte-for-byte.
//!
//! This is deliberately not the [`super::noise`] suite in a second costume,
//! and the difference is the point:
//!
//! - the noise suite answers "does this backend implement the scheme", which
//!   needs secrets, encryption and a noise model, and can only judge a backend
//!   against a bound;
//! - this suite answers "does this backend agree with the reference", which
//!   needs none of those. A bound is a weak oracle: a gadget-product
//!   accumulator one limb too narrow passes the key-switch noise sweep
//!   comfortably (see the width contract on
//!   [`crate::oep::GLWEKeyswitchDefault`]). Equality is not weak.
//!
//! Because nothing is decrypted, the operands need not be valid ciphertexts:
//! they are filled with uniform noise, which exercises the limb arithmetic
//! harder than well-formed inputs do.
//!
//! The reference backend doubles as the staging area: it is host-resident by
//! construction, so it builds the inputs, receives the downloaded results and
//! performs the comparison. Only the backend under test is unconstrained, and
//! it owes just allocation, the prepared-key factory, the operation and
//! transfer, which is what lets a device backend run this suite.

mod automorphism;
mod external_product;
mod keyswitch;
mod operations;

pub use automorphism::*;
pub use external_product::*;
pub use keyswitch::*;
pub use operations::*;

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

/// Allocates a GLWE on the reference module and fills it with uniform noise.
pub(crate) fn ref_glwe<BR, A>(module_ref: &Module<BR>, infos: &A, source: &mut Source) -> BackendGLWE<BR>
where
    BR: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    A: GLWEInfos,
{
    let mut glwe = module_ref.glwe_alloc_from_infos(infos);
    glwe.fill_uniform(infos.base2k().into(), source);
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

/// Declares a `poulpy-core` parity suite for a (reference, test) backend pair.
///
/// Each test receives `(&TestParams, &ParityShapes, &Module<Ref>, &Module<Test>)`.
/// `shapes` is optional and defaults to the full sweep.
///
/// The two modules are `Lazy` statics shared by every test in the generated
/// module, and the test harness runs those tests in parallel. Sharing is fine
/// for the in-tree backends, whose modules are read-only handles, and is what
/// keeps setup off the per-test path. A backend whose module carries mutable
/// state (a device stream or context, say) must make that state `Sync`, or
/// declare one suite per test.
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
