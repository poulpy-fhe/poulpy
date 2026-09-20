use poulpy_core::{
    api::{GLWEAdd, GLWEMulXpMinusOne, GLWERotate},
    layouts::{Base2K, Degree, GLWE, GLWELayout, GLWEToBackendMut, GLWEToBackendRef, ModuleCoreAlloc, Rank, TorusPrecision},
    oep::{GLWEAddImpl, GLWERotateReference},
    reference::operations::{GLWEAddComposition, GLWERotateComposition},
};
use poulpy_hal::AlignedBuf;
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{FillUniform, Module, ScratchArena, ScratchOwned},
    source::Source,
};

use crate::{FFT64Ref, hal_impl::delegating_backend::DelegatingFFT64Ref};
use std::sync::atomic::{AtomicUsize, Ordering};

static ROTATE_OVERRIDE_CALLS: AtomicUsize = AtomicUsize::new(0);

// This fixture implements the HAL requirements of GLWERotateComposition, yet
// can still replace its Reference contract without a conflicting blanket impl.
impl GLWERotateReference<DelegatingFFT64Ref> for Module<DelegatingFFT64Ref> {
    fn glwe_rotate_tmp_bytes_reference(&self) -> usize {
        self.glwe_rotate_tmp_bytes_composition()
    }

    fn glwe_rotate_reference<R, A>(&self, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<DelegatingFFT64Ref>,
        A: GLWEToBackendRef<DelegatingFFT64Ref>,
    {
        ROTATE_OVERRIDE_CALLS.fetch_add(1, Ordering::Relaxed);
        self.glwe_rotate_composition(k, res, a);
    }

    fn glwe_rotate_assign_reference<R>(&self, k: i64, res: &mut R, scratch: &mut ScratchArena<'_, DelegatingFFT64Ref>)
    where
        R: GLWEToBackendMut<DelegatingFFT64Ref>,
    {
        self.glwe_rotate_assign_composition(k, res, scratch);
    }
}

static ADD_IMPL_CALLS: AtomicUsize = AtomicUsize::new(0);

// Deliberately no GLWEAddReference implementation: public dispatch must need
// only the selected unsafe implementation, even when portable HAL is available.
unsafe impl GLWEAddImpl for DelegatingFFT64Ref {
    fn glwe_add_into<R, A, B>(module: &Module<Self>, res: &mut R, a: &A, b: &B)
    where
        R: GLWEToBackendMut<Self>,
        A: GLWEToBackendRef<Self>,
        B: GLWEToBackendRef<Self>,
    {
        ADD_IMPL_CALLS.fetch_add(1, Ordering::Relaxed);
        module.glwe_add_into_composition(res, a, b);
    }

    fn glwe_add_assign<R, A>(module: &Module<Self>, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<Self>,
        A: GLWEToBackendRef<Self>,
    {
        module.glwe_add_assign_composition(res, a);
    }
}

// Other families independently select their portable bodies.
poulpy_core::impl_glwe_mul_xp_minus_one_reference_full!(DelegatingFFT64Ref);

fn sample_glwe() -> GLWE<AlignedBuf, i64> {
    let layout = GLWELayout {
        n: Degree(256),
        base2k: Base2K(17),
        k: TorusPrecision(50),
        rank: Rank(2),
    };
    let module: Module<FFT64Ref> = Module::new(256);
    let mut ct: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&layout);
    let mut source = Source::new([7u8; 32]);
    ct.fill_uniform(40, &mut source);
    ct
}

#[test]
fn delegating_backend_manual_family_matches_fft64_ref() {
    assert_eq!(Module::<DelegatingFFT64Ref>::MAX_BASE2K, Module::<FFT64Ref>::MAX_BASE2K);
    let module_delegating: Module<DelegatingFFT64Ref> = Module::new(256);
    let module_ref: Module<FFT64Ref> = Module::new(256);

    let input = sample_glwe();
    let mut delegating_out = module_delegating.glwe_alloc_from_infos(&input);
    let mut ref_out = module_ref.glwe_alloc_from_infos(&input);

    module_delegating.glwe_mul_xp_minus_one(-7, &mut delegating_out, &input);
    module_ref.glwe_mul_xp_minus_one(-7, &mut ref_out, &input);

    assert_eq!(delegating_out, ref_out);
}

#[test]
fn public_core_dispatch_reaches_override_and_forwards_assign() {
    let module_delegating: Module<DelegatingFFT64Ref> = Module::new(256);
    let module_ref: Module<FFT64Ref> = Module::new(256);

    let input = sample_glwe();
    let mut delegating_out = module_delegating.glwe_alloc_from_infos(&input);
    let mut ref_out = module_ref.glwe_alloc_from_infos(&input);

    let before = ROTATE_OVERRIDE_CALLS.load(Ordering::Relaxed);
    module_delegating.glwe_rotate(11, &mut delegating_out, &input);
    assert_eq!(ROTATE_OVERRIDE_CALLS.load(Ordering::Relaxed), before + 1);
    module_ref.glwe_rotate(11, &mut ref_out, &input);

    assert_eq!(delegating_out, ref_out);
}

#[test]
fn public_core_dispatch_forwards_unchanged_method_with_own_scratch() {
    let module_delegating = Module::<DelegatingFFT64Ref>::new(256);
    let module_ref = Module::<FFT64Ref>::new(256);
    let mut actual = sample_glwe();
    let mut expected = actual.clone();
    let mut actual_scratch = ScratchOwned::<DelegatingFFT64Ref>::alloc(module_delegating.glwe_rotate_tmp_bytes());
    let mut expected_scratch = ScratchOwned::<FFT64Ref>::alloc(module_ref.glwe_rotate_tmp_bytes());
    module_delegating.glwe_rotate_assign(-11, &mut actual, &mut actual_scratch.borrow());
    module_ref.glwe_rotate_assign(-11, &mut expected, &mut expected_scratch.borrow());
    assert_eq!(actual, expected);
}

#[test]
fn public_core_dispatch_accepts_direct_impl_without_reference_trait() {
    let module = Module::<DelegatingFFT64Ref>::new(256);
    let reference = Module::<FFT64Ref>::new(256);
    let a = sample_glwe();
    let b = sample_glwe();
    let mut actual = module.glwe_alloc_from_infos(&a);
    let mut expected = reference.glwe_alloc_from_infos(&a);
    let before = ADD_IMPL_CALLS.load(Ordering::Relaxed);
    module.glwe_add_into(&mut actual, &a, &b);
    reference.glwe_add_into(&mut expected, &a, &b);
    assert_eq!(ADD_IMPL_CALLS.load(Ordering::Relaxed), before + 1);
    assert_eq!(actual, expected);
    module.glwe_add_assign(&mut actual, &a);
    reference.glwe_add_assign(&mut expected, &a);
    assert_eq!(actual, expected);
}
