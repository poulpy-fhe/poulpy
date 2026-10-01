#![cfg(feature = "enable-core")]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

use poulpy_core::{
    EncryptionLayout, GLWECIKeyEncryptSk,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGLWELayout, GLWESecretSampling, ModuleCoreAlloc, Rank, SecretConversion, TorusPrecision,
    },
};
use poulpy_cpu_ref::FFT64Ref;
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Module, ScratchOwned, ZnxView},
    source::Source,
};

#[derive(Clone, Copy, Default)]
struct Observation {
    expected_bytes: usize,
    address: usize,
    dropped: bool,
    zeroed: bool,
}

thread_local! {
    static OBSERVATION: Cell<Observation> = const { Cell::new(Observation {
        expected_bytes: 0,
        address: 0,
        dropped: false,
        zeroed: false,
    }) };
}

struct ObservingAllocator;

// Observe the first secret-sized allocation only during an explicitly scoped
// operation. Both operations allocate their initialized secret temporary first.
// Inspect its bytes before System frees them, never after deallocation.
unsafe impl GlobalAlloc for ObservingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: this allocator forwards the caller's allocation contract.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let _ = OBSERVATION.try_with(|state| {
                let mut observation = state.get();
                if observation.expected_bytes == layout.size() && observation.address == 0 {
                    observation.address = ptr as usize;
                    state.set(observation);
                }
            });
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let _ = OBSERVATION.try_with(|state| {
            let mut observation = state.get();
            if observation.address == ptr as usize && !observation.dropped {
                observation.dropped = true;
                // SAFETY: the tracked secret allocation is still live and its
                // entire buffer was initialized by the operation under test.
                observation.zeroed = unsafe { std::slice::from_raw_parts(ptr, layout.size()) }
                    .iter()
                    .all(|&byte| byte == 0);
                state.set(observation);
            }
        });
        // SAFETY: observation did not change the allocation or its layout.
        unsafe { System.dealloc(ptr, layout) };
    }
}

#[global_allocator]
static ALLOCATOR: ObservingAllocator = ObservingAllocator;

fn assert_temporary_wiped<R>(bytes: usize, operation: impl FnOnce() -> R) -> R {
    OBSERVATION.with(|state| {
        state.set(Observation {
            expected_bytes: bytes,
            ..Observation::default()
        })
    });
    let result = operation();
    let observation = OBSERVATION.with(|state| state.replace(Observation::default()));
    assert_ne!(observation.address, 0, "the secret temporary must have been allocated");
    assert!(observation.dropped, "the secret temporary must be dropped before returning");
    assert!(observation.zeroed, "the secret temporary must be erased before deallocation");
    result
}

#[test]
fn glwe_to_lwe_secret_temporary_is_erased() {
    let module = Module::<FFT64Ref>::new(64);
    let mut secret = module.glwe_secret_alloc(Rank(2));
    module.glwe_secret_fill_ternary_hw(&mut secret, 8, &mut Source::new([1; 32]));
    let result = assert_temporary_wiped(64 * 2 * size_of::<i64>(), || {
        module.lwe_secret_from_glwe_secret(&secret, Degree(96))
    });
    assert!(result.data().at(0, 0).iter().any(|&coefficient| coefficient != 0));
    assert!(secret.data().at(0, 0).iter().any(|&coefficient| coefficient != 0));
}

#[test]
fn ci_key_generation_erases_both_embedded_secret_temporaries() {
    let module = Module::<FFT64Ref>::new(64);
    let small_module = Module::<FFT64Ref>::new(32);
    let mut secret = module.glwe_secret_alloc(Rank(1));
    let mut ci_secret = small_module.glwe_secret_alloc(Rank(1));
    module.glwe_secret_fill_ternary_hw(&mut secret, 8, &mut Source::new([2; 32]));
    small_module.glwe_secret_fill_ternary_hw(&mut ci_secret, 8, &mut Source::new([3; 32]));
    let layout = GGLWELayout {
        n: Degree(64),
        base2k: Base2K(18),
        dnum: Dnum(2),
        dsize: Dsize(1),
        k_aux: TorusPrecision(18),
        rank_in: Rank(1),
        rank_out: Rank(1),
        stride: 1,
    };
    let infos = EncryptionLayout::new_from_default_sigma(layout).unwrap();
    let mut embed = module.glwe_ci_embed_key_alloc_from_infos(&layout);
    let mut trace = module.glwe_ci_trace_key_alloc_from_infos(&layout);
    let mut scratch = ScratchOwned::<FFT64Ref>::alloc(module.glwe_ci_key_encrypt_sk_tmp_bytes(&layout));
    let mut source_xe = Source::new([4; 32]);
    let mut source_xa = Source::new([5; 32]);
    assert_temporary_wiped(64 * size_of::<i64>(), || {
        module.glwe_ci_embed_key_encrypt_sk(
            &mut embed,
            &ci_secret,
            &secret,
            &infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );
    });
    assert_temporary_wiped(64 * size_of::<i64>(), || {
        module.glwe_ci_trace_key_encrypt_sk(
            &mut trace,
            &ci_secret,
            &secret,
            &infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );
    });
    assert!(secret.data().at(0, 0).iter().any(|&coefficient| coefficient != 0));
    assert!(ci_secret.data().at(0, 0).iter().any(|&coefficient| coefficient != 0));
}

#[test]
fn scratch_wipe_erases_buffer_before_drop() {
    assert_temporary_wiped(64, || {
        let mut scratch = ScratchOwned::<FFT64Ref>::alloc(64);
        scratch.data.fill(0xa5);
        scratch.arena().wipe(64);
    });
}
