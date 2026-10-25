//! Reused replicated transforms do not allocate prepared operand buffers.
use poulpy_ckks::{
    CKKSLayout, GLWEPlaintextMeta, Scale, SlotLayout,
    api::{CKKSModuleInfos, CKKSReplicatedOps},
    layouts::{CKKSModuleAlloc, ReplicatedConversion},
    shaping::ShapingUnit,
    test_suite::parity::{
        helpers::{fixture_ciphertext, snapshot},
        keys::{key_layout, prepared_automorphism_key},
    },
};
use poulpy_core::layouts::{GLWELayout, LWEInfos};
use poulpy_cpu_portable::FFT64Portable;
use poulpy_hal::{
    api::{CnvPVecBytesOf, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Module, PrepareHint, ScratchOwned},
};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    collections::HashMap,
    sync::atomic::{AtomicUsize, Ordering::Relaxed},
};

struct Counting;
static THRESHOLD: AtomicUsize = AtomicUsize::new(usize::MAX);
static LARGE_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

fn record(size: usize) {
    if size >= THRESHOLD.load(Relaxed) {
        LARGE_ALLOCATIONS.fetch_add(1, Relaxed);
    }
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record(size);
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static HEAP: Counting = Counting;

#[test]
fn ckks_replicated_workspace_reuses_prepared_buffers() {
    let module = Module::<FFT64Portable>::new(256);
    let conversion = ReplicatedConversion::new::<f64>(
        &module,
        ShapingUnit::U16_40.layout(3, 24).unwrap(),
        19usize.into(),
        20,
        30,
        16,
    )
    .unwrap();
    let layout = CKKSLayout {
        glwe_layout: GLWELayout {
            n: 256usize.into(),
            base2k: 19usize.into(),
            k: 130usize.into(),
            rank: 1usize.into(),
        },
        meta: GLWEPlaintextMeta {
            scale: Scale::Log(40),
            slot_layout: SlotLayout::Full,
            ..Default::default()
        },
    };
    let a = fixture_ciphertext(&module, &layout, 11);
    let b = fixture_ciphertext(&module, &layout, 12);
    let key_layout = key_layout(256, 19, 180, 1, 1, 1);
    let keys: HashMap<_, _> = conversion
        .rotations()
        .into_iter()
        .map(|rotation| {
            let gal = module.ckks_galois_element(rotation);
            (gal, prepared_automorphism_key(&module, &key_layout, gal, 13))
        })
        .collect();
    let mut workspace = conversion.alloc_to_workspace(&module, &a);
    let mut expected_workspace = conversion.alloc_to_workspace(&module, &b);
    let mut out = module.ckks_ciphertext_alloc_from_infos(&layout);
    let mut expected = module.ckks_ciphertext_alloc_from_infos(&layout);
    let mut scratch =
        ScratchOwned::<FFT64Portable>::alloc(module.ckks_replicated_conversion_tmp_bytes(&a, &key_layout, &conversion));
    module
        .ckks_to_replicated_into(&mut out, &a, &conversion, &mut workspace, &keys, &mut scratch.borrow())
        .unwrap();
    let previous = snapshot::<FFT64Portable, _>(&out);
    module
        .ckks_to_replicated_into(
            &mut expected,
            &b,
            &conversion,
            &mut expected_workspace,
            &keys,
            &mut scratch.borrow(),
        )
        .unwrap();
    let prepared_bytes = module.bytes_of_cnv_pvec_left(256, 2, (a.k().as_usize() + 16).div_ceil(19), PrepareHint::Reuse);
    LARGE_ALLOCATIONS.store(0, Relaxed);
    THRESHOLD.store(prepared_bytes, Relaxed);
    let result = module.ckks_to_replicated_into(&mut out, &b, &conversion, &mut workspace, &keys, &mut scratch.borrow());
    THRESHOLD.store(usize::MAX, Relaxed);
    result.unwrap();
    assert_eq!(
        LARGE_ALLOCATIONS.load(Relaxed),
        0,
        "evaluation allocated a prepared operand buffer"
    );
    assert_eq!(snapshot::<FFT64Portable, _>(&out), snapshot::<FFT64Portable, _>(&expected));
    assert_ne!(snapshot::<FFT64Portable, _>(&out), previous);
}
