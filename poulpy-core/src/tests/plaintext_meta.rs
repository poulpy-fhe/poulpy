use std::io::ErrorKind;

use poulpy_hal::layouts::{HostBytesBackend, Module, ReaderFrom, WriterTo};

use crate::layouts::{
    Base2K, Degree, GLWELayout, GLWEPlaintextMeta, LWEInfos, ModuleCoreAlloc, ModuleCoreCompressedAlloc, Rank, ReplicatedLayout,
    Scale, SetGLWEPlaintextInfos, SetK, SlotLayout, SlotsKind, TorusPrecision,
};

fn layout(n: u32, k: u32) -> GLWELayout {
    GLWELayout {
        n: Degree(n),
        base2k: Base2K(16),
        k: TorusPrecision(k),
        rank: Rank(1),
    }
}

const SPARSE_META: GLWEPlaintextMeta = GLWEPlaintextMeta {
    scale: Scale::Log(8),
    slots: SlotsKind::Real,
    log_sparsity: 7,
    slot_layout: SlotLayout::Full,
};

fn replicated_meta() -> GLWEPlaintextMeta {
    GLWEPlaintextMeta {
        scale: Scale::Log(8),
        slot_layout: SlotLayout::Replicated(
            ReplicatedLayout::new(3, 3, 16, &[1, 0, -1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]).unwrap(),
        ),
        ..Default::default()
    }
}

fn assert_accepts_larger_serialized_degree<T>(mut source: T, mut receiver: T, meta: GLWEPlaintextMeta)
where
    T: ReaderFrom + WriterTo + LWEInfos + SetGLWEPlaintextInfos,
{
    assert_eq!(source.n(), Degree(128));
    assert_eq!(receiver.n(), Degree(64));
    assert_eq!(receiver.k(), source.k());
    source.set_plaintext_meta(Some(meta));
    let mut bytes = Vec::new();
    source.write_to(&mut bytes).unwrap();

    receiver.read_from(&mut bytes.as_slice()).unwrap();

    assert_eq!(receiver.n(), source.n());
    assert_eq!(receiver.k(), source.k());
    assert_eq!(receiver.plaintext_meta(), Some(meta));
    let mut round_trip = Vec::new();
    receiver.write_to(&mut round_trip).unwrap();
    assert!(round_trip == bytes, "a read plaintext serializes as the original");
}

fn assert_rejects_meta_exceeding_serialized_degree<T>(mut source: T, mut receiver: T, meta: GLWEPlaintextMeta, what: &str)
where
    T: ReaderFrom + WriterTo + LWEInfos + SetGLWEPlaintextInfos,
{
    assert_eq!(source.n(), Degree(64));
    assert_eq!(receiver.n(), Degree(128));
    assert_eq!(receiver.k(), source.k());
    // The metadata fits the receiver's initial degree, but not the stream's degree.
    source.set_plaintext_meta(Some(meta));
    let mut bytes = Vec::new();
    source.write_to(&mut bytes).unwrap();

    let error = receiver.read_from(&mut bytes.as_slice()).unwrap_err();

    assert_eq!(error.kind(), ErrorKind::InvalidData);
    assert!(error.to_string().contains(what));
    assert!(receiver.plaintext_meta().is_none());
}

#[test]
fn glwe_reader_accepts_sparsity_for_larger_serialized_degree() {
    let module = Module::<HostBytesBackend>::new(128);
    let source = module.glwe_alloc_from_infos(&layout(128, 16));
    // Two limbs at degree 64 reserve the same space as one limb at degree 128.
    let mut receiver = module.glwe_alloc_from_infos(&layout(64, 32));
    receiver.set_k(TorusPrecision(16));
    assert_accepts_larger_serialized_degree(source, receiver, SPARSE_META);
}

#[test]
fn glwe_reader_rejects_sparsity_exceeding_smaller_serialized_degree() {
    let module = Module::<HostBytesBackend>::new(128);
    let source = module.glwe_alloc_from_infos(&layout(64, 16));
    let receiver = module.glwe_alloc_from_infos(&layout(128, 16));
    assert_rejects_meta_exceeding_serialized_degree(source, receiver, SPARSE_META, "sparsity");
}

#[test]
fn compressed_glwe_reader_accepts_sparsity_for_larger_serialized_degree() {
    let module = Module::<HostBytesBackend>::new(128);
    let source = module.glwe_compressed_alloc_from_infos(&layout(128, 16));
    let mut receiver = module.glwe_compressed_alloc_from_infos(&layout(64, 32));
    receiver.k = TorusPrecision(16);
    assert_accepts_larger_serialized_degree(source, receiver, SPARSE_META);
}

#[test]
fn compressed_glwe_reader_rejects_sparsity_exceeding_smaller_serialized_degree() {
    let module = Module::<HostBytesBackend>::new(128);
    let source = module.glwe_compressed_alloc_from_infos(&layout(64, 16));
    let receiver = module.glwe_compressed_alloc_from_infos(&layout(128, 16));
    assert_rejects_meta_exceeding_serialized_degree(source, receiver, SPARSE_META, "sparsity");
}

#[test]
fn glwe_reader_round_trips_replicated_layout() {
    let module = Module::<HostBytesBackend>::new(128);
    let source = module.glwe_alloc_from_infos(&layout(128, 16));
    let mut receiver = module.glwe_alloc_from_infos(&layout(64, 32));
    receiver.set_k(TorusPrecision(16));
    assert_accepts_larger_serialized_degree(source, receiver, replicated_meta());
}

#[test]
fn glwe_reader_rejects_replicated_layout_exceeding_smaller_serialized_degree() {
    let module = Module::<HostBytesBackend>::new(128);
    let source = module.glwe_alloc_from_infos(&layout(64, 16));
    let receiver = module.glwe_alloc_from_infos(&layout(128, 16));
    assert_rejects_meta_exceeding_serialized_degree(source, receiver, replicated_meta(), "slot layout");
}

#[test]
fn compressed_glwe_reader_round_trips_replicated_layout() {
    let module = Module::<HostBytesBackend>::new(128);
    let source = module.glwe_compressed_alloc_from_infos(&layout(128, 16));
    let mut receiver = module.glwe_compressed_alloc_from_infos(&layout(64, 32));
    receiver.k = TorusPrecision(16);
    assert_accepts_larger_serialized_degree(source, receiver, replicated_meta());
}
