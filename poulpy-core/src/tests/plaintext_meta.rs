use std::io::ErrorKind;

use poulpy_hal::layouts::{HostBytesBackend, Module, ReaderFrom, WriterTo};

use crate::layouts::{
    Base2K, Degree, GLWELayout, GLWEPlaintextMeta, LWEInfos, ModuleCoreAlloc, ModuleCoreCompressedAlloc, Rank, Scale,
    SetGLWEPlaintextInfos, SetK, SlotsKind, TorusPrecision,
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
};

fn assert_accepts_larger_serialized_degree<T>(mut source: T, mut receiver: T)
where
    T: ReaderFrom + WriterTo + LWEInfos + SetGLWEPlaintextInfos,
{
    assert_eq!(source.n(), Degree(128));
    assert_eq!(receiver.n(), Degree(64));
    assert_eq!(receiver.k(), source.k());
    source.set_plaintext_meta(Some(SPARSE_META));
    let mut bytes = Vec::new();
    source.write_to(&mut bytes).unwrap();

    receiver.read_from(&mut bytes.as_slice()).unwrap();

    assert_eq!(receiver.n(), source.n());
    assert_eq!(receiver.k(), source.k());
    assert_eq!(receiver.plaintext_meta(), Some(SPARSE_META));
    let mut round_trip = Vec::new();
    receiver.write_to(&mut round_trip).unwrap();
    assert_eq!(round_trip, bytes);
}

fn assert_rejects_sparsity_exceeding_serialized_degree<T>(mut source: T, mut receiver: T)
where
    T: ReaderFrom + WriterTo + LWEInfos + SetGLWEPlaintextInfos,
{
    assert_eq!(source.n(), Degree(64));
    assert_eq!(receiver.n(), Degree(128));
    assert_eq!(receiver.k(), source.k());
    // Sparsity 7 fits the receiver's initial degree, but not the stream's degree.
    source.set_plaintext_meta(Some(SPARSE_META));
    let mut bytes = Vec::new();
    source.write_to(&mut bytes).unwrap();

    let error = receiver.read_from(&mut bytes.as_slice()).unwrap_err();

    assert_eq!(error.kind(), ErrorKind::InvalidData);
    assert!(error.to_string().contains("sparsity"));
    assert!(receiver.plaintext_meta().is_none());
}

#[test]
fn glwe_reader_accepts_sparsity_for_larger_serialized_degree() {
    let module = Module::<HostBytesBackend>::new(128);
    let source = module.glwe_alloc_from_infos(&layout(128, 16));
    // Two limbs at degree 64 reserve the same space as one limb at degree 128.
    let mut receiver = module.glwe_alloc_from_infos(&layout(64, 32));
    receiver.set_k(TorusPrecision(16));
    assert_accepts_larger_serialized_degree(source, receiver);
}

#[test]
fn glwe_reader_rejects_sparsity_exceeding_smaller_serialized_degree() {
    let module = Module::<HostBytesBackend>::new(128);
    let source = module.glwe_alloc_from_infos(&layout(64, 16));
    let receiver = module.glwe_alloc_from_infos(&layout(128, 16));
    assert_rejects_sparsity_exceeding_serialized_degree(source, receiver);
}

#[test]
fn compressed_glwe_reader_accepts_sparsity_for_larger_serialized_degree() {
    let module = Module::<HostBytesBackend>::new(128);
    let source = module.glwe_compressed_alloc_from_infos(&layout(128, 16));
    let mut receiver = module.glwe_compressed_alloc_from_infos(&layout(64, 32));
    receiver.k = TorusPrecision(16);
    assert_accepts_larger_serialized_degree(source, receiver);
}

#[test]
fn compressed_glwe_reader_rejects_sparsity_exceeding_smaller_serialized_degree() {
    let module = Module::<HostBytesBackend>::new(128);
    let source = module.glwe_compressed_alloc_from_infos(&layout(64, 16));
    let receiver = module.glwe_compressed_alloc_from_infos(&layout(128, 16));
    assert_rejects_sparsity_exceeding_serialized_degree(source, receiver);
}
