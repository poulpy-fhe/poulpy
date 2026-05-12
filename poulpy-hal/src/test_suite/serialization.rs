use std::fmt::Debug;

use crate::{
    layouts::{Backend, FillUniform, ReaderFrom, WriterTo},
    source::Source,
};

fn allocate_host_scalar_znx(n: usize, cols: usize) -> crate::layouts::ScalarZnx<Vec<u8>> {
    crate::layouts::ScalarZnx::from_data(
        crate::layouts::HostBytesBackend::alloc_bytes(crate::layouts::ScalarZnx::<Vec<u8>>::bytes_of(n, cols)),
        n,
        cols,
    )
}

fn allocate_host_vec_znx(n: usize, cols: usize, size: usize) -> crate::layouts::VecZnx<Vec<u8>> {
    crate::layouts::VecZnx::from_data(
        crate::layouts::HostBytesBackend::alloc_bytes(crate::layouts::VecZnx::<Vec<u8>>::bytes_of(n, cols, size)),
        n,
        cols,
        size,
    )
}

fn allocate_host_mat_znx(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> crate::layouts::MatZnx<Vec<u8>> {
    crate::layouts::MatZnx::from_data(
        crate::layouts::HostBytesBackend::alloc_bytes(crate::layouts::MatZnx::<Vec<u8>>::bytes_of(
            n, rows, cols_in, cols_out, size,
        )),
        n,
        rows,
        cols_in,
        cols_out,
        size,
    )
}

/// Generic test for serialization and deserialization.
///
/// - `T` must implement I/O traits, zeroing, cloning, and random filling.
pub fn test_reader_writer_interface<T>(mut original: T)
where
    T: WriterTo + ReaderFrom + PartialEq + Eq + Debug + Clone + FillUniform,
{
    // Fill original with uniform random data
    let mut source = Source::new([0u8; 32]);
    original.fill_uniform(50, &mut source);

    // Serialize into a buffer
    let mut buffer = Vec::new();
    original.write_to(&mut buffer).expect("write_to failed");

    // Prepare receiver: same shape, but randomized
    let mut receiver = original.clone();
    receiver.fill_uniform(50, &mut source);

    // Deserialize from buffer
    let mut reader: &[u8] = &buffer;
    receiver.read_from(&mut reader).expect("read_from failed");

    // Ensure serialization round-trip correctness
    assert_eq!(&original, &receiver, "Deserialized object does not match the original");
}

#[test]
fn scalar_znx_serialize() {
    let original = allocate_host_scalar_znx(1024, 3);
    test_reader_writer_interface(original);
}

#[test]
fn vec_znx_serialize() {
    let original = allocate_host_vec_znx(1024, 3, 4);
    test_reader_writer_interface(original);
}

#[test]
fn mat_znx_serialize() {
    let original = allocate_host_mat_znx(1024, 3, 2, 2, 4);
    test_reader_writer_interface(original);
}
