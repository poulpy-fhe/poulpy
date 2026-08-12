use std::fmt::Debug;

use crate::{
    layouts::{Backend, FillUniform, MatZnxOwned, ReaderFrom, ScalarZnxOwned, VecZnxOwned, WriterTo, ZnxWord},
    source::Source,
};

fn allocate_host_scalar_znx<W: ZnxWord>(n: usize, cols: usize) -> ScalarZnxOwned<W> {
    crate::layouts::ScalarZnx::from_data(
        crate::layouts::HostBytesBackend::alloc_bytes(ScalarZnxOwned::<W>::bytes_of(n, cols)),
        n,
        cols,
    )
}

fn allocate_host_vec_znx<W: ZnxWord>(n: usize, cols: usize, size: usize) -> VecZnxOwned<W> {
    crate::layouts::VecZnx::from_data(
        crate::layouts::HostBytesBackend::alloc_bytes(VecZnxOwned::<W>::bytes_of(n, cols, size)),
        n,
        cols,
        size,
    )
}

fn allocate_host_mat_znx<W: ZnxWord>(n: usize, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> MatZnxOwned<W> {
    crate::layouts::MatZnx::from_data(
        crate::layouts::HostBytesBackend::alloc_bytes(MatZnxOwned::<W>::bytes_of(n, rows, cols_in, cols_out, size)),
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
    let original = allocate_host_scalar_znx::<i64>(1024, 3);
    test_reader_writer_interface(original);
}

#[test]
fn vec_znx_serialize() {
    let original = allocate_host_vec_znx::<i64>(1024, 3, 4);
    test_reader_writer_interface(original);
}

#[test]
fn mat_znx_serialize() {
    let original = allocate_host_mat_znx::<i64>(1024, 3, 2, 2, 4);
    test_reader_writer_interface(original);
}
