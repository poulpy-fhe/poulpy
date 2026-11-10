use std::{fmt::Debug, io::ErrorKind};

use crate::{
    api::VecZnxFillUniformSource,
    layouts::{
        ConjugateInvariant, MatZnx, MatZnxAtBackendMut, Module, ReaderFrom, Ring, ScalarZnx, ScalarZnxAsVecZnxBackendMut,
        Standard, VecZnx, WriterTo,
    },
    source::Source,
    test_suite::{TestBackend, download_mat_znx, download_scalar_znx, download_vec_znx, vec_znx_backend_mut},
};

/// Generic test for serialization and deserialization.
///
/// Takes two independently sampled fixtures of one shape: the first is
/// written, then read back into the second, which must then equal it.
pub fn test_reader_writer_interface<T>([original, mut receiver]: [T; 2])
where
    T: WriterTo + ReaderFrom + PartialEq + Eq + Debug,
{
    assert!(original != receiver, "the two fixtures must differ");

    let mut buffer = Vec::new();
    original.write_to(&mut buffer).expect("write_to failed");

    let mut reader: &[u8] = &buffer;
    receiver.read_from(&mut reader).expect("read_from failed");

    assert_eq!(&original, &receiver, "Deserialized object does not match the original");
}

/// Writes `original`, of ring `Rg`, and reads it into a receiver of each ring: the one of `Rg` accepts it, the other
/// refuses it with `InvalidData`, unchanged.
fn assert_ring_checked<Rg: Ring, S, C>(original: &impl WriterTo, standard: S, conjugate_invariant: C)
where
    S: ReaderFrom + Clone + PartialEq + Debug,
    C: ReaderFrom + Clone + PartialEq + Debug,
{
    fn read_into<Rg: Ring, Rx: Ring, T: ReaderFrom + Clone + PartialEq + Debug>(stream: &[u8], mut receiver: T) {
        let before = receiver.clone();
        let read = receiver.read_from(&mut &stream[..]);
        if Rx::WIRE_TAG == Rg::WIRE_TAG {
            read.expect("a stream reads into a receiver of its ring");
        } else {
            assert_eq!(read.unwrap_err().kind(), ErrorKind::InvalidData);
            assert_eq!(receiver, before, "a stream of another ring changed the receiver");
        }
    }

    let mut stream = Vec::new();
    original.write_to(&mut stream).expect("write_to failed");
    read_into::<Rg, Standard, S>(&stream, standard);
    read_into::<Rg, ConjugateInvariant, C>(&stream, conjugate_invariant);
}

/// Round-trips `ScalarZnx`, `VecZnx` and `MatZnx` fixtures sampled by `module`, which a receiver of the other ring
/// refuses.
pub fn test_serialization<BE: TestBackend<Rg>, Rg: Ring>(module: &Module<BE>) {
    let n: usize = module.n();
    let base2k: usize = 50;
    let mut source = Source::new([0u8; 32]);

    let mut scalar = [(); 2].map(|_| module.scalar_znx_alloc(n, 3));
    for s in &mut scalar {
        for col in 0..3 {
            module.vec_znx_fill_uniform_source(
                base2k,
                base2k,
                &mut ScalarZnxAsVecZnxBackendMut::<BE, Rg>::as_vec_znx_backend_mut(s),
                col,
                &mut source,
            );
        }
    }
    let scalar = scalar.each_ref().map(download_scalar_znx::<BE, Rg>);
    assert_ring_checked::<Rg, _, _>(
        &scalar[0],
        ScalarZnx::<_, BE::ZnxWord, Standard>::alloc(n, 3),
        ScalarZnx::<_, BE::ZnxWord, ConjugateInvariant>::alloc(n, 3),
    );
    test_reader_writer_interface(scalar);

    let mut vec = [(); 2].map(|_| module.vec_znx_alloc(n, 3, 4));
    for v in &mut vec {
        for col in 0..3 {
            module.vec_znx_fill_uniform_source(base2k, 4 * base2k, &mut vec_znx_backend_mut::<BE, Rg>(v), col, &mut source);
        }
    }
    let vec = vec.each_ref().map(download_vec_znx::<BE, Rg>);
    assert_ring_checked::<Rg, _, _>(
        &vec[0],
        VecZnx::<_, BE::ZnxWord, Standard>::alloc(n, 3, 4),
        VecZnx::<_, BE::ZnxWord, ConjugateInvariant>::alloc(n, 3, 4),
    );
    test_reader_writer_interface(vec);

    let mut mat = [(); 2].map(|_| module.mat_znx_alloc(n, 3, 2, 2, 4));
    for m in &mut mat {
        for row in 0..3 {
            for col_in in 0..2 {
                for col in 0..2 {
                    module.vec_znx_fill_uniform_source(
                        base2k,
                        4 * base2k,
                        &mut MatZnxAtBackendMut::<BE, Rg>::at_backend_mut(m, row, col_in),
                        col,
                        &mut source,
                    );
                }
            }
        }
    }
    let mat = mat.each_ref().map(download_mat_znx::<BE, Rg>);
    assert_ring_checked::<Rg, _, _>(
        &mat[0],
        MatZnx::<_, BE::ZnxWord, Standard>::alloc(n, 3, 2, 2, 4),
        MatZnx::<_, BE::ZnxWord, ConjugateInvariant>::alloc(n, 3, 2, 2, 4),
    );
    test_reader_writer_interface(mat);
}
