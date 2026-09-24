use std::fmt::Debug;

use crate::{
    api::VecZnxFillUniformSource,
    layouts::{MatZnxAtBackendMut, Module, ReaderFrom, ScalarZnxAsVecZnxBackendMut, WriterTo},
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

/// Round-trips `ScalarZnx`, `VecZnx` and `MatZnx` fixtures sampled by `module`.
pub fn test_serialization<BE: TestBackend>(module: &Module<BE>)
where
    Module<BE>: VecZnxFillUniformSource<BE>,
{
    let n: usize = module.n();
    let base2k: usize = 50;
    let mut source = Source::new([0u8; 32]);

    let mut scalar = [(); 2].map(|_| module.scalar_znx_alloc(n, 3));
    for s in &mut scalar {
        for col in 0..3 {
            module.vec_znx_fill_uniform_source(
                base2k,
                base2k,
                &mut ScalarZnxAsVecZnxBackendMut::<BE>::as_vec_znx_backend_mut(s),
                col,
                &mut source,
            );
        }
    }
    test_reader_writer_interface(scalar.each_ref().map(download_scalar_znx::<BE>));

    let mut vec = [(); 2].map(|_| module.vec_znx_alloc(n, 3, 4));
    for v in &mut vec {
        for col in 0..3 {
            module.vec_znx_fill_uniform_source(base2k, 4 * base2k, &mut vec_znx_backend_mut::<BE>(v), col, &mut source);
        }
    }
    test_reader_writer_interface(vec.each_ref().map(download_vec_znx::<BE>));

    let mut mat = [(); 2].map(|_| module.mat_znx_alloc(n, 3, 2, 2, 4));
    for m in &mut mat {
        for row in 0..3 {
            for col_in in 0..2 {
                for col in 0..2 {
                    module.vec_znx_fill_uniform_source(
                        base2k,
                        4 * base2k,
                        &mut MatZnxAtBackendMut::<BE>::at_backend_mut(m, row, col_in),
                        col,
                        &mut source,
                    );
                }
            }
        }
    }
    test_reader_writer_interface(mat.each_ref().map(download_mat_znx::<BE>));
}
