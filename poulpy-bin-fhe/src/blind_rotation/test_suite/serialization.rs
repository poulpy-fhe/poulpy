use poulpy_core::layouts::{GGSWAtBackendMut, GGSWInfos, GLWEInfos, LWEInfos, compressed::GGSWCompressedToBackendMut};
use poulpy_hal::{
    api::VecZnxFillUniformSource,
    layouts::{Backend, HostDataMut, Module},
    source::Source,
    test_suite::serialization::test_reader_writer_interface,
};

use crate::{
    blind_rotation::{BlindRotationKey, BlindRotationKeyCompressed, BlindRotationKeyLayout, CGGI},
    oep::BlindRotationKeyCompressedFactoryImpl,
};

/// Round-trips a CGGI blind rotation key, standard and compressed, sampled by
/// `module`, through `WriterTo`/`ReaderFrom`.
pub fn test_blind_rotation_key_serialization<BE: Backend + BlindRotationKeyCompressedFactoryImpl<CGGI>>(module: &Module<BE>)
where
    BE::OwnedBuf: HostDataMut,
    Module<BE>: VecZnxFillUniformSource<BE>,
{
    let layout: BlindRotationKeyLayout = BlindRotationKeyLayout {
        n_glwe: (module.n() as u32).into(),
        n_lwe: 64_usize.into(),
        base2k: 12_usize.into(),
        dnum: 2_usize.into(),
        k_aux: 30_usize.into(),
        rank: 2_usize.into(),
    };
    let mut source = Source::new([0u8; 32]);

    let mut brk: [BlindRotationKey<BE::OwnedBuf, CGGI, BE::ZnxWord>; 2] =
        [(); 2].map(|_| BlindRotationKey::alloc(module, &layout));
    for ggsw in brk.iter_mut().flat_map(|x| &mut x.keys) {
        let (rows, cols, size) = (ggsw.dnum().as_usize(), ggsw.rank().as_usize() + 1, ggsw.size());
        for row in 0..rows {
            for col_in in 0..cols {
                let mut glwe = GGSWAtBackendMut::<BE>::at_backend_mut(ggsw, row, col_in);
                for col in 0..cols {
                    module.vec_znx_fill_uniform_source(50, size * 50, glwe.data_mut(), col, &mut source);
                }
            }
        }
    }
    test_reader_writer_interface(brk);

    let mut brk_c: [BlindRotationKeyCompressed<BE::OwnedBuf, CGGI, BE::ZnxWord>; 2] =
        [(); 2].map(|_| BlindRotationKeyCompressed::alloc(module, &layout));
    for ggsw in brk_c.iter_mut().flat_map(|x| &mut x.keys) {
        let (rows, cols, size) = (ggsw.dnum().as_usize(), ggsw.rank().as_usize() + 1, ggsw.size());
        let mut ggsw = GGSWCompressedToBackendMut::<BE>::to_backend_mut(ggsw);
        for row in 0..rows {
            for col in 0..cols {
                module.vec_znx_fill_uniform_source(50, size * 50, ggsw.at_view_mut(row, col).data_mut(), 0, &mut source);
            }
        }
    }
    test_reader_writer_interface(brk_c);
}
