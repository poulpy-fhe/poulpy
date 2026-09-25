use poulpy_hal::{
    AlignedBuf,
    api::{VecZnxFillUniformSource, VecZnxFillUniformSourceAll},
    layouts::{Backend, MatZnxAtBackendMut, Module},
    source::Source,
    test_suite::serialization::test_reader_writer_interface,
};

use crate::api::GLWEMaskFill;
use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWE, GGSW, GLWE, GLWEAutomorphismKey, GLWESwitchingKey, GLWETensorKey, GLWEToLWEKey, LWE,
    LWESwitchingKey, LWEToGLWEKey, Rank, TorusPrecision,
    compressed::{
        GGLWECompressed, GGSWCompressed, GLWEAutomorphismKeyCompressed, GLWECompressed, GLWESwitchingKeyCompressed,
        GLWETensorKeyCompressed, GLWEToLWESwitchingKeyCompressed, LWECompressed, LWESwitchingKeyCompressed,
        LWEToGLWEKeyCompressed,
    },
};

const N_GLWE: Degree = Degree(64);
const N_LWE: Degree = Degree(32);
const BASE2K: Base2K = Base2K(12);
const K: TorusPrecision = TorusPrecision(33);
const DNUM: Dnum = Dnum(3);
const RANK: Rank = Rank(2);
const DSIZE: Dsize = Dsize(1);
/// Auxiliary guard of a key: one full gadget digit (`dsize * base2k`) plus
/// `log2(n)`. Must always be at least `dsize * base2k`.
const K_KEY_AUX: TorusPrecision = TorusPrecision(DSIZE.0 * BASE2K.0 + N_GLWE.0.ilog2());

/// Round-trips every core layout, standard and compressed, through
/// `WriterTo`/`ReaderFrom`, on fixtures sampled by `module` (degree at least 64).
pub fn test_serialization<BE>(module: &Module<BE>)
where
    BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: GLWEMaskFill<BE> + VecZnxFillUniformSource<BE> + VecZnxFillUniformSourceAll<BE>,
{
    let mut source = Source::new([0u8; 32]);

    let mut glwe: [GLWE<AlignedBuf, i64>; 2] = [(); 2].map(|_| GLWE::alloc(N_GLWE, BASE2K, K, RANK));
    let mut glwe_c: [GLWECompressed<AlignedBuf, i64>; 2] = [(); 2].map(|_| GLWECompressed::alloc::<BE>(N_GLWE, BASE2K, K, RANK));
    let mut lwe: [LWE<AlignedBuf, i64>; 2] = [(); 2].map(|_| LWE::alloc(N_LWE, BASE2K, K));
    let mut lwe_c: [LWECompressed<AlignedBuf, i64>; 2] = [(); 2].map(|_| LWECompressed::alloc::<BE>(BASE2K, K));
    for glwe in &mut glwe {
        module.fill_glwe_from_source(glwe, &mut source);
    }
    for v in glwe_c
        .iter_mut()
        .map(|x| &mut x.data)
        .chain(lwe.iter_mut().map(|x| &mut x.mask))
        .chain(lwe_c.iter_mut().map(|x| &mut x.data))
    {
        module.vec_znx_fill_uniform_source_all(50, v.size() * 50, v, &mut source);
    }

    let mut gglwe: [GGLWE<AlignedBuf, i64>; 2] =
        [(); 2].map(|_| GGLWE::alloc(N_GLWE, BASE2K, DNUM, DSIZE, K_KEY_AUX, RANK, RANK));
    let mut gglwe_c: [GGLWECompressed<AlignedBuf, i64>; 2] =
        [(); 2].map(|_| GGLWECompressed::alloc::<BE>(N_GLWE, BASE2K, DNUM, DSIZE, K_KEY_AUX, RANK, RANK));
    let mut swk: [GLWESwitchingKey<AlignedBuf, i64>; 2] =
        [(); 2].map(|_| GLWESwitchingKey::alloc(N_GLWE, BASE2K, DNUM, DSIZE, K_KEY_AUX, RANK, RANK));
    let mut swk_c: [GLWESwitchingKeyCompressed<AlignedBuf, i64>; 2] =
        [(); 2].map(|_| GLWESwitchingKeyCompressed::alloc::<BE>(N_GLWE, BASE2K, DNUM, DSIZE, K_KEY_AUX, RANK, RANK));
    let mut atk: [GLWEAutomorphismKey<AlignedBuf, i64>; 2] =
        [(); 2].map(|_| GLWEAutomorphismKey::alloc(N_GLWE, BASE2K, DNUM, DSIZE, K_KEY_AUX, RANK));
    let mut atk_c: [GLWEAutomorphismKeyCompressed<AlignedBuf, i64>; 2] =
        [(); 2].map(|_| GLWEAutomorphismKeyCompressed::alloc::<BE>(N_GLWE, BASE2K, DNUM, DSIZE, K_KEY_AUX, RANK));
    let mut tsk: [GLWETensorKey<AlignedBuf, i64>; 2] =
        [(); 2].map(|_| GLWETensorKey::alloc(N_GLWE, BASE2K, DNUM, DSIZE, K_KEY_AUX, RANK));
    let mut tsk_c: [GLWETensorKeyCompressed<AlignedBuf, i64>; 2] =
        [(); 2].map(|_| GLWETensorKeyCompressed::alloc::<BE>(N_GLWE, BASE2K, DNUM, DSIZE, K_KEY_AUX, RANK));
    let mut g2l: [GLWEToLWEKey<AlignedBuf, i64>; 2] = [(); 2].map(|_| GLWEToLWEKey::alloc(N_GLWE, BASE2K, DNUM, K_KEY_AUX, RANK));
    let mut g2l_c: [GLWEToLWESwitchingKeyCompressed<AlignedBuf, i64>; 2] =
        [(); 2].map(|_| GLWEToLWESwitchingKeyCompressed::alloc::<BE>(N_GLWE, BASE2K, DNUM, K_KEY_AUX, RANK));
    let mut l2g: [LWEToGLWEKey<AlignedBuf, i64>; 2] = [(); 2].map(|_| LWEToGLWEKey::alloc(N_GLWE, BASE2K, DNUM, K_KEY_AUX, RANK));
    let mut l2g_c: [LWEToGLWEKeyCompressed<AlignedBuf, i64>; 2] =
        [(); 2].map(|_| LWEToGLWEKeyCompressed::alloc::<BE>(N_GLWE, BASE2K, DNUM, K_KEY_AUX, RANK));
    let mut lsk: [LWESwitchingKey<AlignedBuf, i64>; 2] = [(); 2].map(|_| LWESwitchingKey::alloc(N_GLWE, BASE2K, DNUM, K_KEY_AUX));
    let mut lsk_c: [LWESwitchingKeyCompressed<AlignedBuf, i64>; 2] =
        [(); 2].map(|_| LWESwitchingKeyCompressed::alloc::<BE>(N_GLWE, BASE2K, DNUM, K_KEY_AUX));
    let mut ggsw: [GGSW<AlignedBuf, i64>; 2] = [(); 2].map(|_| GGSW::alloc(N_GLWE, BASE2K, DNUM, DSIZE, K_KEY_AUX, RANK));
    let mut ggsw_c: [GGSWCompressed<AlignedBuf, i64>; 2] =
        [(); 2].map(|_| GGSWCompressed::alloc::<BE>(N_GLWE, BASE2K, DNUM, DSIZE, K_KEY_AUX, RANK));
    for m in gglwe
        .iter_mut()
        .map(|x| &mut x.data)
        .chain(gglwe_c.iter_mut().map(|x| &mut x.data))
        .chain(swk.iter_mut().map(|x| &mut x.key.data))
        .chain(swk_c.iter_mut().map(|x| &mut x.key.data))
        .chain(atk.iter_mut().map(|x| &mut x.key.data))
        .chain(atk_c.iter_mut().map(|x| &mut x.key.data))
        .chain(tsk.iter_mut().map(|x| &mut x.0.data))
        .chain(tsk_c.iter_mut().map(|x| &mut x.0.data))
        .chain(g2l.iter_mut().map(|x| &mut x.0.key.data))
        .chain(g2l_c.iter_mut().map(|x| &mut x.0.key.data))
        .chain(l2g.iter_mut().map(|x| &mut x.0.key.data))
        .chain(l2g_c.iter_mut().map(|x| &mut x.0.key.data))
        .chain(lsk.iter_mut().map(|x| &mut x.0.key.data))
        .chain(lsk_c.iter_mut().map(|x| &mut x.0.key.data))
        .chain(ggsw.iter_mut().map(|x| &mut x.data))
        .chain(ggsw_c.iter_mut().map(|x| &mut x.data))
    {
        for row in 0..m.rows() {
            for col_in in 0..m.cols_in() {
                for col in 0..m.cols_out() {
                    module.vec_znx_fill_uniform_source(
                        50,
                        m.size() * 50,
                        &mut MatZnxAtBackendMut::<BE>::at_backend_mut(m, row, col_in),
                        col,
                        &mut source,
                    );
                }
            }
        }
    }

    test_reader_writer_interface(glwe);
    test_reader_writer_interface(glwe_c);
    test_reader_writer_interface(lwe);
    test_reader_writer_interface(lwe_c);
    test_reader_writer_interface(gglwe);
    test_reader_writer_interface(gglwe_c);
    test_reader_writer_interface(swk);
    test_reader_writer_interface(swk_c);
    test_reader_writer_interface(atk);
    test_reader_writer_interface(atk_c);
    test_reader_writer_interface(tsk);
    test_reader_writer_interface(tsk_c);
    test_reader_writer_interface(g2l);
    test_reader_writer_interface(g2l_c);
    test_reader_writer_interface(l2g);
    test_reader_writer_interface(l2g_c);
    test_reader_writer_interface(lsk);
    test_reader_writer_interface(lsk_c);
    test_reader_writer_interface(ggsw);
    test_reader_writer_interface(ggsw_c);
}
