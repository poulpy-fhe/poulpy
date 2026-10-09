use poulpy_hal::{
    AlignedBuf,
    api::{VecZnxFillUniformSource, VecZnxFillUniformSourceAll},
    layouts::{Backend, MatZnx, MatZnxAtBackendMut, Module, ReaderFrom, WriterTo},
    source::Source,
    test_suite::serialization::test_reader_writer_interface as check_roundtrip,
};

use crate::dist::Distribution;
use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWE, GGSW, GLWE, GLWEAutomorphismKey, GLWEInfos, GLWEPublicKey, GLWESwitchingKey,
    GLWETensorKey, GLWEToLWEKey, LWE, LWEInfos, LWESwitchingKey, LWEToGLWEKey, Rank, TorusPrecision,
    compressed::{
        GGLWECompressed, GGSWCompressed, GLWEAutomorphismKeyCompressed, GLWECompressed, GLWESwitchingKeyCompressed,
        GLWETensorKeyCompressed, GLWEToLWESwitchingKeyCompressed, LWECompressed, LWESwitchingKeyCompressed,
        LWEToGLWEKeyCompressed,
    },
};
use crate::{ComponentNoise, FreshNoiseEstimate, api::GLWEMaskFill};

fn test_reader_writer_interface<T>([original, mut receiver]: [T; 2])
where
    T: WriterTo + ReaderFrom + PartialEq + Eq + std::fmt::Debug + LWEInfos,
{
    if original.noise().is_some() {
        let mut bytes = Vec::new();
        original.write_to(&mut bytes).unwrap();
        let marker = bytes.windows(4).position(|word| word == b"PNM3").unwrap();
        let mut malformed = bytes.clone();
        malformed[marker + 25..marker + 33].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(
            receiver
                .read_from(&mut malformed.as_slice())
                .unwrap_err()
                .to_string()
                .contains("count")
        );
        assert!(receiver.noise().is_none());
        receiver.write_to(&mut Vec::new()).unwrap();
        receiver.read_from(&mut bytes.as_slice()).unwrap();
        bytes.pop();
        assert!(receiver.read_from(&mut bytes.as_slice()).is_err());
        assert!(receiver.noise().is_none());
        receiver.write_to(&mut Vec::new()).unwrap();
    }
    check_roundtrip([original, receiver]);
}

/// A rejected shape must not enlarge the metadata allocation bound on a later
/// read into the same receiver. Check the first failure before sending the
/// compact oversized prefix, so a regression cannot request a huge allocation.
fn test_reused_reader_shape_bound<T>(mut receiver: T, malformed_shape: &[u8])
where
    T: WriterTo + ReaderFrom + GLWEInfos,
{
    let trusted_shape = (receiver.n(), receiver.rank());
    assert!(receiver.noise().is_some());
    let mut reader = malformed_shape;
    let error = receiver.read_from(&mut reader).unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    assert_eq!((receiver.n(), receiver.rank()), trusted_shape);
    assert!(receiver.noise().is_none());
    receiver.write_to(&mut Vec::new()).unwrap();

    let mut oversized = b"PNM3".to_vec();
    oversized.extend(1u64.to_le_bytes());
    Distribution::TernaryProb(0.5).write_to(&mut oversized).unwrap();
    oversized.extend(K.0.to_le_bytes());
    oversized.extend((1u64 << 32).to_le_bytes());
    oversized.extend(0u64.to_le_bytes());
    let mut reader = oversized.as_slice();
    let error = receiver.read_from(&mut reader).unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    assert!(error.to_string().contains("count"));
    assert_eq!(
        reader.len(),
        8,
        "the oversized count must fail before reading its stored prefix"
    );
    assert_eq!((receiver.n(), receiver.rank()), trusted_shape);
    assert!(receiver.noise().is_none());
    receiver.write_to(&mut Vec::new()).unwrap();
}

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
    let mut pk: [GLWEPublicKey<AlignedBuf, i64>; 2] = [(); 2].map(|_| GLWEPublicKey::alloc(N_GLWE, BASE2K, K, RANK));
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
        .chain(pk.iter_mut().map(|x| &mut x.data))
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

    let mut fixture = 0;
    let mut metadata = |rank: usize| {
        fixture += 1;
        Some(
            ComponentNoise::from_secret_at(Distribution::TernaryProb(0.3), K, rank).with_components(
                (0..=rank)
                    .map(|component| FreshNoiseEstimate::new((fixture * 10 + component) as f64, K))
                    .collect(),
            ),
        )
    };
    for value in &mut glwe {
        value.noise = metadata(RANK.as_usize());
    }
    for value in &mut glwe_c {
        value.noise = metadata(RANK.as_usize());
    }
    for value in &mut pk {
        value.noise = metadata(RANK.as_usize());
    }
    for value in &mut gglwe {
        value.noise = metadata(RANK.as_usize());
    }
    for value in &mut gglwe_c {
        value.noise = metadata(RANK.as_usize());
    }
    for value in &mut swk {
        value.key.noise = metadata(RANK.as_usize());
    }
    for value in &mut swk_c {
        value.key.noise = metadata(RANK.as_usize());
    }
    for value in &mut atk {
        value.key.noise = metadata(RANK.as_usize());
    }
    for value in &mut atk_c {
        value.key.noise = metadata(RANK.as_usize());
    }
    for value in &mut tsk {
        value.0.noise = metadata(RANK.as_usize());
    }
    for value in &mut tsk_c {
        value.0.noise = metadata(RANK.as_usize());
    }
    for value in &mut g2l {
        value.0.key.noise = metadata(1);
    }
    for value in &mut g2l_c {
        value.0.key.noise = metadata(1);
    }
    for value in &mut l2g {
        value.0.key.noise = metadata(RANK.as_usize());
    }
    for value in &mut l2g_c {
        value.0.key.noise = metadata(RANK.as_usize());
    }
    for value in &mut lsk {
        value.0.key.noise = metadata(1);
    }
    for value in &mut lsk_c {
        value.0.key.noise = metadata(1);
    }
    for value in &mut ggsw {
        value.noise = metadata(RANK.as_usize());
    }
    for value in &mut ggsw_c {
        value.noise = metadata(RANK.as_usize());
    }
    for value in &mut lwe {
        value.noise = metadata(N_LWE.as_usize());
    }
    for value in &mut pk {
        value.dist = Distribution::TernaryProb(0.3);
    }

    let mut none_prefix = Vec::new();
    ComponentNoise::write_optional(None, &mut none_prefix).unwrap();
    for (invalid_rank, partial_next_field) in [(0u32, false), (0, true), (u32::MAX, false), (u32::MAX, true)] {
        let mut glwe_header = none_prefix.clone();
        glwe_header.extend(BASE2K.0.to_le_bytes());
        glwe_header.extend(invalid_rank.to_le_bytes());
        let mut gadget_header = none_prefix.clone();
        gadget_header.extend(K_KEY_AUX.0.to_le_bytes());
        gadget_header.extend(BASE2K.0.to_le_bytes());
        gadget_header.extend(DSIZE.0.to_le_bytes());
        gadget_header.extend(invalid_rank.to_le_bytes());
        if partial_next_field {
            glwe_header.push(0);
            gadget_header.push(0);
        }
        test_reused_reader_shape_bound(glwe_c[1].clone(), &glwe_header);
        test_reused_reader_shape_bound(gglwe_c[1].clone(), &gadget_header);
        test_reused_reader_shape_bound(ggsw_c[1].clone(), &gadget_header);
    }
    // Zero degree makes the advertised byte length zero even with enormous
    // column counts. Reject the header before it replaces the trusted shape.
    let mut glwe_header = none_prefix.clone();
    glwe_header.extend(BASE2K.0.to_le_bytes());
    // Unset plaintext metadata.
    glwe_header.push(0);
    for value in [0u64, 1u64 << 32, 1, 0] {
        glwe_header.extend(value.to_le_bytes());
    }
    test_reused_reader_shape_bound(glwe[1].clone(), &glwe_header);
    let mut gadget_header = none_prefix;
    gadget_header.extend(BASE2K.0.to_le_bytes());
    gadget_header.extend(DSIZE.0.to_le_bytes());
    gadget_header.extend(K_KEY_AUX.0.to_le_bytes());
    for value in [0u64, 1, DNUM.0 as u64, RANK.0 as u64, 1u64 << 32, 0] {
        gadget_header.extend(value.to_le_bytes());
    }
    test_reused_reader_shape_bound(gglwe[1].clone(), &gadget_header);
    test_reused_reader_shape_bound(ggsw[1].clone(), &gadget_header);

    test_reader_writer_interface(glwe);
    // A stream is one row of `rank` encryptions of zero at the stated precision.
    let size: usize = K.0.div_ceil(BASE2K.0) as usize;
    let zeros: Vec<u8> = vec![0u8; GLWEPublicKey::<AlignedBuf, i64>::bytes_of(N_GLWE, BASE2K, K, RANK)];
    let pristine: GLWEPublicKey<AlignedBuf, i64> = GLWEPublicKey::alloc(N_GLWE, BASE2K, K, RANK);
    let mut receiver: GLWEPublicKey<AlignedBuf, i64> = GLWEPublicKey::alloc(N_GLWE, BASE2K, K, RANK);
    let n: usize = N_GLWE.into();
    for (n, base2k, rows, cols_in, cols_out, size) in [
        (n, BASE2K, 2, 1, 2, size),
        (n, BASE2K, 1, 2, 2, size),
        (n, BASE2K, 1, 0, 1, size),
        (n, BASE2K, 1, 2, 3, size - 1),
        (n, Base2K(0), 1, 2, 3, size),
        (0, BASE2K, 1, 2, 3, size),
    ] {
        let mut stream: Vec<u8> = Vec::new();
        ComponentNoise::write_optional(None, &mut stream).unwrap();
        Distribution::TernaryFixed(1).write_to(&mut stream).unwrap();
        stream.extend(base2k.0.to_le_bytes());
        stream.extend(K.0.to_le_bytes());
        MatZnx::<&[u8], i64>::from_data(&zeros, n, rows, cols_in, cols_out, size)
            .write_to(&mut stream)
            .unwrap();
        let error = receiver.read_from(&mut stream.as_slice()).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("invalid public key"));
        assert!(receiver == pristine, "a rejected stream changed the key");
    }
    let mut valid = Vec::new();
    pristine.write_to(&mut valid).unwrap();
    let mut wrong = Vec::new();
    ComponentNoise::write_optional(metadata(0).as_ref(), &mut wrong).unwrap();
    wrong.extend_from_slice(&valid[12..]);
    assert!(
        receiver
            .read_from(&mut wrong.as_slice())
            .unwrap_err()
            .to_string()
            .contains("noise component count")
    );
    test_reader_writer_interface(pk);
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
