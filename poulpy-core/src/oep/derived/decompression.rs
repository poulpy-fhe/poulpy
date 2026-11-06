//! Decompressions of gadget layouts and key wrappers composed from GLWE and GGLWE decompression.
use poulpy_hal::layouts::Module;

use crate::{
    GetDistribution, GetDistributionMut,
    layouts::{
        GGLWECompressedToBackendRef, GGLWEInfos, GGLWEToBackendMut, GGLWEToGGSWKeyCompressedToBackendRef,
        GGLWEToGGSWKeyToBackendMut, GGSWCompressedToBackendRef, GGSWInfos, GGSWToBackendMut, GLWEInfos, GLWEPublicKeyAtViewMut,
        GLWEPublicKeyCompressedToBackendRef, GLWEPublicKeyToBackendMut, GLWESwitchingKeyDegrees, GLWESwitchingKeyDegreesMut,
        GetGaloisElement, LWEInfos, SetGaloisElement,
    },
    oep::DecompressionImpl,
};

pub(crate) fn gglwe_decompress_derived<BE, R, O>(module: &Module<BE>, res: &mut R, other: &O)
where
    BE: DecompressionImpl,
    R: GGLWEToBackendMut<BE> + GGLWEInfos,
    O: GGLWECompressedToBackendRef<BE> + GGLWEInfos,
{
    let noise = other.to_backend_ref().noise();
    {
        let mut res = res.to_backend_mut();
        let other = other.to_backend_ref();

        assert_eq!(res.k_digit(), other.k_digit());
        assert_eq!(res.k_offset(), other.k_offset(), "gglwe_decompress: row offsets differ");
        assert!(res.dnum() <= other.dnum());

        let rank_in: usize = res.rank_in().into();
        let dnum: usize = res.dnum().into();
        for col_i in 0..rank_in {
            for row_i in 0..dnum {
                let mut dst = res.at_view_mut(row_i, col_i);
                let src = other.at_view(row_i, col_i);
                BE::glwe_decompress(module, &mut dst, &src);
            }
        }
    }
    res.set_noise(noise);
}

pub(crate) fn ggsw_decompress_derived<BE, R, O>(module: &Module<BE>, res: &mut R, other: &O)
where
    BE: DecompressionImpl,
    R: GGSWToBackendMut<BE> + GGSWInfos,
    O: GGSWCompressedToBackendRef<BE> + GGSWInfos,
{
    let noise = other.to_backend_ref().noise();
    {
        let mut res = res.to_backend_mut();
        let other = other.to_backend_ref();

        assert_eq!(res.rank(), other.rank());
        let dnum: usize = res.dnum().into();
        let rank: usize = res.rank().into();

        for row_i in 0..dnum {
            for col_j in 0..rank + 1 {
                let mut dst = res.at_view_mut(row_i, col_j);
                let src = other.at_view(row_i, col_j);
                BE::glwe_decompress(module, &mut dst, &src);
            }
        }
    }
    res.set_noise(noise);
}

pub(crate) fn glwe_public_key_decompress_derived<BE, R, O>(module: &Module<BE>, res: &mut R, other: &O)
where
    BE: DecompressionImpl,
    R: GLWEPublicKeyToBackendMut<BE> + GetDistributionMut + GLWEInfos,
    O: GLWEPublicKeyCompressedToBackendRef<BE> + GetDistribution + GLWEInfos,
{
    assert!(
        res.glwe_layout() == other.glwe_layout(),
        "invalid decompression: layouts differ"
    );
    {
        let mut res = res.to_backend_mut();
        let other = other.to_backend_ref();
        for l in 0..other.rank().as_usize() {
            BE::glwe_decompress(module, &mut res.at_view_mut(l), &other.at_view(l));
        }
    }
    res.set_noise(other.noise());
    *res.dist_mut() = *other.dist();
}

pub(crate) fn glwe_automorphism_key_decompress_derived<BE, R, O>(module: &Module<BE>, res: &mut R, other: &O)
where
    BE: DecompressionImpl,
    R: GGLWEToBackendMut<BE> + GGLWEInfos + SetGaloisElement,
    O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement,
{
    BE::gglwe_decompress(module, res, other);
    res.set_p(other.p());
}

pub(crate) fn glwe_switching_key_decompress_derived<BE, R, O>(module: &Module<BE>, res: &mut R, other: &O)
where
    BE: DecompressionImpl,
    R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
    O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GLWESwitchingKeyDegrees,
{
    BE::gglwe_decompress(module, res, other);

    *res.input_degree() = *other.input_degree();
    *res.output_degree() = *other.output_degree();
}

pub(crate) fn glwe_tensor_key_decompress_derived<BE, R, O>(module: &Module<BE>, res: &mut R, other: &O)
where
    BE: DecompressionImpl,
    R: GGLWEToBackendMut<BE> + GGLWEInfos,
    O: GGLWECompressedToBackendRef<BE> + GGLWEInfos,
{
    BE::gglwe_decompress(module, res, other);
}

pub(crate) fn glwe_to_lwe_key_decompress_derived<BE, R, O>(module: &Module<BE>, res: &mut R, other: &O)
where
    BE: DecompressionImpl,
    R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
    O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GLWESwitchingKeyDegrees,
{
    BE::glwe_switching_key_decompress(module, res, other);
}

pub(crate) fn lwe_switching_key_decompress_derived<BE, R, O>(module: &Module<BE>, res: &mut R, other: &O)
where
    BE: DecompressionImpl,
    R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
    O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GLWESwitchingKeyDegrees,
{
    BE::glwe_switching_key_decompress(module, res, other);
}

pub(crate) fn lwe_to_glwe_key_decompress_derived<BE, R, O>(module: &Module<BE>, res: &mut R, other: &O)
where
    BE: DecompressionImpl,
    R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
    O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GLWESwitchingKeyDegrees,
{
    BE::glwe_switching_key_decompress(module, res, other);
}

pub(crate) fn gglwe_to_ggsw_key_decompress_derived<BE, R, O>(module: &Module<BE>, res: &mut R, other: &O)
where
    BE: DecompressionImpl,
    R: GGLWEToGGSWKeyToBackendMut<BE>,
    O: GGLWEToGGSWKeyCompressedToBackendRef<BE>,
{
    let noise = other.to_backend_ref().noise();
    {
        let mut res = res.to_backend_mut();
        let other = other.to_backend_ref();
        assert_eq!(res.keys.len(), other.keys.len());
        for (a, b) in res.keys.iter().zip(other.keys.iter()) {
            assert_eq!(a.k_digit(), b.k_digit());
            assert_eq!(a.k_offset(), b.k_offset(), "gglwe_to_ggsw_key_decompress: row offsets differ");
            assert!(a.dnum() <= b.dnum());
        }
        for i in 0..res.keys.len() {
            BE::gglwe_decompress(module, &mut res.at_view_mut(i), &other.at_view(i));
        }
    }
    res.set_noise(noise);
}
