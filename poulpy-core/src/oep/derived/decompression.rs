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

pub(crate) fn decompress_gglwe_derived<BE, R, O>(module: &Module<BE>, res: &mut R, other: &O)
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
        assert_eq!(res.k_offset(), other.k_offset(), "decompress_gglwe: row offsets differ");
        assert!(res.dnum() <= other.dnum());

        let rank_in: usize = res.rank_in().into();
        let dnum: usize = res.dnum().into();
        for col_i in 0..rank_in {
            for row_i in 0..dnum {
                let mut dst = res.at_view_mut(row_i, col_i);
                let src = other.at_view(row_i, col_i);
                BE::decompress_glwe(module, &mut dst, &src);
            }
        }
    }
    res.set_noise(noise);
}

pub(crate) fn decompress_ggsw_derived<BE, R, O>(module: &Module<BE>, res: &mut R, other: &O)
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
                BE::decompress_glwe(module, &mut dst, &src);
            }
        }
    }
    res.set_noise(noise);
}

pub(crate) fn decompress_glwe_public_key_derived<BE, R, O>(module: &Module<BE>, res: &mut R, other: &O)
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
            BE::decompress_glwe(module, &mut res.at_view_mut(l), &other.at_view(l));
        }
    }
    res.set_noise(other.noise());
    *res.dist_mut() = *other.dist();
}

pub(crate) fn decompress_automorphism_key_derived<BE, R, O>(module: &Module<BE>, res: &mut R, other: &O)
where
    BE: DecompressionImpl,
    R: GGLWEToBackendMut<BE> + GGLWEInfos + SetGaloisElement,
    O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement,
{
    BE::decompress_gglwe(module, res, other);
    res.set_p(other.p());
}

pub(crate) fn decompress_glwe_switching_key_derived<BE, R, O>(module: &Module<BE>, res: &mut R, other: &O)
where
    BE: DecompressionImpl,
    R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
    O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GLWESwitchingKeyDegrees,
{
    BE::decompress_gglwe(module, res, other);

    *res.input_degree() = *other.input_degree();
    *res.output_degree() = *other.output_degree();
}

pub(crate) fn decompress_tensor_key_derived<BE, R, O>(module: &Module<BE>, res: &mut R, other: &O)
where
    BE: DecompressionImpl,
    R: GGLWEToBackendMut<BE> + GGLWEInfos,
    O: GGLWECompressedToBackendRef<BE> + GGLWEInfos,
{
    BE::decompress_gglwe(module, res, other);
}

pub(crate) fn decompress_glwe_to_lwe_key_derived<BE, R, O>(module: &Module<BE>, res: &mut R, other: &O)
where
    BE: DecompressionImpl,
    R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
    O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GLWESwitchingKeyDegrees,
{
    BE::decompress_glwe_switching_key(module, res, other);
}

pub(crate) fn decompress_lwe_switching_key_derived<BE, R, O>(module: &Module<BE>, res: &mut R, other: &O)
where
    BE: DecompressionImpl,
    R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
    O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GLWESwitchingKeyDegrees,
{
    BE::decompress_glwe_switching_key(module, res, other);
}

pub(crate) fn decompress_lwe_to_glwe_key_derived<BE, R, O>(module: &Module<BE>, res: &mut R, other: &O)
where
    BE: DecompressionImpl,
    R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
    O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GLWESwitchingKeyDegrees,
{
    BE::decompress_glwe_switching_key(module, res, other);
}

pub(crate) fn decompress_gglwe_to_ggsw_key_derived<BE, R, O>(module: &Module<BE>, res: &mut R, other: &O)
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
        for i in 0..res.keys.len() {
            let mut a = res.at_view_mut(i);
            let b = other.at_view(i);
            assert_eq!(a.k_digit(), b.k_digit());
            assert!(a.dnum() <= b.dnum());

            let rank_in: usize = a.rank_in().into();
            let dnum: usize = a.dnum().into();
            for col_i in 0..rank_in {
                for row_i in 0..dnum {
                    let mut dst = a.at_view_mut(row_i, col_i);
                    let src = b.at_view(row_i, col_i);
                    BE::decompress_glwe(module, &mut dst, &src);
                }
            }
        }
    }
    res.set_noise(noise);
}
