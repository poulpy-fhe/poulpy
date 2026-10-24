//! Decompression of seed-compressed ciphertexts and keys.

use poulpy_hal::{
    api::VecZnxCopy,
    layouts::{Backend, Module},
};

use crate::{
    GetDistribution, GetDistributionMut,
    api::{GGLWEDecompress, GLWEDecompress, GLWEMaskFill, GLWESwitchingKeyDecompress, LWEFillMask},
    layouts::{
        GGLWECompressedToBackendRef, GGLWEInfos, GGLWEToBackendMut, GGLWEToGGSWKeyCompressedToBackendRef,
        GGLWEToGGSWKeyToBackendMut, GGSWCompressedToBackendRef, GGSWInfos, GGSWToBackendMut, GLWECompressedToBackendRef,
        GLWEInfos, GLWEPublicKeyAtViewMut, GLWEPublicKeyCompressedToBackendRef, GLWEPublicKeyToBackendMut,
        GLWESwitchingKeyDegrees, GLWESwitchingKeyDegreesMut, GLWEToBackendMut, GetDegree, GetGaloisElement,
        LWECompressedToBackendRef, LWEInfos, LWEToBackendMut, SetBase2k, SetGLWEPlaintextInfos, SetGaloisElement, operand_degree,
    },
};

/// Reference decompression bodies. Backend dispatch is selected independently through
/// [`DecompressionImpl`](crate::oep::DecompressionImpl).
pub trait DecompressionReference<BE: Backend> {
    fn decompress_glwe_reference<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWEToBackendMut<BE> + SetBase2k + SetGLWEPlaintextInfos,
        O: GLWECompressedToBackendRef<BE> + GLWEInfos;

    fn decompress_gglwe_reference<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos;

    fn decompress_ggsw_reference<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        O: GGSWCompressedToBackendRef<BE> + GGSWInfos;

    fn decompress_glwe_public_key_reference<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWEPublicKeyToBackendMut<BE> + GetDistributionMut + GLWEInfos,
        O: GLWEPublicKeyCompressedToBackendRef<BE> + GetDistribution + GLWEInfos;

    fn decompress_lwe_reference<R, O>(&self, res: &mut R, other: &O)
    where
        R: LWEToBackendMut<BE> + LWEInfos + SetBase2k,
        O: LWECompressedToBackendRef<BE>;

    fn decompress_automorphism_key_reference<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + SetGaloisElement,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement;

    fn decompress_glwe_switching_key_reference<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GLWESwitchingKeyDegrees;

    fn decompress_tensor_key_reference<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos;

    fn decompress_glwe_to_lwe_key_reference<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GLWESwitchingKeyDegrees;

    fn decompress_lwe_switching_key_reference<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GLWESwitchingKeyDegrees;

    fn decompress_lwe_to_glwe_key_reference<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GLWESwitchingKeyDegrees;

    fn decompress_gglwe_to_ggsw_key_reference<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToGGSWKeyToBackendMut<BE>,
        O: GGLWEToGGSWKeyCompressedToBackendRef<BE>;
}

impl<BE: Backend> DecompressionReference<BE> for Module<BE>
where
    Module<BE>: GLWEMaskFill<BE>
        + LWEFillMask<BE>
        + VecZnxCopy<BE>
        + GLWEDecompress<BE>
        + GGLWEDecompress<BE>
        + GLWESwitchingKeyDecompress<BE>,
{
    fn decompress_glwe_reference<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWEToBackendMut<BE> + SetBase2k + SetGLWEPlaintextInfos,
        O: GLWECompressedToBackendRef<BE> + GLWEInfos,
    {
        let noise = other.to_backend_ref().noise();
        let other = other.to_backend_ref();
        let plaintext_meta = other.plaintext_meta;
        {
            let res = &mut res.to_backend_mut();
            operand_degree(self.ring_degree().as_usize(), &[res.n(), other.n()]);

            assert_eq!(res.glwe_layout(), other.glwe_layout());

            self.vec_znx_copy(&mut res.data, 0, &other.data, 0);
        }
        res.set_base2k(other.base2k());
        self.fill_glwe_mask_from_seed(res, other.seed);
        res.set_noise(noise);
        res.set_canonical(true);
        res.set_plaintext_meta(plaintext_meta);
    }

    fn decompress_gglwe_reference<R, O>(&self, res: &mut R, other: &O)
    where
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
                    self.decompress_glwe(&mut dst, &src);
                }
            }
        }
        res.set_noise(noise);
    }

    fn decompress_ggsw_reference<R, O>(&self, res: &mut R, other: &O)
    where
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
                    self.decompress_glwe(&mut dst, &src);
                }
            }
        }
        res.set_noise(noise);
    }

    fn decompress_glwe_public_key_reference<R, O>(&self, res: &mut R, other: &O)
    where
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
                self.decompress_glwe(&mut res.at_view_mut(l), &other.at_view(l));
            }
        }
        res.set_noise(other.noise());
        *res.dist_mut() = *other.dist();
    }

    fn decompress_lwe_reference<R, O>(&self, res: &mut R, other: &O)
    where
        R: LWEToBackendMut<BE> + LWEInfos + SetBase2k,
        O: LWECompressedToBackendRef<BE>,
    {
        let other = other.to_backend_ref();

        {
            let mut res = res.to_backend_mut();
            assert_eq!(res.base2k(), other.base2k(), "decompress_lwe: base2k mismatch");
            assert_eq!(res.size(), other.size(), "decompress_lwe: limb count mismatch");
            self.vec_znx_copy(&mut res.body, 0, &other.data, 0);
        }
        self.fill_lwe_mask_from_seed(other.base2k().into(), res, other.seed);
        res.set_base2k(other.base2k());
        res.set_noise(None);
    }

    fn decompress_automorphism_key_reference<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + SetGaloisElement,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement,
    {
        self.decompress_gglwe(res, other);
        res.set_p(other.p());
    }

    fn decompress_glwe_switching_key_reference<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GLWESwitchingKeyDegrees,
    {
        self.decompress_gglwe(res, other);

        *res.input_degree() = *other.input_degree();
        *res.output_degree() = *other.output_degree();
    }

    fn decompress_tensor_key_reference<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos,
    {
        self.decompress_gglwe(res, other);
    }

    fn decompress_glwe_to_lwe_key_reference<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GLWESwitchingKeyDegrees,
    {
        self.decompress_glwe_switching_key(res, other);
    }

    fn decompress_lwe_switching_key_reference<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GLWESwitchingKeyDegrees,
    {
        self.decompress_glwe_switching_key(res, other);
    }

    fn decompress_lwe_to_glwe_key_reference<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GLWESwitchingKeyDegrees,
    {
        self.decompress_glwe_switching_key(res, other);
    }

    fn decompress_gglwe_to_ggsw_key_reference<R, O>(&self, res: &mut R, other: &O)
    where
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
                        self.decompress_glwe(&mut dst, &src);
                    }
                }
            }
        }
        res.set_noise(noise);
    }
}
