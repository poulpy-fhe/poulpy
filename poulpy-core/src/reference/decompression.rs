//! Decompression of seed-compressed GLWE and LWE ciphertexts.

use poulpy_hal::{
    api::VecZnxCopy,
    layouts::{Backend, Module},
};

use crate::{
    api::{GLWEMaskFill, LWEFillMask},
    layouts::{
        GLWECompressedToBackendRef, GLWEInfos, GLWEToBackendMut, GetDegree, LWECompressedToBackendRef, LWEInfos, LWEToBackendMut,
        SetBase2k, SetGLWEPlaintextInfos, operand_degree,
    },
};

/// Reference decompression bodies. Backend dispatch is selected independently through
/// [`DecompressionImpl`](crate::oep::DecompressionImpl).
pub trait DecompressionReference<BE: Backend> {
    fn decompress_glwe_reference<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWEToBackendMut<BE> + SetBase2k + SetGLWEPlaintextInfos,
        O: GLWECompressedToBackendRef<BE> + GLWEInfos;

    fn decompress_lwe_reference<R, O>(&self, res: &mut R, other: &O)
    where
        R: LWEToBackendMut<BE> + LWEInfos + SetBase2k,
        O: LWECompressedToBackendRef<BE>;
}

impl<BE: Backend> DecompressionReference<BE> for Module<BE>
where
    Module<BE>: GLWEMaskFill<BE> + LWEFillMask<BE> + VecZnxCopy<BE>,
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
}
