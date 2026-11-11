//! Decompression of seed-compressed GLWE and LWE ciphertexts.

use poulpy_hal::{
    api::VecZnxCopy,
    layouts::{Backend, Module, Ring, Standard},
};

use crate::{
    api::{GLWEFill, LWEFillMask},
    layouts::{
        GLWECompressedToBackendRef, GLWEInfos, GLWEToBackendMut, GetDegree, LWECompressedToBackendRef, LWEInfos, LWEToBackendMut,
        SetBase2k, SetGLWEPlaintextInfos, operand_degree,
    },
};

/// Reference bodies of the required [`DecompressionImpl`](crate::oep::DecompressionImpl) methods, which
/// [`impl_decompression_reference_full!`](crate::impl_decompression_reference_full) forwards to.
pub trait DecompressionReference<BE: Backend, Rg: Ring = Standard> {
    fn glwe_decompress_reference<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWEToBackendMut<BE, Rg> + SetBase2k + SetGLWEPlaintextInfos,
        O: GLWECompressedToBackendRef<BE, Rg> + GLWEInfos;

    fn lwe_decompress_reference<R, O>(&self, res: &mut R, other: &O)
    where
        R: LWEToBackendMut<BE, Rg> + LWEInfos + SetBase2k,
        O: LWECompressedToBackendRef<BE, Rg>;
}

impl<BE: Backend, Rg: Ring> DecompressionReference<BE, Rg> for Module<BE>
where
    Module<BE>: GLWEFill<BE, Rg> + LWEFillMask<BE, Rg> + VecZnxCopy<BE, Rg>,
{
    fn glwe_decompress_reference<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWEToBackendMut<BE, Rg> + SetBase2k + SetGLWEPlaintextInfos,
        O: GLWECompressedToBackendRef<BE, Rg> + GLWEInfos,
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
        self.glwe_fill_mask_from_seed(res, other.seed);
        res.set_noise(noise);
        res.set_canonical(true);
        res.set_plaintext_meta(plaintext_meta);
    }

    fn lwe_decompress_reference<R, O>(&self, res: &mut R, other: &O)
    where
        R: LWEToBackendMut<BE, Rg> + LWEInfos + SetBase2k,
        O: LWECompressedToBackendRef<BE, Rg>,
    {
        let other = other.to_backend_ref();

        {
            let mut res = res.to_backend_mut();
            assert_eq!(res.base2k(), other.base2k(), "lwe_decompress: base2k mismatch");
            assert_eq!(res.size(), other.size(), "lwe_decompress: limb count mismatch");
            self.vec_znx_copy(&mut res.body, 0, &other.data, 0);
        }
        self.lwe_fill_mask_from_seed(other.base2k().into(), res, other.seed);
        res.set_base2k(other.base2k());
        res.set_noise(None);
    }
}
