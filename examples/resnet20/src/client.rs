use anyhow::Result;
use poulpy_ckks::{
    SetCKKSInfos,
    api::{CKKSDecryptOps, CKKSEncodingHostOps, CKKSEncryptOps},
    layouts::{CKKSCiphertextAlloc, CKKSCiphertextOwned, CKKSPlaintextAlloc},
};
use poulpy_core::{
    api::{GLWESecretPreparedFactory, GLWESecretSampling},
    layouts::{BackendGLWESecret, BackendGLWESecretPrepared, GLWESecretAlloc},
};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Module, ScratchOwned},
    source::Source,
};

use crate::{Host, KeySet, Params};

/// The client: the secret, the keys, the image in, the logits out.
pub struct Client {
    params: Params,
    module: Module<Host>,
    sk_raw: BackendGLWESecret<Host>,
    sk: BackendGLWESecretPrepared<Host>,
    scratch: ScratchOwned<Host>,
    source_xs: Source,
    source_xe: Source,
    source_xa: Source,
}

impl Client {
    /// The client of the secret and the randomness of `seed`, one child seed per role: whoever holds `seed` decrypts.
    pub fn new(params: &Params, seed: [u8; 32]) -> Self {
        let mut root = Source::new(seed);
        let preset = params.preset();
        let n = preset.n();
        let module = Module::<Host>::new(n as u64);
        let boot = preset.bootstrap_layout();
        let mut sk_raw = module.glwe_secret_alloc_from_infos(&boot.glwe_layout);
        module.glwe_secret_fill_ternary_hw(
            &mut sk_raw,
            preset.dense_secret_hamming_weight(),
            &mut Source::new(root.new_seed()),
        );
        let mut sk = module.glwe_secret_prepared_alloc_from_infos(&boot.glwe_layout);
        module.glwe_secret_prepare(&mut sk, &sk_raw);
        let scratch = ScratchOwned::<Host>::alloc(
            <Module<Host> as CKKSEncryptOps<Host>>::ckks_encrypt_sk_tmp_bytes(&module, &boot)
                .max(CKKSEncodingHostOps::<Host, f64>::ckks_encoding_host_tmp_bytes(&module, n / 2))
                .max(<Module<Host> as CKKSDecryptOps<Host>>::ckks_decrypt_tmp_bytes(
                    &module, &boot, &boot,
                )),
        );
        Self {
            params: params.clone(),
            module,
            sk_raw,
            sk,
            scratch,
            source_xs: Source::new(root.new_seed()),
            source_xe: Source::new(root.new_seed()),
            source_xa: Source::new(root.new_seed()),
        }
    }

    /// The bootstrapping keys.
    pub fn keys(&mut self) -> Result<KeySet> {
        let preset = self.params.preset();
        let bytes = preset.keys_layout().generate_tmp_bytes(&self.module);
        preset.plan().generate_keys(
            &self.module,
            &self.sk_raw,
            preset.keys_layout(),
            &mut self.source_xs,
            &mut self.source_xe,
            &mut self.source_xa,
            &mut ScratchOwned::<Host>::alloc(bytes).borrow(),
        )
    }

    /// `image`, `[3][32][32]`, in the real slots, as the activation the first layer reads.
    pub fn encrypt_image(&mut self, image: &[f64]) -> Result<CKKSCiphertextOwned<Host>> {
        let (params, base2k) = (&self.params, self.params.preset().base2k().into());
        let re = params.layouts().image.pack_slots(image)?;
        let mut pt = self
            .module
            .ckks_plaintext_alloc_from_slots(re.len(), base2k, params.in_k().into());
        pt.set_meta(params.activation_meta());
        self.module
            .ckks_encode_slots_host_into(&mut pt, &re, &vec![0.0; re.len()], &mut self.scratch.borrow())?;
        let mut ct = self.module.ckks_ciphertext_alloc(base2k, params.in_k().into());
        self.module.ckks_encrypt_sk(
            &mut ct,
            &pt,
            &self.sk,
            &mut self.source_xe,
            &mut self.source_xa,
            &mut self.scratch.borrow(),
        )?;
        Ok(ct)
    }

    /// The ten logits in the coefficients of `ct`, the output of an inference.
    pub fn decrypt_logits(&mut self, ct: &CKKSCiphertextOwned<Host>) -> Result<Vec<f64>> {
        let mut pt = self.module.ckks_plaintext_alloc_from_infos(ct);
        self.module.ckks_decrypt(&mut pt, ct, &self.sk, &mut self.scratch.borrow())?;
        let mut coeffs = vec![0.0; self.module.n()];
        self.module
            .ckks_decode_coeffs_host_into(&pt, &mut coeffs, &mut self.scratch.borrow())?;
        self.params.layouts().logits.unpack(&coeffs)
    }
}

#[cfg(test)]
mod tests {
    use poulpy_core::layouts::LWEInfos;

    use super::Client;
    use crate::{Params, model::test_images};

    /// The preset of the network derives, and the client encrypts an image at the width the server reads.
    #[test]
    fn the_client_encrypts_an_image() {
        let params = Params::new().unwrap();
        let (image, _) = &test_images(1).unwrap()[0];
        let ct = Client::new(&params, [0; 32]).encrypt_image(image).unwrap();
        assert_eq!(ct.k().as_usize(), params.in_k());
    }
}
