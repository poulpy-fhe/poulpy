use std::panic::{AssertUnwindSafe, catch_unwind};

use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc},
    layouts::{Backend, Module, ScratchOwned},
};

use crate::{
    GLWEDecrypt,
    layouts::{GLWESecretPreparedFactory, ModuleCoreAlloc},
};

/// Decryption rejects mismatched plaintext and secret key degrees in all builds.
pub fn test_glwe_decrypt_degree_mismatch<BE: Backend>()
where
    Module<BE>: ModuleNew<BE> + GLWEDecrypt<BE> + GLWESecretPreparedFactory<BE>,
{
    let n = 2 * BE::MIN_DEGREE.max(16);
    let module = Module::<BE>::new(n as u64);
    let ct = module.glwe_alloc(8usize.into(), 8usize.into(), 1usize.into());
    let mut scratch = ScratchOwned::<BE>::alloc(module.glwe_decrypt_tmp_bytes(&ct));
    for other_n in [n / 2, n * 2] {
        let other = Module::<BE>::new(other_n as u64);
        for (pt_module, sk_module, message) in [
            (&module, &other, "GLWE ciphertext and secret key degrees must match"),
            (&other, &module, "GLWE plaintext and secret key degrees must match"),
        ] {
            let mut pt = pt_module.glwe_plaintext_alloc(8usize.into(), 8usize.into());
            let sk = sk_module.glwe_secret_prepared_alloc(1usize.into());
            let err = catch_unwind(AssertUnwindSafe(|| {
                module.glwe_decrypt(&ct, &mut pt, &sk, &mut scratch.arena());
            }))
            .expect_err("decryption accepted mismatched degrees");
            assert!(err.downcast_ref::<String>().is_some_and(|s| s.contains(message)));
        }
    }
}
