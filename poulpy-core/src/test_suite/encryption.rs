use std::panic::{AssertUnwindSafe, catch_unwind};

use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc},
    layouts::{Backend, Module, ScratchOwned},
    source::Source,
};

use super::decryption::panic_message;
use crate::{
    EncryptionLayout, GLWEEncryptSk,
    layouts::{GLWELayout, GLWESecretPreparedFactory, ModuleCoreAlloc},
};

/// Secret-key encryption rejects mismatched operand degrees and degrees above the module in all builds.
pub fn test_glwe_encrypt_degree_mismatch<BE: Backend>()
where
    Module<BE>: ModuleNew<BE> + GLWEEncryptSk<BE> + GLWESecretPreparedFactory<BE>,
{
    let n = 2 * BE::MIN_DEGREE.max(16);
    let module = Module::<BE>::new(n as u64);
    let enc = EncryptionLayout::new_from_default_sigma(GLWELayout {
        n: n.into(),
        base2k: 8usize.into(),
        k: 8usize.into(),
        rank: 1usize.into(),
    })
    .unwrap();
    let mut scratch = ScratchOwned::<BE>::alloc(0);
    let mut xe = Source::new([1; 32]);
    let mut xa = Source::new([2; 32]);

    for other_n in [n / 2, n * 2] {
        let other = Module::<BE>::new(other_n as u64);
        for (ct_module, pt_module, sk_module, message) in [
            (&other, &module, &module, "operand degrees differ"),
            (&module, &other, &module, "operand degrees differ"),
            (&module, &module, &other, "operand degrees differ"),
            (&other, &other, &other, "operand degree exceeds the module degree"),
        ]
        .into_iter()
        // Operands sharing a smaller degree are valid.
        .take(if other_n > n { 4 } else { 3 })
        {
            let mut ct = ct_module.glwe_alloc(8usize.into(), 8usize.into(), 1usize.into());
            let pt = pt_module.glwe_plaintext_alloc(8usize.into(), 8usize.into());
            let sk = sk_module.glwe_secret_prepared_alloc(1usize.into());
            let err = catch_unwind(AssertUnwindSafe(|| {
                module.glwe_encrypt_sk(&mut ct, &pt, &sk, &enc, &mut xe, &mut xa, &mut scratch.arena());
            }))
            .expect_err("encryption accepted mismatched degrees");
            assert!(panic_message(&*err).contains(message));

            if ct_module.n() != n || sk_module.n() != n {
                let err = catch_unwind(AssertUnwindSafe(|| {
                    module.glwe_encrypt_zero_sk(&mut ct, &sk, &enc, &mut xe, &mut xa, &mut scratch.arena());
                }))
                .expect_err("zero encryption accepted mismatched degrees");
                assert!(panic_message(&*err).contains(message));
            }
        }
    }
}
