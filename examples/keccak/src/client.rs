//! The client: the secret, the evaluation keys, a state in and out.

use std::collections::HashMap;

use anyhow::Result;
use poulpy_ckks::{
    layouts::{BootstrappingKeySet, CKKSPlaintextAlloc, CKKSPlaintextOwned},
    prelude::*,
};
use poulpy_core::{
    GLWEAutomorphismKeyEncryptSk, GLWETensorKeyEncryptSk,
    api::{GLWESecretPreparedFactory, GLWESecretSampling},
    layouts::{
        BackendGLWESecret, BackendGLWESecretPrepared, GLWEAutomorphismKeyAlloc, GLWESecretAlloc, GLWETensorKeyAlloc, LWEInfos,
    },
};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, CyclotomicOrder, Module, ScratchArena, ScratchOwned, Standard},
    source::Source,
};

use crate::{
    Ct, Host,
    layout::{SLOTS, STATES, from_slots, to_slots},
    params::{BASE2K, DENSE_SECRET, LOG_N, OUTPUT_LOG_DELTA, Params},
};

type Buf = <Host as Backend>::OwnedBuf;

/// What the server evaluates with: the bootstrapping keys, and the rotation keys and tensor key of the round in a key
/// set of their own, which the server moves to its backend as it does the bootstrapping's.
pub struct EvaluationKeys {
    pub bootstrapping: BootstrappingKeySet<Buf, i64>,
    pub round: BootstrappingKeySet<Buf, i64>,
}

/// The holder of the secret.
pub struct Client {
    module: Module<Host>,
    sk_raw: BackendGLWESecret<Host>,
    sk: BackendGLWESecretPrepared<Host>,
    source_xs: Source,
    source_xe: Source,
    source_xa: Source,
}

/// Real slot values at scale `2^log_delta`, in a plaintext two bits wider, on the client's or the server's backend.
pub(crate) fn encode_slots<BE: Backend>(
    module: &Module<BE>,
    values: &[f64],
    log_delta: usize,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<CKKSPlaintextOwned<BE>>
where
    Module<BE>: CKKSPlaintextAlloc<BE> + CKKSEncodingHostOps<BE, f64, Standard>,
{
    let mut pt = module.ckks_plaintext_alloc_from_slots(SLOTS, BASE2K.into(), (log_delta + 2).into());
    pt.set_meta(GLWEPlaintextMeta {
        scale: Scale::Log(log_delta),
        slots: SlotsKind::Real,
        log_sparsity: 0,
        slot_layout: SlotLayout::Full,
    });
    module.ckks_encode_slots_host_into(&mut pt, values, &[0.0; SLOTS], scratch)?;
    Ok(pt)
}

pub(crate) fn encode_slots_tmp_bytes<BE: Backend>(module: &Module<BE>) -> usize
where
    Module<BE>: CKKSEncodingHostOps<BE, f64, Standard>,
{
    CKKSEncodingHostOps::<BE, f64, Standard>::ckks_encoding_host_tmp_bytes(module, SLOTS)
}

impl Client {
    /// The client of the secret and the randomness of `seed`, one child seed per role: whoever holds `seed` decrypts.
    pub fn new(params: &Params, seed: [u8; 32]) -> Self {
        let mut root = Source::new(seed);
        let module = Module::<Host>::new(1 << LOG_N);
        let sk_layout = params.bootstrapping.raised_layout().glwe_layout;
        let mut sk_raw = module.glwe_secret_alloc_from_infos(&sk_layout);
        module.glwe_secret_fill_ternary_hw(&mut sk_raw, DENSE_SECRET.hamming_weight, &mut Source::new(root.new_seed()));
        let mut sk = module.glwe_secret_prepared_alloc_from_infos(&sk_layout);
        module.glwe_secret_prepare(&mut sk, &sk_raw);
        Self {
            module,
            sk_raw,
            sk,
            source_xs: Source::new(root.new_seed()),
            source_xe: Source::new(root.new_seed()),
            source_xa: Source::new(root.new_seed()),
        }
    }

    /// The evaluation keys.
    pub fn keys(&mut self, params: &Params) -> Result<EvaluationKeys> {
        let module = &self.module;
        let boot = &params.bootstrapping;
        let (key, tsk) = (&params.rotation_key, &params.tensor_key);
        let mut scratch = ScratchOwned::<Host>::alloc(
            boot.keys
                .generate_tmp_bytes(module)
                .max(GLWEAutomorphismKeyEncryptSk::<Host, Standard>::glwe_automorphism_key_encrypt_sk_tmp_bytes(module, key))
                .max(GLWETensorKeyEncryptSk::<Host, Standard>::glwe_tensor_key_encrypt_sk_tmp_bytes(module, tsk)),
        );
        let bootstrapping = boot.plan.generate_keys(
            module,
            &self.sk_raw,
            &boot.keys,
            &mut self.source_xs,
            &mut self.source_xe,
            &mut self.source_xa,
            &mut scratch.borrow(),
        )?;
        let mut rotations = HashMap::new();
        for p in params.round_galois_elements(module.cyclotomic_order::<Standard>()) {
            let mut atk = module.glwe_automorphism_key_alloc_from_infos(key);
            module.glwe_automorphism_key_encrypt_sk(
                &mut atk,
                p,
                &self.sk_raw,
                &mut self.source_xe,
                &mut self.source_xa,
                &mut scratch.borrow(),
            );
            rotations.insert(p, atk);
        }
        let mut tensor = module.glwe_tensor_key_alloc_from_infos(tsk);
        module.glwe_tensor_key_encrypt_sk(
            &mut tensor,
            &self.sk_raw,
            &mut self.source_xa,
            &mut self.source_xe,
            &mut scratch.borrow(),
        );
        Ok(EvaluationKeys {
            bootstrapping,
            round: BootstrappingKeySet {
                rotation_keys: rotations,
                tensor_key: Some(tensor),
                ci_tensor_key: None,
                encapsulation_keys: None,
                first_factor_keys: HashMap::new(),
                conversion_keys: None,
            },
        })
    }

    /// The states as the input of the first round: their bits at the bootstrapping output's scale and width.
    pub fn encrypt(&mut self, params: &Params, states: &[[u64; 25]; STATES]) -> Result<Ct> {
        let module = &self.module;
        let values = to_slots(states);
        let mut ct = module.ckks_ciphertext_alloc(BASE2K.into(), params.bootstrapping.output_k.into());
        let mut scratch = ScratchOwned::<Host>::alloc(
            encode_slots_tmp_bytes(module).max(CKKSEncryptOps::<Host, Standard>::ckks_encrypt_sk_tmp_bytes(module, &ct)),
        );
        let pt = encode_slots(module, &values, OUTPUT_LOG_DELTA, &mut scratch.borrow())?;
        module.ckks_encrypt_sk(
            &mut ct,
            &pt,
            &self.sk,
            &mut self.source_xe,
            &mut self.source_xa,
            &mut scratch.borrow(),
        )?;
        Ok(ct)
    }

    /// The states: each slot rounded, then reduced mod 2.
    pub fn decrypt(&self, ct: &Ct) -> Result<[[u64; 25]; STATES]> {
        let module = &self.module;
        let mut pt = module.ckks_plaintext_alloc_from_slots(SLOTS, ct.base2k(), (ct.log_delta() + 10).into());
        pt.set_meta(ct.meta());
        pt.set_log_budget(10);
        let mut scratch = ScratchOwned::<Host>::alloc(
            CKKSDecryptOps::<Host, Standard>::ckks_decrypt_tmp_bytes(module, &pt, ct).max(encode_slots_tmp_bytes(module)),
        );
        module.ckks_decrypt(&mut pt, ct, &self.sk, &mut scratch.borrow())?;
        let (mut re, mut im) = (vec![0.0; SLOTS], vec![0.0; SLOTS]);
        module.ckks_decode_slots_host_into(&pt, &mut re, &mut im, &mut scratch.borrow())?;
        Ok(from_slots(&re))
    }
}
