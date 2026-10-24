//! Shared code for the runnable walkthroughs.
//!
//! Deterministic seeds make these examples reproducible.
//! Applications must use their own cryptographic randomness and parameter selection.

use anyhow::Result;
use poulpy_ckks::{
    CKKSLayout, CKKSMeta, SetCKKSInfos,
    api::CKKSAllOpsTmpBytes,
    layouts::{CKKSCiphertext, CKKSModuleAlloc},
    test_suite::{
        CKKSTestParams,
        helpers::{
            assert_precision_for_log_delta, ckks_decrypt_decode, ckks_encrypt_with_prec, ckks_spec,
            gen_sk_with_raw, gen_tsk,
        },
        reference_encoder::ReferenceEncoder,
    },
};
use poulpy_core::layouts::{GLWESecretPrepared, GLWETensorKeyPrepared};
use poulpy_cpu_ref::{FFT64ReimTable, NTT4x30Ref};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module, ScratchOwned},
};

pub const N: usize = 4096;
pub const SLOTS: usize = N / 2;
pub const BASE2K: usize = 52;
pub const LOG_DELTA: usize = 32;

pub type Backend = NTT4x30Ref;
pub type Ciphertext = CKKSCiphertext<Vec<u8>, i64>;
pub type SecretKey = GLWESecretPrepared<Vec<u8>, Backend>;
pub type TensorKey = GLWETensorKeyPrepared<Vec<u8>, Backend>;

/// Modules and encoder that do not depend on the circuit depth.
pub struct Setup {
    pub module: Module<Backend>,
    pub host_module: Module<HostBytesBackend>,
    pub encoder: ReferenceEncoder<FFT64ReimTable<f64>>,
}

/// Keys and scratch storage sized for one walkthrough circuit.
pub struct Context {
    pub params: CKKSTestParams,
    pub module: Module<Backend>,
    pub host_module: Module<HostBytesBackend>,
    pub encoder: ReferenceEncoder<FFT64ReimTable<f64>>,
    pub sk: SecretKey,
    pub tsk: TensorKey,
    pub scratch: ScratchOwned<Backend>,
}

impl Setup {
    pub fn new() -> Result<Self> {
        Ok(Self {
            module: Module::<Backend>::new(N as u64),
            host_module: Module::<HostBytesBackend>::new(N as u64),
            encoder: ReferenceEncoder::<FFT64ReimTable<f64>>::new::<f64>(SLOTS)?,
        })
    }

    /// Sizes the modulus and scratch arena, then prepares the CKKS keys.
    pub fn prepare(
        self,
        consumed_bits: usize,
        operation_tmp_bytes: impl FnOnce(&Module<Backend>, &Ciphertext, &CKKSTestParams) -> usize,
    ) -> Context {
        let params = params_for(consumed_bits);
        let mut sizing = self
            .module
            .ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
        sizing.set_meta(params.prec().meta);

        // Key preparation and function evaluation share this temporary memory.
        let scratch_bytes = operation_tmp_bytes(&self.module, &sizing, &params).max(
            self.module
                .ckks_all_ops_tmp_bytes(&sizing, &params.tsk_layout(), &coeff_layout()),
        );
        let mut scratch = ScratchOwned::<Backend>::alloc(scratch_bytes);
        let (sk_raw, sk) = gen_sk_with_raw(&params, &self.module, &self.host_module, [0x11; 32]);
        let tsk = gen_tsk(&params, &self.module, &sk_raw, &mut scratch.borrow());

        Context {
            params,
            module: self.module,
            host_module: self.host_module,
            encoder: self.encoder,
            sk,
            tsk,
            scratch,
        }
    }
}

impl Context {
    /// Encodes and encrypts a dense real slot vector.
    pub fn encrypt(&mut self, values: &[f64]) -> Ciphertext {
        assert_eq!(values.len(), SLOTS);
        let imaginary = vec![0.0; SLOTS];
        ckks_encrypt_with_prec(
            &self.params,
            &self.module,
            &self.host_module,
            &self.encoder,
            &self.sk,
            self.params.k,
            values,
            &imaginary,
            self.params.prec(),
            &mut self.scratch.borrow(),
        )
    }

    /// Allocates an output ciphertext at the circuit modulus.
    pub fn output(&self) -> Ciphertext {
        self.module
            .ckks_ciphertext_alloc(self.params.base2k.into(), self.params.k.into())
    }

    /// Decrypts and returns the real slots.
    pub fn decrypt(&mut self, ciphertext: &Ciphertext) -> Vec<f64> {
        ckks_decrypt_decode::<Backend, f64, FFT64ReimTable<f64>>(
            &self.params,
            &self.module,
            &self.encoder,
            ciphertext,
            &self.sk,
            &mut self.scratch.borrow(),
        )
        .0
    }
}

pub fn coeff_layout() -> CKKSLayout {
    ckks_spec(N, BASE2K, LOG_DELTA, BASE2K)
}

pub fn dense_interval(a: f64, b: f64) -> Vec<f64> {
    (0..SLOTS)
        .map(|i| a + (b - a) * i as f64 / (SLOTS - 1) as f64)
        .collect()
}

pub fn verify(label: &str, got: &[f64], want: &[f64], log_delta: usize) {
    // Poulpy's helper accounts for the expected precision loss from encrypted evaluation.
    assert_precision_for_log_delta(label, got, want, log_delta, N);
    println!("{label}: verified {} encrypted values", got.len());
}

fn params_for(consumed_bits: usize) -> CKKSTestParams {
    let reserve = 2 * LOG_DELTA + 2 * BASE2K;
    let k = (consumed_bits + reserve).next_multiple_of(BASE2K);
    CKKSTestParams {
        n: N,
        base2k: BASE2K,
        k,
        prec_meta: CKKSMeta {
            log_sparsity: 0,
            log_delta: LOG_DELTA,
            slots: Default::default(),
        },
        prec_log_budget: k - LOG_DELTA,
        hw: 192,
        dsize: 1,
        rank: 1,
    }
}
