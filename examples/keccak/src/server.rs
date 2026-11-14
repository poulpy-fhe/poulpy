//! The server: the compiled bootstrapping, the prepared keys, the encoded maps and round constants, and the rounds.
//!
//! The ciphertexts stay on the server's backend between rounds: [`Server::upload`] and [`Server::download`] move them
//! from and to the client.

use std::collections::HashMap;

use anyhow::Result;
use poulpy_ckks::{
    CKKSCtBounds,
    api::{CKKSBootstrappingOps, CKKSLinearTransformationOps, LinearTransformation, LinearTransformationPrepared},
    encoding::{ckks_linear_transformation_encode_from_diagonals, ckks_linear_transformation_encode_tmp_bytes},
    layouts::{BootstrappingContextPrepared, BootstrappingKeysPrepared, CKKSPlaintextOwned, ComplexDiagonals, EncodedLut},
    prelude::*,
};
use poulpy_core::{
    GGLWEBytesOf,
    api::{GLWEAutomorphismKeyPreparedFactory, GLWETensorKeyPreparedFactory, TransferInto},
    layouts::{
        BackendGLWEAutomorphismKeyPrepared, BackendGLWETensorKeyPrepared, Diagonals, GGLWEInfos, GLWELayout, LWEInfos, Rank,
        SetK, TorusPrecision,
    },
};
use poulpy_hal::{
    api::{ModuleSynchronize, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, ScratchOwned, Standard},
};

use crate::{
    Ct, Host, ServerBackend, ServerCt,
    client::{EvaluationKeys, encode_slots, encode_slots_tmp_bytes},
    keccak::ROUND_CONSTANTS,
    layout::{SLOTS, SlotMap, iota_slots},
    params::{BASE2K, LOG_DELTA, LOG_DELTA_MAPS, LOG_N, OUTPUT_LOG_DELTA, Params, bsgs_layout, diagonals},
};

/// The evaluator.
pub struct Server {
    /// Widths of the bootstrapping's input, output and raised ciphertexts.
    input_k: usize,
    output_k: usize,
    raised_k: usize,
    module: Module<ServerBackend>,
    /// The client's backend, which ciphertexts come from and go back to.
    host: Module<Host>,
    context: BootstrappingContextPrepared<ServerBackend, f64>,
    bootstrapping_keys: BootstrappingKeysPrepared<<ServerBackend as Backend>::OwnedBuf, ServerBackend>,
    lut: EncodedLut<CKKSPlaintextOwned<ServerBackend>>,
    /// theta, rho and pi.
    linear: LinearTransformationPrepared<ServerBackend, Standard>,
    /// The `x + 1` and `x + 2` reads of chi.
    shift1: LinearTransformationPrepared<ServerBackend, Standard>,
    shift2: LinearTransformationPrepared<ServerBackend, Standard>,
    /// The round constants as slot values.
    round_constants: Vec<CKKSPlaintextOwned<ServerBackend>>,
    rotation_keys: HashMap<i64, BackendGLWEAutomorphismKeyPrepared<ServerBackend>>,
    tensor_key: BackendGLWETensorKeyPrepared<ServerBackend>,
    scratch: ScratchOwned<ServerBackend>,
}

/// The width `ct` was allocated with, which a transfer copies whatever its `k`.
fn allocated_k<C: CKKSCtBounds>(ct: &C) -> TorusPrecision {
    (ct.max_size() * ct.base2k().as_usize()).into()
}

/// A slot map as a linear transformation of real diagonals at scale `2^LOG_DELTA_MAPS`, prepared.
fn prepare_map(module: &Module<ServerBackend>, map: &SlotMap) -> Result<LinearTransformationPrepared<ServerBackend, Standard>> {
    let re = diagonals(map);
    let strategy = bsgs_layout(&re).strategy;
    let mut scratch =
        ScratchOwned::<ServerBackend>::alloc(
            ckks_linear_transformation_encode_tmp_bytes::<ServerBackend, _, f64, Standard>(module, SLOTS),
        );
    let lt: LinearTransformation<CKKSPlaintextOwned<ServerBackend>> =
        ckks_linear_transformation_encode_from_diagonals::<ServerBackend, f64, Standard>(
            module,
            BASE2K.into(),
            CoeffsMeta::from_delta_budget(LOG_DELTA_MAPS, 2),
            &ComplexDiagonals::new(re, Diagonals::<f64>::new(SLOTS)),
            strategy,
            false,
            &mut scratch.borrow(),
        )?;
    let first = lt.first_diagonal_plaintext().expect("a map with diagonals");
    let mut prepared =
        LinearTransformationPrepared::<ServerBackend, Standard>::alloc_prepared_from_index(module, &lt.index(), first);
    let mut scratch = ScratchOwned::<ServerBackend>::alloc(
        CKKSLinearTransformationOps::<ServerBackend, Standard>::ckks_linear_transformation_rhs_prepare_tmp_bytes(module, first),
    );
    module.ckks_linear_transformation_rhs_prepare(&mut prepared, &lt, &mut scratch.borrow())?;
    Ok(prepared)
}

impl Server {
    /// Compiles the bootstrapping, prepares the client's keys and encodes the round's maps and constants.
    pub fn new(params: &Params, keys: &EvaluationKeys) -> Result<Self> {
        let boot = &params.bootstrapping;
        let module = Module::<ServerBackend>::new(1 << LOG_N);
        let keys_layout = &boot.keys;

        // The bootstrapping.
        let mut scratch = ScratchOwned::<ServerBackend>::alloc(
            BootstrappingContextPrepared::<ServerBackend, f64>::compile_unprepared_tmp_bytes(&module, &boot.plan)?,
        );
        let unprepared = BootstrappingContextPrepared::<ServerBackend, f64>::compile_unprepared(
            &module,
            BASE2K.into(),
            &boot.plan,
            &mut scratch.borrow(),
        )?;
        // The round's keys move as one more key set: each key into a copy carved from the arena, then prepared.
        let moved = |layout: &dyn GGLWEInfos| ServerBackend::scratch_aligned(module.gglwe_bytes_of_from_infos(&layout));
        let (key, tsk) = (&params.rotation_key, &params.tensor_key);
        let mut scratch = ScratchOwned::<ServerBackend>::alloc(
            unprepared
                .prepare_tmp_bytes(&module)
                .max(keys_layout.prepare_tmp_bytes(&module))
                .max(
                    moved(key)
                        + GLWEAutomorphismKeyPreparedFactory::<ServerBackend, Standard>::glwe_automorphism_key_prepare_tmp_bytes(
                            &module, key,
                        ),
                )
                .max(
                    moved(tsk)
                        + GLWETensorKeyPreparedFactory::<ServerBackend, Standard>::glwe_tensor_key_prepare_tmp_bytes(
                            &module, tsk,
                        ),
                )
                .max(encode_slots_tmp_bytes(&module)),
        );
        let context = unprepared.prepare(&module, &mut scratch.borrow())?;
        let bootstrapping_keys = keys.bootstrapping.transfer_and_prepare(&module, &mut scratch.borrow())?;
        let lut = boot.lut.transfer_to(&module);
        let round_keys = keys.round.transfer_and_prepare(&module, &mut scratch.borrow())?;
        let rotation_keys = round_keys.rotation_keys;
        let tensor_key = round_keys.tensor_key.expect("the round's tensor key");

        // The round.
        let [linear, shift1, shift2] = params.maps.each_ref().map(|map| prepare_map(&module, map));
        let round_constants = ROUND_CONSTANTS
            .iter()
            .map(|&rc| {
                let mut values = vec![0.0; SLOTS];
                for s in iota_slots(rc) {
                    values[s] = 1.0;
                }
                encode_slots(&module, &values, OUTPUT_LOG_DELTA, &mut scratch.borrow())
            })
            .collect::<Result<Vec<_>>>()?;

        let mut server = Self {
            host: Module::<Host>::new(1 << LOG_N),
            input_k: boot.input_k,
            output_k: boot.output_k,
            raised_k: boot.raised_k,
            module,
            context,
            bootstrapping_keys,
            lut,
            linear: linear?,
            shift1: shift1?,
            shift2: shift2?,
            round_constants,
            rotation_keys,
            tensor_key,
            scratch: ScratchOwned::<ServerBackend>::alloc(0),
        };
        server.scratch = ScratchOwned::<ServerBackend>::alloc(server.round_tmp_bytes(params));
        Ok(server)
    }

    /// One arena for the bootstrapping and the rest of the round, on ciphertexts up to the bootstrapping output.
    fn round_tmp_bytes(&self, params: &Params) -> usize {
        let module = &self.module;
        let ct = CKKSLayout {
            glwe_layout: GLWELayout {
                n: module.n().into(),
                base2k: BASE2K.into(),
                k: self.output_k.into(),
                rank: Rank(1),
            },
            meta: GLWEPlaintextMeta::default(),
        };
        let mut bytes = module.ckks_functional_bootstrap_tmp_bytes(
            &params.bootstrapping.raised_layout(),
            &params.bootstrapping.input_layout(),
            &self.context,
            std::slice::from_ref(&self.lut),
            &params.bootstrapping.keys,
        );
        for lt in [&self.linear, &self.shift1, &self.shift2] {
            bytes = bytes.max(
                CKKSLinearTransformationOps::<ServerBackend, Standard>::ckks_linear_transformation_eval_self_tmp_bytes(
                    module,
                    false,
                    &ct,
                    lt.first_diagonal_plaintext().expect("a map with diagonals"),
                    &lt.index(),
                    &params.tensor_key,
                    &params.rotation_key,
                ),
            );
        }
        bytes
            .max(CKKSMulOps::<ServerBackend, Standard>::ckks_mul_tmp_bytes(
                module,
                &ct,
                &ct,
                &ct,
                &params.tensor_key,
            ))
            .max(CKKSCopyOps::<ServerBackend, Standard>::ckks_copy_tmp_bytes(module, &ct, &ct))
            // The bootstrapping output, copied down from the raised width.
            .max(CKKSCopyOps::<ServerBackend, Standard>::ckks_copy_tmp_bytes(
                module,
                &ct,
                &params.bootstrapping.raised_layout(),
            ))
            .max(CKKSSubOps::<ServerBackend, Standard>::ckks_sub_tmp_bytes(module, ct.size()))
            .max(CKKSAddOps::<ServerBackend, Standard>::ckks_add_tmp_bytes(module, ct.size()))
            .max(CKKSSubOps::<ServerBackend, Standard>::ckks_sub_pt_vec_tmp_bytes(
                module,
                ct.size(),
            ))
    }

    /// A client ciphertext moved to the server.
    pub fn upload(&self, ct: &Ct) -> ServerCt {
        let mut res = self.module.ckks_ciphertext_alloc(ct.base2k(), allocated_k(ct));
        res.set_k(ct.k());
        ct.transfer_into(&mut *res);
        res.set_meta(ct.meta());
        res
    }

    /// A server ciphertext moved back to the client.
    pub fn download(&self, ct: &ServerCt) -> Ct {
        let mut res = self.host.ckks_ciphertext_alloc(ct.base2k(), allocated_k(ct));
        res.set_k(ct.k());
        ct.transfer_into(&mut *res);
        res.set_meta(ct.meta());
        res
    }

    /// Waits for the backend to finish what it was given.
    pub fn synchronize(&self) {
        self.module.synchronize();
    }

    /// Round `r` of the permutation.
    pub fn round(&mut self, c: &ServerCt, r: usize) -> Result<ServerCt> {
        let v = self.linear_layer(c)?;
        let b = self.bootstrap(&v)?;
        self.chi_iota(&b, r)
    }

    /// `v = L c`, at the bootstrapping input's scale and width.
    fn linear_layer(&mut self, c: &ServerCt) -> Result<ServerCt> {
        let module = &self.module;
        let mut v = module.ckks_ciphertext_alloc(c.base2k(), c.k());
        module.ckks_linear_transformation_eval_self_into(
            &mut v,
            c,
            &self.linear,
            &self.rotation_keys,
            &mut self.scratch.borrow(),
        )?;
        // The bootstrapping takes its integers one bit of scale below the bits it returns.
        v.set_log_delta(LOG_DELTA);
        // A narrower copy drops the budget above the bootstrapping input.
        let mut out = module.ckks_ciphertext_alloc(c.base2k(), self.input_k.into());
        module.ckks_copy(&mut out, &v, &mut self.scratch.borrow())?;
        Ok(out)
    }

    /// `b = v mod 2`.
    fn bootstrap(&mut self, v: &ServerCt) -> Result<ServerCt> {
        // The bootstrapping writes at the raised width, and a narrower copy keeps its output's bits.
        let module = &self.module;
        let mut raised = vec![module.ckks_ciphertext_alloc(BASE2K.into(), self.raised_k.into())];
        module.ckks_functional_bootstrap(
            &mut raised,
            v,
            &self.context,
            std::slice::from_ref(&self.lut),
            &self.bootstrapping_keys,
            &mut self.scratch.borrow(),
        )?;
        let mut out = module.ckks_ciphertext_alloc(BASE2K.into(), self.output_k.into());
        module.ckks_copy(&mut out, &raised[0], &mut self.scratch.borrow())?;
        Ok(out)
    }

    /// chi and iota of round `r` on the bits `b`.
    fn chi_iota(&mut self, b: &ServerCt, r: usize) -> Result<ServerCt> {
        let module = &self.module;
        let mut scratch = self.scratch.borrow();
        let mut a1 = module.ckks_ciphertext_alloc(b.base2k(), b.k());
        let mut a2 = module.ckks_ciphertext_alloc(b.base2k(), b.k());
        module.ckks_linear_transformation_eval_self_into(&mut a1, b, &self.shift1, &self.rotation_keys, &mut scratch)?;
        module.ckks_linear_transformation_eval_self_into(&mut a2, b, &self.shift2, &self.rotation_keys, &mut scratch)?;
        let mut c = module.ckks_ciphertext_alloc(b.base2k(), b.k());
        module.ckks_mul_into(&mut c, &a1, &a2, &self.tensor_key, &mut scratch)?;
        module.ckks_sub_assign(&mut c, &a2, &mut scratch)?;
        module.ckks_add_assign(&mut c, b, &mut scratch)?;
        module.ckks_sub_pt_vec_assign(&mut c, &self.round_constants[r], &mut scratch)?;
        Ok(c)
    }
}
