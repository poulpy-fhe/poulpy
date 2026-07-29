//! Scratch sizing for the SHIP bootstrap.

use crate::CKKSResult as Result;
use poulpy_core::{
    GLWEKeyswitch,
    layouts::{GLWELayout, Rank},
};

use crate::{CKKSLayout, CKKSMeta};
use poulpy_hal::layouts::{Backend, Module};

use super::{
    bootstrap::{ShipBootstrapModule, validate_runtime},
    masking::ship_masking_tmp_bytes,
    mux::ship_mux_rotate_tmp_bytes,
};
use crate::{
    CKKSCtBounds,
    api::{CKKSAddOps, CKKSConjugateOps, CKKSImagOps, CKKSMulOps, CKKSSubOps, ShipScalar},
    layouts::{CKKSCiphertext, ShipKeysPrepared},
    oep::{CKKSEncodingImpl, CKKSShipCoeffEncodingImpl},
};

/// Caller-arena bound of one SHIP bootstrap call; validates the runtime
/// ciphertext and key layouts along the way. The bound covers the complex
/// variant whenever the key bundle carries its masks.
pub(crate) fn ship_bootstrap_tmp_bytes<BE, F, Src>(
    module: &Module<BE>,
    output: &CKKSCiphertext<BE::OwnedBuf>,
    input: &Src,
    keys: &ShipKeysPrepared<BE::OwnedBuf, BE>,
) -> Result<usize>
where
    BE: Backend + CKKSShipCoeffEncodingImpl<BE> + CKKSEncodingImpl<BE, F>,
    F: ShipScalar,
    Module<BE>: ShipBootstrapModule<BE>,
    Src: CKKSCtBounds,
{
    let params = keys.parameters();
    let plan = params.plan();
    let base2k = params.base2k();
    validate_runtime(module, output, input, keys, params.complex())?;

    let kk = plan.raised_k(base2k);
    let bottom = GLWELayout {
        n: (plan.n() as u32).into(),
        base2k: base2k.into(),
        k: base2k.into(),
        rank: Rank(1),
    };
    let raised = CKKSLayout {
        glwe_layout: GLWELayout { k: kk.into(), ..bottom },
        meta: CKKSMeta {
            log_delta: plan.log_delta_work(),
            log_sparsity: 0,
        },
    };

    let mut bytes = module.glwe_keyswitch_tmp_bytes(&bottom, input, keys.dense_to_sparse());
    bytes = bytes.max(BE::ckks_ship_coeff_encodings_tmp_bytes_impl::<F>(
        module,
        plan,
        base2k.into(),
        params.complex(),
    )?);
    bytes = bytes.max(module.ckks_add_pt_vec_tmp_bytes());
    bytes = bytes.max(ship_masking_tmp_bytes(module, plan, base2k));
    for ik in keys.index_keys() {
        for group in ik.mux_keys() {
            if let Some(mux) = group.first() {
                bytes = bytes.max(ship_mux_rotate_tmp_bytes(module, &raised, &mux.key));
            }
        }
    }
    bytes = bytes.max(module.ckks_mul_tmp_bytes(&raised, &raised, &raised, keys.tensor_key()));
    bytes = bytes.max(module.ckks_conjugate_tmp_bytes(&raised, keys.conjugation_key()));
    bytes = bytes.max(module.ckks_add_tmp_bytes());
    bytes = bytes.max(module.ckks_sub_tmp_bytes());
    bytes = bytes.max(module.ckks_mul_i_tmp_bytes());
    Ok(bytes)
}
