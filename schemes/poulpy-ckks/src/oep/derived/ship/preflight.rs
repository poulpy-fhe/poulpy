//! Scratch sizing for the SHIP bootstrap.

use crate::CKKSResult as Result;
use anyhow::Context;
use poulpy_core::{
    GLWEBytesOf, GLWEKeyswitch,
    layouts::{GLWELayout, LWEInfos},
};

use crate::{CKKSLayout, GLWEPlaintextMeta, Scale};
use poulpy_hal::layouts::Module;

use super::bootstrap::{ship_bottom_layout, validate_runtime};
use crate::SlotsKind;
use crate::{
    CKKSCtBounds,
    api::{CKKSAddOps, CKKSConjugateOps, CKKSImagOps, CKKSMulOps, CKKSSubOps, ShipScalar},
    layouts::{CKKSCiphertextOwned, ShipKeysPrepared},
    oep::{CKKSEncodingImpl, CKKSShipImpl},
};

/// Caller-arena bound of one SHIP bootstrap call; validates the runtime
/// ciphertext and key layouts along the way. The bound covers the complex
/// variant whenever the key bundle carries its masks: the bottom ciphertext
/// carved for the encapsulation, then the temporaries of the roots' final
/// combination, each with the stages it spans.
pub(crate) fn ckks_ship_bootstrap_tmp_bytes_derived<BE, F, Src>(
    module: &Module<BE>,
    output: &CKKSCiphertextOwned<BE>,
    input: &Src,
    keys: &ShipKeysPrepared<BE::OwnedBuf, BE>,
) -> Result<usize>
where
    BE: CKKSShipImpl + CKKSEncodingImpl<F>,
    F: ShipScalar,
    Src: CKKSCtBounds,
{
    let params = keys.parameters();
    let plan = params.plan();
    let base2k = params.base2k();
    validate_runtime(module, output, input, keys, params.complex())?;

    let kk = plan.raised_k(base2k);
    let bottom = ship_bottom_layout(plan, base2k);
    let raised = CKKSLayout {
        glwe_layout: GLWELayout { k: kk.into(), ..bottom },
        meta: GLWEPlaintextMeta {
            scale: Scale::Log(plan.log_delta_work()),
            slots: SlotsKind::Complex,
            log_sparsity: 0,
        },
    };

    let encapsulation = module
        .glwe_keyswitch_tmp_bytes(&bottom, input, keys.dense_to_sparse())
        .max(BE::ckks_ship_coeff_encodings_tmp_bytes::<F>(
            module,
            base2k.into(),
            params.complex(),
            plan,
        )?)
        .checked_add(BE::scratch_aligned(module.glwe_bytes_of_from_infos(&bottom)))
        .context("SHIP encapsulation scratch size overflows usize")?;

    let mut tree = module
        .ckks_add_pt_vec_tmp_bytes(raised.size())
        .max(BE::ckks_ship_masking_accumulate_tmp_bytes(module, plan, base2k));
    for ik in keys.index_keys() {
        for group in ik.mux_keys() {
            if let Some(mux) = group.first() {
                tree = tree.max(BE::ckks_ship_mux_rotate_tmp_bytes(module, &raised, &mux.key, group.len()));
            }
        }
    }
    tree = tree.max(module.ckks_mul_tmp_bytes(&raised, &raised, &raised, keys.tensor_key()));

    // The complex combination holds `i v2`, `w+`, `w-` and the conjugate, the real one the conjugate alone.
    let temporaries = if params.complex() { 4 } else { 1 };
    let combination = BE::scratch_aligned(module.glwe_bytes_of_from_infos(&raised))
        .checked_mul(temporaries)
        .and_then(|bytes| {
            bytes.checked_add(
                module
                    .ckks_conjugate_tmp_bytes(&raised, keys.conjugation_key())
                    .max(module.ckks_add_tmp_bytes(raised.size()))
                    .max(module.ckks_sub_tmp_bytes(raised.size()))
                    .max(module.ckks_mul_i_tmp_bytes(raised.size())),
            )
        })
        .context("SHIP combination scratch size overflows usize")?;
    Ok(encapsulation.max(tree).max(combination))
}
