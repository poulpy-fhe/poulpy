//! Backend-generic SHIP half-bootstrap circuit (Algorithm 1).

use crate::{CKKSResult as Result, ckks_ensure};
use poulpy_core::{
    GLWEKeyswitch, GLWEZero,
    layouts::{GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, TorusPrecision},
};
use poulpy_hal::{
    api::{
        CnvPVecBytesOf, Convolution, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAddAssign,
        VecZnxDftApply, VecZnxDftAutomorphism, VecZnxDftBytesOf, VecZnxDftCopy, VecZnxDftZero, VecZnxIdftApplyTmpA,
        VmpApplyDftToDft, VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, Module, ScratchArena},
};

use super::{
    masking::ship_masking_accumulate,
    mux::{ship_mux_plans, ship_mux_rotate},
};
use crate::SlotsKind;
use crate::{
    CKKSCtBounds, CKKSInfos, CKKSMeta,
    api::{CKKSAddOps, CKKSConjugateOps, CKKSImagOps, CKKSMulOps, CKKSSubOps, ShipScalar},
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc, CKKSPlaintextOwned, ShipKeysPrepared},
    oep::{CKKSEncodingImpl, CKKSShipCoeffEncodingImpl},
};

/// Validates the runtime ciphertexts against the key bundle's parameters.
pub(crate) fn validate_runtime<BE, Src>(
    module: &Module<BE>,
    output: &CKKSCiphertextOwned<BE>,
    input: &Src,
    keys: &ShipKeysPrepared<BE::OwnedBuf, BE>,
    complex: bool,
) -> Result<()>
where
    BE: Backend,
    Src: CKKSCtBounds,
{
    const OP: &str = "ckks_ship_bootstrap";
    let params = keys.parameters();
    let plan = params.plan();
    let base2k = params.base2k();
    ckks_ensure!(
        module.n() == plan.n(),
        "{OP}: module degree {} does not match plan degree {}",
        module.n(),
        plan.n()
    );
    ckks_ensure!(
        !complex || params.complex(),
        "{OP}: keys lack the omega_2 masks (generate with complex)"
    );
    ckks_ensure!(
        input.n().as_usize() == plan.n() && input.rank().as_usize() == 1,
        "{OP}: input degree {} / rank {} does not match the plan",
        input.n(),
        input.rank()
    );
    ckks_ensure!(
        input.base2k().as_usize() == base2k && input.k().as_usize() == base2k,
        "{OP}: input must span a single limb of base2k {base2k}, got base2k {} and width {}",
        input.base2k(),
        input.k()
    );
    ckks_ensure!(
        output.n().as_usize() == plan.n() && output.rank().as_usize() == 1,
        "{OP}: output degree {} / rank {} does not match the plan",
        output.n(),
        output.rank()
    );
    ckks_ensure!(
        output.base2k().as_usize() == base2k,
        "{OP}: output base2k {} does not match the key radix {base2k}",
        output.base2k()
    );
    ckks_ensure!(
        output.max_k().as_usize() >= plan.raised_k(base2k),
        "{OP}: output capacity {} is below the raised precision {}",
        output.max_k(),
        plan.raised_k(base2k)
    );
    Ok(())
}

/// Shared core of both bootstrap entry points: encapsulation, masking, blind
/// rotations and product tree per coefficient half, returning the product-tree
/// roots in half order.
fn ship_bootstrap_roots<BE, F, Src>(
    module: &Module<BE>,
    input: &Src,
    keys: &ShipKeysPrepared<BE::OwnedBuf, BE>,
    complex: bool,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<Vec<CKKSCiphertextOwned<BE>>>
where
    BE: Backend + CKKSShipCoeffEncodingImpl<BE> + CKKSEncodingImpl<BE, F>,
    F: ShipScalar,
    Module<BE>: CKKSMulOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSImagOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSModuleAlloc<BE>
        + GLWEKeyswitch<BE>
        + GLWEZero<BE>
        + Convolution<BE>
        + CnvPVecBytesOf
        + VecZnxDftApply<BE>
        + VecZnxDftZero<BE>
        + VecZnxDftCopy<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftAutomorphism<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VmpApplyDftToDft<BE>
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE>,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    const OP: &str = "ckks_ship_bootstrap";
    let params = keys.parameters();
    let plan = *params.plan();
    let base2k = params.base2k();
    let b2k_t: poulpy_core::layouts::Base2K = base2k.into();
    let kk = plan.raised_k(base2k);
    let ld = plan.log_delta_work();
    let theta = plan.theta();
    let halves = if complex { 2 } else { 1 };

    // Encapsulation: switch the bottom ciphertext from the dense to the
    // sparse secret at the bottom modulus.
    let mut a_sparse = module.ckks_ciphertext_alloc(b2k_t, TorusPrecision(base2k as u32));
    module.glwe_keyswitch(&mut a_sparse, input, keys.dense_to_sparse(), scratch);
    a_sparse.set_meta_checked(input.meta())?;

    let enc = BE::ckks_ship_coeff_encodings_impl::<F, _>(module, &a_sparse, &plan, b2k_t, complex, scratch)?;
    ckks_ensure!(
        enc.pi.len() == plan.sparse_hamming_weight() && (!complex || enc.pt0_2.is_some()),
        "{OP}: malformed coefficient encodings"
    );

    // Leaf 0 per half: trivial encryption of pt0 / pt0_2.
    let mut leaves: Vec<Vec<CKKSCiphertextOwned<BE>>> = Vec::with_capacity(halves);
    for half in 0..halves {
        let pt0 = if half == 0 {
            &enc.pt0
        } else {
            enc.pt0_2.as_ref().expect("complex encodings carry pt0_2")
        };
        let mut leaf0 = module.ckks_ciphertext_alloc(b2k_t, TorusPrecision(kk as u32));
        module.glwe_zero(&mut leaf0);
        leaf0.set_meta_checked(CKKSMeta {
            log_delta: ld,
            log_sparsity: 0,
            slots: SlotsKind::Complex,
        })?;
        module.ckks_add_pt_vec_assign(&mut leaf0, pt0, scratch)?;
        let mut half_leaves = Vec::with_capacity(plan.sparse_hamming_weight() + 1);
        half_leaves.push(leaf0);
        leaves.push(half_leaves);
    }

    // Leaves 1..=h: theta-column masking then hoisted base-B mux blind
    // rotation over the remaining digits; the pi plaintexts and mux keys are
    // shared between the halves, the mask sets differ. The mux rotation
    // amounts recur across slots, so their automorphism plans are built once.
    let plans = ship_mux_plans(
        module,
        keys.index_keys()
            .iter()
            .flat_map(|ik| ik.mux_keys().iter().map(Vec::as_slice)),
    );
    for (slot, ik) in keys.index_keys().iter().enumerate() {
        let pi = &enc.pi[slot];
        ckks_ensure!(pi.len() == 4 * theta, "{OP}: malformed pi encodings at slot {slot}");
        for (half, half_leaves) in leaves.iter_mut().enumerate() {
            let masks = if half == 0 { ik.masks() } else { ik.masks2() };
            let mut acc = module.ckks_ciphertext_alloc(b2k_t, TorusPrecision(kk as u32));
            ship_masking_accumulate(module, &mut acc, &plan, masks, pi, scratch)?;
            for group in ik.mux_keys() {
                ship_mux_rotate(module, &mut acc, group, &plans, scratch)?;
            }
            half_leaves.push(acc);
        }
    }

    // Binary product tree per half; odd leftovers carry to the next level.
    let mut roots = Vec::with_capacity(halves);
    for half_leaves in leaves {
        let mut level = half_leaves;
        while level.len() > 1 {
            let mut next = Vec::with_capacity(level.len().div_ceil(2));
            let mut iter = level.into_iter();
            while let Some(x) = iter.next() {
                match iter.next() {
                    Some(y) => {
                        let budget = x.log_budget().min(y.log_budget());
                        let consumed = x.log_delta().max(y.log_delta());
                        ckks_ensure!(budget >= consumed, "{OP}: product tree exhausts the budget");
                        let k_dst = budget - consumed + x.log_delta().min(y.log_delta());
                        let mut dst = module.ckks_ciphertext_alloc(b2k_t, TorusPrecision(k_dst as u32));
                        module.ckks_mul_into(&mut dst, &x, &y, keys.tensor_key(), scratch)?;
                        next.push(dst);
                    }
                    None => next.push(x),
                }
            }
            level = next;
        }
        roots.push(level.pop().expect("product tree is never empty"));
    }
    Ok(roots)
}

/// Real-case SHIP bootstrap: `output = root + Conj(root)`.
pub(crate) fn ship_bootstrap_into<BE, F, Src>(
    module: &Module<BE>,
    output: &mut CKKSCiphertextOwned<BE>,
    input: &Src,
    keys: &ShipKeysPrepared<BE::OwnedBuf, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + CKKSShipCoeffEncodingImpl<BE> + CKKSEncodingImpl<BE, F>,
    F: ShipScalar,
    Module<BE>: CKKSMulOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSImagOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSModuleAlloc<BE>
        + GLWEKeyswitch<BE>
        + GLWEZero<BE>
        + Convolution<BE>
        + CnvPVecBytesOf
        + VecZnxDftApply<BE>
        + VecZnxDftZero<BE>
        + VecZnxDftCopy<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftAutomorphism<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VmpApplyDftToDft<BE>
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE>,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    validate_runtime(module, output, input, keys, false)?;
    let base2k: poulpy_core::layouts::Base2K = keys.parameters().base2k().into();
    let mut roots = ship_bootstrap_roots::<BE, F, _>(module, input, keys, false, scratch)?;

    let root = roots.pop().expect("real bootstrap has one root");
    let mut conj = module.ckks_ciphertext_alloc(base2k, root.k());
    module.ckks_conjugate_into(&mut conj, &root, keys.conjugation_key(), scratch)?;
    module.ckks_add_into(output, &root, &conj, scratch)?;
    Ok(())
}

/// Complex-case SHIP bootstrap:
/// `output = (v1 + i*v2) + Conj(v1 - i*v2)` over the two per-half roots.
pub(crate) fn ship_bootstrap_complex_into<BE, F, Src>(
    module: &Module<BE>,
    output: &mut CKKSCiphertextOwned<BE>,
    input: &Src,
    keys: &ShipKeysPrepared<BE::OwnedBuf, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + CKKSShipCoeffEncodingImpl<BE> + CKKSEncodingImpl<BE, F>,
    F: ShipScalar,
    Module<BE>: CKKSMulOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSImagOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSModuleAlloc<BE>
        + GLWEKeyswitch<BE>
        + GLWEZero<BE>
        + Convolution<BE>
        + CnvPVecBytesOf
        + VecZnxDftApply<BE>
        + VecZnxDftZero<BE>
        + VecZnxDftCopy<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftAutomorphism<BE>
        + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE>
        + VmpApplyDftToDft<BE>
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxBigNormalizeTmpBytes,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE>,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    validate_runtime(module, output, input, keys, true)?;
    let base2k: poulpy_core::layouts::Base2K = keys.parameters().base2k().into();
    let mut roots = ship_bootstrap_roots::<BE, F, _>(module, input, keys, true, scratch)?;

    let v2 = roots.pop().expect("complex bootstrap has two roots");
    let v1 = roots.pop().expect("complex bootstrap has two roots");
    let k_eff = v1.k();
    let mut iv2 = module.ckks_ciphertext_alloc(base2k, v2.k());
    module.ckks_mul_i_into(&mut iv2, &v2, scratch)?;
    let mut w_plus = module.ckks_ciphertext_alloc(base2k, k_eff);
    module.ckks_add_into(&mut w_plus, &v1, &iv2, scratch)?;
    let mut w_minus = module.ckks_ciphertext_alloc(base2k, k_eff);
    module.ckks_sub_into(&mut w_minus, &v1, &iv2, scratch)?;
    let mut conj = module.ckks_ciphertext_alloc(base2k, w_minus.k());
    module.ckks_conjugate_into(&mut conj, &w_minus, keys.conjugation_key(), scratch)?;
    module.ckks_add_into(output, &w_plus, &conj, scratch)?;
    Ok(())
}
