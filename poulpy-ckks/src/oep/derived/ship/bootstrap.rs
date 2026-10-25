//! Backend-generic SHIP half-bootstrap circuit (Algorithm 1).

use crate::{CKKSResult as Result, ckks_ensure};
use poulpy_core::layouts::SetGLWEPlaintextInfos;
use poulpy_core::layouts::prepared::GGLWEPreparedToBackendRef;
use poulpy_core::{
    GLWEKeyswitch, GLWEZero,
    layouts::{Base2K, GLWEInfos, GLWELayout, GLWEToBackendRef, LWEInfos, Rank, TorusPrecision},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use super::ship_mux_plans;
use crate::{
    CKKSCtBounds, CKKSInfos, GLWEPlaintextMeta, Scale, SlotsKind,
    api::{CKKSAddOps, CKKSConjugateOps, CKKSImagOps, CKKSMulOps, CKKSSubOps, ShipScalar},
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc, ScratchArenaTakeCKKS, ShipKeysPrepared, ShipPlan},
    oep::{CKKSEncodingImpl, CKKSShipImpl},
};

/// The layout of the bottom ciphertext the dense-to-sparse switch writes: one limb of the key radix.
pub(super) fn ship_bottom_layout(plan: &ShipPlan, base2k: usize) -> GLWELayout {
    GLWELayout {
        n: (plan.n() as u32).into(),
        base2k: base2k.into(),
        k: base2k.into(),
        rank: Rank(1),
    }
}

/// The layout of a temporary of the final combination of the product-tree roots, at width `k`.
fn ship_root_layout<BE: Backend>(module: &Module<BE>, base2k: Base2K, k: TorusPrecision) -> GLWELayout {
    GLWELayout {
        n: (module.n() as u32).into(),
        base2k,
        k,
        rank: Rank(1),
    }
}

/// Validates the runtime ciphertexts against the key bundle's parameters.
pub(crate) fn validate_runtime<BE, Src>(
    op: &'static str,
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
    let params = keys.parameters();
    let plan = params.plan();
    let base2k = params.base2k();
    ckks_ensure!(
        module.n() == plan.n(),
        "{op}: module degree {} does not match plan degree {}",
        module.n(),
        plan.n()
    );
    ckks_ensure!(
        !complex || params.complex(),
        "{op}: keys lack the omega_2 masks (generate with complex)"
    );
    ckks_ensure!(
        input.n().as_usize() == plan.n() && input.rank().as_usize() == 1,
        "{op}: input degree {} / rank {} does not match the plan",
        input.n(),
        input.rank()
    );
    ckks_ensure!(
        input.base2k().as_usize() == base2k && input.k().as_usize() == base2k,
        "{op}: input must span a single limb of base2k {base2k}, got base2k {} and width {}",
        input.base2k(),
        input.k()
    );
    ckks_ensure!(
        output.n().as_usize() == plan.n() && output.rank().as_usize() == 1,
        "{op}: output degree {} / rank {} does not match the plan",
        output.n(),
        output.rank()
    );
    ckks_ensure!(
        output.base2k().as_usize() == base2k,
        "{op}: output base2k {} does not match the key radix {base2k}",
        output.base2k()
    );
    ckks_ensure!(
        output.max_k().as_usize() >= plan.raised_k(base2k),
        "{op}: output capacity {} is below the raised precision {}",
        output.max_k(),
        plan.raised_k(base2k)
    );
    Ok(())
}

/// Shared core of both bootstrap entry points: encapsulation, masking, blind
/// rotations and product tree per coefficient half, returning the product-tree
/// roots in half order.
fn ship_bootstrap_roots<BE, F, Src>(
    op: &'static str,
    module: &Module<BE>,
    input: &Src,
    keys: &ShipKeysPrepared<BE::OwnedBuf, BE>,
    complex: bool,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<Vec<CKKSCiphertextOwned<BE>>>
where
    BE: CKKSShipImpl + CKKSEncodingImpl<F>,
    F: ShipScalar,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
{
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
    let enc = scratch.scope(|scratch_local| -> Result<_> {
        let (mut a_sparse, mut scratch_local) =
            scratch_local.take_ckks_ciphertext_scratch(&ship_bottom_layout(&plan, base2k), GLWEPlaintextMeta::default());
        module.glwe_keyswitch(
            &mut a_sparse,
            input,
            &keys.dense_to_sparse().to_backend_ref(),
            &mut scratch_local.borrow(),
        );
        a_sparse.set_meta_checked(input.meta())?;
        BE::ckks_ship_coeff_encodings::<F, _>(module, b2k_t, complex, &a_sparse, &plan, &mut scratch_local)
    })?;
    ckks_ensure!(
        enc.pi.len() == plan.sparse_hamming_weight() && (!complex || enc.pt0_2.is_some()),
        "{op}: malformed coefficient encodings"
    );

    // The plan's Hamming weight sets the leaf count, so the leaves and their product tree stay on the heap.
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
        leaf0.set_meta_checked(GLWEPlaintextMeta {
            scale: Scale::Log(ld),
            slots: SlotsKind::Complex,
            log_sparsity: 0,
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
        ckks_ensure!(pi.len() == 4 * theta, "{op}: malformed pi encodings at slot {slot}");
        for (half, half_leaves) in leaves.iter_mut().enumerate() {
            let masks = if half == 0 { ik.masks() } else { ik.masks2() };
            let mut acc = module.ckks_ciphertext_alloc(b2k_t, TorusPrecision(kk as u32));
            BE::ckks_ship_masking_accumulate(module, &mut acc, &plan, masks, pi, scratch)?;
            for group in ik.mux_keys() {
                BE::ckks_ship_mux_rotate(module, &mut acc, group, &plans, scratch)?;
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
                        ckks_ensure!(budget >= consumed, "{op}: product tree exhausts the budget");
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
pub(crate) fn ckks_ship_bootstrap_into_derived<BE, F, Src>(
    module: &Module<BE>,
    output: &mut CKKSCiphertextOwned<BE>,
    input: &Src,
    keys: &ShipKeysPrepared<BE::OwnedBuf, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSShipImpl + CKKSEncodingImpl<F>,
    F: ShipScalar,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    const OP: &str = "ckks_ship_bootstrap_into";
    validate_runtime(OP, module, output, input, keys, false)?;
    let bytes = BE::ckks_ship_bootstrap_tmp_bytes::<F, _>(module, &*output, input, keys)?;
    ckks_ensure!(scratch.available() >= bytes, "{OP}: scratch below its query");
    let base2k: poulpy_core::layouts::Base2K = keys.parameters().base2k().into();
    let mut roots = ship_bootstrap_roots::<BE, F, _>(OP, module, input, keys, false, scratch)?;

    let root = roots.pop().expect("real bootstrap has one root");
    let (mut conj, mut scratch_rest) = scratch
        .borrow()
        .take_ckks_ciphertext_scratch(&ship_root_layout(module, base2k, root.k()), GLWEPlaintextMeta::default());
    module.ckks_conjugate_into(&mut conj, &root, keys.conjugation_key(), &mut scratch_rest.borrow())?;
    module.ckks_add_into(output, &root, &conj, &mut scratch_rest)?;
    // `root + conj(root) = 2·Re(root)`.
    output.set_slots(SlotsKind::Real);
    Ok(())
}

/// Complex-case SHIP bootstrap:
/// `output = (v1 + i*v2) + Conj(v1 - i*v2)` over the two per-half roots.
pub(crate) fn ckks_ship_bootstrap_complex_into_derived<BE, F, Src>(
    module: &Module<BE>,
    output: &mut CKKSCiphertextOwned<BE>,
    input: &Src,
    keys: &ShipKeysPrepared<BE::OwnedBuf, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: CKKSShipImpl + CKKSEncodingImpl<F>,
    F: ShipScalar,
    Src: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    const OP: &str = "ckks_ship_bootstrap_complex_into";
    validate_runtime(OP, module, output, input, keys, true)?;
    let bytes = BE::ckks_ship_bootstrap_tmp_bytes::<F, _>(module, &*output, input, keys)?;
    ckks_ensure!(scratch.available() >= bytes, "{OP}: scratch below its query");
    let base2k: poulpy_core::layouts::Base2K = keys.parameters().base2k().into();
    let mut roots = ship_bootstrap_roots::<BE, F, _>(OP, module, input, keys, true, scratch)?;

    let v2 = roots.pop().expect("complex bootstrap has two roots");
    let v1 = roots.pop().expect("complex bootstrap has two roots");
    let k_eff = v1.k();
    let (mut iv2, scratch_rest) = scratch
        .borrow()
        .take_ckks_ciphertext_scratch(&ship_root_layout(module, base2k, v2.k()), GLWEPlaintextMeta::default());
    let (mut w_plus, scratch_rest) =
        scratch_rest.take_ckks_ciphertext_scratch(&ship_root_layout(module, base2k, k_eff), GLWEPlaintextMeta::default());
    let (mut w_minus, mut scratch_rest) =
        scratch_rest.take_ckks_ciphertext_scratch(&ship_root_layout(module, base2k, k_eff), GLWEPlaintextMeta::default());
    module.ckks_mul_i_into(&mut iv2, &v2, &mut scratch_rest.borrow())?;
    module.ckks_add_into(&mut w_plus, &v1, &iv2, &mut scratch_rest.borrow())?;
    module.ckks_sub_into(&mut w_minus, &v1, &iv2, &mut scratch_rest.borrow())?;
    let (mut conj, mut scratch_rest) =
        scratch_rest.take_ckks_ciphertext_scratch(&ship_root_layout(module, base2k, w_minus.k()), GLWEPlaintextMeta::default());
    module.ckks_conjugate_into(&mut conj, &w_minus, keys.conjugation_key(), &mut scratch_rest.borrow())?;
    module.ckks_add_into(output, &w_plus, &conj, &mut scratch_rest)?;
    Ok(())
}
