//! Shared skeleton of the two carry-verb families (add, sub).
//!
//! [`CKKSAddReference`](crate::reference::CKKSAddReference) and
//! [`CKKSSubReference`](crate::reference::CKKSSubReference) are exact mirrors: same operand alignment, same metadata stamping, same plaintext facades — only the core GLWE verb and its coefficient-wise HAL hooks differ.
//! The [`ckks_carry_verb_reference!`] macro generates both traits from one body so the mirrors cannot drift (the pre-macro hand-written copies had already diverged on the tmp-bytes formulas and error labels).
//!
//! The macro is `pub(crate)`: it is an implementation detail of the two verb modules, not a backend-facing extension point.

use crate::CKKSResult as Result;
use crate::layouts::{CKKSPlaintextViewMut, ScratchArenaTakeCKKS};
use num_traits::{Float, FromPrimitive};
use poulpy_core::layouts::{Base2K, GLWEPlaintextLayout, GLWEToBackendMut, IntPolyInfos, TorusPrecision};
use poulpy_hal::layouts::{Backend, ScratchArena, encode_i128_by_limb};

use crate::SlotsKind;
use crate::{GLWEPlaintextMeta, ReplicatedLayout, Scale, SlotLayout, numerics::CKKSFloat, scalar::Quad, shaping::ShapingRing};

/// Largest number of coefficients of the one of a slot layout, `2R`.
const MAX_ONE_COEFFS: usize = 2 << ReplicatedLayout::MAX_LOG_R;

/// Scratch bytes of [`take_one_pt`] for a destination of `res_size` limbs.
pub(crate) fn one_pt_tmp_bytes<BE: Backend>(res_size: usize) -> usize {
    BE::scratch_aligned(BE::bytes_of_vec_znx(MAX_ONE_COEFFS, 1, res_size + 1))
}

/// The one of `slot_layout`, carved from `scratch`. On a full layout it is the
/// monomial `2` at scale `2`, added as a constant. On a replicated layout it is
/// the masked one `π_S` at scale `2^log_delta`: one on the data slots and zero
/// on the guards, a plaintext of degree `2R` in `Y = X^(N/(2R))` with
/// coefficients `cos(π·j/(2R))/R`, quantized with shaped rounding. It is
/// invariant, and the conjugate-invariant ring holds it as a plaintext of
/// degree `R`, its first `R` coefficients.
pub(crate) fn take_one_pt<'a, BE: Backend + 'a>(
    base2k: Base2K,
    slot_layout: SlotLayout,
    log_delta: usize,
    scratch: ScratchArena<'a, BE>,
) -> Result<(CKKSPlaintextViewMut<'a, BE>, ScratchArena<'a, BE>)> {
    let mut data = [0i128; MAX_ONE_COEFFS];
    let (n, k, meta) = match slot_layout {
        SlotLayout::Full => {
            data[0] = 2;
            let meta = GLWEPlaintextMeta {
                scale: Scale::Log(1),
                slots: SlotsKind::Real,
                log_sparsity: 0,
                slot_layout,
            };
            (1, 1, meta)
        }
        SlotLayout::Replicated(layout) => {
            let r = 1usize << layout.log_r();
            let ring = ShapingRing::of::<BE::Ring>();
            let m = match ring {
                ShapingRing::Standard => 2 * r,
                ShapingRing::ConjugateInvariant => r,
            };
            let r_inv = Quad::from_usize(r).expect("the period is a small integer").recip();
            let mut coeffs = [Quad::from_usize(0).expect("zero"); MAX_ONE_COEFFS];
            for (j, c) in coeffs[..m].iter_mut().enumerate() {
                *c = Quad::ckks_root_of_unity(j as u64, layout.log_r() as u32 + 2).0 * r_inv;
            }
            let (mut low, mut h, mut gh) = ([0i64; MAX_ONE_COEFFS], [0i64; MAX_ONE_COEFFS], [0i64; MAX_ONE_COEFFS]);
            crate::reference::encoding::shaped_quantize_into(
                &coeffs[..m],
                &layout,
                log_delta,
                1,
                ring,
                &mut data[..m],
                &mut low[..m],
                &mut h[..m],
                &mut gh[..m],
            )?;
            // Embedded at the backend's smallest plaintext degree when `m` is below it.
            let n = m.max(BE::MIN_DEGREE);
            if n > MAX_ONE_COEFFS {
                return Err(crate::CKKSCompositionError::InvalidOperand {
                    op: "ckks_one",
                    reason: format!("the masked one needs a plaintext of degree {n} > {MAX_ONE_COEFFS}"),
                }
                .into());
            }
            let stride = n / m;
            if stride > 1 {
                for j in (1..m).rev() {
                    data[j * stride] = data[j];
                    data[j] = 0;
                }
            }
            let meta = GLWEPlaintextMeta {
                scale: Scale::Log(log_delta),
                slots: SlotsKind::Real,
                log_sparsity: 0,
                slot_layout,
            };
            (n, log_delta + 1, meta)
        }
    };
    let layout = GLWEPlaintextLayout {
        n: n.into(),
        base2k,
        k: TorusPrecision(k as u32),
    };
    let (mut pt, scratch) = scratch.take_ckks_plaintext_scratch(&layout, meta);
    let encoded_k = pt.encoded_k().as_usize();
    let (mut carry, mut limb) = ([0i64; MAX_ONE_COEFFS], [0i64; MAX_ONE_COEFFS]);
    {
        let mut backend = pt.to_backend_mut();
        let shape = backend.data().shape();
        encode_i128_by_limb(
            base2k.as_usize(),
            encoded_k,
            &mut data[..n],
            &mut carry[..n],
            &mut limb[..n],
            |j, digits| {
                let offset = shape.scalar_offset(0, j) * size_of::<i64>();
                let mut region = BE::region_mut_ref(backend.data_mut().data_mut(), offset, size_of_val(digits));
                BE::copy_host_to_view(&mut region, bytemuck::cast_slice(digits));
            },
        );
    }
    Ok((pt, scratch))
}

/// Generates a `CKKS{Add,Sub}Reference` trait: the full reference-layer carry-verb
/// family (ct–ct, ct–pt-vector, ct–pt-constant, plus the one-constant
/// facades and the tmp-bytes accounting). None of them normalizes the result.
///
/// Parameters:
/// - `verb`: method-name stem (`add` / `sub`), also the error label.
/// - `doc_verb`: human verb for generated docs (`"addition"` / `"subtraction"`).
/// - `trait_name`: generated trait identifier.
/// - `glwe_bound` / `glwe_into` / `glwe_assign` / `glwe_lsh_verb`: the core
///   GLWE verb trait and its into/assign/shift-accumulate methods.
/// - `pt_vec_bounds`: the verb-specific HAL shift backend traits (each takes
///   a `<BE>` parameter), shared by the pt-vector and pt-constant variants.
///
/// All type and trait names in the expansion resolve at the call site: the
/// invoking module must import the shared bounds (`GLWECopy`, `GLWEShift`, `GLWENormalize`,
/// `VecZnx{Lsh,Rsh}TmpBytes`, `CKKSPlaintextVecReference`, `CKKSModuleAlloc`, the
/// info traits) alongside its verb-specific ones.
macro_rules! ckks_carry_verb_reference {
    (
        verb: $verb:ident,
        doc_verb: $doc_verb:literal,
        trait_name: $Trait:ident,
        glwe_bound: $GLWEVerb:ident,
        glwe_into: $glwe_into:ident,
        glwe_assign: $glwe_assign:ident,
        glwe_lsh_verb: $glwe_lsh_verb:ident,
        pt_vec_bounds: [$($PtVecBound:ident),+ $(,)?],
    ) => {
        ::paste::paste! {
            #[doc = concat!("Backend-generic reference implementation of the CKKS ", $doc_verb, " family.")]
            #[doc = ""]
            #[doc = "Generated by `ckks_carry_verb_reference!` — the add and sub families share one body (see `reference::carry_verb`)."]
            pub trait $Trait<BE: Backend> {
                /// Scratch bytes for the ct–ct variants.
                ///
                /// The three tmp-bytes methods share one formula: the max over the
                /// GLWE shift, GLWE normalize, and coefficient-wise `vec_znx` shift
                /// scratch needs. The ct–ct path never touches the `vec_znx` hooks,
                /// so for it this is a slight over-provision — the price of keeping
                /// one formula for the whole family.
                fn [<ckks_ $verb _tmp_bytes_reference>](&self, res_size: usize) -> usize
                where
                    Self: GLWEShift<BE> + GLWENormalize<BE> + VecZnxLshTmpBytes + VecZnxRshTmpBytes,
                {
                    self.glwe_shift_tmp_bytes(res_size)
                        .max(self.glwe_normalize_tmp_bytes())
                        .max(self.vec_znx_rsh_tmp_bytes(res_size))
                        .max(self.vec_znx_lsh_tmp_bytes(res_size))
                }

                /// Scratch bytes for the ct–pt-vector variants.
                fn [<ckks_ $verb _pt_vec_tmp_bytes_reference>](&self, res_size: usize) -> usize
                where
                    Self: GLWEShift<BE> + GLWENormalize<BE> + VecZnxLshTmpBytes + VecZnxRshTmpBytes,
                {
                    self.[<ckks_ $verb _tmp_bytes_reference>](res_size)
                }

                /// Scratch bytes for the ct–pt-constant variants.
                fn [<ckks_ $verb _pt_const_tmp_bytes_reference>](&self, res_size: usize) -> usize
                where
                    Self: GLWEShift<BE> + GLWENormalize<BE> + VecZnxLshTmpBytes + VecZnxRshTmpBytes,
                {
                    self.[<ckks_ $verb _tmp_bytes_reference>](res_size)
                }

                fn [<ckks_ $verb _into_reference>]<Dst, A, B>(
                    &self,
                    dst: &mut Dst,
                    a: &A,
                    b: &B,
                    scratch: &mut ScratchArena<'_, BE>,
                ) -> Result<()>
                where
                    Self: $GLWEVerb<BE> + GLWEShift<BE>,
                    Dst: GLWEToBackendMut<BE> + LWEInfos + SetCKKSInfos + CKKSInfos,
                    A: GLWEToBackendRef<BE> + CKKSInfos,
                    B: GLWEToBackendRef<BE> + CKKSInfos,
                {
                    let offset = ckks_offset_binary(dst, a, b);
                    // Validate before mutating: on error `dst` must remain untouched.
                    let slot_layout = $crate::reference::mul::product_layout(
                        concat!("ckks_", stringify!($verb)),
                        &[a.slot_layout(), b.slot_layout()],
                    )?;
                    $crate::reference::carry_verb::reject_replicated_scales(
                        concat!("ckks_", stringify!($verb)),
                        slot_layout,
                        a.log_delta(),
                        b.log_delta(),
                    )?;
                    let log_budget =
                        checked_log_budget_sub(concat!("ckks_", stringify!($verb), "_into"), a.log_budget().min(b.log_budget()), offset)?;

                    // Align both operands to the common torus level: operand `x` is
                    // shifted by `x.log_budget − min_budget + offset`. When both
                    // shifts are zero and `dst` holds both operands' limbs, the
                    // plain verb applies directly.
                    let min_budget = a.log_budget().min(b.log_budget());
                    let shift_a = a.log_budget() - min_budget + offset;
                    let shift_b = b.log_budget() - min_budget + offset;
                    if shift_a == 0 && shift_b == 0 && $crate::ckks_holds_limbs(dst, a) && $crate::ckks_holds_limbs(dst, b) {
                        self.$glwe_into(dst, a, b);
                    } else {
                        self.glwe_lsh(dst, a, shift_a, scratch);
                        self.$glwe_lsh_verb(dst, b, shift_b, scratch);
                    }

                    dst.set_log_delta(a.log_delta().min(b.log_delta()));
                    dst.set_log_budget(log_budget);
                    // The sum/product of values sparse at `s` and `t` is sparse at `min(s, t)`.
                    dst.set_log_sparsity(a.log_sparsity().min(b.log_sparsity()));
                    dst.set_slots(a.slots().join(b.slots()));
                    dst.set_slot_layout(slot_layout);
                    Ok(())
                }

                fn [<ckks_ $verb _assign_reference>]<Dst, A>(
                    &self,
                    dst: &mut Dst,
                    a: &A,
                    scratch: &mut ScratchArena<'_, BE>,
                ) -> Result<()>
                where
                    Self: $GLWEVerb<BE> + GLWEShift<BE>,
                    Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
                    A: GLWEToBackendRef<BE> + CKKSInfos,
                {
                    let slot_layout = $crate::reference::mul::product_layout(
                        concat!("ckks_", stringify!($verb), "_assign"),
                        &[dst.slot_layout(), a.slot_layout()],
                    )?;
                    $crate::reference::carry_verb::reject_replicated_scales(
                        concat!("ckks_", stringify!($verb), "_assign"),
                        slot_layout,
                        dst.log_delta(),
                        a.log_delta(),
                    )?;
                    let dst_log_budget = dst.log_budget();

                    if dst_log_budget > a.log_budget() {
                        self.glwe_lsh_assign(dst, dst_log_budget - a.log_budget(), scratch);
                    }
                    let shift_a = a.log_budget().saturating_sub(dst_log_budget);
                    // The shift-accumulate normalizes `a` over all its limbs.
                    if shift_a == 0 && $crate::ckks_holds_limbs(dst, a) {
                        self.$glwe_assign(dst, a);
                    } else {
                        self.$glwe_lsh_verb(dst, a, shift_a, scratch);
                    }

                    dst.set_log_budget(dst_log_budget.min(a.log_budget()));
                    dst.set_log_delta(dst.log_delta().min(a.log_delta()));
                    // The sum/product of values sparse at `s` and `t` is sparse at `min(s, t)`.
                    dst.set_log_sparsity(dst.log_sparsity().min(a.log_sparsity()));
                    dst.set_slots(dst.slots().join(a.slots()));
                    Ok(())
                }

                fn [<ckks_ $verb _pt_vec_into_reference>]<Dst, A, P>(
                    &self,
                    dst: &mut Dst,
                    a: &A,
                    pt: &P,
                    scratch: &mut ScratchArena<'_, BE>,
                ) -> Result<()>
                where
                    Self: GLWECopy<BE> + GLWEShift<BE> $(+ $PtVecBound<BE>)+ + CKKSPlaintextVecReference<BE>,
                    Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
                    A: GLWEToBackendRef<BE> + CKKSInfos,
                    P: GLWEToBackendRef<BE> + ::poulpy_core::layouts::IntPolyInfos + CKKSInfos,
                {
                    let slot_layout = $crate::reference::mul::product_layout(
                        concat!("ckks_", stringify!($verb), "_pt_vec_into"),
                        &[a.slot_layout(), pt.slot_layout()],
                    )?;
                    if pt.log_delta() > a.log_delta() {
                        $crate::reference::carry_verb::reject_replicated_scales(
                            concat!("ckks_", stringify!($verb), "_pt_vec_into"),
                            slot_layout,
                            a.log_delta(),
                            pt.log_delta(),
                        )?;
                    }
                    const OP: &str = concat!("ckks_", stringify!($verb), "_pt_vec");
                    // The copy's budget check, then the plaintext's at the budget the copy leaves, before it writes.
                    let log_budget = checked_log_budget_sub(OP, a.log_budget(), $crate::ckks_offset_unary(dst, a))?;
                    $crate::ensure_plaintext_vec_operand(OP, dst.base2k().as_usize(), dst.n().as_usize(), log_budget, pt, BE::MIN_DEGREE)?;
                    $crate::ckks_copy_stamp_unary(self, OP, dst, a, scratch)?;
                    self.[<ckks_ $verb _pt_vec_assign_reference>](dst, pt, scratch)
                }

                fn [<ckks_ $verb _pt_vec_assign_reference>]<Dst, P>(
                    &self,
                    dst: &mut Dst,
                    pt: &P,
                    scratch: &mut ScratchArena<'_, BE>,
                ) -> Result<()>
                where
                    Self: CKKSPlaintextVecReference<BE> $(+ $PtVecBound<BE>)+,
                    Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
                    P: GLWEToBackendRef<BE> + ::poulpy_core::layouts::IntPolyInfos + CKKSInfos,
                {
                    let slot_layout = $crate::reference::mul::product_layout(
                        concat!("ckks_", stringify!($verb), "_pt_vec_assign"),
                        &[dst.slot_layout(), pt.slot_layout()],
                    )?;
                    if pt.log_delta() > dst.log_delta() {
                        $crate::reference::carry_verb::reject_replicated_scales(
                            concat!("ckks_", stringify!($verb), "_pt_vec_assign"),
                            slot_layout,
                            dst.log_delta(),
                            pt.log_delta(),
                        )?;
                    }
                    CKKSPlaintextVecReference::[<ckks_ $verb _pt_vec_into_reference>](self, dst, pt, scratch)?;
                    dst.set_log_sparsity(dst.log_sparsity().min(dst.log_n() - pt.log_n()));
                    dst.set_slots(dst.slots().join(pt.slots()));
                    Ok(())
                }

                fn [<ckks_ $verb _pt_const_into_reference>]<Dst, A, P>(
                    &self,
                    dst: &mut Dst,
                    dst_coeff: usize,
                    a: &A,
                    cst: &P,
                    const_coeff: usize,
                    scratch: &mut ScratchArena<'_, BE>,
                ) -> Result<()>
                where
                    Self: GLWECopy<BE> + GLWEShift<BE> $(+ $PtVecBound<BE>)+ + CKKSPlaintextVecReference<BE>,
                    Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
                    A: GLWEToBackendRef<BE> + CKKSInfos,
                    P: GLWEToBackendRef<BE> + ::poulpy_core::layouts::IntPolyInfos + CKKSInfos,
                {
                    $crate::reference::carry_verb::reject_replicated_constant(
                        concat!("ckks_", stringify!($verb), "_pt_const"),
                        a.slot_layout(),
                    )?;
                    const OP: &str = concat!("ckks_", stringify!($verb), "_pt_const");
                    // The copy's budget check, then the plaintext's at the budget the copy leaves, before it writes.
                    let log_budget = checked_log_budget_sub(OP, a.log_budget(), $crate::ckks_offset_unary(dst, a))?;
                    $crate::ensure_plaintext_const_operand(
                        OP,
                        dst.base2k().as_usize(),
                        dst.n().as_usize(),
                        dst_coeff,
                        log_budget,
                        cst,
                        const_coeff,
                    )?;
                    $crate::ckks_copy_stamp_unary(self, OP, dst, a, scratch)?;
                    self.[<ckks_ $verb _pt_const_assign_reference>](dst, dst_coeff, cst, const_coeff, scratch)
                }

                fn [<ckks_ $verb _pt_const_assign_reference>]<Dst, P>(
                    &self,
                    dst: &mut Dst,
                    dst_coeff: usize,
                    cst: &P,
                    const_coeff: usize,
                    scratch: &mut ScratchArena<'_, BE>,
                ) -> Result<()>
                where
                    Self: CKKSPlaintextVecReference<BE> $(+ $PtVecBound<BE>)+,
                    Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos,
                    P: GLWEToBackendRef<BE> + ::poulpy_core::layouts::IntPolyInfos + CKKSInfos,
                {
                    $crate::reference::carry_verb::reject_replicated_constant(
                        concat!("ckks_", stringify!($verb), "_pt_const_assign"),
                        dst.slot_layout(),
                    )?;
                    CKKSPlaintextVecReference::[<ckks_ $verb _pt_const_into_reference>](self, dst, dst_coeff, cst, const_coeff, scratch)?;
                    // cst is always tagged as real, it's where dst_coeff lands
                    // that selects if the added constant is real (X^{0}) or imaginary (X^{N/2}).
                    let added = if dst_coeff == 0 { $crate::SlotsKind::Real } else { $crate::SlotsKind::Complex };
                    dst.set_slots(dst.slots().join(added));
                    Ok(())
                }
            }
        }
    };
}
pub(crate) use ckks_carry_verb_reference;

/// A constant lands on every slot, guards included, so a replicated layout
/// takes constants as replicated plaintexts.
/// Rejects a sum of a replicated layout at two scales, which would drop the
/// finer operand's extra bits with plain rounding.
pub(crate) fn reject_replicated_scales(
    op: &'static str,
    slot_layout: crate::SlotLayout,
    log_delta_a: usize,
    log_delta_b: usize,
) -> Result<()> {
    if matches!(slot_layout, crate::SlotLayout::Replicated(_)) && log_delta_a != log_delta_b {
        return Err(crate::CKKSCompositionError::InvalidOperand {
            op,
            reason: format!(
                "a replicated layout adds at one scale, got 2^{log_delta_a} and 2^{log_delta_b}: round the finer operand with shaping first"
            ),
        }
        .into());
    }
    Ok(())
}

pub(crate) fn reject_replicated_constant(op: &'static str, slot_layout: crate::SlotLayout) -> Result<()> {
    if let crate::SlotLayout::Replicated(_) = slot_layout {
        return Err(crate::CKKSCompositionError::InvalidOperand {
            op,
            reason: "a constant would reach the guard slots of a replicated layout".into(),
        }
        .into());
    }
    Ok(())
}
