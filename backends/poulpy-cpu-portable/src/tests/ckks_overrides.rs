//! Downstream specialization probes: reference families remain available while
//! polynomial kernels and EvalMod independently select execution and scratch.
use crate::hal_impl::delegating_backend::DifferentSamplingFFT64Portable as OverrideBackend;
use poulpy_ckks::CKKSModuleInfos;
use poulpy_ckks::Scale;
use poulpy_ckks::api::{
    CKKSAllOpsTmpBytes, CKKSComplexPolynomialEvaluationOps, CKKSCopyOps, CKKSEvalModOps, CKKSPolynomialEvaluationOps,
};
use poulpy_ckks::layouts::{CKKSModuleAlloc, CKKSPlaintextOwned};
use poulpy_ckks::oep::CKKSComplexPolynomialEvaluationImpl;
use poulpy_ckks::test_suite::{
    helpers::{assert_ckks_error, assert_ckks_internal},
    parity::helpers::{fixture_ciphertext, snapshot, with_scratch},
};
use poulpy_ckks::{CKKSCompositionError, CoeffsMeta, GLWEPlaintextMeta, SetCKKSInfos};
use poulpy_core::GLWEMaskFill;
use poulpy_core::layouts::{
    GLWEPlaintextInfos, GetTensorKey, LWEInfos, SetGLWEPlaintextInfos, SetK, TorusPrecision,
    prepared::GLWETensorKeyPreparedBackendRef,
};
use poulpy_cpu_oracle::FFT64Oracle;
use poulpy_hal::layouts::{Backend, Module};
use std::cell::{Cell, RefCell};
thread_local! {
 static REAL_CALLS: Cell<usize> = const { Cell::new(0) };
 static COMPLEX_CALLS: Cell<usize> = const { Cell::new(0) };
 static EVAL_MOD_CALLS: Cell<usize> = const { Cell::new(0) };
 static MAX_COPY_CAPACITY: Cell<usize> = const { Cell::new(0) };
}
const OVERRIDE_SCRATCH: usize = 256;
poulpy_ckks::impl_ckks_plaintext_vec_reference_full!(OverrideBackend);
const COPY_WORKSPACE: usize = 1 << 20;
unsafe impl poulpy_ckks::oep::CKKSCopyImpl for OverrideBackend {
    fn ckks_copy_tmp_bytes<Dst: poulpy_ckks::CKKSCtBounds, Src: poulpy_ckks::CKKSCtBounds>(
        module: &Module<Self>,
        dst: &Dst,
        src: &Src,
    ) -> usize {
        COPY_WORKSPACE * dst.max_size()
            + poulpy_ckks::reference::copy::CKKSCopyReference::ckks_copy_tmp_bytes_reference(module, dst, src)
    }
    fn ckks_copy<Dst, Src>(
        module: &Module<Self>,
        dst: &mut Dst,
        src: &Src,
        scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
    ) -> poulpy_ckks::CKKSResult<()>
    where
        Dst: poulpy_core::layouts::GLWEToBackendMut<Self> + poulpy_ckks::CKKSCtBounds + SetCKKSInfos,
        Src: poulpy_core::layouts::GLWEToBackendRef<Self> + poulpy_ckks::CKKSCtBounds,
    {
        MAX_COPY_CAPACITY.with(|capacity| capacity.set(capacity.get().max(dst.max_size())));
        let workspace = COPY_WORKSPACE * dst.max_size();
        // The reference refuses before any scratch use, so an arena short of the workspace goes to it and must refuse.
        if scratch.available() < workspace {
            let result = poulpy_ckks::reference::copy::CKKSCopyReference::ckks_copy_reference(module, dst, src, scratch);
            assert!(result.is_err(), "ckks_copy: scratch below the probe's query");
            return result;
        }
        let (mut region, mut remaining) = scratch.borrow().take_region(workspace);
        Self::copy_host_to_view(&mut region, &vec![0x5A; workspace]);
        poulpy_ckks::reference::copy::CKKSCopyReference::ckks_copy_reference(module, dst, src, &mut remaining)
    }
}
mod carry;
mod composite;
mod dft;
mod paco_ship;
mod ring_map;
poulpy_ckks::impl_ckks_neg_reference_full!(OverrideBackend);
poulpy_ckks::impl_ckks_pow2_reference_full!(OverrideBackend);
poulpy_ckks::impl_ckks_imag_reference_full!(OverrideBackend);
poulpy_ckks::impl_ckks_rotate_reference_full!(OverrideBackend);
poulpy_ckks::impl_ckks_inner_sum_reference_full!(OverrideBackend);
poulpy_ckks::impl_ckks_conjugate_reference_full!(OverrideBackend);
poulpy_ckks::impl_ckks_encryption_reference_full!(OverrideBackend);
poulpy_ckks::impl_ckks_mul_reference_full!(OverrideBackend);
poulpy_ckks::impl_ckks_linear_transformation_reference_full!(OverrideBackend);
poulpy_ckks::impl_ckks_dot_product_reference_full!(OverrideBackend);
poulpy_ckks::impl_ckks_add_many_reference_full!(OverrideBackend);
poulpy_ckks::impl_ckks_mul_add_derived_full!(OverrideBackend);
poulpy_ckks::impl_ckks_affine_derived_full!(OverrideBackend);
poulpy_ckks::impl_ckks_mul_sub_derived_full!(OverrideBackend);
poulpy_ckks::impl_ckks_approximation_derived_full!(OverrideBackend);

unsafe impl poulpy_ckks::oep::CKKSPolynomialEvaluationImpl for OverrideBackend {
    fn ckks_eval_poly_real_const_coeffs_from_power_basis<R, B, A, G, H>(
        module: &::poulpy_hal::layouts::Module<Self>,
        res: &mut R,
        power_basis: &G,
        poly: &B,
        tsk: &H,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
    ) -> poulpy_ckks::CKKSResult<()>
    where
        R: ::poulpy_core::layouts::GLWEToBackendMut<Self>
            + poulpy_ckks::CKKSCtBounds
            + poulpy_ckks::SetCKKSInfos
            + ::poulpy_core::layouts::SetBSGSMeta,
        B: poulpy_ckks::api::BSGSPolynomialInfos<Self>,
        B::Coeffs: poulpy_ckks::CKKSCtBounds,
        A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + poulpy_ckks::CKKSCtBounds + poulpy_core::layouts::BSGSMeta,
        G: poulpy_ckks::api::PowerBasisHelper<Self, A>,
        H: ::poulpy_core::layouts::GetTensorKey<Self>,
    {
        REAL_CALLS.with(|calls| calls.set(calls.get() + 1));
        poulpy_ckks::reference::polynomial_evaluation::CKKSPolynomialEvaluationReference::ckks_eval_poly_real_const_coeffs_from_power_basis_reference::<R, B, A, G, H>(
            module, res, power_basis, poly, tsk, scratch,
        )
    }
}

unsafe impl CKKSComplexPolynomialEvaluationImpl for OverrideBackend {
    fn ckks_eval_poly_complex_const_coeffs_from_power_basis<R, C, A, G, H>(
        module: &::poulpy_hal::layouts::Module<Self>,
        res: &mut R,
        power_basis: &G,
        poly: &poulpy_ckks::polynomial::ComplexBSGSPolynomial<C>,
        tsk: &H,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
    ) -> poulpy_ckks::CKKSResult<()>
    where
        R: ::poulpy_core::layouts::GLWEToBackendMut<Self>
            + poulpy_ckks::CKKSCtBounds
            + poulpy_ckks::SetCKKSInfos
            + ::poulpy_core::layouts::SetBSGSMeta,
        C: ::poulpy_core::layouts::GLWEToBackendRef<Self>
            + ::poulpy_core::layouts::GLWEInfos
            + poulpy_core::layouts::BSGSMeta
            + poulpy_ckks::CKKSCtBounds
            + ::poulpy_core::layouts::IntPolyInfos,
        A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + poulpy_ckks::CKKSCtBounds + poulpy_core::layouts::BSGSMeta,
        G: poulpy_ckks::api::PowerBasisHelper<Self, A>,
        H: ::poulpy_core::layouts::GetTensorKey<Self>,
    {
        COMPLEX_CALLS.with(|calls| calls.set(calls.get() + 1));
        poulpy_ckks::reference::polynomial_evaluation::CKKSPolynomialEvaluationReference::ckks_eval_poly_complex_const_coeffs_from_power_basis_reference::<R, C, A, G, H>(
            module, res, power_basis, poly, tsk, scratch,
        )
    }
}

unsafe impl poulpy_ckks::oep::CKKSEvalModImpl for OverrideBackend {
    fn ckks_eval_mod_tmp_bytes<R, C, P, F, T>(
        module: &::poulpy_hal::layouts::Module<Self>,
        res: &R,
        ct: &C,
        params: &poulpy_ckks::layouts::eval_mod::EvalMod<F, P>,
        tsk: &T,
    ) -> usize
    where
        R: poulpy_ckks::CKKSCtBounds,
        C: poulpy_ckks::CKKSCtBounds,
        P: poulpy_ckks::CKKSCtBounds,
        T: poulpy_core::layouts::GGLWEInfos,
    {
        let _ = (module, res, ct, params, tsk);
        OVERRIDE_SCRATCH
    }
    fn ckks_eval_mod<R, C, P, F, H>(
        module: &::poulpy_hal::layouts::Module<Self>,
        res: &mut R,
        ct: &C,
        params: &poulpy_ckks::layouts::eval_mod::EvalMod<F, P>,
        tsk: &H,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
    ) -> poulpy_ckks::CKKSResult<()>
    where
        R: ::poulpy_core::layouts::GLWEToBackendMut<Self>
            + ::poulpy_core::layouts::GLWEToBackendRef<Self>
            + poulpy_ckks::CKKSCtBounds
            + poulpy_ckks::SetCKKSInfos
            + ::poulpy_core::layouts::SetBSGSMeta,
        C: ::poulpy_core::layouts::GLWEToBackendRef<Self> + poulpy_ckks::CKKSCtBounds,
        P: ::poulpy_core::layouts::GLWEToBackendRef<Self>
            + ::poulpy_core::layouts::IntPolyInfos
            + poulpy_ckks::CKKSCtBounds
            + ::poulpy_core::layouts::BSGSMeta,
        H: ::poulpy_core::layouts::GetTensorKey<Self>,
    {
        let _ = (module, res, ct, params, tsk);
        EVAL_MOD_CALLS.with(|calls| calls.set(calls.get() + 1));
        let (mut region, _remaining) = scratch.borrow().take_region(OVERRIDE_SCRATCH);
        Self::copy_host_to_view(&mut region, &[0x5A; OVERRIDE_SCRATCH]);
        Err(anyhow::anyhow!("EvalMod override probe").into())
    }
}

struct NoTensorKey;
impl<BE: Backend> GetTensorKey<BE> for NoTensorKey {
    fn lookup_tensor_key(&self, _: TorusPrecision) -> poulpy_core::Result<GLWETensorKeyPreparedBackendRef<'_, BE>> {
        panic!("linear polynomial dispatch must not request a tensor key")
    }
}

fn polynomial(module: &Module<OverrideBackend>) -> poulpy_ckks::polynomial::BSGSPolynomial<CKKSPlaintextOwned<OverrideBackend>> {
    use poulpy_ckks::polynomial::{Basis, Polynomial, SplitStrategy};
    Polynomial::new(Basis::Monomial, vec![0.0f64, 1.0])
        .decompose_bsgs_with(SplitStrategy::MinDepth, |coeffs| -> anyhow::Result<_> {
            let mut pt = module.ckks_pt_coeffs_alloc(coeffs.len(), 16usize.into(), 32usize.into());
            pt.set_meta(GLWEPlaintextMeta {
                scale: Scale::Log(8),
                ..Default::default()
            });
            Ok(pt)
        })
        .unwrap()
}

/// Real and complex one-shot evaluations enter their own prepared overrides once each, on exactly the documented
/// all-operations bound, which the overriding copy sets, and give the oracle's outputs.
#[test]
fn one_shot_polynomials_dispatch_to_independent_prepared_overrides() {
    use poulpy_ckks::polynomial::ComplexBSGSPolynomial;
    let module_ref = Module::<FFT64Oracle>::new(64);
    let module_test = Module::<OverrideBackend>::new(64);
    let layout = poulpy_ckks::CKKSLayout {
        glwe_layout: poulpy_core::layouts::GLWELayout {
            n: 64usize.into(),
            base2k: 16usize.into(),
            k: 64usize.into(),
            rank: 1usize.into(),
        },
        meta: GLWEPlaintextMeta {
            scale: Scale::Log(16),
            ..Default::default()
        },
    };
    let src = fixture_ciphertext(&module_test, &layout, 61);
    let key_infos = poulpy_core::layouts::GLWETensorKeyLayout {
        n: 64usize.into(),
        base2k: 16usize.into(),
        k_aux: 16usize.into(),
        rank: 1usize.into(),
        dnum: 4usize.into(),
        k_digit: 16usize.into(),
    };
    let coeff_meta = poulpy_ckks::CKKSLayout {
        glwe_layout: poulpy_core::layouts::GLWELayout {
            n: 2usize.into(),
            base2k: 16usize.into(),
            k: 32usize.into(),
            rank: 0usize.into(),
        },
        meta: GLWEPlaintextMeta {
            scale: Scale::Log(8),
            log_sparsity: 0,
            ..Default::default()
        },
    };
    let bytes = module_test.ckks_all_ops_tmp_bytes(&src, &coeff_meta, &key_infos);
    // The ordinary reference families use less than COPY_WORKSPACE here.
    assert_eq!(bytes, module_test.ckks_copy_tmp_bytes(&src, &src));
    let bytes_ref = module_ref.ckks_all_ops_tmp_bytes(&src, &coeff_meta, &key_infos);
    let real = polynomial(&module_test);
    let complex = ComplexBSGSPolynomial {
        re: polynomial(&module_test),
        im: polynomial(&module_test),
    };

    let mut want = fixture_ciphertext(&module_ref, &layout, 62);
    with_scratch::<FFT64Oracle, _>(bytes_ref, |s| {
        module_ref.ckks_eval_poly_real_const_coeffs(&mut want, &src, &real, &NoTensorKey, s)
    })
    .unwrap();
    let mut dst = fixture_ciphertext(&module_test, &layout, 62);
    REAL_CALLS.set(0);
    with_scratch::<OverrideBackend, _>(bytes, |s| {
        module_test.ckks_eval_poly_real_const_coeffs(&mut dst, &src, &real, &NoTensorKey, s)
    })
    .unwrap();
    assert_eq!(REAL_CALLS.get(), 1);
    assert_eq!(
        snapshot::<OverrideBackend, _>(&dst),
        snapshot::<FFT64Oracle, _>(&want),
        "ckks_eval_poly_real_const_coeffs"
    );

    let mut want = fixture_ciphertext(&module_ref, &layout, 62);
    with_scratch::<FFT64Oracle, _>(bytes_ref, |s| {
        module_ref.ckks_eval_poly_complex_const_coeffs(&mut want, &src, &complex, &NoTensorKey, s)
    })
    .unwrap();
    let mut dst = fixture_ciphertext(&module_test, &layout, 62);
    COMPLEX_CALLS.set(0);
    with_scratch::<OverrideBackend, _>(bytes, |s| {
        module_test.ckks_eval_poly_complex_const_coeffs(&mut dst, &src, &complex, &NoTensorKey, s)
    })
    .unwrap();
    assert_eq!(COMPLEX_CALLS.get(), 1);
    assert_eq!(
        snapshot::<OverrideBackend, _>(&dst),
        snapshot::<FFT64Oracle, _>(&want),
        "ckks_eval_poly_complex_const_coeffs"
    );
}

#[test]
fn eval_mod_dispatch_uses_its_independent_scratch_query() {
    use poulpy_ckks::layouts::eval_mod::{EvalMod, EvalModBsgs, EvalModPlan, EvalModPoly};
    use poulpy_ckks::polynomial::{Basis, Polynomial, SplitStrategy};
    let module = Module::<OverrideBackend>::new(64);
    let mut src = module.ckks_ciphertext_alloc(16usize.into(), 64usize.into());
    src.set_meta(GLWEPlaintextMeta {
        scale: Scale::Log(16),
        ..Default::default()
    });
    let mut dst = module.ckks_ciphertext_alloc_from_infos(&src);
    let params = EvalMod {
        plan: EvalModPlan::complex_exponential(
            1,
            1,
            0,
            SplitStrategy::MinDepth,
            CoeffsMeta {
                k: 32usize.into(),
                meta: GLWEPlaintextMeta {
                    scale: Scale::Log(8),
                    ..Default::default()
                },
            },
            16,
        ),
        range_extension_consts: None,
        f_mod_input_offset: None,
        f_mod_bsgs: EvalModBsgs::Real(polynomial(&module)),
        f_mod_inv_bsgs: None,
        f_mod_poly: EvalModPoly::Real(Polynomial::new(Basis::Monomial, vec![0.0f64, 1.0])),
        f_mod_inv_poly: None,
    };
    let key_infos = poulpy_core::layouts::GLWETensorKeyLayout {
        n: 64usize.into(),
        base2k: 16usize.into(),
        k_aux: 16usize.into(),
        rank: 1usize.into(),
        dnum: 4usize.into(),
        k_digit: 16usize.into(),
    };
    let bytes = module.ckks_eval_mod_tmp_bytes(&dst, &src, &params, &key_infos);
    assert_eq!(bytes, OVERRIDE_SCRATCH);
    module.fill_glwe_from_source(&mut dst, &mut poulpy_hal::source::Source::new([5u8; 32]));
    let unchanged = snapshot::<OverrideBackend, _>(&dst);
    EVAL_MOD_CALLS.with(|calls| calls.set(0));
    let err = with_scratch::<OverrideBackend, _>(bytes, |s| module.ckks_eval_mod(&mut dst, &src, &params, &NoTensorKey, s))
        .unwrap_err();
    assert_ckks_internal("ckks_eval_mod", &err, "EvalMod override probe");
    assert_eq!(EVAL_MOD_CALLS.with(Cell::get), 1);
    assert_eq!(
        snapshot::<OverrideBackend, _>(&dst),
        unchanged,
        "the probe wrote the destination"
    );
}

#[test]
fn eval_mod_reference_sizes_the_final_destination_copy() {
    use poulpy_ckks::{
        CKKSInfos,
        layouts::eval_mod::{EvalMod, EvalModBsgs, EvalModPlan, EvalModPoly},
        polynomial::{Basis, ComplexBSGSPolynomial, ComplexPolynomial, EncodeBSGS, Polynomial, SplitStrategy},
        reference::eval_mod::{CKKSEvalModReference, ckks_eval_mod_tmp_bytes_reference},
    };
    use poulpy_hal::layouts::{HostBytesBackend, ZnxViewMut};
    let module_test = Module::<OverrideBackend>::new(64);
    let module_ref = Module::<FFT64Oracle>::new(64);
    let host = Module::<HostBytesBackend>::new(64);
    let coeff_meta = CoeffsMeta::from_delta_budget(8, 8);
    let poly = || Polynomial::new(Basis::Monomial, vec![0.125f64, 0.5]);
    let encoded = || {
        poly()
            .encode_bsgs_with(&host, 16usize.into(), coeff_meta, SplitStrategy::MinDepth)
            .unwrap()
    };
    let key_infos = poulpy_core::layouts::GLWETensorKeyLayout {
        n: 64usize.into(),
        base2k: 16usize.into(),
        k_aux: 16usize.into(),
        rank: 1usize.into(),
        dnum: 4usize.into(),
        k_digit: 16usize.into(),
    };
    let mut src = module_test.ckks_ciphertext_alloc(16usize.into(), 64usize.into());
    src.set_meta(GLWEPlaintextMeta {
        scale: Scale::Log(16),
        ..Default::default()
    });
    src.data_mut().at_mut(0, 0)[0] = 128;
    src.data_mut().at_mut(1, 0)[1] = -64;
    let source_before = snapshot::<OverrideBackend, _>(&src);
    for (complex, inverse) in [(false, false), (true, false), (false, true)] {
        let params = EvalMod {
            plan: EvalModPlan::complex_exponential(1, 1, 0, SplitStrategy::MinDepth, coeff_meta, 16),
            range_extension_consts: None,
            f_mod_input_offset: None,
            f_mod_bsgs: if complex {
                EvalModBsgs::Complex(ComplexBSGSPolynomial {
                    re: encoded(),
                    im: encoded(),
                })
            } else {
                EvalModBsgs::Real(encoded())
            },
            f_mod_inv_bsgs: inverse.then(encoded),
            f_mod_poly: if complex {
                EvalModPoly::Complex(ComplexPolynomial::new(Basis::Monomial, vec![0.125, 0.5], vec![0.125, 0.5]))
            } else {
                EvalModPoly::Real(poly())
            },
            f_mod_inv_poly: inverse.then(poly),
        };
        for capacity_k in [128usize, 192] {
            let mut dst = module_test.ckks_ciphertext_alloc(src.base2k(), capacity_k.into());
            dst.set_meta(src.meta());
            dst.set_k(src.k());
            dst.data_mut().raw_mut().fill(0x55);
            assert!(dst.max_size() > src.max_size());
            let capacity = dst.max_size();
            let mut expected = dst.clone();
            let expected_bytes = module_ref.ckks_eval_mod_tmp_bytes(&expected, &src, &params, &key_infos);
            with_scratch::<FFT64Oracle, _>(expected_bytes, |s| {
                module_ref.ckks_eval_mod(&mut expected, &src, &params, &NoTensorKey, s)
            })
            .unwrap();

            // Bypass the independent EvalMod dispatch probe and exercise its
            // real reference circuit with the destination-dependent copy override.
            let bytes = ckks_eval_mod_tmp_bytes_reference(&module_test, &dst, &src, &params, &key_infos);
            MAX_COPY_CAPACITY.set(0);
            with_scratch::<OverrideBackend, _>(bytes, |s| {
                module_test.ckks_eval_mod_reference(&mut dst, &src, &params, &NoTensorKey, s)
            })
            .unwrap();
            assert_eq!(MAX_COPY_CAPACITY.get(), capacity);
            assert_eq!(
                snapshot::<OverrideBackend, _>(&dst),
                snapshot::<FFT64Oracle, _>(&expected),
                "EvalMod differs from the oracle's"
            );
            assert_eq!(dst.max_size(), capacity);
            assert_eq!(
                snapshot::<OverrideBackend, _>(&src),
                source_before,
                "EvalMod changed its source"
            );
        }
    }
}

/// A one-term sum and an empty window copy: both queries are their documented maxima, which the overriding copy sets,
/// both enter that copy at the destination's capacity and give the oracle's outputs, and calls the API refuses before
/// any scratch never reach it.
#[test]
fn add_many_and_inner_sum_queries_cover_the_selected_copy() {
    use poulpy_ckks::api::{CKKSAddManyOps, CKKSAddOps, CKKSInnerSumOps};
    use poulpy_core::layouts::{GLWEAutomorphismKeyLayout, GLWEAutomorphismKeyPrepared};
    use poulpy_core::{GLWENormalize, GLWETrace};
    use poulpy_hal::layouts::ZnxViewMut;
    use std::collections::HashMap;
    let module_test = Module::<OverrideBackend>::new(64);
    let module_ref = Module::<FFT64Oracle>::new(64);
    let mut a = module_test.ckks_ciphertext_alloc(16usize.into(), 64usize.into());
    a.set_meta(GLWEPlaintextMeta {
        scale: Scale::Log(16),
        ..Default::default()
    });
    a.data_mut().at_mut(0, 0)[0] = 128;
    a.data_mut().at_mut(1, 0)[1] = -64;
    let key = GLWEAutomorphismKeyLayout {
        n: 64usize.into(),
        base2k: 16usize.into(),
        dnum: 4usize.into(),
        k_aux: 16usize.into(),
        rank: 1usize.into(),
        k_digit: 16usize.into(),
    };
    let no_keys: HashMap<i64, GLWEAutomorphismKeyPrepared<_, OverrideBackend>> = HashMap::new();
    let no_keys_ref: HashMap<i64, GLWEAutomorphismKeyPrepared<_, FFT64Oracle>> = HashMap::new();
    let destination = || {
        let mut res = module_test.ckks_ciphertext_alloc(16usize.into(), 128usize.into());
        res.data_mut().raw_mut().fill(0x55);
        res
    };

    let mut res = destination();
    let mut expected = destination();
    let bytes = module_ref.ckks_add_many_tmp_bytes(&expected, &a);
    with_scratch::<FFT64Oracle, _>(bytes, |s| module_ref.ckks_add_many(usize::MAX, &mut expected, &[&a], s)).unwrap();
    let bytes = module_test.ckks_add_many_tmp_bytes(&res, &a);
    let copy = module_test.ckks_copy_tmp_bytes(&res, &a);
    assert_eq!(
        bytes,
        module_test
            .ckks_add_tmp_bytes(res.max_size())
            .max(module_test.glwe_normalize_tmp_bytes())
            .max(copy)
    );
    assert_eq!(bytes, copy, "the overriding copy does not bound the sum");
    MAX_COPY_CAPACITY.set(0);
    with_scratch::<OverrideBackend, _>(bytes, |s| module_test.ckks_add_many(usize::MAX, &mut res, &[&a], s)).unwrap();
    assert_eq!(MAX_COPY_CAPACITY.get(), res.max_size());
    assert_eq!(
        snapshot::<OverrideBackend, _>(&res),
        snapshot::<FFT64Oracle, _>(&expected),
        "ckks_add_many"
    );

    let mut res = destination();
    let mut expected = destination();
    let bytes = module_ref.ckks_inner_sum_tmp_bytes(&expected, &a, &key);
    with_scratch::<FFT64Oracle, _>(bytes, |s| {
        module_ref.ckks_inner_sum(0, 0, false, &mut expected, &a, &no_keys_ref, s)
    })
    .unwrap();
    let bytes = module_test.ckks_inner_sum_tmp_bytes(&res, &a, &key);
    let copy = module_test.ckks_copy_tmp_bytes(&res, &a);
    assert_eq!(bytes, copy.max(module_test.glwe_trace_assign_tmp_bytes(&res, &key)));
    assert_eq!(bytes, copy, "the overriding copy does not bound the inner sum");
    MAX_COPY_CAPACITY.set(0);
    with_scratch::<OverrideBackend, _>(bytes, |s| module_test.ckks_inner_sum(0, 0, false, &mut res, &a, &no_keys, s)).unwrap();
    assert_eq!(MAX_COPY_CAPACITY.get(), res.max_size());
    assert_eq!(
        snapshot::<OverrideBackend, _>(&res),
        snapshot::<FFT64Oracle, _>(&expected),
        "ckks_inner_sum"
    );

    MAX_COPY_CAPACITY.set(0);
    let mut res = destination();
    let unchanged = snapshot::<OverrideBackend, _>(&res);
    let err = with_scratch::<OverrideBackend, _>(0, |s| module_test.ckks_add_many(0, &mut res, &[&a], s)).unwrap_err();
    assert_ckks_error(
        "ckks_add_many with normalize_after 0",
        &err,
        CKKSCompositionError::InvalidOperand {
            op: "ckks_add_many",
            reason: "normalize_after must be at least 1".into(),
        },
    );
    assert_eq!(snapshot::<OverrideBackend, _>(&res), unchanged, "a refused sum wrote res");
    let err = with_scratch::<OverrideBackend, _>(0, |s| module_test.ckks_inner_sum(32, 1, false, &mut res, &a, &no_keys, s))
        .unwrap_err();
    assert_ckks_error(
        "ckks_inner_sum past the slots",
        &err,
        CKKSCompositionError::InvalidOperand {
            op: "ckks_inner_sum",
            reason: format!("2^1 batches of 2^32 slots exceed the {} slots", module_test.ckks_max_slots()),
        },
    );
    assert_eq!(
        snapshot::<OverrideBackend, _>(&res),
        unchanged,
        "a refused inner sum wrote res"
    );
    assert_eq!(MAX_COPY_CAPACITY.get(), 0);
}

// The query proxy keeps actual lower-layer arithmetic while independently
// increasing copy, shift, and rotation workspaces. No concrete backend layout is changed.
struct CoreQueryOverrides<'a> {
    module: &'a Module<OverrideBackend>,
    rotate_workspace: usize,
    copy_calls: RefCell<Vec<(TorusPrecision, usize)>>,
    imag_calls: RefCell<Vec<ImagCoreCall>>,
}

#[derive(Debug, PartialEq, Eq)]
enum ImagCoreCall {
    LeftShift(usize),
    Rotate(i64),
    RotateAssign(i64),
}
const CORE_COPY_WORKSPACE: usize = 3 << 20;
const CORE_SHIFT_WORKSPACE: usize = 2 << 20;
impl poulpy_ckks::reference::copy::CKKSCopyReference<OverrideBackend> for CoreQueryOverrides<'_> {}
impl poulpy_ckks::reference::rotate::CKKSRotateReference<OverrideBackend> for CoreQueryOverrides<'_> {}
impl poulpy_ckks::reference::conjugate::CKKSConjugateReference<OverrideBackend> for CoreQueryOverrides<'_> {}
impl poulpy_ckks::reference::imag::CKKSImagReference<OverrideBackend> for CoreQueryOverrides<'_> {}

impl poulpy_core::GLWECopy<OverrideBackend> for CoreQueryOverrides<'_> {
    fn glwe_copy_tmp_bytes<R: poulpy_core::layouts::GLWEInfos, A: poulpy_core::layouts::GLWEInfos>(
        &self,
        res: &R,
        a: &A,
    ) -> usize {
        self.copy_calls.borrow_mut().push((res.k(), res.max_size()));
        CORE_COPY_WORKSPACE
            + (res.k().as_usize() + res.max_size()) * 64
            + poulpy_core::GLWECopy::glwe_copy_tmp_bytes(self.module, res, a)
    }
    fn glwe_copy<R, A>(&self, res: &mut R, a: &A, scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, OverrideBackend>)
    where
        R: poulpy_core::layouts::GLWEToBackendMut<OverrideBackend>,
        A: poulpy_core::layouts::GLWEToBackendRef<OverrideBackend>,
    {
        let layout = {
            let res = res.to_backend_ref();
            (res.k(), res.max_size())
        };
        self.copy_calls.borrow_mut().push(layout);
        let (_workspace, mut remaining) = scratch
            .borrow()
            .take_region(CORE_COPY_WORKSPACE + (layout.0.as_usize() + layout.1) * 64);
        poulpy_core::GLWECopy::glwe_copy(self.module, res, a, &mut remaining)
    }
}

impl poulpy_core::GLWEShift<OverrideBackend> for CoreQueryOverrides<'_> {
    fn glwe_shift_tmp_bytes(&self, res_size: usize) -> usize {
        CORE_SHIFT_WORKSPACE + poulpy_core::GLWEShift::glwe_shift_tmp_bytes(self.module, res_size)
    }
    fn glwe_rsh<R>(&self, k: usize, res: &mut R, scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, OverrideBackend>)
    where
        R: poulpy_core::layouts::GLWEToBackendMut<OverrideBackend>,
    {
        let (_workspace, mut remaining) = scratch.borrow().take_region(CORE_SHIFT_WORKSPACE);
        poulpy_core::GLWEShift::glwe_rsh(self.module, k, res, &mut remaining)
    }
    fn glwe_lsh_assign<R>(&self, res: &mut R, k: usize, scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, OverrideBackend>)
    where
        R: poulpy_core::layouts::GLWEToBackendMut<OverrideBackend>,
    {
        let (_workspace, mut remaining) = scratch.borrow().take_region(CORE_SHIFT_WORKSPACE);
        poulpy_core::GLWEShift::glwe_lsh_assign(self.module, res, k, &mut remaining)
    }
    fn glwe_lsh<R, A>(&self, res: &mut R, a: &A, k: usize, scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, OverrideBackend>)
    where
        R: poulpy_core::layouts::GLWEToBackendMut<OverrideBackend>,
        A: poulpy_core::layouts::GLWEToBackendRef<OverrideBackend>,
    {
        self.imag_calls.borrow_mut().push(ImagCoreCall::LeftShift(k));
        let (_workspace, mut remaining) = scratch.borrow().take_region(CORE_SHIFT_WORKSPACE);
        poulpy_core::GLWEShift::glwe_lsh(self.module, res, a, k, &mut remaining)
    }
    fn glwe_lsh_add<R, A>(
        &self,
        res: &mut R,
        a: &A,
        k: usize,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, OverrideBackend>,
    ) where
        R: poulpy_core::layouts::GLWEToBackendMut<OverrideBackend>,
        A: poulpy_core::layouts::GLWEToBackendRef<OverrideBackend>,
    {
        let (_workspace, mut remaining) = scratch.borrow().take_region(CORE_SHIFT_WORKSPACE);
        poulpy_core::GLWEShift::glwe_lsh_add(self.module, res, a, k, &mut remaining)
    }
    fn glwe_lsh_sub<R, A>(
        &self,
        res: &mut R,
        a: &A,
        k: usize,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, OverrideBackend>,
    ) where
        R: poulpy_core::layouts::GLWEToBackendMut<OverrideBackend>,
        A: poulpy_core::layouts::GLWEToBackendRef<OverrideBackend>,
    {
        let (_workspace, mut remaining) = scratch.borrow().take_region(CORE_SHIFT_WORKSPACE);
        poulpy_core::GLWEShift::glwe_lsh_sub(self.module, res, a, k, &mut remaining)
    }
}

impl poulpy_hal::api::ModuleN for CoreQueryOverrides<'_> {
    fn n(&self) -> usize {
        poulpy_hal::api::ModuleN::n(self.module)
    }
}

impl poulpy_core::GLWERotate<OverrideBackend> for CoreQueryOverrides<'_> {
    fn glwe_rotate_tmp_bytes(&self) -> usize {
        self.rotate_workspace + poulpy_core::GLWERotate::glwe_rotate_tmp_bytes(self.module)
    }
    fn glwe_rotate<R, A>(&self, k: i64, res: &mut R, a: &A)
    where
        R: poulpy_core::layouts::GLWEToBackendMut<OverrideBackend>,
        A: poulpy_core::layouts::GLWEToBackendRef<OverrideBackend>,
    {
        self.imag_calls.borrow_mut().push(ImagCoreCall::Rotate(k));
        poulpy_core::GLWERotate::glwe_rotate(self.module, k, res, a)
    }
    fn glwe_rotate_assign<R>(&self, k: i64, res: &mut R, scratch: &mut poulpy_hal::layouts::ScratchArena<'_, OverrideBackend>)
    where
        R: poulpy_core::layouts::GLWEToBackendMut<OverrideBackend>,
    {
        self.imag_calls.borrow_mut().push(ImagCoreCall::RotateAssign(k));
        let (_workspace, mut remaining) = scratch.borrow().take_region(self.rotate_workspace);
        poulpy_core::GLWERotate::glwe_rotate_assign(self.module, k, res, &mut remaining)
    }
}

impl poulpy_core::GLWEAutomorphism<OverrideBackend> for CoreQueryOverrides<'_> {
    fn glwe_automorphism_tmp_bytes<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: poulpy_core::layouts::GLWEInfos,
        A: poulpy_core::layouts::GLWEInfos,
        K: poulpy_core::layouts::GGLWEInfos,
    {
        poulpy_core::GLWEAutomorphism::glwe_automorphism_tmp_bytes(self.module, res_infos, a_infos, key_infos)
    }
    fn glwe_automorphism<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &poulpy_core::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, OverrideBackend>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, OverrideBackend>,
    ) where
        R: poulpy_core::layouts::GLWEToBackendMut<OverrideBackend> + poulpy_core::layouts::GLWEInfos,
        A: poulpy_core::layouts::GLWEToBackendRef<OverrideBackend> + poulpy_core::layouts::GLWEInfos,
    {
        poulpy_core::GLWEAutomorphism::glwe_automorphism(self.module, res, a, key, scratch)
    }
    fn glwe_automorphism_assign<R>(
        &self,
        res: &mut R,
        key: &poulpy_core::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, OverrideBackend>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, OverrideBackend>,
    ) where
        R: poulpy_core::layouts::GLWEToBackendMut<OverrideBackend> + poulpy_core::layouts::GLWEInfos,
    {
        poulpy_core::GLWEAutomorphism::glwe_automorphism_assign(self.module, res, key, scratch)
    }
    fn glwe_automorphism_add<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &poulpy_core::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, OverrideBackend>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, OverrideBackend>,
    ) where
        R: poulpy_core::layouts::GLWEToBackendMut<OverrideBackend> + poulpy_core::layouts::GLWEInfos,
        A: poulpy_core::layouts::GLWEToBackendRef<OverrideBackend> + poulpy_core::layouts::GLWEInfos,
    {
        poulpy_core::GLWEAutomorphism::glwe_automorphism_add(self.module, res, a, key, scratch)
    }
    fn glwe_automorphism_add_assign<R>(
        &self,
        res: &mut R,
        key: &poulpy_core::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, OverrideBackend>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, OverrideBackend>,
    ) where
        R: poulpy_core::layouts::GLWEToBackendMut<OverrideBackend> + poulpy_core::layouts::GLWEInfos,
    {
        poulpy_core::GLWEAutomorphism::glwe_automorphism_add_assign(self.module, res, key, scratch)
    }
    fn glwe_automorphism_sub<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &poulpy_core::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, OverrideBackend>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, OverrideBackend>,
    ) where
        R: poulpy_core::layouts::GLWEToBackendMut<OverrideBackend> + poulpy_core::layouts::GLWEInfos,
        A: poulpy_core::layouts::GLWEToBackendRef<OverrideBackend> + poulpy_core::layouts::GLWEInfos,
    {
        poulpy_core::GLWEAutomorphism::glwe_automorphism_sub(self.module, res, a, key, scratch)
    }
    fn glwe_automorphism_sub_negate<R, A>(
        &self,
        res: &mut R,
        a: &A,
        key: &poulpy_core::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, OverrideBackend>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, OverrideBackend>,
    ) where
        R: poulpy_core::layouts::GLWEToBackendMut<OverrideBackend> + poulpy_core::layouts::GLWEInfos,
        A: poulpy_core::layouts::GLWEToBackendRef<OverrideBackend> + poulpy_core::layouts::GLWEInfos,
    {
        poulpy_core::GLWEAutomorphism::glwe_automorphism_sub_negate(self.module, res, a, key, scratch)
    }
    fn glwe_automorphism_sub_assign<R>(
        &self,
        res: &mut R,
        key: &poulpy_core::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, OverrideBackend>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, OverrideBackend>,
    ) where
        R: poulpy_core::layouts::GLWEToBackendMut<OverrideBackend> + poulpy_core::layouts::GLWEInfos,
    {
        poulpy_core::GLWEAutomorphism::glwe_automorphism_sub_assign(self.module, res, key, scratch)
    }
    fn glwe_automorphism_sub_negate_assign<R>(
        &self,
        res: &mut R,
        key: &poulpy_core::layouts::prepared::GLWEAutomorphismKeyPreparedBackendRef<'_, OverrideBackend>,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, OverrideBackend>,
    ) where
        R: poulpy_core::layouts::GLWEToBackendMut<OverrideBackend> + poulpy_core::layouts::GLWEInfos,
    {
        poulpy_core::GLWEAutomorphism::glwe_automorphism_sub_negate_assign(self.module, res, key, scratch)
    }
}

#[test]
fn reference_queries_follow_independent_core_copy_and_shift_workspaces() {
    use poulpy_ckks::CKKSInfos;
    use poulpy_ckks::reference::{conjugate::CKKSConjugateReference, copy::CKKSCopyReference, rotate::CKKSRotateReference};
    use poulpy_core::{GLWECopy, GLWEShift};
    use poulpy_hal::layouts::ZnxViewMut;
    let module = Module::<OverrideBackend>::new(64);
    let query = CoreQueryOverrides {
        module: &module,
        rotate_workspace: 0,
        copy_calls: RefCell::default(),
        imag_calls: RefCell::default(),
    };
    let mut src = module.ckks_ciphertext_alloc(16usize.into(), 64usize.into());
    src.set_meta(GLWEPlaintextMeta {
        scale: Scale::Log(16),
        ..Default::default()
    });
    for source_k in [64usize, 61] {
        SetK::set_k(&mut src, source_k.into());
        for col in 0..2 {
            for limb in 0..src.max_size() {
                let padding = ((limb + 1) * 16).saturating_sub(source_k);
                for (j, digit) in src.data_mut().at_mut(col, limb).iter_mut().enumerate() {
                    let value = [-32768i64, -9, 7, 32767][(j + col + limb) % 4];
                    *digit = (value >> padding) << padding;
                }
            }
        }
        for base in [16usize, 15, 17] {
            let mut dst = module.ckks_ciphertext_alloc(base.into(), 128usize.into());
            dst.data_mut().raw_mut().fill(0x55);
            let original_layout = (dst.k(), dst.max_size());
            // The previous ordering stamped the final width before copying.
            let mut expected = dst.clone();
            expected.set_meta(src.meta());
            expected.set_log_budget(src.log_budget());
            with_scratch::<OverrideBackend, _>(module.glwe_copy_tmp_bytes(&expected, &src), |s| {
                module.glwe_copy(&mut expected, &src, s)
            });

            query.copy_calls.borrow_mut().clear();
            let bytes = query.ckks_copy_tmp_bytes_reference(&dst, &src);
            with_scratch::<OverrideBackend, _>(bytes, |s| query.ckks_copy_reference(&mut dst, &src, s)).unwrap();
            assert_eq!(*query.copy_calls.borrow(), [original_layout, original_layout]);
            assert!(dst.data() == expected.data(), "the copy differs from glwe_copy");
            assert_eq!(dst.meta(), src.meta());
            assert_eq!(dst.k(), src.k());
            assert_eq!(dst.max_size(), original_layout.1);
        }
    }
    let key = poulpy_core::layouts::GLWETensorKeyLayout {
        n: 64usize.into(),
        base2k: 16usize.into(),
        k_aux: 16usize.into(),
        rank: 1usize.into(),
        dnum: 4usize.into(),
        k_digit: 16usize.into(),
    };
    // Both queries are the larger of the automorphism's and the aligning shift's, the overriding shift here.
    let shift_bytes = query.glwe_shift_tmp_bytes(src.max_size());
    let automorphism_bytes = poulpy_core::GLWEAutomorphism::glwe_automorphism_tmp_bytes(&query, &src, &src, &key);
    assert!(shift_bytes > automorphism_bytes);
    assert_eq!(query.ckks_rotate_with_key_tmp_bytes_reference(&src, &key), shift_bytes);
    assert_eq!(query.ckks_conjugate_with_key_tmp_bytes_reference(&src, &key), shift_bytes);
}

#[test]
fn division_by_i_uses_negative_monomials_and_selected_core_workspaces() {
    use poulpy_ckks::{CKKSInfos, SlotsKind, reference::imag::CKKSImagReference};
    use poulpy_core::{GLWERotate, GLWEShift};

    let module = Module::<OverrideBackend>::new(64);
    let mut src = module.ckks_ciphertext_alloc(16usize.into(), 64usize.into());
    src.set_meta(GLWEPlaintextMeta {
        scale: Scale::Log(16),
        slots: SlotsKind::Real,
        log_sparsity: 1,
    });
    // Exercise both choices of dominant constituent workspace. The proxy has
    // no CKKS multiplication-by-i or negation contract to compose through.
    for rotate_workspace in [1 << 20, 4 << 20] {
        let query = CoreQueryOverrides {
            module: &module,
            rotate_workspace,
            copy_calls: RefCell::default(),
            imag_calls: RefCell::default(),
        };
        for output_k in [64usize, 47] {
            let mut dst = module.ckks_ciphertext_alloc(16usize.into(), output_k.into());
            let shift_bytes = query.glwe_shift_tmp_bytes(dst.max_size());
            let rotate_bytes = query.glwe_rotate_tmp_bytes();
            assert_eq!(rotate_bytes > shift_bytes, rotate_workspace > CORE_SHIFT_WORKSPACE);
            let bytes = query.ckks_div_i_tmp_bytes_reference(dst.max_size());
            assert_eq!(bytes, shift_bytes.max(rotate_bytes));
            query.imag_calls.borrow_mut().clear();
            with_scratch::<OverrideBackend, _>(bytes, |s| query.ckks_div_i_into_reference(&mut dst, &src, s)).unwrap();
            let expected = if output_k == 64 {
                vec![ImagCoreCall::Rotate(-32)]
            } else {
                vec![ImagCoreCall::LeftShift(64 - output_k), ImagCoreCall::RotateAssign(-32)]
            };
            assert_eq!(*query.imag_calls.borrow(), expected);
            assert_eq!(dst.log_delta(), src.log_delta());
            assert_eq!(dst.log_budget(), output_k - src.log_delta());
            assert_eq!(dst.log_sparsity(), src.log_sparsity());
            assert_eq!(dst.slots(), SlotsKind::Complex);

            dst.set_slots(SlotsKind::Real);
            let meta = GLWEPlaintextMeta {
                slots: SlotsKind::Complex,
                ..dst.meta()
            };
            let k = dst.k();
            query.imag_calls.borrow_mut().clear();
            with_scratch::<OverrideBackend, _>(bytes, |s| query.ckks_div_i_assign_reference(&mut dst, s)).unwrap();
            assert_eq!(*query.imag_calls.borrow(), [ImagCoreCall::RotateAssign(-32)]);
            assert_eq!(dst.meta(), meta);
            assert_eq!(dst.k(), k);
        }
    }
}

#[test]
fn decrypt_query_uses_plaintext_allocation_width() {
    use poulpy_ckks::api::CKKSDecryptOps;
    use poulpy_ckks::reference::CKKSPlaintextVecReference;
    use poulpy_core::{GLWEBytesOf, GLWEDecrypt};

    let module = Module::<OverrideBackend>::new(64);
    let ct = module.ckks_ciphertext_alloc(16usize.into(), 64usize.into());
    let narrow = module.ckks_pt_vec_alloc(module.ckks_max_slots(), 16usize.into(), 64usize.into());
    let mut wide = module.ckks_pt_vec_alloc(module.ckks_max_slots(), 16usize.into(), 1024usize.into());
    SetK::set_k(&mut wide, 32usize.into());
    let expected = |pt_size: usize| {
        module.glwe_plaintext_bytes_of_from_infos(&ct)
            + module
                .glwe_decrypt_tmp_bytes(&ct)
                .max(module.ckks_extract_pt_tmp_bytes_reference(pt_size))
    };
    assert_eq!(module.ckks_decrypt_tmp_bytes(&wide, &ct), expected(wide.max_size()));
    assert_eq!(module.ckks_decrypt_tmp_bytes(&narrow, &ct), expected(narrow.max_size()));
    assert!(
        expected(wide.max_size()) > expected(narrow.max_size()),
        "the plaintext allocation width does not reach the query"
    );
}
