//! Downstream specialization probes: reference families remain available while
//! polynomial kernels and EvalMod independently select execution and scratch.
use crate::hal_impl::delegating_backend::DifferentSamplingFFT64Ref as OverrideBackend;
use poulpy_ckks::api::{CKKSAllOpsTmpBytes, CKKSCopyOps, CKKSEvalModOps, CKKSPolynomialEvaluationOps};
use poulpy_ckks::layouts::{CKKSModuleAlloc, CKKSPlaintextOwned};
use poulpy_ckks::{CKKSMeta, CoeffsMeta, SetCKKSInfos};
use poulpy_core::layouts::{GetTensorKey, LWEInfos, TorusPrecision, prepared::GLWETensorKeyPreparedBackendRef};
use poulpy_hal::api::{ScratchOwnedAlloc, ScratchOwnedBorrow};
use poulpy_hal::layouts::{Backend, Module, ScratchOwned};
use std::cell::{Cell, RefCell};
thread_local! {
 static REAL_CALLS: Cell<usize> = const { Cell::new(0) };
 static COMPLEX_CALLS: Cell<usize> = const { Cell::new(0) };
 static EVAL_MOD_CALLS: Cell<usize> = const { Cell::new(0) };
}
const OVERRIDE_SCRATCH: usize = 256;
poulpy_ckks::impl_ckks_plaintext_reference!(OverrideBackend);
const COPY_WORKSPACE: usize = 1 << 20;
unsafe impl poulpy_ckks::oep::CKKSCopyImpl for OverrideBackend {
    fn ckks_copy_tmp_bytes_impl<Dst: poulpy_ckks::CKKSCtBounds, Src: poulpy_ckks::CKKSCtBounds>(
        module: &Module<Self>,
        dst: &Dst,
        src: &Src,
    ) -> usize {
        COPY_WORKSPACE + poulpy_ckks::reference::copy::CKKSCopyReference::ckks_copy_tmp_bytes_reference(module, dst, src)
    }
    fn ckks_copy_impl<Dst, Src>(
        module: &Module<Self>,
        dst: &mut Dst,
        src: &Src,
        scratch: &mut poulpy_hal::layouts::ScratchArena<'_, Self>,
    ) -> poulpy_ckks::CKKSResult<()>
    where
        Dst: poulpy_core::layouts::GLWEToBackendMut<Self> + poulpy_ckks::CKKSCtBounds + SetCKKSInfos,
        Src: poulpy_core::layouts::GLWEToBackendRef<Self> + poulpy_ckks::CKKSCtBounds,
    {
        let (mut region, mut remaining) = scratch.borrow().take_region(COPY_WORKSPACE);
        Self::copy_host_to_view(&mut region, &vec![0x5A; COPY_WORKSPACE]);
        poulpy_ckks::reference::copy::CKKSCopyReference::ckks_copy_reference(module, dst, src, &mut remaining)
    }
}
poulpy_ckks::impl_ckks_add_reference!(OverrideBackend);
poulpy_ckks::impl_ckks_sub_reference!(OverrideBackend);
poulpy_ckks::impl_ckks_neg_reference!(OverrideBackend);
poulpy_ckks::impl_ckks_pow2_reference!(OverrideBackend);
poulpy_ckks::impl_ckks_imag_reference!(OverrideBackend);
poulpy_ckks::impl_ckks_rotate_reference!(OverrideBackend);
poulpy_ckks::impl_ckks_conjugate_reference!(OverrideBackend);
poulpy_ckks::impl_ckks_encryption_reference!(OverrideBackend);
poulpy_ckks::impl_ckks_mul_reference!(OverrideBackend);

unsafe impl poulpy_ckks::oep::CKKSPolynomialEvaluationImpl for OverrideBackend {
    fn ckks_eval_poly_real_const_coeffs_from_power_basis_impl<R, B, A, G, H>(
        module: &::poulpy_hal::layouts::Module<Self>,
        res: &mut R,
        poly: &B,
        power_basis: &G,
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
        let _ = (module, res, poly, power_basis, tsk, scratch);
        Err(anyhow::anyhow!("polynomial override probe").into())
    }
    fn ckks_eval_poly_complex_const_coeffs_from_power_basis_impl<R, C, A, G, H>(
        module: &::poulpy_hal::layouts::Module<Self>,
        res: &mut R,
        poly: &poulpy_ckks::polynomial::ComplexBSGSPolynomial<C>,
        power_basis: &G,
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
        let _ = (module, res, poly, power_basis, tsk, scratch);
        Err(anyhow::anyhow!("polynomial override probe").into())
    }
}

unsafe impl poulpy_ckks::oep::CKKSEvalModImpl for OverrideBackend {
    fn ckks_eval_mod_tmp_bytes_impl<R, C, P, F, T>(
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
    fn ckks_eval_mod_impl<R, C, P, F, H>(
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
impl GetTensorKey<OverrideBackend> for NoTensorKey {
    fn get_tensor_key(&self, _: TorusPrecision) -> poulpy_core::Result<GLWETensorKeyPreparedBackendRef<'_, OverrideBackend>> {
        panic!("linear polynomial dispatch must not request a tensor key")
    }
}

fn polynomial(module: &Module<OverrideBackend>) -> poulpy_ckks::polynomial::BSGSPolynomial<CKKSPlaintextOwned<OverrideBackend>> {
    use poulpy_ckks::polynomial::{Basis, Polynomial, SplitStrategy};
    Polynomial::new(Basis::Monomial, vec![0.0f64, 1.0])
        .decompose_bsgs_with(SplitStrategy::MinDepth, |coeffs| -> anyhow::Result<_> {
            let mut pt = module.ckks_pt_coeffs_alloc(coeffs.len(), 16usize.into(), 32usize.into());
            pt.set_meta(CKKSMeta {
                log_delta: 8,
                ..Default::default()
            });
            Ok(pt)
        })
        .unwrap()
}

#[test]
fn one_shot_polynomials_dispatch_to_independent_prepared_overrides() {
    use poulpy_ckks::polynomial::ComplexBSGSPolynomial;
    let module = Module::<OverrideBackend>::new(64);
    let mut src = module.ckks_ciphertext_alloc(16usize.into(), 64usize.into());
    src.set_meta(CKKSMeta {
        log_delta: 16,
        ..Default::default()
    });
    let mut dst = module.ckks_ciphertext_alloc_from_infos(&src);
    let key_infos = poulpy_core::layouts::GLWETensorKeyLayout {
        n: 64usize.into(),
        base2k: 16usize.into(),
        k_aux: 16usize.into(),
        rank: 1usize.into(),
        dnum: 4usize.into(),
        dsize: 1usize.into(),
    };
    let coeff_meta = poulpy_ckks::CKKSLayout {
        glwe_layout: poulpy_core::layouts::GLWELayout {
            n: 2usize.into(),
            base2k: 16usize.into(),
            k: 32usize.into(),
            rank: 0usize.into(),
        },
        meta: CKKSMeta {
            log_delta: 8,
            ..Default::default()
        },
    };
    let bytes = module.ckks_all_ops_tmp_bytes(&src, &key_infos, &coeff_meta);
    // The shared query must include this independently selected copy budget;
    // the ordinary reference families use less than COPY_WORKSPACE here.
    assert_eq!(bytes, module.ckks_copy_tmp_bytes(&src, &src));
    let mut owned = ScratchOwned::<OverrideBackend>::alloc(bytes);
    let (mut scratch, _) = owned.borrow().split_at(bytes);
    assert_eq!(scratch.available(), bytes);
    REAL_CALLS.with(|calls| calls.set(0));
    COMPLEX_CALLS.with(|calls| calls.set(0));
    let real = polynomial(&module);
    assert!(
        module
            .ckks_eval_poly_real_const_coeffs(&mut dst, &src, &real, &NoTensorKey, &mut scratch.borrow())
            .is_err()
    );
    assert_eq!(REAL_CALLS.with(Cell::get), 1);
    let complex = ComplexBSGSPolynomial {
        re: real,
        im: polynomial(&module),
    };
    assert!(
        module
            .ckks_eval_poly_complex_const_coeffs(&mut dst, &src, &complex, &NoTensorKey, &mut scratch.borrow())
            .is_err()
    );
    assert_eq!(COMPLEX_CALLS.with(Cell::get), 1);
}

#[test]
fn eval_mod_dispatch_uses_its_independent_scratch_query() {
    use poulpy_ckks::layouts::eval_mod::{EvalMod, EvalModBsgs, EvalModPlan, EvalModPoly};
    use poulpy_ckks::polynomial::{Basis, Polynomial, SplitStrategy};
    let module = Module::<OverrideBackend>::new(64);
    let mut src = module.ckks_ciphertext_alloc(16usize.into(), 64usize.into());
    src.set_meta(CKKSMeta {
        log_delta: 16,
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
                meta: CKKSMeta {
                    log_delta: 8,
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
        dsize: 1usize.into(),
    };
    let bytes = module.ckks_eval_mod_tmp_bytes(&dst, &src, &params, &key_infos);
    assert_eq!(bytes, OVERRIDE_SCRATCH);
    let mut scratch = ScratchOwned::<OverrideBackend>::alloc(bytes);
    EVAL_MOD_CALLS.with(|calls| calls.set(0));
    assert!(
        module
            .ckks_eval_mod(&mut dst, &src, &params, &NoTensorKey, &mut scratch.borrow())
            .is_err()
    );
    assert_eq!(EVAL_MOD_CALLS.with(Cell::get), 1);
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
    use poulpy_hal::layouts::{ZnxView, ZnxViewMut};
    let module = Module::<OverrideBackend>::new(64);
    let query = CoreQueryOverrides {
        module: &module,
        rotate_workspace: 0,
        copy_calls: RefCell::default(),
        imag_calls: RefCell::default(),
    };
    let mut src = module.ckks_ciphertext_alloc(16usize.into(), 64usize.into());
    src.set_meta(CKKSMeta {
        log_delta: 16,
        ..Default::default()
    });
    for source_k in [64usize, 61] {
        SetCKKSInfos::set_k(&mut src, source_k.into());
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
            let mut reference_scratch = ScratchOwned::<OverrideBackend>::alloc(module.glwe_copy_tmp_bytes(&expected, &src));
            module.glwe_copy(&mut expected, &src, &mut reference_scratch.borrow());

            query.copy_calls.borrow_mut().clear();
            let bytes = query.ckks_copy_tmp_bytes_reference(&dst, &src);
            let mut owned = ScratchOwned::<OverrideBackend>::alloc(bytes);
            let (mut exact, _) = owned.borrow().split_at(bytes);
            query.ckks_copy_reference(&mut dst, &src, &mut exact).unwrap();
            assert_eq!(*query.copy_calls.borrow(), [original_layout, original_layout]);
            assert_eq!(dst.data().raw(), expected.data().raw());
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
        dsize: 1usize.into(),
    };
    let shift_bytes = query.glwe_shift_tmp_bytes(src.max_size());
    assert!(query.ckks_rotate_tmp_bytes_reference(&src, &key) >= shift_bytes);
    assert!(query.ckks_conjugate_tmp_bytes_reference(&src, &key) >= shift_bytes);
}

#[test]
fn division_by_i_uses_negative_monomials_and_selected_core_workspaces() {
    use poulpy_ckks::{CKKSInfos, SlotsKind, reference::imag::CKKSImagReference};
    use poulpy_core::{GLWERotate, GLWEShift};

    let module = Module::<OverrideBackend>::new(64);
    let mut src = module.ckks_ciphertext_alloc(16usize.into(), 64usize.into());
    src.set_meta(CKKSMeta {
        log_delta: 16,
        log_sparsity: 1,
        slots: SlotsKind::Real,
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
            let mut owned = ScratchOwned::<OverrideBackend>::alloc(bytes);
            let (mut exact, _) = owned.borrow().split_at(bytes);

            query.imag_calls.borrow_mut().clear();
            query.ckks_div_i_into_reference(&mut dst, &src, &mut exact).unwrap();
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
            let meta = CKKSMeta {
                slots: SlotsKind::Complex,
                ..dst.meta()
            };
            let k = dst.k();
            query.imag_calls.borrow_mut().clear();
            query.ckks_div_i_assign_reference(&mut dst, &mut exact).unwrap();
            assert_eq!(*query.imag_calls.borrow(), [ImagCoreCall::RotateAssign(-32)]);
            assert_eq!(dst.meta(), meta);
            assert_eq!(dst.k(), k);
        }
    }
}

#[test]
fn decrypt_query_uses_plaintext_allocation_width() {
    use poulpy_ckks::api::CKKSDecryptOps;
    use poulpy_ckks::reference::CKKSPlaintextReference;
    use poulpy_core::{GLWEBytesOf, GLWEDecrypt};

    let module = Module::<OverrideBackend>::new(64);
    let ct = module.ckks_ciphertext_alloc(16usize.into(), 64usize.into());
    let narrow = module.ckks_pt_vec_alloc(16usize.into(), 64usize.into());
    let mut wide = module.ckks_pt_vec_alloc(16usize.into(), 1024usize.into());
    SetCKKSInfos::set_k(&mut wide, 32usize.into());
    let expected = module.glwe_plaintext_bytes_of_from_infos(&ct)
        + module
            .glwe_decrypt_tmp_bytes(&ct)
            .max(module.ckks_extract_pt_tmp_bytes_reference(wide.max_size()));
    assert_eq!(module.ckks_decrypt_tmp_bytes(&wide, &ct), expected);
    assert!(expected > module.ckks_decrypt_tmp_bytes(&narrow, &ct));
}
