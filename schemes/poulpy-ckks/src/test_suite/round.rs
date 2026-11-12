//! Rounding tests: plain and shaped precision reduction.
//!
//! # Test inventory
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_round_unit_shaping_is_plain`] | a replicated layout with `U = 1` rounds as a full layout |
//! | [`test_round_error_is_shaped`] | the rounding error is `−g·h·2^-(k+λ)` with `\|\|h\|\|_∞ <= 2^(λ−1)` |
//! | [`test_round_error_is_shaped_conjugate_invariant`] | the same on the conjugate-invariant ring, for the invariant shift of `U` |
//! | [`test_round_beyond_pass_rounds_plainly_first`] | a drop beyond the pass size rounds plainly to `k + λ`, then shaped |
//! | [`test_round_rejects_incompatible_layouts`] | mismatched ranks and degrees, and an arena one byte short, leave operands unchanged |
//! | [`test_round_rejects_invalid_widths`] | a wider destination and a drop beyond `log_delta`, leaving operands unchanged |

use anyhow::Result;
use poulpy_core::layouts::{GLWEInfos, GLWEToBackendRef, LWEInfos};
use poulpy_hal::{
    api::{ReimNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalize, VecZnxNormalizeTmpBytes},
    layouts::{
        ConjugateInvariant, HostBytesBackend, HostDataMut, HostDataRef, Module, Ring, ScratchOwned, VecZnxToBackendMut,
        VecZnxToBackendRef, ZnxView,
    },
};

use super::helpers::{
    TestContextBackend, TestContextModule, TestContextSharedModule, TestScalar, alloc_ct, alloc_scratch, ckks_encrypt,
    ckks_encrypt_pt, gen_sk, gen_sk_with_raw, test_vector_1,
};
use crate::{
    CKKSInfos, GLWEPlaintextInfos, ReplicatedLayout, Scale, SetCKKSInfos, SetGLWEPlaintextInfos, SlotLayout, SlotsKind,
    api::{CKKSEncodingHostOps, CKKSRoundOps},
    layouts::{CKKSCiphertextOwned, CKKSPlaintextAlloc},
    shaping::{ShapingProfile, ShapingUnit},
    test_suite::{CKKSTestParams, reference_encoder::ReferenceEncoder},
};

fn encrypted<BE, F, E>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> CKKSCiphertextOwned<BE>
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let (re, im) = test_vector_1::<F>(m);
    let sk = gen_sk(params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(params, module);
    ckks_encrypt(
        params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re,
        &im,
        &mut scratch.borrow(),
    )
}

fn round<BE, Rg: Ring>(
    module: &Module<BE>,
    src: &CKKSCiphertextOwned<BE, Rg>,
    params: &CKKSTestParams,
    k: usize,
) -> Result<CKKSCiphertextOwned<BE, Rg>>
where
    BE: TestContextBackend,
    Module<BE>: TestContextSharedModule<BE, Rg>,
{
    let mut dst = alloc_ct(params, module, k);
    // The exact queried budget, poisoned.
    crate::test_suite::parity::helpers::with_scratch::<BE, _>(module.ckks_round_tmp_bytes(&dst, src), |scratch| {
        module.ckks_round(&mut dst, src, scratch)
    })?;
    Ok(dst)
}

fn with_layout<BE: TestContextBackend>(ct: &CKKSCiphertextOwned<BE>, layout: SlotLayout) -> CKKSCiphertextOwned<BE> {
    let mut ct = ct.clone();
    ct.set_slot_layout(layout);
    ct
}

fn digits<BE>(ct: &CKKSCiphertextOwned<BE>) -> Vec<i64>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
{
    let glwe = GLWEToBackendRef::<BE>::to_backend_ref(ct);
    let data = glwe.data();
    (0..glwe.rank().as_usize() + 1)
        .flat_map(|col| (0..glwe.size()).flat_map(move |limb| data.at(col, limb).to_vec()))
        .collect()
}

/// `ct · 2^offset` read modulo `2^bits` to `bits` bits of precision, centered.
fn window<BE, Rg: Ring>(module: &Module<BE>, ct: &CKKSCiphertextOwned<BE, Rg>, col: usize, offset: i64, bits: usize) -> Vec<i64>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    Module<BE>: TestContextSharedModule<BE, Rg> + VecZnxNormalize<BE, Rg> + VecZnxNormalizeTmpBytes,
{
    let mut out = module.vec_znx_alloc::<Rg>(ct.n().as_usize(), 1, 1);
    let mut scratch = ScratchOwned::<BE>::alloc(module.vec_znx_normalize_tmp_bytes());
    let glwe = GLWEToBackendRef::<BE, Rg>::to_backend_ref(ct);
    module.vec_znx_normalize(
        &mut VecZnxToBackendMut::<BE, Rg>::to_backend_mut(&mut out),
        bits,
        bits,
        offset,
        0,
        glwe.data(),
        ct.base2k().as_usize(),
        col,
        &mut scratch.borrow(),
    );
    VecZnxToBackendRef::<BE, Rg>::to_backend_ref(&out).at(0, 0).to_vec()
}

/// `Σ_t coeffs[t] · X^(t·stride) · x` in `Z[X]/(X^N + 1)`, in `i128`.
fn sparse_mul(coeffs: &[i128], stride: usize, x: &[i128]) -> Vec<i128> {
    let n = x.len();
    let mut out = vec![0i128; n];
    for (t, &c) in coeffs.iter().enumerate() {
        for (i, &xi) in x.iter().enumerate() {
            let j = i + t * stride;
            if j < n {
                out[j] += c * xi;
            } else {
                out[j - n] -= c * xi;
            }
        }
    }
    out
}

fn centered(x: i128, bits: usize) -> i128 {
    let modulus = 1i128 << bits;
    let r = x.rem_euclid(modulus);
    if r >= modulus / 2 { r - modulus } else { r }
}

pub fn test_round_unit_shaping_is_plain<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
{
    let src = encrypted::<BE, F, E>(&params, module, host_module);
    let dst_k = src.k().as_usize() - src.log_delta() / 2;
    let unit = SlotLayout::Replicated(ReplicatedLayout::new(0, 1, &[1, 0, 0, 0])?);
    let plain = round(module, &src, &params, dst_k)?;
    let shaped = round(module, &with_layout::<BE>(&src, unit), &params, dst_k)?;
    assert_eq!(digits::<BE>(&plain), digits::<BE>(&shaped));
    assert_eq!(plain.log_delta(), src.log_delta() - (src.k().as_usize() - dst_k));
    assert_eq!(plain.log_budget(), src.log_budget());
    assert_eq!(shaped.slot_layout(), unit);
    Ok(())
}

pub fn test_round_error_is_shaped<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + VecZnxNormalize<BE> + VecZnxNormalizeTmpBytes,
    F: TestScalar,
    E: ReimNew<F>,
{
    let layout = ShapingUnit::U32_45.layout(0)?;
    let src = with_layout::<BE>(
        &encrypted::<BE, F, E>(&params, module, host_module),
        SlotLayout::Replicated(layout),
    );
    let lambda = 30.min(src.log_delta());
    let k = src.k().as_usize() - lambda;
    let dst = round(module, &src, &params, k)?;
    assert_eq!(dst.k().as_usize(), k);
    assert_eq!(dst.log_budget(), src.log_budget());
    assert_eq!(dst.slot_layout(), SlotLayout::Replicated(layout));

    // `(dst − src)·2^(k+λ) = −g·h` exactly: read both 8 bits above `k`, where the
    // difference cannot wrap.
    let profile = ShapingProfile::new(&layout);
    let g: Vec<i128> = layout.shaping().iter().map(|&c| c as i128).collect();
    let inverse: Vec<i128> = profile.inverse().iter().map(|&c| c as i64 as i128).collect();
    let stride = params.n >> (layout.log_r() + 1);
    let bits = lambda + 8;
    let mut nonzero = false;
    for col in 0..src.rank().as_usize() + 1 {
        let before = window(module, &src, col, k as i64 - 8, bits);
        let after = window(module, &dst, col, k as i64 - 8, bits);
        let diff: Vec<i128> = after
            .iter()
            .zip(&before)
            .map(|(&a, &b)| centered(a as i128 - b as i128, bits))
            .collect();
        let h: Vec<i128> = sparse_mul(&inverse, stride, &diff)
            .into_iter()
            .map(|x| centered(x, 64))
            .collect();
        assert!(h.iter().all(|x| x.abs() <= 1 << (lambda - 1)), "col {col}");
        assert_eq!(sparse_mul(&g, stride, &h), diff, "col {col}");
        nonzero |= diff.iter().any(|&x| x != 0);
    }
    assert!(nonzero);
    Ok(())
}

/// [`test_round_error_is_shaped`] on the conjugate-invariant ring: on the
/// coefficients unfolded onto `Z[X]/(X^(2N)+1)`, the change is `-g*h` for the
/// invariant shift of `U`, with `g = (Y^s*U)(X^(N/R))`.
pub fn test_round_error_is_shaped_conjugate_invariant<BE>(
    mut params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: TestContextSharedModule<BE, ConjugateInvariant>
        + CKKSEncodingHostOps<BE, f64, ConjugateInvariant>
        + VecZnxNormalize<BE, ConjugateInvariant>
        + VecZnxNormalizeTmpBytes,
{
    params.prec_meta.slots = SlotsKind::Real;
    let (_, sk) = gen_sk_with_raw::<BE, ConjugateInvariant>(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch::<BE, ConjugateInvariant>(&params, module);
    let n = params.n;
    let values: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / (n as f64 + 3.0)).collect();
    let mut pt = module.ckks_plaintext_alloc_from_slots::<ConjugateInvariant>(n, params.base2k.into(), params.prec().k());
    pt.set_meta(params.prec().meta());
    module.ckks_encode_slots_host_into(&mut pt, &values, &vec![0.0; n], &mut scratch.borrow())?;
    let mut src = ckks_encrypt_pt(
        &params,
        module,
        &sk,
        params.k,
        &pt.to_host_owned::<BE>(),
        &mut scratch.borrow(),
    );
    let layout = ShapingUnit::U32_45.layout(0)?;
    src.set_slot_layout(SlotLayout::Replicated(layout));
    let lambda = 30.min(src.log_delta());
    let k = src.k().as_usize() - lambda;
    let dst = round(module, &src, &params, k)?;
    assert_eq!(dst.k().as_usize(), k);
    assert_eq!(dst.log_budget(), src.log_budget());
    assert_eq!(dst.slot_layout(), SlotLayout::Replicated(layout));

    let profile = ShapingProfile::<ConjugateInvariant>::try_new(&layout)?;
    assert_ne!(profile.shift(), 0);
    let g: Vec<i128> = profile.coefficients().iter().map(|&c| c as i128).collect();
    let inverse: Vec<i128> = profile.inverse().iter().map(|&c| c as i64 as i128).collect();
    let unfold = |x: &[i128]| -> Vec<i128> {
        let mut ambient = vec![0i128; 2 * n];
        ambient[..n].copy_from_slice(x);
        for j in 1..n {
            ambient[2 * n - j] = -x[j];
        }
        ambient
    };
    let stride = n >> layout.log_r();
    let bits = lambda + 8;
    let mut nonzero = false;
    for col in 0..src.rank().as_usize() + 1 {
        let before = window(module, &src, col, k as i64 - 8, bits);
        let after = window(module, &dst, col, k as i64 - 8, bits);
        let diff: Vec<i128> = after
            .iter()
            .zip(&before)
            .map(|(&a, &b)| centered(a as i128 - b as i128, bits))
            .collect();
        let diff = unfold(&diff);
        let h: Vec<i128> = sparse_mul(&inverse, stride, &diff)
            .into_iter()
            .map(|x| centered(x, 64))
            .collect();
        assert!(h.iter().all(|x| x.abs() <= 1 << (lambda - 1)), "col {col}");
        assert_eq!(sparse_mul(&g, stride, &h), diff, "col {col}");
        nonzero |= diff.iter().any(|&x| x != 0);
    }
    assert!(nonzero);
    Ok(())
}

pub fn test_round_beyond_pass_rounds_plainly_first<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
{
    let layout = SlotLayout::Replicated(ShapingUnit::U32_45.layout(0)?);
    let pass = 58;
    let mut src = encrypted::<BE, F, E>(&params, module, host_module);
    // Relabel the whole width as precision, so the drop may exceed the pass.
    src.set_scale(Scale::Log(src.k().as_usize()));
    let k = src.k().as_usize() - pass - 6;
    let direct = round(module, &with_layout::<BE>(&src, layout), &params, k)?;
    let plain = round(module, &src, &params, k + pass)?;
    let staged = round(module, &with_layout::<BE>(&plain, layout), &params, k)?;
    assert_eq!(digits::<BE>(&direct), digits::<BE>(&staged));
    Ok(())
}

pub fn test_round_rejects_invalid_widths<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
{
    use crate::test_suite::parity::helpers::{snapshot, with_scratch};
    let src = encrypted::<BE, F, E>(&params, module, host_module);
    let src_before = snapshot::<BE, _, _>(&src);
    let k = src.k().as_usize();
    let drop = src.log_delta() + 1;
    for (dst_k, reason) in [
        (k + 1, format!("res.k()={} exceeds a.k()={k}", k + 1)),
        (
            k - drop,
            format!("dropping {drop} bits exceeds log_delta={}", src.log_delta()),
        ),
    ] {
        let mut dst = alloc_ct(&params, module, dst_k);
        let before = snapshot::<BE, _, _>(&dst);
        let err = with_scratch::<BE, _>(module.ckks_round_tmp_bytes(&dst, &src), |scratch| {
            module.ckks_round(&mut dst, &src, scratch)
        })
        .unwrap_err();
        assert!(
            matches!(err.composition(), Some(crate::CKKSCompositionError::InvalidOperand { op: "ckks_round", reason: got }) if *got == reason),
            "{err}"
        );
        assert_eq!(snapshot::<BE, _, _>(&dst), before);
        assert_eq!(snapshot::<BE, _, _>(&src), src_before);
    }
    Ok(())
}

/// Rank and degree mismatches refuse before changing either operand.
pub fn test_round_rejects_incompatible_layouts<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + poulpy_core::GLWEFill<BE>,
    F: TestScalar,
    E: ReimNew<F>,
{
    use crate::test_suite::parity::helpers::{fixture_ciphertext, snapshot, with_scratch};
    use crate::{CKKSCompositionError, CKKSLayout, GLWEPlaintextMeta};
    use poulpy_core::layouts::GLWELayout;
    for slot_layout in [
        SlotLayout::Full,
        SlotLayout::Replicated(ReplicatedLayout::new(0, 1, &[1, 0, 0, 0])?),
    ] {
        let src_layout = CKKSLayout {
            glwe_layout: GLWELayout {
                n: params.n.into(),
                base2k: params.base2k.into(),
                k: 72usize.into(),
                rank: 1usize.into(),
            },
            meta: GLWEPlaintextMeta {
                scale: Scale::Log(40),
                slot_layout,
                ..Default::default()
            },
        };
        let src = fixture_ciphertext(module, &src_layout, 81);
        let src_before = snapshot::<BE, _, _>(&src);
        for (rank, n, reason) in [
            (0usize, params.n, "source and destination ranks differ"),
            (2, params.n, "source and destination ranks differ"),
            (1, params.n / 2, "source and destination degrees differ"),
        ] {
            let dst_layout = CKKSLayout {
                glwe_layout: GLWELayout {
                    rank: rank.into(),
                    n: n.into(),
                    k: 60usize.into(),
                    ..src_layout.glwe_layout
                },
                meta: GLWEPlaintextMeta {
                    scale: Scale::Log(7),
                    ..Default::default()
                },
            };
            let mut dst = fixture_ciphertext(module, &dst_layout, 83);
            let before = snapshot::<BE, _, _>(&dst);
            let err = with_scratch::<BE, _>(module.ckks_round_tmp_bytes(&dst, &src), |scratch| {
                module.ckks_round(&mut dst, &src, scratch)
            })
            .unwrap_err();
            assert!(
                matches!(err.composition(), Some(CKKSCompositionError::InvalidOperand {op: "ckks_round", reason: got}) if got == reason)
            );
            assert_eq!(snapshot::<BE, _, _>(&dst), before);
            assert_eq!(snapshot::<BE, _, _>(&src), src_before);
        }
        let mut dst = fixture_ciphertext(
            module,
            &CKKSLayout {
                glwe_layout: GLWELayout {
                    k: 60usize.into(),
                    ..src_layout.glwe_layout
                },
                meta: GLWEPlaintextMeta {
                    scale: Scale::Log(7),
                    ..Default::default()
                },
            },
            83,
        );
        let before = snapshot::<BE, _, _>(&dst);
        // One byte short of the query.
        let short = module.ckks_round_tmp_bytes(&dst, &src) - 1;
        let err = with_scratch::<BE, _>(short, |scratch| module.ckks_round(&mut dst, &src, scratch)).unwrap_err();
        assert!(matches!(&err, crate::CKKSError::Internal(_)));
        assert_eq!(err.to_string(), "ckks_round: insufficient scratch");
        assert_eq!(snapshot::<BE, _, _>(&dst), before);
        assert_eq!(snapshot::<BE, _, _>(&src), src_before);
    }
    Ok(())
}
