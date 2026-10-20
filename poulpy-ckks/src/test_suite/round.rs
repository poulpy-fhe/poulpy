//! Rounding tests: plain and shaped precision reduction.
//!
//! # Test inventory
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_round_unit_shaping_is_plain`] | a replicated layout with `U = 1` rounds as a full layout |
//! | [`test_round_error_is_shaped`] | the rounding error is `−g·h·2^-(k+λ)` with `\|\|h\|\|_∞ <= 2^(λ−1)` |
//! | [`test_round_beyond_pass_rounds_plainly_first`] | a drop beyond the pass size rounds plainly to `k + λ`, then shaped |
//! | [`test_round_rejects_invalid_widths`] | a wider destination and a drop beyond `log_delta` |

use anyhow::Result;
use poulpy_core::layouts::{GLWEInfos, GLWEToBackendRef, LWEInfos};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalize, VecZnxNormalizeTmpBytes},
    layouts::{HostBytesBackend, Module, ScratchOwned, Standard, VecZnxToBackendMut, VecZnxToBackendRef, ZnxView},
};

use super::helpers::{
    TestContextBackend, TestContextModule, TestScalar, alloc_ct, alloc_scratch, ckks_encrypt, gen_sk, test_vector_1,
};
use crate::{
    CKKSInfos, GLWEPlaintextInfos, ReplicatedLayout, Scale, SetGLWEPlaintextInfos, SlotLayout,
    api::CKKSRoundOps,
    layouts::CKKSCiphertextOwned,
    shaping::{ShapingProfile, ShapingUnit},
    test_suite::{CKKSTestParams, reference_encoder::ReferenceEncoder},
};

fn encrypted<BE, F, E>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> CKKSCiphertextOwned<BE>
where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
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

fn round<BE>(
    module: &Module<BE>,
    src: &CKKSCiphertextOwned<BE>,
    params: &CKKSTestParams,
    k: usize,
) -> Result<CKKSCiphertextOwned<BE>>
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
{
    let mut dst = alloc_ct(params, module, k);
    let mut scratch = ScratchOwned::<BE>::alloc(module.ckks_round_tmp_bytes(&dst, src));
    module.ckks_round_into(&mut dst, src, &mut scratch.borrow())?;
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
fn window<BE>(module: &Module<BE>, ct: &CKKSCiphertextOwned<BE>, col: usize, offset: i64, bits: usize) -> Vec<i64>
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    Module<BE>: TestContextModule<BE> + VecZnxNormalize<BE> + VecZnxNormalizeTmpBytes,
{
    let mut out = module.vec_znx_alloc(ct.n().as_usize(), 1, 1);
    let mut scratch = ScratchOwned::<BE>::alloc(module.vec_znx_normalize_tmp_bytes());
    let glwe = GLWEToBackendRef::<BE>::to_backend_ref(ct);
    module.vec_znx_normalize(
        &mut VecZnxToBackendMut::<BE>::to_backend_mut(&mut out),
        bits,
        bits,
        offset,
        0,
        glwe.data(),
        ct.base2k().as_usize(),
        col,
        &mut scratch.borrow(),
    );
    VecZnxToBackendRef::<BE>::to_backend_ref(&out).at(0, 0).to_vec()
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
    BE: TestContextBackend<Ring = Standard>,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let src = encrypted::<BE, F, E>(&params, module, host_module);
    let dst_k = src.k().as_usize() - src.log_delta() / 2;
    let unit = SlotLayout::Replicated(ReplicatedLayout::new(0, 1, 8, &[1, 0, 0, 0])?);
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
    BE: TestContextBackend<Ring = Standard>,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + VecZnxNormalize<BE> + VecZnxNormalizeTmpBytes,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let layout = ShapingUnit::U32_45.layout(0, 8)?;
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

pub fn test_round_beyond_pass_rounds_plainly_first<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let layout = SlotLayout::Replicated(ShapingUnit::U32_45.layout(0, 8)?);
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
    BE: TestContextBackend<Ring = Standard>,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let src = encrypted::<BE, F, E>(&params, module, host_module);
    let k = src.k().as_usize();
    assert!(round(module, &src, &params, k + 1).is_err());
    assert!(round(module, &src, &params, k - src.log_delta() - 1).is_err());
    Ok(())
}
