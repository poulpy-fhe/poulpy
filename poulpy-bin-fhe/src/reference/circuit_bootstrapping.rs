#![allow(clippy::too_many_arguments)]
//! Canonical circuit bootstrap algorithms over the public core and scheme operations.
//! Plan preparation constructs its LUT on the host, then uploads it through the LUT factory.
use crate::{
    blind_rotation::{
        BlindRotationAlgo, BlindRotationExecute, BlindRotationKeyEncryptSk, BlindRotationKeyInfos, BlindRotationKeyLayout,
        BlindRotationKeyPrepared, BlindRotationKeyPreparedFactory, LookUpTableLayout, LookUpTableRotationDirection, LookupTable,
        LookupTableFactory,
    },
    circuit_bootstrapping::{
        CircuitBootstrappingEncryptionInfos, CircuitBootstrappingKey, CircuitBootstrappingKeyInfos,
        CircuitBootstrappingKeyPrepared, CircuitBootstrappingPlan, CircuitBootstrappingPlanLayout,
        circuit::{CircuitBootstrappingExecutionConfig, CircuitBootstrappingOutput},
        trace_galois_elements,
    },
};
use itertools::Itertools;
use poulpy_core::{
    Distribution, GGLWEToGGSWKeyEncryptSk, GGSWExpandRows, GLWEAutomorphismKeyEncryptSk, GLWEBytesOf, GLWECopy, GLWEPacking,
    GLWERotate, GLWETrace, GetDistribution, ScratchArenaTakeCore,
    layouts::{
        GGLWELayout, GGLWEToGGSWKeyLayout, GGLWEToGGSWKeyPreparedFactory, GGSWAtViewMut, GGSWAtViewRef, GGSWInfos,
        GGSWToBackendMut, GLWEAutomorphismKeyLayout, GLWEAutomorphismKeyPreparedFactory, GLWEInfos, GLWELayout,
        GLWESecretPreparedFactory, GLWESecretToBackendRef, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LWEInfos,
        LWESecretToBackendRef, LWEToBackendRef, ModuleCoreAlloc, SetGaloisElement, prepared::GGLWEToGGSWKeyPreparedToBackendRef,
    },
};
use poulpy_hal::{
    api::{ModuleLogN, ModuleN},
    layouts::{Backend, ScratchArena},
    source::Source,
};
use std::collections::HashMap;
pub fn circuit_bootstrapping_execute_tmp_bytes_reference<R, A, M, BRA, BE>(
    module: &M,
    block_size: usize,
    extension_factor: usize,
    res_infos: &R,
    cbt_infos: &A,
) -> usize
where
    R: GGSWInfos,
    A: CircuitBootstrappingKeyInfos,
    BRA: BlindRotationAlgo,
    BE: Backend<ZnxWord = i64>,
    M: GLWEBytesOf<BE>
        + BlindRotationExecute<BRA, BE>
        + GLWETrace<BE>
        + GLWEPacking<BE>
        + GGSWExpandRows<BE>
        + GLWERotate<BE>
        + GLWECopy<BE>,
{
    circuit_bootstrapping_prepared_tmp_bytes(
        module,
        res_infos,
        CircuitBootstrappingExecutionConfig {
            output: CircuitBootstrappingOutput::Constant,
            log_domain: 0,
            log_gap_in: None,
            extension_factor,
        },
        block_size,
        cbt_infos,
    )
}
pub fn circuit_bootstrapping_execute_to_exponent_tmp_bytes_reference<R, A, M, BRA, BE>(
    module: &M,
    log_gap_out: usize,
    log_domain: usize,
    block_size: usize,
    extension_factor: usize,
    res_infos: &R,
    cbt_infos: &A,
) -> usize
where
    R: GGSWInfos,
    A: CircuitBootstrappingKeyInfos,
    BRA: BlindRotationAlgo,
    BE: Backend<ZnxWord = i64>,
    M: GLWEBytesOf<BE>
        + BlindRotationExecute<BRA, BE>
        + GLWETrace<BE>
        + GLWEPacking<BE>
        + GGSWExpandRows<BE>
        + GLWERotate<BE>
        + GLWECopy<BE>,
{
    circuit_bootstrapping_prepared_tmp_bytes(
        module,
        res_infos,
        CircuitBootstrappingExecutionConfig {
            output: CircuitBootstrappingOutput::Exponent { log_gap_out },
            log_domain,
            log_gap_in: Some(circuit_bootstrapping_log_gap_in(res_infos, log_domain, extension_factor)),
            extension_factor,
        },
        block_size,
        cbt_infos,
    )
}
pub fn circuit_bootstrapping_key_encrypt_sk_tmp_bytes_reference<A, M, BRA, BE>(module: &M, infos: &A) -> usize
where
    A: CircuitBootstrappingKeyInfos,
    BRA: BlindRotationAlgo,
    BE: Backend,
    M: GGLWEToGGSWKeyEncryptSk<BE>
        + BlindRotationKeyEncryptSk<BRA, BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + ModuleN,
{
    module
        .glwe_automorphism_key_encrypt_sk_tmp_bytes(&infos.atk_infos())
        .max(
            module
                .glwe_secret_prepared_bytes_of(infos.brk_infos().rank())
                .next_multiple_of(BE::SCRATCH_ALIGN)
                + module.blind_rotation_key_encrypt_sk_tmp_bytes(&infos.brk_infos()),
        )
        .max(module.gglwe_to_ggsw_key_encrypt_sk_tmp_bytes(&infos.tsk_infos()))
}
pub fn circuit_bootstrapping_key_encrypt_sk_reference<S0, S1, M, BRA, BE>(
    module: &M,
    res: &mut CircuitBootstrappingKey<BE::OwnedBuf, BRA, BE::ZnxWord>,
    sk_lwe: &S0,
    sk_glwe: &S1,
    enc_infos: &CircuitBootstrappingEncryptionInfos,
    source_xe: &mut Source,
    source_xa: &mut Source,
    scratch: &mut ScratchArena<'_, BE>,
) where
    S0: LWESecretToBackendRef<BE> + GetDistribution + LWEInfos,
    S1: GLWESecretToBackendRef<BE> + GLWEInfos + GetDistribution,
    BRA: BlindRotationAlgo,
    BE: Backend,
    M: GGLWEToGGSWKeyEncryptSk<BE>
        + BlindRotationKeyEncryptSk<BRA, BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + ModuleN,
{
    let brk_infos: &BlindRotationKeyLayout = &res.brk_infos();
    let atk_infos: &GLWEAutomorphismKeyLayout = &res.atk_infos();
    let tsk_infos: &GGLWEToGGSWKeyLayout = &res.tsk_infos();

    assert_eq!(sk_lwe.n(), brk_infos.n_lwe());
    assert_eq!(sk_glwe.n(), brk_infos.n_glwe());
    assert_eq!(sk_glwe.n(), atk_infos.n());
    assert_eq!(sk_glwe.n(), tsk_infos.n());

    assert!(sk_glwe.dist() != &Distribution::NONE);

    let gal_els: Vec<i64> = res.atk.keys().sorted().copied().collect();
    for p in gal_els {
        let atk = res.atk.get_mut(&p).unwrap();
        module.glwe_automorphism_key_encrypt_sk(atk, p, sk_glwe, &enc_infos.atk, source_xe, source_xa, scratch);
    }

    {
        let (mut sk_glwe_prepared, mut op_scratch) = scratch.borrow().take_glwe_secret_prepared_scratch(module, brk_infos.rank());
        module.glwe_secret_prepare(&mut sk_glwe_prepared, sk_glwe);

        module.blind_rotation_key_encrypt_sk(
            &mut res.brk,
            &sk_glwe_prepared,
            sk_lwe,
            &enc_infos.brk,
            source_xe,
            source_xa,
            &mut op_scratch,
        );
    }
    module.gglwe_to_ggsw_key_encrypt_sk(&mut res.tsk, sk_glwe, &enc_infos.tsk, source_xe, source_xa, scratch);
}
pub fn circuit_bootstrapping_key_prepared_alloc_from_infos_reference<A, M, BRA, BE>(
    module: &M,
    infos: &A,
) -> CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE>
where
    A: CircuitBootstrappingKeyInfos,
    BRA: BlindRotationAlgo,
    BE: Backend,
    M: BlindRotationKeyPreparedFactory<BRA, BE> + GGLWEToGGSWKeyPreparedFactory<BE> + GLWEAutomorphismKeyPreparedFactory<BE>,
{
    let atk_infos: &GLWEAutomorphismKeyLayout = &infos.atk_infos();
    let gal_els: Vec<i64> = trace_galois_elements(atk_infos.log_n(), 2 * atk_infos.n().as_usize() as i64);

    CircuitBootstrappingKeyPrepared {
        brk: BlindRotationKeyPrepared::alloc(module, &infos.brk_infos()),
        tsk: module.gglwe_to_ggsw_key_prepared_alloc_from_infos(&infos.tsk_infos()),
        atk: gal_els
            .iter()
            .map(|&gal_el| {
                let mut key = module.glwe_automorphism_key_prepared_alloc_from_infos(atk_infos);
                key.set_p(gal_el);
                (gal_el, key)
            })
            .collect(),
    }
}
pub fn circuit_bootstrapping_key_prepare_tmp_bytes_reference<A, M, BRA, BE>(module: &M, infos: &A) -> usize
where
    A: CircuitBootstrappingKeyInfos,
    BRA: BlindRotationAlgo,
    BE: Backend,
    M: BlindRotationKeyPreparedFactory<BRA, BE> + GGLWEToGGSWKeyPreparedFactory<BE> + GLWEAutomorphismKeyPreparedFactory<BE>,
{
    module
        .blind_rotation_key_prepare_tmp_bytes(&infos.brk_infos())
        .max(module.gglwe_to_ggsw_key_prepare_tmp_bytes(&infos.tsk_infos()))
        .max(module.glwe_automorphism_key_prepare_tmp_bytes(&infos.atk_infos()))
}
pub fn circuit_bootstrapping_key_prepare_reference<M, BRA, BE>(
    module: &M,
    res: &mut CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE>,
    other: &CircuitBootstrappingKey<BE::OwnedBuf, BRA, BE::ZnxWord>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BRA: BlindRotationAlgo,
    BE: Backend,
    M: BlindRotationKeyPreparedFactory<BRA, BE> + GGLWEToGGSWKeyPreparedFactory<BE> + GLWEAutomorphismKeyPreparedFactory<BE>,
{
    assert_eq!(res.brk_infos(), other.brk_infos(), "circuit-bootstrap BRK layout mismatch");
    assert_eq!(res.atk_infos(), other.atk_infos(), "circuit-bootstrap ATK layout mismatch");
    assert_eq!(res.tsk_infos(), other.tsk_infos(), "circuit-bootstrap TSK layout mismatch");
    assert_eq!(
        res.atk.len(),
        other.atk.len(),
        "circuit-bootstrap automorphism key count mismatch"
    );
    assert!(
        res.atk.keys().all(|p| other.atk.contains_key(p)),
        "circuit-bootstrap automorphism keys mismatch"
    );
    res.brk.prepare(module, &other.brk, scratch);
    module.gglwe_to_ggsw_key_prepare(&mut res.tsk, &other.tsk, scratch);

    let gal_els: Vec<i64> = res.atk.keys().sorted().copied().collect();
    for k in gal_els {
        module.glwe_automorphism_key_prepare(
            res.atk.get_mut(&k).unwrap(),
            other
                .atk
                .get(&k)
                .unwrap_or_else(|| panic!("Galois element {k} is present in the prepared key but missing from the source key")),
            scratch,
        );
    }
}
fn circuit_bootstrapping_prepared_tmp_bytes<M, R, A, BRA, BE>(
    module: &M,
    res_infos: &R,
    config: CircuitBootstrappingExecutionConfig,
    block_size: usize,
    cbt_infos: &A,
) -> usize
where
    BRA: BlindRotationAlgo,
    BE: Backend<ZnxWord = i64>,
    R: GGSWInfos,
    A: CircuitBootstrappingKeyInfos,
    M: GLWEBytesOf<BE>
        + BlindRotationExecute<BRA, BE>
        + GLWETrace<BE>
        + GLWEPacking<BE>
        + GGSWExpandRows<BE>
        + GLWERotate<BE>
        + GLWECopy<BE>,
{
    let brk_infos = cbt_infos.brk_infos();
    let atk_infos = cbt_infos.atk_infos();
    let tsk_infos = cbt_infos.tsk_infos();
    let glwe_brk_layout = GLWELayout {
        n: brk_infos.n_glwe,
        base2k: brk_infos.base2k,
        k: brk_infos.k(),
        rank: brk_infos.rank,
    };
    let glwe_atk_layout = GLWELayout {
        n: glwe_brk_layout.n,
        base2k: atk_infos.base2k,
        k: glwe_brk_layout.k,
        rank: glwe_brk_layout.rank,
    };
    let res_glwe_layout = res_infos.glwe_layout();

    let aligned = |bytes: usize| bytes.next_multiple_of(BE::SCRATCH_ALIGN);
    let atk_bytes = aligned(module.glwe_bytes_of_from_infos(&glwe_atk_layout));
    let brk_bytes = aligned(module.glwe_bytes_of_from_infos(&glwe_brk_layout));
    let blind_rotation =
        module.blind_rotation_execute_tmp_bytes(block_size, config.extension_factor, &glwe_brk_layout, &brk_infos);
    let convert = module.glwe_copy_tmp_bytes(&glwe_atk_layout, &glwe_brk_layout);
    let blind_phase = brk_bytes + blind_rotation.max(convert);

    let atk_key_infos: GGLWELayout = GGLWELayout {
        n: atk_infos.n,
        base2k: atk_infos.base2k,
        dnum: atk_infos.dnum,
        k_aux: atk_infos.k_aux,
        dsize: atk_infos.dsize,
        rank_in: atk_infos.rank,
        rank_out: atk_infos.rank,
        stride: 1,
    };
    let trace_atk = module.glwe_trace_tmp_bytes(&glwe_atk_layout, &glwe_atk_layout, &atk_key_infos);
    let trace_res = module.glwe_trace_tmp_bytes(&res_glwe_layout, &glwe_atk_layout, &atk_key_infos);
    let copy_res = module.glwe_copy_tmp_bytes(&res_glwe_layout, &res_glwe_layout);
    let copy_atk = module.glwe_copy_tmp_bytes(&glwe_atk_layout, &glwe_atk_layout);
    let rotate = module.glwe_rotate_tmp_bytes();
    let row_phase = match config.output {
        CircuitBootstrappingOutput::Constant => {
            aligned(module.glwe_bytes_of_from_infos(&res_glwe_layout)) + trace_res.max(copy_res)
        }
        CircuitBootstrappingOutput::Exponent { log_gap_out } => {
            if config.log_gap_in.expect("prepared exponent execution requires its input gap") == log_gap_out {
                aligned(module.glwe_bytes_of_from_infos(&res_glwe_layout)) + trace_res.max(copy_res)
            } else {
                let steps = 1usize
                    .checked_shl(config.log_domain as u32)
                    .expect("circuit-bootstrap domain overflows usize");
                let owned = aligned(module.glwe_bytes_of_from_infos(&glwe_atk_layout))
                    + aligned(module.glwe_bytes_of_from_infos(&res_glwe_layout))
                    + steps * aligned(module.glwe_bytes_of_from_infos(&glwe_atk_layout));
                owned
                    + trace_atk
                        .max(rotate)
                        .max(module.glwe_pack_tmp_bytes(&res_glwe_layout, &glwe_atk_layout, &atk_key_infos))
                        .max(copy_atk)
                        .max(copy_res)
            }
        }
    };
    let online = atk_bytes + blind_phase.max(row_phase).max(rotate);
    online.max(module.ggsw_expand_rows_tmp_bytes(res_infos, &tsk_infos))
}

pub(crate) fn circuit_bootstrapping_log_gap_in<R: GGSWInfos>(res_infos: &R, log_domain: usize, extension_factor: usize) -> usize {
    assert!(
        extension_factor.is_power_of_two(),
        "extension_factor must be a non-zero power of two"
    );
    let dnum = res_infos.dnum().as_usize();
    assert!(dnum > 0, "circuit-bootstrap output must have at least one decomposition row");
    let alpha = dnum.next_power_of_two();
    let domain = 1usize
        .checked_shl(log_domain as u32)
        .expect("circuit-bootstrap domain overflows usize");
    let f_len = domain
        .checked_mul(alpha)
        .expect("circuit-bootstrap LUT length overflows usize");
    assert!(
        f_len <= res_infos.n().as_usize(),
        "circuit-bootstrap LUT length exceeds the polynomial degree"
    );
    let lut_domain = res_infos
        .n()
        .as_usize()
        .checked_mul(extension_factor)
        .expect("circuit-bootstrap LUT domain overflows usize");
    let step = lut_domain
        .checked_add(f_len >> 1)
        .expect("circuit-bootstrap LUT rounding overflows usize")
        / f_len;
    let gap = (step >> 1).checked_mul(2).expect("circuit-bootstrap LUT gap overflows usize") / extension_factor;
    assert!(
        gap > 0,
        "circuit-bootstrap LUT domain exceeds the available polynomial domain"
    );
    let spread = gap.checked_mul(alpha).expect("circuit-bootstrap LUT spread overflows usize");
    (usize::BITS - (spread - 1).leading_zeros()) as usize
}

fn prepare_circuit_bootstrapping_plan<R, M, BRA, BE>(
    module: &M,
    res_infos: &R,
    key: &CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE>,
    log_domain: usize,
    extension_factor: usize,
    output: CircuitBootstrappingOutput,
) -> CircuitBootstrappingPlan<BE::OwnedBuf>
where
    BRA: BlindRotationAlgo,
    BE: Backend<ZnxWord = i64>,
    R: GGSWInfos,
    M: ModuleN + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64> + LookupTableFactory<BE::OwnedBuf, i64>,
{
    assert!(
        extension_factor.is_power_of_two(),
        "extension_factor must be a non-zero power of two"
    );
    assert_eq!(res_infos.n(), key.brk.n());
    assert_eq!(res_infos.n().as_usize(), module.n());
    assert_eq!(res_infos.rank(), key.brk.rank());
    assert_eq!(res_infos.rank(), key.atk_infos().rank);
    assert_eq!(res_infos.rank(), key.tsk.rank());
    assert_eq!(res_infos.n(), key.atk_infos().n);
    assert_eq!(res_infos.n(), key.tsk.n());
    let res_base2k = res_infos.base2k().as_usize();
    let dnum_res = res_infos.dnum().as_usize();
    assert!(
        dnum_res > 0,
        "circuit-bootstrap output must have at least one decomposition row"
    );
    let alpha = dnum_res.next_power_of_two();
    let to_exponent = matches!(output, CircuitBootstrappingOutput::Exponent { .. });

    validate_lut_coefficients(to_exponent, res_base2k, dnum_res, log_domain);

    let domain = 1usize
        .checked_shl(log_domain as u32)
        .expect("circuit-bootstrap domain overflows usize");
    let f_len = domain
        .checked_mul(alpha)
        .expect("circuit-bootstrap LUT length overflows usize");
    assert!(
        f_len <= module.n(),
        "circuit-bootstrap LUT length exceeds the polynomial degree"
    );
    let mut f = vec![0i64; f_len];
    let lut_precision = res_base2k
        .checked_mul(dnum_res)
        .expect("circuit-bootstrap LUT precision overflows usize");

    if to_exponent {
        (0..dnum_res).for_each(|i| {
            f[i] = 1 << (res_base2k * (dnum_res - 1 - i));
        });
    } else {
        (0..domain).for_each(|j| {
            (0..dnum_res).for_each(|i| {
                f[j * alpha + i] = j as i64 * (1 << (res_base2k * (dnum_res - 1 - i)));
            });
        });
    }

    let lut_infos: LookUpTableLayout = LookUpTableLayout {
        n: module.n().into(),
        extension_factor,
        k: lut_precision.into(),
        base2k: key.brk.base2k(),
    };

    let mut lut: LookupTable<BE::OwnedBuf, BE::ZnxWord> = LookupTable::alloc(module, &lut_infos);
    lut.set(module, &f, lut_precision);

    if to_exponent {
        lut.set_rotation_direction(LookUpTableRotationDirection::Right);
    }

    let gap = 2 * lut.drift / lut.extension_factor();
    assert!(
        gap > 0,
        "circuit-bootstrap LUT domain exceeds the available polynomial domain"
    );
    let log_gap_in = (usize::BITS - (gap * alpha - 1).leading_zeros()) as usize;
    debug_assert_eq!(
        log_gap_in,
        circuit_bootstrapping_log_gap_in(res_infos, log_domain, extension_factor)
    );
    CircuitBootstrappingPlan {
        lut,
        output_layout: res_infos.ggsw_layout(),
        output,
        log_domain,
        log_gap_in,
        extension_factor,
        key_layout: crate::circuit_bootstrapping::CircuitBootstrappingKeyLayout {
            brk_layout: key.brk_infos(),
            atk_layout: key.atk_infos(),
            tsk_layout: key.tsk_infos(),
        },
        block_size: key.block_size(),
    }
}

fn validate_lut_coefficients(to_exponent: bool, res_base2k: usize, dnum_res: usize, log_domain: usize) {
    let coefficient_exponent = res_base2k
        .checked_mul(dnum_res.saturating_sub(1))
        .expect("LUT coefficient exponent overflows usize");
    assert!(
        dnum_res == 0 || coefficient_exponent < i64::BITS as usize,
        "LUT coefficient overflow: res_base2k={res_base2k} * (dnum_res-1)={} >= {} bits",
        dnum_res.saturating_sub(1),
        i64::BITS,
    );
    let scaled_exponent = log_domain
        .checked_add(coefficient_exponent)
        .expect("LUT scaled coefficient exponent overflows usize");
    assert!(
        to_exponent || scaled_exponent < i64::BITS as usize,
        "LUT coefficient overflow: log_domain={log_domain} + res_base2k*(dnum_res-1) would exceed i64"
    );
}

fn circuit_bootstrap_prepared<R, L, M, BRA, BE>(
    module: &M,
    res: &mut R,
    lwe: &L,
    key: &CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE>,
    plan: &CircuitBootstrappingPlan<BE::OwnedBuf>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BRA: BlindRotationAlgo,
    BE: Backend<ZnxWord = i64>,
    R: GGSWToBackendMut<BE> + GGSWAtViewRef<BE> + GGSWAtViewMut<BE> + GGSWInfos,
    L: LWEToBackendRef<BE> + LWEInfos,
    M: ModuleLogN
        + GLWEBytesOf<BE>
        + BlindRotationExecute<BRA, BE>
        + GLWETrace<BE>
        + GLWEPacking<BE>
        + GGSWExpandRows<BE>
        + GLWERotate<BE>
        + GLWECopy<BE>,
{
    let dnum_res = res.dnum().as_usize();
    let glwe_brk_layout = &GLWELayout {
        n: key.brk.n(),
        base2k: key.brk.base2k(),
        k: key.brk.k(),
        rank: key.brk.rank(),
    };

    // Every rotation's key shares the radix; read it off the first one.
    let atk_base2k = key
        .atk
        .get_automorphism_key(-1, glwe_brk_layout.k())
        .map(|layout| layout.base2k())
        .unwrap_or_else(|e| panic!("{e}"));
    let glwe_atk_layout: &GLWELayout = &GLWELayout {
        n: glwe_brk_layout.n(),
        base2k: atk_base2k,
        k: glwe_brk_layout.k(),
        rank: glwe_brk_layout.rank(),
    };

    {
        let (mut res_glwe_atk_layout, mut scratch_1) = scratch.borrow().take_glwe_scratch(glwe_atk_layout);

        {
            let (mut res_glwe_brk_layout, mut op_scratch) = scratch_1.borrow().take_glwe_scratch(glwe_brk_layout);
            key.brk
                .execute(module, &mut res_glwe_brk_layout, lwe, &plan.lut, &mut op_scratch.borrow());

            module.glwe_copy(&mut res_glwe_atk_layout, &res_glwe_brk_layout, &mut op_scratch);
        }

        let gap = 2 * plan.lut.drift / plan.lut.extension_factor();

        for i in 0..dnum_res {
            let mut res_row = res.at_view_mut(i, 0);

            match plan.output {
                CircuitBootstrappingOutput::Exponent { log_gap_out } => post_process(
                    module,
                    &mut res_row,
                    &res_glwe_atk_layout,
                    plan.log_gap_in,
                    log_gap_out,
                    plan.log_domain,
                    &key.atk,
                    &mut scratch_1.borrow(),
                ),
                CircuitBootstrappingOutput::Constant => {
                    let (mut tmp_row, mut op_scratch) = scratch_1.borrow().take_glwe_scratch(&res_row);
                    module.glwe_trace(&mut tmp_row, 0, &res_glwe_atk_layout, &key.atk, &mut op_scratch);
                    module.glwe_copy(&mut res_row, &tmp_row, &mut op_scratch);
                }
            }

            if i + 1 < dnum_res {
                module.glwe_rotate_assign(-(gap as i64), &mut res_glwe_atk_layout, &mut scratch_1.borrow());
            }
        }
    }

    // Expands GGLWE to GGSW using GGLWE(s^2)
    module.ggsw_expand_row(res, &key.tsk.to_backend_ref(), scratch);
}

#[allow(clippy::too_many_arguments)]
fn post_process<R, A, M, H, BE>(
    module: &M,
    res: &mut R,
    a: &A,
    log_gap_in: usize,
    log_gap_out: usize,
    log_domain: usize,
    auto_keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend<ZnxWord = i64>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
    H: GetAutomorphismKey<BE>,
    M: ModuleLogN + GLWETrace<BE> + GLWEPacking<BE> + GLWERotate<BE> + GLWECopy<BE>,
{
    if log_gap_in != log_gap_out {
        let steps = 1usize
            .checked_shl(log_domain as u32)
            .expect("circuit-bootstrap domain overflows usize");
        let (mut a_trace, scratch_1) = scratch.borrow().take_glwe_scratch(a);
        let (mut packed, scratch_2) = scratch_1.take_glwe_scratch(res);
        let (mut cts_vec, mut op_scratch) = scratch_2.take_glwe_slice_scratch(steps, a);

        module.glwe_trace(
            &mut a_trace,
            module.log_n() - log_gap_in + 1,
            a,
            auto_keys,
            &mut op_scratch.borrow(),
        );

        for (i, ct) in cts_vec.iter_mut().enumerate().take(steps) {
            if i != 0 {
                module.glwe_rotate_assign(-(1 << log_gap_in), &mut a_trace, &mut op_scratch.borrow());
            }

            module.glwe_copy(ct, &a_trace, &mut op_scratch);
        }

        let mut cts = HashMap::new();
        for (i, ct) in cts_vec.iter_mut().enumerate().take(steps) {
            cts.insert(i * (1 << log_gap_out), ct);
        }

        module.glwe_pack(&mut packed, cts, log_gap_out, auto_keys, &mut op_scratch);
        module.glwe_copy(res, &packed, &mut op_scratch);
    } else {
        let (mut traced, mut op_scratch) = scratch.borrow().take_glwe_scratch(res);
        module.glwe_trace(&mut traced, module.log_n() - log_gap_in + 1, a, auto_keys, &mut op_scratch);
        module.glwe_copy(res, &traced, &mut op_scratch);
    }
}

pub fn circuit_bootstrapping_prepare_to_constant_reference<R: GGSWInfos, M, BRA, BE>(
    module: &M,
    res_infos: &R,
    key: &CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE>,
    log_domain: usize,
    extension_factor: usize,
) -> CircuitBootstrappingPlan<BE::OwnedBuf>
where
    BRA: BlindRotationAlgo,
    BE: Backend<ZnxWord = i64>,
    M: ModuleN + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64> + LookupTableFactory<BE::OwnedBuf, i64>,
{
    prepare_circuit_bootstrapping_plan(
        module,
        res_infos,
        key,
        log_domain,
        extension_factor,
        CircuitBootstrappingOutput::Constant,
    )
}
pub fn circuit_bootstrapping_prepare_to_exponent_reference<R: GGSWInfos, M, BRA, BE>(
    module: &M,
    log_gap_out: usize,
    res_infos: &R,
    key: &CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE>,
    log_domain: usize,
    extension_factor: usize,
) -> CircuitBootstrappingPlan<BE::OwnedBuf>
where
    BRA: BlindRotationAlgo,
    BE: Backend<ZnxWord = i64>,
    M: ModuleN + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64> + LookupTableFactory<BE::OwnedBuf, i64>,
{
    prepare_circuit_bootstrapping_plan(
        module,
        res_infos,
        key,
        log_domain,
        extension_factor,
        CircuitBootstrappingOutput::Exponent { log_gap_out },
    )
}
pub fn circuit_bootstrapping_execute_prepared_tmp_bytes_reference<A: CircuitBootstrappingKeyInfos, M, BRA, BE>(
    module: &M,
    plan: &CircuitBootstrappingPlanLayout,
    key: &A,
) -> usize
where
    BRA: BlindRotationAlgo,
    BE: Backend<ZnxWord = i64>,
    M: GLWEBytesOf<BE>
        + BlindRotationExecute<BRA, BE>
        + GLWETrace<BE>
        + GLWEPacking<BE>
        + GGSWExpandRows<BE>
        + GLWERotate<BE>
        + GLWECopy<BE>,
{
    circuit_bootstrapping_prepared_tmp_bytes(
        module,
        &plan.output_layout,
        CircuitBootstrappingExecutionConfig {
            output: plan.log_gap_out.map_or(CircuitBootstrappingOutput::Constant, |log_gap_out| {
                CircuitBootstrappingOutput::Exponent { log_gap_out }
            }),
            log_domain: plan.log_domain.unwrap_or(0),
            log_gap_in: plan.log_gap_in,
            extension_factor: plan.extension_factor,
        },
        plan.block_size,
        key,
    )
}
pub fn circuit_bootstrapping_execute_prepared_reference<R, L, M, BRA, BE>(
    module: &M,
    res: &mut R,
    lwe: &L,
    key: &CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE>,
    plan: &CircuitBootstrappingPlan<BE::OwnedBuf>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    R: GGSWToBackendMut<BE> + GGSWAtViewRef<BE> + GGSWAtViewMut<BE> + GGSWInfos,
    L: LWEToBackendRef<BE> + LWEInfos,
    BRA: BlindRotationAlgo,
    BE: Backend<ZnxWord = i64>,
    M: ModuleLogN
        + GLWEBytesOf<BE>
        + BlindRotationExecute<BRA, BE>
        + GLWETrace<BE>
        + GLWEPacking<BE>
        + GGSWExpandRows<BE>
        + GLWERotate<BE>
        + GLWECopy<BE>,
{
    assert_eq!(
        res.ggsw_layout(),
        plan.output_layout,
        "circuit-bootstrapping plan/output layout mismatch"
    );
    plan.assert_key_compatible(key);
    let needed = circuit_bootstrapping_execute_prepared_tmp_bytes_reference::<_, _, BRA, BE>(module, &plan.layout(), key);
    assert!(scratch.available() >= needed, "insufficient circuit-bootstrapping scratch");
    circuit_bootstrap_prepared(module, res, lwe, key, plan, scratch);
}

#[cfg(test)]
mod tests {
    use super::validate_lut_coefficients;

    #[test]
    fn exponent_lut_does_not_apply_constant_message_scaling_bound() {
        validate_lut_coefficients(true, 20, 3, 24);
    }

    #[test]
    #[should_panic(expected = "LUT coefficient overflow")]
    fn constant_lut_rejects_message_scaling_overflow() {
        validate_lut_coefficients(false, 20, 3, 24);
    }
}
