use poulpy_hal::layouts::CoeffNormalized;
use std::collections::HashMap;

use poulpy_hal::{
    api::{ModuleLogN, ModuleN},
    layouts::{Backend, Data, Module, ScratchArena},
};

use poulpy_core::{
    GGSWExpandRows, GLWECopy, GLWENormalize, GLWEPacking, GLWERotate, GLWETrace, ScratchArenaTakeCore,
    layouts::{
        GGLWELayout, GGSWAtViewMut, GGSWAtViewRef, GGSWInfos, GGSWLayout, GGSWToBackendMut, GLWEInfos, GLWELayout,
        GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LWEInfos, LWEToBackendRef, ModuleCoreAlloc,
    },
};

use crate::{
    blind_rotation::{
        BlindRotationAlgo, BlindRotationExecute, LookUpTableLayout, LookUpTableRotationDirection, LookupTable, LookupTableFactory,
    },
    circuit_bootstrapping::{CircuitBootstrappingKeyInfos, CircuitBootstrappingKeyPrepared},
};
use poulpy_core::GLWEBytesOf;
use poulpy_core::layouts::prepared::GGLWEToGGSWKeyPreparedToBackendRef;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CircuitBootstrappingOutput {
    Constant,
    Exponent { log_gap_out: usize },
}

#[derive(Clone, Copy)]
struct CircuitBootstrappingExecutionConfig {
    output: CircuitBootstrappingOutput,
    log_domain: usize,
    log_gap_in: Option<usize>,
    extension_factor: usize,
}

/// LUT and dimensional state prepared for repeated circuit bootstrapping.
///
/// Preparing a plan performs the host-side LUT construction and uploads it to
/// the selected backend. Executing it only uses the plan, the prepared key,
/// the input/output ciphertexts, and caller-owned scratch space.
pub struct CircuitBootstrappingPlan<D: Data> {
    lut: LookupTable<D, i64>,
    output_layout: GGSWLayout,
    output: CircuitBootstrappingOutput,
    log_domain: usize,
    log_gap_in: usize,
    extension_factor: usize,
    key_layout: crate::circuit_bootstrapping::CircuitBootstrappingKeyLayout,
    block_size: usize,
}

impl<D: Data> CircuitBootstrappingPlan<D> {
    /// Returns the output layout this plan was prepared for.
    pub fn output_layout(&self) -> GGSWLayout {
        self.output_layout
    }

    /// Returns the minimum scratch-space size for repeated prepared execution.
    pub fn execute_tmp_bytes<M, BRA, BE>(&self, module: &M, key: &CircuitBootstrappingKeyPrepared<D, BRA, BE>) -> usize
    where
        BRA: BlindRotationAlgo,
        BE: Backend<OwnedBuf = D, ZnxWord = i64> + 'static,
        M: ModuleN
            + GLWEBytesOf<BE>
            + BlindRotationExecute<BRA, BE>
            + GLWETrace<BE>
            + GLWEPacking<BE>
            + GGSWExpandRows<BE>
            + GLWERotate<BE>
            + GLWENormalize<BE>,
    {
        self.assert_key_compatible(key);
        circuit_bootstrapping_prepared_tmp_bytes(
            module,
            &self.output_layout,
            CircuitBootstrappingExecutionConfig {
                output: self.output,
                log_domain: self.log_domain,
                log_gap_in: Some(self.log_gap_in),
                extension_factor: self.extension_factor,
            },
            self.block_size,
            key,
        )
    }

    /// Executes a previously prepared circuit-bootstrapping plan.
    pub fn execute<M, R, L, BRA, BE>(
        &self,
        module: &M,
        res: &mut R,
        lwe: &L,
        key: &CircuitBootstrappingKeyPrepared<D, BRA, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        BRA: BlindRotationAlgo,
        BE: Backend<OwnedBuf = D, ZnxWord = i64> + 'static,
        R: GGSWToBackendMut<BE> + GGSWAtViewRef<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        L: LWEToBackendRef<BE> + LWEInfos,
        M: ModuleN
            + BlindRotationExecute<BRA, BE>
            + GLWETrace<BE>
            + GLWEPacking<BE>
            + GGSWExpandRows<BE>
            + GLWERotate<BE>
            + ModuleLogN
            + GLWENormalize<BE>
            + GLWECopy<BE>
            + GLWEBytesOf<BE>,
    {
        assert_eq!(
            res.ggsw_layout(),
            self.output_layout,
            "circuit-bootstrapping plan/output layout mismatch"
        );
        self.assert_key_compatible(key);
        let needed = self.execute_tmp_bytes(module, key);
        assert!(
            scratch.available() >= needed,
            "scratch.available(): {} < CircuitBootstrappingPlan::execute_tmp_bytes: {needed}",
            scratch.available()
        );
        circuit_bootstrap_prepared(module, res, lwe, key, self, scratch);
    }

    fn assert_key_compatible<BRA, BE>(&self, key: &CircuitBootstrappingKeyPrepared<D, BRA, BE>)
    where
        BRA: BlindRotationAlgo,
        BE: Backend<OwnedBuf = D, ZnxWord = i64>,
    {
        assert_eq!(
            key.brk_infos(),
            self.key_layout.brk_layout,
            "circuit-bootstrapping plan/BRK mismatch"
        );
        assert_eq!(
            key.atk_infos(),
            self.key_layout.atk_layout,
            "circuit-bootstrapping plan/ATK mismatch"
        );
        assert_eq!(
            key.tsk_infos(),
            self.key_layout.tsk_layout,
            "circuit-bootstrapping plan/TSK mismatch"
        );
        assert_eq!(
            key.block_size(),
            self.block_size,
            "circuit-bootstrapping plan/block-size mismatch"
        );
    }
}

/// Trait for evaluating a complete circuit bootstrapping.
///
/// Implemented for `Module<BE>` when the backend satisfies the full set of
/// required polynomial-arithmetic trait bounds.  Callers should use the
/// convenience methods on [`CircuitBootstrappingKeyPrepared`] rather than
/// invoking this trait directly.
pub trait CircuitBootstrappingExecute<BRA, BE>
where
    BRA: BlindRotationAlgo,
    BE: Backend,
    Self: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>,
{
    /// Returns the minimum scratch-space size for constant-encoding circuit
    /// bootstrapping.
    ///
    /// This compatibility estimator predates exponent-mode parameters. Use
    /// [`Self::circuit_bootstrapping_execute_to_exponent_tmp_bytes`] for that
    /// mode.
    fn circuit_bootstrapping_execute_tmp_bytes<R, A>(
        &self,
        block_size: usize,
        extension_factor: usize,
        res_infos: &R,
        cbt_infos: &A,
    ) -> usize
    where
        R: GGSWInfos,
        A: CircuitBootstrappingKeyInfos;

    /// Returns the scratch-space size for constant-encoding execution.
    fn circuit_bootstrapping_execute_to_constant_tmp_bytes<R, A>(
        &self,
        block_size: usize,
        extension_factor: usize,
        res_infos: &R,
        cbt_infos: &A,
    ) -> usize
    where
        R: GGSWInfos,
        A: CircuitBootstrappingKeyInfos,
    {
        self.circuit_bootstrapping_execute_tmp_bytes(block_size, extension_factor, res_infos, cbt_infos)
    }

    /// Returns the scratch-space size for exponent-encoding execution.
    #[allow(clippy::too_many_arguments)]
    fn circuit_bootstrapping_execute_to_exponent_tmp_bytes<R, A>(
        &self,
        log_gap_out: usize,
        log_domain: usize,
        block_size: usize,
        extension_factor: usize,
        res_infos: &R,
        cbt_infos: &A,
    ) -> usize
    where
        R: GGSWInfos,
        A: CircuitBootstrappingKeyInfos;

    /// Bootstraps `lwe` into `res`, encoding the plaintext as the constant
    /// term of each GGSW row polynomial.
    ///
    /// `log_domain` controls the number of discrete values representable (the
    /// LUT has `2^log_domain` entries).
    fn circuit_bootstrapping_execute_to_constant<R, L>(
        &self,
        res: &mut R,
        lwe: &L,
        key: &CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE>,
        log_domain: usize,
        extension_factor: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWAtViewRef<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        L: LWEToBackendRef<BE> + LWEInfos;

    /// Bootstraps `lwe` into `res`, encoding the plaintext in the exponent of
    /// the polynomial variable.
    ///
    /// `log_gap_out` controls the spacing of output coefficients (used in
    /// post-processing to adjust the gap for downstream operations).
    /// Allocate scratch with
    /// [`Self::circuit_bootstrapping_execute_to_exponent_tmp_bytes`].
    #[allow(clippy::too_many_arguments)]
    fn circuit_bootstrapping_execute_to_exponent<R, L>(
        &self,
        log_gap_out: usize,
        res: &mut R,
        lwe: &L,
        key: &CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE>,
        log_domain: usize,
        extension_factor: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWAtViewRef<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        L: LWEToBackendRef<BE> + LWEInfos;
}

impl<BRA, BE> CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE>
where
    BRA: BlindRotationAlgo,
    BE: Backend<ZnxWord = i64>,
{
    /// Prepares a reusable constant-encoding circuit-bootstrap plan.
    pub fn prepare_to_constant<M, R>(
        &self,
        module: &M,
        res_infos: &R,
        log_domain: usize,
        extension_factor: usize,
    ) -> CircuitBootstrappingPlan<BE::OwnedBuf>
    where
        M: ModuleN + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64> + LookupTableFactory<BE::OwnedBuf, i64>,
        R: GGSWInfos,
    {
        prepare_circuit_bootstrapping_plan(
            module,
            res_infos,
            self,
            log_domain,
            extension_factor,
            CircuitBootstrappingOutput::Constant,
        )
    }

    /// Prepares a reusable exponent-encoding circuit-bootstrap plan.
    pub fn prepare_to_exponent<M, R>(
        &self,
        module: &M,
        log_gap_out: usize,
        res_infos: &R,
        log_domain: usize,
        extension_factor: usize,
    ) -> CircuitBootstrappingPlan<BE::OwnedBuf>
    where
        M: ModuleN + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64> + LookupTableFactory<BE::OwnedBuf, i64>,
        R: GGSWInfos,
    {
        prepare_circuit_bootstrapping_plan(
            module,
            res_infos,
            self,
            log_domain,
            extension_factor,
            CircuitBootstrappingOutput::Exponent { log_gap_out },
        )
    }

    /// Convenience method: bootstraps `lwe` into the GGSW ciphertext `res`
    /// using the constant-term encoding.
    ///
    /// See [`CircuitBootstrappingExecute::circuit_bootstrapping_execute_to_constant`].
    pub fn execute_to_constant<M, L, R>(
        &self,
        module: &M,
        res: &mut R,
        lwe: &L,
        log_domain: usize,
        extension_factor: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        M: CircuitBootstrappingExecute<BRA, BE>,
        R: GGSWToBackendMut<BE> + GGSWAtViewRef<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        L: LWEToBackendRef<BE> + LWEInfos,
    {
        module.circuit_bootstrapping_execute_to_constant(res, lwe, self, log_domain, extension_factor, scratch);
    }

    /// Convenience method: bootstraps `lwe` into `res` using the exponent
    /// encoding.
    ///
    /// See [`CircuitBootstrappingExecute::circuit_bootstrapping_execute_to_exponent`]
    /// and its mode-specific scratch estimator.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_to_exponent<R, L, M>(
        &self,
        module: &M,
        log_gap_out: usize,
        res: &mut R,
        lwe: &L,
        log_domain: usize,
        extension_factor: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        M: CircuitBootstrappingExecute<BRA, BE>,
        R: GGSWToBackendMut<BE> + GGSWAtViewRef<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        L: LWEToBackendRef<BE> + LWEInfos,
    {
        module.circuit_bootstrapping_execute_to_exponent(log_gap_out, res, lwe, self, log_domain, extension_factor, scratch);
    }
}

impl<BRA, BE> CircuitBootstrappingExecute<BRA, BE> for Module<BE>
where
    BRA: BlindRotationAlgo,
    BE: Backend<ZnxWord = i64> + 'static,
    Self: ModuleN
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + LookupTableFactory<BE::OwnedBuf, BE::ZnxWord>
        + BlindRotationExecute<BRA, BE>
        + GLWETrace<BE>
        + GLWEPacking<BE>
        + GGSWExpandRows<BE>
        + GLWERotate<BE>
        + GLWENormalize<BE>
        + GLWECopy<BE>,
{
    fn circuit_bootstrapping_execute_tmp_bytes<R, A>(
        &self,
        block_size: usize,
        extension_factor: usize,
        res_infos: &R,
        cbt_infos: &A,
    ) -> usize
    where
        R: GGSWInfos,
        A: CircuitBootstrappingKeyInfos,
    {
        circuit_bootstrapping_prepared_tmp_bytes(
            self,
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

    fn circuit_bootstrapping_execute_to_constant_tmp_bytes<R, A>(
        &self,
        block_size: usize,
        extension_factor: usize,
        res_infos: &R,
        cbt_infos: &A,
    ) -> usize
    where
        R: GGSWInfos,
        A: CircuitBootstrappingKeyInfos,
    {
        self.circuit_bootstrapping_execute_tmp_bytes(block_size, extension_factor, res_infos, cbt_infos)
    }

    fn circuit_bootstrapping_execute_to_exponent_tmp_bytes<R, A>(
        &self,
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
    {
        circuit_bootstrapping_prepared_tmp_bytes(
            self,
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

    fn circuit_bootstrapping_execute_to_constant<R, L>(
        &self,
        res: &mut R,
        lwe: &L,
        key: &CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE>,
        log_domain: usize,
        extension_factor: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWAtViewRef<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        L: LWEToBackendRef<BE> + LWEInfos,
    {
        let plan = key.prepare_to_constant(self, res, log_domain, extension_factor);
        plan.execute(self, res, lwe, key, scratch);
    }

    fn circuit_bootstrapping_execute_to_exponent<R, L>(
        &self,
        log_gap_out: usize,
        res: &mut R,
        lwe: &L,
        key: &CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE>,
        log_domain: usize,
        extension_factor: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWAtViewRef<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        L: LWEToBackendRef<BE> + LWEInfos,
    {
        let plan = key.prepare_to_exponent(self, log_gap_out, res, log_domain, extension_factor);
        plan.execute(self, res, lwe, key, scratch);
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
    M: ModuleN
        + GLWEBytesOf<BE>
        + BlindRotationExecute<BRA, BE>
        + GLWETrace<BE>
        + GLWEPacking<BE>
        + GGSWExpandRows<BE>
        + GLWERotate<BE>
        + GLWENormalize<BE>,
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
    let convert = if glwe_brk_layout.base2k == glwe_atk_layout.base2k {
        0
    } else {
        module.glwe_normalize_tmp_bytes()
    };
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
    let rotate = module.glwe_rotate_tmp_bytes();
    let row_phase = match config.output {
        CircuitBootstrappingOutput::Constant => aligned(module.glwe_bytes_of_from_infos(&res_glwe_layout)) + trace_res,
        CircuitBootstrappingOutput::Exponent { log_gap_out } => {
            if config.log_gap_in.expect("prepared exponent execution requires its input gap") == log_gap_out {
                aligned(module.glwe_bytes_of_from_infos(&res_glwe_layout)) + trace_res
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
                        .max(module.glwe_pack_tmp_bytes(&res_glwe_layout, &atk_key_infos))
            }
        }
    };
    let online = atk_bytes + blind_phase.max(row_phase).max(rotate);
    online.max(module.ggsw_expand_rows_tmp_bytes(res_infos, &tsk_infos))
}

fn circuit_bootstrapping_log_gap_in<R: GGSWInfos>(res_infos: &R, log_domain: usize, extension_factor: usize) -> usize {
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
    BE: Backend<ZnxWord = i64> + 'static,
    R: GGSWToBackendMut<BE> + GGSWAtViewRef<BE> + GGSWAtViewMut<BE> + GGSWInfos,
    L: LWEToBackendRef<BE> + LWEInfos,
    M: ModuleN
        + BlindRotationExecute<BRA, BE>
        + GLWETrace<BE>
        + GLWEPacking<BE>
        + GGSWExpandRows<BE>
        + GLWERotate<BE>
        + ModuleLogN
        + GLWENormalize<BE>
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

            if res_glwe_brk_layout.base2k() == res_glwe_atk_layout.base2k() {
                module.glwe_copy(&mut res_glwe_atk_layout, &res_glwe_brk_layout);
            } else {
                module.glwe_normalize(&mut res_glwe_atk_layout, &res_glwe_brk_layout, &mut op_scratch);
            }
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
                    module.glwe_copy(&mut res_row, &tmp_row);
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
    BE: Backend<ZnxWord = i64> + 'static,
    R: GLWEToBackendMut<BE, State = CoeffNormalized> + GLWEInfos,
    A: GLWEToBackendRef<BE, State = CoeffNormalized> + GLWEInfos,
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

            module.glwe_copy(ct, &a_trace);
        }

        let mut cts = HashMap::new();
        for (i, ct) in cts_vec.iter_mut().enumerate().take(steps) {
            cts.insert(i * (1 << log_gap_out), ct);
        }

        module.glwe_pack(&mut packed, cts, log_gap_out, auto_keys, &mut op_scratch);
        module.glwe_copy(res, &packed);
    } else {
        let (mut traced, mut op_scratch) = scratch.borrow().take_glwe_scratch(res);
        module.glwe_trace(&mut traced, module.log_n() - log_gap_in + 1, a, auto_keys, &mut op_scratch);
        module.glwe_copy(res, &traced);
    }
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
