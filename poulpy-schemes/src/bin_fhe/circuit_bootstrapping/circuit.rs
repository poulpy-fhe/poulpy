use std::collections::HashMap;

use poulpy_hal::{
    api::{ModuleLogN, ModuleN, ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, DataRef, Module, Scratch, ScratchOwned},
};

use poulpy_core::{
    GGSWExpandRows, GGSWFromGGLWE, GLWECopy, GLWEDecrypt, GLWENormalize, GLWEPacking, GLWERotate, GLWETrace, ScratchTakeCore,
    layouts::{
        Dsize, GGLWE, GGLWEInfos, GGLWELayout, GGLWEPreparedToRef, GGSWInfos, GGSWToMut, GLWEAutomorphismKeyHelper, GLWEInfos,
        GLWELayout, GLWESecretPreparedFactory, GLWEToMut, GLWEToRef, GetGaloisElement, LWEInfos, LWEToRef, Rank,
    },
};

use poulpy_core::layouts::{GGSW, GLWE, LWE};

use crate::bin_fhe::{
    blind_rotation::{
        BlindRotationAlgo, BlindRotationExecute, LookUpTableLayout, LookUpTableRotationDirection, LookupTable, LookupTableFactory,
    },
    circuit_bootstrapping::{CircuitBootstrappingKeyInfos, CircuitBootstrappingKeyPrepared},
};

/// Trait for evaluating a complete circuit bootstrapping.
///
/// Implemented for `Module<BE>` when the backend satisfies the full set of
/// required polynomial-arithmetic trait bounds.  Callers should use the
/// convenience methods on [`CircuitBootstrappingKeyPrepared`] rather than
/// invoking this trait directly.
pub trait CircuitBootstrappingExecute<BRA: BlindRotationAlgo, BE: Backend> {
    /// Returns the minimum scratch-space size (bytes) required by the circuit
    /// bootstrapping evaluation methods.
    ///
    /// `block_size` and `extension_factor` are forwarded to the underlying
    /// blind-rotation scratch estimator.  The total includes intermediate GLWE
    /// and GGLWE allocations.
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

    /// Bootstraps `lwe` into `res`, encoding the plaintext as the constant
    /// term of each GGSW row polynomial.
    ///
    /// `log_domain` controls the number of discrete values representable (the
    /// LUT has `2^log_domain` entries).
    fn circuit_bootstrapping_execute_to_constant<R, L, D>(
        &self,
        res: &mut R,
        lwe: &L,
        key: &CircuitBootstrappingKeyPrepared<D, BRA, BE>,
        log_domain: usize,
        extension_factor: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGSWToMut + GGSWInfos,
        L: LWEToRef + LWEInfos,
        D: DataRef;

    /// Bootstraps `lwe` into `res`, encoding the plaintext in the exponent of
    /// the polynomial variable.
    ///
    /// `log_gap_out` controls the spacing of output coefficients (used in
    /// post-processing to adjust the gap for downstream operations).
    #[allow(clippy::too_many_arguments)]
    fn circuit_bootstrapping_execute_to_exponent<R, L, D>(
        &self,
        log_gap_out: usize,
        res: &mut R,
        lwe: &L,
        key: &CircuitBootstrappingKeyPrepared<D, BRA, BE>,
        log_domain: usize,
        extension_factor: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGSWToMut + GGSWInfos,
        L: LWEToRef + LWEInfos,
        D: DataRef;
}

impl<D: DataRef, BRA: BlindRotationAlgo, BE: Backend> CircuitBootstrappingKeyPrepared<D, BRA, BE> {
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
        scratch: &mut Scratch<BE>,
    ) where
        M: CircuitBootstrappingExecute<BRA, BE>,
        R: GGSWToMut + GGSWInfos,
        L: LWEToRef + LWEInfos,
    {
        module.circuit_bootstrapping_execute_to_constant(res, lwe, self, log_domain, extension_factor, scratch);
    }

    /// Convenience method: bootstraps `lwe` into `res` using the exponent
    /// encoding.
    ///
    /// See [`CircuitBootstrappingExecute::circuit_bootstrapping_execute_to_exponent`].
    #[allow(clippy::too_many_arguments)]
    pub fn execute_to_exponent<R, L, M>(
        &self,
        module: &M,
        log_gap_out: usize,
        res: &mut R,
        lwe: &L,
        log_domain: usize,
        extension_factor: usize,
        scratch: &mut Scratch<BE>,
    ) where
        M: CircuitBootstrappingExecute<BRA, BE>,
        R: GGSWToMut + GGSWInfos,
        L: LWEToRef + LWEInfos,
    {
        module.circuit_bootstrapping_execute_to_exponent(log_gap_out, res, lwe, self, log_domain, extension_factor, scratch);
    }
}

impl<BRA: BlindRotationAlgo, BE: Backend> CircuitBootstrappingExecute<BRA, BE> for Module<BE>
where
    Self: ModuleN
        + LookupTableFactory
        + BlindRotationExecute<BRA, BE>
        + GLWETrace<BE>
        + GLWEPacking<BE>
        + GGSWFromGGLWE<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEDecrypt<BE>
        + GLWERotate<BE>
        + GLWENormalize<BE>
        + GLWECopy
        + GGSWExpandRows<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
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
        let gglwe_infos: GGLWELayout = GGLWELayout {
            n: res_infos.n(),
            base2k: res_infos.base2k(),
            k: res_infos.max_k(),
            dnum: res_infos.dnum(),
            dsize: Dsize(1),
            rank_in: res_infos.rank().max(Rank(1)),
            rank_out: res_infos.rank(),
        };

        self.blind_rotation_execute_tmp_bytes(block_size, extension_factor, res_infos, &cbt_infos.brk_infos())
            .max(self.glwe_trace_tmp_bytes(res_infos, res_infos, &cbt_infos.atk_infos()))
            .max(self.ggsw_from_gglwe_tmp_bytes(res_infos, &cbt_infos.tsk_infos()))
            + GLWE::<Vec<u8>>::bytes_of_from_infos(res_infos)
            + GGLWE::bytes_of_from_infos(&gglwe_infos)
    }

    fn circuit_bootstrapping_execute_to_constant<R, L, D>(
        &self,
        res: &mut R,
        lwe: &L,
        key: &CircuitBootstrappingKeyPrepared<D, BRA, BE>,
        log_domain: usize,
        extension_factor: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGSWToMut + GGSWInfos,
        L: LWEToRef + LWEInfos,
        D: DataRef,
    {
        assert!(
            scratch.available() >= self.circuit_bootstrapping_execute_tmp_bytes(key.block_size(), extension_factor, res, key)
        );

        circuit_bootstrap_core(false, self, 0, res, lwe, log_domain, extension_factor, key, scratch);
    }

    fn circuit_bootstrapping_execute_to_exponent<R, L, D>(
        &self,
        log_gap_out: usize,
        res: &mut R,
        lwe: &L,
        key: &CircuitBootstrappingKeyPrepared<D, BRA, BE>,
        log_domain: usize,
        extension_factor: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGSWToMut + GGSWInfos,
        L: LWEToRef + LWEInfos,
        D: DataRef,
    {
        assert!(
            scratch.available() >= self.circuit_bootstrapping_execute_tmp_bytes(key.block_size(), extension_factor, res, key)
        );

        circuit_bootstrap_core(true, self, log_gap_out, res, lwe, log_domain, extension_factor, key, scratch);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn circuit_bootstrap_core<R, L, D, M, BRA: BlindRotationAlgo, BE: Backend>(
    to_exponent: bool,
    module: &M,
    log_gap_out: usize,
    res: &mut R,
    lwe: &L,
    log_domain: usize,
    extension_factor: usize,
    key: &CircuitBootstrappingKeyPrepared<D, BRA, BE>,
    scratch: &mut Scratch<BE>,
) where
    R: GGSWToMut,
    L: LWEToRef,
    D: DataRef,
    M: ModuleN
        + LookupTableFactory
        + BlindRotationExecute<BRA, BE>
        + GLWETrace<BE>
        + GLWEPacking<BE>
        + GGSWFromGGLWE<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEDecrypt<BE>
        + GLWERotate<BE>
        + ModuleLogN
        + GLWENormalize<BE>
        + GLWECopy
        + GGSWExpandRows<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let res: &mut GGSW<&mut [u8]> = &mut res.to_mut();
    let lwe: &LWE<&[u8]> = &lwe.to_ref();

    assert_eq!(res.n(), key.brk.n());

    let res_base2k: usize = res.base2k().as_usize();
    let dnum_res: usize = res.dnum().into();

    let alpha: usize = dnum_res.next_power_of_two();

    // Validate that LUT coefficient exponents fit in i64 before building the LUT.
    // The maximum exponent is res_base2k * (dnum_res - 1); 1i64 << that value must not overflow.
    assert!(
        dnum_res == 0 || res_base2k * (dnum_res - 1) < i64::BITS as usize,
        "LUT coefficient overflow: res_base2k={res_base2k} * (dnum_res-1)={} >= {} bits",
        dnum_res.saturating_sub(1),
        i64::BITS,
    );
    // For the constant-mode LUT the coefficient also scales by j < 2^log_domain.
    assert!(
        !to_exponent || log_domain + res_base2k * dnum_res.saturating_sub(1) < i64::BITS as usize,
        "LUT coefficient overflow: log_domain={log_domain} + res_base2k*dnum_res would exceed i64"
    );

    let mut f: Vec<i64> = vec![0i64; (1 << log_domain) * alpha];

    if to_exponent {
        (0..dnum_res).for_each(|i| {
            f[i] = 1 << (res_base2k * (dnum_res - 1 - i));
        });
    } else {
        (0..1 << log_domain).for_each(|j| {
            (0..dnum_res).for_each(|i| {
                f[j * alpha + i] = j as i64 * (1 << (res_base2k * (dnum_res - 1 - i)));
            });
        });
    }

    let lut_infos: LookUpTableLayout = LookUpTableLayout {
        n: module.n().into(),
        extension_factor,
        k: (res_base2k * dnum_res).into(),
        base2k: key.brk.base2k(),
    };

    // Lut precision, basically must be able to hold the decomposition power basis of the GGSW
    let mut lut: LookupTable = LookupTable::alloc(&lut_infos);
    lut.set(module, &f, res_base2k * dnum_res);

    if to_exponent {
        lut.set_rotation_direction(LookUpTableRotationDirection::Right);
    }

    let glwe_brk_layout = &GLWELayout {
        n: key.brk.n(),
        base2k: key.brk.base2k(),
        k: key.brk.max_k(),
        rank: key.brk.rank(),
    };

    let atk_layout: &GGLWELayout = &key.atk.automorphism_key_infos();

    let glwe_atk_layout: &GLWELayout = &GLWELayout {
        n: glwe_brk_layout.n(),
        base2k: atk_layout.base2k(),
        k: glwe_brk_layout.max_k(),
        rank: glwe_brk_layout.rank(),
    };

    let (mut res_glwe_atk_layout, scratch_1) = scratch.take_glwe(glwe_atk_layout);

    // Execute blind rotation over BRK layout and returns result over ATK layout
    {
        let (mut res_glwe_brk_layout, scratch_2) = scratch_1.take_glwe(glwe_brk_layout);
        key.brk.execute(module, &mut res_glwe_brk_layout, lwe, &lut, scratch_2);

        if res_glwe_brk_layout.base2k() == res_glwe_atk_layout.base2k() {
            module.glwe_copy(&mut res_glwe_atk_layout, &res_glwe_brk_layout);
        } else {
            module.glwe_normalize(&mut res_glwe_atk_layout, &res_glwe_brk_layout, scratch_2);
        }
    }

    let gap: usize = 2 * lut.drift / lut.extension_factor();

    assert!(
        gap > 0,
        "gap must be positive (lut.drift={}, extension_factor={}); ensure f_len <= domain_size",
        lut.drift,
        lut.extension_factor(),
    );

    let log_gap_in: usize = (usize::BITS - (gap * alpha - 1).leading_zeros()) as _;

    for i in 0..dnum_res {
        let mut res_row: GLWE<&mut [u8]> = res.at_mut(i, 0);

        if to_exponent {
            // Isolates i-th LUT and moves coefficients according to requested gap.
            post_process(
                module,
                &mut res_row,
                &res_glwe_atk_layout,
                log_gap_in,
                log_gap_out,
                log_domain,
                &key.atk,
                scratch_1,
            );
        } else {
            module.glwe_trace(&mut res_row, 0, &res_glwe_atk_layout, &key.atk, scratch_1);
        }

        if i + 1 < dnum_res {
            module.glwe_rotate_assign(-(gap as i64), &mut res_glwe_atk_layout, scratch_1);
        }
    }

    // Expands GGLWE to GGSW using GGLWE(s^2)
    module.ggsw_expand_row(res, &key.tsk, scratch);
}

#[allow(clippy::too_many_arguments)]
fn post_process<R, A, M, H, K, BE: Backend>(
    module: &M,
    res: &mut R,
    a: &A,
    log_gap_in: usize,
    log_gap_out: usize,
    log_domain: usize,
    auto_keys: &H,
    scratch: &mut Scratch<BE>,
) where
    R: GLWEToMut + GLWEInfos,
    A: GLWEToRef + GLWEInfos,
    H: GLWEAutomorphismKeyHelper<K, BE>,
    K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
    M: ModuleLogN + GLWETrace<BE> + GLWEPacking<BE> + GLWERotate<BE> + GLWECopy,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    // TODO: optimize with packing and final partial trace
    // If gap_out < gap_in, then we need to repack, i.e. reduce the cap between coefficients.
    if log_gap_in != log_gap_out {
        let (mut a_trace, scratch_1) = scratch.take_glwe(a);

        // First partial trace, vanishes all coefficients which are not multiples of gap_in
        // [1, 1, 1, 1, 0, 0, 0, ..., 0, 0, -1, -1, -1, -1] -> [1, 0, 0, 0, 0, 0, 0, ..., 0, 0, 0, 0, 0, 0]
        module.glwe_trace(&mut a_trace, module.log_n() - log_gap_in + 1, a, auto_keys, scratch_1);

        let steps: usize = 1 << log_domain;

        // TODO: from Scratch
        let (mut cts_vec, scratch_2) = scratch_1.take_glwe_slice(steps, a);

        for (i, ct) in cts_vec.iter_mut().enumerate().take(steps) {
            if i != 0 {
                module.glwe_rotate_assign(-(1 << log_gap_in), &mut a_trace, scratch_2);
            }

            module.glwe_copy(ct, &a_trace);
        }

        let mut cts: HashMap<usize, &mut GLWE<&mut [u8]>> = HashMap::new();
        for (i, ct) in cts_vec.iter_mut().enumerate().take(steps) {
            cts.insert(i * (1 << log_gap_out), ct);
        }

        module.glwe_pack(res, cts, log_gap_out, auto_keys, scratch_2);
    } else {
        module.glwe_trace(res, module.log_n() - log_gap_in + 1, a, auto_keys, scratch);
    }
}
