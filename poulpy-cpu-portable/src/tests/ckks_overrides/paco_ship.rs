//! The PaCo and SHIP drivers enter the selected coefficient encoding, masking and
//! mux hooks, and their queries carry the hooks' workspaces.
use super::OverrideBackend;
use super::carry::with_marked_region;
use crate::ckks_encoding::EncodingFFTTable;
use poulpy_ckks::{
    CKKSCtBounds,
    api::PaCoScalar,
    layouts::{CKKSCiphertextOwned, CKKSPlaintextOwned, HMuxRotKeyPrepared, PaCoContext, ShipPlan},
    oep::CKKSEncodingImpl,
    reference::ship::CKKSShipReference,
    test_suite::BASE19_PARAMS_F64,
};
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendRef, LWEInfos};
use poulpy_hal::layouts::{Backend, CnvPVecL, HostBytesBackend, Module, ScratchArena};
use std::{cell::Cell, collections::HashMap};

thread_local! {
    static PACO_ENCODING_CALLS: Cell<usize> = const { Cell::new(0) };
    static SHIP_MASKING_CALLS: Cell<usize> = const { Cell::new(0) };
    static SHIP_MUX_CALLS: Cell<usize> = const { Cell::new(0) };
}
const PACO_ENCODING_EXTRA: usize = 4352;
const SHIP_MASKING_EXTRA: usize = 4608;
const SHIP_MUX_EXTRA: usize = 4864;

unsafe impl poulpy_ckks::oep::CKKSPaCoCoeffEncodingImpl for OverrideBackend {
    fn ckks_paco_coeff_encodings_tmp_bytes<F>(
        _module: &Module<Self>,
        _context: &PaCoContext<Self, F>,
    ) -> poulpy_ckks::CKKSResult<usize>
    where
        F: PaCoScalar,
        Self: CKKSEncodingImpl<F>,
    {
        Ok(PACO_ENCODING_EXTRA)
    }

    fn ckks_paco_coeff_encodings<F, Src>(
        module: &Module<Self>,
        ciphertext: &Src,
        context: &PaCoContext<Self, F>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> poulpy_ckks::CKKSResult<[CKKSPlaintextOwned<Self>; 4]>
    where
        F: PaCoScalar,
        Self: CKKSEncodingImpl<F>,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
    {
        PACO_ENCODING_CALLS.set(PACO_ENCODING_CALLS.get() + 1);
        with_marked_region::<PACO_ENCODING_EXTRA, _>(scratch, |rest| {
            crate::ckks_paco::paco_coeff_encodings_staged::<Self, F, Src>(
                module,
                ciphertext,
                context.plan(),
                context.base2k(),
                rest,
            )
            .map_err(poulpy_ckks::CKKSError::from)
        })
    }
}
poulpy_ckks::impl_ckks_paco_derived_full!(OverrideBackend);
crate::impl_ckks_ship_coeff_encoding!(OverrideBackend);

unsafe impl poulpy_ckks::oep::CKKSShipImpl for OverrideBackend {
    fn ckks_ship_masking_accumulate_tmp_bytes(module: &Module<Self>, plan: &ShipPlan, base2k: usize) -> usize {
        SHIP_MASKING_EXTRA + module.ckks_ship_masking_accumulate_tmp_bytes_reference(plan, base2k)
    }

    fn ckks_ship_masking_accumulate(
        module: &Module<Self>,
        acc: &mut CKKSCiphertextOwned<Self>,
        plan: &ShipPlan,
        masks: &[CnvPVecL<<Self as Backend>::OwnedBuf, <Self as Backend>::DftWord, Self>],
        pis: &[CKKSPlaintextOwned<Self>],
        scratch: &mut ScratchArena<'_, Self>,
    ) -> poulpy_ckks::CKKSResult<()> {
        SHIP_MASKING_CALLS.set(SHIP_MASKING_CALLS.get() + 1);
        with_marked_region::<SHIP_MASKING_EXTRA, _>(scratch, |rest| {
            module.ckks_ship_masking_accumulate_reference(acc, plan, masks, pis, rest)
        })
    }

    fn ckks_ship_mux_rotate_tmp_bytes<C, K>(module: &Module<Self>, ct: &C, key: &K, term_count: usize) -> usize
    where
        C: LWEInfos,
        K: GGLWEInfos,
    {
        SHIP_MUX_EXTRA + module.ckks_ship_mux_rotate_tmp_bytes_reference(ct, key, term_count)
    }

    fn ckks_ship_mux_rotate(
        module: &Module<Self>,
        ct: &mut CKKSCiphertextOwned<Self>,
        keys: &[HMuxRotKeyPrepared<<Self as Backend>::OwnedBuf, Self>],
        plans: &HashMap<i64, <Self as poulpy_hal::oep::HalVecZnxDftImpl>::AutomorphismPlan>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> poulpy_ckks::CKKSResult<()> {
        SHIP_MUX_CALLS.set(SHIP_MUX_CALLS.get() + 1);
        with_marked_region::<SHIP_MUX_EXTRA, _>(scratch, |rest| module.ckks_ship_mux_rotate_reference(ct, keys, plans, rest))
    }
}

#[test]
fn paco_drivers_enter_the_selected_coefficient_encoding() {
    let host = Module::<HostBytesBackend>::new(BASE19_PARAMS_F64.n as u64);
    let module = Module::<OverrideBackend>::new(BASE19_PARAMS_F64.n as u64);
    PACO_ENCODING_CALLS.set(0);
    // The suite runs each driver on exactly the arena its query reports.
    poulpy_ckks::test_suite::paco_bootstrap::test_paco_seq_bootstrap::<OverrideBackend, f64, EncodingFFTTable<f64>>(
        BASE19_PARAMS_F64,
        &module,
        &host,
    );
    assert!(PACO_ENCODING_CALLS.get() > 0);
}

#[test]
fn ship_drivers_enter_the_selected_masking_and_mux() {
    let host = Module::<HostBytesBackend>::new(BASE19_PARAMS_F64.n as u64);
    let module = Module::<OverrideBackend>::new(BASE19_PARAMS_F64.n as u64);
    SHIP_MASKING_CALLS.set(0);
    SHIP_MUX_CALLS.set(0);
    // The suite runs the bootstrap on exactly the arena its query reports.
    poulpy_ckks::test_suite::ship::test_ship_bootstrap::<OverrideBackend, f64, EncodingFFTTable<f64>>(
        BASE19_PARAMS_F64,
        &module,
        &host,
    );
    assert!(SHIP_MASKING_CALLS.get() > 0);
    assert!(SHIP_MUX_CALLS.get() > 0);
}
