//! Runners for the `poulpy-libm` operations on one ciphertext.
//!
//! A runner builds its whole point once, outside the timed loop: the module, the plan on a host module of the same
//! degree, transferred to the backend, the secret and the tensor key, an input and a destination filled on the backend,
//! and the scratch the operation's query reports. The timed circuit does not depend on the slot values, so the inputs
//! are uniform fills rather than encryptions.

use poulpy_ckks::{
    GLWEPlaintextMeta, Scale, SetCKKSInfos, SlotsKind,
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc},
    test_suite::helpers::{TestContextSharedModule, upload_pt},
};
use poulpy_core::{
    GLWEMaskFill, GLWETensorKeyEncryptSk,
    api::{GLWESecretSampling, GLWETensorKeyPreparedFactory},
    layouts::{
        Base2K, Degree, Dsize, GGLWELayout, GLWELayout, GLWETensorKeyLayout, ModuleCoreAlloc, Rank, TorusPrecision, k_digit_of,
        prepared::GLWETensorKeyPrepared,
    },
};
use poulpy_hal::{
    api::{ModuleSynchronize, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostBytesBackend, HostStaged, Module, ScratchOwned, Standard},
    source::Source,
};
use poulpy_libm::{
    exp::{CKKSExpOps, ExpOptions, ExpPlan},
    log::{CKKSLogOps, LogOptions, LogPlan},
    pow::{CKKSPowOps, PowOptions, PowPlan},
};

use criterion::{Bencher, measurement::Measurement};

use crate::{libm::params::LibmBenchParams, timed};

/// Secret-key Hamming weight.
const HW: usize = 192;
/// Domain of the `exp`, `log` and `pow` bases.
const BASE_INTERVAL: (f64, f64) = (0.5, 1.5);
/// Domain of the `pow` exponents.
const EXPONENT_INTERVAL: (f64, f64) = (-1.0, 1.0);

/// What a runner asks of a module: key generation and uniform fills, the timed operations, and synchronization.
///
/// `HostStaged` serves only the transfer of the host-built plans, which poulpy-ckks setup builds on a host module.
pub trait LibmBenchModule<BE: Backend>:
    TestContextSharedModule<BE> + GLWEMaskFill<BE> + ModuleSynchronize<BE> + CKKSExpOps<BE> + CKKSLogOps<BE> + CKKSPowOps<BE>
{
}

impl<BE: Backend, M> LibmBenchModule<BE> for M where
    M: TestContextSharedModule<BE> + GLWEMaskFill<BE> + ModuleSynchronize<BE> + CKKSExpOps<BE> + CKKSLogOps<BE> + CKKSPowOps<BE>
{
}

fn base2k(bp: &LibmBenchParams) -> Base2K {
    Base2K(bp.base2k as u32)
}

/// The plans' coefficient layout: scale `2^log_delta`, one limb of budget.
fn coeff_meta(bp: &LibmBenchParams) -> poulpy_ckks::CKKSLayout {
    poulpy_ckks::CKKSLayout {
        glwe_layout: GLWELayout {
            n: Degree(bp.n as u32),
            base2k: base2k(bp),
            k: TorusPrecision((bp.log_delta + bp.base2k) as u32),
            rank: Rank(1),
        },
        meta: GLWEPlaintextMeta {
            scale: Scale::Log(bp.log_delta),
            slots: SlotsKind::Complex,
            log_sparsity: 0,
        },
    }
}

/// The ciphertext width for a plan consuming `consumed` bits: `2 log_delta` of result headroom and two limbs of guard on
/// top, rounded up to a limb.
fn ct_k(bp: &LibmBenchParams, consumed: usize) -> usize {
    (consumed + 2 * bp.log_delta + 2 * bp.base2k).next_multiple_of(bp.base2k)
}

fn tsk_layout(bp: &LibmBenchParams, k: usize) -> GLWETensorKeyLayout {
    let k_digit = k_digit_of(base2k(bp), Dsize(bp.dsize as u32));
    GLWETensorKeyLayout {
        n: Degree(bp.n as u32),
        base2k: base2k(bp),
        k_aux: TorusPrecision(k_digit.0 + bp.n.ilog2()),
        rank: Rank(1),
        k_digit,
        dnum: GGLWELayout::dnum_for_input(TorusPrecision(k as u32), k_digit),
    }
}

/// The module, the tensor key, two filled operands of width `k` and a filled destination.
struct Point<BE: Backend> {
    module: Module<BE>,
    tsk: GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
    a: CKKSCiphertextOwned<BE>,
    b: CKKSCiphertextOwned<BE>,
    res: CKKSCiphertextOwned<BE>,
}

impl<BE> Point<BE>
where
    BE: Backend<Ring = Standard> + 'static,
    Module<BE>: LibmBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    fn new(bp: &LibmBenchParams, k: usize) -> Self {
        let module = Module::<BE>::new(bp.n as u64);
        let layout = poulpy_ckks::CKKSLayout {
            glwe_layout: GLWELayout {
                k: TorusPrecision(k as u32),
                ..coeff_meta(bp).glwe_layout
            },
            meta: coeff_meta(bp).meta,
        };
        let tsk_layout = tsk_layout(bp, k);
        let mut source_xs = Source::new([0u8; 32]);
        let mut source_xa = Source::new([1u8; 32]);
        let mut source_xe = Source::new([2u8; 32]);
        let mut source_xu = Source::new([3u8; 32]);
        let mut sk_raw = module.glwe_secret_alloc_from_infos(&layout);
        module.glwe_secret_fill_ternary_hw(&mut sk_raw, HW, &mut source_xs);
        let mut arena = ScratchOwned::<BE>::alloc(
            module
                .glwe_tensor_key_encrypt_sk_tmp_bytes(&tsk_layout)
                .max(module.glwe_tensor_key_prepare_tmp_bytes(&tsk_layout)),
        );
        let mut tsk_raw = module.glwe_tensor_key_alloc_from_infos(&tsk_layout);
        module.glwe_tensor_key_encrypt_sk(&mut tsk_raw, &sk_raw, &mut source_xe, &mut source_xa, &mut arena.borrow());
        let mut tsk = module.alloc_tensor_key_prepared_from_infos(&tsk_layout);
        module.glwe_tensor_key_prepare(&mut tsk, &tsk_raw, &mut arena.borrow());
        let mut filled = || {
            let mut ct = module.ckks_ciphertext_alloc_from_infos(&layout);
            module.fill_glwe_from_source(&mut ct, &mut source_xu);
            ct.set_meta(layout.meta);
            ct
        };
        let (a, b, res) = (filled(), filled(), filled());
        Self { module, tsk, a, b, res }
    }
}

/// `ckks_exp` of a plan on `[0.5, 1.5]` at default options.
pub fn runner_ckks_exp<BE, M>(bencher: &mut Bencher<'_, M>, bp: &LibmBenchParams)
where
    BE: Backend<Ring = Standard> + HostStaged + 'static,
    Module<BE>: LibmBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    timed::<BE, M, _>(bencher, "ckks_exp", bp, || {
        let host = Module::<HostBytesBackend>::new(bp.n as u64);
        let plan = ExpPlan::from_precision(
            BASE_INTERVAL.0,
            BASE_INTERVAL.1,
            base2k(bp),
            coeff_meta(bp),
            ExpOptions::default(),
            &host,
        )
        .unwrap();
        let k = ct_k(bp, plan.consumed_bits(bp.log_delta));
        let Point {
            module,
            tsk,
            a,
            res: mut out,
            ..
        } = Point::<BE>::new(bp, k);
        let plan = plan.map_plaintexts(|pt| upload_pt(&module, pt));
        let mut scratch = ScratchOwned::<BE>::alloc(module.ckks_exp_tmp_bytes(&out, &a, &tsk_layout(bp, k), &plan));
        Box::new(move || {
            module.ckks_exp(&mut out, &a, &plan, &tsk, &mut scratch.borrow()).unwrap();
            module.synchronize();
        })
    });
}

/// `ckks_log` of a plan on `[0.5, 1.5]` at default options.
pub fn runner_ckks_log<BE, M>(bencher: &mut Bencher<'_, M>, bp: &LibmBenchParams)
where
    BE: Backend<Ring = Standard> + HostStaged + 'static,
    Module<BE>: LibmBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    timed::<BE, M, _>(bencher, "ckks_log", bp, || {
        let host = Module::<HostBytesBackend>::new(bp.n as u64);
        let plan = LogPlan::from_precision(
            BASE_INTERVAL.0,
            BASE_INTERVAL.1,
            base2k(bp),
            coeff_meta(bp),
            LogOptions::default(),
            &host,
        )
        .unwrap();
        let k = ct_k(bp, plan.consumed_bits(bp.log_delta));
        let Point {
            module,
            tsk,
            a,
            res: mut out,
            ..
        } = Point::<BE>::new(bp, k);
        let plan = plan.map_plaintexts(|pt| upload_pt(&module, pt));
        let mut scratch = ScratchOwned::<BE>::alloc(module.ckks_log_tmp_bytes(&out, &a, &tsk_layout(bp, k), &plan));
        Box::new(move || {
            module.ckks_log(&mut out, &a, &plan, &tsk, &mut scratch.borrow()).unwrap();
            module.synchronize();
        })
    });
}

/// `ckks_pow` of a plan for bases in `[0.5, 1.5]` and exponents in `[-1, 1]` at default options.
pub fn runner_ckks_pow<BE, M>(bencher: &mut Bencher<'_, M>, bp: &LibmBenchParams)
where
    BE: Backend<Ring = Standard> + HostStaged + 'static,
    Module<BE>: LibmBenchModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    M: Measurement,
{
    timed::<BE, M, _>(bencher, "ckks_pow", bp, || {
        let host = Module::<HostBytesBackend>::new(bp.n as u64);
        let plan = PowPlan::from_precision(
            BASE_INTERVAL.0,
            BASE_INTERVAL.1,
            EXPONENT_INTERVAL.0,
            EXPONENT_INTERVAL.1,
            base2k(bp),
            coeff_meta(bp),
            PowOptions::default(),
            &host,
        )
        .unwrap();
        let k = ct_k(bp, plan.consumed_bits(bp.log_delta, bp.log_delta));
        let Point {
            module,
            tsk,
            a,
            b,
            res: mut out,
        } = Point::<BE>::new(bp, k);
        let plan = plan.map_plaintexts(|pt| upload_pt(&module, pt));
        let mut scratch = ScratchOwned::<BE>::alloc(module.ckks_pow_tmp_bytes(&out, &a, &b, &tsk_layout(bp, k), &plan));
        Box::new(move || {
            module.ckks_pow(&mut out, &a, &b, &plan, &tsk, &mut scratch.borrow()).unwrap();
            module.synchronize();
        })
    });
}
