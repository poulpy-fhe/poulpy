//! Thread-count diagnostic for the Rayon backends.
//!
//! Times one primitive per kernel family across pool widths and reports the
//! widths within [`TOLERANCE`] of the best. Backend crates instantiate
//! [`thread_scaling`] for their own types; see their `tuning` tests.
//!
//! Widths are measured round-robin across several rounds, so clock drift and
//! cache warming do not alias onto the thread count. [`Mode::Fast`] takes well
//! under a second per backend; [`Mode::Precise`] uses more rounds.
//!
//! # Scope
//!
//! The probes run the production path, whose worker slices are capped per kernel
//! by [`ScratchWorkers`]. Pool widths above a probe's cap run the same number of
//! slices as the cap, so those curves flatten there. The sweep answers how many
//! threads to give the pool, not whether a kernel would benefit from more slices
//! than its backend allows.

use std::time::Instant;

use poulpy_hal::{
    api::{
        CnvPVecAlloc, Convolution, MatZnxAlloc, ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddIntoBackend,
        VecZnxAlloc, VecZnxBigAlloc, VecZnxDftAlloc, VecZnxDftApply, VecZnxIdftApply, VecZnxIdftApplyTmpBytes, VmpApplyDftToDft,
        VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPrepare, VmpPrepareTmpBytes,
    },
    execution::ScratchWorkers,
    layouts::{
        Backend, CnvPVecLOwned, CnvPVecLToBackendMut, CnvPVecLToBackendRef, CnvPVecROwned, CnvPVecRToBackendMut,
        CnvPVecRToBackendRef, FillUniform, MatZnx, MatZnxToBackendRef, Module, ScratchOwned, VecZnx, VecZnxBigOwned,
        VecZnxBigToBackendMut, VecZnxDftOwned, VecZnxDftToBackendMut, VecZnxDftToBackendRef, VecZnxToBackendMut,
        VecZnxToBackendRef, VmpPMatOwned, VmpPMatToBackendMut, VmpPMatToBackendRef,
    },
    source::Source,
};

/// A pool width within this fraction of the best time is treated as equivalent
/// to it; the smallest such width is the recommendation.
pub const TOLERANCE: f64 = 0.05;

/// How much effort the sweep spends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// A handful of rounds: sub-second, enough for a clear knee on an idle machine.
    Fast,
    /// More rounds and repetitions, for a noisy machine or a close call.
    Precise,
}

impl Mode {
    /// (rounds, timed repetitions per round).
    pub fn rounds_and_reps(self) -> (usize, usize) {
        match self {
            Mode::Fast => (3, 2),
            Mode::Precise => (10, 5),
        }
    }
}

/// Repetition spread above which the machine is considered too busy to trust.
pub const NOISE: f64 = 0.10;

/// One point of a [`thread_scaling`] sweep.
#[derive(Debug, Clone, Copy)]
pub struct ScalingPoint {
    pub threads: usize,
    /// Fastest repetition, in milliseconds.
    pub millis: f64,
    /// Second-fastest repetition over the fastest one. Well above 1 means the
    /// fast samples disagree and the sweep should not be trusted.
    pub spread: f64,
}

/// The outcome of a [`thread_scaling`] sweep.
#[derive(Debug, Clone)]
pub struct ScalingReport {
    pub points: Vec<ScalingPoint>,
    /// Smallest width within [`TOLERANCE`] of the best.
    pub recommended: usize,
    /// Every width within [`TOLERANCE`] of the best: widths the data cannot
    /// separate. A wide band means the knee is flat, not that it is precise.
    pub band: Vec<usize>,
}

impl ScalingReport {
    fn new(mut points: Vec<ScalingPoint>) -> Self {
        assert!(!points.is_empty(), "thread_scaling needs at least one pool width");
        points.sort_by_key(|p| p.threads);
        let best = points.iter().map(|p| p.millis).fold(f64::INFINITY, f64::min);
        let mut band: Vec<usize> = points
            .iter()
            .filter(|p| p.millis <= best * (1.0 + TOLERANCE))
            .map(|p| p.threads)
            .collect();
        band.sort_unstable();
        let recommended = band.first().copied().unwrap_or(1);
        Self {
            points,
            recommended,
            band,
        }
    }

    /// Whether any point varied enough between repetitions to distrust the sweep.
    pub fn noisy(&self) -> bool {
        self.points.iter().any(|p| p.spread > 1.0 + NOISE)
    }

    /// Speed-up of each point over the narrowest pool measured.
    pub fn speedups(&self) -> Vec<f64> {
        let base = self
            .points
            .iter()
            .min_by_key(|p| p.threads)
            .map(|p| p.millis)
            .unwrap_or(f64::NAN);
        self.points.iter().map(|p| base / p.millis).collect()
    }

    pub fn print(&self, label: &str) {
        println!("{label}");
        for (point, speedup) in self.points.iter().zip(self.speedups()) {
            println!(
                "  threads={:<4} {:9.3} ms   {speedup:5.2}x   spread {:.2}",
                point.threads, point.millis, point.spread
            );
        }
        if self.band.len() > 1 {
            let band: Vec<String> = self.band.iter().map(|t| t.to_string()).collect();
            println!(
                "  recommended: {} threads (within noise of {})",
                self.recommended,
                band.join(", ")
            );
        } else {
            println!("  recommended: {} threads", self.recommended);
        }
        if self.noisy() {
            println!(
                "  warning: repetitions vary by more than {:.0}%, rerun on an idle machine",
                NOISE * 100.0
            );
        }
    }
}

/// The operand shape the probes are built at.
///
/// These are HAL quantities. A caller tuning for GLWE work passes `cols` as
/// `rank + 1`, and `rows` as the gadget row count of the key it will use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProbeShape {
    /// Ring degree.
    pub n: usize,
    /// Limbs per column.
    pub size: usize,
    /// Columns of the operands.
    pub cols: usize,
    /// Rows of the prepared matrix the vector-matrix probe multiplies by.
    pub rows: usize,
}

impl ProbeShape {
    /// A shape whose matrix is square in limbs, the usual key-switch case.
    pub fn square(n: usize, size: usize, cols: usize) -> Self {
        Self {
            n,
            size,
            cols,
            rows: size,
        }
    }
}

/// The primitives the sweep times, one per kernel family whose scaling differs.
pub const PROBES: [&str; 4] = ["vmp", "convolution", "idft", "coefficient"];

/// Worker-slice cap each probe runs under, or `None` when the kernel is not
/// scratch-backed and the pool width is its only limit.
fn probe_cap<BE: ScratchWorkers>(probe: usize) -> Option<usize> {
    match probe {
        0 => Some(BE::VMP),
        1 => Some(BE::APPLY),
        2 => Some(BE::IDFT),
        _ => None,
    }
}

/// Timed samples shorter than this are dominated by measurement noise, so each
/// sample batches enough calls to exceed it.
const TARGET_SAMPLE_MS: f64 = 1.0;

/// One [`ScalingReport`] per probe, plus the width that serves them all.
#[derive(Debug, Clone)]
pub struct TuningReport {
    pub probes: Vec<(&'static str, ScalingReport)>,
    /// Cap each probe ran under, in the same order.
    caps: Vec<Option<usize>>,
    /// Width minimizing the worst relative loss across probes: the one no kernel
    /// is badly hurt by, which is not in general the best width for any of them.
    pub recommended: usize,
}

impl TuningReport {
    fn new(probes: Vec<(&'static str, ScalingReport)>, caps: Vec<Option<usize>>) -> Self {
        let trusted: Vec<&ScalingReport> = probes.iter().map(|(_, r)| r).filter(|r| !r.noisy()).collect();
        let voting: Vec<&ScalingReport> = if trusted.is_empty() {
            probes.iter().map(|(_, r)| r).collect()
        } else {
            trusted
        };

        let widths: Vec<usize> = voting
            .first()
            .map(|r| r.points.iter().map(|p| p.threads).collect())
            .unwrap_or_default();
        let recommended = widths
            .iter()
            .map(|&width| {
                let loss = voting
                    .iter()
                    .filter_map(|r| {
                        let best = r.points.iter().map(|p| p.millis).fold(f64::INFINITY, f64::min);
                        r.points.iter().find(|p| p.threads == width).map(|p| p.millis / best)
                    })
                    .fold(1.0f64, f64::max);
                (width, loss)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
            .map(|(width, _)| width)
            .unwrap_or(1);
        Self {
            probes,
            caps,
            recommended,
        }
    }

    pub fn print(&self, label: &str) {
        println!("{label}");
        for (probe, (name, report)) in self.probes.iter().enumerate() {
            match self.caps.get(probe).copied().flatten() {
                Some(cap) => report.print(&format!("{name} (worker slices capped at {cap})")),
                None => report.print(name),
            }
        }
        println!("  => {} threads: the width no probe is badly hurt by", self.recommended);
        if self.probes.iter().any(|(_, r)| r.noisy()) {
            println!("  (noisy probes were excluded from that answer)");
        }
    }
}

/// Times one primitive per kernel family at each pool width in `threads`.
///
/// Pass the shape the application will actually use: the knee moves with it.
pub fn thread_scaling<BE>(shape: ProbeShape, threads: &[usize], mode: Mode) -> TuningReport
where
    BE: Backend<ZnxWord = i64, OwnedBuf: poulpy_hal::layouts::HostDataMut> + ScratchWorkers + 'static,
    Module<BE>: ModuleNew<BE>
        + VecZnxAlloc<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxBigAlloc<BE>
        + MatZnxAlloc<BE>
        + VmpPMatAlloc<BE>
        + CnvPVecAlloc<BE>
        + VmpPrepare<BE>
        + VmpPrepareTmpBytes
        + VecZnxDftApply<BE>
        + VecZnxAddIntoBackend<BE>
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + Convolution<BE>
        + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    validate_widths(threads);

    let ProbeShape { n, size, cols, rows } = shape;
    let module: Module<BE> = Module::<BE>::new(n as u64);
    let cnv_size = 2 * size - 1;
    let mut source: Source = Source::new([0u8; 32]);

    let mut small: VecZnx<BE::OwnedBuf, i64> = module.vec_znx_alloc(cols, size);
    small.fill_uniform(16, &mut source);
    let mut addend: VecZnx<BE::OwnedBuf, i64> = module.vec_znx_alloc(cols, size);
    addend.fill_uniform(16, &mut source);
    let mut sum: VecZnx<BE::OwnedBuf, i64> = module.vec_znx_alloc(cols, size);

    let mut mat: MatZnx<BE::OwnedBuf, i64> = module.mat_znx_alloc(rows, cols, cols, size);
    mat.fill_uniform(16, &mut source);

    let mut pmat: VmpPMatOwned<BE> = module.vmp_pmat_alloc(rows, cols, cols, size);
    let mut a: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(cols, size);
    let mut res: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(cols, size);
    let mut big: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(cols, size);
    let mut left: CnvPVecLOwned<BE> = module.cnv_pvec_left_alloc(cols, size);
    let mut right: CnvPVecROwned<BE> = module.cnv_pvec_right_alloc(cols, size);
    let mut cnv_res: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(cols, cnv_size);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .vmp_prepare_tmp_bytes(size, cols, cols, size)
            .max(module.vmp_apply_dft_to_dft_tmp_bytes(size, size, rows, cols, cols, size))
            .max(module.vec_znx_idft_apply_tmp_bytes())
            .max(module.cnv_prepare_self_tmp_bytes(cnv_size, size))
            .max(module.cnv_apply_dft_tmp_bytes(0, cnv_size, size, size)),
    );

    module.vmp_prepare(
        &mut pmat.to_backend_mut(),
        &MatZnxToBackendRef::<BE>::to_backend_ref(&mat),
        &mut scratch.borrow(),
    );
    for col in 0..cols {
        module.vec_znx_dft_apply(
            1,
            0,
            &mut a.to_backend_mut(),
            col,
            &VecZnxToBackendRef::<BE>::to_backend_ref(&small),
            col,
        );
    }
    module.cnv_prepare_self(
        &mut left.to_backend_mut(),
        &mut right.to_backend_mut(),
        &VecZnxToBackendRef::<BE>::to_backend_ref(&small),
        !0i64,
        &mut scratch.borrow(),
    );

    let (rounds, reps) = mode.rounds_and_reps();
    let pools: Vec<::rayon::ThreadPool> = threads
        .iter()
        .map(|&count| ::rayon::ThreadPoolBuilder::new().num_threads(count).build().unwrap())
        .collect();

    let mut best = vec![vec![f64::INFINITY; threads.len()]; PROBES.len()];
    let mut second = vec![vec![f64::INFINITY; threads.len()]; PROBES.len()];

    // Short kernels need several calls per timed sample; calibrate that count once.
    let mut batch = vec![1usize; PROBES.len()];
    for (probe, count) in batch.iter_mut().enumerate() {
        pools[0].install(|| {
            let start = Instant::now();
            run_probe(
                probe,
                &module,
                &mut res,
                &a,
                &pmat,
                &mut cnv_res,
                &left,
                &right,
                &mut big,
                &mut sum,
                &small,
                &addend,
                &mut scratch,
            );
            let elapsed = start.elapsed().as_secs_f64() * 1e3;
            *count = if elapsed > 0.0 {
                (TARGET_SAMPLE_MS / elapsed).ceil() as usize
            } else {
                1
            }
            .clamp(1, 1024);
        });
    }

    for round in 0..rounds {
        for (probe, _) in PROBES.iter().enumerate() {
            for (index, pool) in pools.iter().enumerate() {
                pool.install(|| {
                    let mut apply = || {
                        run_probe(
                            probe,
                            &module,
                            &mut res,
                            &a,
                            &pmat,
                            &mut cnv_res,
                            &left,
                            &right,
                            &mut big,
                            &mut sum,
                            &small,
                            &addend,
                            &mut scratch,
                        )
                    };
                    if round == 0 {
                        apply();
                    }
                    for _ in 0..reps.max(1) {
                        let start = Instant::now();
                        for _ in 0..batch[probe] {
                            apply();
                        }
                        let elapsed = start.elapsed().as_secs_f64() * 1e3 / batch[probe] as f64;
                        if elapsed < best[probe][index] {
                            second[probe][index] = best[probe][index];
                            best[probe][index] = elapsed;
                        } else if elapsed < second[probe][index] {
                            second[probe][index] = elapsed;
                        }
                    }
                });
            }
        }
    }

    let reports: Vec<(&'static str, ScalingReport)> = PROBES
        .iter()
        .enumerate()
        .map(|(probe, name)| {
            let points: Vec<ScalingPoint> = threads
                .iter()
                .enumerate()
                .map(|(index, &count)| ScalingPoint {
                    threads: count,
                    millis: best[probe][index],
                    spread: if second[probe][index].is_finite() {
                        second[probe][index] / best[probe][index]
                    } else {
                        1.0
                    },
                })
                .collect();
            (*name, ScalingReport::new(points))
        })
        .collect();
    let caps = (0..PROBES.len()).map(probe_cap::<BE>).collect();
    TuningReport::new(reports, caps)
}

/// Runs one call of the probe identified by `probe`.
#[allow(clippy::too_many_arguments)]
fn run_probe<BE>(
    probe: usize,
    module: &Module<BE>,
    res: &mut VecZnxDftOwned<BE>,
    a: &VecZnxDftOwned<BE>,
    pmat: &VmpPMatOwned<BE>,
    cnv_res: &mut VecZnxDftOwned<BE>,
    left: &CnvPVecLOwned<BE>,
    right: &CnvPVecROwned<BE>,
    big: &mut VecZnxBigOwned<BE>,
    sum: &mut VecZnx<BE::OwnedBuf, i64>,
    small: &VecZnx<BE::OwnedBuf, i64>,
    addend: &VecZnx<BE::OwnedBuf, i64>,
    scratch: &mut ScratchOwned<BE>,
) where
    BE: Backend<ZnxWord = i64, OwnedBuf: poulpy_hal::layouts::HostDataMut> + 'static,
    Module<BE>: VecZnxAddIntoBackend<BE> + VecZnxIdftApply<BE> + Convolution<BE> + VmpApplyDftToDft<BE>,
    ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
{
    match probe {
        0 => module.vmp_apply_dft_to_dft(
            &mut res.to_backend_mut(),
            &a.to_backend_ref(),
            &pmat.to_backend_ref(),
            0,
            &mut scratch.borrow(),
        ),
        1 => module.cnv_apply_dft(
            0,
            &mut cnv_res.to_backend_mut(),
            0,
            &left.to_backend_ref(),
            0,
            &right.to_backend_ref(),
            0,
            &mut scratch.borrow(),
        ),
        2 => module.vec_znx_idft_apply(&mut big.to_backend_mut(), 0, &a.to_backend_ref(), 0, &mut scratch.borrow()),
        _ => module.vec_znx_add_into_backend(
            &mut poulpy_hal::oep::SetNormalizationState::set_unnormalized(VecZnxToBackendMut::<BE>::to_backend_mut(sum)),
            0,
            &VecZnxToBackendRef::<BE>::to_backend_ref(small),
            0,
            &VecZnxToBackendRef::<BE>::to_backend_ref(addend),
            0,
        ),
    }
}

/// Rejects sweeps that would produce a meaningless report.
///
/// Rayon reads `num_threads(0)` as "pick a width for me", so a zero would be
/// measured on every core and reported as a zero-thread point.
fn validate_widths(threads: &[usize]) {
    assert!(!threads.is_empty(), "thread_scaling needs at least one pool width");
    assert!(
        threads.iter().all(|&count| count > 0),
        "pool widths must be non-zero, got {threads:?}"
    );
}

/// Pool widths to sweep on this machine: powers of two up to the pool size.
pub fn default_thread_sweep() -> Vec<usize> {
    let max = rayon::current_num_threads().max(1);
    let mut widths: Vec<usize> = (0..).map(|e| 1usize << e).take_while(|&w| w < max).collect();
    widths.push(max);
    widths
}

#[cfg(test)]
mod tests {
    use super::{default_thread_sweep, validate_widths};

    #[test]
    fn the_default_sweep_is_valid() {
        validate_widths(&default_thread_sweep());
    }

    #[test]
    #[should_panic(expected = "at least one pool width")]
    fn an_empty_sweep_is_rejected() {
        validate_widths(&[]);
    }

    #[test]
    #[should_panic(expected = "non-zero")]
    fn a_zero_width_is_rejected() {
        validate_widths(&[1, 0, 4]);
    }
}
