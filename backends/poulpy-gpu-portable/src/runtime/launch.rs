//! What every kernel launch shares: the binding of a view, dispatch sizing,
//! the launch counters and the launch trace.
//!
//! Every launch goes through [`launch!`](crate::runtime::launch), which
//! names its kernel with a [`Label`], counts it, and, while a trace is open
//! on the thread, records what identifies it: the label, the dispatch, the
//! buffers bound, the scalar arguments and the compile-time arguments.

use std::{
    cell::RefCell,
    collections::HashMap,
    hash::{DefaultHasher, Hash, Hasher},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
};

use cubecl::{prelude::*, server::Handle};

/// A view as a kernel sees it: the whole allocation, and where the view starts.
#[derive(Clone)]
pub(crate) struct Binding {
    /// Identity of the allocation.
    pub(crate) id: u64,
    pub(crate) handle: Handle,
    /// Bytes reserved by the allocation.
    pub(crate) reserved: usize,
    /// Byte offset of the view in the allocation.
    pub(crate) offset: usize,
    /// Bytes of the view.
    pub(crate) len: usize,
    /// Whether the buffer that owns the allocation is alive.
    pub(crate) owned: Arc<AtomicBool>,
}

/// An allocation a traced body bound, kept by the trace.
pub(crate) struct Bound {
    /// Keeps the allocation reserved for as long as the trace's graph lives.
    pub(crate) _handle: Handle,
    /// Bytes reserved by the allocation.
    pub(crate) reserved: usize,
    /// Whether the buffer that owns the allocation is alive.
    pub(crate) owned: Arc<AtomicBool>,
    /// Whether the body allocated it, as opposed to receiving it.
    pub(crate) temporary: bool,
}

impl Binding {
    /// The allocation as a slice of `T`.
    pub(crate) fn arg<T>(&self) -> BufferArg {
        // SAFETY: the handle spans `reserved` bytes, a whole number of `T`.
        unsafe { BufferArg::from_raw_parts(self.handle.clone(), self.reserved / size_of::<T>()) }
    }

    /// Index of the view's first element in [`Self::arg`].
    pub(crate) fn base<T>(&self) -> u32 {
        debug_assert!(self.offset.is_multiple_of(size_of::<T>()));
        u32::try_from(self.offset / size_of::<T>()).expect("device view starts beyond the 32-bit element range")
    }
}

/// The launch arguments of the buffers a kernel takes, in signature order.
///
/// CubeCL promises its GPU compilers that distinct buffer arguments do not
/// overlap. Two views of one allocation that may overlap (an operation
/// applied in place) are therefore passed as one buffer: the later argument
/// is declared an alias of the earlier one, which CubeCL compiles as the
/// same pointer. Views of one allocation that are disjoint, or bound with
/// different word types (temporaries carved from one scratch arena), stay
/// separate arguments.
pub(crate) struct BufferArgs<const N: usize> {
    /// Per argument: the rank, among the buffers bound, of the one it aliases, if any.
    alias: [Option<usize>; N],
    next: usize,
}

impl<const N: usize> BufferArgs<N> {
    /// `bindings` pairs each view with the word type it is bound as.
    pub(crate) fn new(bindings: [(&Binding, std::any::TypeId); N]) -> Self {
        let mut alias = [None; N];
        for later in 1..N {
            let (b, b_word) = bindings[later];
            alias[later] = (0..later).find_map(|earlier| {
                let (a, a_word) = bindings[earlier];
                let overlap = a.id == b.id && a.offset < b.offset + b.len && b.offset < a.offset + a.len;
                // Aliasing is transitive through the argument an earlier one already aliases.
                (overlap && a_word == b_word).then(|| alias[earlier].unwrap_or(earlier))
            });
        }
        // CubeCL names the aliased buffer by its rank among the buffers
        // actually bound, which the aliases are not part of.
        let rank = |position: usize| alias[..position].iter().filter(|earlier| earlier.is_none()).count();
        Self {
            alias: std::array::from_fn(|position| alias[position].map(rank)),
            next: 0,
        }
    }

    /// The argument of the next buffer of the signature, which is `binding` bound as `T`.
    pub(crate) fn next<T>(&mut self, binding: &Binding) -> BufferArg {
        let position = self.next;
        self.next += 1;
        match self.alias[position] {
            Some(earlier) => BufferArg::alias(earlier, binding.reserved / size_of::<T>()),
            None => binding.arg::<T>(),
        }
    }
}

/// Dispatch covering `units` threads in groups of `dim`.
pub(crate) fn cubes(client: &Client, units: usize, dim: u32) -> CubeCount {
    // The kernels compare their thread position with their unit count in 32 bits.
    assert!(u32::try_from(units).is_ok(), "dispatch exceeds the 32-bit thread range");
    let groups = u32::try_from(units.div_ceil(dim as usize)).expect("dispatch exceeds the 32-bit group range");
    cubecl::server::CubeCountSelection::new(client, groups.max(1)).cube_count()
}

macro_rules! labels {
    ($($(#[$attr:meta])* $name:ident => $text:literal),* $(,)?) => {
        /// The kernel a launch runs, as the accounting and the trace name it.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[repr(usize)]
        #[allow(non_camel_case_types)]
        pub(crate) enum Label {
            $($(#[$attr])* $name),*
        }

        const LABELS: &[&str] = &[$($(#[$attr])* $text),*];
    };
}

labels! {
    BufferZero => "buffer.zero",
    NttForwardSingle => "ntt.forward.single",
    NttForwardStrided => "ntt.forward.strided",
    NttForwardContiguous => "ntt.forward.contiguous",
    NttInverseSingle => "ntt.inverse.single",
    NttInverseContiguous => "ntt.inverse.contiguous",
    NttInverseStrided => "ntt.inverse.strided",
    CrtToBig => "crt.to_big",
    CrtToBigClearing => "crt.to_big_clearing",
    NormalizeLimbs => "normalize.limbs",
    NormalizeBig => "normalize.big",
    NormalizeResidues => "normalize.residues",
    #[cfg(feature = "enable-ckks")]
    ShapingPreimage => "ckks.shaping_preimage",
    #[cfg(feature = "enable-ckks")]
    ShapingCorrection => "ckks.shaping_correction",
    GadgetDecompose => "znx.gadget_decompose",
    VmpDigitProduct => "vmp.digit_product",
    VmpDigitProductAdd => "vmp.digit_product_add",
    VmpExtractRows => "vmp.extract_rows",
    SvpScale => "svp.scale",
    CnvSingle => "cnv.single",
    CnvSingleAdd => "cnv.single_add",
    CnvSum => "cnv.sum",
    CnvSumAdd => "cnv.sum_add",
    CnvScale => "cnv.scale",
    CnvScaleAdd => "cnv.scale_add",
    CnvPairwise => "cnv.pairwise",
    CnvTensor => "cnv.tensor",
    CnvByConst => "cnv.by_const",
    CnvByConstAdd => "cnv.by_const_add",
    DftZero => "dft.zero",
    DftCopy => "dft.copy",
    DftCombine => "dft.combine",
    DftAutomorphism => "dft.automorphism",
    DftAutomorphismAdd => "dft.automorphism_add",
    ZnxZero => "znx.zero",
    ZnxCopy => "znx.copy",
    ZnxCombine => "znx.combine",
    BigZero => "big.zero",
    BigCopy => "big.copy",
    BigCombine => "big.combine",
    ZnxPermute => "znx.permute",
    ZnxMove => "znx.move",
    BigPermute => "big.permute",
    ZnxRingMap => "znx.ci_map",
    BigScalarProduct => "big.scalar_product",
    BigColWeightedSum => "big.col_weighted_sum",
    BigInnerSum => "big.inner_sum",
    SampleUniform => "sample.uniform",
    SampleTable => "sample.table",
    SampleRejection => "sample.rejection",
    SampleSecret => "sample.secret",
}

impl Label {
    pub(crate) fn name(self) -> &'static str {
        LABELS[self as usize]
    }
}

struct Totals {
    launches: AtomicU64,
    bytes: AtomicU64,
}

static TOTALS: [Totals; LABELS.len()] = [const {
    Totals {
        launches: AtomicU64::new(0),
        bytes: AtomicU64::new(0),
    }
}; LABELS.len()];
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static REPLAYS: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
#[allow(dead_code, reason = "read by the tests of an enabled runtime")]
/// Kernel launches, the bytes they were sized to read plus write, the
/// device allocations requested from CubeCL, and the graph replays.
///
/// A graph replay redoes the launches of a recorded call without issuing
/// them: it adds nothing to `launches`, `bytes` and `allocations`, and one
/// to `replays`. A reader that sees few launches must look at `replays`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LaunchCounters {
    pub launches: u64,
    pub bytes: u64,
    pub allocations: u64,
    pub replays: u64,
}

#[cfg(test)]
#[allow(dead_code, reason = "read by the tests of an enabled runtime")]
/// The launches of one kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelCounters {
    pub label: &'static str,
    pub launches: u64,
    pub bytes: u64,
}

#[cfg(test)]
#[allow(dead_code, reason = "read by the tests of an enabled runtime")]
/// Launches issued by every backend type of this crate since the last reset.
pub fn launch_counters() -> LaunchCounters {
    let mut counters = LaunchCounters {
        allocations: ALLOCATIONS.load(Ordering::Relaxed),
        replays: REPLAYS.load(Ordering::Relaxed),
        ..Default::default()
    };
    for totals in &TOTALS {
        counters.launches += totals.launches.load(Ordering::Relaxed);
        counters.bytes += totals.bytes.load(Ordering::Relaxed);
    }
    counters
}

#[cfg(test)]
#[allow(dead_code, reason = "read by the tests of an enabled runtime")]
/// [`launch_counters`] per kernel, for the kernels launched at least once,
/// largest byte count first.
pub fn launch_breakdown() -> Vec<KernelCounters> {
    let mut breakdown: Vec<KernelCounters> = TOTALS
        .iter()
        .zip(LABELS)
        .map(|(totals, &label)| KernelCounters {
            label,
            launches: totals.launches.load(Ordering::Relaxed),
            bytes: totals.bytes.load(Ordering::Relaxed),
        })
        .filter(|kernel| kernel.launches != 0)
        .collect();
    breakdown.sort_by(|a, b| b.bytes.cmp(&a.bytes).then(a.label.cmp(b.label)));
    breakdown
}

/// Time per kernel, for the tests that profile: with profiling on, every
/// launch is preceded and followed by a device synchronization, and the time
/// between the two is charged to its kernel.
#[cfg(test)]
pub(crate) mod profile {
    // Read by the full-size test of the CKKS layer.
    #![cfg_attr(not(feature = "enable-ckks"), allow(dead_code))]

    use std::{
        sync::atomic::{AtomicBool, AtomicU64, Ordering},
        time::Instant,
    };

    use super::{LABELS, Label};
    use crate::runtime::GpuRuntime;

    static ON: AtomicBool = AtomicBool::new(false);
    static NANOS: [AtomicU64; LABELS.len()] = [const { AtomicU64::new(0) }; LABELS.len()];

    fn sync<RT: GpuRuntime>() {
        pollster::block_on(RT::client().sync()).unwrap_or_else(|error| panic!("device synchronization failed: {error:?}"));
    }

    /// Turns profiling on or off and clears the times.
    pub(crate) fn set(on: bool) {
        ON.store(on, Ordering::Relaxed);
        for nanos in &NANOS {
            nanos.store(0, Ordering::Relaxed);
        }
    }

    pub(crate) fn start<RT: GpuRuntime>() -> Option<Instant> {
        ON.load(Ordering::Relaxed).then(|| {
            sync::<RT>();
            Instant::now()
        })
    }

    pub(crate) fn end<RT: GpuRuntime>(label: Label, start: Option<Instant>) {
        if let Some(start) = start {
            sync::<RT>();
            NANOS[label as usize].fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }

    /// Seconds charged to each kernel launched since profiling was turned on.
    pub(crate) fn seconds() -> Vec<(&'static str, f64)> {
        LABELS
            .iter()
            .zip(&NANOS)
            .map(|(&label, nanos)| (label, nanos.load(Ordering::Relaxed) as f64 * 1e-9))
            .filter(|(_, seconds)| *seconds != 0.0)
            .collect()
    }
}

#[cfg(test)]
#[allow(dead_code, reason = "read by the tests of an enabled runtime")]
/// Resets [`launch_counters`] and [`launch_breakdown`].
pub fn reset_launch_counters() {
    for totals in &TOTALS {
        totals.launches.store(0, Ordering::Relaxed);
        totals.bytes.store(0, Ordering::Relaxed);
    }
    ALLOCATIONS.store(0, Ordering::Relaxed);
    REPLAYS.store(0, Ordering::Relaxed);
}

/// What a body did to the device besides launching kernels. None of it can
/// be part of a recorded graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HostEvents {
    /// Device buffers read back.
    pub reads: u32,
    /// Host bytes written to device buffers.
    pub writes: u32,
    /// Device synchronizations.
    pub syncs: u32,
    /// Device allocations requested from CubeCL.
    pub allocations: u32,
}

impl HostEvents {
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// A host-side action on the device.
#[derive(Clone, Copy)]
pub(crate) enum HostEvent {
    Read,
    Write,
    Sync,
    Allocation,
}

/// The launches of a body, one digest each, and its host events.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct Trace {
    pub(crate) launches: Vec<(Label, u64)>,
    pub(crate) host: HostEvents,
}

impl Trace {
    /// Where `self` and `other` first differ, as a message.
    pub(crate) fn difference(&self, other: &Self) -> Option<String> {
        if self.host != other.host {
            return Some(format!("host events {:?} against {:?}", self.host, other.host));
        }
        let position = self.launches.iter().zip(&other.launches).position(|(a, b)| a != b);
        match position {
            Some(index) => Some(format!(
                "launch {index} of {}: {} against {}",
                self.launches.len(),
                self.launches[index].0.name(),
                other.launches[index].0.name()
            )),
            None if self.launches.len() != other.launches.len() => {
                Some(format!("{} launches against {}", self.launches.len(), other.launches.len()))
            }
            None => None,
        }
    }
}

/// A trace being taken.
#[derive(Default)]
struct OpenTrace {
    trace: Trace,
    /// Allocations made since the trace opened, in order. A launch binding
    /// one is identified by its position here: two runs of a body allocate
    /// their temporaries anew, in the same order.
    allocated: Vec<u64>,
    /// The handles of every allocation bound so far, when the trace keeps them.
    retained: Option<HashMap<u64, Bound>>,
}

/// Threads with an open trace: the per-launch test is one load when there is none.
static TRACING: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static TRACE: RefCell<Option<OpenTrace>> = const { RefCell::new(None) };
}

/// A trace open on this thread, closed by [`Tracer::finish`] or on drop.
pub(crate) struct Tracer(());

impl Tracer {
    /// Opens a trace on this thread; with `retain` it keeps a handle of every
    /// allocation the launches bind. Panics if one is open: traces do not nest.
    pub(crate) fn start(retain: bool) -> Self {
        TRACE.with_borrow_mut(|open| {
            assert!(open.is_none(), "a launch trace is already open on this thread");
            *open = Some(OpenTrace {
                retained: retain.then(HashMap::new),
                ..Default::default()
            });
        });
        TRACING.fetch_add(1, Ordering::Relaxed);
        Self(())
    }

    /// Closes the trace and returns it with the allocations it kept.
    pub(crate) fn finish(self) -> (Trace, Vec<Bound>) {
        let open = TRACE
            .with_borrow_mut(Option::take)
            .expect("the trace is open until it is finished");
        (
            open.trace,
            open.retained
                .map_or_else(Vec::new, |retained| retained.into_values().collect()),
        )
    }
}

impl Drop for Tracer {
    fn drop(&mut self) {
        // Also reached when the traced body panics: the thread must not keep a stale trace.
        TRACE.with_borrow_mut(|open| *open = None);
        TRACING.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Whether a trace is open on some thread: the test a launch makes before
/// it gathers what identifies it.
pub(crate) fn tracing() -> bool {
    TRACING.load(Ordering::Relaxed) != 0
}

/// Counts one graph replay.
pub(crate) fn replayed() {
    REPLAYS.fetch_add(1, Ordering::Relaxed);
}

/// Counts one launch of `label` moving `bytes` of device memory.
pub(crate) fn launched(label: Label, bytes: usize) {
    let totals = &TOTALS[label as usize];
    totals.launches.fetch_add(1, Ordering::Relaxed);
    totals.bytes.fetch_add(bytes as u64, Ordering::Relaxed);
}

/// Adds a launch to the trace open on this thread, if any: its label and a
/// digest of its dispatch, the allocations bound, the scalar arguments and
/// the compile-time arguments.
pub(crate) fn trace_launch(label: Label, units: usize, dim: u32, bindings: &[&Binding], scalars: &[u64], comptime: &impl Hash) {
    TRACE.with_borrow_mut(|open| {
        let Some(open) = open else { return };
        let mut hasher = DefaultHasher::new();
        (units, dim, scalars).hash(&mut hasher);
        comptime.hash(&mut hasher);
        for binding in bindings {
            let position = open.allocated.iter().position(|&id| id == binding.id);
            match position {
                Some(position) => (true, position as u64).hash(&mut hasher),
                None => (false, binding.id).hash(&mut hasher),
            }
            if let Some(retained) = &mut open.retained {
                retained.entry(binding.id).or_insert_with(|| Bound {
                    _handle: binding.handle.clone(),
                    reserved: binding.reserved,
                    owned: binding.owned.clone(),
                    temporary: position.is_some(),
                });
            }
        }
        open.trace.launches.push((label, hasher.finish()));
    });
}

/// Records a host-side action on the device; `id` is the allocation concerned.
pub(crate) fn host_event(event: HostEvent, id: u64) {
    if matches!(event, HostEvent::Allocation) {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    }
    if tracing() {
        TRACE.with_borrow_mut(|open| {
            let Some(open) = open else { return };
            match event {
                HostEvent::Read => open.trace.host.reads += 1,
                HostEvent::Write => open.trace.host.writes += 1,
                HostEvent::Sync => open.trace.host.syncs += 1,
                HostEvent::Allocation => {
                    open.trace.host.allocations += 1;
                    open.allocated.push(id);
                }
            }
        });
    }
}

/// Launches `$kernel` on the runtime `$rt` over `$units` units in groups of
/// `$per`, with workgroups of `$dim` threads, and accounts for it under
/// `$label` as moving `$bytes`. The kernel takes its buffers first, then its
/// scalars, then its compile-time arguments; each is written once here and
/// reaches both the kernel and the trace.
macro_rules! launch {
    (
        $rt:ty, $label:expr, $bytes:expr, $kernel:ident, ($units:expr, $per:expr, $dim:expr);
        [$($binding:expr => $word:ty),* $(,)?];
        [$($scalar:expr),* $(,)?];
        [$($comptime:expr),* $(,)?]
    ) => {{
        let client = <$rt as $crate::runtime::RuntimeDevice>::client();
        let (units, per, dim): (usize, u32, u32) = ($units, $per, $dim);
        let _access = $crate::runtime::access::shared();
        #[cfg(test)]
        let profiled = $crate::runtime::profile::start::<$rt>();
        #[allow(unused_mut)]
        let mut buffers = $crate::runtime::BufferArgs::new([$((&$binding, std::any::TypeId::of::<$word>())),*]);
        // SAFETY: each launcher checks, before it launches, that the indices
        // its kernel forms from the scalar arguments lie inside the bound
        // views; views that may overlap are passed as one buffer.
        unsafe {
            $kernel::launch_unchecked(
                client,
                $crate::runtime::cubes(client, units, per),
                cubecl::prelude::CubeDim::new_1d(dim),
                $(buffers.next::<$word>(&$binding),)*
                $($scalar,)*
                $($comptime,)*
            );
        }
        #[cfg(test)]
        $crate::runtime::profile::end::<$rt>($label, profiled);
        $crate::runtime::launched($label, $bytes);
        if $crate::runtime::tracing() {
            $crate::runtime::trace_launch($label, units, dim, &[$(&$binding),*], &[$($scalar as u64),*], &($($comptime,)*));
        }
    }};
}
pub(crate) use launch;

// The test needs a device handle to build bindings from.
#[cfg(all(
    test,
    any(
        feature = "enable-cuda",
        feature = "enable-hip",
        feature = "enable-metal",
        feature = "enable-vulkan",
        feature = "enable-wgpu"
    )
))]
mod tests {
    use std::any::TypeId;

    use super::*;
    use crate::runtime::RuntimeDevice;

    /// The aliases `BufferArgs` declares for views `(allocation, offset, len)` bound as `u32`.
    fn aliases<const N: usize>(views: [(u64, usize, usize); N], handle: &Handle) -> [Option<usize>; N] {
        let bindings = views.map(|(id, offset, len)| Binding {
            id,
            handle: handle.clone(),
            reserved: 1024,
            offset,
            len,
            owned: Arc::new(AtomicBool::new(true)),
        });
        BufferArgs::new(std::array::from_fn(|i| (&bindings[i], TypeId::of::<u32>()))).alias
    }

    #[test]
    fn overlapping_views_alias_the_first_by_its_rank_among_bound_buffers() {
        let handle = crate::tests::Runtime::client().empty(1024);
        // Distinct allocations and disjoint views of one allocation stay separate.
        assert_eq!(aliases([(0, 0, 64), (1, 0, 64), (0, 64, 64)], &handle), [None, None, None]);
        // An operation in place.
        assert_eq!(aliases([(0, 0, 64), (0, 0, 64), (1, 0, 64)], &handle), [None, Some(0), None]);
        // Aliases are not bound: the fifth argument is the third buffer, of rank 2.
        assert_eq!(
            aliases(
                [(0, 0, 64), (1, 0, 64), (1, 0, 64), (1, 32, 64), (2, 0, 64), (2, 0, 64)],
                &handle
            ),
            [None, None, Some(1), Some(1), None, Some(2)]
        );
    }
}
