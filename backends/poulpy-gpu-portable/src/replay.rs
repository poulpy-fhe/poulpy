//! Replay of a repeated sequence of backend calls.
//!
//! A composite operation (a bootstrap, a linear transformation) issues
//! thousands of launches behind a long host-side orchestration. An
//! application that repeats one on the same buffers records it once with
//! [`Replay::record`] and replays the recording afterwards, as one CubeCL
//! graph on the runtimes that have graphs.

use std::marker::PhantomData;

use crate::runtime::{GpuRuntime, GraphRecording, HostEvents, Trace, Tracer, record};

/// A recorded sequence of backend calls, replayed on demand.
///
/// [`Replay::record`] runs a closure of poulpy calls on the backend types of
/// the runtime `RT` and records the launches it issues. [`Replay::replay`] then
/// redoes the work of the closure: as one graph when the runtime recorded
/// one, and by calling the closure again otherwise.
///
/// # Contract
///
/// - **Same operands.** Every replay acts on the device buffers the
///   recorded call used. The closure must bind the same buffers each time it
///   is called and the caller must keep them alive, and unused by anything
///   else while a replay runs, for as long as the `Replay` lives. New inputs
///   are fed by writing into those buffers between replays; outputs are read
///   from them after a replay.
/// - **Same control flow.** The closure must issue the same launches each
///   time it is called. Host-side state it reads (limb counts, precisions,
///   metadata of its operands) must therefore be the same at every call: a
///   closure that changes such state must restore it itself, since a graph
///   replay does not run the closure and does not touch host-side state.
///   When recording, the closure is called twice in a row on the runtimes
///   with graphs (the second time its launches are recorded, not executed).
/// - **Device work only.** A closure that reads from the device, writes host
///   data to it or synchronizes is not recorded: every replay calls it.
///
/// A graph is never replayed on a buffer its owner dropped: if a buffer the
/// recorded call bound, other than the temporaries it allocated itself, is
/// gone, the recording is discarded and that replay and the following ones
/// call the closure.
///
/// A recording keeps the temporaries of the closure reserved on the device
/// ([`Self::retained_bytes`]) until the `Replay` is dropped.
///
/// While a closure is recorded and while a graph is queued, every other
/// thread that uses the device waits. A recording or a replay cannot be
/// started from inside a recorded closure.
pub struct Replay<'a, RT: GpuRuntime> {
    body: Box<dyn FnMut() + 'a>,
    trace: Trace,
    recording: Option<GraphRecording>,
    _runtime: PhantomData<fn() -> RT>,
}

impl<'a, RT: GpuRuntime> Replay<'a, RT> {
    /// Runs `body` once for its effect and records its launches.
    pub fn record(body: impl FnMut() + 'a) -> Self {
        let mut body: Box<dyn FnMut() + 'a> = Box::new(body);
        let (trace, recording) = record::<RT>(&mut body);
        Self {
            body,
            trace,
            recording,
            _runtime: PhantomData,
        }
    }

    /// Whether replays run a recorded graph rather than the closure.
    pub fn is_graph(&self) -> bool {
        self.recording.as_ref().is_some_and(GraphRecording::valid)
    }

    /// Device bytes only this recording keeps reserved: the temporaries of the closure.
    pub fn retained_bytes(&self) -> u64 {
        self.recording.as_ref().map_or(0, GraphRecording::retained_bytes)
    }

    /// Kernel launches of the recorded call.
    pub fn launches(&self) -> usize {
        self.trace.launches.len()
    }

    /// What the recorded call did besides launching kernels.
    pub fn host_events(&self) -> HostEvents {
        self.trace.host
    }

    /// Redoes the work of the recorded closure.
    pub fn replay(&mut self) {
        if self.recording.as_ref().is_some_and(GraphRecording::replay) {
            return;
        }
        // Not valid any more, or refused by the runtime: the recording goes, with the memory it holds.
        self.recording = None;
        (self.body)();
    }

    /// Calls the closure and panics unless it issues the recorded launches
    /// (kernel, dispatch, buffers bound, scalar and compile-time arguments):
    /// the test that the closure honours the contract. It needs no graph, so
    /// it works on every runtime.
    pub fn check(&mut self) {
        let tracer = Tracer::start(false);
        (self.body)();
        let (trace, _) = tracer.finish();
        if let Some(difference) = self.trace.difference(&trace) {
            panic!("replay check: the closure does not repeat its recording ({difference})");
        }
    }
}
