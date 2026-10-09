//! Recording a launch sequence and replaying it as a CubeCL graph.
//!
//! A body is recorded by running it twice between `graph_prepare` and
//! `stop_capture`: a warm-up run, which executes and produces the body's
//! real result, then a recorded run, whose launches CubeCL records instead
//! of executing. The recorded graph replays against the exact device buffers
//! the recorded run bound, so the recording keeps a handle of each: a buffer
//! the body allocated and dropped stays reserved for the graph instead of
//! returning to the pool.
//!
//! A buffer the body received (an operand, a key, a table of the module) is
//! kept alive the same way, and that must never let a graph run on a buffer
//! its owner dropped: nobody reads those results. Every allocation carries a
//! flag its owning buffer clears when dropped, and a recording is
//! [valid](GraphRecording::valid) only while every buffer its body received
//! is still owned.
//!
//! Inside the recording window CubeCL refuses reads, host writes and
//! synchronizations, and serves allocations only from what the warm-up run
//! returned to its pools. A body whose warm-up run read, wrote or
//! synchronized is therefore never recorded, and a recording CubeCL rejects
//! (reported by `stop_capture`) is dropped. In both cases the warm-up run
//! already produced the result, so nothing changes for the caller.
//!
//! The window records the launches of every thread, so a recording holds the
//! [exclusive access](super::access) to the device from `graph_prepare` to
//! `stop_capture`, and a replay holds it while the graph is queued.
//!
//! The wgpu, CUDA and HIP runtimes of CubeCL 0.11.1 all record graphs. On a
//! runtime that does not, `graph_prepare` fails and the body just runs,
//! traced.

use std::sync::atomic::Ordering;

use cubecl::{client::Graph, prelude::Client};

use super::{Bound, GpuRuntime, Trace, Tracer, access, buffer::parked, replayed};

/// A recorded graph and the buffers it replays against.
pub(crate) struct GraphRecording {
    graph: Graph,
    /// Every allocation the recorded launches bound, kept alive for the graph.
    bound: Vec<Bound>,
}

/// A prepared capture, ended on drop: a body that panics must not leave the
/// client recording the launches that follow.
struct Capture(&'static Client);

impl Capture {
    fn stop(self) -> Option<Graph> {
        let graph = self.0.stop_capture().ok();
        std::mem::forget(self);
        graph
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        let _ = self.0.stop_capture();
    }
}

/// Runs `run` so that it takes effect exactly once, and returns the trace of
/// its launches with a recording of them when one could be made. `run` is
/// called a second time when a graph is recorded, with its launches recorded
/// and not executed; it must issue the same launches both times.
pub(crate) fn record<RT: GpuRuntime>(run: &mut dyn FnMut()) -> (Trace, Option<GraphRecording>) {
    let client = RT::client();
    let _access = access::exclusive();
    // The recorded run allocates again what the warm-up run dropped.
    let _parking = parked::start();
    let capture = client.graph_prepare().is_ok().then_some(Capture(client));

    // The warm-up run: it executes, compiles the kernels and leaves the
    // pools holding every buffer the body allocates.
    let tracer = Tracer::start(false);
    run();
    let (trace, _) = tracer.finish();
    let Some(capture) = capture else {
        return (trace, None);
    };

    if trace.host.reads + trace.host.writes + trace.host.syncs != 0 || client.start_capture().is_err() {
        return (trace, None);
    }
    let tracer = Tracer::start(true);
    run();
    let (recorded, bound) = tracer.finish();
    let recording = match capture.stop() {
        // The graph stands for the warm-up run only if both runs issued the same launches.
        Some(graph) if recorded == trace => Some(GraphRecording { graph, bound }),
        _ => None,
    };
    (trace, recording)
}

impl GraphRecording {
    /// Whether every buffer the body received is still owned by whoever
    /// passed it. An invalid recording never becomes valid again.
    pub(crate) fn valid(&self) -> bool {
        self.bound
            .iter()
            .all(|bound| bound.temporary || bound.owned.load(Ordering::Acquire))
    }

    /// Device bytes only this recording keeps reserved: the allocations it
    /// holds whose owning buffer is gone, which are the temporaries of its
    /// body (and, once it is invalid, the dropped operands).
    pub(crate) fn retained_bytes(&self) -> u64 {
        self.bound
            .iter()
            .filter(|bound| !bound.owned.load(Ordering::Acquire))
            .map(|bound| bound.reserved as u64)
            .sum()
    }

    /// Replays the graph. `false` when the recording is not
    /// [valid](Self::valid) or the runtime refused: nothing ran.
    pub(crate) fn replay(&self) -> bool {
        if !self.valid() {
            return false;
        }
        let _access = access::exclusive();
        // SAFETY: the recording holds a handle of every buffer the graph
        // binds, so each is still allocated; the exclusive access orders the
        // replay against every launch, read and write of the backend; and
        // the caller of the replay promises the operands are not in use
        // elsewhere, as for the body.
        let done = unsafe { self.graph.replay() }.is_ok();
        if done {
            replayed();
        }
        done
    }
}
