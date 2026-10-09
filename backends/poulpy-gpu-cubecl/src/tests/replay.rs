//! Recording and replay on the device: the wgpu runtime records real graphs.

use std::cell::RefCell;

use poulpy_hal::{
    api::{VecZnxAddAssign, VecZnxDftAlloc, VecZnxDftApply},
    layouts::{Backend, Module, VecZnx, VecZnxDftToBackendMut},
    test_suite::{vec_znx_backend_mut, vec_znx_backend_ref},
};

use crate::{Replay, runtime::Wgpu};

type Gpu = crate::NTT4x30Wgpu;
type Vector = VecZnx<<Gpu as Backend>::OwnedBuf, i64>;

const N: usize = 64;

fn vector(values: impl Fn(usize) -> i64, size: usize) -> Vector {
    let coefficients: Vec<i64> = (0..N * size).map(values).collect();
    VecZnx::from_data(Gpu::from_host_bytes(bytemuck::cast_slice(&coefficients)), N, 1, size)
}

fn read(vector: &Vector) -> Vec<i64> {
    bytemuck::pod_collect_to_vec::<u8, i64>(&Gpu::to_host_bytes(poulpy_hal::layouts::DataView::data(vector)))[..N * vector.size()]
        .to_vec()
}

fn add_assign(module: &Module<Gpu>, res: &mut Vector, a: &Vector) {
    module.vec_znx_add_assign(&mut vec_znx_backend_mut::<Gpu>(res), 0, &vec_znx_backend_ref::<Gpu>(a), 0);
}

/// A body of two launches on `res`, one of them through a temporary it allocates.
fn body(module: &Module<Gpu>, res: &mut Vector, a: &Vector) {
    add_assign(module, res, a);
    let mut dft = module.vec_znx_dft_alloc(N, 1, res.size());
    module.vec_znx_dft_apply(1, 0, &mut dft.to_backend_mut(), 0, &vec_znx_backend_ref::<Gpu>(res), 0);
}

#[test]
fn a_replay_redoes_the_closure() {
    let module = Module::<Gpu>::new(N as u64);
    let a = vector(|i| i as i64, 2);
    let mut res = vector(|_| 1, 2);
    {
        let mut replay = Replay::<Wgpu>::record(|| body(&module, &mut res, &a));
        // The closure ran twice, once for its effect and once recorded; the replays run the graph.
        assert!(replay.is_graph());
        // The addition, the zeroing of the temporary and its transform.
        assert_eq!(replay.launches(), 3);
        let host = replay.host_events();
        assert_eq!((host.reads, host.writes, host.syncs, host.allocations), (0, 0, 0, 1));
        // The recording holds the temporary of the body.
        assert_ne!(replay.retained_bytes(), 0);
        let before = crate::launch_counters().replays;
        replay.replay();
        replay.replay();
        // Other tests may replay meanwhile.
        assert!(crate::launch_counters().replays >= before + 2);
    }
    let want: Vec<i64> = (0..2 * N).map(|i| 1 + 3 * i as i64).collect();
    assert_eq!(read(&res), want);
}

#[test]
fn a_check_accepts_a_closure_that_repeats_itself() {
    let module = Module::<Gpu>::new(N as u64);
    let a = vector(|i| i as i64, 2);
    let mut res = vector(|_| 1, 2);
    let mut replay = Replay::<Wgpu>::record(|| body(&module, &mut res, &a));
    // The temporary is another allocation at each call: it is named by its order in the body.
    replay.check();
    replay.check();
}

#[test]
#[should_panic(expected = "does not repeat its recording")]
fn a_check_catches_a_closure_that_changes() {
    let module = Module::<Gpu>::new(N as u64);
    let a = vector(|i| i as i64, 2);
    let mut res = vector(|_| 1, 2);
    let mut calls = 0;
    let mut replay = Replay::<Wgpu>::record(|| {
        calls += 1;
        body(&module, &mut res, &a);
        if calls == 3 {
            // Host-side state the closure did not restore changes what it launches.
            body(&module, &mut res, &a);
        }
    });
    replay.check();
}

#[test]
fn a_closure_that_touches_the_host_is_called_at_every_replay() {
    let module = Module::<Gpu>::new(N as u64);
    let a = vector(|i| i as i64, 2);
    let res = RefCell::new(vector(|_| 1, 2));
    let mut replay = Replay::<Wgpu>::record(|| {
        add_assign(&module, &mut res.borrow_mut(), &a);
        assert_eq!(read(&res.borrow()).len(), 2 * N);
    });
    assert!(!replay.is_graph());
    assert_eq!((replay.host_events().reads, replay.retained_bytes()), (1, 0));
    replay.replay();
    drop(replay);
    let want: Vec<i64> = (0..2 * N).map(|i| 1 + 2 * i as i64).collect();
    assert_eq!(read(&res.borrow()), want);
}

#[test]
fn a_replay_calls_the_closure_once_a_bound_buffer_is_dropped() {
    let module = Module::<Gpu>::new(N as u64);
    let a = RefCell::new(vector(|i| i as i64, 2));
    let res = RefCell::new(vector(|_| 1, 2));
    let mut replay = Replay::<Wgpu>::record(|| body(&module, &mut res.borrow_mut(), &a.borrow()));
    replay.replay();
    assert!(replay.is_graph());
    assert_ne!(replay.retained_bytes(), 0);
    // The closure now binds another buffer: the graph would add the old one.
    *a.borrow_mut() = vector(|i| 5 * i as i64, 2);
    assert!(!replay.is_graph());
    replay.replay();
    // The recording went, with the memory it held.
    assert_eq!(replay.retained_bytes(), 0);
    replay.replay();
    drop(replay);
    let want: Vec<i64> = (0..2 * N).map(|i| 1 + (2 + 10) * i as i64).collect();
    assert_eq!(read(&res.borrow()), want);
}

/// A capture window records the launches of every thread: the launches of
/// another thread must wait for it and still take effect.
#[test]
fn another_thread_keeps_its_launches_while_a_closure_is_recorded() {
    const ROUNDS: usize = 400;
    let module = Module::<Gpu>::new(N as u64);
    let one = vector(|_| 1, 1);
    let mut other = vector(|_| 0, 1);
    let a = vector(|i| i as i64, 2);
    let mut res = vector(|_| 0, 2);
    let mut graphs = 0;
    std::thread::scope(|scope| {
        let worker = scope.spawn(|| {
            for _ in 0..ROUNDS {
                add_assign(&module, &mut other, &one);
            }
        });
        while !worker.is_finished() {
            let mut replay = Replay::<Wgpu>::record(|| body(&module, &mut res, &a));
            replay.replay();
            graphs += replay.is_graph() as usize;
        }
    });
    assert_ne!(graphs, 0);
    assert_eq!(read(&other), vec![ROUNDS as i64; N]);
    // Each round added `a` twice: the recorded call and its replay.
    let want: Vec<i64> = (0..2 * N).map(|i| (2 * graphs * i) as i64).collect();
    assert_eq!(read(&res), want);
}

#[test]
fn a_closure_that_panics_leaves_the_device_usable() {
    let module = Module::<Gpu>::new(N as u64);
    let a = vector(|i| i as i64, 2);
    let mut res = vector(|_| 1, 2);
    let mut calls = 0;
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Replay::<Wgpu>::record(|| {
            calls += 1;
            body(&module, &mut res, &a);
            // The second call is the recorded run, inside the capture window.
            assert!(calls < 2, "the recorded run fails");
        });
    }));
    assert!(panicked.is_err());
    // Launches execute again, and reads are served.
    add_assign(&module, &mut res, &a);
    let want: Vec<i64> = (0..2 * N).map(|i| 1 + 2 * i as i64).collect();
    assert_eq!(read(&res), want);
}

#[test]
#[should_panic(expected = "cannot start inside another one")]
fn a_recording_does_not_nest() {
    let module = Module::<Gpu>::new(N as u64);
    let a = vector(|i| i as i64, 2);
    let res = RefCell::new(vector(|_| 1, 2));
    Replay::<Wgpu>::record(|| {
        Replay::<Wgpu>::record(|| add_assign(&module, &mut res.borrow_mut(), &a));
    });
}
