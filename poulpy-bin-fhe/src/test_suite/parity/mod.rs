//! Differential scheme tests for a caller-selected pair of backends.
//!
//! Fixtures start in the coefficient domain and are transferred identically.
//! Each backend prepares its own keys and uses its own reported scratch budget.
//! The backend under test need not expose host-readable storage. Validation
//! against any already validated backend can establish transitive parity.

pub mod bdd;
pub mod blind_rotation;
pub mod circuit_bootstrapping;
pub mod lifecycle;

use poulpy_core::{
    ComponentNoise, Distribution, TransferInto,
    layouts::{
        GGSW, GGSWInfos, GGSWToBackendMut, GGSWToBackendRef, GLWE, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
        ModuleCoreAlloc,
    },
};
use poulpy_hal::{
    layouts::{Backend, CopyFromHost, CopyToHost, HostBytesBackend, HostDataMut, Module, ScratchArena, ScratchOwned, ZnxViewMut},
    source::Source,
};

/// Storage operations needed to transfer test fixtures, without host-view bounds.
pub trait ParityBackend: Backend<ZnxWord = i64, OwnedBuf: CopyFromHost + CopyToHost> {}
impl<B: Backend<ZnxWord = i64, OwnedBuf: CopyFromHost + CopyToHost>> ParityBackend for B {}

pub(crate) type HostGlwe = GLWE<poulpy_hal::AlignedBuf, i64>;
pub(crate) type HostGgsw = GGSW<poulpy_hal::AlignedBuf, i64>;

/// Asserts evaluation outputs carry no noise tag, which equality between backends would miss.
pub(crate) fn assert_untagged<'a, T: LWEInfos + 'a>(label: &str, outputs: impl IntoIterator<Item = &'a T>) {
    for (i, output) in outputs.into_iter().enumerate() {
        assert!(output.noise().is_none(), "{label} output {i} kept a noise tag");
    }
}

pub(crate) fn host_glwe<B: Backend<ZnxWord = i64>, A: GLWEToBackendRef<B>>(ct: &A) -> HostGlwe {
    use poulpy_core::layouts::SetK;

    let view = ct.to_backend_ref();
    let module = Module::<HostBytesBackend>::new(view.n().as_usize() as u64);
    let mut layout = view.glwe_layout();
    layout.k = (view.max_size() * view.base2k().as_usize()).into();
    let mut host = module.glwe_alloc_from_infos(&layout);
    let bytes = view.n().as_usize() * (view.rank().as_usize() + 1) * view.max_size() * size_of::<i64>();
    B::copy_view_to_host(view.data().data(), &mut host.data_mut().data_mut().as_mut()[..bytes]);
    host.set_k(view.k());
    host.set_canonical(view.is_canonical());
    GLWEToBackendMut::<HostBytesBackend>::set_noise(&mut host, view.noise());
    host
}

pub(crate) fn host_ggsw<B: Backend<ZnxWord = i64>, A: GGSWToBackendRef<B>>(ct: &A) -> HostGgsw {
    let view = ct.to_backend_ref();
    let module = Module::<HostBytesBackend>::new(view.n().as_usize() as u64);
    let mut host = module.ggsw_alloc_from_infos(&view);
    for row in 0..view.dnum().as_usize() {
        for col in 0..=view.rank().as_usize() {
            host_glwe::<B, _>(&view.at_view(row, col)).transfer_into(&mut host.at_mut(row, col));
        }
    }
    GGSWToBackendMut::<HostBytesBackend>::set_noise(&mut host, view.noise());
    host
}

/// Uniform digits in `[-2^(base2k-1), 2^(base2k-1))`, as the backend mask sampler draws them.
pub(crate) fn fill_digits(digits: &mut [i64], base2k: usize, source: &mut Source) {
    let pow2k: u64 = 1 << base2k;
    for digit in digits {
        *digit = source.next_u64n(pow2k, pow2k - 1) as i64 - (pow2k >> 1) as i64;
    }
}

fn canonicalize(ct: &mut GLWE<impl HostDataMut, i64>) {
    let base = ct.base2k().as_usize();
    let live = ct.k().as_usize().div_ceil(base);
    let pad = (base - ct.k().as_usize() % base) % base;
    for col in 0..=ct.rank().as_usize() {
        if pad != 0 && live != 0 {
            for digit in ct.data_mut().at_mut(col, live - 1) {
                *digit &= !0i64 << pad;
            }
        }
        for limb in live..ct.max_size() {
            ct.data_mut().at_mut(col, limb).fill(0);
        }
    }
}

pub(crate) fn fixture_glwe<B: ParityBackend>(module: &Module<B>, infos: &impl GLWEInfos, seed: u8) -> GLWE<B::OwnedBuf, i64> {
    let host = Module::<HostBytesBackend>::new(module.n() as u64);
    let mut input = host.glwe_alloc_from_infos(infos);
    fill_digits(
        input.data_mut().raw_mut(),
        infos.base2k().as_usize(),
        &mut Source::new([seed; 32]),
    );
    canonicalize(&mut input);
    // A fresh estimate, so an evaluation output that keeps a stale one fails the comparison.
    let noise = ComponentNoise::from_secret_at(Distribution::TernaryProb(0.5), infos.k(), infos.rank().as_usize());
    GLWEToBackendMut::<HostBytesBackend>::set_noise(&mut input, Some(noise));
    let mut output = module.glwe_alloc_from_infos(infos);
    input.transfer_into(&mut output);
    output
}

pub(crate) fn fixture_ggsw<B: ParityBackend>(module: &Module<B>, infos: &impl GGSWInfos, seed: u8) -> GGSW<B::OwnedBuf, i64> {
    let host = Module::<HostBytesBackend>::new(module.n() as u64);
    let mut input = host.ggsw_alloc_from_infos(infos);
    let mut source = Source::new([seed; 32]);
    for row in 0..input.dnum().as_usize() {
        for col in 0..=input.rank().as_usize() {
            let mut glwe = input.at_mut(row, col);
            fill_digits(glwe.data_mut().raw_mut(), infos.base2k().as_usize(), &mut source);
            canonicalize(&mut glwe);
        }
    }
    // Tagged after filling, since mutable row access clears the tag.
    let noise = ComponentNoise::from_secret_at(Distribution::TernaryProb(0.5), infos.k(), infos.rank().as_usize());
    GGSWToBackendMut::<HostBytesBackend>::set_noise(&mut input, Some(noise));
    let mut output = module.ggsw_alloc_from_infos(infos);
    input.transfer_into(&mut output);
    output
}

/// Runs with exactly the advertised scratch, poisoned and surrounded by guards.
pub(crate) fn with_scratch<B: Backend, R>(bytes: usize, run: impl FnOnce(&mut ScratchArena<'_, B>) -> R) -> R {
    let guard = B::scratch_aligned(64);
    let mut owned = ScratchOwned::<B> {
        data: B::from_host_bytes(&vec![0xA5; guard + bytes + guard]),
        _phantom: std::marker::PhantomData,
    };
    let result = {
        let (_, rest) = owned.arena().split_at(guard);
        let (mut exact, _) = rest.split_at(bytes);
        assert_eq!(exact.available(), bytes);
        run(&mut exact)
    };
    let host = B::to_host_bytes(&owned.data);
    assert!(host[..guard].iter().all(|v| *v == 0xA5), "scratch prefix overwritten");
    assert!(
        host[guard + bytes..guard + bytes + guard].iter().all(|v| *v == 0xA5),
        "scratch suffix overwritten"
    );
    result
}
