//! Coefficient-domain fixtures and exact scratch checks, using explicit transfers.
use crate::{
    CKKSInfos, CKKSLayout,
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc, CKKSPlaintextOwned},
};
use poulpy_core::{
    GLWEAdd, GLWEMaskFill,
    layouts::{GLWEInfos, GLWEToBackendRef, LWEInfos},
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena, ScratchOwned},
    source::Source,
};

/// Host ciphertext and CKKS metadata, plus the independent canonical-state invariant.
/// GLWE equality intentionally ignores that cached state.
pub(crate) type HostCiphertext = (poulpy_core::layouts::GLWE<poulpy_hal::AlignedBuf, i64>, crate::CKKSMeta, bool);

pub(crate) fn host_ciphertext<B, A>(value: &A) -> HostCiphertext
where
    B: Backend<ZnxWord = i64>,
    A: CKKSInfos + GLWEInfos + GLWEToBackendRef<B>,
{
    use poulpy_core::layouts::{GLWEToBackendMut, ModuleCoreAlloc, SetK};
    use poulpy_hal::layouts::HostBytesBackend;

    let view = value.to_backend_ref();
    let module = Module::<HostBytesBackend>::new(view.n().as_usize() as u64);
    let mut layout = view.glwe_layout();
    layout.k = (view.max_size() * view.base2k().as_usize()).into();
    let mut host = module.glwe_alloc_from_infos(&layout);
    let bytes = view.n().as_usize() * (view.rank().as_usize() + 1) * view.max_size() * size_of::<i64>();
    B::copy_view_to_host(view.data().data(), &mut host.data_mut().data_mut().as_mut()[..bytes]);
    host.set_k(view.k());
    host.set_canonical(view.is_canonical());
    GLWEToBackendMut::<HostBytesBackend>::set_encryption_metadata(&mut host, view.encryption_metadata());
    (host, value.meta(), view.is_canonical())
}

pub(crate) fn fixture_ciphertext<B: Backend<ZnxWord = i64>>(
    module: &Module<B>,
    layout: &CKKSLayout,
    seed: u8,
) -> CKKSCiphertextOwned<B>
where
    Module<B>: GLWEMaskFill<B>,
{
    let mut out = module.ckks_ciphertext_alloc_from_infos(layout);
    module.fill_glwe_from_source(&mut out, &mut Source::new([seed; 32]));
    out
}

/// With `lazy`, adds a fixture filled over whole limbs: the digits leave the
/// canonical range, bits below `k` must round away, and the flag is clear.
pub(crate) fn fixture_operand<B: Backend<ZnxWord = i64>>(
    module: &Module<B>,
    layout: &CKKSLayout,
    seed: u8,
    lazy: bool,
) -> CKKSCiphertextOwned<B>
where
    Module<B>: GLWEMaskFill<B> + GLWEAdd<B>,
{
    let mut out = fixture_ciphertext(module, layout, seed);
    if lazy {
        let mut whole = *layout;
        whole.glwe_layout.k = (out.size() * out.base2k().as_usize()).into();
        module.glwe_add_assign(&mut out, &fixture_ciphertext(module, &whole, !seed));
    }
    out
}

pub(crate) fn fixture_plaintext<B: Backend<ZnxWord = i64>>(
    module: &Module<B>,
    layout: &CKKSLayout,
    seed: u8,
) -> CKKSPlaintextOwned<B>
where
    Module<B>: GLWEMaskFill<B>,
{
    let mut out = module.ckks_plaintext_alloc_from_infos(layout);
    module.fill_glwe_from_source(&mut out, &mut Source::new([seed; 32]));
    out
}

/// Runs with exactly `bytes` accessible poisoned scratch bytes, surrounded by
/// guards. Both allocation and guard downloads work for opaque backend storage.
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
