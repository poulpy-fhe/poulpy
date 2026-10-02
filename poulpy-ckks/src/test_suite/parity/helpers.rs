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

/// Observable CKKS representation. Prepared backend storage is never compared.
#[derive(PartialEq, Eq)]
pub(crate) struct Snapshot {
    pub layout: CKKSLayout,
    pub canonical: bool,
    pub digits: Vec<i64>,
}

impl std::fmt::Debug for Snapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Snapshot")
            .field("layout", &self.layout)
            .field("canonical", &self.canonical)
            .field("total_digits", &self.digits.len())
            .field("first_digits", &&self.digits[..self.digits.len().min(16)])
            .finish()
    }
}

pub(crate) fn snapshot<B, A>(value: &A) -> Snapshot
where
    B: Backend<ZnxWord = i64>,
    A: CKKSInfos + GLWEInfos + GLWEToBackendRef<B>,
{
    let view = value.to_backend_ref();
    let mut digits = vec![0i64; view.n().as_usize() * (view.rank().as_usize() + 1) * view.data().size()];
    B::copy_view_to_host(view.data().data(), bytemuck::cast_slice_mut(&mut digits));
    Snapshot {
        layout: CKKSLayout {
            glwe_layout: poulpy_core::layouts::GLWELayout {
                n: value.n(),
                base2k: value.base2k(),
                k: value.k(),
                rank: value.rank(),
            },
            meta: value.meta(),
        },
        canonical: view.is_canonical(),
        digits,
    }
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
