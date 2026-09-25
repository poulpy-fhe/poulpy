//! Coefficient-domain fixtures and exact scratch checks, using explicit transfers.
use crate::{
    CKKSInfos, CKKSLayout,
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc, CKKSPlaintextOwned},
};
use poulpy_core::layouts::GLWEToBackendMut;
use poulpy_core::{
    GLWEMaskFill,
    layouts::{GLWEInfos, GLWEToBackendRef, LWEInfos},
};
use poulpy_hal::api::VecZnxFillUniformSource;
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena, ScratchOwned},
    source::Source,
};

/// Observable CKKS representation. Prepared backend storage is never compared.
#[derive(PartialEq, Eq)]
pub(crate) struct Snapshot {
    pub layout: CKKSLayout,
    pub digits: Vec<i64>,
}

impl std::fmt::Debug for Snapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Snapshot")
            .field("layout", &self.layout)
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
        digits,
    }
}

pub(crate) fn fixture_ciphertext<B: Backend<ZnxWord = i64>>(
    module: &Module<B>,
    layout: &CKKSLayout,
    seed: u8,
) -> CKKSCiphertextOwned<B>
where
    Module<B>: GLWEMaskFill<B> + VecZnxFillUniformSource<B>,
{
    let mut out = module.ckks_ciphertext_alloc_from_infos(layout);
    let mut mask_source = Source::new([seed; 32]);
    module.vec_znx_fill_uniform_source(
        layout.base2k().as_usize(),
        out.k().as_usize(),
        GLWEToBackendMut::<B>::to_backend_mut(&mut out).data_mut(),
        0,
        &mut mask_source,
    );
    module.fill_glwe_mask_from_source(&mut out, &mut mask_source);
    out
}

pub(crate) fn fixture_plaintext<B: Backend<ZnxWord = i64>>(
    module: &Module<B>,
    layout: &CKKSLayout,
    seed: u8,
) -> CKKSPlaintextOwned<B>
where
    Module<B>: GLWEMaskFill<B> + VecZnxFillUniformSource<B>,
{
    let mut out = module.ckks_plaintext_alloc_from_infos(layout);
    module.vec_znx_fill_uniform_source(
        layout.base2k().as_usize(),
        out.k().as_usize(),
        GLWEToBackendMut::<B>::to_backend_mut(&mut out).data_mut(),
        0,
        &mut Source::new([seed; 32]),
    );
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
