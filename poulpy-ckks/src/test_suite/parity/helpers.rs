//! Coefficient-domain fixtures and exact scratch checks, using explicit transfers.
use crate::{
    CKKSInfos, CKKSLayout,
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc, CKKSPlaintextOwned},
};
use poulpy_core::layouts::{GLWEInfos, GLWEToBackendRef, LWEInfos};
use poulpy_hal::{
    layouts::{Backend, FillUniform, Module, ScratchArena, ScratchOwned, ZnxViewMut},
    source::Source,
    test_suite::{alloc_host_vec_znx, upload_vec_znx},
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

fn fixture_data<B: Backend<ZnxWord = i64>>(
    layout: &CKKSLayout,
    cols: usize,
    seed: u8,
) -> poulpy_hal::layouts::VecZnx<B::OwnedBuf, i64> {
    let b = layout.base2k().as_usize();
    let k = layout.k().as_usize();
    let size = k.div_ceil(b);
    let mut host = alloc_host_vec_znx::<B>(layout.n().as_usize(), cols, size);
    host.fill_uniform(b, &mut Source::new([seed; 32]));
    let pad = (b - k % b) % b;
    if pad != 0 {
        for col in 0..cols {
            for value in host.at_mut(col, size - 1) {
                *value &= !0i64 << pad;
            }
        }
    }
    upload_vec_znx::<B>(&host)
}

pub(crate) fn fixture_ciphertext<B: Backend<ZnxWord = i64>>(
    module: &Module<B>,
    layout: &CKKSLayout,
    seed: u8,
) -> CKKSCiphertextOwned<B> {
    let mut out = module.ckks_ciphertext_alloc_from_infos(layout);
    *out.data_mut() = fixture_data::<B>(layout, layout.rank().as_usize() + 1, seed);
    out
}

pub(crate) fn fixture_plaintext<B: Backend<ZnxWord = i64>>(
    module: &Module<B>,
    layout: &CKKSLayout,
    seed: u8,
) -> CKKSPlaintextOwned<B> {
    let mut out = module.ckks_plaintext_alloc_from_infos(layout);
    *out.data_mut() = fixture_data::<B>(layout, 1, seed);
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
