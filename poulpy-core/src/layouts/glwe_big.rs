use poulpy_hal::layouts::{
    Backend, Data, VecZnxBig, VecZnxBigReborrowBackendMut, VecZnxBigReborrowBackendRef, VecZnxBigToBackendMut,
    VecZnxBigToBackendRef,
};

use crate::layouts::GLWECore;

/// Big/accumulator-domain variant of [`GLWE`](crate::layouts::GLWE).
///
/// This is the deferred-normalization intermediate: the output of an inverse
/// DFT, holding un-carried magnitude across its limbs until
/// `vec_znx_big_normalize` brings it back to the coefficient domain.
///
/// Like [`GLWEPrepared`](crate::layouts::GLWEPrepared) this is [`GLWECore`]
/// over a different payload, so it is the same semantic object in a different
/// computational domain, and `LWEInfos` / `GLWEInfos` come from the
/// payload-generic impls.
///
/// Note the caveat that applies to every non-coefficient domain: `k` on such a
/// value describes the accumulator, not a normalized precision claim. Do not
/// hand one across a public boundary until that contract is pinned down (see
/// Phase 0 of `docs/core-domain-generic-layouts-plan.md`).
pub type GLWEBig<D, B> = GLWECore<VecZnxBig<D, <B as Backend>::BigWord, B>>;

/// Shared backend-native borrow of a [`GLWEBig`].
pub type GLWEBigBackendRef<'a, B> = GLWEBig<<B as Backend>::BufRef<'a>, B>;
/// Mutable backend-native borrow of a [`GLWEBig`].
pub type GLWEBigBackendMut<'a, B> = GLWEBig<<B as Backend>::BufMut<'a>, B>;

pub trait GLWEBigToBackendRef<B: Backend> {
    fn to_backend_ref(&self) -> GLWEBigBackendRef<'_, B>;
}

impl<B: Backend, D: Data> GLWEBigToBackendRef<B> for GLWEBig<D, B>
where
    VecZnxBig<D, B::BigWord, B>: VecZnxBigToBackendRef<B>,
{
    fn to_backend_ref(&self) -> GLWEBigBackendRef<'_, B> {
        GLWECore {
            data: self.data.to_backend_ref(),
            base2k: self.base2k,
            k: self.k,
        }
    }
}

pub trait GLWEBigToBackendMut<B: Backend>: GLWEBigToBackendRef<B> {
    fn to_backend_mut(&mut self) -> GLWEBigBackendMut<'_, B>;
}

impl<B: Backend, D: Data> GLWEBigToBackendMut<B> for GLWEBig<D, B>
where
    VecZnxBig<D, B::BigWord, B>: VecZnxBigToBackendMut<B> + VecZnxBigToBackendRef<B>,
{
    fn to_backend_mut(&mut self) -> GLWEBigBackendMut<'_, B> {
        GLWECore {
            data: self.data.to_backend_mut(),
            base2k: self.base2k,
            k: self.k,
        }
    }
}

/// Reborrows a mutable backend-native [`GLWEBig`] view as a shared one.
pub(crate) fn glwe_big_backend_ref_from_mut<'a, 'b, B: Backend + 'b>(
    glwe: &'a GLWEBig<B::BufMut<'b>, B>,
) -> GLWEBigBackendRef<'a, B> {
    GLWECore {
        data: glwe.data.reborrow_backend_ref(),
        base2k: glwe.base2k,
        k: glwe.k,
    }
}

/// Reborrows a mutable backend-native [`GLWEBig`] view.
pub(crate) fn glwe_big_backend_mut_from_mut<'a, 'b, B: Backend + 'b>(
    glwe: &'a mut GLWEBig<B::BufMut<'b>, B>,
) -> GLWEBigBackendMut<'a, B> {
    GLWECore {
        data: glwe.data.reborrow_backend_mut(),
        base2k: glwe.base2k,
        k: glwe.k,
    }
}
