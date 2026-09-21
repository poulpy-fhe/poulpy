use crate::CKKSResult as Result;
use poulpy_core::{
    GLWEAdd, GLWENormalize, GLWEShift,
    layouts::{GLWEToBackendMut, GLWEToBackendRef, LWEInfos},
};
use poulpy_hal::{
    api::{VecZnxLshAdd, VecZnxLshTmpBytes, VecZnxRshAdd, VecZnxRshTmpBytes},
    layouts::{Backend, ScratchArena},
};

use crate::{
    CKKSInfos, SetCKKSInfos, checked_log_budget_sub, ckks_offset_binary,
    reference::{CKKSPlaintextReference, carry_verb::ckks_carry_verb_reference},
};

ckks_carry_verb_reference! {
    verb: add,
    doc_verb: "addition",
    trait_name: CKKSAddReference,
    glwe_bound: GLWEAdd,
    glwe_into: glwe_add_into,
    glwe_assign: glwe_add_assign,
    glwe_lsh_verb: glwe_lsh_add,
    pt_vec_bounds: [VecZnxLshAdd, VecZnxRshAdd],
}

impl<BE: Backend> CKKSAddReference<BE> for poulpy_hal::layouts::Module<BE> {}

/// Reference entry point for an unnormalized ciphertext wrapper.
pub fn ckks_add_into_unnormalized_wrapped_reference<BE: ::poulpy_hal::layouts::Backend, Dst, A, B>(
    module: &::poulpy_hal::layouts::Module<BE>,
    dst: &mut crate::layouts::UnnormalizedCKKSCiphertext<Dst, BE::ZnxWord>,
    a: &A,
    b: &B,
    scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, BE>,
) -> crate::CKKSResult<()>
where
    ::poulpy_hal::layouts::Module<BE>: ::poulpy_core::GLWEAdd<BE>
        + ::poulpy_core::GLWENormalize<BE>
        + ::poulpy_core::GLWEShift<BE>
        + ::poulpy_hal::api::VecZnxLshAdd<BE>
        + ::poulpy_hal::api::VecZnxRshAdd<BE>
        + ::poulpy_hal::api::VecZnxLshTmpBytes
        + ::poulpy_hal::api::VecZnxRshTmpBytes,
    Dst: ::poulpy_hal::layouts::Data,
    ::poulpy_core::layouts::GLWE<Dst, BE::ZnxWord>: ::poulpy_core::layouts::GLWEToBackendMut<BE>,
    A: ::poulpy_core::layouts::GLWEToBackendRef<BE> + crate::CKKSCtBounds,
    B: ::poulpy_core::layouts::GLWEToBackendRef<BE> + crate::CKKSCtBounds,
{
    crate::reference::add::CKKSAddReference::ckks_add_into_unnormalized_reference(module, &mut dst.write_view(), a, b, scratch)
}

/// Reference entry point for an unnormalized ciphertext wrapper.
pub fn ckks_add_assign_unnormalized_wrapped_reference<BE: ::poulpy_hal::layouts::Backend, Dst, A>(
    module: &::poulpy_hal::layouts::Module<BE>,
    dst: &mut crate::layouts::UnnormalizedCKKSCiphertext<Dst, BE::ZnxWord>,
    a: &A,
    scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, BE>,
) -> crate::CKKSResult<()>
where
    ::poulpy_hal::layouts::Module<BE>: ::poulpy_core::GLWEAdd<BE>
        + ::poulpy_core::GLWENormalize<BE>
        + ::poulpy_core::GLWEShift<BE>
        + ::poulpy_hal::api::VecZnxLshAdd<BE>
        + ::poulpy_hal::api::VecZnxRshAdd<BE>
        + ::poulpy_hal::api::VecZnxLshTmpBytes
        + ::poulpy_hal::api::VecZnxRshTmpBytes,
    Dst: ::poulpy_hal::layouts::Data,
    ::poulpy_core::layouts::GLWE<Dst, BE::ZnxWord>: ::poulpy_core::layouts::GLWEToBackendMut<BE>,
    A: ::poulpy_core::layouts::GLWEToBackendRef<BE> + crate::CKKSInfos,
{
    crate::reference::add::CKKSAddReference::ckks_add_assign_unnormalized_reference(module, &mut dst.write_view(), a, scratch)
}

/// Reference entry point for an unnormalized ciphertext wrapper.
pub fn ckks_add_assign_unnormalized_ref_wrapped_reference<BE: ::poulpy_hal::layouts::Backend, Dst, A>(
    module: &::poulpy_hal::layouts::Module<BE>,
    dst: &mut crate::layouts::ciphertext::UnnormalizedCKKSCiphertextRefMut<'_, Dst, BE::ZnxWord>,
    a: &A,
    scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, BE>,
) -> crate::CKKSResult<()>
where
    ::poulpy_hal::layouts::Module<BE>: ::poulpy_core::GLWEAdd<BE>
        + ::poulpy_core::GLWENormalize<BE>
        + ::poulpy_core::GLWEShift<BE>
        + ::poulpy_hal::api::VecZnxLshAdd<BE>
        + ::poulpy_hal::api::VecZnxRshAdd<BE>
        + ::poulpy_hal::api::VecZnxLshTmpBytes
        + ::poulpy_hal::api::VecZnxRshTmpBytes,
    Dst: ::poulpy_hal::layouts::Data,
    crate::layouts::CKKSCiphertext<Dst, BE::ZnxWord>: ::poulpy_core::layouts::GLWEToBackendMut<BE>,
    A: ::poulpy_core::layouts::GLWEToBackendRef<BE> + crate::CKKSInfos,
{
    crate::reference::add::CKKSAddReference::ckks_add_assign_unnormalized_reference(module, dst.inner, a, scratch)
}

/// Reference entry point for an unnormalized ciphertext wrapper.
pub fn ckks_add_pt_vec_into_unnormalized_wrapped_reference<BE: ::poulpy_hal::layouts::Backend, Dst, A, P>(
    module: &::poulpy_hal::layouts::Module<BE>,
    dst: &mut crate::layouts::UnnormalizedCKKSCiphertext<Dst, BE::ZnxWord>,
    a: &A,
    pt: &P,
    scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, BE>,
) -> crate::CKKSResult<()>
where
    ::poulpy_hal::layouts::Module<BE>: ::poulpy_core::GLWEAdd<BE>
        + ::poulpy_core::GLWENormalize<BE>
        + ::poulpy_core::GLWEShift<BE>
        + ::poulpy_hal::api::VecZnxLshAdd<BE>
        + ::poulpy_hal::api::VecZnxRshAdd<BE>
        + ::poulpy_hal::api::VecZnxLshTmpBytes
        + ::poulpy_hal::api::VecZnxRshTmpBytes,
    Dst: ::poulpy_hal::layouts::Data,
    ::poulpy_core::layouts::GLWE<Dst, BE::ZnxWord>: ::poulpy_core::layouts::GLWEToBackendMut<BE>,
    A: ::poulpy_core::layouts::GLWEToBackendRef<BE> + crate::CKKSCtBounds,
    P: ::poulpy_core::layouts::GLWEToBackendRef<BE> + crate::CKKSCtBounds + ::poulpy_core::layouts::IntPolyInfos,
{
    crate::reference::add::CKKSAddReference::ckks_add_pt_vec_into_unnormalized_reference(
        module,
        &mut dst.write_view(),
        a,
        pt,
        scratch,
    )
}

/// Reference entry point for an unnormalized ciphertext wrapper.
pub fn ckks_add_pt_vec_assign_unnormalized_wrapped_reference<BE: ::poulpy_hal::layouts::Backend, Dst, P>(
    module: &::poulpy_hal::layouts::Module<BE>,
    dst: &mut crate::layouts::UnnormalizedCKKSCiphertext<Dst, BE::ZnxWord>,
    pt: &P,
    scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, BE>,
) -> crate::CKKSResult<()>
where
    ::poulpy_hal::layouts::Module<BE>: ::poulpy_core::GLWEAdd<BE>
        + ::poulpy_core::GLWENormalize<BE>
        + ::poulpy_core::GLWEShift<BE>
        + ::poulpy_hal::api::VecZnxLshAdd<BE>
        + ::poulpy_hal::api::VecZnxRshAdd<BE>
        + ::poulpy_hal::api::VecZnxLshTmpBytes
        + ::poulpy_hal::api::VecZnxRshTmpBytes,
    Dst: ::poulpy_hal::layouts::Data,
    ::poulpy_core::layouts::GLWE<Dst, BE::ZnxWord>: ::poulpy_core::layouts::GLWEToBackendMut<BE>,
    P: ::poulpy_core::layouts::GLWEToBackendRef<BE> + crate::CKKSCtBounds + ::poulpy_core::layouts::IntPolyInfos,
{
    crate::reference::add::CKKSAddReference::ckks_add_pt_vec_assign_unnormalized_reference(
        module,
        &mut dst.write_view(),
        pt,
        scratch,
    )
}

/// Reference entry point for an unnormalized ciphertext wrapper.
pub fn ckks_add_pt_const_into_unnormalized_wrapped_reference<BE: ::poulpy_hal::layouts::Backend, Dst, A, P>(
    module: &::poulpy_hal::layouts::Module<BE>,
    dst: &mut crate::layouts::UnnormalizedCKKSCiphertext<Dst, BE::ZnxWord>,
    a: &A,
    dst_coeff: usize,
    pt: &P,
    pt_coeff: usize,
    scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, BE>,
) -> crate::CKKSResult<()>
where
    ::poulpy_hal::layouts::Module<BE>: ::poulpy_core::GLWEAdd<BE>
        + ::poulpy_core::GLWENormalize<BE>
        + ::poulpy_core::GLWEShift<BE>
        + ::poulpy_hal::api::VecZnxLshAdd<BE>
        + ::poulpy_hal::api::VecZnxRshAdd<BE>
        + ::poulpy_hal::api::VecZnxLshTmpBytes
        + ::poulpy_hal::api::VecZnxRshTmpBytes,
    Dst: ::poulpy_hal::layouts::Data,
    ::poulpy_core::layouts::GLWE<Dst, BE::ZnxWord>: ::poulpy_core::layouts::GLWEToBackendMut<BE>,
    A: ::poulpy_core::layouts::GLWEToBackendRef<BE> + crate::CKKSCtBounds,
    P: ::poulpy_core::layouts::GLWEToBackendRef<BE> + crate::CKKSCtBounds + ::poulpy_core::layouts::IntPolyInfos,
{
    crate::reference::add::CKKSAddReference::ckks_add_pt_const_into_unnormalized_reference(
        module,
        &mut dst.write_view(),
        a,
        dst_coeff,
        pt,
        pt_coeff,
        scratch,
    )
}

/// Reference entry point for an unnormalized ciphertext wrapper.
pub fn ckks_add_pt_const_assign_unnormalized_wrapped_reference<BE: ::poulpy_hal::layouts::Backend, Dst, P>(
    module: &::poulpy_hal::layouts::Module<BE>,
    dst: &mut crate::layouts::UnnormalizedCKKSCiphertext<Dst, BE::ZnxWord>,
    dst_coeff: usize,
    pt: &P,
    pt_coeff: usize,
    scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, BE>,
) -> crate::CKKSResult<()>
where
    ::poulpy_hal::layouts::Module<BE>: ::poulpy_core::GLWEAdd<BE>
        + ::poulpy_core::GLWENormalize<BE>
        + ::poulpy_core::GLWEShift<BE>
        + ::poulpy_hal::api::VecZnxLshAdd<BE>
        + ::poulpy_hal::api::VecZnxRshAdd<BE>
        + ::poulpy_hal::api::VecZnxLshTmpBytes
        + ::poulpy_hal::api::VecZnxRshTmpBytes,
    Dst: ::poulpy_hal::layouts::Data,
    ::poulpy_core::layouts::GLWE<Dst, BE::ZnxWord>: ::poulpy_core::layouts::GLWEToBackendMut<BE>,
    P: ::poulpy_core::layouts::GLWEToBackendRef<BE> + crate::CKKSCtBounds + ::poulpy_core::layouts::IntPolyInfos,
{
    crate::reference::add::CKKSAddReference::ckks_add_pt_const_assign_unnormalized_reference(
        module,
        &mut dst.write_view(),
        dst_coeff,
        pt,
        pt_coeff,
        scratch,
    )
}
