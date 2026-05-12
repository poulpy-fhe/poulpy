use std::collections::HashMap;

use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    ScratchArenaTakeCore,
    default::{glwe_packing::GLWEPackingDefault, glwe_trace::GLWETraceDefault},
    layouts::{
        GGLWEInfos, GGSWAtViewMut, GGSWAtViewRef, GGSWInfos, GGSWToBackendMut, GGSWToBackendRef, GLWE, GLWEAutomorphismKeyHelper,
        GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement,
        prepared::{GGLWEPreparedToBackendRef, GLWETensorKeyPreparedToBackendRef},
    },
    operations::{
        GGSWRotateDefault, GLWEAddDefault, GLWECopyDefault, GLWEMulConstDefault, GLWEMulPlainDefault, GLWEMulXpMinusOneDefault,
        GLWENegateDefault, GLWENormalizeDefault, GLWERotateDefault, GLWEShiftDefault, GLWESubDefault, GLWETensoringDefault,
        GLWEZeroDefault,
    },
};

/// Backend-provided GLWE constant-multiplication operations.
///
/// # Safety
/// Implementations must respect the provided layout metadata, conversion offset, and scratch-space
/// contracts, and must not read or write outside the specified backend-owned buffers.
pub unsafe trait GLWEMulConstImpl<BE: Backend>: Backend {
    fn glwe_mul_const_tmp_bytes<R, A, B>(module: &Module<BE>, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos;

    fn glwe_mul_const<'s, R, A, B>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        a: &A,
        b: &B,
        b_coeff: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        B: crate::layouts::GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_mul_const_assign<'s, R, B>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        b: &B,
        b_coeff: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        B: crate::layouts::GLWEToBackendRef<BE> + GLWEInfos;
}

/// Backend-provided GLWE-by-plaintext multiplication operations.
///
/// # Safety
/// Implementations must interpret the plaintext and ciphertext layouts consistently with the
/// backend and preserve all aliasing and buffer-bound invariants.
pub unsafe trait GLWEMulPlainImpl<BE: Backend>: Backend {
    fn glwe_mul_plain_tmp_bytes<R, A, B>(module: &Module<BE>, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_mul_plain<'s, R, A, B>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        a: &A,
        a_effective_k: usize,
        b: &B,
        b_effective_k: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        B: crate::layouts::GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_mul_plain_assign<'s, R, A>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        res_effective_k: usize,
        a: &A,
        a_effective_k: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;
}

/// Backend-provided GLWE tensoring and relinearization operations.
///
/// # Safety
/// Implementations must preserve tensor layout semantics, respect the temporary-size contracts,
/// and only touch backend-owned storage regions that belong to the supplied operands.
pub unsafe trait GLWETensoringImpl<BE: Backend>: Backend {
    fn glwe_tensor_apply_tmp_bytes<R, A, B>(module: &Module<BE>, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos;

    fn glwe_tensor_square_apply_tmp_bytes<R, A>(module: &Module<BE>, res: &R, a: &A) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_tensor_apply<'s, R, A, B>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        a: &A,
        a_effective_k: usize,
        b: &B,
        b_effective_k: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        B: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_tensor_square_apply<'s, R, A>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        a: &A,
        a_effective_k: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos;

    fn glwe_tensor_relinearize<'s, R, A, T>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        tsk: &T,
        tsk_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>;

    fn glwe_tensor_relinearize_tmp_bytes<R, A, B>(module: &Module<BE>, res: &R, a: &A, tsk: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos;
}

/// Backend-provided GLWE addition operations.
///
/// # Safety
/// Implementations must preserve GLWE layout invariants and respect all backend buffer bounds.
pub unsafe trait GLWEAddImpl<BE: Backend>: Backend {
    fn glwe_add_into<R, A, B>(module: &Module<BE>, res: &mut R, a: &A, b: &B)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        B: GLWEToBackendRef<BE>;

    fn glwe_add_assign<R, A>(module: &Module<BE>, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;
}

/// Backend-provided GLWE negation operations.
///
/// # Safety
/// Implementations must preserve GLWE layout invariants and respect all backend buffer bounds.
pub unsafe trait GLWENegateImpl<BE: Backend>: Backend {
    fn glwe_negate<R, A>(module: &Module<BE>, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;

    fn glwe_negate_assign<R>(module: &Module<BE>, res: &mut R)
    where
        R: GLWEToBackendMut<BE>;
}

/// Backend-provided GLWE subtraction operations.
///
/// # Safety
/// Implementations must preserve GLWE layout invariants and respect all backend buffer bounds.
pub unsafe trait GLWESubImpl<BE: Backend>: Backend {
    fn glwe_sub<R, A, B>(module: &Module<BE>, res: &mut R, a: &A, b: &B)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        B: GLWEToBackendRef<BE>;

    fn glwe_sub_assign<R, A>(module: &Module<BE>, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;

    fn glwe_sub_negate_assign<R, A>(module: &Module<BE>, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;
}

/// Backend-provided GLWE zeroing operations.
///
/// # Safety
/// Implementations must zero every polynomial column in the GLWE without violating layout or
/// backend buffer invariants.
pub unsafe trait GLWEZeroImpl<BE: Backend>: Backend {
    fn glwe_zero<R>(module: &Module<BE>, res: &mut R)
    where
        R: GLWEToBackendMut<BE>;
}

/// Backend-provided GLWE copy operations.
///
/// # Safety
/// Implementations must preserve GLWE layout invariants and respect all backend buffer bounds.
pub unsafe trait GLWECopyImpl<BE: Backend>: Backend {
    fn glwe_copy<R, A>(module: &Module<BE>, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;
}

/// Backend-provided GLWE rotation operations.
///
/// # Safety
/// Implementations must perform rotations according to the polynomial layout without violating
/// scratch-space, aliasing, or buffer-bound guarantees.
pub unsafe trait GLWERotateImpl<BE: Backend>: Backend {
    fn glwe_rotate_tmp_bytes(module: &Module<BE>) -> usize;

    fn glwe_rotate<R, A>(module: &Module<BE>, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;

    fn glwe_rotate_assign<'s, R>(module: &Module<BE>, k: i64, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>;
}

/// Backend-provided GGSW rotation operations.
///
/// # Safety
/// Implementations must preserve the GGSW structure for the backend and may only use scratch space
/// and in-place mutation in ways compatible with the advertised contracts.
pub unsafe trait GGSWRotateImpl<BE: Backend>: Backend {
    fn ggsw_rotate_tmp_bytes(module: &Module<BE>) -> usize;

    fn ggsw_rotate<R, A>(module: &Module<BE>, k: i64, res: &mut R, a: &A)
    where
        R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        A: GGSWToBackendRef<BE> + GGSWAtViewRef<BE> + GGSWInfos;

    fn ggsw_rotate_assign<'s, R>(module: &Module<BE>, k: i64, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos;
}

/// Backend-provided multiplication by `X^p - 1` operations.
///
/// # Safety
/// Implementations must apply the requested ring operation without violating the layout or memory
/// invariants of the supplied ciphertext buffers.
pub unsafe trait GLWEMulXpMinusOneImpl<BE: Backend>: Backend {
    fn glwe_mul_xp_minus_one<R, A>(module: &Module<BE>, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;

    fn glwe_mul_xp_minus_one_assign<'s, R>(module: &Module<BE>, k: i64, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>;
}

/// Backend-provided GLWE shift operations.
///
/// # Safety
/// Implementations must respect the polynomial/ciphertext layout and scratch requirements, and may
/// not read or write beyond the backend-owned regions described by the inputs.
pub unsafe trait GLWEShiftImpl<BE: Backend>: Backend {
    fn glwe_shift_tmp_bytes(module: &Module<BE>) -> usize;

    fn glwe_rsh<'s, R>(module: &Module<BE>, k: usize, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>;

    fn glwe_lsh_assign<'s, R>(module: &Module<BE>, res: &mut R, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>;

    fn glwe_lsh<'s, R, A>(module: &Module<BE>, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;

    fn glwe_lsh_add<'s, R, A>(module: &Module<BE>, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;

    fn glwe_lsh_sub<'s, R, A>(module: &Module<BE>, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;
}

/// Backend-provided GLWE normalization operations.
///
/// # Safety
/// Implementations must return views that remain valid for the advertised lifetime, preserve
/// normalization semantics, and avoid aliasing or out-of-bounds access across temporary buffers.
pub unsafe trait GLWENormalizeImpl<BE: Backend>: Backend {
    fn glwe_normalize_tmp_bytes(module: &Module<BE>) -> usize;

    fn glwe_normalize<'s, R, A>(module: &Module<BE>, res: &mut R, a: &A, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>;

    fn glwe_normalize_assign<'s, R>(module: &Module<BE>, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>;
}

/// Backend-provided GLWE trace operations.
///
/// # Safety
/// Implementations must apply the requested automorphism sequence faithfully, interpret prepared
/// keys correctly, and keep all accesses within the described ciphertext and scratch regions.
pub unsafe trait GLWETraceImpl<BE: Backend>: Backend {
    fn glwe_trace_galois_elements(module: &Module<BE>) -> Vec<i64>;

    fn glwe_trace_tmp_bytes<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_trace<'s, R, A, K, H>(
        module: &Module<BE>,
        res: &mut R,
        skip: usize,
        a: &A,
        keys: &H,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        BE: 's;

    fn glwe_trace_assign<'s, R, K, H>(
        module: &Module<BE>,
        res: &mut R,
        skip: usize,
        keys: &H,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        BE: 's;
}

/// Backend-provided GLWE packing operations.
///
/// # Safety
/// Implementations must maintain ciphertext correctness while combining inputs, and must respect
/// all backend buffer, aliasing, and scratch-space invariants expected by the higher layers.
pub unsafe trait GLWEPackImpl<BE: Backend>: Backend {
    fn glwe_pack_galois_elements(module: &Module<BE>) -> Vec<i64>;

    fn glwe_pack_tmp_bytes<R, K>(module: &Module<BE>, res: &R, key: &K) -> usize
    where
        R: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_pack<'s, R, A, K, H>(
        module: &Module<BE>,
        res: &mut R,
        a: HashMap<usize, &mut A>,
        log_gap_out: usize,
        keys: &H,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendMut<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        BE: 's;
}

#[allow(private_bounds)]
unsafe impl<BE: Backend> GLWEMulConstImpl<BE> for BE
where
    Module<BE>: GLWEMulConstDefault<BE>,
{
    fn glwe_mul_const_tmp_bytes<R, A, B>(module: &Module<BE>, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
    {
        module.glwe_mul_const_tmp_bytes_default(res, a, b)
    }

    fn glwe_mul_const<'s, R, A, B>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        a: &A,
        b: &B,
        b_coeff: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        B: crate::layouts::GLWEToBackendRef<BE> + GLWEInfos,
    {
        module.glwe_mul_const_default(cnv_offset, res, a, b, b_coeff, scratch)
    }

    fn glwe_mul_const_assign<'s, R, B>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        b: &B,
        b_coeff: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        B: crate::layouts::GLWEToBackendRef<BE> + GLWEInfos,
    {
        module.glwe_mul_const_assign_default(cnv_offset, res, b, b_coeff, scratch)
    }
}

#[allow(private_bounds)]
unsafe impl<BE: Backend> GLWEMulPlainImpl<BE> for BE
where
    Module<BE>: GLWEMulPlainDefault<BE>,
{
    fn glwe_mul_plain_tmp_bytes<R, A, B>(module: &Module<BE>, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
    {
        module.glwe_mul_plain_tmp_bytes_default(res, a, b)
    }

    fn glwe_mul_plain<'s, R, A, B>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        a: &A,
        a_effective_k: usize,
        b: &B,
        b_effective_k: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        B: crate::layouts::GLWEToBackendRef<BE> + GLWEInfos,
    {
        module.glwe_mul_plain_default(cnv_offset, res, a, a_effective_k, b, b_effective_k, scratch)
    }

    fn glwe_mul_plain_assign<'s, R, A>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        res_effective_k: usize,
        a: &A,
        a_effective_k: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
    {
        module.glwe_mul_plain_assign_default(cnv_offset, res, res_effective_k, a, a_effective_k, scratch)
    }
}

#[allow(private_bounds)]
unsafe impl<BE: Backend> GLWETensoringImpl<BE> for BE
where
    Module<BE>: GLWETensoringDefault<BE>,
{
    fn glwe_tensor_apply_tmp_bytes<R, A, B>(module: &Module<BE>, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
    {
        module.glwe_tensor_apply_tmp_bytes_default(res, a, b)
    }

    fn glwe_tensor_square_apply_tmp_bytes<R, A>(module: &Module<BE>, res: &R, a: &A) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
    {
        module.glwe_tensor_square_apply_tmp_bytes_default(res, a)
    }

    fn glwe_tensor_apply<'s, R, A, B>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        a: &A,
        a_effective_k: usize,
        b: &B,
        b_effective_k: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        B: GLWEToBackendRef<BE> + GLWEInfos,
    {
        module.glwe_tensor_apply_default(cnv_offset, res, a, a_effective_k, b, b_effective_k, scratch)
    }

    fn glwe_tensor_square_apply<'s, R, A>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut R,
        a: &A,
        a_effective_k: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
    {
        module.glwe_tensor_square_apply_default(cnv_offset, res, a, a_effective_k, scratch)
    }

    fn glwe_tensor_relinearize<'s, R, A, T>(
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        tsk: &T,
        tsk_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    {
        module.glwe_tensor_relinearize_default(res, a, tsk, tsk_size, scratch)
    }

    fn glwe_tensor_relinearize_tmp_bytes<R, A, B>(module: &Module<BE>, res: &R, a: &A, tsk: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos,
    {
        module.glwe_tensor_relinearize_tmp_bytes_default(res, a, tsk)
    }
}

#[allow(private_bounds)]
unsafe impl<BE: Backend> GLWEAddImpl<BE> for BE
where
    Module<BE>: GLWEAddDefault<BE>,
{
    fn glwe_add_into<R, A, B>(module: &Module<BE>, res: &mut R, a: &A, b: &B)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        B: GLWEToBackendRef<BE>,
    {
        module.glwe_add_into_default(res, a, b)
    }

    fn glwe_add_assign<R, A>(module: &Module<BE>, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        module.glwe_add_assign_default(res, a)
    }
}

#[allow(private_bounds)]
unsafe impl<BE: Backend> GLWENegateImpl<BE> for BE
where
    Module<BE>: GLWENegateDefault<BE>,
{
    fn glwe_negate<R, A>(module: &Module<BE>, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        module.glwe_negate_default(res, a)
    }

    fn glwe_negate_assign<R>(module: &Module<BE>, res: &mut R)
    where
        R: GLWEToBackendMut<BE>,
    {
        module.glwe_negate_assign_default(res)
    }
}

#[allow(private_bounds)]
unsafe impl<BE: Backend> GLWESubImpl<BE> for BE
where
    Module<BE>: GLWESubDefault<BE>,
{
    fn glwe_sub<R, A, B>(module: &Module<BE>, res: &mut R, a: &A, b: &B)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        B: GLWEToBackendRef<BE>,
    {
        module.glwe_sub_default(res, a, b)
    }

    fn glwe_sub_assign<R, A>(module: &Module<BE>, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        module.glwe_sub_assign_default(res, a)
    }

    fn glwe_sub_negate_assign<R, A>(module: &Module<BE>, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        module.glwe_sub_negate_assign_default(res, a)
    }
}

#[allow(private_bounds)]
unsafe impl<BE: Backend> GLWEZeroImpl<BE> for BE
where
    Module<BE>: GLWEZeroDefault<BE>,
{
    fn glwe_zero<R>(module: &Module<BE>, res: &mut R)
    where
        R: GLWEToBackendMut<BE>,
    {
        module.glwe_zero_default(res)
    }
}

#[allow(private_bounds)]
unsafe impl<BE: Backend> GLWECopyImpl<BE> for BE
where
    Module<BE>: GLWECopyDefault<BE>,
{
    fn glwe_copy<R, A>(module: &Module<BE>, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        module.glwe_copy_default(res, a)
    }
}

#[allow(private_bounds)]
unsafe impl<BE: Backend> GLWERotateImpl<BE> for BE
where
    Module<BE>: GLWERotateDefault<BE>,
{
    fn glwe_rotate_tmp_bytes(module: &Module<BE>) -> usize {
        module.glwe_rotate_tmp_bytes_default()
    }

    fn glwe_rotate<R, A>(module: &Module<BE>, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        module.glwe_rotate_default(k, res, a)
    }

    fn glwe_rotate_assign<'s, R>(module: &Module<BE>, k: i64, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
    {
        module.glwe_rotate_assign_default(k, res, scratch)
    }
}

#[allow(private_bounds)]
unsafe impl<BE: Backend> GLWEMulXpMinusOneImpl<BE> for BE
where
    Module<BE>: GLWEMulXpMinusOneDefault<BE>,
{
    fn glwe_mul_xp_minus_one<R, A>(module: &Module<BE>, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        module.glwe_mul_xp_minus_one_default(k, res, a)
    }

    fn glwe_mul_xp_minus_one_assign<'s, R>(module: &Module<BE>, k: i64, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
    {
        module.glwe_mul_xp_minus_one_assign_default(k, res, scratch)
    }
}

#[allow(private_bounds)]
unsafe impl<BE: Backend> GLWEShiftImpl<BE> for BE
where
    Module<BE>: GLWEShiftDefault<BE>,
    for<'x> ScratchArena<'x, BE>: ScratchArenaTakeCore<'x, BE>,
{
    fn glwe_shift_tmp_bytes(module: &Module<BE>) -> usize {
        module.glwe_shift_tmp_bytes_default()
    }

    fn glwe_rsh<'s, R>(module: &Module<BE>, k: usize, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
    {
        module.glwe_rsh_default(k, res, scratch)
    }

    fn glwe_lsh_assign<'s, R>(module: &Module<BE>, res: &mut R, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
    {
        module.glwe_lsh_assign_default(res, k, scratch)
    }

    fn glwe_lsh<'s, R, A>(module: &Module<BE>, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        module.glwe_lsh_default(res, a, k, scratch)
    }

    fn glwe_lsh_add<'s, R, A>(module: &Module<BE>, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        module.glwe_lsh_add_default(res, a, k, scratch)
    }

    fn glwe_lsh_sub<'s, R, A>(module: &Module<BE>, res: &mut R, a: &A, k: usize, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        module.glwe_lsh_sub_default(res, a, k, scratch)
    }
}

#[allow(private_bounds)]
unsafe impl<BE: Backend> GLWENormalizeImpl<BE> for BE
where
    Module<BE>: GLWENormalizeDefault<BE>,
    for<'x> ScratchArena<'x, BE>: ScratchArenaTakeCore<'x, BE>,
{
    fn glwe_normalize_tmp_bytes(module: &Module<BE>) -> usize {
        module.glwe_normalize_tmp_bytes_default()
    }

    fn glwe_normalize<'s, R, A>(module: &Module<BE>, res: &mut R, a: &A, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
    {
        module.glwe_normalize_default(res, a, scratch)
    }

    fn glwe_normalize_assign<'s, R>(module: &Module<BE>, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GLWEToBackendMut<BE>,
    {
        module.glwe_normalize_assign_default(res, scratch)
    }
}

#[allow(private_bounds)]
unsafe impl<BE: Backend> GGSWRotateImpl<BE> for BE
where
    Module<BE>: GGSWRotateDefault<BE>,
    for<'x> ScratchArena<'x, BE>: ScratchArenaTakeCore<'x, BE>,
{
    fn ggsw_rotate_tmp_bytes(module: &Module<BE>) -> usize {
        module.ggsw_rotate_tmp_bytes_default()
    }

    fn ggsw_rotate<R, A>(module: &Module<BE>, k: i64, res: &mut R, a: &A)
    where
        R: GGSWToBackendMut<BE> + GGSWAtViewMut<BE> + GGSWInfos,
        A: GGSWToBackendRef<BE> + GGSWAtViewRef<BE> + GGSWInfos,
    {
        module.ggsw_rotate_default(k, res, a)
    }

    fn ggsw_rotate_assign<'s, R>(module: &Module<BE>, k: i64, res: &mut R, scratch: &mut ScratchArena<'s, BE>)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
    {
        let mut res_backend = res.to_backend_mut();
        module.ggsw_rotate_assign_default(k, &mut res_backend, scratch)
    }
}

#[allow(private_bounds)]
unsafe impl<BE: Backend> GLWETraceImpl<BE> for BE
where
    Module<BE>: crate::default::glwe_trace::GLWETraceDefault<BE>,
    for<'x> ScratchArena<'x, BE>: ScratchArenaTakeCore<'x, BE>,
{
    fn glwe_trace_galois_elements(module: &Module<BE>) -> Vec<i64> {
        module.glwe_trace_galois_elements_default()
    }

    fn glwe_trace_tmp_bytes<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos,
    {
        module.glwe_trace_tmp_bytes_default(res_infos, a_infos, key_infos)
    }

    fn glwe_trace<'s, R, A, K, H>(
        module: &Module<BE>,
        res: &mut R,
        skip: usize,
        a: &A,
        keys: &H,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendRef<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        BE: 's,
    {
        let mut scratch_local = scratch.borrow();
        module.glwe_trace_default(res, skip, a, keys, key_size, &mut scratch_local)
    }

    fn glwe_trace_assign<'s, R, K, H>(
        module: &Module<BE>,
        res: &mut R,
        skip: usize,
        keys: &H,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        BE: 's,
    {
        let mut scratch_local = scratch.borrow();
        module.glwe_trace_assign_default(res, skip, keys, key_size, &mut scratch_local)
    }
}

#[allow(private_bounds)]
unsafe impl<BE: Backend> GLWEPackImpl<BE> for BE
where
    Module<BE>: crate::default::glwe_packing::GLWEPackingDefault<BE>,
    GLWE<Vec<u8>>: GLWEToBackendMut<BE>,
    for<'x> ScratchArena<'x, BE>: ScratchArenaTakeCore<'x, BE>,
{
    fn glwe_pack_galois_elements(module: &Module<BE>) -> Vec<i64> {
        module.glwe_pack_galois_elements_default()
    }

    fn glwe_pack_tmp_bytes<R, K>(module: &Module<BE>, res: &R, key: &K) -> usize
    where
        R: GLWEInfos,
        K: GGLWEInfos,
    {
        module.glwe_pack_tmp_bytes_default(res, key)
    }

    fn glwe_pack<'s, R, A, K, H>(
        module: &Module<BE>,
        res: &mut R,
        a: HashMap<usize, &mut A>,
        log_gap_out: usize,
        keys: &H,
        key_size: usize,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        A: GLWEToBackendMut<BE> + GLWEInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        BE: 's,
    {
        let mut scratch_local = scratch.borrow();
        module.glwe_pack_default(res, a, log_gap_out, keys, key_size, &mut scratch_local)
    }
}

/// Delegate the `GLWERotateImpl` family to another host backend through the
/// module-owned transfer API.
///
/// This intentionally routes values through `upload_glwe` / `download_glwe`
/// so a partial backend can keep explicit ownership boundaries when it falls
/// back to another backend's implementation.
#[macro_export]
macro_rules! impl_glwe_rotate_impl_from {
    ($be:ty, $from:ty) => {
        unsafe impl $crate::oep::GLWERotateImpl<$be> for $be {
            fn glwe_rotate_tmp_bytes(module: &poulpy_hal::layouts::Module<$be>) -> usize {
                let delegate: poulpy_hal::layouts::Module<$from> =
                    <poulpy_hal::layouts::Module<$from> as poulpy_hal::api::ModuleNew<$from>>::new(module.n() as u64);
                <poulpy_hal::layouts::Module<$from> as $crate::api::GLWERotate<$from>>::glwe_rotate_tmp_bytes(&delegate)
            }

            fn glwe_rotate<R, A>(module: &poulpy_hal::layouts::Module<$be>, k: i64, res: &mut R, a: &A)
            where
                R: $crate::layouts::GLWEToBackendMut,
                A: $crate::layouts::GLWEToBackendRef,
            {
                let delegate: poulpy_hal::layouts::Module<$from> =
                    <poulpy_hal::layouts::Module<$from> as poulpy_hal::api::ModuleNew<$from>>::new(module.n() as u64);

                let a_host: $crate::layouts::GLWE<Vec<u8>> =
                    poulpy_hal::layouts::ToOwnedDeep::to_owned_deep(&$crate::layouts::GLWEToBackendRef::to_backend_ref(a));
                let a_src: $crate::layouts::GLWE<<$be as poulpy_hal::layouts::Backend>::OwnedBuf> = a_host.reinterpret::<$be>();

                let res_infos = $crate::layouts::GLWEToBackendMut::to_backend_mut(res);
                let res_host: $crate::layouts::GLWE<Vec<u8>> = delegate.glwe_alloc_from_infos(&res_infos);
                let res_src: $crate::layouts::GLWE<<$be as poulpy_hal::layouts::Backend>::OwnedBuf> =
                    res_host.reinterpret::<$be>();

                let a_delegate = $crate::api::ModuleTransfer::upload_glwe::<$be>(&delegate, &a_src);
                let mut res_delegate = $crate::api::ModuleTransfer::upload_glwe::<$be>(&delegate, &res_src);

                <poulpy_hal::layouts::Module<$from> as $crate::api::GLWERotate<$from>>::glwe_rotate(
                    &delegate,
                    k,
                    &mut res_delegate,
                    &a_delegate,
                );

                let res_back: $crate::layouts::GLWE<<$be as poulpy_hal::layouts::Backend>::OwnedBuf> =
                    $crate::api::ModuleTransfer::download_glwe::<$from>(&delegate, &res_delegate);
                let res_back_ref = $crate::layouts::GLWEToBackendRef::to_backend_ref(&res_back);

                let mut bytes = Vec::new();
                poulpy_hal::layouts::WriterTo::write_to(&res_back_ref, &mut bytes)
                    .expect("failed to serialize delegated GLWE rotate result");

                let mut cursor = std::io::Cursor::new(bytes);
                let mut res_mut = $crate::layouts::GLWEToBackendMut::to_backend_mut(res);
                poulpy_hal::layouts::ReaderFrom::read_from(&mut res_mut, &mut cursor)
                    .expect("failed to write delegated GLWE rotate result back");
            }

            fn glwe_rotate_assign<'s, R>(
                module: &poulpy_hal::layouts::Module<$be>,
                k: i64,
                res: &mut R,
                _scratch: &mut poulpy_hal::layouts::ScratchArena<'s, $be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut,
            {
                let delegate: poulpy_hal::layouts::Module<$from> =
                    <poulpy_hal::layouts::Module<$from> as poulpy_hal::api::ModuleNew<$from>>::new(module.n() as u64);

                let res_host: $crate::layouts::GLWE<Vec<u8>> =
                    poulpy_hal::layouts::ToOwnedDeep::to_owned_deep(&$crate::layouts::GLWEToBackendMut::to_backend_mut(res));
                let res_src: $crate::layouts::GLWE<<$be as poulpy_hal::layouts::Backend>::OwnedBuf> =
                    res_host.reinterpret::<$be>();
                let mut res_delegate = $crate::api::ModuleTransfer::upload_glwe::<$be>(&delegate, &res_src);

                let mut scratch_owned: poulpy_hal::layouts::ScratchOwned<$from> =
                    <poulpy_hal::layouts::ScratchOwned<$from> as poulpy_hal::api::ScratchOwnedAlloc<$from>>::alloc(
                        <poulpy_hal::layouts::Module<$from> as $crate::api::GLWERotate<$from>>::glwe_rotate_tmp_bytes(&delegate),
                    );
                let scratch_delegate =
                    <poulpy_hal::layouts::ScratchOwned<$from> as poulpy_hal::api::ScratchOwnedBorrow<$from>>::borrow(
                        &mut scratch_owned,
                    );

                <poulpy_hal::layouts::Module<$from> as $crate::api::GLWERotate<$from>>::glwe_rotate_assign(
                    &delegate,
                    k,
                    &mut res_delegate,
                    scratch_delegate,
                );

                let res_back: $crate::layouts::GLWE<<$be as poulpy_hal::layouts::Backend>::OwnedBuf> =
                    $crate::api::ModuleTransfer::download_glwe::<$from>(&delegate, &res_delegate);
                let res_back_ref = $crate::layouts::GLWEToBackendRef::to_backend_ref(&res_back);

                let mut bytes = Vec::new();
                poulpy_hal::layouts::WriterTo::write_to(&res_back_ref, &mut bytes)
                    .expect("failed to serialize delegated GLWE rotate inplace result");

                let mut cursor = std::io::Cursor::new(bytes);
                let mut res_mut = $crate::layouts::GLWEToBackendMut::to_backend_mut(res);
                poulpy_hal::layouts::ReaderFrom::read_from(&mut res_mut, &mut cursor)
                    .expect("failed to write delegated GLWE rotate inplace result back");
            }
        }
    };
}

/// Implements [`GLWETraceDefault`] for `Module<$be>` by forwarding every method to
/// the corresponding [`glwe_trace_defaults`] free function.
#[macro_export]
macro_rules! impl_glwe_trace_defaults_full {
    ($be:ty) => {
        impl $crate::default::glwe_trace::GLWETraceDefault<$be> for ::poulpy_hal::layouts::Module<$be> {
            fn glwe_trace_assign_tmp_bytes_default<A, K>(&self, a_infos: &A, key_infos: &K) -> usize
            where
                A: $crate::layouts::GLWEInfos,
                K: $crate::layouts::GGLWEInfos,
            {
                $crate::default::glwe_trace::glwe_trace_defaults_impl::glwe_trace_assign_tmp_bytes_default::<$be, _, _, _>(
                    self, a_infos, key_infos,
                )
            }

            fn glwe_trace_galois_elements_default(&self) -> ::std::vec::Vec<i64> {
                $crate::default::glwe_trace::glwe_trace_defaults_impl::glwe_trace_galois_elements_default::<$be, _>(self)
            }

            fn glwe_trace_tmp_bytes_default<R, A, K>(&self, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
            where
                R: $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEInfos,
                K: $crate::layouts::GGLWEInfos,
            {
                $crate::default::glwe_trace::glwe_trace_defaults_impl::glwe_trace_tmp_bytes_default::<$be, _, _, _, _>(
                    self, res_infos, a_infos, key_infos,
                )
            }

            fn glwe_trace_default<'s, R, A, K, H>(
                &self,
                res: &mut R,
                skip: usize,
                a: &A,
                keys: &H,
                key_size: usize,
                scratch: &'s mut ::poulpy_hal::layouts::ScratchArena<'s, $be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
                K: $crate::layouts::prepared::GGLWEPreparedToBackendRef<$be>
                    + $crate::layouts::GetGaloisElement
                    + $crate::layouts::GGLWEInfos,
                H: $crate::layouts::GLWEAutomorphismKeyHelper<K, $be>,
                $be: 's,
            {
                $crate::default::glwe_trace::glwe_trace_defaults_impl::glwe_trace_default::<$be, _, _, _, _, _>(
                    self, res, skip, a, keys, key_size, scratch,
                )
            }

            fn glwe_trace_assign_default<'s, R, K, H>(
                &self,
                res: &mut R,
                skip: usize,
                keys: &H,
                key_size: usize,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'s, $be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                K: $crate::layouts::prepared::GGLWEPreparedToBackendRef<$be>
                    + $crate::layouts::GetGaloisElement
                    + $crate::layouts::GGLWEInfos,
                H: $crate::layouts::GLWEAutomorphismKeyHelper<K, $be>,
                $be: 's,
            {
                $crate::default::glwe_trace::glwe_trace_defaults_impl::glwe_trace_assign_default::<$be, _, _, _, _>(
                    self, res, skip, keys, key_size, scratch,
                )
            }
        }
    };
}

/// Implements [`GLWEPackingDefault`] for `Module<$be>` by forwarding every method to
/// the corresponding [`glwe_packing_defaults`] free function.
#[macro_export]
macro_rules! impl_glwe_packing_defaults_full {
    ($be:ty) => {
        impl $crate::default::glwe_packing::GLWEPackingDefault<$be> for ::poulpy_hal::layouts::Module<$be> {
            fn glwe_pack_galois_elements_default(&self) -> ::std::vec::Vec<i64> {
                $crate::default::glwe_packing::glwe_packing_defaults_impl::glwe_pack_galois_elements_default::<$be, _>(self)
            }

            fn glwe_pack_tmp_bytes_default<R, K>(&self, res: &R, key: &K) -> usize
            where
                R: $crate::layouts::GLWEInfos,
                K: $crate::layouts::GGLWEInfos,
            {
                $crate::default::glwe_packing::glwe_packing_defaults_impl::glwe_pack_tmp_bytes_default::<$be, _, _, _>(
                    self, res, key,
                )
            }

            fn glwe_pack_default<'s, R, A, K, H>(
                &self,
                res: &mut R,
                a: ::std::collections::HashMap<usize, &mut A>,
                log_gap_out: usize,
                keys: &H,
                key_size: usize,
                scratch: &'s mut ::poulpy_hal::layouts::ScratchArena<'s, $be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                K: $crate::layouts::prepared::GGLWEPreparedToBackendRef<$be>
                    + $crate::layouts::GetGaloisElement
                    + $crate::layouts::GGLWEInfos,
                H: $crate::layouts::GLWEAutomorphismKeyHelper<K, $be>,
                $be: 's,
            {
                $crate::default::glwe_packing::glwe_packing_defaults_impl::glwe_pack_default::<$be, _, _, _, _, _>(
                    self,
                    res,
                    a,
                    log_gap_out,
                    keys,
                    key_size,
                    scratch,
                )
            }
        }
    };
}
