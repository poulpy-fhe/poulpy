//! Baby-step rotation materialization.
//!
//! Implements docs/lt_bsgs.md §6.2. The non-trivial baby rotations share one
//! DFT of the input mask columns, then each key performs VMP -> IDFT -> add body
//! -> normalize -> automorphism. The resulting SMALL ciphertexts are prepared as
//! `CnvPVecL` so every giant step can reuse them in convolution form.

use std::collections::{BTreeMap, BTreeSet};

use poulpy_hal::{
    api::{
        CnvPVecAlloc, Convolution, ModuleN, ScratchArenaTakeBasic, VecZnxAutomorphismAssignBackend, VecZnxBigAddSmallAssign,
        VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxDftApply, VecZnxDftBytesOf, VecZnxDftZero, VecZnxIdftApply,
        VecZnxIdftApplyTmpBytes,
    },
    layouts::{
        Backend, CnvPVecL, CnvPVecLToBackendMut, CnvPVecLToBackendRef, ScratchArena, VecZnxBigToBackendRef, VecZnxDftBackendRef,
        VecZnxDftToBackendRef, ZnxInfos,
    },
};

use crate::{
    GLWEAutomorphism, ScratchArenaTakeCore,
    default::{keyswitching::GGLWEProductDefault, operations::msb_mask_bottom_limb},
    layouts::{
        GGLWEInfos, GLWE, GLWEAutomorphismKeyHelper, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement, LWEInfos,
        prepared::GGLWEPreparedToBackendRef,
    },
};

/// Accessor for prepared baby-step rotations.
///
/// The index is the actual baby-step slot rotation `k` used by the current BSGS
/// transform. This mirrors Lattigo's BSGS index map: the giant-step loop asks
/// directly for `rot(v, k)`, regardless of whether the cache was prepared for
/// one transform or for a union of transforms.
pub trait GLWEPreparedBabyStepHelper<BE: Backend> {
    type BabyStep: CnvPVecLToBackendRef<BE> + ZnxInfos;

    fn baby_step(&self, baby: i64) -> &Self::BabyStep;
}

/// Prepared left operands for the baby rotations of one input ciphertext.
pub struct GLWEPreparedBabyRotations<BE: Backend> {
    values: BTreeMap<i64, CnvPVecL<BE::OwnedBuf, BE>>,
}

impl<BE: Backend> GLWEPreparedBabyRotations<BE> {
    fn new(values: BTreeMap<i64, CnvPVecL<BE::OwnedBuf, BE>>) -> Self {
        Self { values }
    }

    /// The slot rotations represented by this prepared baby cache.
    pub fn baby_steps(&self) -> impl ExactSizeIterator<Item = i64> + '_ {
        self.values.keys().copied()
    }

    /// Returns true when `rot` is available in this prepared baby cache.
    pub fn contains_baby_step(&self, rot: i64) -> bool {
        self.values.contains_key(&rot)
    }

    /// Number of prepared baby rotations.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns true when no baby rotations are prepared.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    fn baby_step_by_rotation(&self, rot: i64) -> &CnvPVecL<BE::OwnedBuf, BE> {
        self.values
            .get(&rot)
            .unwrap_or_else(|| panic!("missing prepared baby-step rotation {rot}"))
    }
}

impl<BE: Backend> GLWEPreparedBabyStepHelper<BE> for GLWEPreparedBabyRotations<BE> {
    type BabyStep = CnvPVecL<BE::OwnedBuf, BE>;

    fn baby_step(&self, baby: i64) -> &Self::BabyStep {
        self.baby_step_by_rotation(baby)
    }
}

pub(super) fn glwe_prepare_baby_rotations_tmp_bytes<BE, M, A, K>(module: &M, a_infos: &A, key_infos: &K) -> usize
where
    BE: Backend,
    M: ModuleN
        + Convolution<BE>
        + GLWEAutomorphism<BE>
        + GGLWEProductDefault<BE>
        + VecZnxAutomorphismAssignBackend<BE>
        + VecZnxBigBytesOf
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApplyTmpBytes,
    A: GLWEInfos,
    K: GGLWEInfos,
{
    let cols = a_infos.rank().as_usize() + 1;
    let a_size = a_infos.size();
    let key_size = key_infos.size();
    let baby = GLWE::<Vec<u8>>::bytes_of_from_infos(a_infos);
    let prepare = module.cnv_prepare_left_tmp_bytes(a_infos.size(), a_infos.size());

    let hoisted_a_dft = module.bytes_of_vec_znx_dft(cols - 1, a_size);
    let hoisted_rot = module.bytes_of_vec_znx_dft(cols, key_size)
        + module.bytes_of_vec_znx_big(cols, key_size)
        + module
            .gglwe_product_dft_tmp_bytes_default(key_size, a_size, key_infos)
            .max(module.vec_znx_idft_apply_tmp_bytes());
    let hoisted = hoisted_a_dft + baby + hoisted_rot.max(prepare);

    let fallback = baby + module.glwe_automorphism_tmp_bytes(a_infos, a_infos, key_infos).max(prepare);
    hoisted.max(fallback).max(prepare)
}

#[allow(clippy::too_many_arguments)]
fn glwe_hoisted_baby_rotation<BE, M, R, A, H, K>(
    module: &M,
    baby: &mut R,
    rot: i64,
    a: &A,
    a_dft_ref: &VecZnxDftBackendRef<'_, BE>,
    key_size: usize,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: ModuleN
        + GGLWEProductDefault<BE>
        + VecZnxAutomorphismAssignBackend<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftBytesOf
        + VecZnxDftZero<BE>
        + VecZnxIdftApply<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE> + GLWEInfos,
    K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    H: GLWEAutomorphismKeyHelper<K, BE>,
{
    let cols = a.rank().as_usize() + 1;
    let key: &K = keys
        .get_automorphism_key(rot)
        .unwrap_or_else(|| panic!("missing automorphism key for baby-step rotation {rot}"));
    let key_ref = key.to_backend_ref();
    assert_eq!(key_ref.base2k(), a.base2k());

    let (mut res_dft, mut scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module, cols, key_size);
    if key_ref.dsize().as_usize() > 1 {
        // See `glwe_hoisted_baby_rotations`: multi-digit VMP accumulates into
        // top limbs that must not contain stale scratch contents.
        for col in 0..res_dft.cols() {
            module.vec_znx_dft_zero(&mut res_dft, col);
        }
    }
    module.gglwe_product_dft_default(&mut res_dft, a_dft_ref, &key_ref, &mut scratch_1.borrow());

    let (mut res_big, mut scratch_2) = scratch_1.take_vec_znx_big_scratch(module, cols, key_size);
    let res_dft_ref = res_dft.to_backend_ref();
    for col in 0..cols {
        module.vec_znx_idft_apply(&mut res_big, col, &res_dft_ref, col, &mut scratch_2.borrow());
    }
    {
        let a_ref = a.to_backend_ref();
        module.vec_znx_big_add_small_assign(&mut res_big, 0, &a_ref.data, 0);
    }

    let res_big_ref = res_big.to_backend_ref();
    let baby_base2k = baby.base2k().as_usize();
    let a_base2k = a.base2k().as_usize();
    {
        let mut baby_ref = baby.to_backend_mut();
        for col in 0..cols {
            module.vec_znx_big_normalize(
                &mut baby_ref.data,
                baby_base2k,
                0,
                col,
                &res_big_ref,
                a_base2k,
                col,
                &mut scratch_2.borrow(),
            );
        }
        for col in 0..cols {
            module.vec_znx_automorphism_assign_backend(key.p(), &mut baby_ref.data, col, &mut scratch_2.borrow());
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn glwe_prepare_baby_rotations<BE, M, A, H, K>(
    module: &M,
    baby_steps: &[i64],
    a: &A,
    a_effective_k: usize,
    key_size: usize,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> GLWEPreparedBabyRotations<BE>
where
    BE: Backend,
    M: CnvPVecAlloc<BE>
        + Convolution<BE>
        + GLWEAutomorphism<BE>
        + GGLWEProductDefault<BE>
        + ModuleN
        + VecZnxAutomorphismAssignBackend<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxDftZero<BE>
        + VecZnxIdftApply<BE>,
    A: GLWEToBackendRef<BE> + GLWEInfos,
    K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    H: GLWEAutomorphismKeyHelper<K, BE>,
{
    let baby_steps: Vec<i64> = baby_steps.iter().copied().collect::<BTreeSet<_>>().into_iter().collect();
    let cols = a.rank().as_usize() + 1;
    let a_size = a.size();
    let mask = msb_mask_bottom_limb(a.base2k().as_usize(), a_effective_k);
    let has_nonzero_rotation = baby_steps.iter().any(|&rot| rot != 0);
    let (use_hoisted, key_size) = if has_nonzero_rotation {
        let key_infos = keys.automorphism_key_infos();
        (a.base2k() == key_infos.base2k(), key_size.min(key_infos.size()))
    } else {
        (false, key_size)
    };

    let mut values = BTreeMap::new();
    if use_hoisted {
        let scratch = scratch.borrow();
        let (mut a_dft, mut loop_scratch) = scratch.take_vec_znx_dft_scratch(module, cols - 1, a_size);
        {
            let a_ref = a.to_backend_ref();
            for col_i in 0..cols - 1 {
                module.vec_znx_dft_apply(1, 0, &mut a_dft, col_i, &a_ref.data, col_i + 1);
            }
        }
        let a_dft_ref = a_dft.to_backend_ref();

        for &rot in &baby_steps {
            let mut prepared = module.cnv_pvec_left_alloc(cols, a_size);
            if rot == 0 {
                let a_ref = a.to_backend_ref();
                module.cnv_prepare_left(&mut prepared.to_backend_mut(), &a_ref.data, mask, &mut loop_scratch.borrow());
            } else {
                let (mut baby, mut baby_scratch) = loop_scratch.borrow().take_glwe_scratch(a);
                glwe_hoisted_baby_rotation(
                    module,
                    &mut baby,
                    rot,
                    a,
                    &a_dft_ref,
                    key_size,
                    keys,
                    &mut baby_scratch.borrow(),
                );
                let baby_ref = baby.to_backend_ref();
                module.cnv_prepare_left(
                    &mut prepared.to_backend_mut(),
                    &baby_ref.data,
                    mask,
                    &mut baby_scratch.borrow(),
                );
            }
            values.insert(rot, prepared);
        }
    } else {
        for &rot in &baby_steps {
            let mut prepared = module.cnv_pvec_left_alloc(cols, a_size);
            if rot == 0 {
                let a_ref = a.to_backend_ref();
                module.cnv_prepare_left(&mut prepared.to_backend_mut(), &a_ref.data, mask, scratch);
            } else {
                let key: &K = keys
                    .get_automorphism_key(rot)
                    .unwrap_or_else(|| panic!("missing automorphism key for baby-step rotation {rot}"));
                let (mut baby, mut baby_scratch) = scratch.borrow().take_glwe_scratch(a);
                module.glwe_automorphism(&mut baby, a, key, key_size, &mut baby_scratch.borrow());
                let baby_ref = baby.to_backend_ref();
                module.cnv_prepare_left(
                    &mut prepared.to_backend_mut(),
                    &baby_ref.data,
                    mask,
                    &mut baby_scratch.borrow(),
                );
            }
            values.insert(rot, prepared);
        }
    }

    GLWEPreparedBabyRotations::new(values)
}
