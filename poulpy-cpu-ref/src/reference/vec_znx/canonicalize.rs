use std::mem::size_of_val;

use poulpy_hal::layouts::{
    Backend, HostDataMut, VecZnxBackendMut, ZnxViewMut, vec_znx_backend_mut_with_size, vec_znx_reborrow_backend_mut,
};

use crate::reference::{
    vec_znx::{vec_znx_lsh_assign, vec_znx_rsh_assign, vec_znx_rsh_tmp_bytes},
    znx::{
        ZnxCopy, ZnxNormalizeFinalStepAssign, ZnxNormalizeFirstStepAssign, ZnxNormalizeFirstStepCarryOnly,
        ZnxNormalizeMiddleStep, ZnxNormalizeMiddleStepAssign, ZnxNormalizeMiddleStepCarryOnly, ZnxZero,
    },
};

pub fn vec_znx_canonicalize_tmp_bytes(n: usize) -> usize {
    vec_znx_rsh_tmp_bytes(n)
}

/// Canonicalizes one already-normalized column at semantic precision `k`.
pub fn vec_znx_canonicalize<BE>(base2k: usize, k: usize, a: &mut VecZnxBackendMut<'_, BE>, a_col: usize, tmp: &mut [i64])
where
    BE: Backend<ZnxWord = i64>
        + ZnxZero
        + ZnxCopy
        + ZnxNormalizeFirstStepCarryOnly
        + ZnxNormalizeMiddleStepCarryOnly
        + ZnxNormalizeMiddleStep
        + ZnxNormalizeFirstStepAssign
        + ZnxNormalizeMiddleStepAssign
        + ZnxNormalizeFinalStepAssign,
    for<'x> BE::BufMut<'x>: HostDataMut,
{
    assert_ne!(base2k, 0);
    assert!(
        k <= a.size() * base2k,
        "k ({k}) exceeds VecZnx capacity ({})",
        a.size() * base2k
    );
    assert!(size_of_val(tmp) >= vec_znx_canonicalize_tmp_bytes(a.n()));

    let active_size = k.div_ceil(base2k);
    for limb in active_size..a.size() {
        BE::znx_zero(a.at_mut(a_col, limb));
    }

    let padding = (base2k - k % base2k) % base2k;
    if active_size == 0 || padding == 0 {
        return;
    }

    let mut active = vec_znx_backend_mut_with_size::<BE>(vec_znx_reborrow_backend_mut::<BE>(a), active_size);
    vec_znx_rsh_assign::<BE>(base2k, padding, &mut active, a_col, tmp);
    vec_znx_lsh_assign::<BE>(base2k, padding, &mut active, a_col, tmp);
}

#[cfg(test)]
mod tests {
    use poulpy_hal::layouts::{Module, VecZnxToBackendMut, ZnxView, ZnxViewMut};

    use crate::{FFT64Ref, reference::vec_znx::vec_znx_canonicalize};

    #[test]
    fn clears_partial_bits_and_inactive_limbs() {
        let module = Module::<FFT64Ref>::new(4);
        let mut value = module.vec_znx_alloc(1, 3);
        value.at_mut(0, 0).copy_from_slice(&[3, -2, 1, -1]);
        value.at_mut(0, 1).copy_from_slice(&[15, -1, 12, -12]);
        value.at_mut(0, 2).fill(7);

        let mut tmp = vec![0; 2 * module.n()];
        vec_znx_canonicalize::<FFT64Ref>(
            5,
            8,
            &mut <_ as VecZnxToBackendMut<FFT64Ref>>::to_backend_mut(&mut value),
            0,
            &mut tmp,
        );

        assert_eq!(value.at(0, 0), &[4, -2, 1, -1]);
        assert_eq!(value.at(0, 1), &[-16, 0, 12, -12]);
        assert!(value.at(0, 2).iter().all(|&digit| digit == 0));
    }
}
