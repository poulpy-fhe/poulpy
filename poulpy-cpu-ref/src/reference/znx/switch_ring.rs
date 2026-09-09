use crate::reference::znx::{copy::znx_copy_ref, zero::znx_zero_ref};

pub fn znx_switch_ring_ref(res: &mut [i64], a: &[i64]) {
    let (n_in, n_out) = (a.len(), res.len());

    #[cfg(debug_assertions)]
    {
        assert!(n_in.is_power_of_two());
        assert!(n_in.max(n_out).is_multiple_of(n_in.min(n_out)))
    }

    if n_in == n_out {
        znx_copy_ref(res, a);
        return;
    }

    let (gap_in, gap_out): (usize, usize);
    if n_in > n_out {
        (gap_in, gap_out) = (n_in / n_out, 1)
    } else {
        (gap_in, gap_out) = (1, n_out / n_in);
        znx_zero_ref(res);
    }

    res.iter_mut()
        .step_by(gap_out)
        .zip(a.iter().step_by(gap_in))
        .for_each(|(x_out, x_in)| *x_out = *x_in);
}
