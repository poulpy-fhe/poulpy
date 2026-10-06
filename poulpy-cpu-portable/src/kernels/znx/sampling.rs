use poulpy_hal::source::Source;

pub fn znx_fill_uniform_portable(base2k: usize, res: &mut [i64], source: &mut Source) {
    let pow2k: u64 = 1 << base2k;
    let mask: u64 = pow2k - 1;
    let pow2k_half: i64 = (pow2k >> 1) as i64;
    res.iter_mut()
        .for_each(|xi| *xi = (source.next_u64n(pow2k, mask) as i64) - pow2k_half)
}
