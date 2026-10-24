use poulpy_core::layouts::GGSWInfos;

pub(crate) fn circuit_bootstrapping_log_gap_in<R: GGSWInfos>(res_infos: &R, log_domain: usize, extension_factor: usize) -> usize {
    assert!(
        extension_factor.is_power_of_two(),
        "extension_factor must be a non-zero power of two"
    );
    let dnum = res_infos.dnum().as_usize();
    assert!(dnum > 0, "circuit-bootstrap output must have at least one decomposition row");
    let alpha = dnum.next_power_of_two();
    let domain = 1usize
        .checked_shl(log_domain as u32)
        .expect("circuit-bootstrap domain overflows usize");
    let f_len = domain
        .checked_mul(alpha)
        .expect("circuit-bootstrap LUT length overflows usize");
    assert!(
        f_len <= res_infos.n().as_usize(),
        "circuit-bootstrap LUT length exceeds the polynomial degree"
    );
    let lut_domain = res_infos
        .n()
        .as_usize()
        .checked_mul(extension_factor)
        .expect("circuit-bootstrap LUT domain overflows usize");
    let step = lut_domain
        .checked_add(f_len >> 1)
        .expect("circuit-bootstrap LUT rounding overflows usize")
        / f_len;
    let gap = (step >> 1).checked_mul(2).expect("circuit-bootstrap LUT gap overflows usize") / extension_factor;
    assert!(
        gap > 0,
        "circuit-bootstrap LUT domain exceeds the available polynomial domain"
    );
    let spread = gap.checked_mul(alpha).expect("circuit-bootstrap LUT spread overflows usize");
    (usize::BITS - (spread - 1).leading_zeros()) as usize
}
