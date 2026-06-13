pub mod gglwe;
pub mod ggsw;
pub mod glwe;
pub mod lwe;

pub(crate) use glwe::*;

use crate::layouts::{GGLWEInfos, GGSWInfos};

/// Truncated gadget-product output size (in limbs) for a VMP whose result is
/// IDFT'd and normalized to `out_size` limbs.
///
/// Each VMP output limb is an independent dot product against one column of
/// the prepared key, so the keyswitch (and the IDFT consuming it) scales
/// linearly with the number of output limbs requested. Output limbs whose
/// worst-case un-normalized magnitude lands more than `allowance_bits` below
/// the last limb of the `out_size`-limb normalize target cannot affect the
/// result and are skipped. One un-normalized output limb is bounded by
/// `sqrt(N) · ell · 2^{2(base2k-1)}` (`ell` = total gadget rows accumulated,
/// i.e. digit rows times decomposed input columns; the ring expansion factor
/// is `sqrt(N)` rather than `N` because the operands are signed/centered),
/// giving `ceil((log2(sqrt(N)·ell) + 2(base2k-1) - allowance) / base2k)`
/// guard limbs.
///
/// `allowance_bits` is the number of bits of error the consumer tolerates at
/// the bottom of the output: 0 when the result feeds further multiplications
/// at full precision (e.g. baby-step rotations, whose error is amplified by
/// the plaintext scale in the subsequent product), or the scale of a plaintext
/// factor already applied upstream (e.g. giant-step rotations after the
/// diagonal product, relinearization after tensoring), minus any slack the
/// caller already consumed to fit the result precision in `max_k`.
///
/// **Truncation is only applied for `dsize == 1`.** With `dsize > 1` the
/// gadget product accumulates digits whose per-digit depth is anchored to the
/// full key size and which carry the message across the entire limb range (no
/// spare redundancy below the message precision, especially at `dnum == 1`):
/// dropping any output limb corrupts the result, so this returns `key_limbs`
/// unchanged. A correct multi-digit bound is future work.
#[allow(clippy::too_many_arguments)]
pub fn truncated_gadget_product_size(
    n: usize,
    out_size: usize,
    in_size: usize,
    base2k: usize,
    dsize: usize,
    dnum: usize,
    in_cols: usize,
    key_limbs: usize,
    allowance_bits: usize,
) -> usize {
    if dsize != 1 {
        return key_limbs;
    }
    let ell = in_size.div_ceil(dsize.max(1)).min(dnum.max(1)).max(1) * in_cols.max(1);
    let log_expansion = ((usize::BITS - 1 - n.leading_zeros()) as usize).div_ceil(2);
    let log_ell = (usize::BITS - (ell - 1).leading_zeros()) as usize;
    let log_bound = log_expansion + log_ell + 2 * (base2k - 1);
    let guard = log_bound.saturating_sub(allowance_bits).div_ceil(base2k);
    (out_size - 1 + guard).max(out_size).min(key_limbs)
}

/// [`truncated_gadget_product_size`] for a GGLWE keyswitch key: `rank_in`
/// input columns are decomposed.
pub fn truncated_keyswitch_size<K: GGLWEInfos>(
    n: usize,
    out_size: usize,
    in_size: usize,
    key_infos: &K,
    allowance_bits: usize,
) -> usize {
    truncated_gadget_product_size(
        n,
        out_size,
        in_size,
        key_infos.base2k().as_usize(),
        key_infos.dsize().into(),
        key_infos.dnum().into(),
        key_infos.rank_in().into(),
        key_infos.size(),
        allowance_bits,
    )
}

/// [`truncated_gadget_product_size`] for a GGSW external product: all
/// `rank + 1` columns of the input GLWE are decomposed.
pub fn truncated_external_product_size<G: GGSWInfos>(
    n: usize,
    out_size: usize,
    in_size: usize,
    ggsw_infos: &G,
    allowance_bits: usize,
) -> usize {
    truncated_gadget_product_size(
        n,
        out_size,
        in_size,
        ggsw_infos.base2k().as_usize(),
        ggsw_infos.dsize().into(),
        ggsw_infos.dnum().into(),
        ggsw_infos.rank().as_usize() + 1,
        ggsw_infos.size(),
        allowance_bits,
    )
}
