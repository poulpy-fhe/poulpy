//! Shared coefficient fixtures; each backend prepares its own representation.
use super::helpers::with_scratch;
use poulpy_core::{
    GLWEMaskFill,
    api::{GGLWEPreparedFactory, GLWEAutomorphismKeyPreparedFactory, GLWETensorKeyPreparedFactory},
    layouts::{
        GGLWE, GGLWELayout, GGLWEPrepared, GLWEAutomorphismKeyPrepared, GLWETensorKeyPrepared, ModuleCoreAlloc, SetGaloisElement,
    },
    test_suite::keys::fill_by_digit,
};
use poulpy_hal::{
    layouts::{Backend, Module},
    source::Source,
};

/// Gadget layout of a key covering `k` bits with digit size `dsize`.
pub fn key_layout(n: usize, base2k: usize, k: usize, dsize: usize, rank_in: usize, rank_out: usize) -> GGLWELayout {
    key_layout_bits(n, base2k, k, base2k * dsize, rank_in, rank_out)
}

/// [`key_layout`] with digits of `k_digit` bits, on or off a limb boundary.
pub fn key_layout_bits(n: usize, base2k: usize, k: usize, k_digit: usize, rank_in: usize, rank_out: usize) -> GGLWELayout {
    key_layout_offset(n, base2k, k, k_digit, 0, rank_in, rank_out)
}

/// [`key_layout_bits`] with its rows `k_offset` bits lower.
pub fn key_layout_offset(
    n: usize,
    base2k: usize,
    k: usize,
    k_digit: usize,
    k_offset: usize,
    rank_in: usize,
    rank_out: usize,
) -> GGLWELayout {
    GGLWELayout {
        n: n.into(),
        base2k: base2k.into(),
        dnum: k.div_ceil(k_digit).into(),
        k_digit: k_digit.into(),
        k_offset: k_offset.into(),
        k_aux: k_digit.into(),
        rank_in: rank_in.into(),
        rank_out: rank_out.into(),
        stride: 1,
    }
}

/// A GGLWE of `layout` filled digit by digit from `seed`.
pub fn fixture_gglwe<BE: Backend<ZnxWord = i64>>(module: &Module<BE>, layout: &GGLWELayout, seed: u8) -> GGLWE<BE::OwnedBuf, i64>
where
    Module<BE>: GLWEMaskFill<BE>,
{
    let mut out = module.gglwe_alloc_from_infos(layout);
    fill_by_digit(module, &mut out, 1, &mut Source::new([seed; 32]));
    out
}

/// A tensor key prepared from the fixture GGLWE of `seed`.
pub fn prepared_tensor_key<BE>(module: &Module<BE>, layout: &GGLWELayout, seed: u8) -> GLWETensorKeyPrepared<BE::OwnedBuf, BE>
where
    BE: Backend<ZnxWord = i64>,
    Module<BE>: GLWETensorKeyPreparedFactory<BE> + GLWEMaskFill<BE>,
{
    let coefficients = fixture_gglwe(module, layout, seed);
    let mut prepared = module.alloc_tensor_key_prepared_from_infos(layout);
    with_scratch::<BE, _>(module.glwe_tensor_key_prepare_tmp_bytes(layout), |scratch| {
        module.glwe_tensor_key_prepare(&mut prepared, &coefficients, scratch);
    });
    prepared
}

/// An automorphism key for Galois element `p`, prepared from the fixture GGLWE of `seed`.
pub fn prepared_automorphism_key<BE>(
    module: &Module<BE>,
    layout: &GGLWELayout,
    p: i64,
    seed: u8,
) -> GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>
where
    BE: Backend<ZnxWord = i64>,
    Module<BE>: GLWEAutomorphismKeyPreparedFactory<BE> + GLWEMaskFill<BE>,
{
    let coefficients = fixture_gglwe(module, layout, seed);
    let mut prepared = module.glwe_automorphism_key_prepared_alloc_from_infos(layout);
    with_scratch::<BE, _>(module.gglwe_prepare_tmp_bytes(layout), |scratch| {
        module.gglwe_prepare(&mut prepared, &coefficients, scratch);
    });
    prepared.set_p(p);
    prepared
}

/// A GGLWE prepared from the fixture GGLWE of `seed`.
pub fn prepared_gglwe<BE>(module: &Module<BE>, layout: &GGLWELayout, seed: u8) -> GGLWEPrepared<BE::OwnedBuf, BE>
where
    BE: Backend<ZnxWord = i64>,
    Module<BE>: GGLWEPreparedFactory<BE> + GLWEMaskFill<BE>,
{
    let coefficients = fixture_gglwe(module, layout, seed);
    let mut prepared = module.gglwe_prepared_alloc_from_infos(layout);
    with_scratch::<BE, _>(module.gglwe_prepare_tmp_bytes(layout), |scratch| {
        module.gglwe_prepare(&mut prepared, &coefficients, scratch)
    });
    prepared
}
