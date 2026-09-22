//! Shared coefficient fixtures; each backend prepares its own representation.
use super::helpers::with_scratch;
use poulpy_core::layouts::{
    GGLWE, GGLWELayout, GGLWEPreparedFactory, GLWEAutomorphismKeyPrepared, GLWEAutomorphismKeyPreparedFactory,
    GLWETensorKeyPrepared, GLWETensorKeyPreparedFactory, ModuleCoreAlloc, SetGaloisElement,
};
use poulpy_hal::{
    layouts::{Backend, FillUniform, HostBytesBackend, Module},
    source::Source,
    test_suite::upload_mat_znx,
};

pub(crate) fn key_layout(n: usize, base2k: usize, k: usize, dsize: usize, rank_in: usize, rank_out: usize) -> GGLWELayout {
    GGLWELayout {
        n: n.into(),
        base2k: base2k.into(),
        dnum: k.div_ceil(base2k * dsize).into(),
        dsize: dsize.into(),
        k_aux: (base2k * dsize).into(),
        rank_in: rank_in.into(),
        rank_out: rank_out.into(),
        stride: 1,
    }
}

pub(crate) fn fixture_gglwe<B: Backend<ZnxWord = i64>>(
    module: &Module<B>,
    layout: &GGLWELayout,
    seed: u8,
) -> GGLWE<B::OwnedBuf, i64> {
    let host_module = Module::<HostBytesBackend>::new(module.n() as u64);
    let mut host = host_module.gglwe_alloc_from_infos(layout);
    host.fill_uniform(layout.base2k.as_usize(), &mut Source::new([seed; 32]));
    let mut out = module.gglwe_alloc_from_infos(layout);
    *out.data_mut() = upload_mat_znx::<B>(host.data());
    out
}

pub(crate) fn prepared_tensor_key<B>(module: &Module<B>, layout: &GGLWELayout, seed: u8) -> GLWETensorKeyPrepared<B::OwnedBuf, B>
where
    B: Backend<ZnxWord = i64>,
    Module<B>: GLWETensorKeyPreparedFactory<B>,
{
    let coefficients = fixture_gglwe(module, layout, seed);
    let mut prepared = module.alloc_tensor_key_prepared_from_infos(layout);
    with_scratch::<B, _>(module.prepare_tensor_key_tmp_bytes(layout), |scratch| {
        module.prepare_tensor_key(&mut prepared, &coefficients, scratch);
    });
    prepared
}

pub(crate) fn prepared_automorphism_key<B>(
    module: &Module<B>,
    layout: &GGLWELayout,
    p: i64,
    seed: u8,
) -> GLWEAutomorphismKeyPrepared<B::OwnedBuf, B>
where
    B: Backend<ZnxWord = i64>,
    Module<B>: GLWEAutomorphismKeyPreparedFactory<B>,
{
    let coefficients = fixture_gglwe(module, layout, seed);
    let mut prepared = module.glwe_automorphism_key_prepared_alloc_from_infos(layout);
    with_scratch::<B, _>(module.gglwe_prepare_tmp_bytes(layout), |scratch| {
        module.gglwe_prepare(&mut prepared, &coefficients, scratch);
    });
    prepared.set_p(p);
    prepared
}
