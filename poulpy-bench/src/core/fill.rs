//! Backend-native operand construction for the core benchmarks.
//!
//! Benchmarks measure time, and this arithmetic is data-independent, so the
//! operands need not be genuine ciphertexts: an encryption of zero is uniform
//! limbs too. Filling directly through the backend removes secrets, encryption
//! and any host staging from the runners, which is what keeps them open to a
//! device backend.
//!
//! Uniform noise rather than zeros is deliberate: a float FFT over zeroed
//! buffers can hit denormals and mistime the kernel.

use poulpy_core::layouts::{
    GGLWEAtBackendMut, GGLWEInfos, GGSW, GGSWAtBackendMut, GGSWInfos, GLWE, GLWEBackendMut, GLWEInfos, GLWEPlaintext, GLWETensor,
    LWEInfos,
};
use poulpy_hal::{
    api::VecZnxFillUniformSourceBackend,
    layouts::{Backend, Module, VecZnxBackendMut, vec_znx_backend_mut, vec_znx_reborrow_backend_mut},
    source::Source,
};

/// Fills every column of one GLWE-shaped block.
///
/// The view carries its own column count: `rank + 1` for a ciphertext,
/// `rank_out + 1` for a key row.
pub fn fill_vec_znx<BE>(module: &Module<BE>, data: &mut VecZnxBackendMut<'_, BE>, base2k: usize, source: &mut Source)
where
    BE: Backend<ZnxWord = i64>,
    Module<BE>: VecZnxFillUniformSourceBackend<BE>,
{
    for col in 0..data.cols() {
        module.vec_znx_fill_uniform_source_backend(base2k, data, col, source);
    }
}

/// Fills one borrowed GLWE row.
pub fn fill_glwe_row<BE>(module: &Module<BE>, row: &mut GLWEBackendMut<'_, BE>, base2k: usize, source: &mut Source)
where
    BE: Backend<ZnxWord = i64>,
    Module<BE>: VecZnxFillUniformSourceBackend<BE>,
{
    fill_vec_znx(
        module,
        &mut vec_znx_reborrow_backend_mut::<BE>(row.data_mut()),
        base2k,
        source,
    );
}

/// Fills a GLWE ciphertext.
pub fn fill_glwe<BE>(module: &Module<BE>, ct: &mut GLWE<BE::OwnedBuf, BE::ZnxWord>, source: &mut Source)
where
    BE: Backend<ZnxWord = i64>,
    Module<BE>: VecZnxFillUniformSourceBackend<BE>,
{
    let base2k: usize = ct.base2k().into();
    fill_vec_znx(module, &mut vec_znx_backend_mut::<BE>(ct.data_mut()), base2k, source);
}

/// Fills a GLWE plaintext.
pub fn fill_glwe_plaintext<BE>(module: &Module<BE>, pt: &mut GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord>, source: &mut Source)
where
    BE: Backend<ZnxWord = i64>,
    Module<BE>: VecZnxFillUniformSourceBackend<BE>,
{
    let base2k: usize = pt.base2k().into();
    fill_vec_znx(module, &mut vec_znx_backend_mut::<BE>(pt.data_mut()), base2k, source);
}

/// Fills a GGLWE-shaped key, one GLWE row at a time.
///
/// Generic over the key wrapper, so `GGLWE`, `GLWESwitchingKey` and
/// `GLWEAutomorphismKey` all go through it.
pub fn fill_gglwe<BE, K>(module: &Module<BE>, key: &mut K, source: &mut Source)
where
    BE: Backend<ZnxWord = i64>,
    Module<BE>: VecZnxFillUniformSourceBackend<BE>,
    K: GGLWEAtBackendMut<BE> + GGLWEInfos,
{
    let base2k: usize = key.base2k().into();
    let rows: usize = key.dnum().as_usize();
    let cols: usize = key.rank_in().as_usize();
    for row in 0..rows {
        for col in 0..cols {
            fill_glwe_row(module, &mut key.at_backend_mut(row, col), base2k, source);
        }
    }
}

/// Fills a GLWE tensor.
pub fn fill_glwe_tensor<BE>(module: &Module<BE>, t: &mut GLWETensor<BE::OwnedBuf, BE::ZnxWord>, source: &mut Source)
where
    BE: Backend<ZnxWord = i64>,
    Module<BE>: VecZnxFillUniformSourceBackend<BE>,
{
    let base2k: usize = t.base2k().into();
    fill_vec_znx(module, &mut vec_znx_backend_mut::<BE>(t.data_mut()), base2k, source);
}

/// Fills a GGSW, one GLWE row at a time.
pub fn fill_ggsw<BE>(module: &Module<BE>, ggsw: &mut GGSW<BE::OwnedBuf, BE::ZnxWord>, source: &mut Source)
where
    BE: Backend<ZnxWord = i64>,
    Module<BE>: VecZnxFillUniformSourceBackend<BE>,
    GGSW<BE::OwnedBuf, BE::ZnxWord>: GGSWAtBackendMut<BE>,
{
    let base2k: usize = ggsw.base2k().into();
    let rows: usize = ggsw.dnum().as_usize();
    let cols: usize = ggsw.rank().as_usize() + 1;
    for row in 0..rows {
        for col in 0..cols {
            fill_glwe_row(module, &mut ggsw.at_backend_mut(row, col), base2k, source);
        }
    }
}
