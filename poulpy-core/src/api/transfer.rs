//! Moving a layout from one backend to another.
//!
//! The destination must already exist:
//!
//! ```ignore
//! let mut sk = module.glwe_secret_alloc_from_infos(&sk_host);
//! sk_host.transfer_into(&mut sk);
//! ```
//!
//! Three things follow from requiring an allocated destination, none of which a
//! form returning a fresh value can offer:
//!
//! - no backend is named anywhere. Both buffer types are concrete, so nothing
//!   has to be recovered from an associated-type projection, and there is no
//!   type parameter whose direction a reader can misjudge.
//! - the whole shape is checkable, not just the byte count. Both operands are
//!   present, so `base2k`, `k`, degree, column and limb counts are all compared.
//! - the allocation is visible and hoistable. Moving a key set allocates once,
//!   and the cost is not hidden inside something that reads like a conversion.
//!
//! Impls are one per layout: a shape check, then the container move. Being a
//! trait rather than a closed set of methods, a downstream crate can transfer
//! its own layouts without editing this one.

use poulpy_hal::layouts::{
    CopyFromHost, CopyToHost, Data, DataView, DataViewMut, MatZnx, ScalarZnx, VecZnx, ZnxWord, transfer_buf_into,
};

use crate::layouts::{
    GGLWE, GGLWEToGGSWKey, GGSW, GLWE, GLWEAutomorphismKey, GLWEPlaintext, GLWEPublicKey, GLWESecret, GLWESwitchingKey,
    GLWETensor, GLWETensorKey, GLWEToLWEKey, LWE, LWEPlaintext, LWESecret,
    compressed::{GGLWECompressed, GLWECompressed, GLWEPublicKeyCompressed},
};

/// Moves `Self` into an already-allocated destination.
pub trait TransferInto<Dst> {
    /// # Panics
    ///
    /// If the two values do not agree on shape.
    fn transfer_into(&self, dst: &mut Dst);
}

fn move_vec_znx<D1, D2, W>(src: &VecZnx<D1, W>, dst: &mut VecZnx<D2, W>)
where
    D1: Data + CopyToHost,
    D2: Data + CopyFromHost,
    W: ZnxWord,
{
    assert!(src.n() == dst.n(), "transfer_into: ring degree");
    assert!(src.cols() == dst.cols(), "transfer_into: cols");
    assert!(src.size() == dst.size(), "transfer_into: size");
    transfer_buf_into(src.data(), dst.data_mut());
}

fn move_mat_znx<D1, D2, W>(src: &MatZnx<D1, W>, dst: &mut MatZnx<D2, W>)
where
    D1: Data + CopyToHost,
    D2: Data + CopyFromHost,
    W: ZnxWord,
{
    assert_eq!(src.n(), dst.n(), "transfer_into: ring degree");
    assert_eq!(src.rows(), dst.rows(), "transfer_into: rows");
    assert_eq!(src.cols_in(), dst.cols_in(), "transfer_into: cols_in");
    assert_eq!(src.cols_out(), dst.cols_out(), "transfer_into: cols_out");
    assert_eq!(src.size(), dst.size(), "transfer_into: size");
    transfer_buf_into(src.data(), dst.data_mut());
}

fn move_scalar_znx<D1, D2, W>(src: &ScalarZnx<D1, W>, dst: &mut ScalarZnx<D2, W>)
where
    D1: Data + CopyToHost,
    D2: Data + CopyFromHost,
    W: ZnxWord,
{
    assert!(src.n() == dst.n(), "transfer_into: ring degree");
    assert!(src.cols() == dst.cols(), "transfer_into: cols");
    transfer_buf_into(src.data(), dst.data_mut());
}

impl<D1, D2, W> TransferInto<GLWE<D2, W>> for GLWE<D1, W>
where
    D1: Data + CopyToHost,
    D2: Data + CopyFromHost,
    W: ZnxWord,
{
    fn transfer_into(&self, dst: &mut GLWE<D2, W>) {
        assert_eq!(self.base2k, dst.base2k, "transfer_into: GLWE base2k");
        assert_eq!(self.k, dst.k, "transfer_into: GLWE k");
        move_vec_znx(&self.data, &mut dst.data);
        dst.canonical = self.canonical;
        dst.noise = self.noise.clone();
        dst.plaintext_meta = self.plaintext_meta;
    }
}

impl<D1, D2, W> TransferInto<GLWEPlaintext<D2, W>> for GLWEPlaintext<D1, W>
where
    D1: Data + CopyToHost,
    D2: Data + CopyFromHost,
    W: ZnxWord,
{
    fn transfer_into(&self, dst: &mut GLWEPlaintext<D2, W>) {
        assert!(self.base2k == dst.base2k, "transfer_into: GLWEPlaintext base2k");
        assert!(self.k == dst.k, "transfer_into: GLWEPlaintext k");
        move_vec_znx(&self.data, &mut dst.data);
        dst.plaintext_meta = self.plaintext_meta;
    }
}

impl<D1, D2, W> TransferInto<LWEPlaintext<D2, W>> for LWEPlaintext<D1, W>
where
    D1: Data + CopyToHost,
    D2: Data + CopyFromHost,
    W: ZnxWord,
{
    fn transfer_into(&self, dst: &mut LWEPlaintext<D2, W>) {
        assert!(self.base2k == dst.base2k, "transfer_into: LWEPlaintext base2k");
        assert!(self.k == dst.k, "transfer_into: LWEPlaintext k");
        move_vec_znx(&self.data, &mut dst.data);
    }
}

impl<D1, D2, W> TransferInto<LWE<D2, W>> for LWE<D1, W>
where
    D1: Data + CopyToHost,
    D2: Data + CopyFromHost,
    W: ZnxWord,
{
    fn transfer_into(&self, dst: &mut LWE<D2, W>) {
        assert_eq!(self.base2k, dst.base2k, "transfer_into: LWE base2k");
        assert_eq!(self.k, dst.k, "transfer_into: LWE k");
        move_vec_znx(&self.body, &mut dst.body);
        move_vec_znx(&self.mask, &mut dst.mask);
        dst.noise = self.noise.clone();
    }
}

impl<D1, D2, W> TransferInto<GGLWE<D2, W>> for GGLWE<D1, W>
where
    D1: Data + CopyToHost,
    D2: Data + CopyFromHost,
    W: ZnxWord,
{
    fn transfer_into(&self, dst: &mut GGLWE<D2, W>) {
        assert_eq!(self.base2k, dst.base2k, "transfer_into: GGLWE base2k");
        assert_eq!(self.k_aux, dst.k_aux, "transfer_into: GGLWE k_aux");
        assert_eq!(self.k_digit, dst.k_digit, "transfer_into: GGLWE k_digit");
        assert_eq!(self.k_offset, dst.k_offset, "transfer_into: GGLWE k_offset");
        move_mat_znx(&self.data, &mut dst.data);
        dst.noise = self.noise.clone();
    }
}

impl<D1, D2, W> TransferInto<GGSW<D2, W>> for GGSW<D1, W>
where
    D1: Data + CopyToHost,
    D2: Data + CopyFromHost,
    W: ZnxWord,
{
    fn transfer_into(&self, dst: &mut GGSW<D2, W>) {
        assert_eq!(self.base2k, dst.base2k, "transfer_into: GGSW base2k");
        assert_eq!(self.k_aux, dst.k_aux, "transfer_into: GGSW k_aux");
        assert_eq!(self.k_digit, dst.k_digit, "transfer_into: GGSW k_digit");
        move_mat_znx(&self.data, &mut dst.data);
        dst.noise = self.noise.clone();
    }
}

impl<D1, D2, W> TransferInto<GLWECompressed<D2, W>> for GLWECompressed<D1, W>
where
    D1: Data + CopyToHost,
    D2: Data + CopyFromHost,
    W: ZnxWord,
{
    fn transfer_into(&self, dst: &mut GLWECompressed<D2, W>) {
        assert_eq!(self.base2k, dst.base2k, "transfer_into: GLWECompressed base2k");
        assert_eq!(self.k, dst.k, "transfer_into: GLWECompressed k");
        assert_eq!(self.rank, dst.rank, "transfer_into: GLWECompressed rank");
        move_vec_znx(&self.data, &mut dst.data);
        dst.seed = self.seed;
        dst.noise = self.noise.clone();
        dst.plaintext_meta = self.plaintext_meta;
    }
}

impl<D1, D2, W> TransferInto<GGLWECompressed<D2, W>> for GGLWECompressed<D1, W>
where
    D1: Data + CopyToHost,
    D2: Data + CopyFromHost,
    W: ZnxWord,
{
    fn transfer_into(&self, dst: &mut GGLWECompressed<D2, W>) {
        assert_eq!(self.base2k, dst.base2k, "transfer_into: GGLWECompressed base2k");
        assert_eq!(self.k_aux, dst.k_aux, "transfer_into: GGLWECompressed k_aux");
        assert_eq!(self.k_digit, dst.k_digit, "transfer_into: GGLWECompressed k_digit");
        assert_eq!(self.k_offset, dst.k_offset, "transfer_into: GGLWECompressed k_offset");
        assert_eq!(self.rank_out, dst.rank_out, "transfer_into: GGLWECompressed rank_out");
        assert_eq!(self.seed.len(), dst.seed.len(), "transfer_into: GGLWECompressed seed count");
        move_mat_znx(&self.data, &mut dst.data);
        dst.seed.copy_from_slice(&self.seed);
        dst.noise = self.noise.clone();
    }
}

impl<D1, D2, W> TransferInto<GLWEPublicKey<D2, W>> for GLWEPublicKey<D1, W>
where
    D1: Data + CopyToHost,
    D2: Data + CopyFromHost,
    W: ZnxWord,
{
    fn transfer_into(&self, dst: &mut GLWEPublicKey<D2, W>) {
        assert_eq!(self.base2k, dst.base2k, "transfer_into: GLWEPublicKey base2k");
        assert_eq!(self.k, dst.k, "transfer_into: GLWEPublicKey k");
        move_mat_znx(&self.data, &mut dst.data);
        dst.dist = self.dist;
        dst.noise = self.noise.clone();
    }
}

impl<D1, D2, W> TransferInto<GLWEPublicKeyCompressed<D2, W>> for GLWEPublicKeyCompressed<D1, W>
where
    D1: Data + CopyToHost,
    D2: Data + CopyFromHost,
    W: ZnxWord,
{
    fn transfer_into(&self, dst: &mut GLWEPublicKeyCompressed<D2, W>) {
        assert_eq!(self.base2k, dst.base2k, "transfer_into: GLWEPublicKeyCompressed base2k");
        assert_eq!(self.k, dst.k, "transfer_into: GLWEPublicKeyCompressed k");
        assert_eq!(
            self.seed.len(),
            dst.seed.len(),
            "transfer_into: GLWEPublicKeyCompressed seed count"
        );
        move_mat_znx(&self.data, &mut dst.data);
        dst.seed.copy_from_slice(&self.seed);
        dst.dist = self.dist;
        dst.noise = self.noise.clone();
    }
}

impl<D1, D2, W> TransferInto<GLWESwitchingKey<D2, W>> for GLWESwitchingKey<D1, W>
where
    D1: Data + CopyToHost,
    D2: Data + CopyFromHost,
    W: ZnxWord,
{
    fn transfer_into(&self, dst: &mut GLWESwitchingKey<D2, W>) {
        self.key.transfer_into(&mut dst.key);
        dst.input_degree = self.input_degree;
        dst.output_degree = self.output_degree;
    }
}

impl<D1, D2, W> TransferInto<GLWEAutomorphismKey<D2, W>> for GLWEAutomorphismKey<D1, W>
where
    D1: Data + CopyToHost,
    D2: Data + CopyFromHost,
    W: ZnxWord,
{
    fn transfer_into(&self, dst: &mut GLWEAutomorphismKey<D2, W>) {
        self.key.transfer_into(&mut dst.key);
        dst.p = self.p;
    }
}

impl<D1, D2, W> TransferInto<GLWESecret<D2, W>> for GLWESecret<D1, W>
where
    D1: Data + CopyToHost,
    D2: Data + CopyFromHost,
    W: ZnxWord,
{
    fn transfer_into(&self, dst: &mut GLWESecret<D2, W>) {
        move_scalar_znx(&self.data, &mut dst.data);
        dst.dist = self.dist;
    }
}

impl<D1, D2, W> TransferInto<LWESecret<D2, W>> for LWESecret<D1, W>
where
    D1: Data + CopyToHost,
    D2: Data + CopyFromHost,
    W: ZnxWord,
{
    fn transfer_into(&self, dst: &mut LWESecret<D2, W>) {
        move_scalar_znx(&self.data, &mut dst.data);
        dst.dist = self.dist;
    }
}

impl<D1, D2, W> TransferInto<GLWETensor<D2, W>> for GLWETensor<D1, W>
where
    D1: Data + CopyToHost,
    D2: Data + CopyFromHost,
    W: ZnxWord,
{
    fn transfer_into(&self, dst: &mut GLWETensor<D2, W>) {
        assert_eq!(self.base2k, dst.base2k, "transfer_into: GLWETensor base2k");
        assert_eq!(self.k, dst.k, "transfer_into: GLWETensor k");
        assert_eq!(self.rank, dst.rank, "transfer_into: GLWETensor rank");
        move_vec_znx(&self.data, &mut dst.data);
    }
}

impl<D1, D2, W> TransferInto<GLWETensorKey<D2, W>> for GLWETensorKey<D1, W>
where
    D1: Data + CopyToHost,
    D2: Data + CopyFromHost,
    W: ZnxWord,
{
    fn transfer_into(&self, dst: &mut GLWETensorKey<D2, W>) {
        self.0.transfer_into(&mut dst.0);
    }
}

impl<D1, D2, W> TransferInto<GLWEToLWEKey<D2, W>> for GLWEToLWEKey<D1, W>
where
    D1: Data + CopyToHost,
    D2: Data + CopyFromHost,
    W: ZnxWord,
{
    fn transfer_into(&self, dst: &mut GLWEToLWEKey<D2, W>) {
        self.0.transfer_into(&mut dst.0);
    }
}

impl<D1, D2, W> TransferInto<GGLWEToGGSWKey<D2, W>> for GGLWEToGGSWKey<D1, W>
where
    D1: Data + CopyToHost,
    D2: Data + CopyFromHost,
    W: ZnxWord,
{
    fn transfer_into(&self, dst: &mut GGLWEToGGSWKey<D2, W>) {
        assert_eq!(self.keys.len(), dst.keys.len(), "transfer_into: GGLWEToGGSWKey key count");
        for (src, dst) in self.keys.iter().zip(&mut dst.keys) {
            src.transfer_into(dst);
        }
    }
}

#[cfg(test)]
mod tests {
    use poulpy_hal::layouts::{Backend, DataView, DataViewMut, Module};

    use super::TransferInto;
    use crate::{
        ComponentNoise, Distribution, FreshNoiseEstimate,
        layouts::{
            Base2K, Dnum, GGLWE, GGLWELayout, GLWE, GLWEPlaintextMeta, ModuleCoreAlloc, ModuleCoreCompressedAlloc, Rank, Scale,
            TorusPrecision,
        },
        test_backend::{DstBackend, SrcBackend},
        test_suite::refusal::assert_refusal,
    };

    fn fill_bytes(buf: &mut [u8]) {
        for (i, byte) in buf.iter_mut().enumerate() {
            *byte = (i as u8).wrapping_mul(17).wrapping_add(3);
        }
    }

    #[test]
    fn module_transfer_glwe_roundtrip() {
        let src_module: Module<SrcBackend> = Module::new(64);
        let dst_module: Module<DstBackend> = Module::new(64);
        let mut src: GLWE<<SrcBackend as Backend>::OwnedBuf, <SrcBackend as Backend>::ZnxWord> =
            src_module.glwe_alloc(Base2K(12), TorusPrecision(33), Rank(2));
        fill_bytes(src.data.data_mut());
        src.noise = Some(
            ComponentNoise::from_secret_at(Distribution::TernaryProb(0.3), TorusPrecision(33), 2).with_components(
                [1250.0, 3.0, 7.0]
                    .map(|variance| FreshNoiseEstimate::new(variance, TorusPrecision(47)))
                    .to_vec(),
            ),
        );

        let mut uploaded = dst_module.glwe_alloc_from_infos(&src);
        src.transfer_into(&mut uploaded);
        let mut downloaded = src_module.glwe_alloc_from_infos(&src);
        uploaded.transfer_into(&mut downloaded);

        assert_eq!(downloaded, src);
    }

    #[test]
    fn module_transfer_gglwe_roundtrip() {
        let src_module: Module<SrcBackend> = Module::new(64);
        let dst_module: Module<DstBackend> = Module::new(64);
        let mut src: GGLWE<<SrcBackend as Backend>::OwnedBuf, <SrcBackend as Backend>::ZnxWord> = src_module
            .gglwe_alloc_from_infos(&GGLWELayout {
                n: 64usize.into(),
                base2k: Base2K(12),
                dnum: Dnum(3),
                k_digit: TorusPrecision(12),
                k_aux: TorusPrecision(12 + 6),
                k_offset: TorusPrecision(5),
                rank_in: Rank(1),
                rank_out: Rank(2),
                stride: 1,
            });
        fill_bytes(src.data.data_mut());
        src.noise = Some(
            ComponentNoise::from_secret_at(Distribution::BinaryFixed(11), TorusPrecision(54), 2).with_components(
                [2300.0, 5.0, 9.0]
                    .map(|variance| FreshNoiseEstimate::new(variance, TorusPrecision(61)))
                    .to_vec(),
            ),
        );

        let mut uploaded = dst_module.gglwe_alloc_from_infos(&src);
        src.transfer_into(&mut uploaded);
        let mut downloaded = src_module.gglwe_alloc_from_infos(&src);
        uploaded.transfer_into(&mut downloaded);

        assert_eq!(downloaded, src);
    }

    /// Noise metadata distinct per `seed`, so a receiver's stale estimate differs from the source's.
    fn noise(seed: f64, rank: usize) -> Option<ComponentNoise> {
        Some(
            ComponentNoise::from_secret_at(Distribution::TernaryProb(0.3), TorusPrecision(33), rank).with_components(
                (0..=rank)
                    .map(|i| FreshNoiseEstimate::new(seed + i as f64, TorusPrecision(47)))
                    .collect(),
            ),
        )
    }

    fn other_bytes(buf: &mut [u8]) {
        buf.iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = (i as u8).wrapping_mul(29).wrapping_add(11));
    }

    #[test]
    fn module_transfer_glwe_compressed_roundtrip() {
        let src_module: Module<SrcBackend> = Module::new(64);
        let dst_module: Module<DstBackend> = Module::new(64);
        let mut src = src_module.glwe_compressed_alloc(Base2K(12), TorusPrecision(33), Rank(2));
        fill_bytes(src.data.data_mut());
        src.seed = [7; 32];
        src.noise = noise(1250.0, 2);
        src.plaintext_meta = Some(GLWEPlaintextMeta {
            scale: Scale::Log(17),
            ..Default::default()
        });

        let mut uploaded = dst_module.glwe_compressed_alloc_from_infos(&src);
        other_bytes(uploaded.data.data_mut());
        uploaded.seed = [9; 32];
        uploaded.noise = noise(3.0, 2);
        src.transfer_into(&mut uploaded);
        let mut downloaded = src_module.glwe_compressed_alloc_from_infos(&src);
        other_bytes(downloaded.data.data_mut());
        downloaded.seed = [9; 32];
        downloaded.noise = noise(3.0, 2);
        uploaded.transfer_into(&mut downloaded);

        assert_eq!(downloaded, src);
    }

    #[test]
    fn module_transfer_gglwe_compressed_roundtrip() {
        let src_module: Module<SrcBackend> = Module::new(64);
        let dst_module: Module<DstBackend> = Module::new(64);
        let layout = GGLWELayout {
            n: 64usize.into(),
            base2k: Base2K(12),
            dnum: Dnum(3),
            k_digit: TorusPrecision(12),
            k_aux: TorusPrecision(18),
            k_offset: TorusPrecision(5),
            rank_in: Rank(1),
            rank_out: Rank(2),
            stride: 1,
        };
        let mut src = src_module.gglwe_compressed_alloc_from_infos(&layout);
        fill_bytes(src.data.data_mut());
        src.seed
            .iter_mut()
            .enumerate()
            .for_each(|(i, seed)| *seed = [i as u8 + 1; 32]);
        src.noise = noise(2300.0, 2);

        let mut uploaded = dst_module.gglwe_compressed_alloc_from_infos(&src);
        other_bytes(uploaded.data.data_mut());
        uploaded.seed.fill([9; 32]);
        uploaded.noise = noise(5.0, 2);
        src.transfer_into(&mut uploaded);
        let mut downloaded = src_module.gglwe_compressed_alloc_from_infos(&src);
        other_bytes(downloaded.data.data_mut());
        downloaded.seed.fill([9; 32]);
        downloaded.noise = noise(5.0, 2);
        uploaded.transfer_into(&mut downloaded);

        assert_eq!(downloaded, src);
    }

    #[test]
    fn module_transfer_glwe_public_key_roundtrip() {
        let src_module: Module<SrcBackend> = Module::new(64);
        let dst_module: Module<DstBackend> = Module::new(64);
        let mut src = src_module.glwe_public_key_alloc(Base2K(12), TorusPrecision(33), Rank(2));
        fill_bytes(src.data.data_mut());
        src.dist = Distribution::TernaryProb(0.3);
        src.noise = noise(1250.0, 2);

        let mut uploaded = dst_module.glwe_public_key_alloc_from_infos(&src);
        other_bytes(uploaded.data.data_mut());
        uploaded.dist = Distribution::BinaryProb(0.5);
        uploaded.noise = noise(3.0, 2);
        src.transfer_into(&mut uploaded);
        let mut downloaded = src_module.glwe_public_key_alloc_from_infos(&src);
        other_bytes(downloaded.data.data_mut());
        downloaded.dist = Distribution::BinaryProb(0.5);
        downloaded.noise = noise(3.0, 2);
        uploaded.transfer_into(&mut downloaded);

        assert_eq!(downloaded, src);
    }

    #[test]
    fn module_transfer_glwe_public_key_compressed_roundtrip() {
        let src_module: Module<SrcBackend> = Module::new(64);
        let dst_module: Module<DstBackend> = Module::new(64);
        let mut src = src_module.glwe_public_key_compressed_alloc(Base2K(12), TorusPrecision(33), Rank(2));
        fill_bytes(src.data.data_mut());
        src.seed
            .iter_mut()
            .enumerate()
            .for_each(|(i, seed)| *seed = [i as u8 + 1; 32]);
        src.dist = Distribution::TernaryProb(0.3);
        src.noise = noise(1250.0, 2);

        let mut uploaded = dst_module.glwe_public_key_compressed_alloc_from_infos(&src);
        other_bytes(uploaded.data.data_mut());
        uploaded.seed.fill([9; 32]);
        uploaded.dist = Distribution::BinaryProb(0.5);
        uploaded.noise = noise(3.0, 2);
        src.transfer_into(&mut uploaded);
        let mut downloaded = src_module.glwe_public_key_compressed_alloc_from_infos(&src);
        other_bytes(downloaded.data.data_mut());
        downloaded.seed.fill([9; 32]);
        downloaded.dist = Distribution::BinaryProb(0.5);
        downloaded.noise = noise(3.0, 2);
        uploaded.transfer_into(&mut downloaded);

        assert_eq!(downloaded, src);
    }

    /// Moves `src` into `dst` and asserts the move is refused with `message` and `snapshot(dst)` is as it was.
    fn assert_transfer_refused<S, D, V>(src: &S, dst: &mut D, message: &str, snapshot: impl Fn(&D) -> V)
    where
        S: TransferInto<D>,
        V: PartialEq + std::fmt::Debug,
    {
        assert_refusal(message, dst, snapshot, |dst| src.transfer_into(dst));
    }

    /// A layout move checks the whole shape, not just the byte count.
    #[test]
    fn test_transfer_into_rejects_shape_mismatch() {
        let src_module: Module<SrcBackend> = Module::new(64);
        let dst_module: Module<DstBackend> = Module::new(64);
        let src = src_module.glwe_alloc(Base2K(12), TorusPrecision(24), Rank(1));
        let mut dst = dst_module.glwe_alloc(Base2K(12), TorusPrecision(36), Rank(1));
        fill_bytes(dst.data.data_mut());
        assert_transfer_refused(
            &src,
            &mut dst,
            "assertion `left == right` failed: transfer_into: GLWE k\n  left: TorusPrecision(24)\n right: TorusPrecision(36)",
            |dst| dst.clone(),
        );
    }

    /// The compressed GLWE body has one column whatever the rank, so the move checks the rank itself.
    #[test]
    fn test_transfer_into_rejects_glwe_compressed_rank_mismatch() {
        let src_module: Module<SrcBackend> = Module::new(64);
        let dst_module: Module<DstBackend> = Module::new(64);
        let src = src_module.glwe_compressed_alloc(Base2K(12), TorusPrecision(24), Rank(1));
        let mut dst = dst_module.glwe_compressed_alloc(Base2K(12), TorusPrecision(24), Rank(2));
        fill_bytes(dst.data.data_mut());
        assert_transfer_refused(
            &src,
            &mut dst,
            "assertion `left == right` failed: transfer_into: GLWECompressed rank\n  left: Rank(1)\n right: Rank(2)",
            |dst| dst.clone(),
        );
    }

    /// A row offset that leaves the limb count unchanged still differs.
    #[test]
    fn test_transfer_into_rejects_gglwe_compressed_offset_mismatch() {
        let src_module: Module<SrcBackend> = Module::new(64);
        let dst_module: Module<DstBackend> = Module::new(64);
        let layout = GGLWELayout {
            n: 64usize.into(),
            base2k: Base2K(12),
            dnum: Dnum(2),
            k_digit: TorusPrecision(12),
            k_aux: TorusPrecision(18),
            k_offset: TorusPrecision(0),
            rank_in: Rank(1),
            rank_out: Rank(1),
            stride: 1,
        };
        let src = src_module.gglwe_compressed_alloc_from_infos(&layout);
        let mut dst = dst_module.gglwe_compressed_alloc_from_infos(&GGLWELayout {
            k_offset: TorusPrecision(1),
            ..layout
        });
        fill_bytes(dst.data.data_mut());
        assert_eq!(src.data.size(), dst.data.size());
        assert_transfer_refused(
            &src,
            &mut dst,
            "assertion `left == right` failed: transfer_into: GGLWECompressed k_offset\n  left: TorusPrecision(0)\n right: TorusPrecision(1)",
            |dst| dst.clone(),
        );
    }

    /// A row offset that leaves the limb count unchanged still differs.
    #[test]
    fn test_transfer_into_rejects_gglwe_offset_mismatch() {
        let src_module: Module<SrcBackend> = Module::new(64);
        let dst_module: Module<DstBackend> = Module::new(64);
        let layout = GGLWELayout {
            n: 64usize.into(),
            base2k: Base2K(12),
            dnum: Dnum(2),
            k_digit: TorusPrecision(12),
            k_aux: TorusPrecision(18),
            k_offset: TorusPrecision(0),
            rank_in: Rank(1),
            rank_out: Rank(1),
            stride: 1,
        };
        let src = src_module.gglwe_alloc_from_infos(&layout);
        let mut dst = dst_module.gglwe_alloc_from_infos(&GGLWELayout {
            k_offset: TorusPrecision(1),
            ..layout
        });
        fill_bytes(dst.data.data_mut());
        assert_eq!(src.data.size(), dst.data.size());
        assert_transfer_refused(
            &src,
            &mut dst,
            "assertion `left == right` failed: transfer_into: GGLWE k_offset\n  left: TorusPrecision(0)\n right: TorusPrecision(1)",
            |dst| dst.clone(),
        );
    }

    #[test]
    fn test_transfer_into_rejects_glwe_public_key_precision_mismatch() {
        let src_module: Module<SrcBackend> = Module::new(64);
        let dst_module: Module<DstBackend> = Module::new(64);
        let src = src_module.glwe_public_key_alloc(Base2K(12), TorusPrecision(24), Rank(1));
        let mut dst = dst_module.glwe_public_key_alloc(Base2K(12), TorusPrecision(23), Rank(1));
        fill_bytes(dst.data.data_mut());
        assert_transfer_refused(
            &src,
            &mut dst,
            "assertion `left == right` failed: transfer_into: GLWEPublicKey k\n  left: TorusPrecision(24)\n right: TorusPrecision(23)",
            |dst| (dst.data.data().clone(), dst.noise.clone(), dst.dist),
        );
    }

    #[test]
    fn test_transfer_into_rejects_glwe_public_key_compressed_precision_mismatch() {
        let src_module: Module<SrcBackend> = Module::new(64);
        let dst_module: Module<DstBackend> = Module::new(64);
        let src = src_module.glwe_public_key_compressed_alloc(Base2K(12), TorusPrecision(24), Rank(1));
        let mut dst = dst_module.glwe_public_key_compressed_alloc(Base2K(12), TorusPrecision(23), Rank(1));
        fill_bytes(dst.data.data_mut());
        assert_transfer_refused(
            &src,
            &mut dst,
            "assertion `left == right` failed: transfer_into: GLWEPublicKeyCompressed k\n  left: TorusPrecision(24)\n right: TorusPrecision(23)",
            |dst| (dst.data.data().clone(), dst.seed.clone(), dst.noise.clone(), dst.dist),
        );
    }
}
