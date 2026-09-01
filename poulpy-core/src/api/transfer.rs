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
    CopyFromHost, CopyToHost, Data, DataView, DataViewMut, MatZnx, NormalizationState, ScalarZnx, VecZnx, ZnxWord,
    transfer_buf_into,
};

use crate::layouts::{
    GGLWE, GGLWEToGGSWKey, GGSW, GLWE, GLWEAutomorphismKey, GLWEPlaintext, GLWESecret, GLWESwitchingKey, GLWETensor,
    GLWETensorKey, GLWEToLWEKey, LWE, LWEPlaintext, LWESecret,
};

/// Moves `Self` into an already-allocated destination.
pub trait TransferInto<Dst> {
    /// # Panics
    ///
    /// If the two values do not agree on shape.
    fn transfer_into(&self, dst: &mut Dst);
}

fn move_vec_znx<D1, D2, W, S>(src: &VecZnx<D1, W, S>, dst: &mut VecZnx<D2, W, S>)
where
    D1: Data + CopyToHost,
    D2: Data + CopyFromHost,
    W: ZnxWord,
    S: NormalizationState,
{
    assert_eq!(src.n(), dst.n(), "transfer_into: ring degree");
    assert_eq!(src.cols(), dst.cols(), "transfer_into: cols");
    assert_eq!(src.size(), dst.size(), "transfer_into: size");
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
    assert_eq!(src.n(), dst.n(), "transfer_into: ring degree");
    assert_eq!(src.cols(), dst.cols(), "transfer_into: cols");
    transfer_buf_into(src.data(), dst.data_mut());
}

impl<D1, D2, W, S> TransferInto<GLWE<D2, W, S>> for GLWE<D1, W, S>
where
    S: NormalizationState,
    D1: Data + CopyToHost,
    D2: Data + CopyFromHost,
    W: ZnxWord,
{
    fn transfer_into(&self, dst: &mut GLWE<D2, W, S>) {
        assert_eq!(self.base2k, dst.base2k, "transfer_into: GLWE base2k");
        assert_eq!(self.k, dst.k, "transfer_into: GLWE k");
        move_vec_znx(&self.data, &mut dst.data);
    }
}

impl<D1, D2, W> TransferInto<GLWEPlaintext<D2, W>> for GLWEPlaintext<D1, W>
where
    D1: Data + CopyToHost,
    D2: Data + CopyFromHost,
    W: ZnxWord,
{
    fn transfer_into(&self, dst: &mut GLWEPlaintext<D2, W>) {
        assert_eq!(self.base2k, dst.base2k, "transfer_into: GLWEPlaintext base2k");
        assert_eq!(self.k, dst.k, "transfer_into: GLWEPlaintext k");
        move_vec_znx(&self.data, &mut dst.data);
    }
}

impl<D1, D2, W> TransferInto<LWEPlaintext<D2, W>> for LWEPlaintext<D1, W>
where
    D1: Data + CopyToHost,
    D2: Data + CopyFromHost,
    W: ZnxWord,
{
    fn transfer_into(&self, dst: &mut LWEPlaintext<D2, W>) {
        assert_eq!(self.base2k, dst.base2k, "transfer_into: LWEPlaintext base2k");
        assert_eq!(self.k, dst.k, "transfer_into: LWEPlaintext k");
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
        assert_eq!(self.dsize, dst.dsize, "transfer_into: GGLWE dsize");
        move_mat_znx(&self.data, &mut dst.data);
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
        assert_eq!(self.dsize, dst.dsize, "transfer_into: GGSW dsize");
        move_mat_znx(&self.data, &mut dst.data);
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
