use poulpy_hal::layouts::{Backend, DataView, MatZnx, Module, ScalarZnx, TransferFrom, VecZnx};

use crate::layouts::{
    BackendGGLWE, BackendGGSW, BackendGLWE, BackendGLWEPlaintext, BackendGLWESecret, BackendLWE, BackendLWEPlaintext,
    BackendLWESecret, GGLWE, GGSW, GLWE, GLWEPlaintext, GLWESecret, LWE, LWEPlaintext, LWESecret,
};

fn transfer_vec_znx<From, To>(src: &VecZnx<From::OwnedBuf>) -> VecZnx<To::OwnedBuf>
where
    From: Backend,
    To: Backend + TransferFrom<From>,
{
    VecZnx::from_data_with_max_size(
        <To as TransferFrom<From>>::transfer_buf(src.data()),
        src.n(),
        src.cols(),
        src.size(),
        src.max_size(),
    )
}

fn transfer_mat_znx<From, To>(src: &MatZnx<From::OwnedBuf>) -> MatZnx<To::OwnedBuf>
where
    From: Backend,
    To: Backend + TransferFrom<From>,
{
    MatZnx::from_data(
        <To as TransferFrom<From>>::transfer_buf(src.data()),
        src.n(),
        src.rows(),
        src.cols_in(),
        src.cols_out(),
        src.size(),
    )
}

fn transfer_scalar_znx<From, To>(src: &ScalarZnx<From::OwnedBuf>) -> ScalarZnx<To::OwnedBuf>
where
    From: Backend,
    To: Backend + TransferFrom<From>,
{
    ScalarZnx::from_data(<To as TransferFrom<From>>::transfer_buf(src.data()), src.n(), src.cols())
}

pub trait ModuleTransfer<To: Backend> {
    fn upload_glwe<From>(&self, src: &BackendGLWE<From>) -> BackendGLWE<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn download_glwe<From>(&self, src: &BackendGLWE<From>) -> BackendGLWE<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn upload_lwe<From>(&self, src: &BackendLWE<From>) -> BackendLWE<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn download_lwe<From>(&self, src: &BackendLWE<From>) -> BackendLWE<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn upload_gglwe<From>(&self, src: &BackendGGLWE<From>) -> BackendGGLWE<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn download_gglwe<From>(&self, src: &BackendGGLWE<From>) -> BackendGGLWE<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn upload_ggsw<From>(&self, src: &BackendGGSW<From>) -> BackendGGSW<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn download_ggsw<From>(&self, src: &BackendGGSW<From>) -> BackendGGSW<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn upload_glwe_secret<From>(&self, src: &BackendGLWESecret<From>) -> BackendGLWESecret<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn download_glwe_secret<From>(&self, src: &BackendGLWESecret<From>) -> BackendGLWESecret<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn upload_lwe_secret<From>(&self, src: &BackendLWESecret<From>) -> BackendLWESecret<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn download_lwe_secret<From>(&self, src: &BackendLWESecret<From>) -> BackendLWESecret<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn upload_glwe_plaintext<From>(&self, src: &BackendGLWEPlaintext<From>) -> BackendGLWEPlaintext<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn download_glwe_plaintext<From>(&self, src: &BackendGLWEPlaintext<From>) -> BackendGLWEPlaintext<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn upload_lwe_plaintext<From>(&self, src: &BackendLWEPlaintext<From>) -> BackendLWEPlaintext<To>
    where
        From: Backend,
        To: TransferFrom<From>;

    fn download_lwe_plaintext<From>(&self, src: &BackendLWEPlaintext<From>) -> BackendLWEPlaintext<To>
    where
        From: Backend,
        To: TransferFrom<From>;
}

impl<To: Backend> ModuleTransfer<To> for Module<To> {
    fn upload_glwe<From>(&self, src: &BackendGLWE<From>) -> BackendGLWE<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        let _ = self;
        GLWE {
            data: transfer_vec_znx::<From, To>(&src.data),
            base2k: src.base2k,
        }
    }

    fn download_glwe<From>(&self, src: &BackendGLWE<From>) -> BackendGLWE<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        self.upload_glwe(src)
    }

    fn upload_lwe<From>(&self, src: &BackendLWE<From>) -> BackendLWE<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        let _ = self;
        LWE {
            data: transfer_vec_znx::<From, To>(&src.data),
            base2k: src.base2k,
        }
    }

    fn download_lwe<From>(&self, src: &BackendLWE<From>) -> BackendLWE<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        self.upload_lwe(src)
    }

    fn upload_gglwe<From>(&self, src: &BackendGGLWE<From>) -> BackendGGLWE<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        let _ = self;
        GGLWE {
            data: transfer_mat_znx::<From, To>(&src.data),
            base2k: src.base2k,
            dsize: src.dsize,
        }
    }

    fn download_gglwe<From>(&self, src: &BackendGGLWE<From>) -> BackendGGLWE<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        self.upload_gglwe(src)
    }

    fn upload_ggsw<From>(&self, src: &BackendGGSW<From>) -> BackendGGSW<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        let _ = self;
        GGSW {
            data: transfer_mat_znx::<From, To>(&src.data),
            base2k: src.base2k,
            dsize: src.dsize,
        }
    }

    fn download_ggsw<From>(&self, src: &BackendGGSW<From>) -> BackendGGSW<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        self.upload_ggsw(src)
    }

    fn upload_glwe_secret<From>(&self, src: &BackendGLWESecret<From>) -> BackendGLWESecret<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        let _ = self;
        GLWESecret {
            data: transfer_scalar_znx::<From, To>(&src.data),
            dist: src.dist,
        }
    }

    fn download_glwe_secret<From>(&self, src: &BackendGLWESecret<From>) -> BackendGLWESecret<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        self.upload_glwe_secret(src)
    }

    fn upload_lwe_secret<From>(&self, src: &BackendLWESecret<From>) -> BackendLWESecret<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        let _ = self;
        LWESecret {
            data: transfer_scalar_znx::<From, To>(&src.data),
            dist: src.dist,
        }
    }

    fn download_lwe_secret<From>(&self, src: &BackendLWESecret<From>) -> BackendLWESecret<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        self.upload_lwe_secret(src)
    }

    fn upload_glwe_plaintext<From>(&self, src: &BackendGLWEPlaintext<From>) -> BackendGLWEPlaintext<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        let _ = self;
        GLWEPlaintext {
            data: transfer_vec_znx::<From, To>(&src.data),
            base2k: src.base2k,
        }
    }

    fn download_glwe_plaintext<From>(&self, src: &BackendGLWEPlaintext<From>) -> BackendGLWEPlaintext<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        self.upload_glwe_plaintext(src)
    }

    fn upload_lwe_plaintext<From>(&self, src: &BackendLWEPlaintext<From>) -> BackendLWEPlaintext<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        let _ = self;
        LWEPlaintext {
            data: transfer_vec_znx::<From, To>(&src.data),
            base2k: src.base2k,
        }
    }

    fn download_lwe_plaintext<From>(&self, src: &BackendLWEPlaintext<From>) -> BackendLWEPlaintext<To>
    where
        From: Backend,
        To: TransferFrom<From>,
    {
        self.upload_lwe_plaintext(src)
    }
}
