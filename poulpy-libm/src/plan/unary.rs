//! Direct unary operation macros.

macro_rules! define_unary_plan {
    ($plan:ident, $name:literal) => {
        #[doc = concat!("Prepared `", $name, "` approximation.")]
        pub struct $plan<P> {
            /// Prepared polynomial.
            pub approximation: $crate::approximation::PolynomialApproximation<P>,
            /// Fitted absolute-error bits.
            pub approximation_bits: f64,
        }

        impl<P> $plan<P> {
            /// Consumed modulus bits.
            pub fn consumed_bits(&self, input_log_delta: usize) -> usize {
                self.approximation.consumed_bits(input_log_delta)
            }

            /// Multiplicative depth.
            pub fn depth(&self) -> usize {
                self.approximation.depth()
            }

            /// Input interval.
            pub fn interval(&self) -> (f64, f64) {
                self.approximation.interval()
            }

            /// Polynomial degree.
            pub fn degree(&self) -> usize {
                self.approximation.degree()
            }

            /// Maps prepared plaintexts.
            pub fn map_plaintexts<Q>(self, f: impl FnMut(&P) -> Q) -> $plan<Q> {
                $plan {
                    approximation: self.approximation.map_plaintexts(f),
                    approximation_bits: self.approximation_bits,
                }
            }
        }
    };
}

macro_rules! declare_unary_op {
    ($tmp:ident, $eval:ident, $plan:ident, $name:literal) => {
        #[doc = concat!("Scratch bytes for [`Self::", stringify!($eval), "`].")]
        fn $tmp<R, T, P>(&self, res: &R, tsk: &T, plan: &$plan<P>) -> usize
        where
            R: poulpy_ckks::CKKSCtBounds,
            T: poulpy_core::layouts::GGLWEInfos,
            P: poulpy_ckks::CKKSInfos + poulpy_core::layouts::LWEInfos;

        #[doc = concat!("Evaluates `", $name, "(input)`.")]
        fn $eval<P>(
            &self,
            res: &mut poulpy_ckks::layouts::CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
            input: &poulpy_ckks::layouts::CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
            plan: &$plan<P>,
            tsk: &poulpy_core::layouts::GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, BE>,
        ) -> anyhow::Result<()>
        where
            P: poulpy_core::layouts::GLWEToBackendRef<BE>
                + poulpy_ckks::CKKSCtBounds
                + poulpy_core::layouts::BSGSMeta
                + poulpy_core::layouts::IntPolyInfos;
    };
}

macro_rules! impl_unary_op {
    ($tmp:ident, $eval:ident, $plan:ident) => {
        fn $tmp<R, T, P>(&self, res: &R, tsk: &T, plan: &$plan<P>) -> usize
        where
            R: poulpy_ckks::CKKSCtBounds,
            T: poulpy_core::layouts::GGLWEInfos,
            P: poulpy_ckks::CKKSInfos + poulpy_core::layouts::LWEInfos,
        {
            self.ckks_approximation_tmp_bytes(res, res, tsk, &plan.approximation)
        }

        fn $eval<P>(
            &self,
            res: &mut poulpy_ckks::layouts::CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
            input: &poulpy_ckks::layouts::CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
            plan: &$plan<P>,
            tsk: &poulpy_core::layouts::GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, BE>,
        ) -> anyhow::Result<()>
        where
            P: poulpy_core::layouts::GLWEToBackendRef<BE>
                + poulpy_ckks::CKKSCtBounds
                + poulpy_core::layouts::BSGSMeta
                + poulpy_core::layouts::IntPolyInfos,
        {
            self.ckks_eval_approximation(res, input, &plan.approximation, tsk, scratch)
                .map_err(|e| anyhow::anyhow!("{}: {e}", stringify!($eval)))
        }
    };
}

pub(crate) use {declare_unary_op, define_unary_plan, impl_unary_op};
