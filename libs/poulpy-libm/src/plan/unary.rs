//! Direct unary operation macros.

macro_rules! define_unary_plan {
    ($plan:ident, $name:literal) => {
        #[doc = concat!("Prepared `", $name, "` approximation.")]
        pub struct $plan<P> {
            pub(crate) approximation: $crate::approximation::PolynomialApproximation<P>,
            pub(crate) approximation_bits: f64,
        }

        impl<P> $plan<P> {
            /// Prepared polynomial.
            pub fn approximation(&self) -> &$crate::approximation::PolynomialApproximation<P> {
                &self.approximation
            }

            /// Fitted absolute-error bits.
            pub fn approximation_bits(&self) -> f64 {
                self.approximation_bits
            }

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
        fn $tmp<R, I, T, P>(&self, res: &R, input: &I, tsk: &T, plan: &$plan<P>) -> usize
        where
            R: poulpy_ckks::CKKSCtBounds,
            I: poulpy_ckks::CKKSCtBounds,
            T: poulpy_core::layouts::GGLWEInfos,
            P: poulpy_ckks::CKKSInfos + poulpy_core::layouts::LWEInfos;

        #[doc = concat!("Evaluates `", $name, "(input)`.")]
        ///
        /// The slots must lie in `plan.interval()`. `input` needs `plan.consumed_bits(input.log_delta())` bits of budget,
        /// plus the headroom of the result: the remaining budget `b` is guaranteed to hold a slot value `v` with
        /// `|v| < 2^(b - 1)`, and a larger value can wrap without error.
        ///
        /// # Metadata
        ///
        /// As `ckks_eval_approximation`.
        ///
        /// # Errors
        ///
        /// The errors of `ckks_eval_approximation`, with its guarantee on `res`.
        fn $eval<R, I, P, H>(
            &self,
            res: &mut R,
            input: &I,
            plan: &$plan<P>,
            tsk: &H,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, BE>,
        ) -> anyhow::Result<()>
        where
            R: poulpy_core::layouts::GLWEToBackendMut<BE>
                + poulpy_ckks::CKKSCtBounds
                + poulpy_ckks::SetCKKSInfos
                + poulpy_core::layouts::SetBSGSMeta,
            I: poulpy_core::layouts::GLWEToBackendRef<BE> + poulpy_ckks::CKKSCtBounds,
            P: poulpy_core::layouts::GLWEToBackendRef<BE>
                + poulpy_ckks::CKKSCtBounds
                + poulpy_core::layouts::BSGSMeta
                + poulpy_core::layouts::IntPolyInfos
                + poulpy_ckks::layouts::CKKSMaskedConstants<BE>,
            H: poulpy_core::layouts::GetTensorKey<BE>;
    };
}

macro_rules! impl_unary_op {
    ($tmp:ident, $eval:ident, $plan:ident) => {
        fn $tmp<R, I, T, P>(&self, res: &R, input: &I, tsk: &T, plan: &$plan<P>) -> usize
        where
            R: poulpy_ckks::CKKSCtBounds,
            I: poulpy_ckks::CKKSCtBounds,
            T: poulpy_core::layouts::GGLWEInfos,
            P: poulpy_ckks::CKKSInfos + poulpy_core::layouts::LWEInfos,
        {
            self.ckks_approximation_tmp_bytes(res, input, &plan.approximation, tsk)
        }

        fn $eval<R, I, P, H>(
            &self,
            res: &mut R,
            input: &I,
            plan: &$plan<P>,
            tsk: &H,
            scratch: &mut poulpy_hal::layouts::ScratchArena<'_, BE>,
        ) -> anyhow::Result<()>
        where
            R: poulpy_core::layouts::GLWEToBackendMut<BE>
                + poulpy_ckks::CKKSCtBounds
                + poulpy_ckks::SetCKKSInfos
                + poulpy_core::layouts::SetBSGSMeta,
            I: poulpy_core::layouts::GLWEToBackendRef<BE> + poulpy_ckks::CKKSCtBounds,
            P: poulpy_core::layouts::GLWEToBackendRef<BE>
                + poulpy_ckks::CKKSCtBounds
                + poulpy_core::layouts::BSGSMeta
                + poulpy_core::layouts::IntPolyInfos
                + poulpy_ckks::layouts::CKKSMaskedConstants<BE>,
            H: poulpy_core::layouts::GetTensorKey<BE>,
        {
            anyhow::Context::context(
                self.ckks_eval_approximation(res, input, &plan.approximation, tsk, scratch),
                stringify!($eval),
            )
        }
    };
}

pub(crate) use {declare_unary_op, define_unary_plan, impl_unary_op};
