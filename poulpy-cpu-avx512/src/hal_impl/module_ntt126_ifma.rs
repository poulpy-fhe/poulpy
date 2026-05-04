macro_rules! hal_impl_module_ntt126_ifma {
    () => {
        fn new(n: u64) -> Module<Self> {
            crate::ntt126_ifma::module::module_new(n)
        }
    };
}
