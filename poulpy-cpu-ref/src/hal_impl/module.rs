#[macro_export]
macro_rules! hal_impl_module {
    ($defaults:ident) => {
        fn new(n: u64) -> Module<Self> {
            <Self as $defaults<Self>>::module_new_default(n)
        }
    };
}
