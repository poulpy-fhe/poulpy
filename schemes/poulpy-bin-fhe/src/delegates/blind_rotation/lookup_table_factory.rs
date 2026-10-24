use crate::api::LookupTableFactory;
use crate::blind_rotation::LookupTable;
use crate::oep::LookupTableFactoryImpl;
use poulpy_hal::layouts::Module;

impl<BE: LookupTableFactoryImpl> LookupTableFactory<BE::OwnedBuf, BE::ZnxWord> for Module<BE> {
    fn lookup_table_set(&self, res: &mut LookupTable<BE::OwnedBuf, BE::ZnxWord>, f: &[i64], k: usize) {
        BE::lookup_table_set(self, res, f, k)
    }

    fn lookup_table_rotate(&self, k: i64, res: &mut LookupTable<BE::OwnedBuf, BE::ZnxWord>) {
        BE::lookup_table_rotate(self, k, res)
    }
}
