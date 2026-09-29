use poulpy_core::layouts::{GGLWECompressedSeed, GGSWInfos};
use poulpy_hal::layouts::Module;

use crate::{layouts::GGSWShareOwned, oep::GGLWEPatCompressedImpl};

pub(crate) fn mhe_ggsw_share_aggregate_derived<BE: GGLWEPatCompressedImpl>(
    module: &Module<BE>,
    res: &mut GGSWShareOwned<BE>,
    a: &GGSWShareOwned<BE>,
) {
    assert!(res.ggsw_layout() == a.ggsw_layout(), "invalid aggregation: layouts differ");
    // Every part is checked before any is summed, so a rejected share leaves `res` unchanged.
    for (res_part, a_part) in res.parts().zip(a.parts()) {
        assert!(res_part.seed() == a_part.seed(), "invalid aggregation: seeds differ");
    }
    for (res_part, a_part) in res.parts_mut().zip(a.parts()) {
        BE::gglwe_pat_compressed_aggregate_assign(module, res_part, a_part);
    }
}
