use poulpy_hal::layouts::ZnxWord;
use std::fmt;

use poulpy_hal::{
    layouts::{Backend, Data, FillUniform, HostDataMut, HostDataRef, ReaderFrom, WriterTo},
    source::Source,
};

use crate::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEAtViewMut, GGLWEAtViewRef, GGLWEBackendMut, GGLWEBackendRef, GGLWEInfos, GGLWEToBackendMut,
    GGLWEToBackendRef, GLWEInfos, GLWESwitchingKey, GLWESwitchingKeyDegrees, GLWESwitchingKeyDegreesMut, GLWEViewMut,
    GLWEViewRef, LWEInfos, Rank, TorusPrecision,
};

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct LWESwitchingKeyLayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub dnum: Dnum,
    pub k_aux: TorusPrecision,
}

impl LWEInfos for LWESwitchingKeyLayout {
    fn n(&self) -> Degree {
        self.n
    }

    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn max_size(&self) -> usize {
        crate::layouts::key_size(self.base2k, self.dnum, Dsize(1), self.k_aux)
    }

    fn k(&self) -> TorusPrecision {
        crate::layouts::key_k(self.base2k, self.dnum, Dsize(1), self.k_aux)
    }
}

impl GLWEInfos for LWESwitchingKeyLayout {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl GGLWEInfos for LWESwitchingKeyLayout {
    fn k_aux(&self) -> TorusPrecision {
        self.k_aux
    }

    fn dnum(&self) -> Dnum {
        self.dnum
    }

    fn rank_in(&self) -> Rank {
        Rank(1)
    }

    fn dsize(&self) -> Dsize {
        Dsize(1)
    }

    fn rank_out(&self) -> Rank {
        Rank(1)
    }
}

#[derive(PartialEq, Eq, Clone)]
pub struct LWESwitchingKey<D: Data, W: ZnxWord>(pub(crate) GLWESwitchingKey<D, W>);

impl<D: Data, W: ZnxWord> LWEInfos for LWESwitchingKey<D, W> {
    fn base2k(&self) -> Base2K {
        self.0.base2k()
    }

    fn n(&self) -> Degree {
        self.0.n()
    }

    fn max_size(&self) -> usize {
        self.0.max_size()
    }

    fn k(&self) -> TorusPrecision {
        self.0.k()
    }
}

impl<D: Data, W: ZnxWord> GLWEInfos for LWESwitchingKey<D, W> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl<D: Data, W: ZnxWord> GGLWEInfos for LWESwitchingKey<D, W> {
    fn k_aux(&self) -> TorusPrecision {
        self.0.k_aux()
    }

    fn dsize(&self) -> Dsize {
        self.0.dsize()
    }

    fn rank_in(&self) -> Rank {
        self.0.rank_in()
    }

    fn rank_out(&self) -> Rank {
        self.0.rank_out()
    }

    fn dnum(&self) -> Dnum {
        self.0.dnum()
    }
}

#[expect(
    dead_code,
    reason = "host-owned constructors are kept for serialization and host-only staging"
)]
impl<W: ZnxWord> LWESwitchingKey<Vec<u8>, W> {
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        assert_eq!(infos.dsize().0, 1, "dsize > 1 is not supported for LWESwitchingKey");
        assert_eq!(infos.rank_in().0, 1, "rank_in > 1 is not supported for LWESwitchingKey");
        assert_eq!(infos.rank_out().0, 1, "rank_out > 1 is not supported for LWESwitchingKey");
        Self::alloc(infos.n(), infos.base2k(), infos.dnum(), infos.k_aux())
    }

    pub(crate) fn alloc(n: Degree, base2k: Base2K, dnum: Dnum, k_aux: TorusPrecision) -> Self {
        LWESwitchingKey(GLWESwitchingKey::alloc(n, base2k, dnum, Dsize(1), k_aux, Rank(1), Rank(1)))
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(infos.dsize().0, 1, "dsize > 1 is not supported for LWESwitchingKey");
        assert_eq!(infos.rank_in().0, 1, "rank_in > 1 is not supported for LWESwitchingKey");
        assert_eq!(infos.rank_out().0, 1, "rank_out > 1 is not supported for LWESwitchingKey");
        Self::bytes_of(infos.n(), infos.base2k(), infos.dnum(), infos.k_aux())
    }

    pub fn bytes_of(n: Degree, base2k: Base2K, dnum: Dnum, k_aux: TorusPrecision) -> usize {
        GLWESwitchingKey::<Vec<u8>, W>::bytes_of(n, base2k, dnum, Dsize(1), k_aux, Rank(1), Rank(1))
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Debug for LWESwitchingKey<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataMut, W: ZnxWord> FillUniform for LWESwitchingKey<D, W> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.0.fill_uniform(log_bound, source);
    }
}

impl<D: HostDataRef, W: ZnxWord> fmt::Display for LWESwitchingKey<D, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(LWESwitchingKey) {}", self.0)
    }
}

impl<D: HostDataMut, W: ZnxWord> ReaderFrom for LWESwitchingKey<D, W> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.0.read_from(reader)
    }
}

impl<D: HostDataRef, W: ZnxWord> WriterTo for LWESwitchingKey<D, W> {
    fn write_to<Wr: std::io::Write>(&self, writer: &mut Wr) -> std::io::Result<()> {
        self.0.write_to(writer)
    }
}

impl_gglwe_to_backend_for_field!(LWESwitchingKey<D, BE::ZnxWord>, 0, GLWESwitchingKey<D, BE::ZnxWord>);

impl_gglwe_at_view_for_field!(LWESwitchingKey<BE::OwnedBuf, BE::ZnxWord>; 0.key);

impl<D: Data, W: ZnxWord> GLWESwitchingKeyDegreesMut for LWESwitchingKey<D, W> {
    fn input_degree(&mut self) -> &mut Degree {
        &mut self.0.input_degree
    }

    fn output_degree(&mut self) -> &mut Degree {
        &mut self.0.output_degree
    }
}

impl<D: Data, W: ZnxWord> GLWESwitchingKeyDegrees for LWESwitchingKey<D, W> {
    fn input_degree(&self) -> &Degree {
        &self.0.input_degree
    }

    fn output_degree(&self) -> &Degree {
        &self.0.output_degree
    }
}
