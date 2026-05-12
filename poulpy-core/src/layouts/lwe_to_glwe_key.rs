use std::fmt;

use poulpy_hal::{
    layouts::{Backend, Data, FillUniform, HostDataMut, HostDataRef, ReaderFrom, WriterTo},
    source::Source,
};

use crate::{
    DeclaredK,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGLWEAtViewMut, GGLWEAtViewRef, GGLWEBackendMut, GGLWEBackendRef, GGLWEInfos,
        GGLWEToBackendMut, GGLWEToBackendRef, GLWEInfos, GLWESwitchingKey, GLWESwitchingKeyDegrees, GLWESwitchingKeyDegreesMut,
        GLWEViewMut, GLWEViewRef, LWEInfos, Rank, TorusPrecision,
    },
};

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct LWEToGLWEKeyLayout {
    pub n: Degree,
    pub base2k: Base2K,
    pub k: TorusPrecision,
    pub rank_out: Rank,
    pub dnum: Dnum,
}

impl DeclaredK for LWEToGLWEKeyLayout {
    fn k(&self) -> TorusPrecision {
        self.k
    }
}

impl LWEInfos for LWEToGLWEKeyLayout {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn n(&self) -> Degree {
        self.n
    }

    fn size(&self) -> usize {
        self.k.as_usize().div_ceil(self.base2k.as_usize())
    }
}

impl GLWEInfos for LWEToGLWEKeyLayout {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}

impl GGLWEInfos for LWEToGLWEKeyLayout {
    fn rank_in(&self) -> Rank {
        Rank(1)
    }

    fn dsize(&self) -> Dsize {
        Dsize(1)
    }

    fn rank_out(&self) -> Rank {
        self.rank_out
    }

    fn dnum(&self) -> Dnum {
        self.dnum
    }
}

#[derive(PartialEq, Eq, Clone)]
pub struct LWEToGLWEKey<D: Data>(pub(crate) GLWESwitchingKey<D>);

impl<D: Data> LWEInfos for LWEToGLWEKey<D> {
    fn base2k(&self) -> Base2K {
        self.0.base2k()
    }

    fn n(&self) -> Degree {
        self.0.n()
    }

    fn size(&self) -> usize {
        self.0.size()
    }
}

impl<D: Data> GLWEInfos for LWEToGLWEKey<D> {
    fn rank(&self) -> Rank {
        self.rank_out()
    }
}
impl<D: Data> GGLWEInfos for LWEToGLWEKey<D> {
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

impl<D: HostDataRef> fmt::Debug for LWEToGLWEKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl<D: HostDataMut> FillUniform for LWEToGLWEKey<D> {
    fn fill_uniform(&mut self, log_bound: usize, source: &mut Source) {
        self.0.fill_uniform(log_bound, source);
    }
}

impl<D: HostDataRef> fmt::Display for LWEToGLWEKey<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(LWEToGLWEKey) {}", self.0)
    }
}

impl<D: HostDataMut> ReaderFrom for LWEToGLWEKey<D> {
    fn read_from<R: std::io::Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        self.0.read_from(reader)
    }
}

impl<D: HostDataRef> WriterTo for LWEToGLWEKey<D> {
    fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.0.write_to(writer)
    }
}

#[expect(
    dead_code,
    reason = "host-owned constructors are kept for serialization and host-only staging"
)]
impl LWEToGLWEKey<Vec<u8>> {
    pub(crate) fn alloc_from_infos<A>(infos: &A) -> Self
    where
        A: GGLWEInfos,
    {
        assert_eq!(infos.rank_in().0, 1, "rank_in > 1 is not supported for LWEToGLWEKey");
        assert_eq!(infos.dsize().0, 1, "dsize > 1 is not supported for LWEToGLWEKey");

        Self::alloc(infos.n(), infos.base2k(), infos.max_k(), infos.rank_out(), infos.dnum())
    }

    pub(crate) fn alloc(n: Degree, base2k: Base2K, k: TorusPrecision, rank_out: Rank, dnum: Dnum) -> Self {
        LWEToGLWEKey(GLWESwitchingKey::alloc(n, base2k, k, Rank(1), rank_out, dnum, Dsize(1)))
    }

    pub fn bytes_of_from_infos<A>(infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(infos.rank_in().0, 1, "rank_in > 1 is not supported for LWEToGLWEKey");
        assert_eq!(infos.dsize().0, 1, "dsize > 1 is not supported for LWEToGLWEKey");
        Self::bytes_of(infos.n(), infos.base2k(), infos.max_k(), infos.rank_out(), infos.dnum())
    }

    pub fn bytes_of(n: Degree, base2k: Base2K, k: TorusPrecision, rank_out: Rank, dnum: Dnum) -> usize {
        GLWESwitchingKey::bytes_of(n, base2k, k, Rank(1), rank_out, dnum, Dsize(1))
    }
}

impl_gglwe_to_backend_for_field!(LWEToGLWEKey<D>, 0, GLWESwitchingKey<D>);

impl_gglwe_at_view_for_field!(LWEToGLWEKey<BE::OwnedBuf>; 0.key);

impl<D: HostDataMut> GLWESwitchingKeyDegreesMut for LWEToGLWEKey<D> {
    fn input_degree(&mut self) -> &mut Degree {
        &mut self.0.input_degree
    }

    fn output_degree(&mut self) -> &mut Degree {
        &mut self.0.output_degree
    }
}

impl<D: HostDataRef> GLWESwitchingKeyDegrees for LWEToGLWEKey<D> {
    fn input_degree(&self) -> &Degree {
        &self.0.input_degree
    }

    fn output_degree(&self) -> &Degree {
        &self.0.output_degree
    }
}
