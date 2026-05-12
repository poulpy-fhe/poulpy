use crate::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, ScratchArena, ScratchOwned},
};

impl<B> ScratchOwnedAlloc<B> for ScratchOwned<B>
where
    B: Backend,
{
    fn alloc(size: usize) -> Self {
        ScratchOwned {
            data: B::alloc_bytes(size),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B> ScratchOwnedBorrow<B> for ScratchOwned<B>
where
    B: Backend,
{
    fn borrow(&mut self) -> ScratchArena<'_, B> {
        self.arena()
    }
}

impl<B: Backend> ScratchAvailable for ScratchArena<'_, B> {
    fn available(&self) -> usize {
        ScratchArena::available(self)
    }
}
