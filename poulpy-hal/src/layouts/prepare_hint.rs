/// How a prepared operand is expected to be used.
///
/// The hint selects a backend representation; it never changes the value a
/// prepared object denotes. `prepare` writes `prep(M)` in the representation
/// named by the destination's hint, and `apply` accepts every hint the backend
/// can produce. Backends with a single representation ignore it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PrepareHint {
    /// The prepared object is applied many times; optimise `apply`.
    #[default]
    Reuse,
    /// The prepared object is applied once; optimise `prepare` + `apply` together.
    /// "Once" means consumed by a single product operation, which may apply it
    /// to every column of one operand.
    OneShot,
}

#[cfg(test)]
mod tests {
    use super::PrepareHint;

    #[test]
    fn default_is_reuse() {
        assert_eq!(PrepareHint::default(), PrepareHint::Reuse);
    }
}
