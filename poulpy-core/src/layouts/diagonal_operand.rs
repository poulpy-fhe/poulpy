/// The right operand a ciphertext diagonal gives the products of its giant step: prepared as a right operand, on
/// `ceil(k / base2k)` limbs, or a GLWE the product prepares; both give the same bits.
pub enum DiagonalOperand<'a, P, C> {
    /// Prepared as a right operand, on `ceil(k / base2k)` limbs.
    Prepared(&'a P),
    /// Not prepared yet: the product prepares it.
    Unprepared(&'a C),
}
