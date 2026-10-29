/// The right operand a ciphertext diagonal gives the products of its giant step: prepared as a right operand, on
/// `ceil(k / base2k)` limbs, or a ciphertext the product prepares; both give the same bits.
pub enum MulRightOperand<'a, P, C> {
    /// Prepared as a right operand, on `ceil(k / base2k)` limbs.
    Prepared(&'a P),
    /// A ciphertext, prepared in the product.
    Ciphertext(&'a C),
}
