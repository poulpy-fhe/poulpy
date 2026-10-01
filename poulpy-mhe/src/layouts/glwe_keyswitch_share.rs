use crate::layouts::glwe_share::glwe_share;

glwe_share!(
    /// One party's share of a collective key switch to a secret key: a rank-0
    /// GLWE, the body the finalization adds to the ciphertext.
    GLWEKeyswitchShare,
    GLWEKeyswitchShareOwned
);

glwe_share!(
    /// One party's share of a collective key switch to a public key: a GLWE
    /// under the output public key, at its rank.
    GLWEPublicKeyswitchShare,
    GLWEPublicKeyswitchShareOwned
);
