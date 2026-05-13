# 🔒 Security Policy for Poulpy

## Report a Vulnerability

To report a vulnerability, please contact us at: **[jean-philippe@phantom.zone](mailto:jean-philippe@phantom.zone)**

Include in your report (if possible):

* Affected crate
* Steps to reproduce
* Impact
* Potential fix

We will acknowledge receipt and work with you on resolution.

## Security Model

Poulpy implements Module-LWE-based cryptography and follows the standard **IND-CPA security model** when used with appropriate parameters.

To select secure parameters, we recommend using the [Lattice Estimator](https://github.com/malb/lattice-estimator).

Like other FHE libraries, Poulpy does **not** provide stronger security notions out of the box and users should be aware that:

* FHE ciphertexts are malleable by design and are not inherently CCA-secure.
* Circuit privacy is not guaranteed without additional techniques (e.g., re-randomization, modulus switching, noise flooding).

Users should therefore design protocols accordingly and apply standard safeguards (e.g., private & authenticated channels, key rotation, limiting decryption queries).

Additional context on security notions beyond CPA can be found in [Relations among new CCA security notions for approximate FHE](https://eprint.iacr.org/2024/812).
