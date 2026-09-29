# Multiparty contracts

## Randomness and seeds

A public mask seed is common to all parties contributing to one result.
Use a fresh seed for every result, separated by protocol, session and key
identity. Derive separate seeds from the CRS
for a key set.

Private `source_xe` streams must be independently
seeded for each party and purpose, kept secret and consumed without replay.
Never initialize a private stream from a public mask seed. An advancing error
stream can supply successive fresh samples.
