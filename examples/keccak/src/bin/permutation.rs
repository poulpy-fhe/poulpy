//! Keccak-f[1600] on eight encrypted states: the client encrypts eight pseudorandom states in one ciphertext, the
//! server runs the 24 rounds, and the decrypted states are checked against the plain permutation.

use std::{fs::File, io::Read, time::Instant};

use anyhow::{Result, ensure};
use poulpy_keccak::{
    client::Client,
    keccak::{ROUND_CONSTANTS, keccak_f},
    layout::STATES,
    params::Params,
    server::Server,
};

fn main() -> Result<()> {
    let params = Params::new()?;
    let mut seed = [0; 32];
    File::open("/dev/urandom")?.read_exact(&mut seed)?;
    let mut client = Client::new(&params, seed);
    let keys = client.keys(&params)?;
    let mut server = Server::new(&params, &keys)?;
    drop(keys);

    // Pseudorandom states.
    let mut x = 0x243F6A8885A308D3u64;
    let states: [[u64; 25]; STATES] = std::array::from_fn(|_| {
        std::array::from_fn(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        })
    });

    let mut c = server.upload(&client.encrypt(&params, &states)?);
    for r in 0..ROUND_CONSTANTS.len() {
        let start = Instant::now();
        c = server.round(&c, r)?;
        server.synchronize();
        println!("round {r:2}: {:.3} s", start.elapsed().as_secs_f64());
    }

    for (s, (got, mut want)) in client.decrypt(&server.download(&c))?.into_iter().zip(states).enumerate() {
        keccak_f(&mut want);
        ensure!(got == want, "decrypted state {s} differs from Keccak-f[1600]");
    }
    println!("all {STATES} states match Keccak-f[1600]");
    Ok(())
}
