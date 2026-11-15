//! Encrypted ResNet-20 inference on the first images of the CIFAR-10 test batch, `encrypted_inference [images]`. Ten
//! are built in; the crate's README gets the others. The server runs on the CPU, or on the GPU with the `enable-cuda`
//! feature, which records the whole circuit once as a graph. Each image is compared with the cleartext forward pass
//! that uses the same polynomial ReLU.

use std::time::{Duration, Instant};

use anyhow::{Result, ensure};
use poulpy_resnet20::{
    Client, Params, Server,
    model::{ResNet20, Variant, argmax, max_error, test_images},
};

fn main() -> Result<()> {
    let count = std::env::args().nth(1).map_or(Ok(1), |a| a.parse())?;
    let model = ResNet20::load()?;
    let images = test_images(count)?;
    ensure!(!images.is_empty(), "an inference takes at least one image");
    let params = Params::new()?;
    println!(
        "widths: input {} output {} raised {}",
        params.preset().input_k(),
        params.preset().output_k(),
        params.preset().bootstrap_k()
    );

    let start = Instant::now();
    let mut client = Client::new(&params, rand::random());
    let key_set = client.keys()?;
    println!("client keys: {:.1} s", start.elapsed().as_secs_f64());
    let start = Instant::now();
    let server = Server::new(&params, &model, key_set)?;
    println!("server setup: {:.1} s", start.elapsed().as_secs_f64());
    #[cfg(feature = "enable-cuda")]
    let mut server = {
        let start = Instant::now();
        let mut recorded = server.record(&client.encrypt_image(&images[0].0)?)?;
        let host = recorded.replay.host_events();
        println!(
            "recorded in {:.1} s: {} launches, graph {}, {:.2} GiB retained, host reads {}, writes {}, syncs {}, \
             allocations {}",
            start.elapsed().as_secs_f64(),
            recorded.replay.launches(),
            recorded.replay.is_graph(),
            recorded.replay.retained_bytes() as f64 / (1u64 << 30) as f64,
            host.reads,
            host.writes,
            host.syncs,
            host.allocations,
        );
        recorded.replay.check();
        recorded
    };

    let (mut correct, mut agree) = (0, 0);
    // Wall clock of every image but the first.
    let mut warm = Duration::ZERO;
    for (i, (image, label)) in images.iter().enumerate() {
        let ct = client.encrypt_image(image)?;
        let start = Instant::now();
        let out = server.infer(&ct)?;
        let wall = start.elapsed();
        if i > 0 {
            warm += wall;
        }
        let logits = client.decrypt_logits(&out)?;
        let mirror = model.forward(image, Variant::Polynomial);
        let (class, want) = (argmax(&logits), argmax(&mirror));
        correct += usize::from(class == *label);
        agree += usize::from(class == want);
        println!(
            "image {i}: label {label}, encrypted {class}, cleartext {want}, logit error 2^{:.1}, {:.1} ms",
            max_error(&logits, &mirror).log2(),
            1e3 * wall.as_secs_f64()
        );
    }
    if images.len() > 1 {
        println!(
            "warm wall clock: {:.1} ms per image",
            1e3 * warm.as_secs_f64() / (images.len() - 1) as f64
        );
    }
    println!(
        "{correct} / {} correct, {agree} agree with the cleartext polynomial network",
        images.len()
    );
    Ok(())
}
