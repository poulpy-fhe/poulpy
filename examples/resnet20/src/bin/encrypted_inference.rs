//! Encrypted ResNet-20 inference on the first images of the CIFAR-10 test batch, `encrypted_inference [images]`. Ten
//! are built in; the crate's README gets the others. The server runs on the CPU, or on the GPU with the `enable-cuda`
//! feature, where `--replay` records the whole circuit once as a graph. Each image is compared with the cleartext
//! forward pass that uses the same polynomial ReLU; `--trace` decrypts every layer of the first image on the client and
//! compares it with the cleartext; `--no-stage-sync` leaves the per-stage times meaningless and the wall clock free of
//! the waits they need.

use std::{
    collections::HashMap,
    fs::File,
    io::Read,
    time::{Duration, Instant},
};

use anyhow::{Result, ensure};
use poulpy_ckks::{CKKSInfos, layouts::CKKSCiphertextOwned};
use poulpy_core::layouts::LWEInfos;
use poulpy_resnet20::{
    CoeffLayout,
    model::{Relu, ResNet20, SPAN, argmax, log_divisor, relu, test_images},
    network::{Client, Host, Layouts, Params, Server, Stage},
};

/// A root seed for the client, from the operating system.
fn fresh_seed() -> Result<[u8; 32]> {
    let mut seed = [0; 32];
    File::open("/dev/urandom")?.read_exact(&mut seed)?;
    Ok(seed)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| args.iter().any(|a| a == name);
    let count = args.iter().find(|a| !a.starts_with("--")).map_or(Ok(1), |a| a.parse())?;
    ensure!(
        cfg!(feature = "enable-cuda") || !flag("--replay"),
        "--replay takes the enable-cuda feature"
    );
    let model = ResNet20::load()?;
    let images = test_images(count)?;
    let params = Params::new()?;
    let layouts = Layouts::new(params.n())?;
    println!(
        "widths: input {} output {} raised {}",
        params.preset.input_k(),
        params.preset.output_k(),
        params.preset.bootstrap_k()
    );

    let start = Instant::now();
    let mut client = Client::new(&params, fresh_seed()?);
    let key_set = client.keys(&params)?;
    println!("client keys: {:.1} s", start.elapsed().as_secs_f64());
    let start = Instant::now();
    let mut server = Server::new(params.clone(), &model, &layouts, &key_set)?;
    drop(key_set);
    println!("server setup: {:.1} s", start.elapsed().as_secs_f64());

    #[cfg(feature = "enable-cuda")]
    if flag("--replay") {
        return replayed(&model, &images, &params, &layouts, &mut client, &server);
    }
    let trace = flag("--trace");
    server.set_stage_sync(!flag("--no-stage-sync"));
    let (mut correct, mut agree) = (0, 0);
    let mut warm = Duration::ZERO;
    for (i, (image, label)) in images.iter().enumerate() {
        let ct = client.encrypt_image(&params, &layouts.image, image)?;
        let start = Instant::now();
        let (out, t) = if trace && i == 0 {
            // Each layer's output against the cleartext one, normalized by its divisor.
            let mut inputs = HashMap::new();
            model.forward_traced(image, Relu::Polynomial, &mut |name, x| {
                inputs.insert(name.to_string(), x.to_vec());
            })?;
            let mut probe = |name: &str, stage: Stage, ct: &CKKSCiphertextOwned<Host>, layout: &CoeffLayout| {
                let d = f64::from(1u32 << log_divisor(name).unwrap());
                let want: Vec<f64> = inputs[name]
                    .iter()
                    .map(|x| x / d)
                    .map(|u| if stage == Stage::Input { SPAN * u } else { relu(u) })
                    .collect();
                let have = if stage == Stage::Activation {
                    client.decrypt_slots(&params, ct, layout)
                } else {
                    client.decrypt_coeffs(ct, layout)
                }
                .unwrap();
                let err = have.iter().zip(&want).fold(0.0f64, |e, (h, w)| e.max((h - w).abs()));
                let max = want.iter().fold(0.0f64, |m, w| m.max(w.abs()));
                println!(
                    "  {name} {stage:?}: max error 2^{:.1}, max value {max:.3}, k {}, scale 2^{}",
                    err.log2(),
                    ct.k().as_usize(),
                    ct.log_delta()
                );
            };
            server.infer_traced(&ct, Some(&mut probe))?
        } else {
            server.infer(&ct)?
        };
        let wall = start.elapsed();
        if i > 0 {
            warm += wall;
        }
        let logits = client.decrypt_coeffs(&out, &layouts.logits)?;
        let mirror = model.forward(image, Relu::Polynomial)?;
        let err = logits.iter().zip(&mirror).fold(0.0f64, |e, (a, b)| e.max((a - b).abs()));
        let (class, want) = (argmax(&logits), argmax(&mirror));
        correct += usize::from(class == *label);
        agree += usize::from(class == want);
        println!(
            "image {i}: label {label}, encrypted {class}, cleartext {want}, logit error 2^{:.1}, {:.2} s \
             (s2c {:.2}, conv {:.2}, raise {:.2}, relu {:.2})",
            err.log2(),
            t.total().as_secs_f64(),
            t.s2c.as_secs_f64(),
            t.conv.as_secs_f64(),
            t.raise.as_secs_f64(),
            t.relu.as_secs_f64(),
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

/// The images through the circuit recorded once as a graph.
#[cfg(feature = "enable-cuda")]
fn replayed(
    model: &ResNet20,
    images: &[(Vec<f64>, usize)],
    params: &Params,
    layouts: &Layouts,
    client: &mut Client,
    server: &Server,
) -> Result<()> {
    let first = client.encrypt_image(params, &layouts.image, &images[0].0)?;
    let start = Instant::now();
    let mut recorded = server.record(&first)?;
    let host = recorded.host_events();
    println!(
        "recorded in {:.1} s: {} launches, graph {}, {:.2} GiB retained, host reads {}, writes {}, syncs {}, allocations {}",
        start.elapsed().as_secs_f64(),
        recorded.launches(),
        recorded.is_graph(),
        recorded.retained_bytes() as f64 / (1u64 << 30) as f64,
        host.reads,
        host.writes,
        host.syncs,
        host.allocations,
    );
    recorded.check();

    let (mut correct, mut agree) = (0, 0);
    let mut warm = Duration::ZERO;
    for (i, (image, label)) in images.iter().enumerate() {
        let encrypted = if i == 0 {
            None
        } else {
            Some(client.encrypt_image(params, &layouts.image, image)?)
        };
        let start = Instant::now();
        let out = recorded.infer(encrypted.as_ref().unwrap_or(&first));
        let wall = start.elapsed();
        if i > 0 {
            warm += wall;
        }
        let logits = client.decrypt_coeffs(&out, &layouts.logits)?;
        let mirror = model.forward(image, Relu::Polynomial)?;
        let err = logits.iter().zip(&mirror).fold(0.0f64, |e, (a, b)| e.max((a - b).abs()));
        let (class, want) = (argmax(&logits), argmax(&mirror));
        correct += usize::from(class == *label);
        agree += usize::from(class == want);
        println!(
            "image {i}: label {label}, encrypted {class}, cleartext {want}, logit error 2^{:.1}, {:.1} ms",
            err.log2(),
            1e3 * wall.as_secs_f64(),
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
