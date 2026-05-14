use std::env;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use xq_vision::ModelSource;
use xq_vision::XqVision;

const TEST_IMAGE: &str = "examples/test.jpg";
const TOTAL_ITERS: usize = 101;
const WARMUP_ITERS: usize = 1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let board_model = required_env("BOARD_MODEL")?;
    let piece_model = required_env("PIECE_MODEL")?;
    let image_path = manifest_dir.join(TEST_IMAGE);

    let image = image::open(&image_path)?.to_rgb8();
    let builder = XqVision::builder()
        .graph_optimization(xq_vision::GraphOptimization::All)
        .board_model(ModelSource::file(board_model))
        .piece_model(ModelSource::file(piece_model));
    println!("execution providers: {:?}", builder.session_config().execution_providers());
    let mut vision = builder.build()?;

    let measured = TOTAL_ITERS - WARMUP_ITERS;
    let mut samples: Vec<Duration> = Vec::with_capacity(measured);
    for i in 0..TOTAL_ITERS {
        let start = Instant::now();
        let _ = vision.recognize(&image)?;
        let elapsed = start.elapsed();
        if i < WARMUP_ITERS {
            println!("warmup #{i}: {elapsed:?}");
        } else {
            samples.push(elapsed);
        }
    }

    print_report(&samples);
    Ok(())
}

fn required_env(name: &str) -> Result<String, Box<dyn std::error::Error>> {
    env::var(name).map_err(|_| format!("missing required environment variable {name}").into())
}

fn print_report(samples: &[Duration]) {
    let n = samples.len();
    if n == 0 {
        println!("no samples collected");
        return;
    }

    let mut sorted: Vec<Duration> = samples.to_vec();
    sorted.sort_unstable();

    let total: Duration = sorted.iter().sum();
    let min = sorted[0];
    let max = sorted[n - 1];
    let mean = total / n as u32;
    let p50 = sorted[n / 2];
    let p95 = sorted[(n * 95) / 100];
    let p99 = sorted[(n * 99) / 100];

    let mean_us = mean.as_secs_f64() * 1_000_000.0;
    let variance_us2: f64 = samples
        .iter()
        .map(|d| {
            let delta = d.as_secs_f64() * 1_000_000.0 - mean_us;
            delta * delta
        })
        .sum::<f64>()
        / n as f64;
    let stddev = Duration::from_secs_f64(variance_us2.sqrt() / 1_000_000.0);
    let throughput = n as f64 / total.as_secs_f64();

    println!("---");
    println!("samples:    {n}");
    println!("total:      {total:?}");
    println!("min:        {min:?}");
    println!("max:        {max:?}");
    println!("mean:       {mean:?}");
    println!("stddev:     {stddev:?}");
    println!("p50:        {p50:?}");
    println!("p95:        {p95:?}");
    println!("p99:        {p99:?}");
    println!("throughput: {throughput:.2} ops/s");
}
