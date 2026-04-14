//! benchmark comparing latency vs solana-measure using each crate's public api
//!
//! measures:
//! - measurement overhead (timing an empty closure)
//! - bounded work across multiple sizes
#![allow(deprecated)]

use latency::TimePoint;
use solana_measure::measure::Measure;

const ITERATIONS: usize = 100_000;
const WORKLOAD_ROUNDS: [u64; 5] = [128, 256, 512, 1_024, 4_096];

fn main() {
    latency::init();

    println!("\n=== latency vs solana-measure benchmark ===\n");
    println!("compares public timing apis on identical workloads");
    println!("all reported durations include each api's own measurement cost\n");

    // Warm up
    for _ in 0..1000 {
        let _ = TimePoint::now();
    }

    println!("benchmark 1: measurement overhead (empty closure)");
    benchmark_overhead();

    println!("\nbenchmark 2: bounded work sweep");
    benchmark_work_sweep();
}

fn benchmark_overhead() {
    benchmark_pair(|| {});
}

fn benchmark_work_sweep() {
    for rounds in WORKLOAD_ROUNDS {
        println!("\n  work rounds={rounds}");
        benchmark_pair(|| do_work(rounds));
    }
}

fn benchmark_pair<F>(work: F)
where
    F: FnMut() + Copy,
{
    let latency_stats = run_latency_benchmark(work);
    let solana_stats = run_solana_benchmark(work);

    println!("\n  latency (TimePoint):");
    print_stats(&latency_stats);

    println!("\n  solana-measure (Measure):");
    print_stats(&solana_stats);

    println!(
        "\n  mean ratio (solana-measure / latency): {:.2}x",
        ratio(solana_stats.mean, latency_stats.mean)
    );
    println!(
        "  p99 ratio (solana-measure / latency): {:.2}x",
        ratio(solana_stats.p99, latency_stats.p99)
    );
}

fn run_latency_benchmark<F>(mut work: F) -> BenchmarkStats
where
    F: FnMut(),
{
    let mut samples = Vec::with_capacity(ITERATIONS);

    for _ in 0..ITERATIONS {
        let start = TimePoint::now();
        work();
        samples.push(start.elapsed_ns());
    }

    BenchmarkStats::from_samples(samples)
}

fn run_solana_benchmark<F>(mut work: F) -> BenchmarkStats
where
    F: FnMut(),
{
    let mut samples = Vec::with_capacity(ITERATIONS);

    for _ in 0..ITERATIONS {
        let mut measure = Measure::start("");
        work();
        measure.stop();
        samples.push(measure.as_ns());
    }

    BenchmarkStats::from_samples(samples)
}

fn do_work(rounds: u64) {
    let mut acc = 0x9e37_79b9_7f4a_7c15u64;

    for i in 0..rounds {
        let mixed = i
            .wrapping_mul(0xbf58_476d_1ce4_e5b9)
            .rotate_left((i as u32) & 31);
        acc = acc.wrapping_add(mixed ^ acc.rotate_left(7));
    }

    std::hint::black_box(acc);
}

fn print_stats(stats: &BenchmarkStats) {
    println!(
        "    count={} mean={}ns min={}ns p50={}ns p99={}ns p999={}ns max={}ns",
        stats.count, stats.mean, stats.min, stats.p50, stats.p99, stats.p999, stats.max
    );
}

fn ratio(lhs: u64, rhs: u64) -> f64 {
    lhs as f64 / rhs.max(1) as f64
}

#[derive(Clone, Copy, Debug)]
struct BenchmarkStats {
    count: usize,
    mean: u64,
    min: u64,
    p50: u64,
    p99: u64,
    p999: u64,
    max: u64,
}

impl BenchmarkStats {
    fn from_samples(mut samples: Vec<u64>) -> Self {
        debug_assert!(!samples.is_empty());

        let count = samples.len();
        let sum = samples.iter().map(|&sample| sample as u128).sum::<u128>();

        samples.sort_unstable();

        Self {
            count,
            mean: (sum / count as u128) as u64,
            min: samples[0],
            p50: samples[percentile_index(count, 50, 100)],
            p99: samples[percentile_index(count, 99, 100)],
            p999: samples[percentile_index(count, 999, 1000)],
            max: samples[count - 1],
        }
    }
}

fn percentile_index(count: usize, numerator: usize, denominator: usize) -> usize {
    if count <= 1 {
        0
    } else {
        ((count - 1) * numerator) / denominator
    }
}
