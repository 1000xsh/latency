# latency

low-overhead timing for x86_64.

it uses serialized `rdtsc` reads for `TimePoint`, provides a lock-free histogram for latency distributions and includes simple timer helpers for manual and scoped measurement.

## quick start

```rust
use latency::{Histogram, TimePoint, Timer};

fn main() {
    latency::init();

    let start = TimePoint::now();
    do_work();
    println!("elapsed: {}ns", start.elapsed_ns());

    let timer = Timer::start();
    do_work();
    println!("elapsed: {}ns", timer.elapsed_ns());

    let histogram = Histogram::new();
    histogram.record(42);
    println!("{}", histogram.stats().format());
}

fn do_work() {}
```

## api

- `TimePoint::now()` and `elapsed_ns()` for direct timing
- `Timer` and `TimerGuard` for manual or scoped timing
- `ScopedTimer` for closure-based timing
- `Histogram` for `min`, `max`, `mean`, and percentile summaries
- `time_block!` and `time_if_enabled!` for lightweight instrumentation

`ScopedTimer` threshold warnings are debug-only, so release builds do not write to stderr from the measured path.

## benchmark

the example benchmark compares `latency::TimePoint` against `solana_measure::measure::Measure` across an empty measurement and a bounded work sweep.

on 7950x, repeated pinned-core runs showed:

- empty measurement: `latency` is clearly lower overhead
- small work around `80ns` to `300ns`: `latency` is still measurably faster
- medium to larger work around `0.58us` to `2.33us`: both are roughly equal, with `latency` a few nanoseconds lower

run it with:

```bash
cargo run --release --example compare_timing
```

timing is enabled by default. to compile the crate without active timing code:

```toml
[features]
default = []
```

## platform

real tsc-based timing requires x86_64. on other targets, the raw tsc helpers return `0`, so `TimePoint`-based measurement is effectively disabled.
