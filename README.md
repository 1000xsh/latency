# latency

low-latency timing library for performance-critical applications. provides nanosecond-precision measurements using x86_64 tsc (time-stamp counter) with minimal overhead.

## features

- **cpu cycle counting** via `rdtsc`/`rdtscp` on x86_64
- **lock-free histogram** with logarithmic bucketing for latency distributions
- **multiple timer apis** manual, raii guards, scoped timers
- **zero-cost when disabled** via feature flag
- **fast statistics** percentiles (p50, p90, p99, p999, p9999) with o(1) bucket access

## quick start

```rust
use latency::{TimePoint, Histogram, Timer, ScopedTimer};
use std::sync::Arc;

fn main() {
    latency::init();

    // method 1: simple time point
    let start = TimePoint::now();
    do_work();
    println!("elapsed: {}ns", start.elapsed_ns());

    // method 2: manual timer
    let timer = Timer::start();
    do_work();
    println!("elapsed: {}μs", timer.elapsed_micros());

    // method 3: RAII guard
    let histogram = Arc::new(Histogram::new());
    {
        let _guard = latency::TimerGuard::new(Arc::clone(&histogram));
        do_work();
    }
    println!("{}", histogram.stats().format_micros());

    // method 4: scoped timer with closures
    let timer = ScopedTimer::new(Arc::clone(&histogram));
    let result = timer.time(|| {
        do_work()
    });
}

fn do_work() {}
```

## api

### timing primitives
- `TimePoint::now()` - snapshot of current tsc value
- `TimePoint::elapsed_ns()` / `elapsed_cycles()` - elapsed time since snapshot
- `time_block!` macro - time a code block and get result + elapsed time
- `time_if_enabled!` macro - conditional timing based on feature flag

### histograms
- `Histogram::record(value_ns)` - lock-free atomic recording
- `histogram.stats()` - get latencystats with percentiles
- `histogram.percentiles(&[p1, p2, ...])` - arbitrary percentile queries

### timers
- `Timer` - manual start/stop with optional histogram recording
- `TimerGuard` - raii guard that records on drop
- `ScopedTimer` - closures with threshold-based logging

### global registry (optional)
```rust
let hist = timer::global::histogram("my_op");
timer::global::report_all();  // print all histograms
```

## configuration

add to `Cargo.toml`:
```toml
[dependencies]
latency = { path = "." }

[features]
default = ["enabled"]
```

disable timing overhead:
```toml
default = []  # timing disabled by default
```

## performance

- `rdtsc()` – ~20 cycles (inline)
- `histogram.record()` – ~50 cycles atomic update (lock-free)
- when disabled – zero runtime cost via feature flag

## x86_64 only

requires x86_64 for tsc access. other platforms return 0 with feature checks.
