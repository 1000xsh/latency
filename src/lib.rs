//! low overhead timing infrastructure for perf measurement
//!
//! provides rdtsc-based timing on x86_64 and histogram tracking

#![cfg_attr(not(feature = "enabled"), allow(dead_code))]

pub mod histogram;
pub mod rdtsc;
pub mod timer;

pub use histogram::{Histogram, Percentiles};
pub use rdtsc::{cycles_to_nanos, rdtsc, rdtsc_end, rdtsc_start, rdtscp};
pub use timer::{ScopedTimer, Timer, TimerGuard};

use std::sync::atomic::{AtomicU64, Ordering};

/// global cycle frequency for conversion to nanoseconds
static CYCLES_PER_SECOND: AtomicU64 = AtomicU64::new(0);

/// Q32 fixed-point multiplier: nanos = (cycles * multiplier) >> 32
/// avoids u128 division on hot path (~3-4 cycles vs 40-80 for __udivti3)
static NANOS_MULTIPLIER: AtomicU64 = AtomicU64::new(0);

/// Q32 fixed-point multiplier: cycles = (nanos * multiplier) >> 32
static CYCLES_MULTIPLIER: AtomicU64 = AtomicU64::new(0);

/// initializes the timing subsystem
///
/// calibrates rdtsc frequency for accurate time measurements.
/// should be called once at program start.
pub fn init() {
    #[cfg(all(target_arch = "x86_64", feature = "enabled"))]
    {
        assert!(
            rdtsc::has_invariant_tsc(),
            "cpu does not support invariant tsc -- rdtsc measurements will be unreliable"
        );

        let freq = rdtsc::calibrate_frequency();
        CYCLES_PER_SECOND.store(freq, Ordering::Relaxed);

        // precompute fixed-point multipliers to avoid u128 division on hot path
        // nanos_mult = (1e9 << 32) / freq -- for cycles_to_nanos
        // cycles_mult = (freq << 32) / 1e9 -- for nanos_to_cycles
        let nanos_mult = ((1_000_000_000u128 << 32) / freq as u128) as u64;
        let cycles_mult = ((freq as u128) << 32) / 1_000_000_000u128;
        NANOS_MULTIPLIER.store(nanos_mult, Ordering::Relaxed);
        CYCLES_MULTIPLIER.store(cycles_mult as u64, Ordering::Relaxed);

        eprintln!(
            "timing: calibrated cpu frequency: {} ghz",
            freq as f64 / 1_000_000_000.0
        );
    }
}

/// gets the calibrated cpu frequency in cycles per second
pub fn cpu_frequency() -> u64 {
    CYCLES_PER_SECOND.load(Ordering::Relaxed)
}

/// gets the precomputed Q32 nanos multiplier
pub fn nanos_multiplier() -> u64 {
    NANOS_MULTIPLIER.load(Ordering::Relaxed)
}

/// gets the precomputed Q32 cycles multiplier
pub fn cycles_multiplier() -> u64 {
    CYCLES_MULTIPLIER.load(Ordering::Relaxed)
}

/// timing measurement point
#[derive(Debug, Clone, Copy)]
pub struct TimePoint {
    cycles: u64,
}

impl TimePoint {
    /// creates a new time point at current time
    ///
    /// uses lfence+rdtsc to serialize the start measurement,
    /// preventing ooo reordering past the code being measured.
    #[inline(always)]
    pub fn now() -> Self {
        #[cfg(all(target_arch = "x86_64", feature = "enabled"))]
        {
            Self {
                cycles: rdtsc::rdtsc_start(),
            }
        }
        #[cfg(not(all(target_arch = "x86_64", feature = "enabled")))]
        {
            Self { cycles: 0 }
        }
    }

    /// calculates elapsed time since this point in nanoseconds
    ///
    /// uses rdtscp+lfence to serialize the end measurement.
    #[inline(always)]
    pub fn elapsed_ns(&self) -> u64 {
        #[cfg(all(target_arch = "x86_64", feature = "enabled"))]
        {
            let now = rdtsc::rdtsc_end();
            if now > self.cycles {
                cycles_to_nanos(now - self.cycles)
            } else {
                0
            }
        }
        #[cfg(not(all(target_arch = "x86_64", feature = "enabled")))]
        {
            0
        }
    }

    /// calculates elapsed time since this point in cycles
    ///
    /// uses rdtscp+lfence to serialize the end measurement.
    #[inline(always)]
    pub fn elapsed_cycles(&self) -> u64 {
        #[cfg(all(target_arch = "x86_64", feature = "enabled"))]
        {
            let now = rdtsc::rdtsc_end();
            if now > self.cycles {
                now - self.cycles
            } else {
                0
            }
        }
        #[cfg(not(all(target_arch = "x86_64", feature = "enabled")))]
        {
            0
        }
    }
}

/// macro for timing a block of code
///
/// usage:
/// ```no_run
/// let (result, elapsed) = latency::time_block!({
///     // code to time
///     42
/// });
/// ```
#[macro_export]
macro_rules! time_block {
    ($block:block) => {{
        let __start = $crate::TimePoint::now();
        let __result = $block;
        (__result, __start.elapsed_ns())
    }};
}

/// macro for conditionally timing based on feature flag
///
/// usage:
/// ```no_run
/// let histogram = latency::Histogram::new();
/// latency::time_if_enabled!(histogram, {
///     // code to time
/// });
/// ```
#[macro_export]
macro_rules! time_if_enabled {
    ($histogram:expr, $block:block) => {{
        #[cfg(feature = "enabled")]
        {
            let __start = $crate::TimePoint::now();
            let __result = $block;
            $histogram.record(__start.elapsed_ns());
            __result
        }
        #[cfg(not(feature = "enabled"))]
        {
            $block
        }
    }};
}

/// latency statistics
#[derive(Debug, Clone, Copy, Default)]
pub struct LatencyStats {
    pub min: u64,
    pub max: u64,
    pub mean: u64,
    pub p50: u64,
    pub p90: u64,
    pub p99: u64,
    pub p999: u64,
    pub p9999: u64,
    pub count: u64,
}

impl LatencyStats {
    /// formats the stats as a string
    pub fn format(&self) -> String {
        format!(
            "count={} min={}ns p50={}ns p99={}ns p999={}ns max={}ns",
            self.count, self.min, self.p50, self.p99, self.p999, self.max
        )
    }

    /// formats with microsecond units
    pub fn format_micros(&self) -> String {
        format!(
            "count={} min={}μs p50={}μs p99={}μs p999={}μs max={}μs",
            self.count,
            self.min / 1000,
            self.p50 / 1000,
            self.p99 / 1000,
            self.p999 / 1000,
            self.max / 1000
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_point() {
        init();

        let start = TimePoint::now();
        // do some work
        let mut sum = 0u64;
        for i in 0..1000 {
            sum = sum.wrapping_add(i);
        }
        let elapsed = start.elapsed_ns();

        // should take some time
        #[cfg(feature = "enabled")]
        assert!(elapsed > 0);

        // prevent optimization
        std::hint::black_box(sum);
    }

    #[test]
    fn test_time_block_macro() {
        init();

        let (result, elapsed) = time_block!({
            let mut sum = 0u64;
            for i in 0..1000 {
                sum = sum.wrapping_add(i);
            }
            sum
        });

        assert_eq!(result, 499500);
        #[cfg(feature = "enabled")]
        assert!(elapsed > 0);
    }
}
