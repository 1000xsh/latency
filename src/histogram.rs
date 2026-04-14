//! high-performance histogram for latency tracking
//!
//! uses logarithmic buckets for efficient storage and fast percentile calculation.
//! designed for nanosecond-precision measurements with minimal overhead.

use std::sync::atomic::{AtomicU64, Ordering};

/// sub-bucket precision bits: 3 gives 8 sub-buckets per power-of-2 range (~12.5% error).
/// hot-path cost is identical to pure log2 -- just bit shifts and masks.
const SUB_BITS: u32 = 3;
const SUB_BUCKETS: usize = 1 << SUB_BITS; // 8

/// total buckets: linear region (8) + 61 magnitude groups * 8 sub-buckets = 496
/// covers u64 range from 1ns to ~584 years with ~12.5% precision at every scale.
/// bucket array = 496 * 8 = ~4KB (fits in L1 cache).
const NUM_BUCKETS: usize = SUB_BUCKETS + (64 - SUB_BITS as usize) * SUB_BUCKETS; // 496

/// cache-line-padded atomic to prevent false sharing between threads.
/// each padded field occupies its own 64-byte cache line so concurrent
/// writes to different fields don't bounce the same line.
#[repr(align(64))]
struct PaddedAtomicU64(AtomicU64);

impl PaddedAtomicU64 {
    const fn new(val: u64) -> Self {
        Self(AtomicU64::new(val))
    }

    #[inline(always)]
    fn load(&self, order: Ordering) -> u64 {
        self.0.load(order)
    }

    #[inline(always)]
    fn store(&self, val: u64, order: Ordering) {
        self.0.store(val, order)
    }

    #[inline(always)]
    fn fetch_add(&self, val: u64, order: Ordering) -> u64 {
        self.0.fetch_add(val, order)
    }

    #[inline(always)]
    fn compare_exchange_weak(
        &self,
        current: u64,
        new: u64,
        success: Ordering,
        failure: Ordering,
    ) -> Result<u64, u64> {
        self.0.compare_exchange_weak(current, new, success, failure)
    }
}

/// histogram for tracking latency distributions
///
/// thread-safe using atomics for lock-free updates.
/// hot statistics fields are cache-line-padded to prevent false sharing.
pub struct Histogram {
    /// bucket counts
    buckets: [AtomicU64; NUM_BUCKETS],
    /// total count of samples (padded to own cache line)
    count: PaddedAtomicU64,
    /// sum of all samples for mean calculation (padded)
    sum: PaddedAtomicU64,
    /// minimum value seen (padded)
    min: PaddedAtomicU64,
    /// maximum value seen (padded)
    max: PaddedAtomicU64,
}

impl Histogram {
    /// creates a new empty histogram
    pub fn new() -> Self {
        Self {
            buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            count: PaddedAtomicU64::new(0),
            sum: PaddedAtomicU64::new(0),
            min: PaddedAtomicU64::new(u64::MAX),
            max: PaddedAtomicU64::new(0),
        }
    }

    /// records a value in nanoseconds
    #[inline(always)]
    pub fn record(&self, value_ns: u64) {
        // determine bucket index using logarithmic scaling
        let bucket = Self::value_to_bucket(value_ns);

        // increment bucket count
        self.buckets[bucket].fetch_add(1, Ordering::Relaxed);

        // update statistics
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum.fetch_add(value_ns, Ordering::Relaxed);

        // update min/max (may have races but that's ok for statistics)
        self.update_min(value_ns);
        self.update_max(value_ns);
    }

    /// updates minimum value
    fn update_min(&self, value: u64) {
        let mut current = self.min.load(Ordering::Relaxed);
        while value < current {
            match self.min.compare_exchange_weak(
                current,
                value,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current = x,
            }
        }
    }

    /// updates maximum value
    fn update_max(&self, value: u64) {
        let mut current = self.max.load(Ordering::Relaxed);
        while value > current {
            match self.max.compare_exchange_weak(
                current,
                value,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current = x,
            }
        }
    }

    /// converts a value to bucket index
    ///
    /// for values < 2^SUB_BITS (8): linear mapping (full precision).
    /// for values >= 8: magnitude + top SUB_BITS after leading 1-bit.
    /// all bit ops, no branches on hot path (except the magnitude check).
    #[inline(always)]
    fn value_to_bucket(value: u64) -> usize {
        if value == 0 {
            return 0;
        }
        let mag = 63 - value.leading_zeros(); // floor(log2(value))
        if mag < SUB_BITS {
            // linear region: values 1..7 map to buckets 1..7
            value as usize
        } else {
            let shift = mag - SUB_BITS;
            let sub = ((value >> shift) as usize) & (SUB_BUCKETS - 1);
            let base = ((mag - SUB_BITS) as usize + 1) * SUB_BUCKETS;
            let bucket = base + sub;
            bucket.min(NUM_BUCKETS - 1)
        }
    }

    /// converts a bucket index to the minimum value in that bucket
    #[inline(always)]
    fn bucket_to_value(bucket: usize) -> u64 {
        if bucket < SUB_BUCKETS as usize {
            // linear region
            bucket as u64
        } else {
            let group = (bucket / SUB_BUCKETS) - 1; // 0-based magnitude group
            let sub = bucket & (SUB_BUCKETS - 1);
            let mag = group + SUB_BITS as usize; // actual magnitude
            let shift = mag - SUB_BITS as usize;
            (1u64 << mag) | ((sub as u64) << shift)
        }
    }

    /// clears the histogram
    pub fn clear(&self) {
        for bucket in &self.buckets {
            bucket.store(0, Ordering::Relaxed);
        }
        self.count.store(0, Ordering::Relaxed);
        self.sum.store(0, Ordering::Relaxed);
        self.min.store(u64::MAX, Ordering::Relaxed);
        self.max.store(0, Ordering::Relaxed);
    }

    /// gets the total count of samples
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// gets the minimum value
    pub fn min(&self) -> u64 {
        let min = self.min.load(Ordering::Relaxed);
        if min == u64::MAX {
            0
        } else {
            min
        }
    }

    /// gets the maximum value
    pub fn max(&self) -> u64 {
        self.max.load(Ordering::Relaxed)
    }

    /// calculates the mean value
    pub fn mean(&self) -> u64 {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            0
        } else {
            self.sum.load(Ordering::Relaxed) / count
        }
    }

    /// calculates percentiles
    pub fn percentiles(&self, percentiles: &[f64]) -> Vec<u64> {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            return vec![0; percentiles.len()];
        }

        // collect cumulative bucket counts on the stack (no allocation)
        let mut cum_counts = [0u64; NUM_BUCKETS];
        let mut cumulative = 0u64;
        for i in 0..NUM_BUCKETS {
            cumulative += self.buckets[i].load(Ordering::Relaxed);
            cum_counts[i] = cumulative;
        }

        // calculate percentiles
        let mut results = Vec::with_capacity(percentiles.len());
        for &p in percentiles {
            let target = ((count as f64 * p / 100.0) as u64).max(1);

            let mut value = 0u64;
            for i in 0..NUM_BUCKETS {
                if cum_counts[i] >= target {
                    value = Self::bucket_to_value(i);
                    break;
                }
            }
            results.push(value);
        }

        results
    }

    /// gets common percentiles (p50, p90, p99, p999, p9999)
    pub fn common_percentiles(&self) -> Percentiles {
        let values = self.percentiles(&[50.0, 90.0, 99.0, 99.9, 99.99]);
        Percentiles {
            p50: values[0],
            p90: values[1],
            p99: values[2],
            p999: values[3],
            p9999: values[4],
        }
    }

    /// gets latency statistics
    pub fn stats(&self) -> super::LatencyStats {
        let percentiles = self.common_percentiles();
        super::LatencyStats {
            min: self.min(),
            max: self.max(),
            mean: self.mean(),
            p50: percentiles.p50,
            p90: percentiles.p90,
            p99: percentiles.p99,
            p999: percentiles.p999,
            p9999: percentiles.p9999,
            count: self.count(),
        }
    }
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

/// common percentile values
#[derive(Debug, Clone, Copy, Default)]
pub struct Percentiles {
    pub p50: u64,
    pub p90: u64,
    pub p99: u64,
    pub p999: u64,
    pub p9999: u64,
}

impl Percentiles {
    /// formats percentiles as a string
    pub fn format(&self) -> String {
        format!(
            "p50={}ns p90={}ns p99={}ns p999={}ns p9999={}ns",
            self.p50, self.p90, self.p99, self.p999, self.p9999
        )
    }

    /// formats with microsecond units
    pub fn format_micros(&self) -> String {
        format!(
            "p50={}μs p90={}μs p99={}μs p999={}μs p9999={}μs",
            self.p50 / 1000,
            self.p90 / 1000,
            self.p99 / 1000,
            self.p999 / 1000,
            self.p9999 / 1000
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_basic() {
        let hist = Histogram::new();

        // record some values
        hist.record(100);
        hist.record(200);
        hist.record(300);
        hist.record(400);
        hist.record(500);

        assert_eq!(hist.count(), 5);
        assert_eq!(hist.min(), 100);
        assert_eq!(hist.max(), 500);
        assert_eq!(hist.mean(), 300);
    }

    #[test]
    fn test_percentiles() {
        let hist = Histogram::new();

        // record values 1-100 * 1000ns
        for i in 1..=100 {
            hist.record(i * 1000);
        }

        let percentiles = hist.common_percentiles();

        // p50 should be near 50_000ns (within ~12.5% bucket precision)
        assert!(
            percentiles.p50 >= 40_000 && percentiles.p50 <= 56_000,
            "p50={} expected near 50000",
            percentiles.p50
        );

        // p99 should be near 99_000ns
        assert!(
            percentiles.p99 >= 88_000 && percentiles.p99 <= 104_000,
            "p99={} expected near 99000",
            percentiles.p99
        );
    }

    #[test]
    fn test_bucket_conversion() {
        // linear region: values 0-7 map directly
        assert_eq!(Histogram::value_to_bucket(0), 0);
        assert_eq!(Histogram::value_to_bucket(1), 1);
        assert_eq!(Histogram::value_to_bucket(7), 7);

        // magnitude 3 (values 8-15): 8 sub-buckets, each covers 1
        assert_eq!(Histogram::value_to_bucket(8), 8);
        assert_eq!(Histogram::value_to_bucket(15), 15);

        // magnitude 4 (values 16-31): 8 sub-buckets, each covers 2
        assert_eq!(Histogram::value_to_bucket(16), 16);
        assert_eq!(Histogram::value_to_bucket(17), 16);
        assert_eq!(Histogram::value_to_bucket(30), 23);
        assert_eq!(Histogram::value_to_bucket(31), 23);

        // round-trip: bucket_to_value gives lower bound of bucket
        for v in [1u64, 7, 8, 15, 16, 31, 100, 1000, 1_000_000, 1_000_000_000] {
            let bucket = Histogram::value_to_bucket(v);
            let lower = Histogram::bucket_to_value(bucket);
            assert!(
                lower <= v,
                "bucket_to_value({})={} > original value {}",
                bucket,
                lower,
                v
            );
        }
    }

    #[test]
    fn test_sub_bucket_precision() {
        let hist = Histogram::new();

        // at 1μs scale, should distinguish 1000ns from 1200ns
        hist.record(1000);
        hist.record(1200);

        let b1 = Histogram::value_to_bucket(1000);
        let b2 = Histogram::value_to_bucket(1200);
        assert_ne!(b1, b2, "1000ns and 1200ns should be in different buckets");

        // verify bucket lower bounds are reasonable
        let v1 = Histogram::bucket_to_value(b1);
        let v2 = Histogram::bucket_to_value(b2);
        assert!(v1 <= 1000);
        assert!(v2 <= 1200);
        assert!(v2 > v1);
    }

    #[test]
    fn test_clear() {
        let hist = Histogram::new();

        hist.record(100);
        hist.record(200);
        assert_eq!(hist.count(), 2);

        hist.clear();
        assert_eq!(hist.count(), 0);
        assert_eq!(hist.min(), 0);
        assert_eq!(hist.max(), 0);
    }

    #[test]
    fn test_zero_value_is_recorded() {
        let hist = Histogram::new();

        hist.record(0);
        hist.record(10);

        assert_eq!(hist.count(), 2);
        assert_eq!(hist.min(), 0);
        assert_eq!(hist.max(), 10);

        let percentiles = hist.common_percentiles();
        assert_eq!(percentiles.p50, 0);

        let exact = hist.percentiles(&[100.0]);
        assert_eq!(exact[0], 10);
    }
}
