//! rdtsc (read time-stamp counter) wrapper for x86_64
//!
//! provides access to cpu cycle counter for ultra-precise timing.
//! on non-x86_64 platforms, falls back to standard time functions.

use std::time::{Duration, Instant};

/// reads the time-stamp counter
///
/// returns the current value of the processor's time-stamp counter.
/// this is the number of clock cycles since the last reset.
#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub fn rdtsc() -> u64 {
    unsafe {
        let lo: u32;
        let hi: u32;
        // use inline assembly to read tsc
        std::arch::asm!(
            "rdtsc",
            out("eax") lo,
            out("edx") hi,
            options(nostack, nomem),
        );
        ((hi as u64) << 32) | (lo as u64)
    }
}

/// reads the time-stamp counter with serialization
///
/// rdtscp is a serializing variant that waits for all previous instructions
/// to complete before reading the counter. slightly slower but more accurate.
#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub fn rdtscp() -> u64 {
    unsafe {
        let lo: u32;
        let hi: u32;
        let _aux: u32;
        // use inline assembly to read tsc with serialization
        std::arch::asm!(
            "rdtscp",
            out("eax") lo,
            out("edx") hi,
            out("ecx") _aux,
            options(nostack, nomem),
        );
        ((hi as u64) << 32) | (lo as u64)
    }
}

/// serialized start timestamp: lfence drains the pipeline then rdtsc reads.
/// use this for the START of a measurement interval.
/// cost: ~5-10 cycles more than plain rdtsc, but prevents ooo reordering.
#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub fn rdtsc_start() -> u64 {
    unsafe {
        let lo: u32;
        let hi: u32;
        std::arch::asm!(
            "lfence",
            "rdtsc",
            out("eax") lo,
            out("edx") hi,
            options(nostack, nomem),
        );
        ((hi as u64) << 32) | (lo as u64)
    }
}

/// serialized end timestamp: rdtscp waits for prior instructions,
/// lfence prevents later instructions from speculatively executing before read.
/// use this for the END of a measurement interval.
#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub fn rdtsc_end() -> u64 {
    unsafe {
        let lo: u32;
        let hi: u32;
        let _aux: u32;
        std::arch::asm!(
            "rdtscp",
            "lfence",
            out("eax") lo,
            out("edx") hi,
            out("ecx") _aux,
            options(nostack, nomem),
        );
        ((hi as u64) << 32) | (lo as u64)
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn rdtsc() -> u64 {
    0
}

#[cfg(not(target_arch = "x86_64"))]
pub fn rdtscp() -> u64 {
    0
}

#[cfg(not(target_arch = "x86_64"))]
pub fn rdtsc_start() -> u64 {
    0
}

#[cfg(not(target_arch = "x86_64"))]
pub fn rdtsc_end() -> u64 {
    0
}

/// checks if the cpu supports invariant tsc (constant rate regardless of p-states)
///
/// queries cpuid leaf 0x80000007, edx bit 8.
/// without invariant tsc, rdtsc frequency changes with cpu frequency scaling.
#[cfg(target_arch = "x86_64")]
pub fn has_invariant_tsc() -> bool {
    unsafe {
        let edx: u32;
        std::arch::asm!(
            "push rbx",
            "mov eax, 0x80000007",
            "cpuid",
            "pop rbx",
            out("edx") edx,
            inout("eax") 0x80000007u32 => _,
            out("ecx") _,
            options(nomem, preserves_flags),
        );
        (edx & (1 << 8)) != 0
    }
}

/// calibrates the cpu frequency by measuring rdtsc over a known time period
///
/// takes multiple samples and uses the maximum observed frequency
/// to reduce impact of os scheduling/preemption on accuracy.
/// returns the frequency in cycles per second.
#[cfg(target_arch = "x86_64")]
pub fn calibrate_frequency() -> u64 {
    // warm up tsc and caches
    for _ in 0..100 {
        rdtsc();
    }

    // take 5 samples of 20ms each, use max (least affected by preemption)
    let mut best_freq: u64 = 0;

    for _ in 0..5 {
        let duration = Duration::from_millis(20);

        let start_instant = Instant::now();
        let start_cycles = rdtsc();

        while start_instant.elapsed() < duration {
            std::hint::spin_loop();
        }

        let end_cycles = rdtsc();
        let elapsed = start_instant.elapsed();

        let cycles = end_cycles - start_cycles;
        let nanos = elapsed.as_nanos();

        let freq = ((cycles as u128 * 1_000_000_000) / nanos) as u64;
        best_freq = best_freq.max(freq);
    }

    best_freq
}

/// converts cpu cycles to nanoseconds using precomputed Q32 fixed-point multiplier
///
/// uses (cycles * multiplier) >> 32 instead of u128 division.
/// single mulq + shift (~3-4 cycles) vs __udivti3 (~40-80 cycles).
#[inline(always)]
pub fn cycles_to_nanos(cycles: u64) -> u64 {
    let mult = super::nanos_multiplier();
    if mult == 0 {
        // not calibrated, assume ~3ghz
        cycles / 3
    } else {
        ((cycles as u128 * mult as u128) >> 32) as u64
    }
}

/// converts nanoseconds to cpu cycles using precomputed Q32 fixed-point multiplier
#[inline(always)]
pub fn nanos_to_cycles(nanos: u64) -> u64 {
    let mult = super::cycles_multiplier();
    if mult == 0 {
        // not calibrated, assume ~3ghz
        nanos * 3
    } else {
        ((nanos as u128 * mult as u128) >> 32) as u64
    }
}

/// memory fence to prevent instruction reordering
#[inline(always)]
pub fn fence() {
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
}

/// serializing fence using cpuid instruction
#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub fn serialize() {
    unsafe {
        // cpuid is a serializing instruction.
        // rbx is reserved by llvm for PIC/GOT base -- lateout("ebx") is rejected,
        // so we manually save/restore via push/pop. nostack is intentionally omitted
        // to let llvm account for the stack usage.
        std::arch::asm!(
            "push rbx",
            "cpuid",
            "pop rbx",
            inout("eax") 0 => _,
            out("ecx") _,
            out("edx") _,
            options(nomem, preserves_flags),
        );
    }
}

#[inline(always)]
#[cfg(not(target_arch = "x86_64"))]
pub fn serialize() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rdtsc() {
        let start = rdtsc();
        // do some work
        let mut sum = 0u64;
        for i in 0..1000 {
            sum = sum.wrapping_add(i);
        }
        let end = rdtsc();

        // should advance
        assert!(end >= start);

        // prevent optimization
        std::hint::black_box(sum);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_calibration() {
        let freq = calibrate_frequency();

        // typical cpu frequencies are 1-5 ghz
        assert!(freq > 500_000_000); // > 0.5 ghz
        assert!(freq < 10_000_000_000); // < 10 ghz

        eprintln!("calibrated frequency: {} ghz", freq as f64 / 1e9);
    }

    #[test]
    fn test_conversion() {
        super::super::init();

        let cycles = 1_000_000_000; // 1 billion cycles
        let nanos = cycles_to_nanos(cycles);
        let cycles2 = nanos_to_cycles(nanos);

        // should be approximately equal (some rounding error ok)
        let diff = if cycles2 > cycles {
            cycles2 - cycles
        } else {
            cycles - cycles2
        };

        assert!(diff < cycles / 100); // less than 1% error
    }
}
