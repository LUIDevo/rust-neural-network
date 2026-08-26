// AI Generated RNG - I'm probably going to reimplement this with the rand crate but this will do for now

/// xorshift64 generator.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Avoid the all-zero state, which xorshift can't escape.
        Rng(if seed == 0 { 0x9E3779B97F4A7C15 } else { seed })
    }

    /// Uniform f32 in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        // Top 24 bits -> uniform float.
        (x >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Standard normal sample via Box-Muller.
    pub fn next_gaussian(&mut self) -> f32 {
        // u1 must be > 0 for ln; nudge away from 0.
        let u1 = self.next_f32().max(f32::MIN_POSITIVE);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}
