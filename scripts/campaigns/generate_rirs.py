"""
Generate diverse synthetic Room Impulse Responses (RIRs) using pyroomacoustics.

Creates a bank of 600+ RIRs with randomized room parameters for
Yakura-style Expectation-over-Transformation (EoT) OTA attack training.

Room parameters are sampled uniformly from realistic ranges:
  - Room dimensions: small (3x3x2.5m) to large (15x10x4m)
  - RT60: 0.1s to 1.2s (anechoic to reverberant)
  - Source/mic positions: random within room bounds
  - Source-mic distance: 0.3m to 5m

Usage:
    python generate_rirs.py --output-dir data/rirs --n-rirs 615
    python generate_rirs.py --output-dir data/rirs --n-rirs 615 --sample-rate 24000
"""

import argparse
import os

import numpy as np
import soundfile as sf


def generate_rir_pyroomacoustics(
    room_dim, rt60, src_pos, mic_pos, sample_rate=24000
):
    """Generate a single RIR using pyroomacoustics image source method."""
    import pyroomacoustics as pra

    # Use Sabine's formula to compute absorption and max_order
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

    room = pra.ShoeBox(
        room_dim,
        fs=sample_rate,
        materials=pra.Material(e_absorption),
        max_order=max_order,
    )
    room.add_source(src_pos)
    room.add_microphone(mic_pos)
    room.compute_rir()

    rir = room.rir[0][0]  # [T] numpy array
    # Normalize peak to 1.0
    rir = rir / (np.max(np.abs(rir)) + 1e-10)
    return rir.astype(np.float32)


def generate_rir_scipy(room_dim, rt60, src_pos, mic_pos, sample_rate=24000):
    """
    Fallback: generate a synthetic RIR without pyroomacoustics.

    Uses a simple exponential decay model with randomized early reflections.
    Less physically accurate but captures the main characteristics.
    """
    distance = np.sqrt(sum((s - m) ** 2 for s, m in zip(src_pos, mic_pos)))
    speed_of_sound = 343.0  # m/s

    # Direct path delay
    direct_delay = distance / speed_of_sound
    direct_sample = int(direct_delay * sample_rate)

    # RIR length from RT60
    rir_length = int(rt60 * sample_rate * 1.5)
    rir_length = max(rir_length, sample_rate // 4)  # at least 250ms
    rir = np.zeros(rir_length, dtype=np.float32)

    # Direct path
    if direct_sample < rir_length:
        rir[direct_sample] = 1.0

    # Early reflections (6 walls)
    for dim_idx in range(3):
        for sign in [-1, 1]:
            # Image source position
            img_pos = list(src_pos)
            img_pos[dim_idx] = sign * room_dim[dim_idx] - src_pos[dim_idx]
            img_dist = np.sqrt(sum((i - m) ** 2 for i, m in zip(img_pos, mic_pos)))
            img_delay = int(img_dist / speed_of_sound * sample_rate)
            if img_delay < rir_length:
                # Attenuation from distance + wall absorption
                atten = (distance / (img_dist + 1e-6)) * np.random.uniform(0.3, 0.8)
                rir[img_delay] += atten * np.random.choice([-1, 1])

    # Late reverberation: exponential decay noise
    if rt60 > 0.05:
        decay_rate = 6.9 / (rt60 * sample_rate)  # -60dB in rt60 seconds
        t = np.arange(direct_sample, rir_length)
        envelope = np.exp(-decay_rate * (t - direct_sample))
        noise = np.random.randn(len(t)).astype(np.float32) * 0.1
        rir[direct_sample:] += noise * envelope

    # Normalize
    rir = rir / (np.max(np.abs(rir)) + 1e-10)
    return rir


def random_room_params(rng):
    """Sample random room parameters."""
    # Room dimensions (meters)
    length = rng.uniform(3.0, 15.0)
    width = rng.uniform(3.0, 10.0)
    height = rng.uniform(2.4, 4.0)
    room_dim = [length, width, height]

    # RT60 (reverberation time)
    rt60 = rng.uniform(0.1, 1.2)

    # Source position (inside room with 0.5m margin)
    margin = 0.5
    src_pos = [
        rng.uniform(margin, length - margin),
        rng.uniform(margin, width - margin),
        rng.uniform(0.8, min(2.0, height - margin)),  # ~speaker height
    ]

    # Mic position (inside room with margin, different from source)
    mic_pos = [
        rng.uniform(margin, length - margin),
        rng.uniform(margin, width - margin),
        rng.uniform(0.8, min(1.8, height - margin)),  # ~mic height
    ]

    # Ensure minimum distance between source and mic
    distance = np.sqrt(sum((s - m) ** 2 for s, m in zip(src_pos, mic_pos)))
    if distance < 0.3:
        # Push mic away from source
        direction = np.array(mic_pos) - np.array(src_pos)
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        mic_pos = (np.array(src_pos) + direction * 0.5).tolist()
        # Clamp to room bounds
        for i in range(3):
            mic_pos[i] = max(margin, min(room_dim[i] - margin, mic_pos[i]))

    return room_dim, rt60, src_pos, mic_pos


def main():
    parser = argparse.ArgumentParser(description="Generate diverse synthetic RIRs")
    parser.add_argument("--output-dir", type=str, default="data/rirs",
                        help="Output directory for RIR .wav files")
    parser.add_argument("--n-rirs", type=int, default=615,
                        help="Number of RIRs to generate (default: 615)")
    parser.add_argument("--sample-rate", type=int, default=24000,
                        help="Sample rate (default: 24000, EnCodec native)")
    parser.add_argument("--max-length-ms", type=float, default=500.0,
                        help="Maximum RIR length in ms (default: 500)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    max_samples = int(args.max_length_ms / 1000.0 * args.sample_rate)

    # Try pyroomacoustics first, fall back to scipy-based generation
    try:
        import pyroomacoustics  # noqa: F401
        use_pra = True
        print(f"Using pyroomacoustics for physically-accurate RIR generation")
    except ImportError:
        use_pra = False
        print(f"pyroomacoustics not available, using scipy-based synthetic RIRs")
        print(f"Install with: pip install pyroomacoustics")

    print(f"Generating {args.n_rirs} RIRs at {args.sample_rate}Hz...")
    print(f"Output directory: {args.output_dir}")
    print(f"Max length: {args.max_length_ms:.0f}ms ({max_samples} samples)")

    generated = 0
    failed = 0

    for i in range(args.n_rirs):
        room_dim, rt60, src_pos, mic_pos = random_room_params(rng)

        try:
            if use_pra:
                rir = generate_rir_pyroomacoustics(
                    room_dim, rt60, src_pos, mic_pos, args.sample_rate
                )
            else:
                rir = generate_rir_scipy(
                    room_dim, rt60, src_pos, mic_pos, args.sample_rate
                )

            # Truncate to max length
            if len(rir) > max_samples:
                rir = rir[:max_samples]

            # Save as .wav
            fname = f"rir_{i:04d}.wav"
            fpath = os.path.join(args.output_dir, fname)
            sf.write(fpath, rir, args.sample_rate)
            generated += 1

            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{args.n_rirs}...")

        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  Warning: RIR {i} failed: {e}")
            continue

    print(f"\nDone: {generated} RIRs generated, {failed} failed")
    print(f"Saved to: {args.output_dir}")

    # Save metadata
    import json
    meta = {
        "n_rirs": generated,
        "sample_rate": args.sample_rate,
        "max_length_ms": args.max_length_ms,
        "seed": args.seed,
        "use_pyroomacoustics": use_pra,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
