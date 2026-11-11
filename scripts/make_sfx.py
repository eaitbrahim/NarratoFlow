"""Synthesize the four marker squeaks into assets/sfx/ (needs ffmpeg). Run manually:

    uv run python scripts/make_sfx.py

Band-passed noise, 150-300 ms, attack/decay envelope, peak -6 dBFS, fixed seeds so the
output is reproducible. NarratoFlow's own work, Apache-2.0.
"""

from pathlib import Path

import numpy as np

from narratoflow.audio.io import SAMPLE_RATE, encode_flac

OUT = Path(__file__).parents[1] / "assets" / "sfx"
PEAK = 10 ** (-6 / 20)
# (seed, duration ms, band low Hz, band high Hz)
SPECS = [(11, 150, 1800, 3600), (22, 200, 1500, 3200), (33, 250, 1200, 2800), (44, 300, 1000, 2400)]


def squeak(seed: int, ms: int, low: float, high: float) -> np.ndarray:
    n = SAMPLE_RATE * ms // 1000
    spec = np.fft.rfft(np.random.default_rng(seed).standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1 / SAMPLE_RATE)
    spec[(freqs < low) | (freqs > high)] = 0
    band = np.fft.irfft(spec, n)
    t = np.arange(n) / n
    env = np.minimum(t / 0.1, 1.0) * np.exp(-3.0 * t) * np.minimum((1 - t) / 0.05, 1.0)
    out = band * env
    return (out / np.max(np.abs(out)) * PEAK).astype(np.float32)


def main() -> None:
    for i, (seed, ms, low, high) in enumerate(SPECS, start=1):
        encode_flac(squeak(seed, ms, low, high), OUT / f"marker-{i}.flac")
    entries = "".join(
        f'\n[sfx.marker-{i}]\nfile = "marker-{i}.flac"\nlicense = "Apache-2.0"\n'
        f'source = "scripts/make_sfx.py"\n'
        for i in range(1, len(SPECS) + 1)
    )
    (OUT / "index.toml").write_text(entries.lstrip("\n"), encoding="utf-8")


if __name__ == "__main__":
    main()
