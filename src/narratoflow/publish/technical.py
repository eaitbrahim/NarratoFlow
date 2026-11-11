"""Technical checks on the exported MP4: platform limits and the shipped loudness."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from narratoflow.audio.io import MIN_GATED_MS, Loudness, parse_loudnorm
from narratoflow.errors import RenderError
from narratoflow.publish.checks import CheckId, CheckResult, PartInputs, Severity, summarize
from narratoflow.publish.rules import LoudnessRules, PlatformRules, Rules
from narratoflow.render.ffmpeg import MediaProbe, RenderTools, probe, run_tool_capture

LOUDNESS_TIMEOUT_S = 600
_CHUNK = 1 << 20


@dataclass(frozen=True)
class MediaFacts:
    probe: MediaProbe
    loudness: Loudness
    sha256: str
    size_bytes: int


def measure_loudness(tools: RenderTools, video: Path, audio_ms: int | None) -> Loudness:
    """Loudness of the video's first audio stream (the AAC stereo track that ships)."""
    _, stderr = run_tool_capture(
        tools.ffmpeg,
        [
            "-hide_banner", "-nostdin", "-v", "info", "-i", str(video),
            "-map", "0:a:0", "-af", "loudnorm=I=-14:TP=-1:LRA=11:print_format=json",
            "-f", "null", "-",
        ],
        timeout_s=LOUDNESS_TIMEOUT_S,
    )  # fmt: skip
    return parse_loudnorm(stderr, gated=audio_ms is not None and audio_ms >= MIN_GATED_MS)


def _sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as fh:
            while chunk := fh.read(_CHUNK):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise RenderError(f"cannot read {path}: {exc}") from exc
    return digest.hexdigest(), size


def inspect_video(tools: RenderTools, video: Path) -> MediaFacts:
    """Probe, loudness and sha256 of `video`; `RenderError`/`AudioError` propagate."""
    media = probe(tools, video)
    loudness = measure_loudness(tools, video, media.audio_ms)
    sha256, size = _sha256(video)
    return MediaFacts(probe=media, loudness=loudness, sha256=sha256, size_bytes=size)


def _s(ms: int) -> str:
    return f"{ms / 1000:.1f}"


def _show(value: object) -> str:
    return "unknown" if value is None else str(value)


def _platform_findings(p: PlatformRules, media: MediaProbe) -> list[str]:
    reasons: list[str] = []
    if media.video_ms is None:
        reasons.append(f"video duration unknown; {p.name} allows at most {p.max_duration_s} s")
    elif media.video_ms > p.max_duration_s * 1000:
        reasons.append(
            f"{_s(media.video_ms)} s is longer than the {p.name} limit of {p.max_duration_s} s"
        )
    if (media.width, media.height) != (p.width, p.height):
        reasons.append(
            f"{_show(media.width)}x{_show(media.height)} is not the {p.name} size "
            f"{p.width}x{p.height}"
        )
    if media.fps != Fraction(p.fps):
        reasons.append(f"{_show(media.fps)} fps is not the {p.name} rate of {p.fps} fps")
    listed = (
        ("video codec", media.video_codec, p.video_codecs),
        ("pixel format", media.pix_fmt, p.pix_fmts),
        ("audio codec", media.audio_codec, p.audio_codecs),
    )
    for label, value, allowed in listed:
        if value not in allowed:
            reasons.append(
                f"{label} {_show(value)} is not allowed by {p.name} ({', '.join(allowed)})"
            )
    if media.audio_rate != p.audio_rate:
        reasons.append(
            f"audio rate {_show(media.audio_rate)} Hz is not the {p.name} rate of {p.audio_rate} Hz"
        )
    return reasons


def _finite(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


def _loudness_findings(loud: Loudness, rules: LoudnessRules) -> list[CheckResult]:
    found: list[CheckResult] = []

    def add(severity: Severity, reason: str) -> None:
        found.append(CheckResult(check=CheckId.TECHNICAL, severity=severity, reason=reason))

    integrated = _finite(loud.integrated_lufs)
    if integrated is None:
        add(Severity.FAIL, "loudness not measurable")
    elif abs(integrated - rules.target_lufs) > rules.tolerance_lu:
        add(
            Severity.FAIL,
            f"integrated loudness {integrated:.1f} LUFS is outside "
            f"{rules.target_lufs:.1f} +/-{rules.tolerance_lu:.1f} LU",
        )
    peak = _finite(loud.true_peak_dbtp)
    if peak is None:
        add(Severity.FAIL, "true peak not measurable")
    elif peak > rules.fail_true_peak_dbtp:
        add(
            Severity.FAIL,
            f"true peak {peak:.1f} dBTP is above {rules.fail_true_peak_dbtp:.1f} dBTP (clipping)",
        )
    elif peak > rules.warn_true_peak_dbtp:
        add(
            Severity.WARN,
            f"true peak {peak:.1f} dBTP is above {rules.warn_true_peak_dbtp:.1f} dBTP; "
            "AAC encoding may overshoot",
        )
    return found


def technical_checks(
    inputs: PartInputs, facts: MediaFacts | None, rules: Rules
) -> list[CheckResult]:
    """Platform limits, loudness and the channel's part-length target for the shipped MP4."""
    if facts is None:
        return [
            CheckResult(check=CheckId.TECHNICAL, severity=Severity.FAIL, reason="no video to check")
        ]
    media = facts.probe
    findings: list[CheckResult] = []
    for pid, p in rules.file.platforms.items():
        findings.extend(
            CheckResult(
                check=CheckId.TECHNICAL, severity=Severity.FAIL, reason=reason, platform=pid
            )
            for reason in _platform_findings(p, media)
        )
    findings.extend(_loudness_findings(facts.loudness, rules.file.loudness))
    channel = inputs.channel
    if media.video_ms is not None and not (
        channel.part_min_s * 1000 <= media.video_ms <= channel.part_max_s * 1000
    ):
        findings.append(
            CheckResult(
                check=CheckId.TECHNICAL,
                severity=Severity.WARN,
                reason=(
                    f"{_s(media.video_ms)} s is outside the channel's "
                    f"{channel.part_min_s}-{channel.part_max_s} s part target"
                ),
            )
        )
    if findings:
        return findings
    # No findings: duration, size, fps, integrated loudness and true peak are all known.
    names = ", ".join(p.name for p in rules.file.platforms.values())
    ok = (
        f"{_s(media.video_ms or 0)} s, {media.width}x{media.height} @ {media.fps} fps, "
        f"{facts.loudness.integrated_lufs:.1f} LUFS, {facts.loudness.true_peak_dbtp:.1f} dBTP; "
        f"fits {names}"
    )
    return summarize(CheckId.TECHNICAL, [], ok)
