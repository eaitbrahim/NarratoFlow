from __future__ import annotations

import dataclasses
import hashlib
from fractions import Fraction
from pathlib import Path

import pytest
from fixtures.audio import tone
from fixtures.publish import PACKAGED_RULES, make_inputs, write_rules

from narratoflow.audio import io as aio
from narratoflow.audio.io import Loudness
from narratoflow.channels.config import ChannelConfig
from narratoflow.errors import RenderError
from narratoflow.library.repository import SketchLibrary
from narratoflow.publish.checks import CheckId, CheckResult, PartInputs, Severity
from narratoflow.publish.rules import Rules, load_rules
from narratoflow.publish.technical import (
    MediaFacts,
    inspect_video,
    measure_loudness,
    technical_checks,
)
from narratoflow.render.encoder import AAC_ARGS, X264_ARGS
from narratoflow.render.ffmpeg import MediaProbe, RenderTools, run_tool

GOOD_PROBE = MediaProbe(
    width=1080,
    height=1920,
    fps=Fraction(30),
    frames=2250,
    video_ms=75_000,
    audio_ms=75_000,
    video_codec="h264",
    pix_fmt="yuv420p",
    audio_codec="aac",
    audio_rate=48000,
)
GOOD_LOUDNESS = Loudness(-14.2, -1.5, 3.0, -24.2, 0.1)


def _facts(probe: MediaProbe = GOOD_PROBE, loudness: Loudness = GOOD_LOUDNESS) -> MediaFacts:
    return MediaFacts(probe=probe, loudness=loudness, sha256="0" * 64, size_bytes=1)


def _probe(**changes: object) -> MediaProbe:
    return dataclasses.replace(GOOD_PROBE, **changes)  # type: ignore[arg-type]


def _loud(**changes: object) -> Loudness:
    return dataclasses.replace(GOOD_LOUDNESS, **changes)  # type: ignore[arg-type]


@pytest.fixture
def rules() -> Rules:
    return load_rules(PACKAGED_RULES)


@pytest.fixture
def inputs(tmp_path: Path, library: SketchLibrary, channel: ChannelConfig) -> PartInputs:
    return make_inputs(tmp_path, library, channel)


def _by(results: list[CheckResult], severity: Severity) -> list[CheckResult]:
    return [r for r in results if r.severity is severity]


def test_good_facts_pass(inputs: PartInputs, rules: Rules) -> None:
    results = technical_checks(inputs, _facts(), rules)
    assert results == [
        CheckResult(
            check=CheckId.TECHNICAL,
            severity=Severity.PASS,
            reason=(
                "75.0 s, 1080x1920 @ 30 fps, -14.2 LUFS, -1.5 dBTP; fits YouTube Shorts, TikTok"
            ),
        )
    ]


def test_too_long_fails_per_platform(inputs: PartInputs, rules: Rules) -> None:
    results = technical_checks(inputs, _facts(_probe(video_ms=200_000)), rules)
    fails = _by(results, Severity.FAIL)
    assert [(f.platform, f.reason) for f in fails] == [
        ("youtube-shorts", "200.0 s is longer than YouTube Shorts's 180 s limit")
    ]
    assert all(r.check is CheckId.TECHNICAL for r in results)


def test_unknown_duration_fails_per_platform(inputs: PartInputs, rules: Rules) -> None:
    fails = _by(technical_checks(inputs, _facts(_probe(video_ms=None)), rules), Severity.FAIL)
    assert [f.platform for f in fails] == ["youtube-shorts", "tiktok"]
    assert all("duration" in f.reason for f in fails)


def test_wrong_size_or_fps_fails(inputs: PartInputs, rules: Rules) -> None:
    size = _by(technical_checks(inputs, _facts(_probe(width=720)), rules), Severity.FAIL)
    assert [f.platform for f in size] == ["youtube-shorts", "tiktok"]
    assert "720x1920" in size[0].reason and "1080x1920" in size[0].reason
    fps = _by(
        technical_checks(inputs, _facts(_probe(fps=Fraction(30000, 1001))), rules),
        Severity.FAIL,
    )
    assert [f.platform for f in fps] == ["youtube-shorts", "tiktok"]
    assert "30000/1001 fps" in fps[0].reason
    unknown = _by(
        technical_checks(inputs, _facts(_probe(fps=None, height=None)), rules), Severity.FAIL
    )
    assert len(unknown) == 4


def test_wrong_codecs_fail(inputs: PartInputs, rules: Rules) -> None:
    bad = _probe(video_codec="hevc", pix_fmt="yuv444p", audio_codec="opus", audio_rate=44100)
    fails = _by(technical_checks(inputs, _facts(bad), rules), Severity.FAIL)
    assert len(fails) == 8
    reasons = " | ".join(f.reason for f in fails)
    for word in ("hevc", "yuv444p", "opus", "44100"):
        assert word in reasons
    none = _probe(audio_codec=None, audio_rate=None)
    assert len(_by(technical_checks(inputs, _facts(none), rules), Severity.FAIL)) == 4


def test_unmeasurable_loudness_fails(inputs: PartInputs, rules: Rules) -> None:
    for loud in (_loud(integrated_lufs=None), _loud(integrated_lufs=float("nan"))):
        fails = _by(technical_checks(inputs, _facts(loudness=loud), rules), Severity.FAIL)
        assert [(f.platform, f.reason) for f in fails] == [(None, "loudness not measurable")]
    fails = _by(
        technical_checks(inputs, _facts(loudness=_loud(true_peak_dbtp=None)), rules),
        Severity.FAIL,
    )
    assert len(fails) == 1 and "true peak" in fails[0].reason
    assert fails[0].platform is None


def test_true_peak_between_minus_one_and_zero_warns_above_zero_fails(
    inputs: PartInputs, rules: Rules
) -> None:
    at_warn = technical_checks(inputs, _facts(loudness=_loud(true_peak_dbtp=-1.0)), rules)
    assert [r.severity for r in at_warn] == [Severity.PASS]
    warn = technical_checks(inputs, _facts(loudness=_loud(true_peak_dbtp=-0.4)), rules)
    assert [(r.severity, r.platform) for r in warn] == [(Severity.WARN, None)]
    assert "-0.4 dBTP" in warn[0].reason
    at_fail = technical_checks(inputs, _facts(loudness=_loud(true_peak_dbtp=0.0)), rules)
    assert [r.severity for r in at_fail] == [Severity.WARN]
    fail = technical_checks(inputs, _facts(loudness=_loud(true_peak_dbtp=0.3)), rules)
    assert [(r.severity, r.platform) for r in fail] == [(Severity.FAIL, None)]
    assert "0.3 dBTP" in fail[0].reason


def test_peak_thresholds_come_from_rules(tmp_path: Path, inputs: PartInputs) -> None:
    rules = load_rules(
        write_rules(tmp_path, **{"loudness.warn_true_peak_dbtp": -3.0, "loudness.tolerance_lu": 3})
    )
    results = technical_checks(inputs, _facts(loudness=_loud(true_peak_dbtp=-2.0)), rules)
    assert [r.severity for r in results] == [Severity.WARN]
    off = technical_checks(
        inputs, _facts(loudness=_loud(integrated_lufs=-16.5, true_peak_dbtp=-4.0)), rules
    )
    assert [r.severity for r in off] == [Severity.PASS]


def test_off_target_lufs_fails(inputs: PartInputs, rules: Rules) -> None:
    fails = _by(
        technical_checks(inputs, _facts(loudness=_loud(integrated_lufs=-16.04)), rules),
        Severity.FAIL,
    )
    assert [(f.platform, f.reason) for f in fails] == [
        (None, "integrated loudness -16.0 LUFS is outside -14.0 +/-1.0 LU")
    ]
    edge = technical_checks(inputs, _facts(loudness=_loud(integrated_lufs=-13.0)), rules)
    assert [r.severity for r in edge] == [Severity.PASS]


def test_outside_channel_target_warns(inputs: PartInputs, rules: Rules) -> None:
    for ms in (45_000, 120_000):
        results = technical_checks(inputs, _facts(_probe(video_ms=ms)), rules)
        assert [(r.severity, r.platform) for r in results] == [(Severity.WARN, None)]
        assert results[0].reason == (
            f"{ms / 1000:.1f} s is outside the channel's 60-90 s part target"
        )
    for ms in (60_000, 90_000):
        results = technical_checks(inputs, _facts(_probe(video_ms=ms)), rules)
        assert [r.severity for r in results] == [Severity.PASS]


def test_missing_video_single_fail(inputs: PartInputs, rules: Rules) -> None:
    assert technical_checks(inputs, None, rules) == [
        CheckResult(check=CheckId.TECHNICAL, severity=Severity.FAIL, reason="no video to check")
    ]


def _mux(tools: RenderTools, tmp_path: Path, flac: Path, seconds: int) -> Path:
    out = tmp_path / "part.mp4"
    run_tool(
        tools.ffmpeg,
        [
            "-hide_banner", "-nostdin", "-v", "error", "-y",
            "-f", "lavfi", "-i", f"color=c=white:s=1080x1920:r=30:d={seconds}",
            "-i", str(flac),
            "-map", "0:v:0", "-map", "1:a:0",
            *X264_ARGS, *AAC_ARGS, "-shortest",
            str(out),
        ],
    )  # fmt: skip
    return out


def test_inspect_master_after_aac_stereo_mux(
    probe_tools: RenderTools,
    tmp_path: Path,
    library: SketchLibrary,
    channel: ChannelConfig,
    rules: Rules,
) -> None:
    mastered, _ = aio.master(tone(5000))
    flac = tmp_path / "mix.flac"
    aio.encode_flac(mastered, flac)
    video = _mux(probe_tools, tmp_path, flac, 5)
    facts = inspect_video(probe_tools, video)
    assert facts.loudness.integrated_lufs is not None
    assert abs(facts.loudness.integrated_lufs + 14) <= 1.0
    assert facts.loudness.true_peak_dbtp is not None
    assert facts.loudness.true_peak_dbtp <= -1.0
    assert facts.probe.audio_codec == "aac" and facts.probe.width == 1080
    assert facts.sha256 == hashlib.sha256(video.read_bytes()).hexdigest()
    assert facts.size_bytes == video.stat().st_size
    inputs = make_inputs(tmp_path / "render", library, channel)
    results = technical_checks(inputs, facts, rules)
    assert _by(results, Severity.FAIL) == []
    # 5 s is short of the channel's 60 s target: a WARN, never a FAIL.
    assert [r.severity for r in results] == [Severity.WARN]


def test_measure_loudness_short_clip_is_ungated(probe_tools: RenderTools, tmp_path: Path) -> None:
    flac = tmp_path / "mix.flac"
    aio.encode_flac(tone(1000), flac)
    video = _mux(probe_tools, tmp_path, flac, 1)
    assert measure_loudness(probe_tools, video, 300).integrated_lufs is None
    assert measure_loudness(probe_tools, video, None).integrated_lufs is None
    assert measure_loudness(probe_tools, video, 1000).integrated_lufs is not None


def test_inspect_video_errors_propagate(probe_tools: RenderTools, tmp_path: Path) -> None:
    corrupt = tmp_path / "bad.mp4"
    corrupt.write_bytes(b"not an mp4")
    with pytest.raises(RenderError):
        inspect_video(probe_tools, corrupt)
    silent = tmp_path / "silent.mp4"
    run_tool(
        probe_tools.ffmpeg,
        [
            "-hide_banner", "-nostdin", "-v", "error", "-y",
            "-f", "lavfi", "-i", "color=c=white:s=64x64:r=30:d=0.5",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(silent),
        ],
    )  # fmt: skip
    with pytest.raises(RenderError):
        inspect_video(probe_tools, silent)


def test_unreadable_video_is_render_error(tmp_path: Path) -> None:
    from narratoflow.publish import technical

    with pytest.raises(RenderError, match="cannot read"):
        technical._sha256(tmp_path)
