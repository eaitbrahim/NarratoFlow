import hashlib
import re
from datetime import date, timedelta
from pathlib import Path

import pytest

from narratoflow.errors import PublishError
from narratoflow.publish.checks import Severity
from narratoflow.publish.rules import default_rules, home_free, load_rules, rules_checks
from narratoflow.settings import Settings

PACKAGED = Path(__file__).parents[2] / "src" / "narratoflow" / "data" / "platform_rules.toml"


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "platform_rules.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _packaged_text() -> str:
    return PACKAGED.read_text(encoding="utf-8")


def _settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path)


def test_packaged_rules_load_with_pinned_values(tmp_path: Path) -> None:
    rules = default_rules(_settings(tmp_path))
    f = rules.file
    assert rules.origin == "packaged"
    assert f.version == "2026-11-14.1"
    assert f.checked == date(2026, 11, 14)
    assert f.platforms["youtube-shorts"].max_duration_s == 180
    assert f.platforms["tiktok"].max_duration_s == 600
    for p in f.platforms.values():
        assert (p.width, p.height, p.fps) == (1080, 1920, 30)
    lo = f.loudness
    assert (lo.target_lufs, lo.tolerance_lu) == (-14.0, 1.0)
    assert (lo.warn_true_peak_dbtp, lo.fail_true_peak_dbtp) == (-1.0, 0.0)
    assert f.licences.assets["CC-BY-4.0"].attribution is True
    assert f.licences.voice["edge-tts-terms"].verdict == "warn"
    aligner = f.licences.assets["ctc-forced-aligner-terms"]
    assert aligner.verdict == "warn"
    assert "not cleared for commercial use" in aligner.note


def test_every_platform_has_https_sources_and_verify_note(tmp_path: Path) -> None:
    f = default_rules(_settings(tmp_path)).file
    assert f.loudness.verify.strip()
    for p in f.platforms.values():
        assert p.sources
        assert all(s.startswith("https://") for s in p.sources)
        assert p.verify.strip()
    for table in (f.licences.assets, f.licences.generated, f.licences.voice):
        for rule in table.values():
            assert rule.verdict == "pass" or rule.note


def test_religious_discussion_fail_rejected(tmp_path: Path) -> None:
    text = _packaged_text()
    marker = "[content.religious_discussion]"
    head, tail = text.split(marker)
    bad = head + marker + tail.replace('max_severity = "warn"', 'max_severity = "fail"', 1)
    with pytest.raises(PublishError, match="religious_discussion"):
        load_rules(_write(tmp_path, bad))
    no_other = re.sub(r"\[content\.other\].*", "", text, flags=re.S)
    with pytest.raises(PublishError, match="other"):
        load_rules(_write(tmp_path, no_other))


def test_rules_sha256_is_of_file_bytes(tmp_path: Path) -> None:
    path = _write(tmp_path, _packaged_text())
    rules = load_rules(path)
    assert rules.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert rules.origin == str(path.resolve())


def test_data_dir_override_wins(tmp_path: Path) -> None:
    assert default_rules(_settings(tmp_path)).origin == "packaged"
    path = _write(tmp_path, _packaged_text().replace("2026-11-14.1", "override.1"))
    rules = default_rules(_settings(tmp_path))
    assert rules.origin == str(path.resolve())
    assert rules.file.version == "override.1"


def test_broken_rules_names_file(tmp_path: Path) -> None:
    text = _packaged_text()
    cases = [
        "this is = not [ toml",
        text.replace('version = "2026-11-14.1"\n', ""),
        text + '\nsurprise = "x"\n',
    ]
    for bad in cases:
        path = _write(tmp_path, bad)
        with pytest.raises(PublishError, match=re.escape(f"platform rules {path}")):
            load_rules(path)
    with pytest.raises(PublishError, match="cannot be read"):
        load_rules(tmp_path / "missing.toml")


def test_rules_age_warns_after_180_days(tmp_path: Path) -> None:
    rules = default_rules(_settings(tmp_path))
    checked = rules.file.checked
    (ok,) = rules_checks(rules, checked + timedelta(days=180))
    assert ok.severity is Severity.PASS
    assert ok.reason == "platform rules 2026-11-14.1 (checked 2026-11-14)"
    (warn,) = rules_checks(rules, checked + timedelta(days=181))
    assert warn.severity is Severity.WARN
    assert "over 180 days ago" in warn.reason


def test_rules_checked_in_the_future_warns(tmp_path: Path) -> None:
    rules = default_rules(_settings(tmp_path))
    (warn,) = rules_checks(rules, rules.file.checked - timedelta(days=1))
    assert warn.severity is Severity.WARN
    assert warn.reason == (
        "platform rules 2026-11-14.1 have a checked date 2026-11-14 in the future; "
        "fix data/platform_rules.toml"
    )


def test_loudness_peak_order_rejected(tmp_path: Path) -> None:
    bad = _packaged_text().replace("warn_true_peak_dbtp = -1.0", "warn_true_peak_dbtp = 0.5")
    with pytest.raises(PublishError, match="warn_true_peak_dbtp"):
        load_rules(_write(tmp_path, bad))


def test_content_category_keys_are_normalised_on_load(tmp_path: Path) -> None:
    text = _packaged_text().replace("[content.graphic_violence]", "[content.graphic-violence]")
    assert "[content.graphic-violence]" in text
    rules = load_rules(_write(tmp_path, text))
    assert "graphic_violence" in rules.file.content
    assert "graphic-violence" not in rules.file.content


def test_content_category_keys_colliding_once_normalised_are_refused(tmp_path: Path) -> None:
    text = _packaged_text() + (
        '\n[content.Graphic-Violence]\ndescription = "dup"\nmax_severity = "warn"\npolicies = []\n'
    )
    with pytest.raises(PublishError, match="Graphic-Violence"):
        load_rules(_write(tmp_path, text))


def test_own_work_licence_passes(tmp_path: Path) -> None:
    rules = default_rules(_settings(tmp_path))
    assert rules.file.version == "2026-11-14.1"
    own = rules.file.licences.assets["own-work"]
    assert (own.verdict, own.attribution) == ("pass", False)


def test_home_free_shows_a_home_prefix_as_tilde() -> None:
    home = Path.home()
    assert home_free(str(home / "data" / "rules.toml")) == "~/data/rules.toml"
    assert home_free(str(home)) == "~"
    assert home_free("packaged") == "packaged"
    assert home_free(str(home) + "x/rules.toml") == str(home) + "x/rules.toml"
