from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from conftest import FIXTURES
from fixtures.audio import speechlike, to_wav_bytes

from narratoflow.audio import io as aio
from narratoflow.audio import recitation
from narratoflow.audio.recitation import (
    AYAH_GAP_MS,
    AyahSegment,
    RecitationSource,
    display_words,
    verse_clip,
)
from narratoflow.audio.timeline import TimingSource
from narratoflow.errors import ProviderError, ReciterError
from narratoflow.models import VerseRef
from narratoflow.quran.corpus import QuranCorpus
from narratoflow.quran.reciters import Reciter

Handler = Callable[[httpx.Request], httpx.Response]
CORPUS = QuranCorpus.from_files(
    FIXTURES / "quran/quran-simple-clean.txt", FIXTURES / "quran/quran-uthmani.txt"
)
WAV = to_wav_bytes(speechlike(2))
API = "https://api.quran.com/api/v4/recitations/7/by_ayah/"

# U+06D6 small high ligature sad with lam with alef maksura (a pause mark).
PAUSE = "\u06d6"
# U+06DE rub el hizb, which may open an ayah.
HIZB = "\u06de"
WORD_A = "\u0628\u0650\u0633\u0652\u0645\u0650"
WORD_B = "\u0671\u0644\u0644\u0651\u064e\u0647\u0650"
WORD_C = "\u062d\u064e\u0642\u0651\u064c"


def qurancom() -> Reciter:
    return Reciter(
        id="qc-hafs",
        name="QC",
        riwaya="hafs",
        kind="qurancom",
        recitation_id=7,
        license="Example",
        license_url="https://example.org/terms",
        source="https://quran.com/",
        segments_source="quran.com",
        checked=date(2026, 11, 11),
    )


def files_reciter() -> Reciter:
    return Reciter(
        id="files-hafs",
        name="Files",
        riwaya="hafs",
        kind="files",
        audio_url="https://audio.example.org/r/{surah:03d}{ayah:03d}.mp3",
        license="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        source="https://audio.example.org/r/",
        checked=date(2026, 11, 11),
    )


def source(reciter: Reciter, tmp_path: Path, handler: Handler) -> RecitationSource:
    return RecitationSource(
        reciter, tmp_path, http=httpx.Client(transport=httpx.MockTransport(handler))
    )


def api_server(
    audio: dict[str, bytes],
    segments: dict[str, Any],
    seen: list[str],
    relative: bool = True,
) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        seen.append(url)
        if url.startswith(API):
            key = url.removeprefix(API)
            if key not in audio:
                return httpx.Response(404, json={"status": 404, "error": "SECRETBODY"})
            s, a = key.split(":")
            path = f"Alafasy/mp3/{int(s):03d}{int(a):03d}.mp3"
            file_url = path if relative else f"https://cdn.example.org/{path}"
            body = {
                "audio_files": [
                    {"verse_key": key, "url": file_url, "segments": segments.get(key, [])}
                ],
                "pagination": {},
            }
            return httpx.Response(200, json=body)
        for key, data in audio.items():
            s, a = key.split(":")
            if url.endswith(f"/{int(s):03d}{int(a):03d}.mp3"):
                return httpx.Response(200, content=data, headers={"content-type": "audio/mpeg"})
        return httpx.Response(404)

    return handler


def test_qurancom_fetch_reads_url_and_segments(tmp_path: Path) -> None:
    seen: list[str] = []
    segs = [[1, 0, 400], [2, 450, 900]]
    src = source(qurancom(), tmp_path, api_server({"19:2": WAV}, {"19:2": segs}, seen))
    got = src.fetch(19, 2)
    assert seen == [API + "19:2", "https://verses.quran.com/Alafasy/mp3/019002.mp3"]
    assert got.surah == 19
    assert got.ayah == 2
    assert got.audio == WAV
    assert got.segments == [
        AyahSegment(word=1, start_ms=0, end_ms=400),
        AyahSegment(word=2, start_ms=450, end_ms=900),
    ]
    cache = tmp_path / "qc-hafs"
    assert (cache / "019002.mp3").read_bytes() == WAV
    assert json.loads((cache / "019002.segments.json").read_text()) == segs
    assert not [p for p in cache.iterdir() if p.name.startswith(".")]


def test_qurancom_absolute_url_is_used(tmp_path: Path) -> None:
    seen: list[str] = []
    handler = api_server({"19:2": WAV}, {}, seen, relative=False)
    got = source(qurancom(), tmp_path, handler).fetch(19, 2)
    assert seen[1] == "https://cdn.example.org/Alafasy/mp3/019002.mp3"
    assert got.segments is None  # an empty segment list is no timing


def test_second_fetch_uses_cache_no_requests(tmp_path: Path) -> None:
    seen: list[str] = []
    segs = [[1, 0, 400]]
    first = source(qurancom(), tmp_path, api_server({"19:2": WAV}, {"19:2": segs}, seen)).fetch(
        19, 2
    )

    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request {request.url}")

    again = source(qurancom(), tmp_path, refuse).fetch(19, 2)
    assert again == first


def test_cached_ayah_without_segments_file_has_no_segments(tmp_path: Path) -> None:
    cache = tmp_path / "files-hafs"
    cache.mkdir()
    (cache / "019002.mp3").write_bytes(WAV)

    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    got = source(files_reciter(), tmp_path, refuse).fetch(19, 2)
    assert got.segments is None
    assert got.audio == WAV


@pytest.mark.parametrize(
    "cached",
    [
        "{not json",
        json.dumps([[2, 0, 100], [1, 100, 200]]),  # non-monotonic
        json.dumps([{"word": 1, "start_ms": 0, "end_ms": 100}]),  # not triples
        json.dumps([]),
    ],
)
def test_invalid_cached_segments_refetch_the_ayah(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, cached: str
) -> None:
    cache = tmp_path / "qc-hafs"
    cache.mkdir()
    (cache / "019002.mp3").write_bytes(WAV)
    (cache / "019002.segments.json").write_text(cached, encoding="utf-8")
    seen: list[str] = []
    segs = [[1, 0, 400]]
    with caplog.at_level(logging.WARNING):
        got = source(qurancom(), tmp_path, api_server({"19:2": WAV}, {"19:2": segs}, seen)).fetch(
            19, 2
        )
    assert len(seen) == 2  # fetched again
    assert got.segments == [AyahSegment(word=1, start_ms=0, end_ms=400)]
    assert json.loads((cache / "019002.segments.json").read_text()) == segs
    assert "19:2" in caplog.text


def test_cached_non_audio_is_deleted_and_refetched(tmp_path: Path) -> None:
    cache = tmp_path / "files-hafs"
    cache.mkdir()
    (cache / "019002.mp3").write_bytes(b"")
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=WAV)

    got = source(files_reciter(), tmp_path, handler).fetch(19, 2)
    assert seen == ["https://audio.example.org/r/019002.mp3"]
    assert got.audio == WAV
    assert (cache / "019002.mp3").read_bytes() == WAV


def test_refetch_without_segments_removes_stale_segments(tmp_path: Path) -> None:
    cache = tmp_path / "qc-hafs"
    cache.mkdir()
    (cache / "019002.mp3").write_bytes(b"not audio")  # forces a refetch
    (cache / "019002.segments.json").write_text(json.dumps([[1, 0, 100]]), encoding="utf-8")
    seen: list[str] = []
    got = source(qurancom(), tmp_path, api_server({"19:2": WAV}, {}, seen)).fetch(19, 2)
    assert got.segments is None
    assert not (cache / "019002.segments.json").exists()

    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    assert source(qurancom(), tmp_path, refuse).fetch(19, 2).segments is None


def test_https_redirect_is_followed(tmp_path: Path) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.host == "audio.example.org":
            return httpx.Response(302, headers={"location": "https://mirror.example.org/a.mp3"})
        return httpx.Response(200, content=WAV)

    got = source(files_reciter(), tmp_path, handler).fetch(2, 255)
    assert seen == ["https://audio.example.org/r/002255.mp3", "https://mirror.example.org/a.mp3"]
    assert got.audio == WAV


def test_redirect_to_http_is_refused(tmp_path: Path) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.scheme == "http":
            return httpx.Response(200, content=WAV)
        return httpx.Response(302, headers={"location": "http://evil.example.org/a.mp3"})

    with pytest.raises(ProviderError, match="non-https"):
        source(files_reciter(), tmp_path, handler).fetch(2, 255)
    assert seen == ["https://audio.example.org/r/002255.mp3"]  # http hop never requested
    assert not (tmp_path / "files-hafs" / "002255.mp3").exists()


def test_redirect_loop_is_provider_error(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": str(request.url)})

    with pytest.raises(ProviderError, match="too many redirects"):
        source(files_reciter(), tmp_path, handler).fetch(2, 255)


def test_audio_body_over_cap_is_provider_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recitation, "MAX_AUDIO_BYTES", 1000)
    stream = httpx.ByteStream(b"ID3" + b"x" * 2000)  # no content-length: streamed check
    src = source(files_reciter(), tmp_path, lambda request: httpx.Response(200, stream=stream))
    with pytest.raises(ProviderError, match="exceeds 1000 bytes"):
        src.fetch(2, 255)
    declared = httpx.Response(200, content=b"ID3" + b"x" * 2000)  # content-length header
    src = source(files_reciter(), tmp_path, lambda request: declared)
    with pytest.raises(ProviderError, match="exceeds 1000 bytes"):
        src.fetch(2, 255)
    assert not (tmp_path / "files-hafs" / "002255.mp3").exists()


def test_api_body_over_cap_is_provider_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recitation, "MAX_JSON_BYTES", 100)
    body = {"audio_files": [{"url": "a/019002.mp3", "segments": [[1, 0, 1]] * 50}]}
    src = source(qurancom(), tmp_path, lambda request: httpx.Response(200, json=body))
    with pytest.raises(ProviderError, match="exceeds 100 bytes"):
        src.fetch(19, 2)


def test_source_closes_only_its_own_client(tmp_path: Path) -> None:
    with RecitationSource(files_reciter(), tmp_path) as owned:
        client = owned.http
    assert client.is_closed
    given = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    with RecitationSource(files_reciter(), tmp_path, http=given):
        pass
    assert not given.is_closed
    given.close()


def test_404_is_reciter_error_naming_ayah(tmp_path: Path) -> None:
    seen: list[str] = []
    src = source(qurancom(), tmp_path, api_server({}, {}, seen))
    with pytest.raises(ReciterError, match="QC has no recording of 19:9") as exc:
        src.fetch(19, 9)
    assert "SECRETBODY" not in str(exc.value)
    assert not (tmp_path / "qc-hafs" / "019009.mp3").exists()


def test_audio_404_is_reciter_error(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    with pytest.raises(ReciterError, match="Files has no recording of 2:255"):
        source(files_reciter(), tmp_path, handler).fetch(2, 255)


def test_files_fetch_uses_template_and_no_segments(tmp_path: Path) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=WAV)

    got = source(files_reciter(), tmp_path, handler).fetch(2, 255)
    assert seen == ["https://audio.example.org/r/002255.mp3"]
    assert got.segments is None
    assert (tmp_path / "files-hafs" / "002255.mp3").read_bytes() == WAV
    assert not (tmp_path / "files-hafs" / "002255.segments.json").exists()


@pytest.mark.parametrize(
    "segments",
    [
        [[1, 500, 400]],  # end before start
        [[2, 0, 100], [1, 100, 200]],  # words out of order
        [[1, 300, 400], [2, 100, 200]],  # starts go backwards
        [[0, 0, 100]],  # word index below 1
        [[1, -5, 100]],  # negative time
        [[1, 0]],  # short tuple
        [[1, 0, 100, 5]],  # long tuple
        [["1", 0, 100]],  # non-integer
        [[True, 0, 100]],  # bool is not an int here
        "nope",
        [[1, 0, 1e12]],  # absurd time
    ],
)
def test_malformed_segments_are_dropped_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, segments: Any
) -> None:
    seen: list[str] = []
    handler = api_server({"19:2": WAV}, {"19:2": segments}, seen)
    with caplog.at_level(logging.WARNING):
        got = source(qurancom(), tmp_path, handler).fetch(19, 2)
    assert got.segments is None
    assert "19:2" in caplog.text


def test_overlapping_segments_are_kept(tmp_path: Path) -> None:
    # Only starts must be non-decreasing; a word may end after the next starts.
    seen: list[str] = []
    segs = [[1, 0, 300], [2, 200, 400]]
    got = source(qurancom(), tmp_path, api_server({"19:2": WAV}, {"19:2": segs}, seen)).fetch(19, 2)
    assert got.segments is not None
    assert len(got.segments) == 2


@pytest.mark.parametrize(
    ("response", "fragment"),
    [
        (httpx.Response(500, text="SECRETBODY"), "HTTP 500"),
        (httpx.Response(429, text="SECRETBODY"), "HTTP 429"),
        (httpx.Response(200, text="SECRETBODY not json"), "JSON"),
        (httpx.Response(200, json={"audio_files": []}), "audio"),
        (httpx.Response(200, json=[1, 2]), "audio"),
        (httpx.Response(200, json={"audio_files": [{"url": 3}]}), "audio"),
        (httpx.Response(200, json={"audio_files": [{"url": "http://x.org/a.mp3"}]}), "https"),
    ],
)
def test_api_failures_are_provider_errors_without_body(
    tmp_path: Path, response: httpx.Response, fragment: str
) -> None:
    src = source(qurancom(), tmp_path, lambda request: response)
    with pytest.raises(ProviderError) as exc:
        src.fetch(19, 2)
    assert not isinstance(exc.value, ReciterError)
    assert fragment in str(exc.value)
    assert "SECRETBODY" not in str(exc.value)


def test_non_audio_body_is_provider_error(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>SECRETBODY</html>")

    with pytest.raises(ProviderError, match="not audio") as exc:
        source(files_reciter(), tmp_path, handler).fetch(2, 255)
    assert "SECRETBODY" not in str(exc.value)
    assert not (tmp_path / "files-hafs" / "002255.mp3").exists()


def test_timeout_and_transport_errors_are_provider_errors(tmp_path: Path) -> None:
    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ProviderError, match="timed out"):
        source(files_reciter(), tmp_path, slow).fetch(2, 255)

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    with pytest.raises(ProviderError, match="reach"):
        source(files_reciter(), tmp_path, down).fetch(2, 255)


def test_fetch_rejects_bad_ayah_numbers(tmp_path: Path) -> None:
    src = source(files_reciter(), tmp_path, lambda request: httpx.Response(200, content=WAV))
    for surah, ayah in ((0, 1), (115, 1), (2, 0), (2, 1000)):
        with pytest.raises(ReciterError):
            src.fetch(surah, ayah)


def test_display_words_attach_pause_marks() -> None:
    text = f"{WORD_A} {PAUSE} {WORD_B}  {WORD_C} {PAUSE}"
    assert display_words(text) == [f"{WORD_A} {PAUSE}", WORD_B, f"{WORD_C} {PAUSE}"]
    assert display_words(f"{HIZB} {WORD_A} {WORD_B}") == [f"{HIZB} {WORD_A}", WORD_B]
    assert display_words("") == []
    assert display_words(PAUSE) == [PAUSE]
    uthmani = CORPUS.ayah(19, 2).uthmani
    assert display_words(uthmani) == uthmani.split()


def _segments(words: int, lead_ms: int, word_ms: int = 300, gap_ms: int = 80) -> list[list[int]]:
    out = []
    for k in range(words):
        start = lead_ms + k * (word_ms + gap_ms)
        out.append([k + 1, start, start + word_ms])
    return out


def _ref(surah: int, start: int, end: int) -> VerseRef:
    return VerseRef(
        surah=surah,
        ayah_start=start,
        ayah_end=end,
        certain=True,
        uthmani_text=CORPUS.text_range(surah, start, end),
    )


def test_verse_clip_segments_shifted_by_trim(ffmpeg_bin: str, tmp_path: Path) -> None:
    words = display_words(CORPUS.ayah(19, 2).uthmani)
    audio = to_wav_bytes(speechlike(len(words), lead_ms=600))
    seen: list[str] = []
    handler = api_server({"19:2": audio}, {"19:2": _segments(len(words), 600)}, seen)
    clip, warnings = verse_clip(_ref(19, 2, 2), CORPUS, source(qurancom(), tmp_path, handler))
    assert warnings == []
    assert clip.timing_source is TimingSource.SEGMENTS
    [ayah] = clip.ayat
    assert (ayah.surah, ayah.ayah, ayah.start_ms) == (19, 2, 0)
    assert ayah.end_ms == len(clip.samples) * 1000 // aio.SAMPLE_RATE
    assert [w.text for w in ayah.words] == words
    assert ayah.words[0].start_ms <= 20
    assert ayah.words[1].start_ms - ayah.words[0].start_ms == 380
    assert all(w.end_ms <= ayah.end_ms for w in ayah.words)
    assert len(clip.samples) < len(aio.decode(audio))


def test_segment_count_mismatch_falls_back_to_whole_ayah_with_warning(
    ffmpeg_bin: str, tmp_path: Path
) -> None:
    audio = to_wav_bytes(speechlike(3))
    seen: list[str] = []
    handler = api_server({"19:2": audio}, {"19:2": _segments(3, 0)}, seen)
    clip, warnings = verse_clip(_ref(19, 2, 2), CORPUS, source(qurancom(), tmp_path, handler))
    assert warnings == ["19:2: no word timings; whole ayah revealed"]
    assert clip.timing_source is TimingSource.AYAH
    assert clip.ayat[0].words == []


def test_files_reciter_clip_is_ayah_timed(ffmpeg_bin: str, tmp_path: Path) -> None:
    audio = to_wav_bytes(speechlike(3))
    src = source(files_reciter(), tmp_path, lambda request: httpx.Response(200, content=audio))
    clip, warnings = verse_clip(_ref(19, 2, 2), CORPUS, src)
    assert clip.timing_source is TimingSource.AYAH
    assert warnings == ["19:2: no word timings; whole ayah revealed"]


def test_multi_ayah_joined_with_gap(ffmpeg_bin: str, tmp_path: Path) -> None:
    n2 = len(display_words(CORPUS.ayah(19, 2).uthmani))
    n3 = len(display_words(CORPUS.ayah(19, 3).uthmani))
    a2 = to_wav_bytes(speechlike(n2, seed=1))
    a3 = to_wav_bytes(speechlike(n3, seed=2))
    seen: list[str] = []
    handler = api_server(
        {"19:2": a2, "19:3": a3},
        {"19:2": _segments(n2, 0), "19:3": [[1, 0, 300]]},
        seen,
    )
    clip, warnings = verse_clip(_ref(19, 2, 3), CORPUS, source(qurancom(), tmp_path, handler))
    first, second = clip.ayat
    assert warnings == ["19:3: no word timings; whole ayah revealed"]
    assert clip.timing_source is TimingSource.AYAH
    assert len(first.words) == n2  # words are kept where available
    assert second.words == []
    assert second.start_ms - first.end_ms == AYAH_GAP_MS
    assert second.end_ms == len(clip.samples) * 1000 // aio.SAMPLE_RATE
    gap = clip.samples[aio.ms_to_samples(first.end_ms + 5) : aio.ms_to_samples(second.start_ms - 5)]
    assert gap.size > 0
    assert float(np.max(np.abs(gap))) == 0.0
    assert clip.samples.dtype == np.float32


def test_missing_ayah_in_corpus_fails_before_download(tmp_path: Path) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    ref = VerseRef(surah=19, ayah_start=90, ayah_end=90, certain=True, uthmani_text="x")
    with pytest.raises(ReciterError, match="19:90"):
        verse_clip(ref, CORPUS, source(qurancom(), tmp_path, refuse))
