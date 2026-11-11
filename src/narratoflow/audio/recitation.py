"""Per-ayah recitation download (cached) and verse clips timed to Uthmani words.

quran.com API v4, checked 2026-11-11:
- ``GET https://api.quran.com/api/v4/recitations/{recitation_id}/by_ayah/{s}:{a}``
  returns ``audio_files``, a list of objects with ``verse_key`` and ``url``
  (https://api-docs.quran.foundation/docs/content_apis_versioned/4.0.0/list-ayah-recitation/).
  The documented examples show ``url`` both as a relative path such as
  ``AbdulBaset/Mujawwad/mp3/001001.mp3`` (served from https://verses.quran.com/)
  and as a full https URL; both are handled.
- Word segments are ``[word_index, start_ms, end_ms]`` with 1-based word indexes
  and all times in milliseconds
  (https://api-docs.quran.foundation/docs/content_apis_versioned/4.0.0/chapter-reciter-audio-file/).
  The by_ayah page does not document ``segments``; any other layout is dropped
  with a warning and the ayah is revealed whole.
- The Quran Foundation now documents the v4 Content API behind OAuth2 client
  credentials (``x-auth-token`` and ``x-client-id``) and calls
  ``api.quran.com/api/v4`` the "older unauthenticated API"
  (https://api-docs.quran.foundation/docs/quickstart/migration/). Plan 4 adds no
  credentials, and the shipped allow-list has no ``qurancom`` reciter.

Downloads are cached as ``<cache_dir>/<reciter id>/<SSSAAA>.mp3`` plus
``<SSSAAA>.segments.json`` (the same ``[word, start_ms, end_ms]`` triples) when
segments exist. Both are written atomically and the audio file is written last,
so it marks a complete entry. A cached file that is not audio, or cached
segments that fail validation, make the entry a miss: it is deleted and fetched
again. Redirects are followed only to https URLs, and bodies are capped
(`MAX_AUDIO_BYTES` per recording, `MAX_JSON_BYTES` per API response).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import numpy as np
from pydantic import BaseModel, Field, model_validator

from narratoflow.audio import io as aio
from narratoflow.audio.timeline import AyahTiming, TimelineWord, TimingSource
from narratoflow.errors import AudioError, ProviderError, ReciterError, VerseCorpusError
from narratoflow.models import VerseRef
from narratoflow.quran.corpus import QuranCorpus
from narratoflow.quran.reciters import Reciter

logger = logging.getLogger(__name__)

QURANCOM_API = "https://api.quran.com/api/v4"
QURANCOM_AUDIO_BASE = "https://verses.quran.com/"
TIMEOUT_S = 60.0
AYAH_GAP_MS = 300
MAX_SEGMENT_MS = 3_600_000
MAX_AUDIO_BYTES = 20 * 1024 * 1024  # one ayah's recording
MAX_JSON_BYTES = 1024 * 1024  # an API response or cached segments
MAX_REDIRECTS = 5
MAX_SURAH = 114
MAX_AYAH = 286  # the longest surah (al-Baqarah)
_FILE_MODE = 0o644

# Qur'anic annotation marks that Tanzil writes as their own whitespace token.
_MARKS = frozenset(chr(c) for c in range(0x06D6, 0x06EE))


class AyahSegment(BaseModel):
    word: int = Field(ge=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> AyahSegment:
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms is before start_ms")
        return self


@dataclass(frozen=True)
class AyahAudio:
    surah: int
    ayah: int
    audio: bytes
    segments: list[AyahSegment] | None


@dataclass
class VerseClip:
    samples: aio.Samples
    ayat: list[AyahTiming]  # times relative to the clip start
    timing_source: TimingSource


def display_words(uthmani: str) -> list[str]:
    """Whitespace words with each mark-only token joined to its neighbour word."""
    words: list[str] = []
    leading: list[str] = []
    for token in uthmani.split():
        if all(ch in _MARKS for ch in token):
            if words:
                words[-1] = f"{words[-1]} {token}"
            else:
                leading.append(token)
            continue
        words.append(" ".join([*leading, token]))
        leading = []
    if leading:
        words.append(" ".join(leading))
    return words


def _key(surah: int, ayah: int) -> str:
    return f"{surah:03d}{ayah:03d}"


def _is_audio(data: bytes) -> bool:
    if data[:3] == b"ID3" or data[:4] in (b"RIFF", b"OggS", b"fLaC") or data[4:8] == b"ftyp":
        return True
    return len(data) >= 2 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0


def _unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        raise AudioError(f"cannot remove recitation cache file {path}: {exc}") from exc


def _read_capped(response: httpx.Response, limit: int, host: str, surah: int, ayah: int) -> bytes:
    declared = response.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise ProviderError(f"{host} response for {surah}:{ayah} exceeds {limit} bytes")
    chunks: list[bytes] = []
    size = 0
    for chunk in response.iter_bytes():
        size += len(chunk)
        if size > limit:
            raise ProviderError(f"{host} response for {surah}:{ayah} exceeds {limit} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


def _write_atomic(path: Path, data: bytes) -> None:
    tmp_name: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as tmp:
            tmp_name = tmp.name
            tmp.write(data)
            tmp.flush()
            os.fchmod(tmp.fileno(), _FILE_MODE)
            os.fsync(tmp.fileno())
        os.replace(tmp_name, path)
    except OSError as exc:
        if tmp_name is not None:
            Path(tmp_name).unlink(missing_ok=True)
        raise AudioError(f"cannot write recitation cache {path}: {exc}") from exc


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _parse_segments(raw: object, where: str) -> list[AyahSegment] | None:
    """Validated segments, or None (with a warning when they were present but bad)."""
    if raw is None or raw == []:
        return None
    try:
        return _checked_segments(raw)
    except ValueError as exc:
        logger.warning("%s: dropping word segments (%s)", where, exc)
        return None


def _checked_segments(raw: object) -> list[AyahSegment]:
    if not isinstance(raw, list):
        raise ValueError("not a list")
    out: list[AyahSegment] = []
    for item in raw:
        if not (isinstance(item, list) and len(item) == 3 and all(_is_int(v) for v in item)):
            raise ValueError("not [word, start_ms, end_ms] integer triples")
        word, start, end = item
        if word < 1 or not 0 <= start <= end <= MAX_SEGMENT_MS:
            raise ValueError("word index or times out of range")
        if out and (word <= out[-1].word or start < out[-1].start_ms):
            raise ValueError("not monotonic")
        out.append(AyahSegment(word=word, start_ms=start, end_ms=end))
    return out


class RecitationSource:
    """Fetches one reciter's per-ayah recordings, caching them under ``cache_dir``.

    Use it as a context manager (or call `close`) to close the HTTP client it
    created; a client passed in by the caller is left open.
    """

    def __init__(self, reciter: Reciter, cache_dir: Path, http: httpx.Client | None = None) -> None:
        self.reciter = reciter
        self.cache_dir = cache_dir / reciter.id
        self._owns_http = http is None
        self.http = http if http is not None else httpx.Client(timeout=TIMEOUT_S)

    def __enter__(self) -> RecitationSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_http:
            self.http.close()

    def fetch(self, surah: int, ayah: int) -> AyahAudio:
        if not (1 <= surah <= MAX_SURAH and 1 <= ayah <= MAX_AYAH):
            raise ReciterError(f"no such ayah {surah}:{ayah}")
        cached = self._cached(surah, ayah)
        if cached is not None:
            return cached
        if self.reciter.kind == "qurancom":
            url, segments = self._qurancom_lookup(surah, ayah)
        else:
            url, segments = self.reciter.file_url(surah, ayah), None
        audio = self._get(url, surah, ayah, MAX_AUDIO_BYTES)
        if not _is_audio(audio):
            raise ProviderError(
                f"{self.reciter.name}: recording of {surah}:{ayah} is not audio "
                f"({len(audio)} bytes)"
            )
        key = _key(surah, ayah)
        seg_path = self.cache_dir / f"{key}.segments.json"
        if segments is not None:
            payload = json.dumps([[s.word, s.start_ms, s.end_ms] for s in segments]).encode()
            _write_atomic(seg_path, payload)
        else:
            _unlink(seg_path)  # never pair new audio with stale timings
        _write_atomic(self.cache_dir / f"{key}.mp3", audio)
        return AyahAudio(surah, ayah, audio, segments)

    def _cached(self, surah: int, ayah: int) -> AyahAudio | None:
        """The cached entry, or None (a miss). A damaged entry is deleted and refetched."""
        key = _key(surah, ayah)
        audio_path = self.cache_dir / f"{key}.mp3"
        seg_path = self.cache_dir / f"{key}.segments.json"
        if not audio_path.is_file():
            return None
        where = f"{surah}:{ayah}"
        try:
            audio = audio_path.read_bytes()
        except OSError as exc:
            raise AudioError(f"cannot read recitation cache {audio_path}: {exc}") from exc
        if not _is_audio(audio):
            logger.warning("%s: cached recording is not audio; fetching it again", where)
            _unlink(audio_path)
            return None
        if not seg_path.is_file():
            return AyahAudio(surah, ayah, audio, None)
        try:
            text = seg_path.read_text(encoding="utf-8")
            segments = _checked_segments(json.loads(text)) if len(text) <= MAX_JSON_BYTES else None
        except (OSError, ValueError, RecursionError):
            segments = None
        if not segments:
            # Damaged timings: drop the whole entry so audio and timings come back together.
            logger.warning("%s: cached word segments are invalid; fetching the ayah again", where)
            _unlink(audio_path)
            _unlink(seg_path)
            return None
        return AyahAudio(surah, ayah, audio, segments)

    def _get(self, url: str, surah: int, ayah: int, limit: int) -> bytes:
        """The body of ``url``, following only https redirects, read up to ``limit`` bytes."""
        request = self.http.build_request("GET", url, timeout=TIMEOUT_S)
        for _ in range(MAX_REDIRECTS + 1):
            host = request.url.host
            if request.url.scheme != "https":
                raise ProviderError(f"refusing a non-https URL on {host} for {surah}:{ayah}")
            try:
                response = self.http.send(request, stream=True, follow_redirects=False)
                try:
                    if response.is_redirect and response.next_request is not None:
                        request = response.next_request
                        continue
                    self._check_status(response, host, surah, ayah)
                    return _read_capped(response, limit, host, surah, ayah)
                finally:
                    response.close()
            except httpx.TimeoutException as exc:
                raise ProviderError(f"{host} request timed out ({type(exc).__name__})") from exc
            except httpx.HTTPError as exc:
                raise ProviderError(f"cannot reach {host} ({type(exc).__name__})") from exc
        raise ProviderError(f"too many redirects for {surah}:{ayah}")

    def _check_status(self, response: httpx.Response, host: str, surah: int, ayah: int) -> None:
        if response.status_code == 404:
            raise ReciterError(f"{self.reciter.name} has no recording of {surah}:{ayah}")
        if not response.is_success:
            raise ProviderError(f"{host} returned HTTP {response.status_code} for {surah}:{ayah}")

    def _qurancom_lookup(self, surah: int, ayah: int) -> tuple[str, list[AyahSegment] | None]:
        rid = self.reciter.recitation_id
        url = f"{QURANCOM_API}/recitations/{rid}/by_ayah/{surah}:{ayah}"
        raw = self._get(url, surah, ayah, MAX_JSON_BYTES)
        try:
            body: Any = json.loads(raw)
        except (ValueError, RecursionError) as exc:
            raise ProviderError("quran.com response is not valid JSON") from exc
        files = body.get("audio_files") if isinstance(body, dict) else None
        if not isinstance(files, list) or not files or not isinstance(files[0], dict):
            raise ProviderError(f"quran.com response has no audio_files for {surah}:{ayah}")
        first = files[0]
        audio_url = first.get("url")
        if not isinstance(audio_url, str) or not audio_url.strip():
            raise ProviderError(f"quran.com response has no audio file url for {surah}:{ayah}")
        return _absolute(audio_url.strip()), _parse_segments(
            first.get("segments"), f"{surah}:{ayah}"
        )


def _absolute(url: str) -> str:
    if url.startswith("//"):
        url = f"https:{url}"
    parts = urlsplit(url)
    if not parts.scheme:
        return QURANCOM_AUDIO_BASE + url.lstrip("/")
    if parts.scheme != "https" or not parts.netloc:
        raise ProviderError("quran.com audio file url is not https")
    return url


def _ayah_words(
    segments: list[AyahSegment] | None, words: list[str], offset_ms: int, lead_ms: int, end_ms: int
) -> list[TimelineWord]:
    if segments is None or [s.word for s in segments] != list(range(1, len(words) + 1)):
        return []

    def clamp(ms: int) -> int:
        return min(max(offset_ms + ms - lead_ms, offset_ms), end_ms)

    return [
        TimelineWord(text=words[s.word - 1], start_ms=clamp(s.start_ms), end_ms=clamp(s.end_ms))
        for s in segments
    ]


def verse_clip(
    ref: VerseRef, corpus: QuranCorpus, source: RecitationSource
) -> tuple[VerseClip, list[str]]:
    """The whole ayat of ``ref`` recited back to back, with Uthmani word timings."""
    surah = ref.surah
    numbers = range(ref.ayah_start, ref.ayah_end + 1)
    texts: list[str] = []
    for n in numbers:
        try:
            texts.append(corpus.ayah(surah, n).uthmani)
        except VerseCorpusError as exc:
            raise ReciterError(f"ayah {surah}:{n} is not in the corpus") from exc
    gap = aio.silence(AYAH_GAP_MS)
    pieces: list[aio.Samples] = []
    ayat: list[AyahTiming] = []
    warnings: list[str] = []
    cursor = 0  # samples
    for n, text in zip(numbers, texts, strict=True):
        got = source.fetch(surah, n)
        trimmed, lead_ms = aio.trim_silence(aio.decode(got.audio))
        if trimmed.size == 0:
            raise ReciterError(f"{source.reciter.name}: recording of {surah}:{n} is silent")
        if pieces:
            pieces.append(gap)
            cursor += gap.size
        start_ms = cursor * 1000 // aio.SAMPLE_RATE
        cursor += trimmed.size
        end_ms = cursor * 1000 // aio.SAMPLE_RATE
        pieces.append(trimmed)
        words = _ayah_words(got.segments, display_words(text), start_ms, lead_ms, end_ms)
        if not words:
            warnings.append(f"{surah}:{n}: no word timings; whole ayah revealed")
        ayat.append(AyahTiming(surah=surah, ayah=n, start_ms=start_ms, end_ms=end_ms, words=words))
    source_kind = TimingSource.SEGMENTS if all(a.words for a in ayat) else TimingSource.AYAH
    samples = np.concatenate(pieces).astype(np.float32)
    return VerseClip(samples=samples, ayat=ayat, timing_source=source_kind), warnings
