# ADR 0003: TTS strategy is a voice chain with edge as the default free fallback

**Status:** Accepted, 2026-11-13

## Context

`docs/REVIEW.md` (finding C2) found that `edge-tts` is an open-source client of an undocumented
Microsoft endpoint that can be rate-limited or changed, and whose terms for commercial reuse are
unclear. It recommended keeping edge as the default engine behind an interface, making a local
engine (Piper) a first-class fallback, caching all synthesized audio, and documenting the risk.
Captions also need per-word timings, which not every engine provides.

## Decision

- **Chain.** `build_voice_chain` (`src/narratoflow/tts/factory.py`) returns the channel's primary
  voice followed by each `voice.fallback` entry (`VoiceConfig.fallback`, default `["edge"]`,
  values `edge` or `piper`, in `src/narratoflow/channels/config.py`), skipping duplicates. It
  validates everything up front, before any spend.
- **Retry inside a voice.** Edge and ElevenLabs are wrapped in `RetryingSpeechProvider`
  (`src/narratoflow/tts/retry.py`): `TransientProviderError` only, `RETRY_ATTEMPTS = 3`, delays of
  1 s and 4 s. Piper is not wrapped.
- **Budget.** Paid calls go through `TrackedSpeechProvider` (`tts/factory.py`), which checks the
  budget before each call and charges after success.
- **Fallback.** `narrate_part` (`src/narratoflow/audio/narrate.py`) voices every narration scene
  of a part with one voice. If that voice raises `ProviderError`, `ExtraMissingError` or an
  undecodable or silent clip, the whole part is re-voiced with the next voice; the reasons are
  recorded in `VoiceUsed.fallback_from` and `fallback_reasons` (`src/narratoflow/audio/timeline.py`).
- **Cache.** The TTS cache key (`_key` in `narrate.py`, hashed by `cache_key` in
  `src/narratoflow/pipeline/cache.py`) covers provider, model, voice id, spoken text and, for a
  local model, the model file path, size and mtime.
- **Word timings.** `_timings` in `narrate.py` uses the provider's spans when they map onto the
  spoken words (`tokens(spoken.text)`; `TimingSource.PROVIDER`), else the CTC aligner `CtcAligner`
  (`src/narratoflow/align/ctc.py`, extra `align`; `ALIGNED`), else a length-weighted spread over
  the clip's non-silent span (`estimate_timings` in `audio/words.py`; `ESTIMATED`, with a
  warning). Piper reports no spans (`src/narratoflow/tts/piper.py`).
- **Terms.** Each part's `VoiceUsed.terms` is a terms id from `voice_terms` (`tts/factory.py`):
  `elevenlabs-terms`, `edge-tts-terms`, `piper-model-license`. These feed the publish rules in
  `src/narratoflow/data/platform_rules.toml`, where all three are `warn`.

## Consequences

- Edge has no explicit commercial licence. The `edge-tts-terms` rule warns on it, and
  `piper-model-license` warns that the verdict depends on the licence of the chosen voice model.
  The code records which terms apply but does not record a Piper model's licence; that is checked
  by the operator.
- The size of the edge-versus-CTC word-start offset on Arabic is not settled. Measured
  disagreement is about 115 ms in `docs/spikes/2026-11-12-arabic-word-timing.md`; attributing it to
  one side is step 1 of #133, with tooling and decoder-priming findings in
  `docs/spikes/2026-11-13-word-onset-ground-truth.md`. Until then provider spans are used as given.
- A failed voice re-voices the whole part, so a part never mixes voices, at the cost of re-spending
  the scenes already voiced (cached scenes are free).
- Optional engines stay behind extras (`tts-local`, `align`), so the base install imports neither.

## Alternatives considered

- **ElevenLabs as default:** better quality, but paid and keyed. It is supported as a primary
  voice instead.
- **XTTS-v2 and F5-TTS:** non-commercial weights (REVIEW C3); not implemented.
- **Per-scene fallback:** would mix voices inside one video; rejected in favour of per-part.
- **Always align with CTC:** slower and needs an extra; provider spans are used when present.
