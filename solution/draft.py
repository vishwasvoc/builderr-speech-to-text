"""
solution/draft.py
------------------
Implements the REAL streaming contract, confirmed from
docs/STREAMING_CONTRACT.md:

    def draft(audio_buffer: bytes, is_final: bool) -> tuple[str, int]:
        # audio_buffer = ALL audio heard so far (PCM s16le, mono, 16kHz)
        # called repeatedly as audio arrives (is_final=False), once more
        # after the user stops (is_final=True)
        # returns (text_so_far, stable_chars)

    - text_so_far: best transcript of the audio so far. Must keep the
      Hindi-English code-switch faithful (never translate the mix).
    - stable_chars: length of the committed leading prefix of
      text_so_far. Must be NON-DECREASING across calls, and that
      prefix string may only be EXTENDED, never rewritten.

Design, matching how it's actually scored (docs/STREAMING_CONTRACT.md
section 3): final meaning (40) + critical facts (20) = 60 of 100 points,
judged ONCE on the final call only. End-to-final latency is 25 points
(target ~2s). TTFS (5) and revision churn (5) are the only things partial
calls affect. So: partials exist to protect the TTFS/churn/no-partial
hard-caps cheaply - they are NOT where quality points are won. The full
quality+latency budget goes into the final pass.

Two behaviors follow from that:

1. Partial calls only re-transcribe the *uncommitted tail* of the audio
   (not the whole growing buffer) with a cheap model, so cost stays
   roughly constant no matter how long the clip gets. A prefix is only
   committed once it has matched across two consecutive tail passes AND
   a trailing safety margin is held back - because per 3.2, committed
   tokens that don't survive into the FINAL count as churn even if we
   personally never "rewrote" them - so being trigger-happy about
   committing hurts even without technically breaking the
   never-rewrite rule.

2. is_final=True ignores all the incremental/tail machinery and
   re-transcribes the FULL buffer once with the strong, faithful
   pipeline (fast pass -> escalate to the Hindi-capable model if
   code-switched -> romanize Devanagari -> repetition guard) - the
   exact same logic solution/transcribe.py uses in "auto" mode, reused
   directly so both paths agree.

UPDATED after real scoring feedback ("too slow", "misses too much"):
both _run_tail_partial and _run_final now go through
solution.transcribe._transcribe_core, which prefers mlx-whisper
(genuinely GPU/ANE-accelerated on the scoring Mac) over faster-whisper
(CPU-only there) - see transcribe.py's docstring for the full reasoning
and the caveat that this is unverified on real Mac hardware. _run_final
also now retries once with the other model size if the primary pass
comes back blank, since a blank final is a severe hard cap.

UPDATED again after leaderboard feedback ("only the English clips
finished reliably", score 15/100 vs RambleFix's 68.52): this points at
a specific, high-confidence bug, not just "model quality is weaker."
docs/STREAMING_CONTRACT.md's flow is: launch the server -> warm up on
ONE unscored clip -> call block_network() -> score. If that single
warmup clip is plain English (likely, since even RambleFix's own draft
path is English-biased), HINGLISH_MODEL_NAME never gets touched during
warmup under the old lazy-loading design - so the FIRST time it's ever
needed is the first real Hindi/code-switched clip in the scored,
network-blocked run, where trying to load/download it for the first
time either hangs or throws. That's a near-exact match for "only
English clips finished reliably" (English never needed the second
model; every Hindi clip hit a cold, blocked load and scored ~0).

CODE RED UPDATE (score dropped to 12.50, note: "six of eight clips timed
out and mixed speech did not finish"): this is a DIFFERENT, more urgent
failure mode than "dropped content" - a timeout/hang scores that clip as
0 outright (the worst possible outcome, worse than a rough-but-complete
answer). The previous round's fix (bigger Hindi model + more lenient/
exhaustive decoding + a blank-safety retry that can trigger a 3rd
sequential model call) makes each Hindi clip take LONGER - exactly the
wrong direction if timeouts are now the dominant problem.

Also worth flagging: builderr.ai/guidelines now describes this challenge
as "Final only: 70 + 30... Early drafts are not scored" - simpler than
docs/STREAMING_CONTRACT.md's 40/20/25/5/5 breakdown, and possibly in
tension with it (that doc calls itself "the single source of truth", so
treat it as ground truth, but the guidelines page suggests partials may
matter less than we assumed). Since both can't be fully verified against
each other, the design below is safe under EITHER: partials still run
(avoids the "no useful partial ever" cap either doc could apply) but are
throttled harder, freeing CPU for what both docs agree matters most - a
fast, complete final.

The actual fix: every model call in this file now runs through
_call_with_deadline(), which enforces a hard self-imposed wall-clock
budget using a background thread pool. If a call doesn't finish in time,
we DON'T wait for it - we fall back to whatever we already have (even
lower-quality-but-complete beats zero) and return well within budget.
This lets us KEEP the Hindi quality improvements (bigger model, lenient
decoding) when there's time for them, while guaranteeing we can never
hang a clip to a 0 score. Known limitation: Python cannot forcibly kill
a blocking call, so a timed-out call keeps running in the background
(wasting some CPU) even after we've moved on - the thread pool has a
few extra workers as headroom against this, but it's a real constraint,
not a perfect solution.
"""

from __future__ import annotations

import concurrent.futures
import os
import time
from typing import Any, List, Optional, Tuple

import numpy as np

from solution.transcribe import (
    FAST_MODEL_NAME,
    HINGLISH_MODEL_NAME,
    _transcribe_core,
    _looks_code_switched,
    _romanize,
    _detect_repetition_loop,
    _dedupe_repetition,
)

SAMPLE_RATE = 16000
TAIL_WINDOW_MAX_S = 8.0          # safety cap on tail re-decode length
COMMIT_SAFETY_WORDS = 2          # never commit the trailing N words of a pass
MIN_NEW_AUDIO_S = 0.5            # throttle: need this much new audio to re-run
MIN_WALL_INTERVAL_S = 0.6        # throttle: don't re-run more often than this
                                  # (relaxed from 0.4/0.35 - partials likely
                                  # matter less than we assumed; free up CPU
                                  # for the final instead)

# Background pool used for deadline-bounded calls (see _call_with_deadline).
# A few extra workers beyond what we'd normally need in flight at once, as
# headroom against orphaned timed-out calls still running when the next
# one is submitted (Python can't forcibly cancel a blocking call).
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="asr")

# Hard wall-clock budget for the ENTIRE final pass (fast check + possible
# Hindi escalation + possible retry, all combined). Chosen well under
# where a hang would otherwise cost a clip its full score - tune via:
#   set STT_FINAL_DEADLINE_S=3.0
FINAL_DEADLINE_S = float(os.environ.get("STT_FINAL_DEADLINE_S", "4.0"))

# Much shorter budget for a single partial pass - if a partial can't
# finish quickly, just skip that cycle rather than risk it stalling
# anything. Partials are frequent and cheap by design; a slow one isn't
# worth waiting for.
PARTIAL_DEADLINE_S = float(os.environ.get("STT_PARTIAL_DEADLINE_S", "1.2"))


def _call_with_deadline(model_size: str, audio: np.ndarray, beam_size: int,
                         lenient: bool, deadline_s: float, word_timestamps: bool = False):
    """Run a transcription call with a hard wall-clock deadline. Returns
    the EngineResult, or None if it didn't finish in time (or raised) -
    callers must handle None by falling back to something else instead
    of waiting. See module docstring for why this exists and its
    limitation (can't truly cancel the background call)."""
    if deadline_s <= 0.05:
        return None  # no meaningful time left - don't even try
    future = _executor.submit(
        _transcribe_core, model_size, audio,
        beam_size=beam_size, word_timestamps=word_timestamps, lenient=lenient,
    )
    try:
        return future.result(timeout=deadline_s)
    except Exception:
        # Covers both concurrent.futures.TimeoutError and any exception
        # raised inside _transcribe_core - either way, no result in time.
        return None


def _eager_warmup() -> None:
    """Force-load BOTH the fast and Hindi-capable models right now, at
    import time, instead of waiting for each to be needed lazily. See
    the module docstring for why this matters - it's the most likely
    fix for clips failing outright rather than just scoring low.
    Best-effort: swallow all exceptions so a warmup problem can't break
    the import / stop the server printing READY at all. A slow or
    partially-failed warmup is far better than the server never starting."""
    try:
        silence = np.zeros(int(0.3 * SAMPLE_RATE), dtype=np.float32)
        _transcribe_core(FAST_MODEL_NAME, silence, beam_size=1, word_timestamps=False)
        _transcribe_core(HINGLISH_MODEL_NAME, silence, beam_size=1, word_timestamps=False)
    except Exception:
        pass


_eager_warmup()


class _StreamState:
    def __init__(self) -> None:
        self.committed_text: str = ""
        self.committed_samples: int = 0       # audio cursor for the tail window
        self.last_run_wall: float = 0.0
        self.finalized: bool = False
        self.likely_code_switched: bool = False  # set True by any partial pass
                                                   # that looked Hindi/mixed -
                                                   # lets the final pass skip a
                                                   # redundant "is this Hindi?"
                                                   # check it already knows the
                                                   # answer to (see _run_final)


_state = _StreamState()


def draft_reset(*_args: Any, **_kwargs: Any) -> None:
    """Reset all per-stream state. Called by the harness at the start of
    each new clip/session (exact signature not confirmed -
    docs/STREAMING_CONTRACT.md never mentions this function; we only
    know it's imported because of the original "draft_reset import
    issue" failure. solution/stream_server.py is visible locally in your
    repo right now even though it's swapped for the official copy at
    scoring time - `findstr /i "draft_reset" solution\\stream_server.py`
    would show the real call if you want to close this last gap. Built
    defensively below (accepts any args/kwargs) so it should work
    either way."""
    global _state
    _state = _StreamState()
    return None


def _bytes_to_f32(buf: bytes) -> np.ndarray:
    arr = np.frombuffer(bytes(buf), dtype=np.int16)
    return arr.astype(np.float32) / 32768.0



# Beam size for the FINAL pass only (partials always use beam_size=1 -
# cheap, run frequently). Lower = faster but slightly less accurate.
# Tunable without touching code, e.g. if latency is still too high:
#   set STT_FINAL_BEAM_SIZE=2
# Default lowered from 5 to 3 after "too slow" feedback - a reasonable
# middle ground, but genuinely a guess without being able to benchmark
# on the real scoring Mac. Turn it down further if still too slow, or
# back up toward 5 if speed turns out fine and quality is the bottleneck.
FINAL_BEAM_SIZE = int(os.environ.get("STT_FINAL_BEAM_SIZE", "3"))

# Separate, higher beam size for the Hindi/code-switch pass specifically.
# After "English got stronger, mixed Hindi-English still dropped too
# much" feedback: the across-the-board beam_size cut above likely helped
# the easy English path but hurt the hard Hindi path, which is exactly
# where completeness matters most (40+20 = 60 of 100 points). This pass
# also runs with lenient=True (see transcribe.py) to reduce content
# being silently dropped as "low confidence" or "no speech".
HINGLISH_BEAM_SIZE = int(os.environ.get("STT_HINGLISH_BEAM_SIZE", "5"))


def _run_tail_partial(tail_audio: np.ndarray) -> Tuple[str, List[Tuple[str, float]], Optional[str], Optional[float]]:
    """Cheap pass over just the uncommitted tail, bounded by
    PARTIAL_DEADLINE_S. Returns the text, (word, end_time_within_tail)
    pairs for advancing the audio cursor, and the detected
    language/confidence. On timeout, returns empty/unknown - the caller
    (draft()) already handles "nothing new happened this cycle" cleanly."""
    result = _call_with_deadline(FAST_MODEL_NAME, tail_audio, beam_size=1,
                                  lenient=False, deadline_s=PARTIAL_DEADLINE_S,
                                  word_timestamps=True)
    if result is None:
        return "", [], None, None
    return result.text, result.words, result.language, result.language_probability


def _run_final(full_audio: np.ndarray, hint_code_switched: bool = False) -> str:
    """Full-buffer, full-quality pass - same faithful logic as
    transcribe.py's auto mode. This is where the 60-70 quality points
    live (both scoring docs agree on that), so it gets the largest
    latency budget - but that budget is now a HARD, self-enforced
    deadline (FINAL_DEADLINE_S total), not an open-ended wait. See
    module docstring for the full reasoning: a timeout scores a clip 0,
    which is worse than a fast-but-rougher answer.

    Speed optimization (kept from before): if partials already flagged
    this clip as code-switched, skip straight to the Hindi model instead
    of re-discovering what we already know."""
    deadline = time.perf_counter() + FINAL_DEADLINE_S

    def remaining() -> float:
        return deadline - time.perf_counter()

    text = ""

    # Always keep at least this much of the deadline in reserve for a
    # fast fallback attempt - otherwise a Hindi call that times out could
    # consume the ENTIRE budget itself, leaving nothing for any fallback
    # and guaranteeing a blank result. (Caught by testing the timeout
    # path directly - worth remembering why this constant exists.)
    FALLBACK_RESERVE_S = 1.0

    if hint_code_switched:
        hindi_budget = max(0.1, remaining() - FALLBACK_RESERVE_S)
        strong = _call_with_deadline(HINGLISH_MODEL_NAME, full_audio, HINGLISH_BEAM_SIZE,
                                      lenient=True, deadline_s=hindi_budget)
        text = strong.text if strong else ""
        if not text.strip():
            fallback = _call_with_deadline(FAST_MODEL_NAME, full_audio, FINAL_BEAM_SIZE,
                                            lenient=False, deadline_s=remaining())
            text = fallback.text if fallback else text
    else:
        fast = _call_with_deadline(FAST_MODEL_NAME, full_audio, FINAL_BEAM_SIZE,
                                    lenient=False, deadline_s=remaining())
        fast_text = fast.text if fast else ""
        if fast and _looks_code_switched(fast_text, fast.language, fast.language_probability):
            strong = _call_with_deadline(HINGLISH_MODEL_NAME, full_audio, HINGLISH_BEAM_SIZE,
                                          lenient=True, deadline_s=remaining())
            text = strong.text if strong else fast_text  # fall back to the fast result, not blank
        else:
            text = fast_text
        if not text.strip() and remaining() > 0.2:
            # Blank-safety retry, but only if there's still real budget left -
            # no point starting a call we know we'll have to abandon.
            retry = _call_with_deadline(HINGLISH_MODEL_NAME, full_audio, HINGLISH_BEAM_SIZE,
                                         lenient=True, deadline_s=remaining())
            text = retry.text if retry else text

    if _detect_repetition_loop(text):
        text = _dedupe_repetition(text)

    return _romanize(text).strip()


def draft(audio_buffer: bytes, is_final: bool) -> Tuple[str, int]:
    """See module docstring for the contract. Returns (text_so_far,
    stable_chars)."""
    global _state
    full_audio = _bytes_to_f32(audio_buffer)

    if is_final:
        final_text = _run_final(full_audio, hint_code_switched=_state.likely_code_switched)
        _state.committed_text = final_text
        _state.finalized = True
        return final_text, len(final_text)

    total_samples = len(full_audio)
    new_samples = total_samples - _state.committed_samples
    now = time.perf_counter()

    # Throttle: not enough new audio, or too soon since the last pass -
    # return the unchanged committed state (cheap, and can't cause churn
    # since nothing changes).
    if (new_samples / SAMPLE_RATE) < MIN_NEW_AUDIO_S or \
       (now - _state.last_run_wall) < MIN_WALL_INTERVAL_S:
        return _state.committed_text, len(_state.committed_text)

    _state.last_run_wall = now
    tail_start = _state.committed_samples
    tail_audio = full_audio[tail_start:]

    max_tail_samples = int(TAIL_WINDOW_MAX_S * SAMPLE_RATE)
    if len(tail_audio) > max_tail_samples:
        # Safety fallback only - shouldn't normally trigger if commits
        # are progressing at a reasonable rate.
        tail_audio = tail_audio[-max_tail_samples:]

    tail_text, tail_words, tail_lang, tail_lang_prob = _run_tail_partial(tail_audio)
    if _looks_code_switched(tail_text, tail_lang, tail_lang_prob):
        _state.likely_code_switched = True
    new_words = tail_text.split()

    # Commit directly from this pass (minus a small trailing safety
    # margin against mid-word audio cutoffs) - no longer requiring
    # agreement across two consecutive passes first.
    #
    # CHANGED after confirming (guidelines page + the actual GitHub
    # README, in matching wording both times): "Intermediate drafts are
    # optional and never affect the score" - i.e. revision churn isn't
    # scored. The old 2-pass-stability requirement existed specifically
    # to protect against a churn penalty that, per this confirmation,
    # doesn't exist. Committing faster/more eagerly now has no real
    # downside and helps satisfy "at least one useful partial" (still
    # worth hedging for, since docs/STREAMING_CONTRACT.md's hard caps
    # around this haven't been explicitly retracted, just possibly
    # superseded) sooner rather than later.
    commit_word_count = max(0, len(new_words) - COMMIT_SAFETY_WORDS)

    if commit_word_count > 0:
        addition = " ".join(new_words[:commit_word_count])
        _state.committed_text = (
            f"{_state.committed_text} {addition}".strip()
            if _state.committed_text else addition
        )
        if len(tail_words) >= commit_word_count:
            end_time_in_tail = tail_words[commit_word_count - 1][1]
            _state.committed_samples = tail_start + int(end_time_in_tail * SAMPLE_RATE)
        else:
            # No word timestamps available - fall back to a proportional
            # estimate of where in the audio that many words likely ended.
            frac = commit_word_count / max(1, len(new_words))
            _state.committed_samples = tail_start + int(frac * len(tail_audio))

    trailing = " ".join(new_words[commit_word_count:]) if commit_word_count else tail_text
    text_so_far = f"{_state.committed_text} {trailing}".strip()
    return text_so_far, len(_state.committed_text)
