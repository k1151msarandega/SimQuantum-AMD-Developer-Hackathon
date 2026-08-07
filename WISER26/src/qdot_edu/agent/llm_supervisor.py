"""LLM-supervised triage: a slow reasoning layer above triage.py's fast rule-based decisions.

PORTED from qdot-live-twin (Act II), src/qdot_twin/agent/llm_supervisor.py,
with only the import path updated (qdot_twin -> qdot_edu). The logic here
never depended on the GPU -- it calls an external LLM API, not
twin/batch_estimator.py -- so nothing else needed to change for the CPU
port. See docs/PORTING_NOTES.md.

WISER26 framing note: this is an explicit, judgment-call addition to an
otherwise laptop-reproducible submission -- it requires FIREWORKS_API_KEY
and a network call. The rest of the console (serial/batched/batched_triage,
the instrument panel, the potential-well panel) works fully offline with
no external dependency; this supervisor is opt-in on top of that, not a
requirement to use the console. See app.py for how the console surfaces
that tradeoff to the learner (a visible "no key set" state, not a failure).

Design constraint, non-negotiable given this pipeline's real-time framing:
stream() is a blocking, wall-clock-paced generator (see pipeline.py's own
docstring on the synchronous-approximation design). A network call to
Fireworks on every frame or every micro-batch flush would contradict the
pipeline's entire "keep up with a fast stream" premise. So this supervisor:

  1. Runs in a background thread, on its own slower cadence
     (SUPERVISOR_INTERVAL_S), never blocking stream()/estimate_batch().
  2. Reads a rolling window of recent (queue_depth, time_since_full,
     drift_activity, tier_chosen) tuples that the main pipeline loop
     appends to via RollingHistory.
  3. Asks the LLM, given that window, for adjusted cheap_queue_depth /
     skip_queue_depth / stale_threshold_s values.
  4. Writes the result into a shared TriageThresholds object (thresholds.py)
     that triage.decide() reads on its very next call.

This makes the LLM's output a real, consumed decision -- it changes which
tier subsequent frames get routed to -- without ever sitting in the hot
path. The fast, per-frame FULL/CHEAP/SKIP decision itself stays exactly as
rule-based as it was -- weighing queue depth, staleness, and drift
together in one call is what makes that decision real regardless of who
set the thresholds it uses.

Safety: a malformed or nonsensical LLM response must never be able to
break the stream. Every value is clamped into a hard-coded bound before
being written, an ordering invariant (skip > cheap) is enforced
regardless of what the model said, and any parse/API failure leaves the
previous thresholds untouched and logs the failure rather than raising.
"""
import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass

from qdot_edu.agent.thresholds import TriageThresholds

SUPERVISOR_INTERVAL_S = 1.0  # short enough that the quick-iteration config
# (configs/trajectory_quick.yaml, 300 frames) has a real chance to see at
# least one tick before the run ends.
HISTORY_MAXLEN = 300
MODEL = "accounts/fireworks/models/llama-v3p1-8b-instruct"
# A plain instruct model, not a reasoning model: under a tight token cap a
# reasoning model can spend its whole budget on hidden chain-of-thought and
# return empty content. A plain instruct model reliably returns the JSON
# directly. Verify this model is still on Fireworks' serverless tier
# (not on-demand/dedicated-only) before relying on it -- the catalog does
# change.

# Hard safety bounds. The LLM can tune within these; it can never push a
# value outside them, no matter what it returns.
CHEAP_QUEUE_DEPTH_BOUNDS = (2, 200)
SKIP_QUEUE_DEPTH_BOUNDS = (5, 500)
STALE_THRESHOLD_S_BOUNDS = (0.01, 2.0)

SYSTEM_PROMPT = """You tune three thresholds for a real-time triage agent that classifies incoming sensor frames as FULL, CHEAP, or SKIP based on backlog and staleness.

Thresholds:
- cheap_queue_depth: above this queue depth, switch from FULL to CHEAP tier (if not stale/drifting)
- skip_queue_depth: above this queue depth, switch to SKIP tier (shed load entirely)
- stale_threshold_s: if this long has passed since the last FULL update, force a FULL update regardless of backlog

Goal: keep worst-case staleness bounded while minimizing wasted FULL-tier compute. Respond with ONLY a JSON object, no markdown, no prose outside the JSON:
{"cheap_queue_depth": <int>, "skip_queue_depth": <int>, "stale_threshold_s": <float>, "reasoning": "<one short sentence>"}"""


@dataclass
class SupervisorEvent:
    t: float
    old: tuple
    new: tuple
    reasoning: str


class RollingHistory:
    """Thread-safe fixed-length window of recent triage-relevant signals."""

    def __init__(self, maxlen: int = HISTORY_MAXLEN):
        self._lock = threading.Lock()
        self._buf = deque(maxlen=maxlen)

    def record(self, queue_depth: int, time_since_full: float, drift_active: bool, tier_name: str) -> None:
        with self._lock:
            self._buf.append((queue_depth, time_since_full, drift_active, tier_name))

    def snapshot(self) -> list:
        with self._lock:
            return list(self._buf)


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def _summarize(window: list) -> str:
    depths = [w[0] for w in window]
    stales = [w[1] for w in window]
    drift_count = sum(1 for w in window if w[2])
    tier_counts: dict = {}
    for w in window:
        tier_counts[w[3]] = tier_counts.get(w[3], 0) + 1
    return (
        f"Recent window ({len(window)} decisions):\n"
        f"queue_depth: min={min(depths)} max={max(depths)} avg={sum(depths) / len(depths):.1f}\n"
        f"time_since_full_update_s: min={min(stales):.4f} max={max(stales):.4f}\n"
        f"drift_flags_active: {drift_count}/{len(window)}\n"
        f"tier_counts_this_window: {tier_counts}\n"
        f"If backlog is consistently near or above the current skip threshold, consider lowering "
        f"cheap_queue_depth to shed load earlier. If queue_depth never approaches current thresholds, "
        f"consider raising them slightly to save compute on FULL updates -- but do not overcorrect "
        f"in either direction."
    )


class LLMSupervisor:
    def __init__(self, thresholds: TriageThresholds, history: RollingHistory):
        self.thresholds = thresholds
        self.history = history
        self.events: list = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._client = None  # lazily constructed so importing this module never requires the key

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url="https://api.fireworks.ai/inference/v1",
                api_key=os.environ["FIREWORKS_API_KEY"],
            )
        return self._client

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=SUPERVISOR_INTERVAL_S + 2)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(SUPERVISOR_INTERVAL_S)
            if self._stop.is_set():
                return
            self._tick()

    def _tick(self) -> None:
        window = self.history.snapshot()
        if len(window) < 3:  # not enough signal yet to make a meaningful call
            return

        old = self.thresholds.snapshot()
        try:
            client = self._get_client()
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _summarize(window)},
                ],
                max_tokens=300,
                temperature=0.2,
            )
            # No reasoning_effort/Harmony handling needed -- plain instruct
            # models return the answer directly in .content, no hidden
            # chain-of-thought step that can silently consume the budget.
            content = resp.choices[0].message.content
            if not content:
                raise ValueError("empty response content (model likely exhausted max_tokens on reasoning)")
            text = content.strip()
            if text.startswith("```"):
                text = text.strip("`")
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)

            cheap = _clamp(int(data["cheap_queue_depth"]), *CHEAP_QUEUE_DEPTH_BOUNDS)
            skip = _clamp(int(data["skip_queue_depth"]), *SKIP_QUEUE_DEPTH_BOUNDS)
            stale = _clamp(float(data["stale_threshold_s"]), *STALE_THRESHOLD_S_BOUNDS)
            if skip <= cheap:
                skip = cheap + 1  # never trust LLM ordering blindly; invariant enforced regardless

            self.thresholds.update(cheap_queue_depth=cheap, skip_queue_depth=skip, stale_threshold_s=stale)
            self.events.append(SupervisorEvent(
                t=time.time(), old=old, new=(cheap, skip, stale),
                reasoning=str(data.get("reasoning", "")),
            ))
        except Exception as e:
            self.events.append(SupervisorEvent(
                t=time.time(), old=old, new=old, reasoning=f"[error, thresholds unchanged] {e!r}",
            ))
