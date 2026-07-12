"""Triage agent: the real decision-making piece.

Consumes real signals -- queue depth, time-since-last-full-update, and
recent drift-flag activity -- and decides, per incoming frame, one of:
  FULL    -- run the full ensemble (twin/serial_estimator.py:estimate)
  CHEAP   -- run a reduced ensemble (twin/serial_estimator.py:estimate_cheap)
  SKIP    -- skip and flag stale

This decision must be *consumed* by the pipeline (i.e. it actually changes
which code path a frame goes through) -- this is exactly the mistake to
avoid repeating from Dr. Q in Act I, where the LLM's recommendation was
computed but never wired into the control flow.

Deliberately rule-based on the hot path: multi-signal weighing (not a
single threshold) is what makes this a real triage decision rather than a
control loop. Policy, in plain terms: minimize cost EXCEPT when correctness
is actually at stake, in which case always pay for FULL regardless of
backlog. "At stake" means either (a) a drift event is active right now, or
(b) it's been too long since a trustworthy full update, even if the queue
is currently short.

Optional slow layer (agent/llm_supervisor.py): the three thresholds below
can be tuned in the background by an LLM reasoning over recent backlog/
staleness/drift trends, via a shared TriageThresholds object passed in as
`thresholds`. That LLM never sits in this function's call path and never
makes the per-frame decision itself -- it only adjusts the numbers this
function weighs, on its own slower cadence. decide() with thresholds=None
behaves exactly as it always has.

STALE_THRESHOLD_S is grounded in step 2's own measured data, not guessed:
the finalized serial-baseline run's worst observed lag was ~0.0625s, so
0.05s sits just inside "clearly degraded, worth escalating" territory
based on what this exact pipeline has actually shown it can produce under
load. QUEUE_DEPTH thresholds are reasonable placeholders pending step 6's
pipeline wiring, which will exercise real queue dynamics end to end --
flagged here rather than presented as equally well-grounded.
"""
from enum import Enum, auto

from qdot_twin.agent.thresholds import TriageThresholds


class Tier(Enum):
    FULL = auto()
    CHEAP = auto()
    SKIP = auto()


# Placeholders pending real queue dynamics from pipeline.py (step 6).
CHEAP_QUEUE_DEPTH = 10
SKIP_QUEUE_DEPTH = 50

# Grounded in step 2's measured worst-case lag (~0.0625s) -- see docstring.
STALE_THRESHOLD_S = 0.05


def decide(
    queue_depth: int,
    time_since_full_update: float,
    recent_drift_activity: bool,
    thresholds: "TriageThresholds | None" = None,
) -> Tier:
    """Weigh queue depth, staleness, and drift activity together.

    Order matters and is deliberate:
      1. A drift event happening right now overrides everything else --
         correctness during a real event matters more than saving compute.
      2. Backlog so severe that even a cheap update can't keep the queue
         from growing -- shed load entirely rather than fall further
         behind on every single frame.
      3. Moderate backlog, but a full update happened recently enough that
         a cheap update is a safe way to bridge until the queue clears.
      4. Too long since a trustworthy full update, regardless of current
         queue depth -- correctness debt accumulated and is worth paying
         down even though it costs more.
      5. No real pressure -- run FULL since there's budget to spare.

    `thresholds`, if given, overrides the module-level constants with a
    live snapshot from a TriageThresholds object (see llm_supervisor.py).
    Left as None, behavior is identical to the original rule-based-only
    version.
    """
    if thresholds is not None:
        cheap_q, skip_q, stale_s = thresholds.snapshot()
    else:
        cheap_q, skip_q, stale_s = CHEAP_QUEUE_DEPTH, SKIP_QUEUE_DEPTH, STALE_THRESHOLD_S

    if recent_drift_activity:
        return Tier.FULL

    if queue_depth > skip_q:
        return Tier.SKIP

    if queue_depth > cheap_q and time_since_full_update < stale_s:
        return Tier.CHEAP

    if time_since_full_update > stale_s:
        return Tier.FULL

    return Tier.FULL
