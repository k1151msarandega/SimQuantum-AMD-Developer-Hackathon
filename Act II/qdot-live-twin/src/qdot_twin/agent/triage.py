"""Triage agent: the real decision-making piece.

Consumes real signals -- queue depth, time-since-last-full-update, and
recent drift-flag activity -- and decides, per incoming frame, one of:
  FULL    -- run the full batched estimator
  CHEAP   -- run a cheap approximate update
  SKIP    -- skip and flag stale

This decision must be *consumed* by the pipeline (i.e. it actually changes
which code path a frame goes through) -- this is exactly the mistake to
avoid repeating from Dr. Q in Act I, where the LLM's recommendation was
computed but never wired into the control flow.

Deliberately rule-based, not LLM-backed: multi-signal weighing (not a
single threshold) is what makes this a real triage decision rather than a
control loop, per the hackathon-scoping discussion.
"""
from enum import Enum, auto


class Tier(Enum):
    FULL = auto()
    CHEAP = auto()
    SKIP = auto()


def decide(queue_depth: int, time_since_full_update: float, recent_drift_activity: bool) -> Tier:
    """Weigh queue depth, staleness, and drift activity together.

    E.g.: high backlog + no recent drift -> CHEAP or SKIP.
          high backlog + recent drift flag -> FULL anyway, over budget.
    This reconciliation across signals (not a single comparison) is what
    makes the decision non-trivial.
    """
    raise NotImplementedError
