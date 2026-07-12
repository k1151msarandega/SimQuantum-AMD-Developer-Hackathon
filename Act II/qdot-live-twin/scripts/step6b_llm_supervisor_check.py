"""Sanity check for the LLM supervisor, isolated from the full pipeline.

Run this BEFORE running mode="batched_triage_llm" in the full pipeline.
It exercises the real Fireworks round-trip and the clamp/parse logic
against synthetic history, with no QArray/GPU dependency, so a bad
API key, wrong model id, or JSON-parsing bug shows up here in seconds
instead of after a multi-minute full pipeline run.

Requires: FIREWORKS_API_KEY set in the environment.

Usage:
    python scripts/step6b_llm_supervisor_check.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qdot_twin.agent.thresholds import TriageThresholds
from qdot_twin.agent.llm_supervisor import LLMSupervisor, RollingHistory


def main():
    if "FIREWORKS_API_KEY" not in os.environ:
        print("FIREWORKS_API_KEY not set -- export it before running this check.")
        sys.exit(1)

    thresholds = TriageThresholds()
    history = RollingHistory()

    # Synthetic scenario: backlog consistently near/above the default
    # skip threshold (50), drift quiet. A reasonable LLM should suggest
    # lowering cheap_queue_depth to shed load earlier.
    print("Scenario A: sustained heavy backlog, no drift")
    for i in range(40):
        history.record(queue_depth=45 + (i % 10), time_since_full=0.01, drift_active=False, tier_name="CHEAP")

    supervisor = LLMSupervisor(thresholds, history)
    before = thresholds.snapshot()
    supervisor._tick()  # direct call -- no background thread needed for this check
    after = thresholds.snapshot()

    print(f"  before: cheap_queue_depth={before[0]} skip_queue_depth={before[1]} stale_threshold_s={before[2]}")
    print(f"  after:  cheap_queue_depth={after[0]} skip_queue_depth={after[1]} stale_threshold_s={after[2]}")
    if supervisor.events:
        print(f"  reasoning: {supervisor.events[-1].reasoning}")
    else:
        print("  WARNING: no event recorded -- _tick() returned without updating or logging anything")

    print()
    print("Scenario B: light backlog, thresholds never approached")
    history2 = RollingHistory()
    for i in range(40):
        history2.record(queue_depth=2, time_since_full=0.005, drift_active=False, tier_name="FULL")
    thresholds2 = TriageThresholds()
    supervisor2 = LLMSupervisor(thresholds2, history2)
    before2 = thresholds2.snapshot()
    supervisor2._tick()
    after2 = thresholds2.snapshot()
    print(f"  before: {before2}")
    print(f"  after:  {after2}")
    if supervisor2.events:
        print(f"  reasoning: {supervisor2.events[-1].reasoning}")

    print()
    print("If both scenarios show a real event with sane (non-error) reasoning and")
    print("thresholds that stayed within bounds, the supervisor is ready to wire into")
    print("the full pipeline via mode='batched_triage_llm'.")


if __name__ == "__main__":
    main()
