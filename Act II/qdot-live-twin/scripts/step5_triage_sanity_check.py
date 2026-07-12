"""Step 5 sanity check: exercise agent.triage.decide() across the signal
space it's meant to reconcile, to confirm the branching logic actually
behaves as documented before wiring it into pipeline.py.
"""
from qdot_twin.agent.triage import decide, Tier

cases = [
    # (queue_depth, time_since_full_update, recent_drift_activity, note)
    (0, 0.001, False, "no pressure at all"),
    (5, 0.001, False, "light backlog, fresh -- should still be FULL, budget available"),
    (20, 0.01, False, "moderate backlog, fresh -- CHEAP, bridging"),
    (20, 0.08, False, "moderate backlog, stale -- FULL wins, correctness debt"),
    (80, 0.01, False, "severe backlog, fresh -- SKIP, shed load"),
    (80, 0.08, False, "severe backlog AND stale -- still SKIP, queue check comes first"),
    (80, 0.01, True, "severe backlog, but drift active -- FULL overrides everything"),
    (0, 0.09, False, "no backlog but very stale -- FULL, pay down debt"),
]

print(f"{'queue':>6} {'stale(s)':>9} {'drift':>6}  {'decision':>8}   note")
for queue_depth, staleness, drift, note in cases:
    tier = decide(queue_depth, staleness, drift)
    print(f"{queue_depth:>6} {staleness:>9.3f} {str(drift):>6}  {tier.name:>8}   {note}")
