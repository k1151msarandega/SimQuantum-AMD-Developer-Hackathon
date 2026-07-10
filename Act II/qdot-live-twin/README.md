cat > README.md << 'EOF'
# qdot-live-twin

A GPU-accelerated live digital twin of a quantum dot device, with a
throughput-aware triage agent.

A **twin** passively mirrors the device's current state as it changes; it
does not steer the device anywhere. Success is staying synchronised (low
staleness), not "reaching a target."

## Setup
