# Target audience and learning objectives

(Required WISER submission section -- drafted here first, then
summarized/linked from the top-level README.)

## Target audience
Undergraduate or early-graduate students in physics, electrical
engineering, or CS who have working Python but little or no hands-on
exposure to real quantum-hardware control systems. Assumes comfort with
basic programming (functions, classes, reading a config file) and basic
probability/statistics (mean, variance, z-score) -- does NOT assume prior
knowledge of quantum dot devices, digital twins, or systems/control
concepts like triage or backpressure. Also usable by a slightly broader
"quantum-curious, comfortable with code" audience.

## Learning objectives
By the end of the lesson arc, a learner should be able to:

1. **Explain what a digital twin is** in the context of quantum hardware,
   and why device teams build them (cheap/fast virtual iteration vs.
   costly, slow cryogenic hardware time).
2. **Explain the constant-capacitance model** for a gate-defined
   semiconductor quantum dot array: how gate voltages and
   capacitance couplings (Cdd, Cgd) determine charge configurations, and
   how that produces the honeycomb-pattern "charge stability diagrams"
   real experimentalists read.
3. **Articulate the throughput/latency tradeoff** between processing data
   one frame at a time (serial) and processing it in batches, and predict
   qualitatively when batching helps vs. when it doesn't.
4. **Reason about system staleness under load**: why a fixed-throughput
   estimator falls behind a data stream whose rate increases, and how to
   read a staleness-over-time chart.
5. **Design and critique a triage policy**: given multiple real signals
   (queue depth, time since last full update, drift activity), explain
   why a multi-signal rule beats a single threshold, and predict how
   changing a threshold changes system behavior.
6. **Interpret drift/out-of-distribution detection** via rolling
   statistics, and distinguish a gradual, legitimate change (slow gate
   drift) from a sudden discrete event (a jump) using the same detector.
7. **Connect an electrostatic energy model to spatial intuition**: read a
   3D confinement-potential surface and explain, qualitatively, when an
   electron is more likely to be trapped -- while correctly stating what
   that visualization is (a pedagogical interpolation) and is not (a full
   Poisson-equation solve).
8. **Critically evaluate a simulation's own assumptions and limitations**
   -- modeled directly on this codebase's own practice of documenting
   design tradeoffs and open TODOs rather than hiding them.

## Non-goals
This is not a rigorous course in electrostatics, quantum mechanics, or
distributed systems -- it borrows one running example from a real
experimental-quantum-computing engineering problem to teach systems
thinking and quantum-hardware intuition together. Learners will not come
away able to design a real charge-stability-diagram autotuner; they will
come away understanding why one is hard and what tradeoffs it has to make.
