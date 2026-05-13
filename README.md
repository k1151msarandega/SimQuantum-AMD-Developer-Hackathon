# ⚛ SimQuantum Tuning Lab
**Autonomous quantum dot tuning · AMD Developer Hackathon 2026 · Kudzai Musarandega**

[![Live Demo](https://img.shields.io/badge/Demo-HuggingFace%20Spaces-yellow)](https://huggingface.co/spaces/lablab-ai-amd-developer-hackathon/simquantum-tuning-lab)
[![AMD MI300X](https://img.shields.io/badge/Hardware-AMD%20MI300X-red)](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
[![Model](https://img.shields.io/badge/LLM-Qwen3--8B-blue)](https://huggingface.co/Qwen/Qwen3-8B)

---

## The Problem

Before a quantum chip can run a single calculation, a PhD physicist must manually
nudge gate voltages — one by one — until exactly one electron sits in each quantum
dot. This process takes **hours to days per cooldown cycle**, requires expert
intuition, and does not scale. It is the primary operational bottleneck preventing
quantum hardware from reaching its potential.

SimQuantum automates it.

---

## What It Does

A six-stage agentic loop navigates a 2D gate voltage space and autonomously
locates the **(1,1) charge state** — one electron per dot — required for spin
qubit operation. No human guidance. No manual tuning.

```
BOOTSTRAPPING → COARSE_SURVEY → HYPERSURFACE_SEARCH → CHARGE_ID
     ↓               ↓                  ↓                  ↓
  DQC gate      Gradient peak      Local refinement    CNN ensemble
  SNR check     detection          16×16 scan          91.4% val acc
                                                       + OOD detection
```

**NAVIGATION → VERIFICATION** — Phase 3, active development.

---

## AMD Hardware Integration

The full inference stack targets **AMD Instinct MI300X** via ROCm:

- **Qwen2.5-1.5B-Instruct** served via vLLM on MI300X — OpenAI-compatible endpoint
- **Dr. Q (Qwen3-8B)** embedded into the agent architecture with live access to all
  instrument state, classifications, and anomaly flags
- PyTorch ROCm build (`2.10.0+rocm7.0`) for CNN ensemble inference
- MI300X unified 192GB HBM3 eliminates host-device transfer overhead across the
  multi-model pipeline

The LLM backend is fully endpoint-agnostic — one environment variable switches
Dr. Q between MI300X, Fireworks AI, or any OpenAI-compatible server:

```bash
export QDOT_LLM_BASE_URL="http://<mi300x-host>:8000/v1"
export QDOT_LLM_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
export QDOT_LLM_API_KEY="your-key"
```

---

## Agent Architecture

| Agent | Role |
|---|---|
| **Executive** | Orchestrates the 6-stage POMDP loop |
| **Perception** | DQC gatekeeper + 5-model CNN ensemble + Mahalanobis OOD detector |
| **Planning** | Particle filter belief state + MultiResBO Bayesian optimiser |
| **Safety** | Hard voltage bounds, step caps, 12-condition HITL risk scoring |
| **Dr. Q** | Qwen3-8B with live agent state injection — explains, alerts, summarises |

Dr. Q is not a chatbot bolted onto the interface. It receives a dynamically
constructed system prompt on every call containing current stage, voltages,
CNN classification confidence, ensemble disagreement, belief probability P(1,1),
SNR, budget remaining, and active anomaly flags. It has access to the last 16
turns of conversation history and can answer questions about the running
experiment in any register — from a curious student to a device physicist.

---

## Business Value

Every quantum hardware lab in the world tunes devices manually today.
IBM operates over 100 quantum systems. Each requires tuning every time
environmental conditions shift. A working automated tuner means:

- **Researchers get their time back** — hours of manual tuning becomes a
  background process
- **Higher throughput** — more experiments per cooldown cycle
- **Lower barrier to entry** — labs without deep tuning expertise can operate
  quantum hardware
- **Scalability** — the path from one device to a rack of devices requires
  automation at this layer

The real device adapter interface is already defined in the codebase.
Pointing it at a Si/SiGe or GaAs device replaces the simulator with
real hardware — zero architecture changes required.

---

## Results

| Stage | Status | Notes |
|---|---|---|
| Bootstrapping | ✅ Working | DQC gate + SNR validation |
| Coarse Survey | ✅ Working | Gradient-based peak detection |
| Hypersurface Search | ✅ Working | Local boundary refinement |
| Charge ID | ✅ Working | 91.4% ensemble accuracy, 51k training diagrams |
| Navigation | 🔬 Active development | Reward signal design — open research problem |
| Verification | 🔬 Active development | Pending Navigation |

The system locates a charge transition autonomously in a 256 V² gate voltage
space using ~2,400 measurements out of an 8,096 budget.

---

## Technical Walkthrough

Full architecture writeup: [[`docs/simquantum_technical_walkthrough.pdf`](docs/simquantum_technical_walkthrough.pdf)](https://pdflink.to/5c1e4dd3/)

Covers: AMD ROCm integration, vLLM deployment, agent architecture,
infrastructure challenges, and future work.

---

## Run the Demo

```bash
git clone https://github.com/k1151msarandega/SimQuantum-AMD-Developer-Hackathon.git
cd SimQuantum-AMD-Developer-Hackathon
pip install -r requirements.txt
streamlit run app.py
```

Or visit the live Space (Dr. Q pre-connected via Fireworks AI):
**[huggingface.co/spaces/lablab-ai-amd-developer-hackathon/simquantum-tuning-lab](https://huggingface.co/spaces/lablab-ai-amd-developer-hackathon/simquantum-tuning-lab)**
