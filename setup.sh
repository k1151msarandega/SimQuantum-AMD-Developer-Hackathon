#!/bin/bash
# setup.sh — SimQuantum MI300X one-time setup
# Run this ONCE on a fresh droplet.
# After this, use start.sh every time.
set -e

CONDA_ENV="qdots"
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
REPO_URL="https://huggingface.co/spaces/lablab-ai-amd-developer-hackathon/simquantum-tuning-lab"
REPO_DIR="/root/simquantum-tuning-lab"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SimQuantum — One-Time Droplet Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── 1. Clone repo ─────────────────────────────────────────────────────────────
if [ -d "$REPO_DIR/.git" ]; then
    echo "► Repo already exists — pulling latest..."
    cd "$REPO_DIR"
    git pull origin main
else
    echo "► Cloning SimQuantum from HuggingFace..."
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi
echo "  ✓ Repo ready at $REPO_DIR"

# ── 2. Conda init ─────────────────────────────────────────────────────────────
echo "► Initializing conda..."
source /root/miniconda3/etc/profile.d/conda.sh

# ── 3. Create env if it doesn't exist ────────────────────────────────────────
if conda env list | grep -q "^${CONDA_ENV} "; then
    echo "  ✓ Conda env '$CONDA_ENV' already exists"
else
    echo "► Creating conda env '$CONDA_ENV' (Python 3.11)..."
    conda create -y -n "$CONDA_ENV" python=3.11
    echo "  ✓ Created"
fi
conda activate "$CONDA_ENV"
echo "  ✓ Python: $(python --version)"

# ── 4. ROCm PyTorch ───────────────────────────────────────────────────────────
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  ✓ ROCm PyTorch already installed"
else
    echo "► Installing ROCm PyTorch (this takes a few minutes)..."
    pip install torch torchvision \
        --index-url https://download.pytorch.org/whl/rocm6.2 \
        --quiet
    echo "  ✓ Done"
fi

# ── 5. vLLM ───────────────────────────────────────────────────────────────────
if python -c "import vllm" 2>/dev/null; then
    echo "  ✓ vLLM already installed"
else
    echo "► Installing vLLM..."
    pip install vllm --quiet
    echo "  ✓ Done"
fi

# ── 6. App dependencies ───────────────────────────────────────────────────────
echo "► Installing app dependencies..."
pip install streamlit==1.57.0 plotly openai numpy scipy scikit-learn tqdm --quiet
pip install -e . --quiet
echo "  ✓ Done"

# ── 7. Pre-download the model weights ────────────────────────────────────────
# Do this now so start.sh doesn't spend credits downloading later
echo "► Pre-downloading $MODEL weights (one-time, ~3GB)..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL')
print('  ✓ Model weights cached')
"

# ── 8. Write start.sh into the repo dir so it's always there ─────────────────
cat > "$REPO_DIR/start.sh" << 'STARTSCRIPT'
#!/bin/bash
# start.sh — run this every time you boot the droplet
CONDA_ENV="qdots"
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
VLLM_PORT=8000
STREAMLIT_PORT=8501
REPO_DIR="/root/simquantum-tuning-lab"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SimQuantum — Starting Up"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "$REPO_DIR"

# Pull latest code (cheap, always do it)
git pull origin main --quiet && echo "► Code up to date ✓" || echo "► (git pull skipped)"

# Init conda
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

PYTHON="$(which python)"
VLLM_BIN="$(which vllm)"

# ── Start vLLM if not running ─────────────────────────────────────────────────
if curl -s http://localhost:$VLLM_PORT/v1/models > /dev/null 2>&1; then
    echo "► vLLM already running ✓"
else
    echo "► Starting vLLM (Qwen2.5-1.5B on MI300X)..."
    export HIP_VISIBLE_DEVICES=0
    export ROCR_VISIBLE_DEVICES=0
    export VLLM_TARGET_DEVICE=rocm
    export HSA_OVERRIDE_GFX_VERSION=9.4.2

    nohup "$VLLM_BIN" serve "$MODEL" \
        --host 0.0.0.0 \
        --port $VLLM_PORT \
        --gpu-memory-utilization 0.45 \
        --max-model-len 4096 \
        > /tmp/vllm.log 2>&1 &
    VLLM_PID=$!

    echo -n "  Waiting for vLLM"
    for i in $(seq 1 120); do
        curl -s http://localhost:$VLLM_PORT/v1/models > /dev/null 2>&1 && echo " ✓" && break
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo ""
            echo "  ✗ vLLM crashed. Last 20 lines:"
            tail -20 /tmp/vllm.log
            exit 1
        fi
        printf "."; sleep 1
    done
fi

# ── Kill old Streamlit only ───────────────────────────────────────────────────
pkill -f "streamlit run" 2>/dev/null || true
sleep 1

# ── Start Streamlit ───────────────────────────────────────────────────────────
echo "► Starting Streamlit..."
export QDOT_LLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"
export QDOT_LLM_MODEL="$MODEL"

nohup "$PYTHON" -m streamlit run app.py \
    --server.port "$STREAMLIT_PORT" \
    --server.address 0.0.0.0 \
    --server.headless true \
    > /tmp/streamlit.log 2>&1 &

echo -n "  Waiting for Streamlit"
for i in $(seq 1 30); do
    curl -s http://localhost:$STREAMLIT_PORT > /dev/null 2>&1 && echo " ✓" && break
    printf "."; sleep 1
done

PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "YOUR_IP")
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✓ SimQuantum is live!"
echo ""
echo "  Open this in your browser:"
echo "  http://${PUBLIC_IP}:${STREAMLIT_PORT}"
echo ""
echo "  If something looks wrong:"
echo "    tail -f /tmp/vllm.log"
echo "    tail -f /tmp/streamlit.log"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STARTSCRIPT

chmod +x "$REPO_DIR/start.sh"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✓ Setup complete!"
echo ""
echo "  From now on, just run:"
echo "    cd /root/simquantum-tuning-lab"
echo "    bash start.sh"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
