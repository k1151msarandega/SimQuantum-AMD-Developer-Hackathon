cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e .

COPY configs/ configs/
COPY scripts/ scripts/

ENTRYPOINT ["python", "scripts/run_demo.py"]
EOF
