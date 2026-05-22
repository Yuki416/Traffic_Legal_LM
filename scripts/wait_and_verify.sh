#!/bin/bash
LOG=/tmp/setup_verify.log
PERSISTENT_LOG=/home/under115b/work/DL_project/setup_verify.log
WORKDIR=/home/under115b/work/DL_project

exec > >(tee -a "$LOG" "$PERSISTENT_LOG") 2>&1

echo "============================================"
echo "[$(date)] Waiting for Docker build to finish..."
echo "============================================"

# Wait until legal-lora:latest image exists
while ! docker image inspect legal-lora:latest &>/dev/null; do
    echo "[$(date)] Build still in progress..."
    sleep 30
done

echo ""
echo "[$(date)] ✓ Docker image legal-lora:latest is ready!"
echo ""

echo "============================================"
echo "[$(date)] Running Step 6: verification script"
echo "============================================"

docker run --gpus all --rm \
    -v "$WORKDIR":/workspace \
    -w /workspace \
    legal-lora:latest \
    python scripts/test_setup.py

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date)] ✓ All done! Environment setup verified successfully."
else
    echo "[$(date)] ✗ Verification failed with exit code $EXIT_CODE."
fi
echo "============================================"
