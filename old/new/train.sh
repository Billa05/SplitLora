#!/bin/bash
# Run distributed training with live terminal output

set -e

echo "========================================================================"
echo "  GPT-2 Split Model Training - 2 Device Pipeline"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  - Training steps: 2000"
echo "  - Batch size: 4"
echo "  - LoRA rank: 16, alpha: 16"
echo "  - Learning rate: 1e-5"
echo ""
echo "Device allocation:"
echo "  - Device 0: Embeddings + Layers 0-5"
echo "  - Device 1: Layers 6-11 + LM Head + Loss"
echo ""
echo "========================================================================"
echo ""

# Clean up old log files
rm -f device_0.log device_1.log

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping training processes..."
    kill $DEVICE0_PID $DEVICE1_PID 2>/dev/null || true
    wait 2>/dev/null
    echo "Cleanup complete"
}

trap cleanup EXIT INT TERM

echo "Starting Device 1 (listener)..."
python ws_client.py --device_id 1 2>&1 | sed 's/^/[D1] /' &
DEVICE1_PID=$!

echo "Waiting 3 seconds for Device 1 to initialize..."
sleep 3

echo "Starting Device 0 (driver)..."
python ws_client.py --device_id 0 2>&1 | sed 's/^/[D0] /' &
DEVICE0_PID=$!

echo ""
echo "========================================================================"
echo "  Training in progress..."
echo "  Press Ctrl+C to stop"
echo "========================================================================"
echo ""

# Wait for both processes to complete
wait $DEVICE0_PID
wait $DEVICE1_PID

echo ""
echo "========================================================================"
echo "  Training Complete!"
echo "========================================================================"
echo ""
echo "Models saved:"
echo "  ✓ lora_device_0.pth"
echo "  ✓ lora_device_1.pth"
echo ""
echo "Test with:"
echo '  python infer_2device.py --prompt "name : Blue Spice | Type : restaurant | food : French | price : more than £ 30 | area : riverside<|endoftext|>"'
echo ""
