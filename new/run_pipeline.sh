#!/bin/bash

# Pipeline Training Script
# This script runs the 2-device sequential pipeline in the correct order

echo "Starting 2-device sequential pipeline training..."
echo "================================================"
echo ""

# Kill any existing processes on the ports
echo "Cleaning up existing processes..."
lsof -ti:8765 | xargs kill -9 2>/dev/null || true
lsof -ti:8766 | xargs kill -9 2>/dev/null || true
sleep 1

# Start devices in reverse order (last to first)
# Device 1 must start first (it listens on 8766)
echo "Starting Device 1 (layers 6-11 + LM Head, listening on 8766)..."
python ws_client.py --device_id 1 > device_1.log 2>&1 &
DEVICE_1_PID=$!
sleep 2

# Device 0 starts last (connects to 8766, has the data)
echo "Starting Device 0 (embeddings + layers 0-5, connecting to 8766)..."
python ws_client.py --device_id 0 > device_0.log 2>&1 &
DEVICE_0_PID=$!

echo ""
echo "All devices started!"
echo "  Device 0 PID: $DEVICE_0_PID"
echo "  Device 1 PID: $DEVICE_1_PID"
echo ""
echo "Monitoring logs (Ctrl+C to stop)..."
echo "================================================"

# Monitor all logs
tail -f device_0.log device_1.log &
TAIL_PID=$!

# Wait for Device 0 to finish (it has the training loop)
wait $DEVICE_0_PID

echo ""
echo "================================================"
echo "Device 0 finished. Waiting for other devices to save adapters..."
sleep 3  # Give devices time to save their LoRA adapters

# Kill the tail process
kill $TAIL_PID 2>/dev/null || true

# Kill remaining devices
kill $DEVICE_1_PID 2>/dev/null || true

echo "All devices stopped."
echo ""
echo "LoRA adapters saved:"
ls -lh lora_device_*.pth 2>/dev/null || echo "No LoRA files found"
echo ""
echo "Log files:"
ls -lh device_*.log
