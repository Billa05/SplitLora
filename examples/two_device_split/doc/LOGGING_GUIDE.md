# Logging Guide for Split Learning

This document describes the comprehensive logging system implemented in the distributed split learning framework.

## Overview

The training system now provides detailed progress logging every 10 steps on both the **client** and **server** sides, including:

- Current step and total steps
- Loss metrics
- Training speed (steps per second)
- Estimated Time to Arrival (ETA) for completion
- Perplexity (PPL) on client side

## Client Logging

The client (device_id=0) displays progress every 10 steps with the following format:

```
[Client] [Epoch 1] Step 10/500 | Loss 4.1234 | PPL 61.83 | 2.45 steps/s | ETA: 3.3m
[Client] [Epoch 1] Step 20/500 | Loss 3.8765 | PPL 48.27 | 2.50 steps/s | ETA: 3.2m
[Client] [Epoch 1] Step 30/500 | Loss 3.6543 | PPL 38.62 | 2.48 steps/s | ETA: 3.2m
```

### Client Log Fields

- **[Client]**: Identifies this as client-side output
- **[Epoch N]**: Current training epoch
- **Step X/Y**: Current step (X) out of total steps (Y) in this epoch
- **Loss**: Average loss over the steps logged so far
- **PPL**: Perplexity, calculated as `exp(loss)`, a measure of model uncertainty
- **steps/s**: Training speed in steps per second
- **ETA**: Estimated time remaining for the current epoch

### ETA Format

The ETA is displayed in the most appropriate unit:
- **Seconds** (s): For < 1 minute remaining (e.g., `45s`)
- **Minutes** (m): For 1 minute to < 1 hour remaining (e.g., `3.5m`)
- **Hours** (h): For >= 1 hour remaining (e.g., `2.3h`)

## Server Logging

The server (device_id=1) displays progress every 10 steps with the following format:

```
[Server] Step 10/500 | Loss 4.1234 | 2.45 steps/s | ETA: 3.3m
[Server] Step 20/500 | Loss 3.8765 | 2.50 steps/s | ETA: 3.2m
[Server] Step 30/500 | Loss 3.6543 | 2.48 steps/s | ETA: 3.2m
```

### Server Log Fields

- **[Server]**: Identifies this as server-side output
- **Step X/Y**: Current step (X) out of total steps (Y)
- **Loss**: Current average loss
- **steps/s**: Training speed in steps per second
- **ETA**: Estimated time remaining

## Implementation Details

### Calculation Method

The logging system tracks:

1. **Step counts**: Global step counter incremented after each optimizer step
2. **Time tracking**: Timestamps at each log interval (every 10 steps)
3. **Speed calculation**: `steps_per_sec = steps_since_last_log / time_since_last_log`
4. **ETA calculation**: `eta = remaining_steps / steps_per_sec`

### Communication

The client sends progress information to the server during the optimizer step message:

```python
message = {
    "type": "optimizer_step",
    "clip": args.clip,
    "current_step": global_step,
    "total_steps": total_steps_in_epoch,
    "loss": avg_loss.avg
}
```

This allows the server to display synchronized progress information.

## Monitoring Training

### What to Look For

**Healthy Training Signs:**
- Loss steadily decreasing
- PPL (perplexity) decreasing
- Consistent steps/s (training speed)
- ETA converging as training progresses

**Potential Issues:**
- Loss increasing or oscillating wildly → learning rate too high, or data issues
- Very slow steps/s → possible bottleneck in data loading, network, or computation
- ETA increasing instead of decreasing → training is slowing down over time

### Example Training Output

Here's what you might see during a typical training run:

**Terminal 1 (Server):**
```
[Server] Listening on ws://0.0.0.0:8765
[Server] Client connected!
[Server] Step 10/500 | Loss 4.5234 | 2.12 steps/s | ETA: 3.9m
[Server] Step 20/500 | Loss 4.2156 | 2.15 steps/s | ETA: 3.7m
[Server] Step 30/500 | Loss 3.9845 | 2.14 steps/s | ETA: 3.7m
...
```

**Terminal 2 (Client):**
```
[Client] Connecting to server at ws://localhost:8765
[Client] Connected to server!
[Client] [Epoch 1] Step 10/500 | Loss 4.5234 | PPL 91.97 | 2.12 steps/s | ETA: 3.9m
[Client] [Epoch 1] Step 20/500 | Loss 4.2156 | PPL 67.63 | 2.15 steps/s | ETA: 3.7m
[Client] [Epoch 1] Step 30/500 | Loss 3.9845 | PPL 53.68 | 2.14 steps/s | ETA: 3.7m
...
```

## Legacy Logging

The original `--log_interval` parameter is still supported for backward compatibility. If you set `--log_interval 50`, you'll get additional log messages at steps 50, 100, 150, etc., but the every-10-steps logging always runs.

To disable the legacy logging completely, just ignore it—the new every-10-steps logging provides more frequent and detailed progress information.

## Configuration

No additional configuration is needed! The logging is automatically enabled when you run distributed training with `ws_client.py`.

### Customizing Log Frequency

To change the log frequency from every 10 steps to another interval, edit `ws_client.py`:

**Client side:**
```python
# Change this line (around line 538):
if global_step % 10 == 0:
# To:
if global_step % 20 == 0:  # Log every 20 steps
```

**Server side:**
```python
# Change this line (around line 302):
if self.step_count % 10 == 0:
# To:
if self.step_count % 20 == 0:  # Log every 20 steps
```

## Troubleshooting

### Logs Not Appearing

If you don't see logs every 10 steps:

1. **Check if training is actually progressing**: The optimizer step must complete for logging to occur
2. **Verify gradient accumulation**: If `--grad_acc` is high, actual optimizer steps happen less frequently
3. **Check max_train_steps**: If set too low, training may end before 10 steps

### Mismatched Step Counts

If client and server show different step counts:

1. **This is expected briefly**: There may be a 1-2 step lag due to asynchronous communication
2. **If persistent**: Check for network issues or dropped messages

### ETA Not Showing

ETA requires at least one log interval (10 steps) to calculate speed. It will appear starting from step 20.

## Performance Impact

The logging system has minimal performance impact:

- **Computation**: Simple arithmetic every 10 steps
- **Network**: Adds ~20 bytes per optimizer step message
- **Memory**: Stores only 2-3 timestamps and counters

## Summary

The new logging system provides real-time, detailed progress information for both client and server, making it easy to monitor distributed training runs. The ETA calculation helps you estimate completion time, and the step-by-step metrics help identify training issues early.

For questions or issues, refer to the main README.md or open an issue on GitHub.
