# Reflex CLI Command Reference

> Complete reference for every `reflex` command. Run `reflex <command> --help` for the latest flags.

---

## Quick Orientation

| Verb | Purpose | Typical User |
|---|---|---|
| `go` | One-command deploy: probe → pull → export → serve | Everyone — start here |
| `serve` | Start the inference server from an exported model | Production deployments |
| `ros2-serve` | ROS2 bridge: subscribes to topics, publishes actions | ROS2-based robots & drones |
| `export` | Convert a VLA checkpoint to ONNX | Model engineers |
| `validate` | Round-trip parity check (PyTorch vs ONNX) | QA / verification |
| `doctor` | Diagnose GPU, CUDA, and deploy issues | Debugging |
| `guard` | Initialize or check runtime safety limits | Safety-critical deployments |
| `bench` | Benchmark inference latency & throughput | Performance tuning |
| `models list` | Browse the model registry | Discovery |
| `models pull` | Download a model from the registry | Setup |
| `models info` | Show metadata for a specific model | Inspection |
| `chat` | Interactive VLA chat (experimental) | Exploration |

---

## `reflex go`

**One-command deploy.** Probes your hardware, selects the right model variant, pulls weights, exports to ONNX (if needed), and starts serving.

```bash
# Robotic arm (warehouse pick-and-place)
reflex go --model smolvla-base --embodiment franka --port 8000

# Drone / UAV (aerial inspection)
reflex go --model smolvla-base --embodiment quadcopter --port 8001

# SO-100 hobby arm (farm / lab)
reflex go --model smolvla-base --embodiment so100

# Dry run — show the plan without pulling or serving
reflex go --model pi05-libero --dry-run
```

| Flag | Default | Description |
|---|---|---|
| `--model` | _(required)_ | Registry ID (e.g. `pi05-libero`) or family (`pi05`, `smolvla`, `pi0`) |
| `--embodiment` | _(none)_ | Preset name: `franka`, `so100`, `ur5`, `quadcopter`, or custom JSON path |
| `--device-class` | _(auto)_ | Override probe: `h200`, `h100`, `a100`, `a10g`, `thor`, `agx_orin`, `orin_nano`, `cpu` |
| `--target-dir` | `~/.cache/reflex/models/<id>/` | Weight cache directory |
| `--port` | `8000` | HTTP port for `/act` + `/health` |
| `--host` | `0.0.0.0` | Server bind address |
| `--api-key` | _(none)_ | If set, `/act` requires `X-Reflex-Key` header |
| `--dry-run` | `false` | Print the resolution plan without executing |

---

## `reflex serve`

**Production inference server.** Serves an already-exported model directory via HTTP.

```bash
# Basic serve
reflex serve ./reflex_export/ --port 8000

# With safety limits (clamping actions to joint bounds)
reflex serve ./reflex_export/ --safety-config safety_limits.json

# With API authentication
reflex serve ./reflex_export/ --api-key $REFLEX_API_KEY

# Adaptive denoising for lower latency
reflex serve ./reflex_export/ --adaptive-steps
```

| Flag | Default | Description |
|---|---|---|
| `export_dir` | _(required)_ | Path to the exported model directory |
| `--port` | `8000` | Server port |
| `--host` | `0.0.0.0` | Server host |
| `--device` | `cuda` | Execution device: `cuda` or `cpu` |
| `--providers` | _(auto)_ | Comma-separated ORT execution providers |
| `--no-strict-providers` | `false` | Allow silent fallback to CPU |
| `--safety-config` | _(none)_ | Path to SafetyLimits JSON from `reflex guard init` |
| `--adaptive-steps` | `false` | Early-stop denoising when velocity norm converges |
| `--deadline-ms` | `0` | Per-request deadline; returns last-good action if exceeded |
| `--max-batch-cost-ms` | `100.0` | Chunk-budget batch scheduler flush threshold |
| `--batch-timeout-ms` | `5.0` | Maximum wait per batch flush |
| `--api-key` | _(none)_ | Require `X-Reflex-Key` header for `/act` and `/config` |
| `--cloud-fallback` | _(none)_ | URL of a remote reflex serve for cloud-edge routing |

### Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/act` | Yes (if `--api-key`) | Send image + instruction + state → receive actions |
| `GET` | `/health` | No | Readiness probe (always unauthenticated) |
| `GET` | `/config` | Yes (if `--api-key`) | Server configuration and model metadata |

---

## `reflex ros2-serve`

**ROS2 bridge node.** Subscribes to image, state, and task topics; publishes action chunks.

```bash
# Robotic arm via ROS2
reflex ros2-serve ./reflex_export/ \
  --image-topic /camera/image_raw \
  --state-topic /joint_states \
  --action-topic /reflex/actions \
  --rate-hz 20

# Drone via MAVROS
reflex ros2-serve ./reflex_export/ \
  --image-topic /camera/image_raw \
  --state-topic /mavros/imu/data \
  --action-topic /reflex/actions \
  --rate-hz 50
```

| Flag | Default | Description |
|---|---|---|
| `export_dir` | _(required)_ | Path to exported model directory |
| `--device` | `cuda` | ORT execution device |
| `--image-topic` | `/camera/image_raw` | `sensor_msgs/Image` topic |
| `--state-topic` | `/joint_states` | `sensor_msgs/JointState` (arms) or `sensor_msgs/Imu` (drones) |
| `--task-topic` | `/reflex/task` | `std_msgs/String` — text instruction |
| `--action-topic` | `/reflex/actions` | `std_msgs/Float32MultiArray` — published actions |
| `--rate-hz` | `20.0` | Inference rate. Use 50 Hz for drones, 20 Hz for arms |
| `--safety-config` | _(none)_ | SafetyLimits JSON path |
| `--node-name` | `reflex_vla` | ROS2 node name |
| `--mcp` | `false` | Enable MCP server alongside ROS2 |

> **Vertical guidance:** For warehouse AGVs subscribing to `/odom`, or farm robots using GPS-fused state, pass the appropriate topic via `--state-topic`. The bridge auto-detects `JointState.position` vs `Imu.orientation` fields.

---

## `reflex export`

**Convert a VLA checkpoint to ONNX.** Produces an export directory with ONNX files, config, and a `VERIFICATION.md` receipt.

```bash
reflex export lerobot/smolvla-base --target orin-nano --output ./reflex_export/
reflex export pi0-base --precision fp16 --chunk-size 50
reflex export pi05-libero --dry-run  # check exportability without building
```

| Flag | Default | Description |
|---|---|---|
| `model` | _(required)_ | HuggingFace model ID or local checkpoint path |
| `--target` | `desktop` | Target hardware: `orin-nano`, `orin`, `orin-64`, `thor`, `desktop` |
| `--output` | `./reflex_export` | Output directory |
| `--precision` | `fp16` | Precision: `fp16`, `fp8`, `int8` |
| `--opset` | `19` | ONNX opset version |
| `--chunk-size` | `50` | Action chunk size |
| `--no-validate` | `false` | Skip ONNX validation after export |
| `--dry-run` | `false` | Check exportability without building |
| `--verbose` | `false` | Verbose logging |

---

## `reflex validate`

**Round-trip parity check.** Compares PyTorch model output against ONNX export to verify numerical fidelity.

```bash
reflex validate ./reflex_export/
reflex validate ./reflex_export/ --threshold 1e-4
```

See [How to Read VERIFICATION.md](./understanding_verification.md) for interpreting results.

---

## `reflex doctor`

**Diagnose your environment.** Checks Python, CUDA, ORT providers, and optionally runs deploy-specific diagnostics.

```bash
# System probe only
reflex doctor

# System probe + deploy diagnostics for a specific export
reflex doctor --model ./reflex_export/ --embodiment franka

# Drone deployment diagnostics
reflex doctor --model ./reflex_export/ --embodiment quadcopter

# Machine-readable output for CI
reflex doctor --model ./reflex_export/ --format json

# Show auto-calibration cache
reflex doctor --show-calibration
```

| Flag | Default | Description |
|---|---|---|
| `--model` | _(none)_ | Export directory for deploy diagnostics |
| `--embodiment` | `custom` | Preset for cross-checks: `franka`, `so100`, `ur5`, `quadcopter` |
| `--rtc` | `false` | Validate RTC chunk-boundary alignment |
| `--format` | `human` | Output format: `human` (table) or `json` |
| `--skip` | _(none)_ | Check IDs to skip (repeatable) |
| `--show-calibration` | `false` | Print the auto-calibration cache |

---

## `reflex guard`

**Runtime safety constraints.** Initialize safety limits from a URDF or JSON, then validate actions at inference time.

```bash
# Initialize from URDF
reflex guard init --urdf robot.urdf --output safety_limits.json

# Check a running server's config
reflex guard check --config safety_limits.json
```

> **For drones:** generate safety limits from the quadcopter preset config rather than URDF, since UAVs don't have joint limits in the traditional sense.

---

## `reflex bench`

**Benchmark inference performance.**

```bash
reflex bench ./reflex_export/
reflex bench ./reflex_export/ --warmup 10 --iterations 100
```

---

## `reflex models list`

**Browse the model registry.**

```bash
reflex models list
```

## `reflex models pull`

**Download a model from the registry.**

```bash
reflex models pull smolvla-base
reflex models pull pi05-libero --target-dir ./my_models/
```

## `reflex models info`

**Show metadata for a specific model.**

```bash
reflex models info smolvla-base
```

---

## `reflex validate-dataset`

**Validate a LeRobot v2/v3 dataset.**

```bash
reflex validate-dataset ./dataset/
```

---

## `reflex turbo`

**Adaptive denoising calibration.** Profiles your export to find optimal early-stopping thresholds.

```bash
reflex turbo ./reflex_export/
```

---

## `reflex chat`

**Interactive VLA chat.** Experimental conversational interface for exploring model behavior.

```bash
reflex chat --model smolvla-base
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `REFLEX_NO_UPGRADE_CHECK=1` | Suppress the daily PyPI upgrade nag |
| `REFLEX_API_KEY` | Default API key (alternative to `--api-key` flag) |
| `CUDA_VISIBLE_DEVICES` | Restrict which GPUs reflex uses |

---

## Vertical Quick-Start Matrix

| Vertical | Command | Key Flags |
|---|---|---|
| **Warehouse arm** | `reflex go --model pi05 --embodiment franka` | `--port 8000` |
| **Farm robotics** | `reflex go --model smolvla-base --embodiment so100` | `--device-class orin_nano` |
| **Aerial drone** | `reflex go --model smolvla-base --embodiment quadcopter` | `--port 8001` |
| **Retail camera** | `reflex serve ./export/ --deadline-ms 100` | `--adaptive-steps` |
| **Traffic AI** | `reflex serve ./export/ --max-batch-cost-ms 200` | High throughput batching |
