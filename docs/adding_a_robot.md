# Adding a Robot — Embodiment Cookbook

> Step-by-step guide to adding a new robot (arm, drone, AGV, or camera rig) to Reflex VLA.

---

## Overview

Reflex ships with embodiment presets for common platforms (Franka, SO-100, UR5, Quadcopter). Adding your own robot takes 4 steps:

1. **Create the JSON config** — define action space, normalization, control rates
2. **Place it in the presets directory** — so `reflex go --embodiment <name>` finds it
3. **Add it to the schema enum** — so validation passes
4. **Test it** — confirm `reflex doctor` and the test suite are happy

This tutorial walks through two examples:
- **MyArm-6** — a fictional 6-DOF robotic arm (warehouse / farm use case)
- **SkyScout** — a fictional surveillance drone (retail / traffic monitoring use case)

---

## Step 1: Create the JSON Config

### Example A: Robotic Arm (MyArm-6)

Create `src/reflex/embodiments/presets/myarm6.json`:

```json
{
  "schema_version": 1,
  "embodiment": "myarm6",
  "action_space": {
    "dim": 7,
    "labels": ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"],
    "type": "continuous",
    "ranges": [
      [-3.14, 3.14],
      [-1.57, 1.57],
      [-3.14, 3.14],
      [-1.57, 1.57],
      [-3.14, 3.14],
      [-3.14, 3.14],
      [0.0, 1.0]
    ]
  },
  "normalization": {
    "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
    "std":  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5]
  },
  "gripper": {
    "component_idx": 6,
    "close_threshold": 0.5,
    "inverted": false
  },
  "cameras": [
    {
      "name": "wrist_cam",
      "width": 640,
      "height": 480,
      "fps": 30,
      "encoding": "rgb8"
    }
  ],
  "control": {
    "frequency_hz": 30.0,
    "chunk_size": 50,
    "rtc_execution_horizon": 10
  },
  "constraints": {
    "max_ee_velocity": 1.5,
    "max_gripper_velocity": 2.0,
    "workspace_bbox": [[-1.0, -1.0, 0.0], [1.0, 1.0, 1.5]]
  }
}
```

### Example B: Surveillance Drone (SkyScout)

Create `src/reflex/embodiments/presets/skyscout.json`:

```json
{
  "schema_version": 1,
  "embodiment": "skyscout",
  "action_space": {
    "dim": 5,
    "labels": ["roll_rate", "pitch_rate", "yaw_rate", "thrust", "payload"],
    "type": "continuous",
    "ranges": [
      [-3.14, 3.14],
      [-3.14, 3.14],
      [-3.14, 3.14],
      [0.0, 1.0],
      [0.0, 1.0]
    ]
  },
  "normalization": {
    "mean": [0.0, 0.0, 0.0, 0.5, 0.0],
    "std":  [1.0, 1.0, 1.0, 0.5, 1.0]
  },
  "gripper": {
    "component_idx": 4,
    "close_threshold": 0.5,
    "inverted": false
  },
  "cameras": [
    {
      "name": "front_rgb",
      "width": 640,
      "height": 480,
      "fps": 30,
      "encoding": "rgb8"
    },
    {
      "name": "downward_rgb",
      "width": 320,
      "height": 240,
      "fps": 15,
      "encoding": "rgb8"
    }
  ],
  "control": {
    "frequency_hz": 50.0,
    "chunk_size": 20,
    "rtc_execution_horizon": 5
  },
  "constraints": {
    "max_ee_velocity": 8.0,
    "max_gripper_velocity": 1.0,
    "workspace_bbox": [[-50.0, -50.0, 0.5], [50.0, 50.0, 120.0]]
  }
}
```

> **Key differences for drones:** Higher `frequency_hz` (50 Hz vs 30 Hz for arms), smaller `chunk_size` (20 vs 50), and much larger `workspace_bbox` (meters of airspace vs a tabletop).

---

## Step 2: Place the Config

The preset needs to exist in **two locations**:

```
src/reflex/embodiments/presets/<name>.json   ← ships with pip install
configs/embodiments/<name>.json              ← editable dev copy
```

```bash
# Copy to both locations
cp myarm6.json src/reflex/embodiments/presets/myarm6.json
cp myarm6.json configs/embodiments/myarm6.json
```

---

## Step 3: Add to Schema Enum

Edit `src/reflex/embodiments/schema.json` — add your embodiment slug to the enum:

```diff
 "embodiment": {
   "type": "string",
-  "enum": ["franka", "so100", "ur5", "trossen", "stretch", "quadcopter", "custom"],
+  "enum": ["franka", "so100", "ur5", "trossen", "stretch", "quadcopter", "myarm6", "custom"],
   "description": "Embodiment slug. Must match the file name minus .json for presets."
 },
```

> **Or use `"custom"`:** If you don't want to modify the schema, you can load any JSON config via `--custom-embodiment-config /path/to/myarm6.json` instead. The `"custom"` slug skips enum validation.

---

## Step 4: Test It

### 4a. Add to the test matrix

Edit `tests/test_embodiments.py`:

```python
ALL_PRESETS = ["franka", "myarm6", "quadcopter", "so100", "ur5"]
```

### 4b. Run the test suite

```bash
pytest tests/test_embodiments.py -v
```

You should see your new preset loading and passing all schema + cross-field validations.

### 4c. Run reflex doctor

```bash
reflex doctor --model ./reflex_export/ --embodiment myarm6
```

This runs the 5 deploy diagnostics (gripper config, normalization sanity, action dim match, etc.) against your custom config.

---

## Vertical-Specific Guidance

### Warehouse Robots (AGVs, Pick-and-Place Arms)

- **Action space:** Typically 6-7 DOF (6 joints + gripper)
- **Control rate:** 20–30 Hz (sufficient for manipulation tasks)
- **Camera setup:** Wrist cam + overhead cam for bin picking
- **State topic (ROS2):** `/joint_states` (standard `sensor_msgs/JointState`)
- **Safety:** Tight `workspace_bbox` to prevent collisions with shelving

### Farm Robotics (SO-100, Custom Arms)

- **Action space:** 6-7 DOF
- **Control rate:** 20 Hz
- **Camera setup:** Front RGB; consider adding depth for outdoor environments
- **Hardware:** Jetson Orin Nano for low-power deployments
- **Key constraint:** Outdoor lighting variation — ensure camera `fps` is high enough

### Aerial Drones (Quadcopters, Fixed-Wing)

- **Action space:** 4-5 DOF (roll/pitch/yaw rates + thrust ± payload)
- **Control rate:** 50 Hz (PX4/ArduPilot outer loop rate)
- **Camera setup:** Front + downward for VLA inference
- **State topic (ROS2):** `/mavros/imu/data` (not `/joint_states`)
- **Safety:** Large `workspace_bbox` for airspace, altitude floor > 0

### Retail Loss Prevention (Camera-Only Inference)

- **Action space:** Can be 0-DOF (pure perception) or PTZ camera control
- **Control rate:** 10–15 Hz (frame-rate bound)
- **Camera setup:** Fixed overhead or PTZ security cameras
- **Deployment:** `reflex serve` with `--deadline-ms` for real-time alerts

### Traffic Management AI

- **Action space:** Typically perception-only (signal classification, vehicle tracking)
- **Deployment:** `reflex serve` with high `--max-batch-cost-ms` for throughput
- **Camera setup:** Multi-camera intersection feeds

---

## Checklist

- [ ] JSON config created with correct `schema_version: 1`
- [ ] `embodiment` slug matches the filename (minus `.json`)
- [ ] `action_space.dim` matches the number of `ranges` entries
- [ ] `normalization.mean` and `normalization.std` have length == `dim`
- [ ] `gripper.component_idx` is within `[0, dim)`
- [ ] Config placed in both `presets/` and `configs/embodiments/`
- [ ] Slug added to `schema.json` enum (or using `custom`)
- [ ] Tests pass: `pytest tests/test_embodiments.py`
- [ ] Doctor passes: `reflex doctor --embodiment <name>`

---

## Further Reading

- [Embodiment Schema Reference](./embodiment_schema.md)
- [CLI Command Reference](./cli_reference.md)
- [Getting Started](./getting_started.md)
