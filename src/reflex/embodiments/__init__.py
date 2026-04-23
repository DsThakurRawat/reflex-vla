"""Per-embodiment configs (Franka, SO-100, UR5, Trossen, Stretch, custom).

Read by `reflex serve --embodiment <name>` so the runtime knows the robot's
action space, normalization stats, gripper layout, control rate, and safety
constraints. Designed to be loaded once at server startup, passed to the
RTC adapter (B.3), action denormalization (B.6), and reflex doctor (D.1).

Pattern mirrors `src/reflex/config.py:HARDWARE_PROFILES` — module-level
registry + frozen dataclass + getter that raises a descriptive error.

Schema is at `schema.json` next to this file. Preset JSON files are at
`<repo>/configs/embodiments/{franka,so100,ur5}.json` (NOT in the package —
they're user-editable + ship in the repo, not in the wheel).

Plan: features/01_serve/subfeatures/_rtc_a2c2/per-embodiment-configs_plan.md
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Repo-relative path to preset JSONs. Relative to the repo root, NOT this
# file (the JSONs live under configs/embodiments/, outside the package).
_REPO_ROOT = Path(__file__).resolve().parents[3]  # src/reflex/embodiments/__init__.py → repo
_PRESETS_DIR = _REPO_ROOT / "configs" / "embodiments"

# Path to the JSON schema file (lives inside the package).
_SCHEMA_PATH = Path(__file__).parent / "schema.json"


@dataclass(frozen=True)
class EmbodimentConfig:
    """A robot's per-embodiment config. Frozen — load once, pass around safely."""

    schema_version: int
    embodiment: str
    action_space: dict[str, Any]
    normalization: dict[str, list[float]]
    gripper: dict[str, Any]
    cameras: list[dict[str, Any]]
    control: dict[str, float | int]
    constraints: dict[str, Any]

    # Where this config came from (for debugging + audit trail). Not part
    # of the schema; populated by the loader.
    _source_path: str = field(default="")

    @classmethod
    def from_dict(cls, d: dict[str, Any], source_path: str = "") -> EmbodimentConfig:
        """Construct from a parsed JSON dict. Doesn't validate — call validate()
        separately if you want to know if it's well-formed."""
        return cls(
            schema_version=d["schema_version"],
            embodiment=d["embodiment"],
            action_space=d["action_space"],
            normalization=d["normalization"],
            gripper=d["gripper"],
            cameras=d["cameras"],
            control=d["control"],
            constraints=d["constraints"],
            _source_path=source_path,
        )

    @classmethod
    def load_preset(cls, name: str) -> EmbodimentConfig:
        """Load a shipped preset by name. Raises ValueError if unknown."""
        path = _PRESETS_DIR / f"{name}.json"
        if not path.exists():
            available = sorted(p.stem for p in _PRESETS_DIR.glob("*.json"))
            raise ValueError(
                f"Unknown embodiment preset '{name}'. "
                f"Available: {available or '(none — run scripts/emit_embodiment_presets.py)'}"
            )
        return cls.load_custom(str(path))

    @classmethod
    def load_custom(cls, path: str) -> EmbodimentConfig:
        """Load from an arbitrary JSON file path."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Embodiment config not found: {path}")
        with p.open() as f:
            data = json.load(f)
        return cls.from_dict(data, source_path=str(p))

    def to_dict(self) -> dict[str, Any]:
        """Serialize back to a dict matching the schema (drops _source_path)."""
        return {
            "schema_version": self.schema_version,
            "embodiment": self.embodiment,
            "action_space": self.action_space,
            "normalization": self.normalization,
            "gripper": self.gripper,
            "cameras": self.cameras,
            "control": self.control,
            "constraints": self.constraints,
        }

    @property
    def action_dim(self) -> int:
        """Convenience accessor — number of action dimensions."""
        return int(self.action_space["dim"])

    @property
    def state_dim(self) -> int:
        """Convenience accessor — number of state dimensions
        (inferred from mean_state)."""
        return len(self.normalization["mean_state"])

    @property
    def gripper_idx(self) -> int:
        return int(self.gripper["component_idx"])


def list_presets() -> list[str]:
    """Return slugs of all shipped presets (alphabetical)."""
    if not _PRESETS_DIR.exists():
        return []
    return sorted(p.stem for p in _PRESETS_DIR.glob("*.json"))


def get_schema_path() -> Path:
    """Path to the embodiment JSON schema (for jsonschema validation +
    VSCode hookup)."""
    return _SCHEMA_PATH


__all__ = [
    "EmbodimentConfig",
    "list_presets",
    "get_schema_path",
]
