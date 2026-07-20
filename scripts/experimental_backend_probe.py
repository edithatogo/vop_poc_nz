#!/usr/bin/env python3
"""Exercise the opt-in JAX backend and emit durable, deterministic CI evidence."""

from __future__ import annotations

import argparse
import json
from importlib import import_module
from pathlib import Path
from typing import Any


def build_evidence() -> dict[str, Any]:
    """Compile and execute a bounded x64 JAX calculation on the selected device."""
    jax = import_module("jax")
    jnp = import_module("jax.numpy")
    jax.config.update("jax_enable_x64", True)
    values = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)
    squared_sum = jax.jit(lambda array: jnp.sum(jnp.square(array)))(values)
    device = values.devices().pop()
    return {
        "backend": device.platform,
        "device_kind": device.device_kind,
        "jax_version": jax.__version__,
        "result": float(squared_sum),
        "x64_enabled": bool(jax.config.x64_enabled),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = build_evidence()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    main()
