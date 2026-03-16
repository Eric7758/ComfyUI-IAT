"""
ComfyUI-IAT Plugin
Version: 1.0.0
Author: Eric7758
Description: Image & Text utilities with Qwen translator and prompt optimizer.
"""

from __future__ import annotations

import importlib
import os
import sys

__version__ = "1.0.0"

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]

cwd = os.path.dirname(os.path.realpath(__file__))

config: dict = {}
config_path = os.path.join(cwd, "config.yaml")
if os.path.isfile(config_path):
    try:
        import yaml  # type: ignore

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[IAT] WARN: Failed to load config.yaml: {e}")

verbose = bool((config.get("logging") or {}).get("verbose", False))

# Expose config to node modules.
if "comfyui_iat_config" not in sys.modules:
    sys.modules["comfyui_iat_config"] = type("Config", (), {"data": config})()

# Keep implementation under `py/` to avoid colliding with ComfyUI's own `nodes` module.
nodes_dir = os.path.join(cwd, "py", "nodes")
node_modules: list[str] = []
try:
    if not os.path.isdir(nodes_dir):
        raise FileNotFoundError(nodes_dir)
    for file in sorted(os.listdir(nodes_dir)):
        if file.endswith(".py") and file != "__init__.py":
            module_name = file[:-3]
            node_modules.append(f".py.nodes.{module_name}")
except Exception as e:
    print(f"[IAT] ERROR: Failed to scan py/nodes directory: {e}")

for module_path in node_modules:
    try:
        module = importlib.import_module(module_path, package=__name__)
        if hasattr(module, "NODE_CLASS_MAPPINGS"):
            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
        if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
        if verbose:
            print(f"[IAT] Loaded module: {module_path}")
    except Exception as e:
        print(f"[IAT] ERROR: Failed to load {module_path}: {e}")

node_count = len(NODE_CLASS_MAPPINGS)
if node_count > 0:
    print(f"\033[34m[IAT]\033[0m v{__version__} \033[92mLoaded {node_count} nodes\033[0m")
else:
    print(f"\033[31m[IAT]\033[0m v{__version__} \033[93mNo nodes loaded\033[0m")
