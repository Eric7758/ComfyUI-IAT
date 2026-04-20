"""ComfyUI-IAT 插件入口。

职责：
1. 读取 `config.yaml`。
2. 动态加载 `py/nodes` 下的节点模块。
3. 汇总导出 ComfyUI 需要的 `NODE_*_MAPPINGS`。
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Iterable

__version__ = "2.0.0"
WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY", "__version__"]

cwd = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cwd, "config.yaml")


def _load_config(path: str) -> dict:
    """读取配置文件并返回字典；失败时返回空字典。"""
    config: dict = {}
    try:
        import yaml  # type: ignore

        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            if not isinstance(config, dict):
                print(f"[IAT] WARN: Ignoring non-dict config file: {os.path.basename(path)}")
                return {}
    except Exception as e:
        print(f"[IAT] WARN: Failed to load config.yaml: {e}")
        return {}
    return config


def _iter_node_modules(nodes_root: str) -> Iterable[str]:
    """扫描 `py/nodes`，返回可导入模块路径。"""
    if not os.path.isdir(nodes_root):
        raise FileNotFoundError(nodes_root)
    for file in sorted(os.listdir(nodes_root)):
        if file.endswith(".py") and file != "__init__.py":
            yield f".py.nodes.{file[:-3]}"


def _register_nodes(module_paths: Iterable[str], verbose: bool) -> None:
    """导入节点模块并合并映射表。"""
    for module_path in module_paths:
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


config = _load_config(config_path)
verbose = bool((config.get("logging") or {}).get("verbose", False))

# Expose config to node modules.
if "comfyui_iat_config" not in sys.modules:
    sys.modules["comfyui_iat_config"] = type(
        "Config",
        (),
        {
            "data": config,
            "path": config_path,
        },
    )()

# 节点实现固定放在 `py/`，避免与 ComfyUI 内置 `nodes` 模块冲突。
nodes_dir = os.path.join(cwd, "py", "nodes")
try:
    node_modules = list(_iter_node_modules(nodes_dir))
except Exception as e:
    print(f"[IAT] ERROR: Failed to scan py/nodes directory: {e}")
    node_modules = []

_register_nodes(node_modules, verbose)

node_count = len(NODE_CLASS_MAPPINGS)
if node_count > 0:
    print(f"\033[34m[IAT]\033[0m v{__version__} \033[92mLoaded {node_count} nodes\033[0m")
else:
    print(f"\033[31m[IAT]\033[0m v{__version__} \033[93mNo nodes loaded\033[0m")
