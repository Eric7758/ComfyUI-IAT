"""
ComfyUI-IAT Plugin
Version: 1.0.0
Author: Eric7758
Description: Image & Text utilities with Qwen translator and prompt optimizer.
"""

import os
import importlib
import sys

# 插件元信息
__version__ = "1.0.0"

# 初始化映射
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# 获取当前插件目录
cwd = os.path.dirname(os.path.realpath(__file__))

# ==============================
# 1. 加载配置
# ==============================
config = {}
config_path = os.path.join(cwd, "config.yaml")
if os.path.isfile(config_path):
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[IAT] WARN: Failed to load config.yaml: {e}")

# ==============================
# 2. 动态加载 nodes/ 目录下的所有模块
# ==============================
node_modules = []

nodes_dir = os.path.join(cwd, "nodes")
try:
    for file in os.listdir(nodes_dir):
        if file.endswith(".py") and file != "__init__.py":
            module_name = file[:-3]  # 去掉 .py
            node_modules.append(f".nodes.{module_name}")
except Exception as e:
    print(f"[IAT] ERROR: Failed to scan nodes/ directory: {e}")

# ==============================
# 3. 导入并合并节点
# ==============================
for module_path in node_modules:
    try:
        module = importlib.import_module(module_path, package=__name__)
        if hasattr(module, "NODE_CLASS_MAPPINGS"):
            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
        if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
        print(f"[IAT] Loaded module: {module_path}")
    except Exception as e:
        print(f"[IAT] ERROR: Failed to load {module_path}: {e}")

# ==============================
# 4. 输出加载结果
# ==============================
node_count = len(NODE_CLASS_MAPPINGS)
if node_count > 0:
    print(f"\033[34m[IAT]\033[0m v{__version__} \033[92mLoaded {node_count} nodes\033[0m")
else:
    print(f"\033[31m[IAT]\033[0m v{__version__} \033[93mNo nodes loaded\033[0m")

# 暴露配置给节点
if "comfyui_iat_config" not in sys.modules:
    sys.modules["comfyui_iat_config"] = type("Config", (), {"data": config})()
