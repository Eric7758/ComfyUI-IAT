# Update Notes

## 2026-04-20 (v2.0.0)

### 1) 版本号更新
- 插件版本更新到 `2.0.0`。
- 默认 `user_agent` 更新为 `ComfyUI-IAT/2.0`。

### 2) Qwen3.5 运行时策略收敛（仅官方原版）
- 移除量化与 GGUF 路径，文本与视觉节点统一走官方原版模型分支。
- 运行时不再自动升级依赖，`transformers` 不满足要求时返回 `E5001` 并提示手动修复命令。
- 模型下载目录固定为 `ComfyUI/models/diffusion_models`。
- 下载顺序统一为 `ModelScope -> HuggingFace`。

### 3) 下载稳定性增强
- 保留并沿用下载锁、状态文件、分片完整性校验与缺失补齐逻辑。
- 使用 `resume_download=True` 进行断点续传，降低中断下载的重复成本。

### 4) 文档与安装器一致性修正
- README / INSTALL / USAGE / TROUBLESHOOTING 全量同步到当前真实行为。
- `install.py` 改为仅做依赖安装与兼容性检查，不再自动拉取 transformers 源码包。
- 补充 v2.0.0 GitHub Release 模板：`.github/release-v2.0.0.md`。

## 2026-04-17 (v1.22.0)

### 1) 版本号更新
- 插件版本更新到 `1.22.0`。
- 默认 `user_agent` 更新为 `ComfyUI-IAT/1.22`。

### 2) Qwen3.5 兼容基线修正
- 依赖基线统一为 `transformers>=5.2.0`。
- 运行时最低版本检查同步更新为 `5.2.0`。
- 文档中的 Transformers 版本要求已同步修正。

## 2026-04-17 (v1.21.0)

### 1) 版本号更新
- 插件版本更新到 `1.21.0`。
- 默认 `user_agent` 更新为 `ComfyUI-IAT/1.21`。

### 2) 依赖声明调整
- 恢复基础依赖声明：`torch`、`numpy`、`Pillow`。
- 保留并强调 `transformers>=5.2.0`，降低 `qwen3_5` 架构不兼容风险。

## 2026-04-17 (v1.2.0)

### 1) 插件结构精简
- 重构根入口加载逻辑，拆分配置读取与节点注册函数，便于维护。
- 清理废弃空壳模块 `py/nodes/llm_nodes.py`。
- 补充中文注释与说明，提升可读性。

### 2) 依赖与发布信息统一
- `requirements.txt` 与 `pyproject.toml` 依赖保持一致。
- 默认 `user_agent` 更新为 `ComfyUI-IAT/1.2`。

### 3) 测试节点隔离
- 从 IAT 插件移除 XYZ 测试节点文件，避免混入 IAT 节点组。
- XYZ 测试节点独立放在 `ComfyUI-XYZ_test`，分类与 IAT 分组隔离。

## 2026-03-24

### 1) LLM 全面重构为 Qwen3.5 统一运行时
- 新增统一运行时模块：`py/nodes/qwen35_runtime.py`
- 所有 LLM 节点统一走该运行时：模型解析、自动下载、缓存、推理统一管理
- 旧 `llm_nodes.py` 改为兼容 shim，避免旧加载逻辑继续生效

### 2) 节点统一与兼容
- 保留并兼容以下节点名：
  - `QwenTranslator by IAT`
  - `QwenKontextTranslator by IAT`
  - `Qwen35PromptEnhancer by IAT`
  - `Qwen35ReversePrompt by IAT`
- 简单工具节点（输入/图像类）保持不变

### 3) 模型策略与自动下载
- 统一模型家族：Qwen3.5
- 默认模型变体：`Qwen3.5-Latest`
- 支持最小模型测试变体：`Qwen3.5-0.8B`
- 模型下载目录：`ComfyUI/models/LLM`
- 下载源回退策略：ModelScope -> HuggingFace

### 4) 关键修复
- 修复模型文件检测 bug：避免将空目录误判为有效模型
- 增加 tokenizer 回退加载策略，提升不同 Qwen 检查点兼容性
- 优化 0.8B 候选顺序，优先可用仓库

### 5) 配置与依赖更新
- `config.yaml` 改为统一配置项：
  - `model.default_variant`
  - `model.quantization`
  - `model.device`
- `requirements.txt` 新增：`huggingface_hub>=0.24.0`

### 6) 已完成验证
- WSL + Conda + ComfyUI 真实环境加载测试通过
- GPU 启动测试通过（RTX 4090, 端口 8100）
- `Qwen3.5-0.8B` 端到端翻译推理测试通过
