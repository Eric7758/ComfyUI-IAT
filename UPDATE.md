# Update Notes

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
