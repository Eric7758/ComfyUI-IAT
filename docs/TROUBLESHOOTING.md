# Troubleshooting (Production)

本页用于生产环境排障，优先提供：
- 简洁用户可读错误信息
- 可追溯错误码（`IAT:EXXXX`）+ trace id
- 最短恢复路径

## Error Format

运行时统一错误格式：

`[IAT:EXXXX][trace_id] message`

- `EXXXX`：错误类别
- `trace_id`：本次错误追踪 ID（用于日志定位）

## Runtime Behavior Baseline

- 仅支持官方原版模型（无量化/GGUF 分支）。
- 模型目录固定为 `ComfyUI/models/diffusion_models`。
- 下载顺序固定为 `ModelScope -> HuggingFace`。
- 运行时不会自动升级依赖。

## Error Codes

### `E1001` Unsupported model variant
- 现象：选择了不存在或拼写错误的 `model_variant`
- 处理：
1. 从节点下拉中重新选择，不要手填
2. 重启 ComfyUI 后再次尝试

### `E2001` Model download/validation failed
- 现象：模型下载或校验失败
- 处理：
1. 检查网络连通性与代理
2. 确认 `modelscope`、`huggingface_hub` 已安装
3. 检查 `ComfyUI/models/diffusion_models` 的目录权限和可写性
4. 结合 `trace_id` 查看同时间段日志细节

### `E2003` Download lock acquisition failed
- 现象：模型目录锁获取失败（系统锁异常）
- 处理：
1. 检查文件系统是否支持文件锁
2. 检查目录权限，确保当前用户可读写
3. 重启 ComfyUI 后重试

### `E2004` Download lock timeout
- 现象：等待模型下载锁超时，常见于并发流程同时下载同一模型
- 处理：
1. 等待正在进行的下载任务完成
2. 关闭重复工作流，保留一个下载任务
3. 如确认无下载进程，重启 ComfyUI 后重试

### `E5001` Transformers version/architecture mismatch
- 现象：`transformers` 版本过低，或不包含 `qwen3_5` 架构注册
- 处理：
1. 在 ComfyUI 当前 Python 环境执行：
```bash
python -m pip install --upgrade "transformers>=5.2.0"
```
2. 重启 ComfyUI 后重试

## Fast Checks

1. 依赖检查：
- `transformers>=5.2.0`
- `modelscope>=1.18.0`
- `huggingface_hub>=0.24.0`

2. 路径检查：
- 目标目录：`ComfyUI/models/diffusion_models`
- 确认目录存在且可写

3. 日志检查：
- 用错误里的 `trace_id` 在 ComfyUI 日志中检索对应详情
