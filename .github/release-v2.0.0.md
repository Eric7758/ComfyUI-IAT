# ComfyUI-IAT v2.0.0

## Highlights

- Runtime strategy is now official-model only.
- Quantization and GGUF runtime branches are removed.
- Model storage path is fixed to `ComfyUI/models/diffusion_models`.
- Model download order is fixed to `ModelScope -> HuggingFace`.
- Runtime dependency auto-upgrade is removed; dependency mismatch now returns `E5001` with a manual fix command.

## Reliability Improvements

- Download lock mechanism for concurrent model downloads.
- Persistent model state file (`.iat.model_state.json`).
- Shard integrity validation with missing-shard recovery.
- Resume download enabled to reduce restart cost.

## Error Handling

- Unified production error format: `[IAT:EXXXX][trace_id] message`.
- Error codes aligned with runtime behavior:
  - `E1001`: Unsupported model variant
  - `E2001`: Model download/validation failed
  - `E2003`: Download lock acquisition failed
  - `E2004`: Download lock timeout
  - `E5001`: Transformers version/architecture mismatch

## Upgrade Notes

- This is a behavior-changing release.
- If you previously used quantized or GGUF variants, switch to official variants in node dropdowns.
- If startup or runtime reports `E5001`, run:

```bash
python -m pip install --upgrade "transformers>=5.2.0"
```

## Docs

- README / INSTALL / USAGE / TROUBLESHOOTING have been synchronized to v2.0 behavior.
- Full changelog: `UPDATE.md`
