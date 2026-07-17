Production datasets must live below the external `datasets.root` configured in
`config.yaml`. This directory documents the canonical layout only; JSON files
containing a standalone `captions` array are not discovered or loaded.

Rules:
- One dataset per directory.
- Put `dataset.json` at the dataset root.
- Put paired files under `images/`: `0001.png` + `0001.txt`.
- Missing pairs are skipped and reported by the node diagnostics.
- The index cache is rebuilt automatically when files change.

Directory schema:

```text
dataset_A/
├── dataset.json
└── images/
    ├── 0001.png
    ├── 0001.txt
    ├── 0002.png
    └── 0002.txt
```

`dataset.json` schema:

```json
{
  "dataset_name": "CMF_ChampagneSilver",
  "version": "1.0",
  "trigger_words": ["champagne silver", "satin metallic"],
  "base_model": "Flux.2 Klein 9B",
  "lora_name": "model_A",
  "language": "zh"
}
```
