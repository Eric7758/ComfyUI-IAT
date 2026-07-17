Production datasets must live below the external `datasets.root` configured in
`config.yaml`. This directory documents the canonical layout only; JSON files
containing a standalone `captions` array are not discovered or loaded.

Rules:
- One dataset per directory.
- Put `dataset.json` at the dataset root.
- A multi-view sample uses the same filename stem across `control1`, `control2`,
  `control3`, and `result` directories. The caption is `result/<stem>.txt`.
- Directory names may be `dataset_A_control1` / `dataset_A_result` or nested
  `images/control1` / `images/result`.
- A missing control image is reported as a warning; a missing result image or
  result caption skips that sample.
- The index cache is rebuilt automatically when any image or caption changes.

Single-image schema:

```text
dataset_A/
├── dataset.json
└── images/
    ├── 0001.png
    ├── 0001.txt
    ├── 0002.png
    └── 0002.txt
```

Multi-view schema (maximum four images per sample):

```text
dataset_A/
├── dataset.json
├── dataset_A_control1/
│   ├── 000000.jpg
│   └── 000b00.jpg
├── dataset_A_control2/
│   ├── 000000.jpg
│   └── 000b00.jpg
├── dataset_A_control3/
│   └── 000000.jpg
└── dataset_A_result/
    ├── 000000.png
    ├── 000000.txt
    ├── 000b00.png
    └── 000b00.txt
```

`dataset.json` schema:

```json
{
  "dataset_name": "CMF_ChampagneSilver",
  "version": "1.0",
  "trigger_words": ["champagne silver", "satin metallic"],
  "base_model": "Flux.2 Klein 9B",
  "lora_name": "model_A",
  "language": "zh",
  "image_roles": ["control1", "control2", "control3", "result"],
  "caption_role": "result"
}
```

`image_roles` and `caption_role` are optional. When omitted, the node infers
the four roles from directory names and uses `result` captions.
