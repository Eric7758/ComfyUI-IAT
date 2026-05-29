`datasets/prompts_kb/` stores local prompt knowledge-base JSON files for the dataset-backed RAG reverse prompt node.

Rules:
- One dataset per JSON file.
- File name: `[dataset_name].json` is recommended.
- Changes require a ComfyUI restart to refresh dataset dropdown options.
- Keep only validated prompt datasets here.

Schema:

```json
{
  "dataset_name": "CMF_ChampagneSilver",
  "version": "1.0",
  "trigger_words": ["champagne silver", "satin metallic"],
  "captions": [
    "product render, champagne silver finish, satin metallic surface, soft studio reflection, three quarter view",
    "industrial design close-up, curved aluminum shell, premium champagne silver coating, clean background"
  ]
}
```
