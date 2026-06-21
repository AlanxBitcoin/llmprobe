# LLM Probe

LLM Probe is a local research tool for Llama-family models.
It focuses on hidden-state and neuron-level analysis, with both CLI and built-in UI workflows.

Single entrypoint:

- `src/main.py`

## What This Project Is For

- Run reproducible local studies over hidden states and logits.
- Perform neuron and layer intervention experiments.
- Reuse token hidden-state cache for faster repeated experiments.
- Execute studies from CLI or from the local Web UI.
- Keep the study system modular so new studies can be added continuously.

## Requirements

- Python 3.11+ (3.12 is commonly used)
- CUDA GPU (recommended)
- Local Hugging Face-compatible Llama model directory

Install dependencies:

```bash
pip install -r docs/requirements.txt
```

## Quick Start

1. Configure model and runtime options in `configs/custom.yaml`.
2. Start app (no subcommand):

```bash
python src/main.py --config configs/custom.yaml
```

Default UI URL:

- `http://127.0.0.1:8000`

When no subcommand is provided, startup behavior is controlled by config, including:

- UI server boot
- model/runtime warmup
- hidden-store disk initialization

## Important Config Keys

In `configs/custom.yaml`:

- `model.model_name_or_path`: local model path (required)
- `runtime.start_llama_api_on_boot`: warm model/runtime on boot
- `ui.enabled`, `ui.start_server_on_boot`, `ui.host`, `ui.port`
- `hidden_store.protocol`: `bos0_assistant0`, `bos1_assistant0`, or `bos1_assistant1`
- `hidden_store.init_on_main_boot`, `hidden_store.preload_on_boot`

## CLI Usage

Base format:

```bash
python src/main.py --config configs/custom.yaml <command> [args...]
```

Show all commands:

```bash
python src/main.py --help
```

Show command-specific arguments:

```bash
python src/main.py <command> --help
```

## Common Command Examples

Single-word hidden state:

```bash
python src/main.py --config configs/custom.yaml run-single-word-hidden-state apple
```

Top-K neuron intervention:

```bash
python src/main.py --config configs/custom.yaml run-single-word-top-100-neurons apple --top-k-neurons 100 --intervention-layer 30
```

Build token hidden-store cache:

```bash
python src/main.py --config configs/custom.yaml build-token-hidden-store --bos true --assistant false --limit 0 --start-token-id 0
```

## Study Inventory (Current)

The following are the currently registered `study` commands from `src/study/cli.py`:

- `run-single-word-hidden-state`
- `run-single-word-hidden-state-batch-average`
- `run-sentence-next-word`
- `run-token-diff`
- `run-one-on-one-attention`
- `run-chat-attention-word-replacement`
- `run-qk-params`
- `run-single-word-top-100-neurons`
- `run-layer-neuron-logits-table`
- `run-layer-ffn-neuron-logits-table`
- `run-layer-neurons`
- `run-layer-shortcut`
- `run-attribute-group-neurons`

This list is not fixed. New studies will be added over time as research needs evolve.

## Other CLI Commands (Current)

Batch and global analysis:

- `run-single-batch`
- `run-multi-batch`
- `run-global-analysis`
- `run-global-analysis-all`
- `run-color-words-experiment`

Word and sentence studies:

- `run-single-word`
- `run-single-word-hidden-state`
- `run-single-word-hidden-state-batch-average`
- `run-sentence-next-word`
- `run-token-diff`
- `run-one-on-one-attention`
- `run-chat-attention-word-replacement`
- `run-qk-params`
- `run-word-sum`
- `run-word-diff`
- `run-multi-word`

Neuron and layer intervention:

- `run-single-word-top-100-neurons`
- `run-layer-neuron-logits-table`
- `run-layer-ffn-neuron-logits-table`
- `run-layer-neurons`
- `run-layer-shortcut`
- `run-attribute-group-neurons`

Reports and probes:

- `run-dim-report`
- `run-word-family-report`
- `run-word-contrast-report`
- `run-probe`
- `run-attribute-probe`
- `predict-attributes`

Cache build:

- `build-token-hidden-store`

## Output and Cache

- Outputs: `data/outputs/`
- Hidden-store files: `data/cache/`
  - `hidden_states.<protocol>.f16.bin`
  - `hidden_states.<protocol>.done.bin`

The `data/` directory is intended for generated runtime artifacts and is usually ignored by git.

## Screenshots

Recommended location:

- `docs/images/`

Example:

```md
![UI Home](docs/images/ui-home.png)
![Study Result](docs/images/study-result.png)
```

Tips:

- Keep image names lowercase with hyphens, for example `ui-home.png`.
- Prefer PNG for charts/plots and JPEG for large photos.
- If a screenshot is very large, compress before commit to keep repo size manageable.

## Source Layout

- `src/main.py`: app and CLI entrypoint
- `src/study/`: study implementations and CLI registration
- `src/probes/`: probe logic
- `src/utils/`: hidden extraction, cache/store, utilities
- `src/ui/`: routes, registry/forms, static frontend assets

## Notes

- If UI actions do not refresh after code changes, restart the process.
- Startup time and GPU memory usage depend on model size, quantization, and hardware setup.
