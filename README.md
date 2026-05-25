# LLM Probe (Current Version)

`probe` is a local analysis tool for inspecting hidden states of a Llama-family model, running study/probe experiments, and viewing results in a built-in UI.

This README matches the current codebase (`src/main.py` as the single entry).

## 1) What This Project Does

- Runs local model-based studies and probes.
- Supports a **hidden state store** (`data/cache/*.bin`) for token-level read-through cache.
- Provides a local UI (default `http://127.0.0.1:8000`) for study execution and visualization.
- Includes two single-word hidden-state study flows:
  - `run-single-word-hidden-state`
  - `run-single-word-top-100-neurons` (with configurable neuron top-k and intervention layer)

## 2) Project Entry

- Main entry: `src/main.py`
- Default config: `configs/custom.yaml`

Common usage:

```bash
python src/main.py --config configs/custom.yaml
```

No subcommand starts the app server flow (UI + runtime behavior controlled by config).

## 3) Environment

Recommended:

- Python 3.11+ (current setup uses Python 3.12)
- CUDA GPU
- Local Hugging Face-compatible Llama model directory

Install dependencies from:

```bash
pip install -r docs/requirements.txt
```

## 4) Key Config Points

In `configs/custom.yaml`:

- `model.model_name_or_path`: local model path (required)
- `runtime.start_llama_api_on_boot`: whether to warm model on app boot
- `ui.enabled`, `ui.start_server_on_boot`, `ui.host`, `ui.port`
- `hidden_store.protocol`: one of:
  - `bos0_assistant0`
  - `bos1_assistant0` (default)
  - `bos1_assistant1`
- `hidden_store.init_on_main_boot`, `hidden_store.preload_on_boot`

## 5) Run UI

```bash
python src/main.py --config configs/custom.yaml
```

Then open:

- `http://127.0.0.1:8000`

## 6) Core CLI Commands

### Single-word hidden state

```bash
python src/main.py --config configs/custom.yaml run-single-word-hidden-state apple
```

### Single-word top-100 neuron intervention

```bash
python src/main.py --config configs/custom.yaml run-single-word-top-100-neurons apple --top-k-neurons 100 --intervention-layer 30
```

### Build hidden-store cache (protocol build)

```bash
python src/main.py --config configs/custom.yaml build-token-hidden-store --bos true --assistant false --limit 0 --start-token-id 0
```

### Other study/probe commands

- `run-single-word`
- `run-single-batch`
- `run-multi-batch`
- `run-global-analysis`
- `run-global-analysis-all`
- `run-color-words-experiment`
- `run-dim-report`
- `run-word-family-report`
- `run-word-contrast-report`
- `run-probe`
- `run-attribute-probe`
- `predict-attributes`

Use command help:

```bash
python src/main.py --help
python src/main.py run-single-word-top-100-neurons --help
```

## 7) Single-Word Study Flow (Current)

For hidden-state studies, runtime flow is:

1. UI triggers study action.
2. Study layer calls probe layer.
3. Probe reads hidden-state store first.
4. On cache miss, probe/runtime fallback runs model and writes back.
5. Study composes returned payload for UI:
   - heatmap matrix
   - top logits table(s)
   - metadata / task hints

For `run-single-word-top-100-neurons`, extra parameters are included in the response:

- `top_k_neurons`
- `intervention_layer`

## 8) Output and Cache

- General outputs: `data/outputs/`
- Hidden-store files: `data/cache/`
  - `hidden_states.<protocol>.f16.bin`
  - `hidden_states.<protocol>.done.bin`

The `data/` directory is intended for generated runtime artifacts and should typically stay ignored in git.

## 9) Source Structure (Practical)

- `src/main.py`: CLI/app entry
- `src/study/`: study orchestration
- `src/probes/`: probe-layer logic
- `src/utils/`: hooks, hidden extraction, logits, cache/store utilities
- `src/ui/`: routes, registry, forms, static frontend

## 10) Notes

- If UI does not show newly added actions after code updates, restart app (old process may still be serving old action registry).
- Model loading speed and memory behavior depend on quantization and your GPU setup.
