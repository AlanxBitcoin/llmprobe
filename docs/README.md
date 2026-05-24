# LLM Probe MVP

This project analyzes hidden states from a local `Llama 3 8B Instruct` model for two workflows:

- Single-word layer-8 concept probing
- Multi-word all-layer activation flow visualization

It also supports batch processing from a 100-word text file and exports frame sequences plus synthesized videos.

## Docs

- Usage guide: [USAGE.md](D:\vbcode2026\LLM探针\docs\USAGE.md)
- Prompt variants: [prompt_variants.md](D:\vbcode2026\LLM探针\docs\prompt_variants.md)

## Features

- Load a local Llama-family model with `transformers`
- Run hidden-state extraction for bare English words
- Match activations against predefined concept type/value vocabularies
- Train a simple linear probe on the labeled 100-word set
- Save plots and structured JSON summaries
- Batch through a word list in single-word mode or grouped multi-word mode
- Export MP4 videos from generated frames

## Quick Start

1. Install Python dependencies:

```powershell
pip install -r requirements.txt
```

2. Update [`configs/default.yaml`](D:\vbcode2026\LLM探针\configs\default.yaml) with a Hugging Face compatible local model directory if needed.

3. Run single-word batch analysis:

```powershell
python main.py run-single-batch
```

4. Run multi-word batch analysis with group size 3:

```powershell
python main.py run-multi-batch --batch-size 3
```

5. Run global analysis for every discovered word list under `data/*.txt`:

```powershell
python main.py run-global-analysis-all
```

6. Train a linear probe on layer-8 word representations:

```powershell
python main.py run-probe
```

7. Train attribute-family probes:

```powershell
python main.py run-attribute-probe
```

8. Predict attributes for one word:

```powershell
python main.py predict-attributes apple
```

## Notes

- The first version uses concept-space similarity as an interpretable approximation.
- `C:\Users\qiang\.ollama\models` is usually an Ollama cache layout, not a direct `transformers` model folder. If loading fails, point the config at a Hugging Face style local snapshot instead.
- If 4-bit loading is unavailable in your environment, switch to `load_in_8bit: true` or full precision in the config.
- Video export uses generated frames rather than desktop capture.
