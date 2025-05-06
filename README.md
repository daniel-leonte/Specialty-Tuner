# SpecialtyTuner – Compact ML Code Generation Assistant

SpecialtyTuner is a lightweight *instruction-tuned* language model that writes **production-ready Python code for common machine-learning and data-science tasks**.  
The project shows how far you can push a modern open-source LLM by combining:

* CodeLlama-7b-hf as a base model (≈7 B parameters)
* Parameter-efficient **LoRA** fine-tuning (rank = 16) on hand-curated ML examples
* 4-bit / 8-bit quantisation for memory-friendly inference

---

## ✨ Key Features

|      | What you get |
|------|--------------|
| ⚡ **Fast offline inference** | Quantised weights + small LoRA adapter (≈30 MB) |
| 🪄 **One-shot code generation** | CLI returns a fully formatted `black`-compliant script for your prompt (`pandas`, `scikit-learn`, `torch`, …) |
| 🔍 **Quality gates** | Built-in evaluation script checks syntax, execution and *pass@k* unit tests |
| 🛠 **End-to-end pipeline** | `collect_data → clean_data → train → eval → infer` – all reproducible with plain Python |
| 📦 **Self-contained repo** | No hidden services – everything runs locally with open-source libraries |
| 🧰 **Modular architecture** | Shared utilities module for consistent device handling, code extraction and processing |

---

## Quick Start

```bash
# 1. Clone & install
$ git clone <your-repo-url> specialtytuner && cd specialtytuner
$ python3 -m venv .venv && source .venv/bin/activate
$ pip install -r requirements.txt

# 2. (Optional) download the released LoRA checkpoint
$ wget -O checkpoints/best/adapter_model.bin <link-to-checkpoint>

# 3. Generate code ✨
$ python scripts/infer.py \
        --prompt "Load a CSV with pandas, drop NA rows, then train a RandomForestClassifier" \
        --temperature 0.2 --top_p 0.95 --streaming
```

The assistant prints a fully formatted script enclosed in markdown fences:

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# ...
```

---

## Repository Layout

```
.
├── data/              # train / val / test JSONL after cleaning
├── scripts/           # all pipeline steps
│   ├── collect_data.py   # fetch & merge open datasets
│   ├── clean_data.py     # format, lint, quality-score
│   ├── train.py          # LoRA fine-tuning
│   ├── eval.py           # syntax / exec / unit-test evaluation
│   ├── infer.py          # CLI inference
│   ├── generate_demo.py  # build nice markdown demo gallery
│   └── utils.py          # shared utilities for device detection and code handling
├── checkpoints/       # tiny LoRA adapter (generated after training)
├── demo_outputs.md    # example generations
├── requirements.txt   # python deps
└── README.md          # you are here
```

---

## 🔧 Training Pipeline

1. **Collect** open-source tasks (DS-1000, HumanEval, CodeAlpaca subset):
   ```bash
   python scripts/collect_data.py
   ```
2. **Clean & score** the raw snippets → `train.jsonl / val.jsonl / test.jsonl`:
   ```bash
   python scripts/clean_data.py
   ```
3. **LoRA Fine-tune** on 7-B base model (4-bit CUDA or 8-bit CPU):
   ```bash
   python scripts/train.py
   ```
4. **Evaluate** on held-out tasks (20 examples by default):
   ```bash
   python scripts/eval.py
   ```

---

## 📊 Current Results (20 hold-out prompts)

| Metric | Score |
|--------|-------|
| Syntax correctness | **95 %** |
| Execution success  | 5 % |
| pass@5 unit tests  | 0 % |

> Tip: Most failures come from missing runtime dependencies – improving the import heuristic is next on the roadmap.

---

## In-depth Usage

### Inference flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--temperature` | 0.2 | Sampling temperature |
| `--top_p`       | 0.95 | Nucleus sampling cut-off |
| `--max_length`  | 800  | Max tokens in generation |
| `--streaming`   | off  | Stream tokens as they arrive |

### Generate a demo gallery

```bash
python scripts/generate_demo.py --output_file demo_outputs.md
```

### Run evaluation on more examples

```bash
python scripts/eval.py   # uses data/test.jsonl
```

---

## Code Architecture

The codebase follows software engineering best practices:

* **DRY Principle**: Common functionality extracted to `utils.py` to eliminate code duplication
* **Hardware Adaptability**: Automatic detection and configuration for CUDA, MPS (Apple Silicon), or CPU environments
* **Consistent Formatting**: Standardized code formatting via Black
* **Error Handling**: Robust error handling throughout the pipeline
* **Documentation**: Clear docstrings with type hints for better maintainability

---

## Limitations & Future Work

* The base model has no internet or file-system awareness; paths and URLs are placeholders.
* Execution success is still low – integrating an auto-fix loop is planned.
* Only single-file scripts are produced; multi-module projects are out-of-scope for now.
* The dataset is relatively small (~600 curated examples) – scaling the corpus should unlock further quality gains.

---

## License

This repository is released under the MIT License.  
The fine-tuned weights inherit the original CodeLlama licence.

---

## Acknowledgements

* [CodeLlama](https://github.com/facebookresearch/codellama) for the stellar base model
* HuggingFace 🤗 – *transformers*, *datasets*, *peft*, *accelerate*
* [DS-1000](https://huggingface.co/datasets/xlangai/DS-1000), HumanEval & CodeAlpaca datasets
* The open-source community for tools like *black*, *bitsandbytes* and *pytest*