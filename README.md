# Specialty-Tuner

Fine-tuned Phi-3 for ML code generation. Uses LoRA, works on 16GB VRAM.

## Install

```bash
git clone <repo>
cd Specialty-Tuner
python -m venv codegen_env && source codegen_env/bin/activate
pip install -r requirements.txt
```

## Use

```bash
python finetune_phi3.py  # train
python inference.py      # test
python app.py           # web ui at localhost:7860
```

## Files

- `finetune_phi3.py` - training script
- `inference.py` - inference utils  
- `app.py` - gradio web interface
- `ml_code_dataset.jsonl` - training data
- `requirements.txt` - deps

## Config

**Model**: microsoft/Phi-3-mini-4k-instruct + LoRA (r=16, alpha=32)  
**Training**: 4 batch size, 2e-4 lr, 500 steps, fp16  
**Hardware**: 16GB VRAM minimum

## Deploy

`python app.py` or upload to HF Spaces. 