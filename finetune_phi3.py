import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset
import json

# Load and format data in one go
data = [{"text": f"<|user|>\n{json.loads(line)['prompt']}<|end|>\n<|assistant|>\n```python\n{json.loads(line)['code']}\n```<|end|>"}
        for line in open("ml_code_dataset.jsonl")]

# Model setup - no unnecessary variables
model_name = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    ),
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16
)

# Apply LoRA - minimal config
model = get_peft_model(
    prepare_model_for_kbit_training(model),
    LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.1, task_type="CAUSAL_LM")
)

# Train it
SFTTrainer(
    model=model,
    train_dataset=Dataset.from_list(data),
    tokenizer=tokenizer,
    args=TrainingArguments(
        output_dir="./phi3_ml_finetuned",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_steps=500,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        warmup_steps=50,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to=None
    ),
    max_seq_length=1024,
    dataset_text_field="text"
).train()

print("done.") 