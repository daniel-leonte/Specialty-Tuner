import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from utils import get_device_config

def main():
    # Define model name
    model_name = "codellama/CodeLlama-7b-hf"
    
    # Check for GPU/MPS availability
    device, torch_dtype, use_fp16, device_name = get_device_config()
    print(f"Using {device_name}")
    
    # Load model with optimized settings
    print("Loading base model...")
    if torch.cuda.is_available():
        # Use quantization for CUDA
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            load_in_4bit=True,
            device_map="auto"
        )
    else:
        # Use regular loading for MPS/CPU but with 8-bit if possible
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                load_in_8bit=True,
                device_map="auto"
            )
            print("Using 8-bit quantization")
        except:
            print("8-bit quantization not available, using standard precision")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype
            )
            model = model.to(device)
    
    # Set up LoRA configuration with improved parameters
    print("Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=16,  # Increased rank for better expressivity
        lora_alpha=32,
        # Target more modules for better fine-tuning
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA adapters
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Check data files
    if not os.path.exists("data/train.jsonl") or not os.path.exists("data/val.jsonl"):
        raise FileNotFoundError("Data files not found. Run collect_data.py and clean_data.py first.")
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("json", data_files={
        "train": "data/train.jsonl",
        "validation": "data/val.jsonl"
    })
    
    # Format dataset with prompts
    def create_prompt(example):
        prompt = f"Write Python ML code for: {example['instruction']}\n\n```python\n{example['response']}\n```"
        return {"text": prompt}
    
    print("Processing dataset...")
    formatted_dataset = dataset.map(
        create_prompt,
        remove_columns=["instruction", "response"]
    )
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=1024
        )
    
    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # Set up training arguments
    print("Setting up trainer...")
    training_args = TrainingArguments(
        output_dir="checkpoints",
        per_device_train_batch_size=2,  # Reduced batch size for larger model
        per_device_eval_batch_size=2,
        eval_strategy="epoch",
        logging_dir="logs",
        logging_steps=10,
        learning_rate=2e-5,
        num_train_epochs=5,  # Increased epochs
        weight_decay=0.01,
        fp16=use_fp16,
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        save_total_limit=2,
        warmup_ratio=0.1,  # Added warmup for better training stability
        gradient_accumulation_steps=4,  # Added gradient accumulation to handle larger model
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    # Save model
    print("Saving model...")
    os.makedirs("checkpoints/best", exist_ok=True)
    model.save_pretrained("checkpoints/best")
    tokenizer.save_pretrained("checkpoints/best")
    
    print("Training completed!")

if __name__ == "__main__":
    main() 