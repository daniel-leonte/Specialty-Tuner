import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from utils import get_device_config, format_code, extract_code

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate ML code using fine-tuned model")
    parser.add_argument("--prompt", required=True, help="Prompt describing the ML task")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (default: 0.2)")
    parser.add_argument("--max_length", type=int, default=800, help="Maximum response length (default: 800)")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling value")
    parser.add_argument("--streaming", action="store_true", help="Enable streaming output")
    args = parser.parse_args()

    # Check if model exists
    checkpoint_path = "checkpoints/best"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("Model checkpoint not found. Run train.py first.")
    
    # Determine device
    device, torch_dtype, use_fp16, device_name = get_device_config()
    print(f"Using {device_name}")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch_dtype,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch_dtype
        )
        model = model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    # Create text generation pipeline
    print("Setting up generation pipeline...")
    if torch.cuda.is_available():
        model = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=args.max_length,
            torch_dtype=torch_dtype
        )
    else:
        model = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=args.max_length,
            torch_dtype=torch_dtype
        )
    
    # Create prompt
    prompt = f"Write Python ML code for: {args.prompt}"
    
    # Generate code
    print(f"Generating code for: {args.prompt}")
    outputs = model(
        prompt, 
        temperature=args.temperature,
        max_length=args.max_length,
        do_sample=True,
        top_p=args.top_p,
        repetition_penalty=1.1,  # Discourage repetitive output
        num_return_sequences=1,
    )
    output_text = outputs[0]['generated_text']
    
    # Extract and format code
    code = extract_code(output_text, prompt)
    formatted_code = format_code(code)
    
    # Print code with markdown code block formatting
    print("\n```python")
    print(formatted_code)
    print("```")

if __name__ == "__main__":
    main() 