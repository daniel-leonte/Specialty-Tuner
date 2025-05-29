"""
Simple code generator using fine-tuned Phi-3.
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


class CodeGenerator:
    def __init__(self, model_path: str = "./phi3_ml_finetuned/final", quantize: bool = True):
        self.model_path = Path(model_path)
        self.base_model = "microsoft/Phi-3-mini-4k-instruct"
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Simple device detection - disable quantization on MPS
        is_mps = torch.backends.mps.is_available()
        use_quantization = quantize and not is_mps
        
        # Model loading
        kwargs = {
            "trust_remote_code": True, 
            "torch_dtype": torch.float16,
            "attn_implementation": "eager"  # Avoid flash attention issues
        }
        if use_quantization:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        kwargs["device_map"] = "mps" if is_mps else "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(self.base_model, **kwargs)
        self.device = "mps" if is_mps else "cuda" if torch.cuda.is_available() else "cpu"
        
        # Apply fine-tuning if available
        if self.model_path.exists():
            self.model = PeftModel.from_pretrained(self.model, str(self.model_path))
        
        self.model.eval()
    
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        # Simple prompt format
        text = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        
        # Generate
        inputs = self.tokenizer(text, return_tensors="pt")
        # Move inputs to the same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=False  # Avoid cache compatibility issues
            )
        
        # Debug: Print full output
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"DEBUG - Full output: {full_text}")
        print(f"DEBUG - Input length: {len(inputs['input_ids'][0])}")
        print(f"DEBUG - Output length: {len(outputs[0])}")
        
        # Extract response - simpler approach
        response = full_text[len(text):].strip()
        
        # Clean up
        if response.startswith("```python"):
            response = response[9:]  # Remove ```python
        if "```" in response:
            response = response.split("```")[0]  # Remove closing ```
        
        return response.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", help="Code generation prompt")
    parser.add_argument("--no-quant", action="store_true", help="Disable quantization")
    parser.add_argument("--max-tokens", type=int, default=512)
    
    args = parser.parse_args()
    
    generator = CodeGenerator(quantize=not args.no_quant)
    
    if args.prompt:
        print(generator.generate(args.prompt, args.max_tokens))
    else:
        # Interactive mode
        while True:
            try:
                prompt = input("\n> ")
                if not prompt.strip():
                    break
                print(generator.generate(prompt, args.max_tokens))
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main() 