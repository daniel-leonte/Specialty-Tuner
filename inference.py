"""
Minimal code generator using fine-tuned Phi-3. Works on any device.
"""

import argparse
import re
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class CodeGenerator:
    def __init__(self, model_path: str = "./phi3_ml_finetuned/final", quantize: bool = True):
        self.device = self._get_device()
        self.model_path = Path(model_path)
        self.base_model = "microsoft/Phi-3-mini-4k-instruct"
        self.quantize = quantize
        self._load_model()
    
    def _get_device(self) -> torch.device:
        if torch.backends.mps.is_available(): return torch.device("mps")
        if torch.cuda.is_available(): return torch.device("cuda")
        return torch.device("cpu")
    
    def _get_quant_config(self) -> Optional[BitsAndBytesConfig]:
        if not self.quantize or self.device.type == "mps": return None
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    def _load_model(self):
        # Load tokenizer
        tokenizer_path = str(self.model_path) if self.model_path.exists() else self.base_model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Model config
        quant_config = self._get_quant_config()
        kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "attn_implementation": "eager" if self.device.type == "mps" else "flash_attention_2"
        }
        
        if quant_config:
            kwargs["quantization_config"] = quant_config
        else:
            kwargs["torch_dtype"] = torch.bfloat16 if self.device.type == "mps" else torch.float16
            if self.device.type == "mps":
                kwargs["device_map"] = "auto"
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(self.base_model, **kwargs)
        
        if not quant_config and self.device.type != "mps":
            self.model = self.model.to(self.device)
        
        # Apply fine-tuning if available
        if self.model_path.exists():
            self.model = PeftModel.from_pretrained(self.model, str(self.model_path))
        
        self.model.eval()
        
        # Clear cache for MPS
        if self.device.type == "mps":
            torch.mps.empty_cache()
    
    def generate(self, prompt: str, max_tokens: int = 512, for_code: bool = True) -> str:
        # Clear MPS cache
        if self.device.type == "mps":
            torch.mps.empty_cache()
        
        # Format prompt
        formatted = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        if for_code:
            formatted += "```python\n"
        
        # Tokenize and generate
        inputs = self.tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=False
            )
        
        # Clear MPS cache
        if self.device.type == "mps":
            torch.mps.empty_cache()
        
        # Decode and extract
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = text[len(formatted):].strip()
        
        # Extract code if needed
        if for_code and "```python" in response:
            if match := re.search(r'```python\n(.*?)(?:```|<\|end\||$)', response, re.DOTALL):
                return match.group(1).strip()
        
        return response


def main():
    parser = argparse.ArgumentParser(description="Minimal Code Generator")
    parser.add_argument("--test", action="store_true", help="Run test")
    parser.add_argument("--prompt", help="Generate code for prompt")
    parser.add_argument("--no-quant", action="store_true", help="Disable quantization")
    
    args = parser.parse_args()
    
    generator = CodeGenerator(quantize=not args.no_quant)
    
    if args.test:
        result = generator.generate("Write a function to add two numbers", for_code=False)
        print(f"\n{result}\n")
    elif args.prompt:
        code = generator.generate(args.prompt)
        print(f"\n{code}\n")
    else:
        # Demo
        for prompt in ["Train a logistic regression", "Create preprocessing function", "Build neural network"]:
            print(f"\n--- {prompt} ---")
            print(generator.generate(prompt))


if __name__ == "__main__":
    main() 