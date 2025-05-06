import os
import json
from datasets import load_dataset
import random


def main():
    """Download datasets and convert them to our instruction/response JSONL format."""

    # Ensure output directory exists
    os.makedirs("data", exist_ok=True)

    # Track total examples collected
    total_examples = 0
    
    # Output path for combined raw data
    output_path = "data/raw.jsonl"
    
    with open(output_path, "w", encoding="utf-8") as fp:
        # 1. Load the DS-1000 dataset (Python code for ML & data science tasks)
        print("Loading DS-1000 dataset...")
        ds_1000 = load_dataset("xlangai/DS-1000", split="test")
        
        for row in ds_1000:
            record = {
                "instruction": row["prompt"],
                "response": row["reference_code"],
                "source": "ds-1000"
            }
            json.dump(record, fp, ensure_ascii=False)
            fp.write("\n")
            total_examples += 1
        
        print(f"Added {len(ds_1000)} examples from DS-1000")
        
        # 2. Load HumanEval dataset for additional Python examples
        print("Loading HumanEval dataset...")
        try:
            humaneval = load_dataset("openai_humaneval", split="test")
            
            for row in humaneval[:200]:  # Limit to 200 examples
                # Extract the docstring as instruction and the canonical solution
                instruction = row["prompt"].split("def ")[0].strip()
                solution = row["canonical_solution"]
                
                # Skip solutions that are too short
                if len(solution.strip().split("\n")) < 3:
                    continue
                    
                record = {
                    "instruction": f"Write a Python function that {instruction}",
                    "response": solution,
                    "source": "humaneval"
                }
                json.dump(record, fp, ensure_ascii=False)
                fp.write("\n")
                total_examples += 1
            
            print(f"Added examples from HumanEval")
        except Exception as e:
            print(f"Error loading HumanEval: {e}")
            
        # 3. Load examples from the CodeAlpaca dataset
        print("Loading CodeAlpaca dataset...")
        try:
            alpaca = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
            
            # Filter for Python examples and ML/data science tasks
            ml_keywords = ["machine learning", "data", "model", "train", "sklearn", 
                          "pandas", "numpy", "tensorflow", "torch", "keras", 
                          "plot", "visualize", "classify", "regression", "cluster"]
            
            ml_examples = []
            for row in alpaca:
                instruction = row["instruction"]
                # Check if it's likely an ML/data science task
                if any(keyword in instruction.lower() for keyword in ml_keywords):
                    if "python" in instruction.lower() or "code" in instruction.lower():
                        ml_examples.append(row)
            
            # Randomly select up to 200 examples
            selected = random.sample(ml_examples, min(200, len(ml_examples)))
            
            for row in selected:
                record = {
                    "instruction": row["instruction"],
                    "response": row["output"],
                    "source": "codealpaca"
                }
                json.dump(record, fp, ensure_ascii=False)
                fp.write("\n")
                total_examples += 1
                
            print(f"Added {len(selected)} examples from CodeAlpaca")
        except Exception as e:
            print(f"Error loading CodeAlpaca: {e}")

    print(f"Created {total_examples} total examples in {output_path}")
    print("Run clean_data.py next to prepare the dataset for training.")


if __name__ == "__main__":
    main() 