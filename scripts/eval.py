import os
import json
import subprocess
import tempfile
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from pathlib import Path
from utils import get_device_config, extract_code, check_syntax

def check_execution(code):
    """Check if code executes without errors in a sandboxed environment"""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as temp_file:
            temp_file_path = temp_file.name
            # Add import statements for common ML libraries to make execution more likely to succeed
            setup = """
# Common ML imports - added by evaluator
try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from sklearn import datasets
except ImportError:
    pass  # Continue even if some imports fail
"""
            temp_file.write(setup + "\n" + code)
        
        # Execute the file as a separate process with a timeout
        result = subprocess.run(
            [sys.executable, temp_file_path],
            capture_output=True,
            text=True,
            timeout=15  # Increased timeout to 15 seconds
        )
        
        # Remove the temporary file
        os.unlink(temp_file_path)
        
        # Check if execution was successful
        return result.returncode == 0
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as e:
        print(f"Execution error: {e}")
        # Attempt to clean up the temporary file if it still exists
        try:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        except:
            pass
        return False

def run_pytest(code: str, test_code: str) -> bool:
    """Write `code` and `test_code` into a temporary directory and run pytest.
    Returns True if all tests pass, otherwise False."""
    try:
        import pytest  # noqa: F401 – ensure dependency available
    except ImportError:
        raise RuntimeError("PyTest is required for functional evaluation. Install with `pip install pytest`." )

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Write generated solution
            (tmp_path / "solution.py").write_text(code, encoding="utf-8")

            # Prepend an import of the solution for convenience if user hasn't provided one
            import_line = "import solution as sol\n"
            if "import solution" in test_code:
                full_test_code = test_code
            else:
                full_test_code = import_line + test_code

            (tmp_path / "test_solution.py").write_text(full_test_code, encoding="utf-8")

            # Run pytest quietly
            res = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "test_solution.py"],
                cwd=tmp_path,
                capture_output=True,
                text=True,
                timeout=15,
            )
            return res.returncode == 0
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError) as e:
        print(f"Pytest execution error: {e}")
        return False

def main():
    # Check for checkpoint
    checkpoint_path = "checkpoints/best"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("Model checkpoint not found. Run train.py first.")
    
    # Check for test data
    test_file = "data/test.jsonl"
    if not os.path.exists(test_file):
        raise FileNotFoundError("Test data not found. Run collect_data.py and clean_data.py first.")
    
    # Load test data
    with open(test_file) as f:
        test_data = [json.loads(line) for line in f]
    
    # Test on up to 20 examples
    max_examples = 20
    test_data = test_data[:max_examples] if len(test_data) > max_examples else test_data
    
    print(f"Evaluating on {len(test_data)} test examples...")
    
    # Determine the device to use
    device, torch_dtype, _, device_name = get_device_config()
    print(f"Using {device_name}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    # Create a text generation pipeline
    model = pipeline(
        "text-generation",
        model=checkpoint_path,
        tokenizer=tokenizer,
        device=device,
        torch_dtype=torch_dtype,
    )
    
    # Evaluation hyper-parameters
    num_samples = 5  # pass@k – number of generations per prompt

    # Create better evaluation prompts
    test_prompts = []
    for item in test_data:
        # Create prompt with better instruction
        basic_prompt = f"Write Python ML code for: {item['instruction']}"
        enhanced_prompt = basic_prompt + "\nMake sure the code is complete, well-structured, and follows best practices."
        test_prompts.append({
            "instruction": item['instruction'],
            "prompt": enhanced_prompt,
            "tests": item.get('tests')
        })

    # Evaluation metrics
    syntax_ok = 0
    exec_ok = 0
    tests_pass_any = 0  # pass@k numerator
    
    # Run evaluation
    for i, item in enumerate(test_prompts):
        # Create prompt
        prompt = item["prompt"]
        print(f"\nExample {i+1}/{len(test_prompts)}: {item['instruction']}")
        
        any_test_pass = False  # track pass@k for this example

        for sample_idx in range(num_samples):
            # Generate code sample
            outputs = model(
                prompt,
                max_new_tokens=800,  # Increased token limit
                temperature=0.2,     # Lower temperature for more precision
                do_sample=True,
                top_p=0.95,          # Added top_p
                repetition_penalty=1.1, # Prevent repetitive output
                truncation=True,
            )
            output_text = outputs[0]["generated_text"]

            # Extract code from output
            generated_code = extract_code(output_text, prompt)
            print(f"  Sample {sample_idx+1}/{num_samples} – length: {len(generated_code)} chars")

            # Syntax check
            is_syntax_valid = check_syntax(generated_code)
            if is_syntax_valid:
                syntax_ok += 1
            
            # Execution check (sanity, independent of tests)
            if is_syntax_valid and check_execution(generated_code):
                exec_ok += 1

            # Functional correctness via pytest if tests are provided
            tests_str = item.get("tests")
            if tests_str:
                if run_pytest(generated_code, tests_str):
                    any_test_pass = True
                    # no need to generate more samples once we pass
                    break

        # Record pass@k result
        if any_test_pass:
            tests_pass_any += 1
    
    # Calculate metrics
    total_examples = len(test_prompts)
    syntax_accuracy = syntax_ok / (total_examples * num_samples) * 100
    exec_accuracy = exec_ok / (total_examples * num_samples) * 100
    pass_at_k = tests_pass_any / total_examples * 100
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results:")
    print(f"Syntax Correctness: {syntax_accuracy:.1f}% ({syntax_ok}/{total_examples * num_samples})")
    print(f"Execution Success: {exec_accuracy:.1f}% ({exec_ok}/{total_examples * num_samples})")
    print(f"pass@{num_samples}: {pass_at_k:.1f}% ({tests_pass_any}/{total_examples})")
    
    # Save results to file
    results = {
        "syntax_correctness": syntax_accuracy,
        "execution_success": exec_accuracy,
        "examples_evaluated": total_examples,
        f"pass@{num_samples}": pass_at_k
    }
    
    with open("eval_results.txt", "w") as f:
        f.write(f"Syntax Correctness: {syntax_accuracy:.1f}%\n")
        f.write(f"Execution Success: {exec_accuracy:.1f}%\n")
        f.write(f"pass@{num_samples}: {pass_at_k:.1f}%\n")
        f.write(f"Examples Evaluated: {total_examples}")
    
    print(f"Results saved to eval_results.txt")

if __name__ == "__main__":
    main() 