import os
import ast
import torch
import black

def get_device_config():
    """
    Determine the optimal device configuration based on available hardware.
    
    Returns:
        tuple: (device, torch_dtype, use_fp16, device_name)
    """
    if torch.backends.mps.is_available():
        return "mps", torch.float32, False, "MPS (Apple Silicon GPU)"
    elif torch.cuda.is_available():
        return 0, torch.float16, True, "CUDA GPU"  # Use device index 0 for CUDA
    else:
        return "cpu", torch.float32, False, "CPU"

def format_code(code):
    """
    Format code with black if possible.
    
    Args:
        code (str): Raw code string to format
        
    Returns:
        str: Formatted code or original code if formatting fails
    """
    try:
        # Verify it's valid Python
        ast.parse(code)
        # Format with black
        mode = black.Mode()
        formatted_code = black.format_str(code, mode=mode)
        return formatted_code
    except (SyntaxError, black.parsing.InvalidInput, ValueError) as e:
        # If there's an error, return the original code
        print(f"Warning: Could not format code - {e}")
        return code

def extract_code(generated_text, prompt):
    """
    Extract code from the model's generated text, handling potential surrounding text.
    
    Args:
        generated_text (str): The full model output
        prompt (str): The prompt provided to the model
        
    Returns:
        str: Extracted code portion
    """
    # Remove the prompt part first
    response = generated_text[len(prompt):].strip()
    
    # Find the start of the code block (either ```python or ```)
    start_marker_py = "```python"
    start_marker_generic = "```"
    end_marker = "```"
    
    start_index_py = response.find(start_marker_py)
    start_index_generic = response.find(start_marker_generic)
    
    if start_index_py != -1:
        # Found ```python
        start_index = start_index_py + len(start_marker_py)
    elif start_index_generic != -1:
        # Found generic ``` (and not ```python)
        start_index = start_index_generic + len(start_marker_generic)
    else:
        # No code block markers found, use heuristics to extract code
        # Look for potential Python code indicators
        for line in response.split('\n'):
            if line.strip().startswith(('import ', 'from ', 'def ', 'class ', '# ')):
                # Probably code, use the entire response
                return response.strip()
        # If no code indicators found, return response as is
        return response.strip()

    # Find the end marker after the start index
    end_index = response.find(end_marker, start_index)
    
    if end_index != -1:
        # Found end marker
        code = response[start_index:end_index].strip()
    else:
        # No end marker found after start marker, take the rest of the string
        code = response[start_index:].strip()
        
    return code

def check_syntax(code):
    """
    Check if code has valid Python syntax.
    
    Args:
        code (str): Code to check
        
    Returns:
        bool: True if syntax is valid, False otherwise
    """
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False 