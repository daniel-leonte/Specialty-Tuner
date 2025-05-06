import json
import ast
import random
import os
import re
from typing import List, Dict, Any
from utils import check_syntax, format_code

def is_ml_code(code: str) -> bool:
    """Check if the code is related to machine learning by looking for ML-related imports."""
    ml_libraries = [
        'sklearn', 'scikit-learn', 'numpy', 'pandas', 'matplotlib', 'seaborn', 
        'tensorflow', 'torch', 'keras', 'xgboost', 'lightgbm'
    ]
    
    try:
        # Parse the code
        tree = ast.parse(code)
        
        # Extract import statements
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(name.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module.split('.')[0])
        
        # Check if any ML library is imported
        return any(lib in imports for lib in ml_libraries)
    except SyntaxError:
        # If code can't be parsed, it's not valid Python
        return False

def clean_code(code: str) -> str:
    """Clean and format a code snippet, ensuring it follows best practices."""
    try:
        # Check if valid Python code
        if not check_syntax(code):
            return None
        
        # Format with black
        formatted_code = format_code(code)
        
        # Remove unused imports (simple version)
        formatted_code = remove_unused_imports(formatted_code)
        
        return formatted_code
    except (SyntaxError, ValueError) as e:
        print(f"Cleaning error: {e}")
        return None

def remove_unused_imports(code: str) -> str:
    """Simple heuristic to remove obviously unused imports."""
    try:
        tree = ast.parse(code)
        
        # Get all import names
        import_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    import_names.append(name.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for name in node.names:
                        import_names.append(name.name)
        
        # Get all names used in the code
        used_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
        
        # Filter out imports that are definitely unused
        unused_imports = [name for name in import_names if name not in used_names]
        
        # If we found unused imports, remove them from the code
        if unused_imports:
            lines = code.split('\n')
            filtered_lines = []
            for line in lines:
                if any(f"import {name}" in line or f"from {name}" in line or f", {name}" in line for name in unused_imports):
                    # Check if this is a multi-import line and only remove the unused import
                    if ',' in line:
                        for name in unused_imports:
                            # Replace ", name" or "name, " patterns
                            line = re.sub(rf',\s*{re.escape(name)}\s*', '', line)
                            line = re.sub(rf'\s*{re.escape(name)},\s*', ' ', line)
                    else:
                        # Skip the line if it's a single import of an unused module
                        continue
                filtered_lines.append(line)
            return '\n'.join(filtered_lines)
        
        return code
    except (SyntaxError, Exception):
        # If there's any error, return the original code
        return code

def evaluate_code_quality(code: str) -> float:
    """Evaluate the quality of code on a scale of 0 to 1."""
    score = 0.0
    
    try:
        # Parse the code
        tree = ast.parse(code)
        
        # 1. Check if it has ML-related imports (0.3 points)
        if is_ml_code(code):
            score += 0.3
        else:
            # Check for any imports at all (partial credit: 0.1)
            has_imports = False
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    has_imports = True
                    score += 0.1
                    break
        
        # 2. Check for docstrings (0.2 points) - reduced to 0.1
        has_docstring = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(node):
                has_docstring = True
                break
        if has_docstring:
            score += 0.1
        
        # 3. Check for comments (0.1 points)
        if '#' in code:
            score += 0.1
        
        # 4. Check for function definitions (0.2 points) - now 0.1
        has_functions = any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
        if has_functions:
            score += 0.1
        
        # 5. Check for reasonable length (0.2 points)
        lines = code.strip().split('\n')
        if 5 <= len(lines) <= 100:  # Reduced minimum from 10 to 5 lines
            score += 0.2
        elif len(lines) > 3:
            # Partial credit for even shorter code
            score += 0.1
        
        return score
    except (SyntaxError, Exception):
        # Even with syntax errors, give minimal points if it has ML-related text
        if any(lib in code for lib in ['sklearn', 'numpy', 'pandas', 'matplotlib', 'torch', 'tensorflow']):
            return 0.1
        return 0.0

def try_fix_common_syntax_errors(code: str) -> str:
    """Attempt to fix common syntax errors in code."""
    if code is None:
        return None
        
    # Fix 1: Inconsistent indentation - replace tabs with spaces
    code = code.replace('\t', '    ')
    
    # Fix 2: Remove trailing spaces
    lines = code.split('\n')
    code = '\n'.join(line.rstrip() for line in lines)
    
    # Fix 3: Try to fix missing parentheses
    open_parens = code.count('(')
    close_parens = code.count(')')
    if open_parens > close_parens:
        code += ')' * (open_parens - close_parens)
    
    # Fix 4: Try to fix common quote issues
    if code.count('"') % 2 == 1:
        code = code.replace('"', "'")
    
    # Fix 5: Handle incomplete code blocks
    if code.strip().endswith(':'):
        code += '\n    pass'
    
    return code

print("Loading and cleaning data...")

# Load the raw data
data = []
skipped_count = 0
fixed_count = 0

with open("data/raw.jsonl") as f:
    for line in f:
        try:
            item = json.loads(line)
            
            # First try cleaning the original code
            cleaned_code = clean_code(item["response"])
            
            # If cleaning failed, try fixing common errors first
            if cleaned_code is None:
                fixed_code = try_fix_common_syntax_errors(item["response"])
                cleaned_code = clean_code(fixed_code)
                if cleaned_code:
                    fixed_count += 1
            
            if cleaned_code:
                quality_score = evaluate_code_quality(cleaned_code)
                
                # Lower the quality threshold to keep more examples
                if quality_score >= 0.3:  # Changed from 0.5 to 0.3
                    data.append({
                        "instruction": item["instruction"],
                        "response": cleaned_code,
                        "quality": quality_score
                    })
                else:
                    print(f"Discarded low quality example (score: {quality_score:.2f})")
                    skipped_count += 1
            else:
                skipped_count += 1
                print(f"Couldn't clean code - skipping example")
        except Exception as e:
            print(f"Error processing example: {e}")
            skipped_count += 1

print(f"Kept {len(data)} quality examples out of the original dataset")
print(f"Fixed {fixed_count} examples with syntax issues")
print(f"Skipped {skipped_count} examples that couldn't be cleaned/fixed")

# Sort by quality score
data = sorted(data, key=lambda x: x["quality"], reverse=True)

# Remove quality field for final output
for item in data:
    if "quality" in item:
        del item["quality"]

# Shuffle data for randomization
random.seed(42)
random.shuffle(data)

# Make sure output directories exist
os.makedirs("data", exist_ok=True)

# Calculate split sizes
total = len(data)
train_size = int(total * 0.8)
val_size = int(total * 0.1)
# test_size is the remainder

# Split data into train, validation, and test sets
train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

# Write to files
def write_jsonl(filename, data_list):
    with open(filename, 'w') as f:
        for item in data_list:
            json.dump(item, f)
            f.write('\n')

write_jsonl("data/train.jsonl", train_data)
write_jsonl("data/val.jsonl", val_data)
write_jsonl("data/test.jsonl", test_data)

# Output stats
print(f"Total examples processed: {total}")
print(f"Train set: {len(train_data)} examples")
print(f"Validation set: {len(val_data)} examples")
print(f"Test set: {len(test_data)} examples") 