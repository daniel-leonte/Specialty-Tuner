# Machine Learning Code Tuner Scripts

This directory contains scripts for fine-tuning and using a language model to generate ML code.

## Architecture

The pipeline consists of several key components:

1. **Data Collection** (`collect_data.py`) - Downloads and processes datasets containing ML code examples
2. **Data Cleaning** (`clean_data.py`) - Cleans and formats code, removes poor quality examples
3. **Model Training** (`train.py`) - Fine-tunes the CodeLlama model on the ML code examples
4. **Evaluation** (`eval.py`) - Evaluates model performance with syntax checking, execution success, and functional testing
5. **Inference** (`infer.py`) - Generates ML code from prompts using the fine-tuned model
6. **Demo Generation** (`generate_demo.py`) - Creates demonstration examples using the model

## Shared Utilities

The `utils.py` module centralizes common functionality across scripts:

- **Device Detection** - Consistent handling of CUDA, MPS (Apple Silicon), and CPU configurations
- **Code Extraction** - Parsing ML code from model outputs
- **Code Formatting** - Standardized code formatting with black
- **Syntax Checking** - Validation of Python syntax

## Usage

The scripts should be run in the following order:

1. `python collect_data.py` - Gather training data
2. `python clean_data.py` - Clean and prepare the data
3. `python train.py` - Train the model
4. `python eval.py` - Evaluate model performance
5. `python infer.py --prompt "your ML task"` - Generate code for a specific task
6. `python generate_demo.py` - Create demonstration examples

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- PEFT
- Black
- Datasets 