# CodeGen-ML: ML Code Generator

A fine-tuned language model that generates Python code for machine learning tasks based on natural language prompts. Built using Microsoft's Phi-3-mini-4k-instruct model with LoRA fine-tuning.

## 🚀 Project Overview

CodeGen-ML is a specialized code generation tool that understands machine learning concepts and generates Python code for various ML tasks including:

- Data preprocessing and feature engineering
- Model training and evaluation
- Neural network implementations
- Cross-validation and hyperparameter tuning
- Visualization and analysis

The model is fine-tuned using Parameter-Efficient Fine-Tuning (PEFT) with LoRA, making it memory-efficient and suitable for single GPU setups.

## 🛠️ Features

- **Fine-tuned Phi-3 Model**: Specialized for ML code generation
- **Memory Efficient**: Uses 4-bit quantization and LoRA for 16GB VRAM compatibility
- **Web Interface**: Beautiful Gradio UI for easy interaction
- **Flexible Generation**: Adjustable parameters for code generation
- **Ready for Deployment**: Compatible with Hugging Face Spaces

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU with 16GB VRAM (recommended)
- 20GB+ free disk space for model and dependencies

## 🔧 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/CodeGen-ML.git
cd CodeGen-ML
```

2. **Create a virtual environment**:
```bash
python -m venv codegen_env
source codegen_env/bin/activate  # On Windows: codegen_env\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 📊 Dataset

The project includes a sample dataset (`ml_code_dataset.jsonl`) with prompt-code pairs covering various ML tasks. Each entry contains:

- **Prompt**: Natural language description of an ML task
- **Code**: Corresponding Python implementation

### Dataset Format
```json
{"prompt": "Write a Python function to preprocess a dataset...", "code": "from sklearn.preprocessing import StandardScaler..."}
```

The dataset covers:
- Data preprocessing with scikit-learn
- Model training and evaluation
- PyTorch neural network implementations
- Statistical analysis and visualization

## 🎯 Usage

### 1. Fine-tune the Model

Run the fine-tuning script to train the model on the ML code dataset:

```bash
python finetune_phi3.py
```

**Training Configuration**:
- LoRA rank: 16, alpha: 32
- Target modules: q_proj, v_proj
- Batch size: 4 with gradient accumulation: 4
- Learning rate: 2e-4
- Max steps: 500
- Mixed precision: FP16

The fine-tuned model will be saved to `./phi3_ml_finetuned/final/`.

### 2. Test Inference

Test the fine-tuned model with sample prompts:

```bash
python inference.py
```

### 3. Launch Web Interface

Start the Gradio web application:

```bash
python app.py
```

The interface will be available at `http://localhost:7860`.

### 4. Generate Code

Use the web interface or Python API:

```python
from inference import CodeGenerator

generator = CodeGenerator()
code = generator.generate_code("Train a logistic regression model on a dataset")
print(code)
```

## 🌐 Web Interface Features

- **Intuitive UI**: Clean, modern interface for code generation
- **Example Prompts**: Pre-built examples for common ML tasks
- **Advanced Settings**: Adjustable generation parameters
- **Code Highlighting**: Syntax-highlighted Python output
- **Real-time Generation**: Instant code generation from prompts

## 🚀 Deployment

### Local Deployment

The Gradio app runs locally and can be accessed via web browser. Set `share=True` in `app.py` for temporary public sharing.

### Hugging Face Spaces Deployment

1. **Create a new Space** on Hugging Face with Gradio SDK
2. **Upload files** to the Space repository:
   ```
   app.py
   inference.py
   requirements.txt
   README.md
   ```
3. **Configure Space settings**:
   - SDK: Gradio
   - Hardware: GPU (T4 or better recommended)
   - Python version: 3.8+

4. **Add model files**: Upload the fine-tuned model to the Space or use Hugging Face Model Hub

### Environment Variables

For Hugging Face Spaces, you may need to set:
```
HF_TOKEN=your_hugging_face_token  # If using private models
```

## 📁 Project Structure

```
CodeGen-ML/
├── requirements.txt          # Python dependencies
├── ml_code_dataset.jsonl    # Training dataset
├── finetune_phi3.py         # Fine-tuning script
├── inference.py             # Inference utilities
├── app.py                   # Gradio web interface
├── README.md                # Project documentation
├── .gitignore              # Git ignore rules
└── phi3_ml_finetuned/      # Fine-tuned model (excluded from git)
    └── final/
        ├── adapter_config.json
        ├── adapter_model.safetensors
        └── tokenizer files
```

## 🔧 Configuration

### Model Parameters

- **Base Model**: microsoft/Phi-3-mini-4k-instruct
- **LoRA Configuration**:
  - Rank (r): 16
  - Alpha: 32
  - Target modules: ["q_proj", "v_proj"]
  - Dropout: 0.1

### Training Parameters

- **Batch Size**: 4 (effective: 16 with gradient accumulation)
- **Learning Rate**: 2e-4
- **Max Steps**: 500
- **Warmup Steps**: 50
- **Scheduler**: Cosine annealing

### Generation Parameters

- **Max Length**: 512 tokens
- **Temperature**: 0.7
- **Top-p**: 0.9
- **Do Sample**: True

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Microsoft for the Phi-3-mini-4k-instruct model
- Hugging Face for the Transformers library
- The PEFT team for LoRA implementation
- Gradio team for the web interface framework

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/CodeGen-ML/issues) page
2. Create a new issue with detailed description
3. Join our community discussions

## 🔮 Future Enhancements

- [ ] Support for more ML frameworks (JAX, XGBoost)
- [ ] Code execution and validation
- [ ] Multi-language support
- [ ] Advanced prompt engineering
- [ ] Model performance metrics dashboard
- [ ] Integration with popular ML platforms

---

**Made with ❤️ for the ML community** 