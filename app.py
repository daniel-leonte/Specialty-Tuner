import gradio as gr
import torch
from inference import CodeGenerator
import os

# Global variable to store the model
generator = None

def initialize_model():
    """Initialize the code generator model."""
    global generator
    if generator is None:
        try:
            generator = CodeGenerator()
            return "Model loaded successfully!"
        except Exception as e:
            return f"Error loading model: {str(e)}"
    return "Model already loaded!"

def generate_ml_code(prompt, max_length, temperature, top_p):
    """Generate ML code from a prompt using the fine-tuned model."""
    global generator
    
    if generator is None:
        return "Error: Model not loaded. Please wait for model initialization."
    
    if not prompt.strip():
        return "Please enter a prompt describing the ML task you want to implement."
    
    try:
        # Generate code
        code = generator.generate_code(
            prompt=prompt,
            max_length=int(max_length),
            temperature=temperature,
            top_p=top_p
        )
        return code
    except Exception as e:
        return f"Error generating code: {str(e)}"

def create_interface():
    """Create and configure the Gradio interface."""
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        text-align: center;
        margin-bottom: 30px;
    }
    .description {
        text-align: center;
        margin-bottom: 20px;
        color: #666;
    }
    """
    
    with gr.Blocks(css=css, title="CodeGen-ML") as interface:
        # Header
        gr.Markdown(
            """
            # 🤖 CodeGen-ML: ML Code Generator
            
            Generate Python code for machine learning tasks using natural language prompts.
            Powered by fine-tuned Phi-3-mini-4k-instruct model.
            """,
            elem_classes=["main-header"]
        )
        
        # Model status
        with gr.Row():
            model_status = gr.Textbox(
                label="Model Status",
                value="Initializing model...",
                interactive=False
            )
            init_btn = gr.Button("Initialize Model", variant="secondary")
        
        # Main interface
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### 📝 Describe your ML task")
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="e.g., Train a logistic regression model on a dataset with cross-validation",
                    lines=3,
                    max_lines=5
                )
                
                # Advanced settings
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    max_length = gr.Slider(
                        minimum=100,
                        maximum=1000,
                        value=512,
                        step=50,
                        label="Max Length"
                    )
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature"
                    )
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.1,
                        label="Top-p"
                    )
                
                generate_btn = gr.Button("🚀 Generate Code", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                # Output section
                gr.Markdown("### 🐍 Generated Python Code")
                code_output = gr.Code(
                    label="Generated Code",
                    language="python",
                    lines=20
                )
        
        # Example prompts
        gr.Markdown("### 💡 Example Prompts")
        examples = gr.Examples(
            examples=[
                ["Train a logistic regression model on a dataset with train-test split and evaluate performance"],
                ["Create a function to preprocess data by handling missing values and scaling features"],
                ["Implement a PyTorch neural network for binary classification with training loop"],
                ["Write code to perform k-fold cross-validation on a machine learning model"],
                ["Create a function to visualize feature importance from a trained random forest model"],
                ["Implement gradient descent optimization for linear regression from scratch"],
            ],
            inputs=[prompt_input],
            label="Click on any example to use it"
        )
        
        # Event handlers
        init_btn.click(
            fn=initialize_model,
            outputs=[model_status]
        )
        
        generate_btn.click(
            fn=generate_ml_code,
            inputs=[prompt_input, max_length, temperature, top_p],
            outputs=[code_output]
        )
        
        # Auto-initialize model on startup
        interface.load(
            fn=initialize_model,
            outputs=[model_status]
        )
    
    return interface

def main():
    """Launch the Gradio application."""
    # Check if model exists
    model_path = "./phi3_ml_finetuned/final"
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}")
        print("Please run finetune_phi3.py first to train the model.")
    
    # Create and launch interface
    interface = create_interface()
    
    # Launch with appropriate settings
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True for public sharing
        debug=False
    )

if __name__ == "__main__":
    main() 