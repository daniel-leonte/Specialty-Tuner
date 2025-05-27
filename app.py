import gradio as gr
from inference import CodeGenerator

generator = CodeGenerator()

def generate_code(prompt, max_length=512, temperature=0.7, top_p=0.9):
    return generator.generate_code(prompt, max_length, temperature, top_p) if prompt.strip() else "Enter a prompt."

with gr.Blocks() as app:
    gr.Markdown("# ML Code Generator")
    
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", placeholder="Train a logistic regression model...", lines=2)
        with gr.Column():
            max_length = gr.Slider(100, 1000, 512, label="Max Length")
            temperature = gr.Slider(0.1, 1.0, 0.7, label="Temperature") 
            top_p = gr.Slider(0.1, 1.0, 0.9, label="Top-p")
    
    generate_btn = gr.Button("Generate")
    code_output = gr.Code(language="python")
    
    generate_btn.click(generate_code, [prompt, max_length, temperature, top_p], code_output)

app.launch() 