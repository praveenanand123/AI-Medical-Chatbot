import gradio as gr
from brain_of_the_doctor import analyze_input

# Function to handle user inputs
def handle_inputs(text_input, image_input, audio_input):
    return analyze_input(text_input=text_input, image_input=image_input, audio_input=audio_input)

# Building the Gradio interface
with gr.Blocks(css="""
body {
    background: linear-gradient(135deg, #d4e9ff, #e3f4ff, #f4fcff);
    animation: gradientBG 10s ease infinite;
    background-size: 200% 200%;
}

@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.center-heading {
    text-align: center;
    font-size: 2.5rem;
    color: #1e3a8a;
    margin-bottom: 25px;
    font-weight: bold;
}

button.submit-button {
    background-color: #1e3a8a;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    font-size: 1rem;
    transition: background 0.3s ease;
}

button.submit-button:hover {
    background-color: #164270;
}

.response-area {
    padding: 15px;
    background: #ffffffcc;
    border: 1px solid #1e3a8a;
    border-radius: 10px;
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
}
""") as demo:
    gr.HTML("<h2 class='center-heading'>🩺 AI Doctor - Diagnostics Specialist</h2>")

    with gr.Row():
        text_input = gr.Textbox(label="Enter your query", placeholder="Type your question here...", lines=2)
        image_input = gr.Image(label="Upload an image (optional)", type="pil", sources=["upload", "webcam"])
        audio_input = gr.Audio(label="Record or upload audio (optional)", type="numpy", sources=["microphone", "upload"])

    submit_button = gr.Button("Submit", elem_classes=["submit-button"])
    output = gr.Markdown(label="Response", elem_classes=["response-area"])

    submit_button.click(fn=handle_inputs, inputs=[text_input, image_input, audio_input], outputs=output)

demo.launch()