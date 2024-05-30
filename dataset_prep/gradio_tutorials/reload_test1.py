from turtle import title
import gradio as gr

with gr.Blocks(title="Greeting!") as demo:
    gr.Markdown("# Greetings from Gradio!")
    inp = gr.Textbox(placeholder="What is your name?", label="name")
    out = gr.Textbox(label="greeting")

    inp.change(fn=lambda x: f"Welcome, {x}!", inputs=inp, outputs=out)

if __name__ == "__main__":
    demo.launch()
