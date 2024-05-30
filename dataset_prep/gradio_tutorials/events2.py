import gradio as gr


def welcome(name):
    return f"Welcome to Gradio, {name}!"


with gr.Blocks() as demo:
    gr.Markdown(
        """
    # Hello World!
    Start typing below to see the output.
    """
    )
    inp = gr.Textbox(label="Name", placeholder="What is your name?")
    out = gr.Textbox(label="Output")
    inp.change(welcome, inp, out)

demo.launch()
