import gradio as gr

with gr.Blocks() as demo:
    with gr.Row():
        text1 = gr.Textbox(label="t1")
        slider2 = gr.Textbox(label="s2")
        drop3 = gr.Dropdown(["a", "b", "c"], label="d3")
    with gr.Row():
        # with gr.Column(scale=1, min_width=600):
        with gr.Column():

            text1 = gr.Textbox(label="prompt 1")
            text2 = gr.Textbox(label="prompt 2")
            inbtw = gr.Button("Between")
            text4 = gr.Textbox(label="prompt 1")
            text5 = gr.Textbox(label="prompt 2")
        # with gr.Column(scale=2, min_width=600):
        with gr.Column():
            img1 = gr.Image("images/cheetah.jpg")
            btn = gr.Button("Go")

demo.launch()
