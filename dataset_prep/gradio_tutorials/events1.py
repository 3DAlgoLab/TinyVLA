import gradio as gr


# def greet(name):
#     return "Hello " + name + "!"


def greet(name):
    return "Hello " + name + "!!!"


with gr.Blocks() as demo:
    name = gr.Text(label="Name")
    # output = gr.Textbox(label="Output Box", interactive=True)
    greet_btn = gr.Button("Greet")
    greet_btn.click(greet, name, name)

    # @greet_btn.click(inputs=name, outputs=output)


demo.launch()
