import random
import gradio as gr


def random_response(message, history):
    return random.choice(["Yes", "No"])


gr.ChatInterface(random_response).launch()
