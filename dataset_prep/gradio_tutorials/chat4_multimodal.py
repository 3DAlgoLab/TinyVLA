import gradio as gr
import time


def count_files(message, history):
    num_files = len(message["files"])
    return f"You uploaded {num_files} files"


# demo = gr.ChatInterface(
#     count_files,
#     chatbot=gr.Chatbot(
#         placeholder="<strong>Your Personal Yes-Man</strong><br>Ask Me Anything"
#     ),
#     textbox=gr.Textbox(
#         placeholder="Ask me a yes or no question", container=False, scale=7
#     ),
#     title="Yes Man",
#     description="Ask Yes Man any question",
#     theme="soft",
#     examples=[{"text": "Hello", "files": []}],
#     cache_examples=True,
#     retry_btn=None,
#     undo_btn="Delete Previous",
#     clear_btn="Clear",
#     multimodal=True,
# )

demo = gr.ChatInterface(
    fn=count_files,
    examples=[{"text": "Hello", "files": []}],
    title="Echo bot",
    multimodal=True,
)

demo.launch()
