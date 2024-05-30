import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


def predict(message, history):
    history_openai_format = []
    for human, ai in history:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({"role": "assistant", "content": ai})

    history_openai_format.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=history_openai_format, stream=True
    )

    partial_msg = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            partial_msg += chunk.choices[0].delta.content
            yield partial_msg


gr.ChatInterface(predict).launch()
