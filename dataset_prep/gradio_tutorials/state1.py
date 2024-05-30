import gradio as gr

secret_word = "gradio"
css = """
    #warning {background-color: #FFCCCB}
    .feedback textarea {font-size: 24px !important}
"""

with gr.Blocks(css=css) as demo:
    used_letters_var = gr.State([])
    with gr.Row() as row:
        with gr.Column():
            input_letter = gr.Textbox(label="Enter Letter", elem_classes="feedback")
            btn = gr.Button("Guess Letter")
        with gr.Column():
            hangman = gr.Textbox(label="Hangman", value="_ " * len(secret_word))
            used_letters_box = gr.Textbox(label="Used Letters")

    def guess_letter(letter, used_letters):
        # if len(letter) != 1:
        #     input_letter.value = ""
        #     raise gr.Error("Input one letter")

        used_letters.append(letter)
        answer = " ".join(
            [letter if letter in used_letters else "_" for letter in secret_word]
        )

        return {
            used_letters_var: used_letters,
            used_letters_box: " ".join(used_letters),
            hangman: answer,
        }

    btn.click(
        guess_letter,
        [input_letter, used_letters_var],
        [used_letters_var, used_letters_box, hangman],
    )

demo.launch()
