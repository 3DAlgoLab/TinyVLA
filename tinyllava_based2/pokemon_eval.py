from tinyllava.eval.run_tiny_llava import eval_model
from pathlib import Path


model_path = str(
    Path().absolute() / "outputs" / "pokemon-finetune-TinyLLaVA-Phi-2-SigLIP-3.1B-lora"
)

print("model path:", model_path)
# prompt = "What are the things I should be cautious about when I visit here?"
prompt = "Provide a brief description of the given image."
# image_file = "https://llava-vl.github.io/static/images/view.jpg"
# image_file = str(Path().absolute() / "pokemon_like_t.png")
image_file = "/data/PokemonBLIPCaptions/image_18.png"
print("image file:", image_file)

conv_mode = "phi"  # or llama, gemma, etc

args = type(
    "Args",
    (),
    {
        "model_path": model_path,
        "model": None,
        "query": prompt,
        "conv_mode": conv_mode,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512,
    },
)()

eval_model(args)
