import shortuuid
from PIL import Image
import random
import json
import tqdm
import os
import pandas
from pathlib import Path
import shutil

meta_data_file = "/data/PokemonBLIPCaptions/metadata.csv"
original_image_folder = Path(meta_data_file).parent
df = pandas.read_csv(meta_data_file)

pokemon_data = []
pokemon_image_path = "/data/pokemon/image"
pokemon_data_path = "/data/pokemon/pokemon_blip_captions.json"

target_folder = Path(pokemon_data_path).parent
if target_folder.exists():
    shutil.rmtree(target_folder)

os.makedirs(pokemon_image_path, exist_ok=True)
os.makedirs(Path(pokemon_data_path).parent, exist_ok=True)

description_list = [
    "Describe the image concisely.",
    "Provide a brief description of the given image.",
    "Offer a succinct explanation of the picture presented.",
    "Summarize the visual content of the image.",
    "Give a short and clear explanation of the subsequent image.",
    "Share a concise interpretation of the image provided.",
    "Present a compact description of the photo's key features.",
    "Relay a brief, clear account of the picture shown.",
    "Render a clear and concise summary of the photo.",
    "Write a terse but informative summary of the picture.",
    "Create a compact narrative representing the image presented.",
]

for index, row in tqdm.tqdm(df.iterrows()):
    uuid = shortuuid.uuid()
    sample_dict = dict()
    sample_dict["id"] = uuid
    sample_dict["image"] = uuid + ".png"

    sample_img = Image.open(original_image_folder / row["file_name"])
    sample_img.save(os.path.join(pokemon_image_path, uuid + ".png"))
    conversations = [
        {"from": "human", "value": "<image>\n" + random.choice(description_list)},
        {"from": "gpt", "value": row["text"]},
    ]
    sample_dict["conversations"] = conversations
    pokemon_data.append(sample_dict)

with open(pokemon_data_path, "w") as f:
    json.dump(pokemon_data, f, indent=4)
