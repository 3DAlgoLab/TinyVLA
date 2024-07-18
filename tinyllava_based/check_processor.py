# %%
from transformers import AutoProcessor
from PIL import Image
import requests
import io
from transformers import LlavaConfig

hf_path = "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B"
processor = AutoProcessor.from_pretrained(hf_path, trust_remote_code=True)


prompt = "What are these?"
image_url = "http://images.cocodataset.org/test-stuff2017/000000000001.jpg"


r = requests.get(image_url)
raw_image = Image.open(io.BytesIO(r.content))

inputs = processor(str(prompt), raw_image.convert("RGB"), return_tensors="pt").to(
    "cuda"
)

# %%
