# %%
import torch
from icecream import ic
from transformers import AutoTokenizer, pipeline

# %%

torch.manual_seed(0)
model = "tiiuae/falcon-7b-instruct"
# model = "google/gemma-2b-it"
# model = "arcee-ai/Arcee-Spark" # Great!

tokenizer = AutoTokenizer.from_pretrained(model)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
# %%
torch.manual_seed(0)
prompt = """Classify the text into neutral, negative or positive. 
Text: This movie is definitely one of my favorite movies of its kind. The interaction between respectable and morally strong characters is an ode to chivalry and the honor code amongst thieves and policemen.
Sentiment:
"""

sequences = pipe(prompt, max_new_tokens=256)
ic(len(sequences))
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
# %%
torch.manual_seed(1)
prompt = """Return a list of named entities in the text. 
Text: The Golden State Warriors are an American professional basketball team based in San Francisco.
Named entities:
"""

sequences = pipe(prompt, max_new_tokens=256, return_full_text=False)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

# %%
torch.manual_seed(2)
prompt = """Translate the English text to Korean.
Text: Sometimes, I've believed as many as six impossible things before breakfast.
Translation:
"""

sequences = pipe(prompt, max_new_tokens=256, do_sample=True, return_full_text=False)


# %%
def print_sequences(sequences):
    for seq in sequences:
        print(f"Result: \n{seq['generated_text']}")


# %%
torch.manual_seed(3)
prompt = """
Write a summary of the following text within 100 characters. 

Text: Permaculture is a design process mimicking the diversity, functionality and resilience of natural ecosystems. The principles and practices are drawn from traditional ecological knowledge of indigenous cultures combined with modern scientific understanding and technological innovations. Permaculture design provides a framework helping individuals and communities develop innovative, creative and effective strategies for meeting basic needs while preparing for and mitigating the projected impacts of climate change.

Summary:
"""
sequences = pipe(prompt, max_new_tokens=256, do_sample=True, return_full_text=False)
print_sequences(sequences)

# %%
prompt = """Answer the question using the context below.
Context: Gazpacho is a cold soup and drink made of raw, blended vegetables. Most gazpacho includes stale bread, tomato, cucumbers, onion, bell peppers, garlic, olive oil, wine vinegar, water, and salt. Northern recipes often include cumin and/or pimentón (smoked sweet paprika). Traditionally, gazpacho was made by pounding the vegetables in a mortar with a pestle; this more laborious method is still sometimes used as it helps keep the gazpacho cool and avoids the foam and silky consistency of smoothie versions made in blenders or food processors.

Question: What modern tool is used to make gazpacho?
Answer:
"""
sequences = pipe(prompt, max_new_tokens=256, do_sample=True, return_full_text=False)
print_sequences(sequences)

# %%
# Reasoning Test
prompt = """There are 5 groups of students in the class. Each group has 4 students. How many students are there in the class?"""

sequences = pipe(prompt, max_new_tokens=256, do_sample=True, return_full_text=False)
print_sequences(sequences)

# %%
prompt = """I baked 15 muffins. I ate 2 muffins and gave 5 muffins to a neighbor. My partner then bought 6 more muffins and ate 2. How many muffins do we now have?"""

sequences = pipe(prompt, max_new_tokens=256, do_sample=True, return_full_text=False)
print_sequences(sequences)

# %%
