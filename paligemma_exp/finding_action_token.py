# %%
from math import pi
import sentencepiece as spm
import os
from icecream import ic

#%%
TOKENIZER_PATH = "./paligemma_tokenizer.model"
if not os.path.exists(TOKENIZER_PATH):
    print("Downloading the model tokenizer...")
    !gsutil cp gs://big_vision/paligemma_tokenizer.model {TOKENIZER_PATH}
    print(f"Tokenizer path: {TOKENIZER_PATH}")
else:
    print(f"Tokenizer file: {TOKENIZER_PATH} is already downloaded")


# %%
sp = spm.SentencePieceProcessor(TOKENIZER_PATH)

# encode: text => id
print(sp.EncodeAsPieces("This is a test"))
print(sp.EncodeAsPieces("This is a test_case"))
print(sp.EncodeAsIds("This is a test"))
print(sp.EncodeAsIds("Hello World"))

# decode: id => text
print(sp.DecodePieces(["This", "▁is", "▁a", "▁t", "est"]))

# print(sp.decode_ids([209, 31, 9, 375, 586]))
print(sp.DecodeIds([1596, 603, 476, 2121]))


# %%
sp.GetPieceSize()
# %%
reserved_size = 256

last_id = sp.GetPieceSize() - 1
ic(last_id)
ids = [i for i in range(last_id-reserved_size, last_id+1)]
ic(ids)
# print(sp.DecodeIds(ids))

for i in range(last_id-reserved_size, last_id):
    piece = sp.IdToPiece(i)
    print(f"id:{i} -->piece:{piece}")
    
# %%
special_ids = [sp.bos_id(), sp.eos_id(), sp.pad_id(), sp.unk_id()]
ic(sp.IdToPiece([i for i in special_ids]))

# %%
real_last_id = sp.GetPieceSize() - 1
ic(real_last_id)

# 257_151

#%%
ic(real_last_id)

reserved_size = 255
start_id = 255_700
last_id = start_id + reserved_size

num_to_id = {}
id_to_num = {}
pieces = [] 

for n, i in enumerate(range(last_id-reserved_size, last_id+1)):
    piece = sp.IdToPiece(i)
    print(f"id({n}):{i} -->piece:{piece}, bytes:{piece.encode('utf-8')}")
    pieces.append(piece)

#%%
pieces_str = sp.DecodePieces(pieces)
# print(pieces_str)

ic(len(pieces_str))
ic(pieces_str)


# %%
# Test full string conversion 
# real value (meter dimension or degree for rotation)
# action value -> id -> token

# scale for translation  1/128 (meter)
# scale for rotation 180./128 (degree)
# action value range(-scale, scale) -> action value (-128, 128)

# %%
