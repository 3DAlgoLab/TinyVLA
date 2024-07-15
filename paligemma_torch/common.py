from datetime import datetime
from datasets import load_dataset
from pathlib import Path
from datasets import Image as dsImage

def generate_timestamp():
    """
    Example usage:
        formatted_timestamp = generate_timestamp()
        print("Formatted timestamp:", formatted_timestamp)
    """
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M")
    return timestamp


def load_vla_data_from_folder(folder:str, split='train'):
    anno_file = str(Path(folder) / "_annotations.jsonl")    
    ds_raw = load_dataset('json', data_files={split:anno_file})
    def add_abs_path(example):
        example["image"] = str(Path(folder) / example["image"])
        return example
    
    ds_raw = ds_raw.map(add_abs_path)
    ds_raw = ds_raw.cast_column("image", dsImage(mode="RGB"))
    return ds_raw


