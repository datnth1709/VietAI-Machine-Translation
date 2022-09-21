from cmath import exp
import os
import jsonlines
from datasets import load_dataset

DATA_TYPE_HELSINKI = "helsinki"
DATA_TYPE_VIT = "vit5"

def preprocess_data_helsinki(tokenization_path):
    subsets = ["train", "test", "dev"]
    langs = ["en", "vi"]
    for subset in subsets:
        temp = {}
        data = []
        for lang in langs:
            path = os.path.join(tokenization_path,subset,f"{subset}.{lang}")
            with open(path, "r", encoding="utf-8") as f:    
                contents = f.readlines()
            print(f"{path}:", len(contents), "lines")
            for i in range(len(contents)):
                line = contents[i].strip()
                if line[-1:] == "\n":
                    line = line[:-1]
                contents[i] = line
            temp[lang] = contents
        for en, vi in zip(temp["en"], temp["vi"]):
            data.append({
                "translation": {
                "en": en,
                "vi": vi
                }
            })
        with jsonlines.open(os.path.join(tokenization_path, f"{subset}.json"), mode="w") as writer:        
            writer.write_all(data)
    dataset = load_dataset(
        "json", 
        data_files={
            "train":os.path.join(tokenization_path, "train.json"),
            "validation":os.path.join(tokenization_path, "dev.json"),
            "test":os.path.join(tokenization_path, "test.json")
        }
    )
    return dataset

def preprocess_data_vit(tokenization_path):
    subsets = ["train", "dev", "test"]
    langs = ["en", "vi"]
    for subset in subsets:
        temp = {}
        data = []
        for lang in langs:
            path = os.path.join(tokenization_path, subset,f"{subset}.{lang}")
            with open(path, "r", encoding='utf-8') as f:    
                contents = f.readlines()
            print(f"{path}:", len(contents), "lines")
            for i in range(len(contents)):
                line = contents[i].strip()
                if line[-1:] == "\n":
                    line = line[:-1]
                if lang == "vi":
                    contents[i] = "translate Vietnamese to English: " + line + " </s>"
                else:
                    contents[i] = line
            temp[lang] = contents
        for en, vi in zip(temp["en"], temp["vi"]):
            data.append({
                "en": en,
                "vi": vi
            })
        with jsonlines.open(os.path.join(tokenization_path, f'{subset}.jsonl'), mode='w') as writer:        
            writer.write_all(data)
    dataset = load_dataset(
        "json", 
        data_files={
            "train":os.path.join(tokenization_path, "train.jsonl"),
            "dev":os.path.join(tokenization_path, "dev.jsonl"),
            "test":os.path.join(tokenization_path, "test.jsonl")
        }
    )
    return dataset

def get_dataset(data_type, data_path):
    if data_type not in [DATA_TYPE_HELSINKI, DATA_TYPE_VIT]:
        raise Exception("Only support data_type = helsinki or vit5")
    if data_type == DATA_TYPE_HELSINKI:
        return preprocess_data_helsinki(data_path)
    if data_type == DATA_TYPE_VIT:
        return preprocess_data_vit(data_path)
