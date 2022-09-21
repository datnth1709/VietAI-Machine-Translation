from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def init_model(pretrained_name):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_name)
    return tokenizer, model