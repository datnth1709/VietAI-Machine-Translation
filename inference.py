from transformers import pipeline, TranslationPipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import argparse
import torch
device = torch.device(0 if torch.cuda.is_available() else "cpu")
print(device)
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, choices=['QyQy/VietAi-FinalProject-VIT5', 'datnth1709/finetuned_HelsinkiNLP-opus-mt-vi-en_PhoMT'], help='The name of the pretrained weights from HuggingFace Hub')
    parser.add_argument('--text', type=str, help='The text to translate')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_opt()

    model_checkpoint = args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    translator = TranslationPipeline(model=model, tokenizer=tokenizer,clean_up_tokenization_spaces=True, device=device)
    output = translator(args.text)[0]['translation_text']

    print(output)