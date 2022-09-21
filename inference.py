from transformers import pipeline
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, choices=['QyQy/VietAi-FinalProject-VIT5', 'datnth1709/finetuned_HelsinkiNLP-opus-mt-vi-en_PhoMT'], help='The name of the pretrained weights from HuggingFace Hub')
    parser.add_argument('--text', type=str, help='The text to translate')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_opt()
    translator = pipeline("translation", model=args.checkpoint)
    output = translator(args.text)[0]['translation_text']
    print(output)