from transformers import DataCollatorForSeq2Seq
from models import init_model
from metrics import get_metric_compute_fn
from data_processors import get_dataset
from trainers import init_trainer
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, choices=['VietAI/vit5-base', 'Helsinki-NLP/opus-mt-vi-en', 'QyQy/VietAi-FinalProject-VIT5', 'datnth1709/finetuned_HelsinkiNLP-opus-mt-vi-en_PhoMT'], help='The name of the pretrained weights from HuggingFace Hub')
    parser.add_argument('--dataset_tokenization_dir', type=str, help='The path to tokenization folder of PhoMT dataset')
    parser.add_argument('--epochs', type=int, default=1, help='Total training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='total batch size for each GPUs')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--data_type', type=str, choices=['helsinki', 'vit5'], default='helsinki', help='Only support either helsinki or vit5 for preprocessing data')
    parser.add_argument('--output_dir', type=str, help='folder name to save training')
    parser.add_argument('--src_max_length', type=int, default=128, help='Max length of input sentence')
    parser.add_argument('--tgt_max_length', type=str, default=128, help='Max length of label sentence')
    parser.add_argument('--accumulation', type=int, default=1, help='Gradient Accumulation steps')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning Rate')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_opt()
    print("PREPARING DATASET")
    dataset = get_dataset(args.data_type, args.dataset_tokenization_dir)
    tokenizer, model = init_model(args.model_name)
    max_input_length = args.src_max_length
    max_target_length = args.tgt_max_length

    print("TOKENIZE DATASET")

    def tokenize_helsinki(examples):
        inputs = [ex["vi"] for ex in examples["translation"]]
        targets = [ex["en"] for ex in examples["translation"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        # Set up the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def tokenize_vit(examples):
        inputs = examples["vi"] 
        targets =examples["en"] 
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        # Set up the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    if args.data_type == "helsinki":
        tokenized_dataset = dataset.map(
            tokenize_helsinki,
            batched=True,
            remove_columns=dataset["validation"].column_names,
        )
    elif args.data_type == "vit5":
        tokenized_dataset = dataset.map(
            tokenize_vit,
            batched=True,
            remove_columns=["en", "vi"]
        )
    else:
        raise Exception("Only support data_type = helsinki or vit5")

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")
    trainer = init_trainer(
        args.data_type, 
        model, 
        tokenizer, 
        tokenized_dataset, 
        data_collator, 
        args.output_dir, 
        epochs=args.epochs, 
        batch_size=args.batch_size,
        lr=args.lr,
        accumulation_steps=args.accumulation
    )
    print("START TRAINING")
    if args.resume == True:
        trainer.train(args.model_name)
    else:
        trainer.train()
