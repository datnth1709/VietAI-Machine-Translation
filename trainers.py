from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import os
from metrics import get_metric_compute_fn
from transformers.optimization import Adafactor, AdafactorSchedule

DATA_TYPE_HELSINKI = "helsinki"
DATA_TYPE_VIT = "vit5"

def init_trainer(data_type, model, tokenizer, tokenized_datasets, data_collator, output_dir, epochs=5, batch_size=16, lr=2e-5, accumulation_steps=1):
    if data_type not in [DATA_TYPE_HELSINKI, DATA_TYPE_VIT]:
        raise Exception("Only support data_type = helsinki or vit5")
    if data_type == DATA_TYPE_HELSINKI:
        train_args = Seq2SeqTrainingArguments(
            output_dir,
            evaluation_strategy = 'epoch',
            logging_strategy = 'steps',
            logging_steps = 1000,
            eval_steps = 1,                   
            save_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=5,
            num_train_epochs=epochs,
            predict_with_generate=True,
            fp16=True,
        )
        trainer = Seq2SeqTrainer(
            model,
            train_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=get_metric_compute_fn,
        )
        return trainer

    if data_type == DATA_TYPE_VIT:
        train_args = Seq2SeqTrainingArguments(
            output_dir,
            evaluation_strategy = 'steps',
            save_strategy="steps",
            logging_steps = 500,                   
            eval_steps = 5000, 
            save_steps=1000,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=accumulation_steps,
            per_device_eval_batch_size=batch_size,
            save_total_limit=3,
            num_train_epochs=epochs,
            predict_with_generate=True,
            fp16=True,
            half_precision_backend = "auto",
        )
        optim = Adafactor(
            model.parameters(), 
            scale_parameter=True, 
            relative_step=True, 
            warmup_init=True, 
            lr=None
        )
        lr_scheduler = AdafactorSchedule(optim)
        trainer = Seq2SeqTrainer(
            model,
            train_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["dev"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=get_metric_compute_fn,
            optimizers=(optim, lr_scheduler)
        )
        return trainer
    