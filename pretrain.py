import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

model_checkpoint = "pucpr/biobertpt-all"  
train_file = "/proj/berzbiomedicalimagingkth/users/x_zexch/documents/spr2026/train_new.csv"
output_dir = "./mlm_pretrain_ckpt"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

dataset = load_dataset("text", data_files={"train": train_file})

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=384)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=2,
    per_device_train_batch_size=64,
    save_steps=1000,
    save_total_limit=2,
    prediction_loss_only=True,
    fp16=True,
    logging_steps=100,
)

# Trainer
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

class LossLoggerCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        # Clear file at start
        with open(self.log_path, 'w') as f:
            f.write('step\tloss\n')

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is not None and 'loss' in logs:
            with open(self.log_path, 'a') as f:
                f.write(f"{state.global_step}\t{logs['loss']}\n")

loss_log_path = os.path.join(output_dir, 'loss_log.txt')

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
    callbacks=[LossLoggerCallback(loss_log_path)],
)

trainer.train()

# save the model and tokenizer
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)