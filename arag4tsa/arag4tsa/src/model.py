from transformers import Trainer, TrainingArguments, QLoRA

def fine_tune_model(model, train_dataset, val_dataset):
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        evaluation_strategy="epoch",
        save_total_limit=2,
        learning_rate=1e-5,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    qlora = QLoRA(
        low_rank=16, 
        alpha=32, 
        dropout=0.05
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=(qlora, None)
    )

    trainer.train()