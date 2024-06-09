import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

def train(train_dataset, eval_dataset, lr, train_batch, test_batch, epochs, decay, output_dir):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)

    # Training arguments

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=train_batch,
        per_device_eval_batch_size=test_batch,
        num_train_epochs=epochs,
        weight_decay=decay,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Train the model
    trainer.train()

    return trainer

def evaluate(trainer):
    # Evaluate the model
    results = trainer.evaluate()
    print(f"results{results}")