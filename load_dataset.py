from datasets import load_dataset
from transformers import AutoTokenizer

def load_data(training_data_root):
    # Load dataset
    dataset = load_dataset(training_data_root)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Tokenize the data
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Split the dataset
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"]
    return train_dataset, eval_dataset