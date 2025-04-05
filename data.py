from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import pandas as pd

df = pd.read_csv("pm_training_data.csv")

# Combine goal and working_memory into one input text field
df["input_text"] = df["goal"] + " | " + df["working_memory"]

# Shuffle and split the dataset
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
split_index = int(len(df_shuffled) * 0.8)
df_train = df_shuffled[:split_index]
df_val = df_shuffled[split_index:]

# Convert to Hugging Face Dataset
dataset_train = Dataset.from_pandas(df_train[["input_text", "correct_agent"]])
dataset_val = Dataset.from_pandas(df_val[["input_text", "correct_agent"]])

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Preprocessing function
def preprocess(batch):
    texts = [str(x) if x is not None else "" for x in batch["input_text"]]
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=128
    )


# Encode labels and tokenize
dataset_train = dataset_train.class_encode_column("correct_agent")
dataset_val = dataset_val.class_encode_column("correct_agent")

label_names = dataset_train.features["correct_agent"].names

num_labels = len(label_names)

encoded_train = dataset_train.map(preprocess, batched=True)
encoded_val = dataset_val.map(preprocess, batched=True)

encoded_train = encoded_train.rename_column("correct_agent", "labels")
encoded_val = encoded_val.rename_column("correct_agent", "labels")

encoded_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
encoded_val.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Load model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

# Training arguments
training_args = TrainingArguments(
    output_dir="./pm-agent-selector",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# Trainer setup
def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_train,
    eval_dataset=encoded_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Start training
trainer.train()

# Save the model
trainer.save_model("./pm-agent-selector")