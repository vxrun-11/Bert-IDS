import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import torch

def preprocess_data(data):
    data["text"] = data["protocol_type"].astype(str) + " " + data["service"].astype(str) + " " + data["flag"].astype(str)
    return data

train_data = pd.read_csv("kdd_train_data.csv")
test_data = pd.read_csv("kdd_test_data.csv")

# Preprocess the data to add the "text" column
train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# Load a pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize and encode the text data
max_length = 128

train_encodings = tokenizer(
    train_data["text"].tolist(),
    truncation=True,
    padding=True,
    max_length=max_length,
    return_tensors='pt',
)

test_encodings = tokenizer(
    test_data["text"].tolist(),
    truncation=True,
    padding=True,
    max_length=max_length,
    return_tensors='pt',
)

# Create PyTorch datasets
train_dataset = TensorDataset(
    train_encodings["input_ids"],
    train_encodings["token_type_ids"],
    train_encodings["attention_mask"],
    torch.tensor(train_data["malicious"].values)
)

test_dataset = TensorDataset(
    test_encodings["input_ids"],
    test_encodings["token_type_ids"],
    test_encodings["attention_mask"],
    torch.tensor(test_data["malicious"].values)
)

# Create data loaders
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Load pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Define optimizer and learning rate
optimizer = AdamW(model.parameters(), lr=1e-5, no_deprecation_warning=True)

# Training loop
model.train()
epochs = 0
for epoch in range(epochs):
    for batch in train_dataloader:
        input_ids, token_type_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Testing
test_predictions = []
model.eval()
with torch.no_grad():
    for batch in test_dataloader:
        input_ids, token_type_ids, attention_mask, labels = batch
        outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        logits = outputs.logits
        test_predictions.extend(logits.argmax(dim=1).tolist())

# Calculate test accuracy
test_accuracy = accuracy_score(test_data["malicious"], test_predictions)
print(f"Test Accuracy: {test_accuracy}")

# Save the trained model
model.save_pretrained("kdd_bert_model")

# To load the trained model, you can use the following line:
# model = BertForSequenceClassification.from_pretrained("kdd_bert_model")
