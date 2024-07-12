import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# โหลดข้อมูล
df = pd.read_excel('C:\\Users\\kong0\\Desktop\\AI\\kong\\Diw\\data\\Gas.xlsx')
data_X = df['message'].tolist()
data_y = df['type'].tolist()

# แปลง labels เป็นตัวเลข
label_encoder = LabelEncoder()
data_y_encoded = label_encoder.fit_transform(data_y)

# กำหนดพารามิเตอร์
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

# โหลด pre-trained BERT model และ tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
num_labels = len(label_encoder.classes_)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# แบ่งข้อมูลเป็น train และ test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(data_X, data_y_encoded, test_size=0.2, random_state=42)

# เตรียมข้อมูลสำหรับ BERT
def prepare_data(texts, labels):
    input_ids = []
    attention_masks = []
    
    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return TensorDataset(input_ids, attention_masks, labels)

# สร้าง DataLoader
train_dataset = prepare_data(train_texts, train_labels)
test_dataset = prepare_data(test_texts, test_labels)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# เตรียม optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(EPOCHS):
    model.train()
    for batch in train_dataloader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}/{EPOCHS} completed')
    

# Evaluation
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        
        outputs = model(input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, dim=1)
        
        predictions.extend(preds.cpu().tolist())
        true_labels.extend(labels.cpu().tolist())

accuracy = accuracy_score(true_labels, predictions)
print(f'Test Accuracy: {accuracy:.4f}')

# แปลงกลับเป็น labels เดิม
predictions_original = label_encoder.inverse_transform(predictions)
true_labels_original = label_encoder.inverse_transform(true_labels)

print("ตัวอย่างการทำนาย:")
for pred, true in zip(predictions_original[:5], true_labels_original[:5]):
    print(f"ทำนาย: {pred}, ค่าจริง: {true}")