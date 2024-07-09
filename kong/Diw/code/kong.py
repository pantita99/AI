import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# โหลดข้อมูลจากไฟล์ Excel
df = pd.read_excel('AI_diw.xlsx')

# ตรวจสอบประเภทของคอลัมน์ labels
print(df['tag'].dtype)

# ถ้าหากคอลัมน์ labels เป็น string ให้ทำการแปลงเป็นตัวเลข
# สมมติว่ามี 4 คลาส: ['class1', 'class2', 'class3', 'class4']
label_mapping = {'AskDate': 0, 'Greet': 1, 'AskAI': 2, 'Gas': 3}
df['tag'] = df['tag'].map(label_mapping)

# แบ่งข้อมูลเป็น train และ test sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# โหลด tokenizer และ model
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=8)

# สร้าง Dataset สำหรับการฝึก
class Dataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_texts = train_data['patterns'].tolist()
train_labels = train_data['tag'].tolist()
test_texts = test_data['patterns'].tolist()
test_labels = test_data['tag'].tolist()

train_dataset = Dataset(train_texts, train_labels)
test_dataset = Dataset(test_texts, test_labels)

# การตั้งค่าและฝึกโมเดล
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

# ประเมินผลโมเดล
trainer.evaluate()

# บันทึกโมเดลและ tokenizer
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')
