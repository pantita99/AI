import pandas as pd
from pythainlp import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# อ่านข้อมูลจากไฟล์ Excel
df = pd.read_excel('C:\\Users\\kong0\\Desktop\\AI\\kong\\Diw\\data\\Gas.xlsx')

# แยกข้อความและป้ายกำกับ
data_X = df['message']
data_y = df['type']

# ฟังก์ชันสำหรับการตัดคำ
def tokenize(text):
    return word_tokenize(text, engine='newmm')

# สร้าง Vectorizer สำหรับแปลงข้อความเป็น feature vectors
vectorizer = CountVectorizer(tokenizer=tokenize)

# แปลงข้อความ
X = vectorizer.fit_transform(data_X)

# แปลงป้ายกำกับเป็นตัวเลข
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data_y)
print(y)

# แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างโมเดล Naive Bayes
model = MultinomialNB()

# ฝึกโมเดล
model.fit(X_train, y_train)

# ทำนายผล
y_pred = model.predict(X_test)

# แสดงผลลัพธ์
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
