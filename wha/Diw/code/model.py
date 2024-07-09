import pandas as pd
from pythainlp.corpus import thai_stopwords
from pythainlp.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

# ฟังก์ชันสำหรับการตัดคำภาษาไทยและลบ stop words
def tokenize_and_remove_stopwords(text):
    stop_words = set(thai_stopwords())
    tokens = word_tokenize(text, engine='newmm')
    return [token for token in tokens if token not in stop_words]

# โหลดข้อมูลจากไฟล์ Excel
df = pd.read_excel('ModelGas.xlsx')

# แสดงข้อมูลตัวอย่าง
print(df.head())

# สมมุติว่า DataFrame มีสองคอลัมน์: 'message' และ 'type'
X = df['message']
y = df['type']

# การแปลงข้อความเป็น Bag of Words
vectorizer = CountVectorizer(tokenizer=tokenize_and_remove_stopwords)
X_vectorized = vectorizer.fit_transform(X)

# แบ่งข้อมูลสำหรับ train และ test
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# สร้างและเทรนโมเดล Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# ทำนายผลและประเมินผลโมเดล
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# # Save the model to a file
with open('LogisticRegression.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the vectorizer to a file
with open('vectorizer_naive_bayes.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)