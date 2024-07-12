import pandas as pd
from pythainlp.corpus import thai_stopwords
from pythainlp.tokenize import word_tokenize
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer  #ช่วยแปลงข้อความเป็นเวกเตอร์


#โหลดข้อมูลจากexcel
data = pd.read_excel('')


X = data['message'] 
y = data['type'] 


#แบ่งข้อมูลเป็นชุดฝึกและทดสอบ

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#ฟังก์ชั่นตัดคำไทยและลบstop word
def tokenize_and_remove_stopwords(text): #กำหนดฟังก์ชั่น
    stop_word = set(thai_stopwords())
    tokens = word_tokenize (text,engine='newmm')
    return [token for token in tokens if token not in stop_word]


#สร้าง CountVectorizer
vectorizer = CountVectorizer(tokenizer=tokenize_and_remove_stopwords)

#แปลงข้อความเป็นเวกเตอร์

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

0
                             

#สร้างโมเดล  
model = GaussianNB()

#ฝึกโมเดล
model.fit(X_train_vectorized.toarray(), y_train)

#ทำนายต่อจากชุดทดสอบ
y_pred = model.predict(X_test_vectorized.toarray())

#ประเมินโมเดล
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.2f}") 









