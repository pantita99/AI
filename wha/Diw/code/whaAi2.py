import pandas as pd #สำหรับการจัดการเอกสารและวิเคราห์ข้อมูล โครงสร้างหลักคือ datafream จัดการข้อมูลในตารางได้ดี
from pythainlp.corpus import thai_stopwords #ตัดคำหรือกรองคำที่ไม่สำคัญ และ หรือ ที่
from pythainlp.tokenize import word_tokenize #แยกคำออกเป็นคำๆ 
from sklearn.feature_extraction.text import CountVectorizer #แปลงข้อความเป็นรูปแบบที่เหมาะกับการวิเคราะห์ จะนับจำนวนคำและสร้างเวกเตอร์
from sklearn.model_selection import train_test_split #แบ่งข้อมูลออกเป็นชุดฝึกและทดสอบ
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score, classification_report #ผลลัพธ์

data = pd.read_excel('C:\\Users\\mhewwha\\OneDrive\\Desktop\\AI\\wha\\Diw\\data\\Gas1.xlsx')


X = data['message']
y = data ['type']


X_train,X_test,y_train,y_test=train_test_split(X,y, test_size = 0.2, random_state=42)


def tokenize_and_remove_stopwords(text):
    stop_word = set(thai_stopwords())
    tokens = word_tokenize(text,engine='newmm')
    return[token for token in tokens if token not in stop_word]


#สร้าง CountVectorizer
vectorizer = CountVectorizer(tokenizer=tokenize_and_remove_stopwords) #ฟังก์ชันสำหรับการตัดคำและการกรองคำที่ไม่สำคัญ (stop words) ก่อนที่จะทำการแปลงข้อความเป็นเวกเตอร์


#แปลงงข้อความเป็น เวกเตอร์ 
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized =  vectorizer.transform(X_test)


#สร้างโมเดล 
model = SVC(kernel='linear')
#ฝึกโมเดล
model.fit(X_train_vectorized,y_train)

#ทำนาย
y_pred = model.predict(X_test_vectorized)

#ประเมินผล
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))







    




