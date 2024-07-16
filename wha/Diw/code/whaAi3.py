import pandas as pd 
from pythainlp.corpus import thai_stopwords 
from pythainlp.tokenize import word_tokenize 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report 
import pickle 


data = pd.read_excel('C:\\Users\\mhewwha\\OneDrive\\Desktop\\AI\\wha\\Diw\\data\\law.xlsx')


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
model3 = LogisticRegression()
#ฝึกโมเดล
model3.fit(X_train_vectorized, y_train)

#ทำนาย
y_pred = model3.predict(X_test_vectorized)

#ประเมินผล
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))


print("Vectorizer saved as vectorizer1.pkl") #สั่งให้save
with open('vectorizer3.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('svm_model3.pkl', 'wb') as f:
    pickle.dump(model3, f)
   







