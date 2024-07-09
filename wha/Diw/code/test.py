import pandas as pd
from pythainlp.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the data
df = pd.read_excel('AI_diw.xlsx')

# Display the first few rows of the dataframe
print(df.head())

# Assuming the dataframe has two columns: 'text' and 'label'
texts = df['message'].astype(str)
labels = df['type']

# Preprocess the text data
def preprocess_text(text):
    return ' '.join(word_tokenize(text, engine='newmm'))

texts = texts.apply(preprocess_text)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test_vec)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

# Save the model to a file
with open('model1.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)

# Save the vectorizer to a file
with open('vectorizer1.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)