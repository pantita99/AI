import pandas as pd
import numpy as np
import tensorflow as tf
from pythainlp.corpus import thai_stopwords
from pythainlp.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import pickle

# Load the data
df = pd.read_excel('ModelGas.xlsx')
data_X = df['message']
data_y = df['type']

# Define stopwords
stopwords = thai_stopwords()

# Preprocessing function
def preprocess(text):
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stopwords]
    return ' '.join(tokens)

# Apply preprocessing
data_X = data_X.apply(preprocess)

# Vectorize text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data_X)

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(data_y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=len(np.unique(y)))
y_test = to_categorical(y_test, num_classes=len(np.unique(y)))

# Reshape X to have a single feature dimension
X_train = X_train.toarray().reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.toarray().reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the model
model = Sequential()
model.add(Conv1D(128, 5, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.35))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(len(np.unique(y)), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=200, batch_size=16, validation_data=(X_test, y_test))

# Evaluate the model
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
acc = accuracy_score(y_test.argmax(axis=1), y_pred)
print("Accuracy: {:.2f}%".format(acc * 100))

# Show confusion matrix
cm = confusion_matrix(y_test.argmax(axis=1), y_pred)
print("Confusion matrix:")
print(cm)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

#model.save('text_neural_network1.h5')



