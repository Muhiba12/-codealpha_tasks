import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical

#  Generate Synthetic Audio Dataset
print("Generating synthetic emotion dataset...")

n_samples = 200  
duration = 2.0    
sr = 22050        
t = np.linspace(0, duration, int(sr*duration))

emotions = ['happy', 'sad', 'angry']
frequencies = [440, 220, 880]   

X = []
y = []

for emotion, freq in zip(emotions, frequencies):
    for _ in range(n_samples):
       
        audio = 0.5*np.sin(2*np.pi*freq*t) + 0.05*np.random.randn(len(t))
        
    
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc = mfcc.T   
        
        X.append(mfcc)
        y.append(emotion)


max_len = max([mf.shape[0] for mf in X])
X_padded = np.array([np.pad(mf, ((0,max_len-mf.shape[0]),(0,0)), mode='constant') for mf in X])

print("Dataset shape:", X_padded.shape)


encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)


X_train, X_test, y_train, y_test = train_test_split(X_padded, y_categorical, test_size=0.2, random_state=42)

print("Training set:", X_train.shape, "Testing set:", X_test.shape)


model = Sequential([
    LSTM(128, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(emotions), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)


y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=emotions))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=emotions, yticklabels=emotions)
plt.title("Confusion Matrix - Emotion Recognition")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
