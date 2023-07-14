#!/usr/bin/env python
# coding: utf-8

# Required Libraries

# In[17]:


import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[66]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
import os as os
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping


# In[19]:


# data = pd.read_csv('your_data.csv') # Alter accordingly
audio_dir = '/Users/jovanwong/Pictures/Desktop/GroundUp AI/ML classifier/audios_split'  # replace with your directory


# Generate the MFCCs

# In[20]:


# Load the audio files and generate the MFCC
mfccs = []
labels = []
for filename in os.listdir(audio_dir):
    if filename.endswith(".mp3"):  # assuming the files are in .mp3 format
        file_path = os.path.join(audio_dir, filename)
        y, sr = librosa.load(file_path)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        mfccs.append(mfcc)
        label = filename.split('(')[1].split(')')[0]  # extract label from filename
        labels.append(label)
    
# Convert lists to numpy arrays
mfccs = np.array(mfccs)
labels = np.array(labels)


# Train test split

# In[30]:


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(mfccs, labels, test_size=0.8, random_state=42)
# print(y_train)
# print(y_test)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
print(X_train)
print(X_test)
# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode the target labels
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
print(y_train_encoded)
print(y_test_encoded)


# Crafting neural network model with 256 neurons, dropout rate of 0.5.
# Rectified linear unit, and sigmoid function

# In[68]:


# Model architecture
model = Sequential()
model.add(Flatten(input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))  # binary classification

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])


# Compile and train the model

# In[71]:


# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Define early stopping callback
early_stopping = EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True)

# Train the model
model.fit(X_train, y_train_encoded, batch_size=32, epochs=20, validation_data=(X_test, y_test_encoded))

# Evaluate the model
score = model.evaluate(X_test, y_test_encoded, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# Finding best model

# In[72]:


# Define the model
def create_model(optimizer='adam', dropout_rate=0.5):
    model = Sequential()
    model.add(Flatten(input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Define the hyperparameters to search
optimizers = ['adam', 'rmsprop']
dropout_rates = [0.2, 0.4, 0.6]

best_model = None
best_val_loss = float('inf')

# Perform grid search
for optimizer in optimizers:
    for dropout_rate in dropout_rates:
        model = create_model(optimizer=optimizer, dropout_rate=dropout_rate)
        model.fit(X_train, y_train_encoded, batch_size=32, epochs=20, verbose=0)
        val_loss, val_acc = model.evaluate(X_test, y_test_encoded, verbose=0)
        print(f"Optimizer: {optimizer}, Dropout Rate: {dropout_rate}, Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")
        if val_loss < best_val_loss:
            best_model = model
            best_val_loss = val_loss

# Evaluate the best model
train_loss, train_acc = best_model.evaluate(X_train, y_train_encoded, verbose=0)
val_loss, val_acc = best_model.evaluate(X_test, y_test_encoded, verbose=0)
print('Best Model - Train loss:', train_loss)
print('Best Model - Train accuracy:', train_acc)
print('Best Model - Validation loss:', val_loss)
print('Best Model - Validation accuracy:', val_acc)


# Loss vs Val Loss plot

# In[73]:


# Train the model
history = model.fit(X_train, y_train_encoded, batch_size=32, epochs=20, validation_data=(X_test, y_test_encoded))

# Extract loss values from history
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Create plot of training loss and validation loss
epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

