import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load a .wav file and extract features (MFCCs or Mel Spectrograms)
def extract_features(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)  # Take mean of MFCCs over time

# Dataset directory and labels
data_dir = '../Dataset/Train/'
labels = []  # Replace with your actual categories
file_paths = []
file_labels = []

# # Load files and extract features
# features = []
# for label in labels:
#     label_dir = os.path.join(data_dir, label)
#     for file_name in os.listdir(label_dir):
#         if file_name.endswith('.wav'):
#             file_path = os.path.join(label_dir, file_name)
#             feature = extract_features(file_path)
#             features.append(feature)
#             file_labels.append(label)

# Load files and extract features
features = []
for label in os.listdir(data_dir):
    labels.append(label)
    label_dir = os.path.join(data_dir, label)
    for file_name in os.listdir(label_dir):
        if file_name.endswith('.wav'):
            file_path = os.path.join(label_dir, file_name)
            feature = extract_features(file_path)
            features.append(feature)
            file_labels.append(label)

# Convert labels to integers
label_map = {label: i for i, label in enumerate(labels)}
labels_numeric = [label_map[l] for l in file_labels]

# Convert to NumPy arrays for training
X = np.array(features)
y = np.array(labels_numeric)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),  # Input layer with the shape of features
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(labels), activation='softmax')  # Output layer for classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc}')
