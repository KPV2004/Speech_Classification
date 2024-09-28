import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load a .wav file and extract Mel-Spectrogram features (faster than MFCC)
def extract_features(file_path, n_mels=128):
    y, sr = librosa.load(file_path, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    return np.mean(librosa.power_to_db(mel_spectrogram, ref=np.max).T, axis=0)

# Function to load or extract features and save them to .npy files
def load_or_extract_features(data_dir, label_map, file_prefix):
    features = []
    labels = []
    npy_file = f'{file_prefix}_features.npy'
    npy_labels = f'{file_prefix}_labels.npy'

    # Check if precomputed features exist
    if os.path.exists(npy_file) and os.path.exists(npy_labels):
        features = np.load(npy_file)
        labels = np.load(npy_labels)
    else:
        for label in os.listdir(data_dir):
            label_dir = os.path.join(data_dir, label)
            for file_name in os.listdir(label_dir):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(label_dir, file_name)
                    feature = extract_features(file_path)
                    features.append(feature)
                    labels.append(label_map[label])

        features = np.array(features)
        labels = np.array(labels)

        # Save features to .npy file
        np.save(npy_file, features)
        np.save(npy_labels, labels)

    return features, labels


# Define the directories
train_data_dir = '../Dataset/Train/'
test_data_dir = '../Dataset/Test/'

# Get the list of labels from training data
labels = [label for label in os.listdir(train_data_dir)]
label_map = {label: i for i, label in enumerate(labels)}

# Load or extract training features
X_train, y_train = load_or_extract_features(train_data_dir, label_map, 'train')

# Load or extract test features
X_test_new, y_test_new = load_or_extract_features(test_data_dir, label_map, 'test')

# Build a simplified model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),  # Batch normalization speeds up training
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(labels), activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with fewer epochs
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test_new, y_test_new))

# Evaluate the model on the test dataset
test_loss_new, test_acc_new = model.evaluate(X_test_new, y_test_new)
print(f'New Test Accuracy: {test_acc_new}')



