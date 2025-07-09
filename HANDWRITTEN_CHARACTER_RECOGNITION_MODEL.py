#!/usr/bin/env python
# coding: utf-8

# In[9]:


import sys
print(sys.executable)
print(sys.version)


# # CODEALPHA TASK-3: HANDWRITTEN CHARACTER RECOGNITION

# In[13]:


import sys
import pkg_resources

# List of required packages
required_packages = [
    'numpy',
    'matplotlib',
    'tensorflow',
    'tensorflow-datasets'
]

# Check each package
for package in required_packages:
    try:
        dist = pkg_resources.get_distribution(package)
        print(f"✅ {package} is installed (Version: {dist.version})")
    except pkg_resources.DistributionNotFound:
        print(f"❌ {package} is NOT installed")


# In[12]:


import sys
get_ipython().system('{sys.executable} -m pip install tensorflow-datasets')


# In[14]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load MNIST
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()

# Print MNIST info
print("MNIST Dataset Info:")
print(f"Training images shape: {x_train_mnist.shape}")  # (60000, 28, 28)
print(f"Training labels shape: {y_train_mnist.shape}")  # (60000,)
print("\nFirst 5 MNIST training samples:")

# Display first 5 samples
for i in range(5):
    print(f"Label: {y_train_mnist[i]}")
    print(f"Image shape: {x_train_mnist[i].shape}")
    print(f"Pixel range: {np.min(x_train_mnist[i])}-{np.max(x_train_mnist[i])}\n")


# In[15]:


import tensorflow_datasets as tfds

# Load EMNIST Letters
emnist_train, emnist_test = tfds.load(
    'emnist/letters',
    split=['train', 'test'],
    as_supervised=True,
    shuffle_files=True
)

# Convert to numpy (first 5 samples only)
print("\nEMNIST Letters Dataset Info:")
emnist_samples = list(emnist_train.take(5))  # Get first 5 samples

for i, (image, label) in enumerate(emnist_samples):
    print(f"\nSample {i+1}:")
    print(f"Label (before adjustment): {label}")  # A=1, B=2,... (will be adjusted to 0-25 later)
    print(f"Image shape: {image.shape}")  # (28, 28, 1)
    print(f"Pixel range: {np.min(image)}-{np.max(image)}")


# In[17]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.utils import to_categorical
import tensorflow_datasets as tfds
from sklearn.utils import resample
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==================== ENVIRONMENT CHECK ====================
print("🔍 TensorFlow:", tf.__version__)
print("🔍 NumPy:", np.__version__)

# ==================== LOAD DATASETS ====================
def load_datasets():
    print("📥 Loading MNIST...")
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()

    print("📥 Loading EMNIST Letters...")
    emnist_train, emnist_test = tfds.load('emnist/letters', split=['train', 'test'], as_supervised=True)

    def convert_emnist(ds):
        images, labels = [], []
        for img, label in tfds.as_numpy(ds):
            images.append(img)
            labels.append(label)
        return np.array(images), np.array(labels)

    x_train_emnist, y_train_emnist = convert_emnist(emnist_train)
    x_test_emnist, y_test_emnist = convert_emnist(emnist_test)

    return (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist), (x_train_emnist, y_train_emnist), (x_test_emnist, y_test_emnist)

(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist), (x_train_emnist, y_train_emnist), (x_test_emnist, y_test_emnist) = load_datasets()

# ==================== PREPROCESS ====================
def preprocess(images, labels):
    images = images.astype("float32") / 255.0
    if images.ndim == 3:
        images = np.expand_dims(images, axis=-1)
    return images, labels

x_train_mnist, y_train_mnist = preprocess(x_train_mnist, y_train_mnist)
x_test_mnist, y_test_mnist = preprocess(x_test_mnist, y_test_mnist)
x_train_emnist, y_train_emnist = preprocess(x_train_emnist, y_train_emnist)
x_test_emnist, y_test_emnist = preprocess(x_test_emnist, y_test_emnist)

# Adjust EMNIST labels (1–26 → 10–35 for A-Z)
y_train_emnist = y_train_emnist - 1 + 10
y_test_emnist = y_test_emnist - 1 + 10

# ==================== BALANCE DATA ====================
x_train_emnist, y_train_emnist = resample(x_train_emnist, y_train_emnist,
                                          replace=False, n_samples=len(x_train_mnist),
                                          stratify=y_train_emnist, random_state=42)

x_test_emnist, y_test_emnist = resample(x_test_emnist, y_test_emnist,
                                        replace=False, n_samples=len(x_test_mnist),
                                        stratify=y_test_emnist, random_state=42)

# ==================== COMBINE & ENCODE ====================
x_train = np.concatenate([x_train_mnist, x_train_emnist])
y_train = np.concatenate([y_train_mnist, y_train_emnist])
x_test = np.concatenate([x_test_mnist, x_test_emnist])
y_test = np.concatenate([y_test_mnist, y_test_emnist])

num_classes = 36  # Digits 0–9 and Letters A–Z
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# ==================== AUGMENTATION ====================
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(x_train)

# ==================== BUILD CRNN ====================
def build_crnn_model(input_shape=(28, 28, 1), num_classes=36):
    model = models.Sequential([
        Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Reshape((7, 7 * 64)),  # 7x7x64 -> (7, 448)

        layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)),
        layers.Bidirectional(layers.LSTM(64, dropout=0.3, recurrent_dropout=0.3)),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_crnn_model()
model.summary()

# ==================== TRAIN ====================
print("\n🚀 Training model...")
history = model.fit(
    datagen.flow(x_train, y_train_cat, batch_size=128),
    epochs=25,
    validation_data=(x_test, y_test_cat),
    verbose=1
)

# ==================== PLOT TRAINING ====================
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Training Acc')
    plt.plot(epochs, val_acc, 'g--', label='Validation Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r--', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_training_history(history)

# ==================== EVALUATE ====================
print("\n📊 Evaluating on test set...")
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=2)
print(f"✅ Final Test Accuracy: {test_acc*100:.2f}%")

# ==================== PREDICT & SHOW ====================
def decode_label(label):
    return str(label) if label < 10 else chr(ord('A') + label - 10)

def show_predictions(model, images, labels, num=10):
    plt.figure(figsize=(15, 3))
    for i in range(num):
        img = images[i]
        true = labels[i]
        pred = model.predict(np.expand_dims(img, axis=0), verbose=0)
        pred_label = np.argmax(pred)
        plt.subplot(1, num, i + 1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f"True: {decode_label(true)}\nPred: {decode_label(pred_label)}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# ==================== SAMPLE PREDICTIONS ====================
print("\n🔍 Sample Predictions (Digits & Letters):")
mixed_indices = np.random.choice(len(x_test), 10, replace=False)
show_predictions(model, x_test[mixed_indices], y_test[mixed_indices], num=10)

# ==================== SENTENCE PREDICTION ====================
print("\n✍️ Predicted Sentence:")
sentence_indices = np.random.choice(len(x_test), 8, replace=False)
plt.figure(figsize=(16, 2))
sentence = ""
for i, idx in enumerate(sentence_indices):
    img = x_test[idx]
    pred = np.argmax(model.predict(np.expand_dims(img, axis=0), verbose=0))
    char = decode_label(pred)
    sentence += char
    plt.subplot(1, 8, i + 1)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(f"{char}")
    plt.axis('off')
plt.suptitle(f"Predicted Sentence: {sentence}", fontsize=16)
plt.tight_layout()
plt.show()

# ==================== EXAMPLES: DIGITS ====================
print("\n🔢 Predicting Digits 0–9:")
digit_indices = np.where(y_test < 10)[0]
np.random.shuffle(digit_indices)
show_predictions(model, x_test[digit_indices[:4]], y_test[digit_indices[:4]], num=4)

# ==================== EXAMPLES: LETTERS ====================
print("\n🔤 Predicting Letters A–Z:")
letter_indices = np.where((y_test >= 10) & (y_test < 36))[0]
np.random.shuffle(letter_indices)
show_predictions(model, x_test[letter_indices[:4]], y_test[letter_indices[:4]], num=4)

# ==================== EXAMPLES: MIXED ====================
print("\n🔀 Predicting Mixed Characters:")
mixed_indices = np.random.choice(len(x_test), 4, replace=False)
show_predictions(model, x_test[mixed_indices], y_test[mixed_indices], num=4)

# ==================== SAVE MODEL ====================
model.save("crnn_emnist_mnist_model.h5")
print("💾 Model saved as crnn_emnist_mnist_model.h5")

