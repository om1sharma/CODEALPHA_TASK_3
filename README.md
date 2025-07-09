# CODEALPHA_TASK_3
This project uses a CRNN model trained on MNIST and EMNIST datasets to recognize handwritten digits (0–9) and letters (A–Z). It combines CNNs for feature extraction and LSTMs for sequence learning. The system supports character and word prediction, ideal for OCR and handwriting recognition.
======================================================================================EXTENDED DESCRIPTION==============================================================================================================
✍️ Handwritten Character Recognition using CRNN
CodeAlpha Task 3

📌 Project Overview
This project is focused on building a machine learning model that can recognize handwritten digits (0–9) and uppercase letters (A–Z) from image data. It uses a Convolutional Recurrent Neural Network (CRNN) architecture that combines the strengths of CNNs (for spatial feature extraction) and RNNs (for sequential pattern learning).

The model is trained on two widely-used datasets: MNIST (digits) and EMNIST Letters (uppercase A–Z) to classify handwritten inputs into one of 36 classes (0–9 + A–Z). The final system supports character-wise predictions and can even generate short words or phrases based on consecutive character images.

📂 Datasets Used
MNIST

Contains 60,000 training and 10,000 testing grayscale images of handwritten digits (0–9).

Each image is 28×28 pixels.

EMNIST Letters

Contains uppercase letters (A–Z) extracted from NIST Special Database 19.

Also 28×28 grayscale images.

🔧 Preprocessing Steps
Normalized pixel values to [0, 1]

Reshaped images to fit the model input

Adjusted EMNIST labels from (1–26) to (10–35) for proper classification

Combined and balanced both datasets

Applied data augmentation using ImageDataGenerator for better generalization

One-hot encoded all output labels for training

🧠 Model Architecture – CRNN
The model is built using TensorFlow Keras, and consists of:

CNN layers for extracting spatial features:

Conv2D, BatchNormalization, MaxPooling, and Dropout

Reshape layer to convert CNN output to time-series format

Bidirectional LSTM layers for sequential learning

Dense layers with softmax output for final classification

This hybrid architecture allows the model to understand both visual patterns and temporal context, which is especially helpful for handwritten characters that vary in shape.

📊 Evaluation
Evaluated on a balanced test set containing digits and letters

Metrics used:

Accuracy

Confusion matrix

Loss/Accuracy plots

High accuracy achieved across all character classes

==>🔍 Features & Capabilities
Predict individual digits or letters from input images

Predict and visualize short sequences (like words or codes)

Sample outputs include prediction of:

Digits only

Letters only

Mixed characters

Reconstructed sentences

🧪 Technologies Used
Python 3.11

TensorFlow 2.19

TensorFlow Datasets

NumPy, Matplotlib

Scikit-learn

