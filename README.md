# Audio Sentiment Detection using LSTM
This project builds an LSTM-based deep learning model to classify emotions from audio using the RAVDESS dataset. The system processes MFCC features extracted from .wav files and predicts one of the following emotions:

neutral, calm, happy, sad, angry, fearful, disgust, surprised

# LSTM for Audio Sentiment Detection:

1. Audio = Sequential Data
   
Audio signals are time-series — they change over time.

Emotional tone depends on how pitch, energy, and rhythm evolve.

So we need a model that remembers past information — not just see a snapshot.

3. LSTM = Memory-Powered Neural Network

LSTM stands for Long Short-Term Memory, a special type of RNN.

Unlike simple neural nets or CNNs, LSTMs can store long-term dependencies.

They’re designed to avoid the vanishing gradient problem seen in basic RNNs.


# Dataset: RAVDESS

8 emotion categories

24 actors (male & female)

735 audio files used (selected subset)

# Preprocessing includes:

MFCC extraction (40 coefficients)

Padding/truncation to ensure equal time steps

Normalization

# Preprocessing & Splitting

Uses preprocess.py to extract and store MFCC features in:

X_datanew.json, Y_datanew.json for training

x_test_data.json, y_test_data.json for testing

Preprocessing time: ~3–5 minutes on Colab GPU

# Model Architecture (LSTM)

Input: (40, 174) MFCC feature matrix

LSTM layers + Dropout

Dense softmax output (8 classes)

Loss: categorical_crossentropy

Optimizer: Adam

# Training

80% training / 20% test split

stratify used for balanced label distribution

EarlyStopping on validation loss

Best model saved as best_weights.keras

# Graphs

Accuracy & Loss Plot

Shows training vs validation accuracy/loss across epochs

Saved as: loss_accuracy_plot.png

Confusion Matrix

Shows test set predictions vs true labels

Normalized row-wise accuracy

Saved as: confusion_matrix.png

# Per-Class Accuracy:

neutral : 1.0000
calm    : 0.6667
happy   : 0.3333
sad     : 0.3333
angry   : 0.5000
fearful : 0.9000


# Output Files
File	Description
X_datanew.json	MFCC features for training set
Y_datanew.json	Emotion labels for training set
x_test_data.json	MFCC features for test set
y_test_data.json	Emotion labels for test set
best_weights.keras	Best model weights (from training)
confusion_matrix.png	Confusion matrix image
loss_accuracy_plot.png	Training vs validation curves



Final test set accuracy: ~80–85%, varies by class
