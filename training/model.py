import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
                                   BatchNormalization)
from tensorflow.keras.callbacks import (ReduceLROnPlateau, EarlyStopping, 
                                      ModelCheckpoint)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Load features
X = np.load('features/X.npy')
y = np.load('features/y.npy')

# It changes shape from (samples, height, width) to (samples, height, width, 1) so CNN knows it’s grayscale (1 channel).
if len(X.shape) == 3:
    X = X[..., np.newaxis]

print("Loaded data shape:", X.shape, y.shape)

# Normalize
X = X / (np.max(np.abs(X), axis=(1, 2, 3), keepdims=True) + 1e-8)

# One-hot encode labels
y_categorical = to_categorical(y, num_classes=8)

# Handle class imbalance with balanced weights
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class weights: {dict(zip(range(8), [f'{w:.2f}' for w in class_weights]))}")

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical # stratify_y => Ensures the train/validation split keeps the same class proportion as the original dataset.
)

print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

# Improved model - closer to your original but better
model = Sequential([
    # Block 1 : Block usually means multiple layers.
    Conv2D(32, (3, 3), activation='relu', input_shape=(40, 128, 1), padding='same'),
    BatchNormalization(), # It normalizes layer outputs to speed up training and improve stability.
    MaxPooling2D((2, 2)), # It downsamples the feature map by taking the maximum in each 2×2 window, reducing size and computation.
    Dropout(0.2),  # Randomly turns off 20% of neurons during training to reduce overfitting.
    
    # Block 2  
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    
    # Block 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    
    # Dense layers
    Flatten(), # Converts the 2D feature maps into a 1D vector so they can be fed into dense layers.
    Dense(256, activation='relu'),  # A fully connected layer with 256 neurons using ReLU for non-linearity; increased size gives more learning capacity.
    BatchNormalization(),
    Dropout(0.4), 
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(8, activation='softmax')
])

# Dense(256) – learn complex patterns from flattened features.
# Dense(128) – refine and compress learned patterns.
# Dense(8) – output probabilities for 8 emotion classes (softmax).


# Compile with good settings
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model Summary:")
model.summary()

# Smart callbacks
callbacks = [
    ReduceLROnPlateau( # If val_accuracy stops improving for 6 epochs, reduce learning rate by ×0.6 (but not below 1e-6) to fine-tune learning.
        monitor='val_accuracy',
        factor=0.6, # Multiplies the current learning rate by 0.6 
        patience=6, # Waits 6 epochs without improvement before reducing the learning rate.
        min_lr=1e-6, # Sets the lowest learning rate allowed to 0.000001 during training.
        verbose=1,
        mode='max'
    ),
    EarlyStopping( # If val_accuracy doesn’t improve for 12 epochs, stop training and restore the best model weights to avoid overfitting.
        monitor='val_accuracy',
        patience=12,       
        restore_best_weights=True,
        verbose=1,
        mode='max'
    ),
    ModelCheckpoint( # Save the model file only when val_accuracy reaches a new high, so you keep the best-performing version.
        'saved_model/best_emotion_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max'
    )
]

# Train
print("Starting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=32,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# Load best weights
try:
    model.load_weights('saved_model/best_emotion_model.keras')
except:
    print("Using final model weights")

# Evaluate
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\nFinal Validation Accuracy: {val_acc * 100:.2f}%")

# Detailed analysis
y_pred = model.predict(X_val, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_val, axis=1)


# For Analysis: 
print("\nPer-class Performance:")
report = classification_report(y_true_classes, y_pred_classes, output_dict=True)
for class_id in range(8):
    if str(class_id) in report:
        precision = report[str(class_id)]['precision']
        recall = report[str(class_id)]['recall']
        f1 = report[str(class_id)]['f1-score']
        support = report[str(class_id)]['support']
        print(f"Class {class_id}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, Support={support}")

print(f"\nOverall Metrics:")
print(f"Macro Avg F1: {report['macro avg']['f1-score']:.3f}")
print(f"Weighted Avg F1: {report['weighted avg']['f1-score']:.3f}")

# Check prediction distribution
pred_dist = np.bincount(y_pred_classes, minlength=8) # np.bincount() => counts how many times each integer value appears in an array.
true_dist = np.bincount(y_true_classes, minlength=8)

print(f"\nPrediction vs True Distribution:")
for i in range(8):
    print(f"Class {i}: Predicted={pred_dist[i]:2d}, True={true_dist[i]:2d}")

print("Model saved as best_emotion_model.keras")

# Save training history
import pickle
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)


# Final Validation Accuracy: 72.10%

# Per-class Performance:
# Class 0: P=0.765, R=0.722, F1=0.743, Support=18.0
# Class 1: P=0.654, R=0.919, F1=0.764, Support=37.0
# Class 2: P=0.833, R=0.541, F1=0.656, Support=37.0
# Class 3: P=0.696, R=0.432, F1=0.533, Support=37.0
# Class 4: P=0.698, R=0.811, F1=0.750, Support=37.0
# Class 5: P=0.686, R=0.649, F1=0.667, Support=37.0
# Class 6: P=0.725, R=0.784, F1=0.753, Support=37.0
# Class 7: P=0.786, R=0.917, F1=0.846, Support=36.0

# Overall Metrics:
# Macro Avg F1: 0.714
# Weighted Avg F1: 0.712

# Prediction vs True Distribution:
# Class 0: Predicted=17, True=18
# Class 1: Predicted=52, True=37
# Class 2: Predicted=24, True=37
# Class 3: Predicted=23, True=37
# Class 4: Predicted=43, True=37
# Class 5: Predicted=35, True=37
# Class 6: Predicted=40, True=37
# Class 7: Predicted=42, True=36
# Model saved as best_emotion_model.keras