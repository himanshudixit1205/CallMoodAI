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

# Add channel dimension if missing
if len(X.shape) == 3:
    X = X[..., np.newaxis]

print("Loaded data shape:", X.shape, y.shape)

# Normalize
X = X / (np.max(np.abs(X), axis=(1, 2, 3), keepdims=True) + 1e-8)

# One-hot encode labels
y_categorical = to_categorical(y, num_classes=8)

# Class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class weights: {dict(zip(range(8), [f'{w:.2f}' for w in class_weights]))}")

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)

print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

# CNN Model
model = Sequential([
    
    Conv2D(32, (3, 3), activation='relu', input_shape=(40, 128, 1), padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(128, activation='relu'),
    Dropout(0.3),
    
    Dense(8, activation='softmax')
])

# Compile
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model Summary:")
model.summary()

# Callbacks
callbacks = [
    ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.6,
        patience=6,
        min_lr=1e-6,
        verbose=1,
        mode='max'
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=12,
        restore_best_weights=True,
        verbose=1,
        mode='max'
    ),
    ModelCheckpoint(
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

# Predictions
y_pred = model.predict(X_val, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_val, axis=1)

# Classification report
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

# Distribution check
pred_dist = np.bincount(y_pred_classes, minlength=8)
true_dist = np.bincount(y_true_classes, minlength=8)

print(f"\nPrediction vs True Distribution:")
for i in range(8):
    print(f"Class {i}: Predicted={pred_dist[i]:2d}, True={true_dist[i]:2d}")

print("Model saved as best_emotion_model.keras")

# Save training history
import pickle
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
