import kagglehub
import pandas as pd
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet50 import preprocess_input

# =============================
# 1. Load file paths + labels
# =============================

# Download latest version
path = kagglehub.dataset_download("shreyasraghav/shutterstock-dataset-for-ai-vs-human-gen-image")

print("Path to dataset files:", path)

# Point directly to train.csv inside the downloaded dataset
df = pd.read_csv(os.path.join(path, "train.csv"))

y = df["label"].values # 0 or 1
X = df['file_name'].values

base_dir = "/Users/ansonlin/Desktop/SPIS/archive"
X = df['file_name'].apply(lambda fn: os.path.join(base_dir, str(fn).strip())).values

x_train, x_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)


# =============================
# 2. Preprocessing function
# =============================
IMG_SIZE = (160, 160)

def preprocess_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)

    img = tf.image.resize(img, IMG_SIZE)
    img = preprocess_input(img)
    return img, label

# =============================
# 3. TF Datasets
# =============================
BATCH_SIZE = 64

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))

train_ds = (train_ds
            .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(1000)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE))

test_ds = (test_ds
           .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
           .batch(BATCH_SIZE)
           .prefetch(tf.data.AUTOTUNE))

val_ds = (val_ds
          .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
          .batch(BATCH_SIZE)
          .prefetch(tf.data.AUTOTUNE))

# =============================
# 4. Model (Transfer Learning)
# =============================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

base_model = tf.keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(160,160,3)
)
base_model.trainable = True
for layer in base_model.layers[:-20]:  # freeze all but last 20 layers
    layer.trainable = False


model = tf.keras.Sequential([
    data_augmentation,
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="sigmoid")  # binary classification
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(name="auc"), "accuracy"]

)

from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

# =============================
# 5. Train
# =============================
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=3e-4,
    decay_steps=len(train_ds) * 10,
    alpha=1e-6
)


base_model.trainable = False
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(name="auc"), "accuracy"]
)
print("Stage 1: Training only top layers...")
model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=[early_stop])


for layer in base_model.layers[:-20]:
    layer.trainable = False
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(name="auc"), "accuracy"]
)
print("Stage 2: Fine-tuning last 20 layers...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[early_stop],
    class_weight=class_weights
)

# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=20,
#     callbacks=[early_stop, lr_schedule],
#     class_weight=class_weights
# )

# =============================
# 6. Evaluate
# =============================
test_loss, test_auc, test_acc = model.evaluate(test_ds, verbose=2)
print(f"\n✅ Test accuracy: {test_acc:.4f}")

# =============================
# 7. Predictions
# =============================
predictions = model.predict(test_ds)
predicted_labels = (predictions > 0.5).astype(int).flatten()

# =============================
# 8. Visualization
# =============================
def plot_prediction_results(file_paths, labels, predictions, predicted_labels, title, correct=True):
    plt.figure(figsize=(10,10))
    
    
    correct_indices = [i for i in range(len(labels)) if predicted_labels[i] == labels[i]]
    incorrect_indices = [i for i in range(len(labels)) if predicted_labels[i] != labels[i]]
    indices = correct_indices if correct else incorrect_indices
    
    # show up to 25 images
    for i, idx in enumerate(indices[:25]):
        img = tf.io.read_file(file_paths[idx])
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        plt.subplot(5,5,i+1)
        plt.imshow(img.numpy().astype("uint8"))
        plt.axis("off")
        plt.xlabel(f"T:{labels[idx]}, P:{predicted_labels[idx]}")
    
    plt.suptitle(title)
    plt.show()

# Save the trained model
model.save("ai_detector_model.keras")


# plot_prediction_results(x_test, y_test, predictions, "✅ Correct Predictions", correct=True)
# plot_prediction_results(x_test, y_test, predictions, "❌ Incorrect Predictions", correct=False) # show incorrect predictions

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, predicted_labels))
print(confusion_matrix(y_test, predicted_labels))

def plot_history(history):
    plt.figure(figsize=(18,5))

    # Accuracy
    plt.subplot(1,3,1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title("Accuracy")

    # Loss
    plt.subplot(1,3,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title("Loss")

    # AUC
    plt.subplot(1,3,3)
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Val AUC')
    plt.legend()
    plt.title("AUC")

    plt.show()
