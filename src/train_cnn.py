"""
train_cnn.py
------------
Trains a CNN binary classifier on mel spectrogram images.

Dataset structure:
    data/spectrograms/
        FAKE/   -> label 0
        REAL/   -> label 1

Run:
    python -m src.train_cnn
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR    = "data/spectrograms"
MODELS_DIR  = "models"
IMG_SIZE    = (128, 128)
BATCH_SIZE  = 32
EPOCHS      = 10
# ───────────────────────────────────────────────────────────────────────────────


def build_model() -> tf.keras.Model:
    """Build and return the CNN model."""
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(*IMG_SIZE, 3)),
        layers.MaxPooling2D(2, 2),

        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),

        # Block 3
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),

        # Classifier head
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ])
    return model


def plot_history(history: tf.keras.callbacks.History, output_dir: str) -> None:
    """Save training/validation accuracy and loss curves as a PNG."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    # Loss
    axes[1].plot(history.history["loss"],     label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "cnn_training_history.png")
    fig.savefig(plot_path, dpi=100)
    plt.close(fig)
    print(f"  History plot saved → {plot_path}")


def train(data_dir: str = DATA_DIR, models_dir: str = MODELS_DIR) -> None:
    """
    Load spectrogram images, train the CNN, evaluate, and save the model.

    Args:
        data_dir:   Root folder containing REAL/ and FAKE/ sub-folders.
        models_dir: Destination folder for the saved model and history plot.
    """
    os.makedirs(models_dir, exist_ok=True)

    # ── 1. Data generators ──────────────────────────────────────────────────
    # Augment only the training split to reduce overfitting.
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.08,
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        shuffle=True,
        seed=42,
    )

    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False,
        seed=42,
    )

    print(f"\n  Class indices : {train_generator.class_indices}")
    print(f"  Train batches : {len(train_generator)}")
    print(f"  Val   batches : {len(val_generator)}\n")

    # ── 2. Build & compile ──────────────────────────────────────────────────
    model = build_model()
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # ── 3. Train ────────────────────────────────────────────────────────────
    print(f"\nTraining for {EPOCHS} epochs ...\n")

    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # ── 4. Final metrics ────────────────────────────────────────────────────
    final_train_acc = history.history["accuracy"][-1]
    final_val_acc   = history.history["val_accuracy"][-1]

    print(f"\n{'─' * 42}")
    print(f"  Training   Accuracy : {final_train_acc:.4f}  ({final_train_acc * 100:.2f}%)")
    print(f"  Validation Accuracy : {final_val_acc:.4f}  ({final_val_acc * 100:.2f}%)")
    print(f"{'─' * 42}")

    # ── 5. Save model ────────────────────────────────────────────────────────
    model_path = os.path.join(models_dir, "cnn_model.keras")
    model.save(model_path)
    print(f"\n  Model saved → {model_path}")

    # ── 6. Save history plot ─────────────────────────────────────────────────
    plot_history(history, models_dir)


if __name__ == "__main__":
    train()
