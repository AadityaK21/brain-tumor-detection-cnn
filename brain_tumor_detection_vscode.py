import os
import random
from typing import List, Tuple
import cv2
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import BinaryCrossentropy


IMAGE_SIZE = 150
EPOCHS = 15
BATCH_SIZE = 25
SEED = 10
MODEL_PATH = "brain_tumor_model.keras"

TUMOR_FOLDERS = [
    "Brain tumor - Recurrenceremnant of previous lesion",
    "Brain tumor operated with ventricular hemorrhage",
    "Brain Tumor",
    "Brain Tumor (Hemangioblastoma  Pleomorphic xanthroastrocytoma  metastasis)",
    "Brain tumor (Dermoid cyst craniopharyngioma)",
    "small meningioma",
    "pituitary tumor",
    "Brain Tumor (Ependymoma)",
    "meningioma",
    "Brain tumor (Astrocytoma Ganglioglioma)",
    "Glioma",
    "Left Retro-orbital Haemangioma",
]
NORMAL_FOLDER = "Normal"


def download_dataset() -> str:
    """Download dataset using kagglehub and return the dataset root path."""
    path = kagglehub.dataset_download("sudipde25/mri-dataset-for-detection-and-analysis")
    print(f"Dataset downloaded to: {path}")
    return path



def collect_image_paths(dataset_path: str) -> Tuple[List[str], List[str]]:
    """Collect tumor and normal image paths."""
    nins_dir = os.path.join(dataset_path, "NINS_Dataset", "NINS_Dataset")
    if not os.path.isdir(nins_dir):
        raise FileNotFoundError(f"Dataset folder not found: {nins_dir}")

    yes_images: List[str] = []
    no_images: List[str] = []

    for folder in TUMOR_FOLDERS:
        folder_path = os.path.join(nins_dir, folder)
        if os.path.isdir(folder_path):
            for fname in os.listdir(folder_path):
                if fname.lower().endswith(".jpg"):
                    yes_images.append(os.path.join(folder_path, fname))

    normal_path = os.path.join(nins_dir, NORMAL_FOLDER)
    if not os.path.isdir(normal_path):
        raise FileNotFoundError(f"Normal folder not found: {normal_path}")

    for fname in os.listdir(normal_path):
        if fname.lower().endswith(".jpg"):
            no_images.append(os.path.join(normal_path, fname))

    print(f"Tumor images found: {len(yes_images)}")
    print(f"Normal images found: {len(no_images)}")
    return yes_images, no_images



def show_samples(image_list: List[str], label: str, num: int = 5) -> None:
    """Display random image samples."""
    if len(image_list) == 0:
        print(f"No images found for {label}.")
        return

    num = min(num, len(image_list))
    plt.figure(figsize=(15, 5))

    for i, img_path in enumerate(random.sample(image_list, num)):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, num, i + 1)
        plt.imshow(img)
        plt.title(label)
        plt.axis("off")

    plt.tight_layout()
    plt.show()



def load_and_preprocess_images(yes_images: List[str], no_images: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Load images, resize them, normalize them, and create labels."""
    data = []
    labels = []

    for img_path in yes_images:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img / 255.0
        data.append(img)
        labels.append(1)

    for img_path in no_images:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img / 255.0
        data.append(img)
        labels.append(0)

    data = np.array(data, dtype=np.float32).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    labels = np.array(labels, dtype=np.int32)

    indices = np.arange(len(data))
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    print(f"Total processed images: {len(data)}")
    return data, labels



def build_model() -> tf.keras.Model:
    """Create the CNN model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss=BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    return model



def plot_history(history: tf.keras.callbacks.History) -> None:
    """Plot training and validation metrics."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    plt.show()



def main() -> None:
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    dataset_path = download_dataset()
    yes_images, no_images = collect_image_paths(dataset_path)

    print("\nShowing sample tumor images...")
    show_samples(yes_images, "Tumor")

    print("\nShowing sample normal images...")
    show_samples(no_images, "Normal")

    data, labels = load_and_preprocess_images(yes_images, no_images)

    x_train, x_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=0.2,
        random_state=SEED,
        stratify=labels,
    )

    print(f"Training samples: {len(x_train)}")
    print(f"Testing samples: {len(x_test)}")

    model = build_model()
    model.summary()

    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1,
    )

    plot_history(history)

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    model.save(MODEL_PATH)
    print(f"Model saved as: {MODEL_PATH}")


if __name__ == "__main__":
    main()
