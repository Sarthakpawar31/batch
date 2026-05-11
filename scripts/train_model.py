"""
scripts/train_model.py

Fine-tune MobileNetV2 on PlantVillage dataset and export to TFLite.

Run on a desktop/cloud machine with a GPU.  NOT on the Raspberry Pi.

Requirements (desktop only):
    pip install tensorflow tensorflow-model-optimization kaggle

Dataset:
    https://www.kaggle.com/datasets/emmarex/plantdisease

Usage:
    kaggle datasets download -d emmarex/plantdisease
    unzip plantdisease.zip -d data/
    python scripts/train_model.py
"""

from __future__ import annotations

import pathlib

# ── Lazy imports (not available on RPi) ─────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications import MobileNetV2
except ImportError:
    raise SystemExit("Install tensorflow: pip install tensorflow")

DATA_DIR   = pathlib.Path("data/PlantVillage")
OUTPUT     = pathlib.Path("ai/disease_model.tflite")
IMG_SIZE   = (224, 224)
BATCH      = 32
EPOCHS     = 10
NUM_CLASSES = 10   # adjust to your label count

# ── Data pipeline ─────────────────────────────────────────────────────────────

def build_datasets():
    train = keras.preprocessing.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="training",
        seed=42, image_size=IMG_SIZE, batch_size=BATCH,
    )
    val = keras.preprocessing.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation",
        seed=42, image_size=IMG_SIZE, batch_size=BATCH,
    )
    AUTOTUNE = tf.data.AUTOTUNE
    train = train.prefetch(AUTOTUNE)
    val   = val.prefetch(AUTOTUNE)
    return train, val


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model() -> keras.Model:
    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False   # freeze base; fine-tune top

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── TFLite export ─────────────────────────────────────────────────────────────

def export_tflite(model: keras.Model) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Dynamic-range quantisation (reduces model size ~4×)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(tflite_model)
    size_mb = OUTPUT.stat().st_size / 1e6
    print(f"TFLite model saved: {OUTPUT}  ({size_mb:.1f} MB)")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading dataset …")
    train, val = build_datasets()

    print("Building model …")
    model = build_model()
    model.summary()

    print("Training …")
    model.fit(train, validation_data=val, epochs=EPOCHS)

    print("Exporting TFLite …")
    export_tflite(model)

    print("Done!  Copy ai/disease_model.tflite to your Raspberry Pi.")
