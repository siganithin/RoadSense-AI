import os
import pickle
import yaml
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from src.data_preparation import load_dataset, create_data_generators
from src.model import build_model, unfreeze_base

# Suppress oneDNN warnings (optional)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def main():
    print("Creating models directory...")
    os.makedirs("models", exist_ok=True)
    
    print("Loading dataset...")
    image_paths, labels = load_dataset(config['data']['dataset_path'])
    print(f"Found {len(image_paths)} images with labels.")
    
    print("Creating data generators...")
    train_gen, val_gen = create_data_generators(
        image_paths, labels,
        batch_size=config['data']['batch_size'],
        img_size=tuple(config['data']['image_size']),
        validation_split=config['data']['validation_split'],
        seed=config['data']['seed']
    )
    print("Data generators created.")
    
    num_classes = len(config['classes'])
    print(f"Number of classes: {num_classes}")
    
    print("Building model...")
    model = build_model(
        num_classes=num_classes,
        input_shape=(*config['data']['image_size'], 3),
        dropout_rate=config['model']['dropout_rate'],
        learning_rate=config['model']['learning_rate']
    )
    print("Model built.")
    
    checkpoint = ModelCheckpoint(
        "models/damage_classifier.h5",
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    
    print("Starting Phase 1 training (10 epochs)...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1
    )
    print("Phase 1 completed.")
    
    print("Starting Phase 2 fine-tuning (additional 20 epochs)...")
    model = unfreeze_base(model, layers_to_unfreeze=20)
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=30,
        initial_epoch=10,
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1
    )
    print("Phase 2 completed.")
    
    # Save class names
    class_names = {v: k for k, v in config['classes'].items()}
    with open("models/class_names.pkl", "wb") as f:
        pickle.dump(class_names, f)
    
    print("Training completed. Model saved to models/")

if __name__ == "__main__":
    main()