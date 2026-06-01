import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def load_dataset(dataset_path):
    """Load images and assign a single label per image based on the most frequent class."""
    image_dir = os.path.join(dataset_path, "data/images")
    label_dir = os.path.join(dataset_path, "data/labels")
    
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not os.path.exists(label_dir):
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    
    image_paths = []
    labels = []
    valid_class_ids = set(config['classes'].keys())  # {0,1,2}
    
    for fname in os.listdir(image_dir):
        if fname.lower().endswith(('.jpg', '.png')):
            img_path = os.path.join(image_dir, fname)
            label_file = os.path.join(label_dir, fname.replace('.jpg', '.txt').replace('.png', '.txt'))
            
            if not os.path.exists(label_file):
                continue
                
            with open(label_file, 'r') as lf:
                lines = lf.readlines()
                if not lines:
                    continue
                # Extract class IDs (first number in each line)
                class_ids = [int(line.split()[0]) for line in lines]
                # Filter only valid class IDs
                class_ids = [cid for cid in class_ids if cid in valid_class_ids]
                if not class_ids:
                    continue
                
            class_counts = pd.Series(class_ids).value_counts()
            main_class = class_counts.idxmax()
            
            image_paths.append(img_path)
            labels.append(main_class)
    
    return image_paths, labels

def create_data_generators(image_paths, labels, batch_size, img_size, validation_split, seed):
    """Create train/validation data generators."""
    # Map numeric labels to class names
    class_names = {v: k for k, v in config['classes'].items()}
    df = pd.DataFrame({'filename': image_paths, 'class_id': labels})
    # Convert class_id to class name safely
    df['class_name'] = df['class_id'].map(class_names)
    
    # Remove any rows where class_name is NaN (should not happen, but safety)
    initial_len = len(df)
    df = df.dropna(subset=['class_name'])
    if len(df) < initial_len:
        print(f"Warning: Removed {initial_len - len(df)} images with unknown class IDs")
    
    # Use class_name for stratification (no NaN)
    train_df, val_df = train_test_split(
        df, test_size=validation_split, random_state=seed, stratify=df['class_name']
    )
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='filename',
        y_col='class_name',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    val_generator = val_datagen.flow_from_dataframe(
        val_df,
        x_col='filename',
        y_col='class_name',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    return train_generator, val_generator