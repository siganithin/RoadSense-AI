"""
test_data_pipeline.py — Tests for data preprocessing and augmentation pipeline
"""

import pytest
import numpy as np
import os
import sys
import tempfile
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def create_dummy_dataset(tmp_dir: str, n_per_class: int = 5):
    """Create a minimal fake dataset with images and YOLO labels."""
    img_dir = os.path.join(tmp_dir, "data", "images")
    lbl_dir = os.path.join(tmp_dir, "data", "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    class_ids = [0, 1, 2]  # Pothole, Crack, Manhole
    for cls_id in class_ids:
        for i in range(n_per_class):
            fname = f"class{cls_id}_img{i}.jpg"
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img.save(os.path.join(img_dir, fname))
            with open(os.path.join(lbl_dir, fname.replace(".jpg", ".txt")), "w") as f:
                f.write(f"{cls_id} 0.5 0.5 0.3 0.3\n")
    return tmp_dir


class TestDataPipeline:
    def test_load_dataset_returns_lists(self):
        from src.data_preparation import load_dataset
        with tempfile.TemporaryDirectory() as tmp:
            ds_path = create_dummy_dataset(tmp)
            paths, labels = load_dataset(ds_path)
            assert isinstance(paths, list)
            assert isinstance(labels, list)
            assert len(paths) == len(labels)

    def test_load_dataset_correct_count(self):
        from src.data_preparation import load_dataset
        with tempfile.TemporaryDirectory() as tmp:
            ds_path = create_dummy_dataset(tmp, n_per_class=4)
            paths, labels = load_dataset(ds_path)
            assert len(paths) == 12   # 3 classes × 4 images

    def test_labels_are_valid_class_ids(self):
        from src.data_preparation import load_dataset
        with tempfile.TemporaryDirectory() as tmp:
            ds_path = create_dummy_dataset(tmp)
            _, labels = load_dataset(ds_path)
            assert all(l in {0, 1, 2} for l in labels)

    def test_missing_label_file_skipped(self):
        from src.data_preparation import load_dataset
        with tempfile.TemporaryDirectory() as tmp:
            ds_path = create_dummy_dataset(tmp, n_per_class=3)
            # Remove one label file
            img_dir = os.path.join(ds_path, "data", "images")
            lbl_dir = os.path.join(ds_path, "data", "labels")
            first_img = os.listdir(img_dir)[0]
            lbl_file  = os.path.join(lbl_dir, first_img.replace(".jpg", ".txt"))
            if os.path.exists(lbl_file):
                os.remove(lbl_file)
            paths, labels = load_dataset(ds_path)
            assert len(paths) == len(labels)

    def test_data_generators_created(self):
        from src.data_preparation import load_dataset, create_data_generators
        with tempfile.TemporaryDirectory() as tmp:
            ds_path = create_dummy_dataset(tmp, n_per_class=6)
            paths, labels = load_dataset(ds_path)
            train_gen, val_gen = create_data_generators(
                paths, labels, batch_size=4,
                img_size=(64, 64), validation_split=0.2, seed=42
            )
            assert train_gen is not None
            assert val_gen is not None

    def test_generator_batch_shape(self):
        from src.data_preparation import load_dataset, create_data_generators
        with tempfile.TemporaryDirectory() as tmp:
            ds_path = create_dummy_dataset(tmp, n_per_class=6)
            paths, labels = load_dataset(ds_path)
            train_gen, _ = create_data_generators(
                paths, labels, batch_size=4,
                img_size=(64, 64), validation_split=0.2, seed=42
            )
            batch_x, batch_y = next(train_gen)
            assert batch_x.shape[1:] == (64, 64, 3)
            assert batch_y.shape[1] == 3   # one-hot, 3 classes
