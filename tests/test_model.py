"""
test_model.py — Unit tests for model building and inference pipeline
Run: pytest tests/ -v
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ── Model builder tests ───────────────────────────────────────────────────────
class TestModelBuilder:
    def test_model_output_shape(self):
        from src.model import build_model
        model = build_model(num_classes=3, input_shape=(224, 224, 3))
        assert model.output_shape == (None, 3), "Output shape should be (None, 3)"

    def test_model_input_shape(self):
        from src.model import build_model
        model = build_model(num_classes=3, input_shape=(224, 224, 3))
        assert model.input_shape == (None, 224, 224, 3)

    def test_model_compiles(self):
        from src.model import build_model
        model = build_model(num_classes=3)
        assert model.optimizer is not None
        assert model.loss is not None

    def test_model_predict_shape(self):
        from src.model import build_model
        model = build_model(num_classes=3, input_shape=(224, 224, 3))
        dummy = np.random.rand(2, 224, 224, 3).astype(np.float32)
        preds = model.predict(dummy, verbose=0)
        assert preds.shape == (2, 3)

    def test_model_softmax_sums_to_one(self):
        from src.model import build_model
        model = build_model(num_classes=3, input_shape=(224, 224, 3))
        dummy = np.random.rand(4, 224, 224, 3).astype(np.float32)
        preds = model.predict(dummy, verbose=0)
        sums  = preds.sum(axis=1)
        np.testing.assert_allclose(sums, np.ones(4), atol=1e-5)

    def test_unfreeze_base(self):
        from src.model import build_model, unfreeze_base
        model = build_model(num_classes=3)
        model = unfreeze_base(model, layers_to_unfreeze=10)
        trainable_count = sum(1 for l in model.layers if l.trainable)
        assert trainable_count > 0

    def test_different_num_classes(self):
        from src.model import build_model
        for n in [2, 3, 5, 10]:
            model = build_model(num_classes=n)
            assert model.output_shape == (None, n)


# ── Augmentation tests ────────────────────────────────────────────────────────
class TestAugmentation:
    def _dummy_img(self):
        return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    def test_horizontal_flip_shape(self):
        from src.augmentation import random_horizontal_flip
        img = self._dummy_img()
        out = random_horizontal_flip(img)
        assert out.shape == img.shape

    def test_rotation_shape(self):
        from src.augmentation import random_rotation
        img = self._dummy_img()
        out = random_rotation(img)
        assert out.shape == img.shape

    def test_brightness_range(self):
        from src.augmentation import random_brightness
        img = self._dummy_img()
        out = random_brightness(img)
        assert out.min() >= 0 and out.max() <= 255

    def test_noise_shape(self):
        from src.augmentation import random_gaussian_noise
        img = self._dummy_img()
        out = random_gaussian_noise(img)
        assert out.shape == img.shape

    def test_zoom_shape(self):
        from src.augmentation import random_zoom
        img = self._dummy_img()
        out = random_zoom(img)
        assert out.shape == img.shape

    def test_full_augment_pipeline(self):
        from src.augmentation import augment_image
        img = self._dummy_img()
        out = augment_image(img)
        assert out.shape == img.shape
        assert out.dtype == np.uint8


# ── Utils tests ───────────────────────────────────────────────────────────────
class TestUtils:
    def test_normalize(self):
        from src.utils import normalize
        img = np.ones((224, 224, 3), dtype=np.uint8) * 128
        out = normalize(img)
        assert out.max() <= 1.0
        assert out.min() >= 0.0

    def test_denormalize(self):
        from src.utils import denormalize
        img = np.ones((224, 224, 3), dtype=np.float32) * 0.5
        out = denormalize(img)
        assert out.dtype == np.uint8
        assert out.max() <= 255

    def test_timestamp_format(self):
        from src.utils import timestamp
        ts = timestamp()
        assert len(ts) == 15   # YYYYMMDD_HHMMSS

    def test_load_config(self):
        from src.utils import load_config
        cfg = load_config("config.yaml")
        assert "data" in cfg
        assert "model" in cfg
        assert "classes" in cfg
