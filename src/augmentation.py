"""
augmentation.py — Data augmentation pipeline for RoadSense AI
Provides both Keras ImageDataGenerator-based and albumentations-based augmentation.
"""

import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ── Keras-based augmentation (used in training) ───────────────────────────────
def get_train_augmentor() -> ImageDataGenerator:
    """Returns augmented ImageDataGenerator for training."""
    return ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.25,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.75, 1.25],
        channel_shift_range=20.0,
        fill_mode="nearest",
    )


def get_val_augmentor() -> ImageDataGenerator:
    """Returns rescale-only ImageDataGenerator for validation."""
    return ImageDataGenerator(rescale=1.0 / 255)


# ── Manual augmentation helpers ───────────────────────────────────────────────
def random_brightness(img: np.ndarray, low: float = 0.7, high: float = 1.3) -> np.ndarray:
    factor = np.random.uniform(low, high)
    return np.clip(img * factor, 0, 255).astype(np.uint8)


def random_gaussian_noise(img: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def random_horizontal_flip(img: np.ndarray) -> np.ndarray:
    if np.random.rand() > 0.5:
        return cv2.flip(img, 1)
    return img


def random_rotation(img: np.ndarray, max_angle: float = 20.0) -> np.ndarray:
    angle = np.random.uniform(-max_angle, max_angle)
    h, w  = img.shape[:2]
    M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def random_zoom(img: np.ndarray, zoom_range: float = 0.2) -> np.ndarray:
    h, w   = img.shape[:2]
    factor = np.random.uniform(1 - zoom_range, 1 + zoom_range)
    new_h, new_w = int(h * factor), int(w * factor)
    resized = cv2.resize(img, (new_w, new_h))
    if factor > 1:
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        return resized[start_h:start_h + h, start_w:start_w + w]
    else:
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        return cv2.copyMakeBorder(resized, pad_h, h - new_h - pad_h,
                                  pad_w, w - new_w - pad_w, cv2.BORDER_REFLECT)


def augment_image(img: np.ndarray) -> np.ndarray:
    """Apply a random chain of augmentations to a single image."""
    img = random_horizontal_flip(img)
    img = random_rotation(img)
    img = random_zoom(img)
    img = random_brightness(img)
    img = random_gaussian_noise(img, sigma=5.0)
    return img
