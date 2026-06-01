"""
RoadSense AI — Source Package
Modules:
    data_preparation  : Dataset loading and data generators
    model             : EfficientNetB0 model builder and fine-tuning
    evaluate          : Evaluation metrics and visualizations
    predict           : Single-image and batch inference
    gradcam           : GradCAM explainability
    augmentation      : Data augmentation pipeline
    utils             : Shared utilities (logging, config, image helpers)
"""

from src.utils import load_config, get_logger
from src.model import build_model, unfreeze_base
from src.data_preparation import load_dataset, create_data_generators

__version__ = "2.0.0"
__author__  = "RoadSense AI Team"
