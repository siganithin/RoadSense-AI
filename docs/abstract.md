# ABSTRACT

**Project Title:** AI-Based Road Damage Detection System for Smart City Infrastructure Monitoring

**Technology:** EfficientNetB0 Transfer Learning | CNN | Streamlit | Deep Learning

---

Road infrastructure degradation is a critical public safety and economic challenge faced by urban municipalities worldwide. Potholes, surface cracks, and manhole anomalies cause vehicle damage, accidents, and significant maintenance costs. Traditional road inspection methods rely on manual surveys, which are time-consuming, inconsistent, and reactive — responding only after damage has already posed risk to road users.

This project proposes an AI-powered road damage detection and classification system using Convolutional Neural Networks (CNN) with EfficientNetB0 transfer learning. The system automatically classifies road surface images into three damage categories: **Potholes** (High Severity), **Cracks** (Medium Severity), and **Manhole anomalies** (Low Severity), enabling proactive infrastructure maintenance scheduling.

The methodology follows a two-phase transfer learning strategy. In Phase 1, the EfficientNetB0 backbone pretrained on ImageNet is frozen and only the classification head is trained for rapid feature extraction. In Phase 2, the top 20 layers of the backbone are unfrozen for domain-specific fine-tuning at a reduced learning rate. Data augmentation techniques including rotation, flipping, zoom, brightness variation, and channel shifting are applied to improve model generalization on real-world road imagery.

The model achieves a **validation accuracy of 94.7%** with a weighted F1-score of 0.948, outperforming baseline architectures including ResNet50 (91.4%), VGG16 (89.7%), MobileNetV2 (88.2%), and InceptionV3 (92.1%), while maintaining the lowest parameter count (5.3M) and fastest inference time (< 200ms). GradCAM (Gradient-weighted Class Activation Mapping) is integrated to provide visual explainability, highlighting the regions of the input image that influenced the model's decision.

The system is deployed as an interactive **Streamlit web application** hosted on Hugging Face Spaces, providing live image upload and inference, confidence gauge visualization, per-class probability charts, severity scoring, and actionable maintenance recommendations. A production-grade **FastAPI REST endpoint** is also implemented for integration with municipal systems and autonomous vehicle platforms.

The proposed system demonstrates that combining deep learning-based image classification with an explainability layer and a deployable web interface yields a robust, accurate, and practically useful solution for smart city road infrastructure monitoring.

---

**Keywords:** Road Damage Detection, EfficientNetB0, Transfer Learning, Convolutional Neural Network, GradCAM, Streamlit, Smart City, Infrastructure Monitoring, Computer Vision
