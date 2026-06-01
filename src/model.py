import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def build_model(num_classes, input_shape=(224,224,3), dropout_rate=0.3, learning_rate=0.0001):
    """Build EfficientNetB0 based classifier."""
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def unfreeze_base(model, layers_to_unfreeze=20):
    """Unfreeze top layers of the base model for fine-tuning."""
    base_model = model.layers[1]  # EfficientNet is the second layer
    base_model.trainable = True
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[-layers_to_unfreeze:]:
        layer.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['model']['learning_rate'] / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model