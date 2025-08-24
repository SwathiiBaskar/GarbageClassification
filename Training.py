import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.applications import EfficientNetB3,ResNet50,DenseNet121
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from sklearn.metrics import accuracy_score,classification_report
from dotenv import load_dotenv

img_size=(128, 128)
input_shape=img_size + (3,)
batch_size=32
epochs=30
num_classes=9 
load_dotenv()
dataset_dir=os.getenv("DATASET_DIR","garbage_classification")

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    image_size=img_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=42
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    image_size=img_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=42
)

class_names=train_ds.class_names
AUTOTUNE=tf.data.AUTOTUNE
train_ds=train_ds.prefetch(AUTOTUNE)
val_ds=val_ds.prefetch(AUTOTUNE)

def build_cnn_light():
    model=models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32,3,activation='relu',padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64,3,activation='relu',padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128,3,activation='relu',padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),

        layers.Dense(128,activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes,activation='softmax')
    ])
    return model

def build_cnn_deep():
    inputs=tf.keras.Input(shape=input_shape)
    x=layers.Conv2D(64,3,activation='relu',padding='same')(inputs)
    x=layers.BatchNormalization()(x)
    x=layers.MaxPooling2D()(x)
    
    x1=layers.Conv2D(64,3,activation='relu',padding='same')(x)
    x1=layers.BatchNormalization()(x1)
    x1=layers.Conv2D(64,3,activation='relu',padding='same')(x1)
    x1=layers.BatchNormalization()(x1)
    
    x=layers.Add()([x, x1])  # Residual connection
    
    x=layers.Conv2D(128,3,activation='relu',padding='same')(x)
    x=layers.BatchNormalization()(x)
    x=layers.GlobalAveragePooling2D()(x)
    x=layers.Dense(256,activation='relu')(x)
    x=layers.Dropout(0.5)(x)
    outputs=layers.Dense(num_classes,activation='softmax')(x)
    return models.Model(inputs,outputs)

def build_cnn_wide():
    inputs = tf.keras.Input(shape=input_shape)
    
    # Parallel convolutions with different kernel sizes
    conv1=layers.Conv2D(64,1,activation='relu',padding='same')(inputs)
    conv3=layers.Conv2D(64,3,activation='relu',padding='same')(inputs)
    conv5=layers.Conv2D(64,5,activation='relu',padding='same')(inputs)
    
    # Concatenating feature maps
    x=layers.Concatenate()([conv1,conv3,conv5])
    x=layers.BatchNormalization()(x)
    x=layers.MaxPooling2D()(x)
    
    # Additional processing
    x=layers.Conv2D(128,3,activation='relu',padding='same')(x)
    x=layers.BatchNormalization()(x)
    x=layers.GlobalAveragePooling2D()(x)
    x=layers.Dense(128,activation='relu')(x)
    x=layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes,activation='softmax')(x)
    return models.Model(inputs,outputs)

def build_base_model(base_class,name):
    base=base_class(include_top=False,weights='imagenet',input_shape=input_shape)
    base.trainable=False

    inputs=tf.keras.Input(shape=input_shape)
    x=base(inputs, training=False)
    x=layers.GlobalAveragePooling2D()(x)
    x=layers.Dense(256, activation='relu')(x) 
    x=layers.Dropout(0.3)(x)
    outputs=layers.Dense(num_classes,activation='softmax')(x)

    return models.Model(inputs,outputs,name=name)

def train_model(model,name):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    checkpoint_cb = ModelCheckpoint(
        f"{name}.keras",
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    )
    
    earlystop_cb = EarlyStopping(
        patience=5,
        restore_best_weights=True,
        monitor='val_accuracy',
        mode='max'
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6
    )
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint_cb,earlystop_cb,reduce_lr]
    )
    return model

try:
    print("Training Light CNN...")
    model_light=build_cnn_light()
    model_light=train_model(model_light, "cnn_light")
except Exception as e:
    print(f"Error training Light CNN: {e}")
    model_light=None

try:
    print("Training Deep CNN...")
    model_deep=build_cnn_deep()
    model_deep=train_model(model_deep, "cnn_deep")
except Exception as e:
    print(f"Error training Deep CNN: {e}")
    model_deep=None

try:
    print("Training Wide CNN...")
    model_wide=build_cnn_wide()
    model_wide=train_model(model_wide, "cnn_wide")
except Exception as e:
    print(f"Error training Wide CNN: {e}")
    model_wide=None

try:
    print("Training EfficientNetB3...")
    model_effnet=build_base_model(EfficientNetB3, "effnet")
    model_effnet=train_model(model_effnet, "effnet")
except Exception as e:
    print(f"Error training EfficientNetB3: {e}")
    model_effnet=None

try:
    print("Training ResNet50...")
    model_resnet=build_base_model(ResNet50, "resnet")
    model_resnet=train_model(model_resnet, "resnet")
except Exception as e:
    print(f"Error training ResNet50: {e}")
    model_resnet=None

try:
    print("Training DenseNet121...")
    model_densenet=build_base_model(DenseNet121, "densenet")
    model_densenet=train_model(model_densenet, "densenet")
except Exception as e:
    print(f"Error training DenseNet121: {e}")
    model_densenet=None

# Creating ensemble from successfully trained models
ensemble_models=[m for m in [model_light, model_deep, model_wide, 
                              model_effnet, model_resnet, model_densenet] if m is not None]

if not ensemble_models:
    raise ValueError("No models were successfully trained for the ensemble.")

def ensemble_predict(models,image_batch):
    preds = [model.predict(image_batch, verbose=0) for model in models]
    avg_preds = np.mean(preds,axis=0)
    return np.argmax(avg_preds,axis=1)

def weighted_ensemble_predict(models,weights,image_batch):
    preds = [model.predict(image_batch, verbose=0) * w for model, w in zip(models, weights)]
    avg_preds = np.sum(preds, axis=0) / sum(weights)
    return np.argmax(avg_preds, axis=1)

def evaluate_ensemble(models,dataset):
    y_true, y_pred = [],[]
    for images, labels in dataset:
        pred = ensemble_predict(models,images)
        y_pred.extend(pred)
        y_true.extend(labels.numpy())  # Changed from argmax since labels are already integers
    
    acc = accuracy_score(y_true,y_pred)
    print(f"Ensemble Accuracy: {acc * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_true,y_pred,target_names=class_names))

def preprocess_image(path):
    img = tf.keras.utils.load_img(path,target_size=img_size)
    img = tf.keras.utils.img_to_array(img)/255.0
    return np.expand_dims(img,axis=0)

def predict_single_image(models,image_path):
    img = preprocess_image(image_path)
    preds = [model.predict(img, verbose=0) for model in models]
    avg_pred = np.mean(preds, axis=0)[0]
    top_class = np.argmax(avg_pred)
    print(f"Predicted: {class_names[top_class]} ({avg_pred[top_class]:.2f} confidence)")
    return top_class

# Evaluating performance
if ensemble_models:
    print("\nEvaluating Ensemble Model...")
    evaluate_ensemble(ensemble_models, val_ds)
    
    # Example prediction
    test_image_path = os.path.join("Samples", "Test_image.png")
    if os.path.exists(test_image_path):
        print("\nMaking prediction on test image...")
        predict_single_image(ensemble_models, test_image_path)
    else:
        print(f"\nTest image not found at: {test_image_path}")
else:
    print("No models available for evaluation.")
