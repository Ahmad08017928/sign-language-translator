import os
import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi dataset
data_dir = 'data/'
labels = {chr(i+65): i for i in range(26)}  # A-Z
IMG_SIZE = 224  # MobileNetV2 standar resolution

def load_data():
    X, y = [], []
    for label, idx in labels.items():
        folder = os.path.join(data_dir, label)
        if not os.path.exists(folder):
            continue
        print(f"Loading class {label} ({idx})...")
        count = 0
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Tidak dapat membaca gambar {img_path}")
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = preprocess_input(img)  # Sesuai standar MobileNetV2
            X.append(img)
            y.append(idx)
            count += 1
        print(f"Loaded {count} images for class {label}")
    return np.array(X), np.array(y)

print("Loading dataset...")
X, y = load_data()
print(f"Dataset loaded: {len(X)} images, {len(np.unique(y))} classes")

# Cek distribusi kelas
for label, idx in labels.items():
    count = np.sum(y == idx)
    print(f"Class {label}: {count} images")

y = to_categorical(y, num_classes=26)

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set: {X_train.shape[0]} images")
print(f"Validation set: {X_val.shape[0]} images")

# Augmentasi data dengan transformasi yang lebih ringan untuk isyarat tangan
datagen = ImageDataGenerator(
    rotation_range=15,  # Lebih konservatif untuk isyarat tangan
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,  # Tetap matikan flip horizontal
    fill_mode='nearest',
    brightness_range=[0.8, 1.2],  # Variasi pencahayaan untuk ketahanan
)

datagen.fit(X_train)

# Load base MobileNetV2 dengan input shape yang benar
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Freeze layer awal, unfreeze beberapa layer terakhir untuk fine tuning
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Build model dengan multiple regularization techniques
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Lapisan pertama dengan batch normalization dan dropout
x = Dense(512, activation=None, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)  # Dropout yang cukup agresif

# Output layer dengan regulasi L2 yang lebih ringan
output = Dense(26, activation='softmax', kernel_regularizer=l1_l2(l1=0, l2=1e-4))(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile model dengan learning rate yang lebih kecil dan decay
optimizer = Adam(learning_rate=0.0001, decay=1e-6)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks untuk mengoptimalkan pembelajaran
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
model_checkpoint = ModelCheckpoint(
    'models/sign_language_checkpoint.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Training dengan augmentasi dan callbacks
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=250, 
    callbacks=[reduce_lr, early_stop, model_checkpoint]
)

# Evaluasi model
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Validation accuracy: {val_acc:.4f}")
print(f"Validation loss: {val_loss:.4f}")


y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

# Plot konfusi matrix
plt.figure(figsize=(12, 10))
conf_matrix = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[chr(i+65) for i in range(26)], 
            yticklabels=[chr(i+65) for i in range(26)])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('history/confusion_matrix.png')
plt.close()

# Simpan model final
os.makedirs("models", exist_ok=True)
model.save('models/sign_language_mobilenetv2_regularized.h5')
print("Model saved to 'models/sign_language_mobilenetv2_regularized.h5'")