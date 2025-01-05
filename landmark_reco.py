import tensorflow as tf
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from PIL import Image  # Add this import to fix the Image error

# Paths to train and validation CSV files
train_csv = 'F:/coding/projects/landmark rec/landmarks_dataset_train.csv'
test_csv = 'F:/coding/projects/landmark rec/landmarks_dataset_test.csv'

# Load CSV files
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# Validate image files and remove invalid entries
def validate_image_files(df, directory):
    valid_files = []
    for idx, row in df.iterrows():
        img_path = os.path.join(directory, row['filename'])  # Ensure 'filename' exists in the dataframe
        try:
            img = Image.open(img_path)
            img.verify()  # Check if valid image
            valid_files.append(row)
        except Exception as e:
            print(f"Invalid image file: {img_path}, Error: {e}")
    return pd.DataFrame(valid_files)

# Paths for training and validation data
train_dir = 'F:/coding/projects/landmark rec/datasettrain'
val_dir = 'F:/coding/projects/landmark rec/datasettest'

# Validate train and test datasets
train_df = validate_image_files(train_df, train_dir)
test_df = validate_image_files(test_df, val_dir)

# If the labels column is found, map labels to integers
if 'label' in train_df.columns:
    class_labels = train_df['label'].unique()  # Use 'label' column, replace if different
    class_dict = {label: idx for idx, label in enumerate(class_labels)}
    train_df['label'] = train_df['label'].map(class_dict).astype(str)
    test_df['label'] = test_df['label'].map(class_dict).astype(str)
else:
    raise ValueError("No 'label' column found in the train data.")

# Number of classes
num_classes = len(class_labels)

# Load pre-trained InceptionV3 and enable fine-tuning
base_model = tf.keras.applications.InceptionV3(input_shape=(299, 299, 3), include_top=False, weights='imagenet')
base_model.trainable = True
for layer in base_model.layers[:-20]:  # Freeze all layers except the last 20
    layer.trainable = False

# Build the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),  # Dropout layer to prevent overfitting
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Fine-tuned learning rate
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Set up image data generators with increased augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,  # Increased rotation range for more variation
    width_shift_range=0.3,  # Increased width shift range
    height_shift_range=0.3,  # Increased height shift range
    shear_range=0.3,  # Increased shear range
    zoom_range=0.3,  # Increased zoom range
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    x_col='filename',
    y_col='label',
    target_size=(299, 299),  # InceptionV3 requires 299x299 input
    batch_size=32,
    class_mode='sparse'
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=val_dir,
    x_col='filename',
    y_col='label',
    target_size=(299, 299),  # InceptionV3 requires 299x299 input
    batch_size=32,
    class_mode='sparse'
)

# Implement learning rate scheduler and early stopping
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=40,  # Increased epochs for better training
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    callbacks=[lr_scheduler, early_stopping]
)

# Save the trained model
model.save('landmark_recognition_model_inceptionv3.h5')

# Evaluate the model
val_loss, val_acc = model.evaluate(val_generator)
print(f"Validation Accuracy: {val_acc * 100:.2f}%")

# Plot accuracy and loss curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.show()

# Test the model on a sample image
img_path = 'F:/coding/projects/landmark rec/test_image.jpeg'
img = load_img(img_path, target_size=(299, 299))  # InceptionV3 requires 299x299 input
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict the landmark
pred = model.predict(img_array)
class_idx = np.argmax(pred)
predicted_label = list(class_dict.keys())[list(class_dict.values()).index(class_idx)]

print(f"The predicted landmark is: {predicted_label}")

# Display the test image with prediction
plt.imshow(img)
plt.title(f"Predicted Landmark: {predicted_label}")
plt.axis('off')
plt.show()

# Generate confusion matrix and classification report
true_labels = [class_dict[label] for label in test_df['label']]
pred_labels = []
for img_path in test_df['filename']:
    img = load_img(os.path.join(val_dir, img_path), target_size=(299, 299))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    pred_labels.append(np.argmax(pred))

cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(true_labels, pred_labels, target_names=class_labels))
