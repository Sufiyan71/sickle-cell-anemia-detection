import zipfile
import os
import shutil
import random

# Step 1: Unzip the archive
def unzip_file(zip_file_path, extract_dir):
    # Ensure the extraction directory exists
    os.makedirs(extract_dir, exist_ok=True)

    # Unzip the archive
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    print(f"Files extracted to: {extract_dir}")

# Step 2: Split the data into training and testing sets
def split_data(source_dir, train_dir, test_dir, split_ratio=0.8):
    # Create directories if they don't exist
    os.makedirs(os.path.join(train_dir, 'Positive'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'Negative'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'Positive'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'Negative'), exist_ok=True)

    # List directories
    categories = ['Positive', 'Negative']

    for category in categories:
        source_path = os.path.join(source_dir, category)
        files = os.listdir(source_path)
        random.shuffle(files)  # Shuffle to ensure randomness

        # Determine split index
        split_index = int(len(files) * split_ratio)

        # Split files into training and testing sets
        train_files = files[:split_index]
        test_files = files[split_index:]

        # Move files to respective folders
        for file_name in train_files:
            shutil.move(os.path.join(source_path, file_name), os.path.join(train_dir, category, file_name))
        for file_name in test_files:
            shutil.move(os.path.join(source_path, file_name), os.path.join(test_dir, category, file_name))

# Step 3: Define paths
zip_file_path = 'archive.zip'  # Path to the zip file
extract_dir = 'extracted_data'  # Path where the zip will be extracted
source_dir = os.path.join(extract_dir, 'archive')  # Directory containing the extracted data
train_dir = 'sickle_train_data'  # Path for training data
test_dir = 'sickle_test_data'    # Path for testing data

# Step 4: Execute the unzip function
unzip_file(zip_file_path, extract_dir)

# Step 5: Split the data
split_data(source_dir, train_dir, test_dir)

print("Data split into training and testing sets.")

# Standard Libraries
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Scikit-learn
from sklearn.model_selection import train_test_split

# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3  # Import for InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras import optimizers, losses
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import image
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
# Path for all files
train_data_dir = 'sickle_train_data'
test_data_dir = 'sickle_test_data'

# Global Variables -- Constants for InceptionV3
RESCALE = None  # Use the InceptionV3's preprocess_input function instead of rescaling
TARGET_SIZE = (299, 299)  # InceptionV3 requires a minimum input size of 299x299
COLOR_MODE = "rgb"  # InceptionV3 expects 3-channel RGB images
CLASS_MODE = "categorical"  # Suitable for multi-class classification
BATCH_SIZE = 32  # Standard batch size, no need to change

from tensorflow.keras.applications.inception_v3 import preprocess_input  # Import preprocess function for InceptionV3

# Data augmentation for training and validation sets
data_augmentation = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Use InceptionV3's preprocess_input instead of rescale
    validation_split=0.2
)

# Training data generator
train_datagen = data_augmentation.flow_from_directory(
    directory=train_data_dir,
    target_size=(299, 299),  # InceptionV3 requires at least 299x299 input size
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    seed=42
)

# Validation data generator
validation_datagen = data_augmentation.flow_from_directory(
    directory=train_data_dir,
    target_size=(299, 299),  # Match InceptionV3's input size
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    seed=42
)

# Test data generator
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # Use preprocess_input for the test set
test_generator = test_datagen.flow_from_directory(
    directory=test_data_dir,
    target_size=(299, 299),  # InceptionV3 input size
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)
# Initializing InceptionV3 (pretrained) model with input image shape as (299, 299, 3)
base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(299, 299, 3))

# Setting the Training of all layers of InceptionV3 model to False
base_model.trainable = False

# Add custom layers on top of the InceptionV3 base model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),  # Replaces Flatten, more efficient with fewer parameters
    Dropout(0.4),              # Increase dropout to 0.4 to prevent overfitting
    Dense(1024, activation='relu'),
    Dropout(0.4),              # Added another Dropout layer for regularization
    Dense(2, activation='softmax')  # For binary classification, adjust the last Dense layer
])


model.save('sickle_cell_detection_model.h5')

# Using the Adam Optimizer to set the learning rate of our final model
opt = optimizers.Adam(learning_rate=0.0001)

# Compiling and setting the parameters we want our model to use
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

# Setting variables for the model
epochs = 20

# Separating Training and Testing Data
train_generator = train_datagen
valid_generator = validation_datagen

# Calculating variables for the model
steps_per_epoch = train_generator.n // BATCH_SIZE
validation_steps = valid_generator.n // BATCH_SIZE

print("steps_per_epoch :", steps_per_epoch)
print("validation_steps :", validation_steps)

# File Path to store the trained models
model_dir = "./CNN-Models"
os.makedirs(model_dir, exist_ok=True)
filepath = os.path.join(model_dir, "model_{epoch:02d}-{val_accuracy:.2f}.keras")

# Using the ModelCheckpoint function to train and store all the best models
model_chkpt = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# Callback list for the model
callbacks_list = [model_chkpt]

# Training the Model
history = model.fit(
    train_generator,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_generator,
    validation_steps=validation_steps,
    callbacks=callbacks_list
)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

# ________________ Graph 1 ________________

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

# ________________ Graph 2 ________________

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.show()

# Calculate the Loss and Accuracy on the Validation Data
val_loss, val_acc = model.evaluate(valid_generator)
print('val accuracy : ', val_acc)

# Calculate the Loss and Accuracy on the Testing Data
test_loss, test_acc = model.evaluate(test_generator)
print('test accuracy : ', test_acc)

# Make predictions on the test data
# We use predict_generator since test data is loaded using a generator
y_pred_prob = model.predict(test_generator)

# Convert probabilities to class predictions
y_pred = np.argmax(y_pred_prob, axis=1)

# Get the true labels from the test generator
y_true = test_generator.classes

# Import the necessary function from sklearn.metrics
from sklearn.metrics import precision_score, recall_score, f1_score

# Calculate Precision, Recall, and F1 Score
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Print the results
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Custom function to load and predict label for the image
def predict(img_rel_path):
    # Import Image from the path with size of (200, 200)
    img = image.load_img(img_rel_path, target_size=(299, 299))

    # Convert Image to a numpy array
    img = image.img_to_array(img, dtype=np.uint8)

    # Scaling the Image Array values between 0 and 1
    img = np.array(img)/255.0

    # Plotting the Loaded Image
    plt.title("Loaded Image")
    plt.axis('off')
    plt.imshow(img.squeeze())
    plt.show()

    # Get the Predicted Label for the loaded Image
    p = model.predict(img[np.newaxis, ...])

    # Label array
    labels = {0: 'Negative', 1: 'Positive'}

    print("\n\nMaximum Probability: ", np.max(p[0], axis=-1))
    predicted_class = labels[np.argmax(p[0], axis=-1)]
    print("Classified:", predicted_class, "\n\n")

    classes=[]
    prob=[]
    print("\n-------------------Individual Probability--------------------------------\n")

    for i,j in enumerate (p[0],0):
        print(labels[i].upper(),':',round(j*100,2),'%')
        classes.append(labels[i])
        prob.append(round(j*100,2))

    def plot_bar_x():
        # this is for plotting purpose
        index = np.arange(len(classes))
        plt.bar(index, prob)
        plt.xlabel('Labels', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.xticks(index, classes, fontsize=12, rotation=20)
        plt.title('Probability for loaded image')
        plt.show()

    plot_bar_x()

    predict("/content/sickle_test_data/Positive/218.jpg")