import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import os
import numpy as np
import matplotlib.pyplot as plt

# Load pre-trained ResNet model without the top (fully connected) layers
base_model = ResNet50(weights='imagenet', include_top=False)
# 'conv5_block3_out', 'conv4_block6_out', "conv2_block3_out", conv5_block2_out
layer_name = "conv5_block2_out"
selected_layer = base_model.get_layer(layer_name).output

activation_model = tf.keras.models.Model(inputs=base_model.input, outputs=selected_layer)
dataset_path = '/Users/mihir/Documents/CS 7651/Final Project/FACES'

emotion_activations = {}
dirs = [folder for folder in os.listdir(dataset_path) if not folder.startswith('.DS_Store')]
# print(dirs)
for emotion_folder in dirs:
    emotion_path = os.path.join(dataset_path, emotion_folder)
    emotion_activations[emotion_folder] = []
    for image_name in os.listdir(emotion_path):
        image_path = os.path.join(emotion_path, image_name)
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict emotion using the model
        activations = activation_model.predict(img_array)
        emotion_activations[emotion_folder].append(activations)

for emotion_folder, activations_list in emotion_activations.items():
    # Convert the list of activations to a numpy array
    activations_array = np.array(activations_list)
    # Take the mean along the batch dimension to get a representative activation
    mean_activation = np.mean(activations_array, axis=0)
    # Visualize the mean activation
    plt.imshow(mean_activation[0, :, :, 0], cmap='viridis')
    plt.title('Emotion: {} - Layer: {}'.format(emotion_folder, layer_name))
    cbar = plt.colorbar()
    cbar.set_label('Mean Activation Value')
    plt.show()
