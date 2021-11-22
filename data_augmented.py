__author__ = "Mira Syahirah"
__copyright__ = "Copyright 2021, Sevtech"
__maintainer__ = "Mira Syahirah"
__email__ = "myransr02@gmail.com"

# import Augmentor
 # Common imports
import os
import numpy as np

# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

 # TensorFlow imports
# may differs from version to versions
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset information

image_folder = os.path.join('datasets', 'face_dataset_train_images')
img_height, img_width = 250, 250  # size of images
num_classes = 2  #  myself


dataset = keras.preprocessing.image_dataset_from_directory(
    image_folder,
    seed=42,
    image_size=(img_height, img_width),
    label_mode='categorical',
    shuffle=True)

class_names = dataset.class_names
class_names

# Helper function to get classname of the image
def get_classname(class_names, mask):


    assert len(class_names) == len(
        mask), "The arrays must be of the same length"

    return class_names[np.array(mask).argmax(axis=0)]

sqrt_img = 2  # images per row / col.
# The square root of the total number of images shown

plt.figure(figsize=(8, 8))
for images, labels in dataset.take(3):
    for index in range(sqrt_img**2):
        # grid 'sqrt_img' x 'sqrt_img'
        plt.subplot(sqrt_img, sqrt_img, index + 1)
        plt.imshow(images[index] / 255)
        class_name = get_classname(class_names, labels[index])
        plt.title("Class: {}".format(class_name))
        plt.axis("off")



batch_size = 16

# Create data generator based on ImageDataGenerator object

train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=(0.7, 1),
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    image_folder,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

# To see next augmented image
image, label = train_generator.next()

plt.figure(figsize=(6, 6))
plt.imshow(image[0] / 255)  # first image from batch
plt.title("Augmented image from ImageDataGenerator")
plt.axis("off")

## Cara yang pertama : Generate n * batch_size random samples

n = 10

aug_image_folder = os.path.join('datasets', 'face_dataset_train_aug_images')
if not os.path.exists(aug_image_folder):
    os.makedirs(aug_image_folder)  # create folder if doesn't exist



train_generator.save_to_dir = aug_image_folder
train_generator.save_format = 'jpg'

# If 'save_to_dir' is set, `next()` method
# will generate `batch_size` images each time 
# and save them to 'save_to_dir' folder

for i in range(n):
    print("Step {} of {}".format(i+1, n))
    train_generator.next()
    print("\tGenerate {} random images".format(train_generator.batch_size))

print("\nTotal number images generated = {}".format(n*train_generator.batch_size))

## Cara yang kedua : Generate n samples for each image
n = 5

aug_image_folder = os.path.join('datasets', 'face_dataset_train_aug_images')
if not os.path.exists(aug_image_folder):
    os.makedirs(aug_image_folder)  # create folder if doesn't exist

train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=(0.7, 1),
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest')

 # classes: 'myself' and 'bapak dafi'
image_folder_to_generate = os.path.join(image_folder, 'bapak dafi')
image_folder_to_save = os.path.join(aug_image_folder, 'new bapak dafi')
if not os.path.exists(image_folder_to_save):
    os.makedirs(image_folder_to_save)  # create folder if doesn't exist

i = 0
total = len(os.listdir(image_folder_to_generate))  # berapa banyak filenya dalam folder 
for filename in os.listdir(image_folder_to_generate):
    print("Step {} of {}".format(i+1, total))
    #untuk setiap image dalam folder : bacanya
    image_path = os.path.join(image_folder_to_generate, filename)
    image = keras.preprocessing.image.load_img(
        image_path, target_size=(img_height, img_width, 3))
    image = keras.preprocessing.image.img_to_array(
        image)  # dari gambar ke array
    # bentuk dari (250, 250, 3) ke (1, 250, 250, 3)
    image = np.expand_dims(image, axis=0)

    # create ImageDataGenerator object for it
    current_image_gen = train_datagen.flow(image,
                                           batch_size=1,
                                           save_to_dir=image_folder_to_save,
                                           save_prefix=filename,
                                           save_format="jpg")

    # generate n samples
    count = 0
    for image in current_image_gen:  # accessing the object saves the image to disk
        count += 1
        if count == n:  # n gambar akan digenerate
            break
    print('\tGenerate {} samples for file {}'.format(n, filename))
    i += 1

print("\nTotal number images generated = {}".format(n*total))




##SIMPLE ONE AUGMENTED IMAGE

# # Build a new pipeline for image processing.
# p = Augmentor.Pipeline(
#     source_directory="data/mail",
#     output_directory="data/mail")

# # Manipulate the example card.
# p.rotate_without_crop(
#     probability=.5,
#     max_left_rotation=10,
#     max_right_rotation=10,
#     expand=True)
# p.zoom(
#     probability=.3,
#     min_factor=.8,
#     max_factor=1.1)
# p.skew(
#     probability=.3,
#     magnitude=.15)
# p.random_brightness(
#     probability=.5,
#     min_factor=.5,
#     max_factor=1.75)

# # Generate and save 100 new images to the output directory.
# p.sample(300)

