import os

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split


def load_images(path_list):
    '''
    Loads and preprocesses images

    Args
        path_list: list of paths to the images
        im_class: class of the images

    Returns
        list of images
    '''

    # preprocess images
    images_data = []
    for ip in path_list:
        img = Image.open(ip) #load image
        img = img.convert('L') #to grayscale

        # crop around spectrogram
        box = (213, 60, img.width - 432, img.height - 54)
        img = img.crop(box)

        # resize
        width, height = img.size
        aspect_ratio = width / height
        new_width = int(128 * aspect_ratio)
        img = img.resize((new_width, 128), resample= Image.LANCZOS)

        img = np.array(img)/255.0
        img = np.expand_dims(img, axis= -1)
        
        # image to tensor
        images_data.append(tf.convert_to_tensor(img, dtype= tf.float32))

    return images_data

def prepare_data(dataset_path, ratio= .8):
    '''
    Takes images paths for training

    Args
        dataset_path: path to the dataset containing class folders
        ratio: dimension of training set

    Returns
        training_dataset: class divided paths for images used in training
        train_classes: classes in the training dataset
        test_classes: left out classes
    '''

    classes = os.listdir(dataset_path)
    train_classes, test_classes = train_test_split(classes, train_size= ratio, random_state= 666)
    print(f"Dataset split: \n\tTraining= {train_classes}\
        \n\tTesting= {test_classes}")
    
    # create training dataset
    training_dataset = {}
    for cl in train_classes:
        class_path = os.path.join(dataset_path, cl)
        training_dataset[cl] = [os.path.join(class_path, img) for img in os.listdir(class_path)] # list of image paths

    return training_dataset, train_classes, test_classes