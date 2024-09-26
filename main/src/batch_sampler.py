import random

import numpy as np
import tensorflow as tf
from PIL import Image


class BatchSampler:

    def __init__(self, data: dict, n: int, k: int, batch: int, query_as_support= False):
        '''
        Class to generate batches of tasks
        
        Args
            data: data from which it creates batches
            n: the number of classes per task
            k: the number of samples per class inside a task
            batch: the size of the batch
            query_as_support: if set then query as K samples per class otherwise only 1
        '''

        self.dataset = data
        self.classes = list(data.keys())
        self.n = min(len(self.classes), n)
        self.k = k
        self.batch_size = batch
        
        # create a mapping from class to label number
        self.labels = {cl: idx for idx, cl in enumerate(self.classes)}

        self.q_as_s = query_as_support
        if query_as_support:
            self.k = k*2
        else:
            self.k = k+1

    def get_labels(self) -> dict:
        '''
        Returns
            mapping from label to number
        '''

        return self.labels
     
    def _load_images(self, path_list, im_class):
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
            images_data.append((tf.convert_to_tensor(img), self.labels[im_class]))


        return images_data
    
    def _switch_class_with_samples(self, batch):
        '''
        Batch division in support and query sets
        
        Args
            batch: the batch of tasks from which it creates the batch with samples
        Returns
            list of task batches with samples
        '''

        batch_with_samples = []
        for task in batch:
            support_set = []
            query_set = []
            for cl in task:
                # get K random samples of images for the current class
                sample_paths = random.sample(self.dataset[cl], self.k)
                class_samples = self._load_images(sample_paths, cl)
                
                split = -1
                if self.q_as_s:
                    split = self.k//2

                support_set.extend(class_samples[:split])
                query_set.extend(class_samples[split:])
            
            # add task as support and query sets
            batch_with_samples.append([support_set, query_set])
        
        return batch_with_samples

    def get_batch(self):
        '''
        Creates a batch of size batch_size for tasks 
        each containing K samples per class

        Returns
            batch of tasks
        '''

        batch = [tuple(np.random.choice(self.classes, self.n, replace=False)) for _ in range(self.batch_size)]
        batch_with_samples = self._switch_class_with_samples(batch)

        return batch_with_samples