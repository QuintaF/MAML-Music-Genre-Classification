import os

import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam

class MAML:
    def __init__(self, model, alpha= .01, beta= .001, steps= 1, epochs = 10):
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.steps = steps
        self.epochs = epochs

        self.loss_fn= tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer= Adam(learning_rate= self.beta) 

    def save(self, obj, path, name):
        '''
        Saves the final model

        Args
            obj: final model
            path: path for the save file
        '''

        obj.save(os.path.join(path, name))
        print(f'{name} saved to {path}')

    def get_model(self):
        '''
        Returns
            the current model
        '''
        return self.model
        
    def get_accuracy(self, predictions, labels):
        '''
        Computes accuracy 

        Args
            predictions: straight from the model, transformed into integers
            labels: one hot labels, transformed into integers
        
        Returns
            accuracy value
        '''

        pred = np.argmax(predictions, axis= -1)
        real = np.argmax(labels, axis=-1)

        return np.mean(pred == real)

    def _task_training(self, support):
        '''
        performs one or more gradient steps for task training

        Args
            support set to perform training on

        Returns
            gradient values for the task
        '''

        #in_optimizer= SGD(learning_rate= self.alpha)

        images, labels = zip(*support)
        labels = np.array(labels)
        images = np.array(images)

        # training on support set
        with tf.GradientTape() as tape:
            predictions = self.model(images)
            loss = self.loss_fn(labels, predictions)
            loss = tf.reduce_mean(loss)

        # gradient w.r.t. base network variables
        grad = tape.gradient(loss, self.model.trainable_variables)
        
        # if more than 1 inner step is required
        for _ in range(self.steps-1):
            # apply inner gradient update: θ'i = θ - α∇θLTi(fθ)
            #in_optimizer.apply_gradients( zip(grad, self.model.trainable_variables))
            for var, grad in zip(self.model.trainable_variables, grad):
                var.assign(var - self.alpha * grad)

            # more training on support set
            with tf.GradientTape() as tape:
                predictions = self.model(images)
                loss = self.loss_fn(labels, predictions)
                loss = tf.reduce_mean(loss)

            # gradient w.r.t. base network variables
            grad = tape.gradient(loss, self.model.trainable_variables)

        return grad

    def _inner_loop(self, batch):
        '''
        inner loop on all training tasks

        Args
            batch: the list of tasks to train the model on

        Returns
            new network weights for each task trained
        '''

        meta_weights = []
        old_weights = self.model.get_weights()
        for support, _ in batch:
            # reset model weights for new task
            self.model.set_weights(old_weights)

            # inner training steps on task
            grad = self._task_training(support)

            # compute last gradient update weights: θ'i = θ - α∇θLTi(fθ)
            updated_weights = []
            for var, grad in zip(self.model.trainable_variables, grad):
                updated_weights.append(var - self.alpha * grad)

            # save θ'i per task net weights
            meta_weights.append(updated_weights)

        return meta_weights
    
    def meta_training(self, batch_sampler):
        '''
        Meta Training loop

        Args
            batch_sampler: the class from which batch of tasks for the inner loop are samples

        Returns
            loss history for training epoch
        '''
        
        history = {'loss': [], 'acc': []}
        print(f"MAML-training started for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            old_weights = self.model.get_weights()

            # get batch of tasks
            training_batch = batch_sampler.get_batch()
            
            # cycle through training tasks
            meta_weights= self._inner_loop(training_batch)

            meta_loss = 0
            total_acc = 0
            with tf.GradientTape() as meta_tape:
                # meta training loop
                for idx, data in enumerate(training_batch):
                    _, query = data
                    
                    images, labels = zip(*query)
                    labels = np.array(labels)
                    images = np.array(images)

                    # apply updated task weights to the model
                    for var, new_val in zip(self.model.trainable_variables, meta_weights[idx]):
                        var.assign(new_val)

                    #with tf.GradientTape() as meta_tape:
                    predictions = self.model(images)
                    total_acc += self.get_accuracy(predictions, labels)
                    meta_loss += tf.reduce_mean(self.loss_fn(labels, predictions))

            self.model.set_weights(old_weights)
            #apply the gradient
            grad = meta_tape.gradient(meta_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients( zip(grad, self.model.trainable_variables))
            
            history['loss'].append(meta_loss)            
            history['acc'].append(total_acc/len(training_batch))
            print(f"{epoch}/{self.epochs-1}, Loss: {meta_loss}, Acc: {total_acc/len(training_batch):.3f}")

        return history