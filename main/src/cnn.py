from keras import regularizers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, \
                                    Conv2D, MaxPooling2D, BatchNormalization


class CNN(Model):

    def __init__(self, input= (349, 128, 1), output= 10, **kwargs):
        '''
        CNN model

        Args
            input: input shape
            output: number of output classes(softmax outputs)
        '''

        super(CNN, self).__init__(**kwargs)
        self.num_classes= output
        self.input_shape = input

        model = Sequential([
            Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape= self.input_shape),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.04)),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.2),

            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.2),

            Flatten(),
            Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.04)),
            Dropout(0.5),
            
            Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.04)),
            Dense(self.num_classes, activation='softmax')
        ])

        self.model = model

    def call(self, x):
        '''
        forward pass through the CNN

        Args
            x: input for the model
        '''
        return self.model(x)