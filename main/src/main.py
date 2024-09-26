'''
Deep Learning for Music Genre Classification with CNN
'''
import os
import time
import pickle
from datetime import datetime
from contextlib import contextmanager

#directory change ->  ...\main\src
import os
file_path = os.path.dirname(__file__)
os.chdir(file_path)

#ignore WARNINGS
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

#modules
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

#local imports
from arg_parser import parse_args
from data_preparation import prepare_data, load_images
from batch_sampler import BatchSampler
from maml import MAML
from cnn import CNN


#globals
DATASET = "..\\dataset\\mel_spectrograms"
MODELS = "..\\models"


def plot_history(history, dir= None):
    '''
    Plot training results from the model history

    Args
        history: dictionary containing losses and accuracies over training
        dir: if set, saves the plots to the specified path
    '''
    
    acc = history['acc']
    loss = history['loss']
    
    tick_step = 1
    if len(loss) > 99:
        tick_step = 10**(np.floor(np.log10(len(loss))).astype(int))
    epochs = range(0, len(acc), tick_step)
    tick_loss = [min(loss), max(loss)]
    tick_acc = [min(acc), max(acc)]

    # plot
    plt.figure(figsize=(12, 5))
    
    # loss plot
    plt.subplot(1, 2, 2)
    plt.plot(range(len(loss)), loss, 'b-', label='Training loss')
    plt.title('Training and Validation Loss')
    plt.xticks(epochs)
    plt.yticks(tick_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(range(len(acc)), acc, 'b-', label='Training accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xticks(epochs)
    plt.yticks(tick_acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    if dir:
        plt.savefig(os.path.join(dir, 'training_plots.jpg'), format='jpg')
    plt.show()

    # Print the final results in a clean format
    print("\nFinal Training Accuracy: {:.2f}".format(acc[-1]))
    print("Final Training Loss: {:.4f}".format(loss[-1]))


def test_model(model, labels):
    '''
    Test the trained model adaptability

    Args
        model: model to test
        labels: outputs mappings ussed during training
        limit: number of classes to test the model on

    Returns
        loss history for training epoch
    '''


    # get paths to files
    paths = {}
    classes = dict(list(labels.items())[:-2])
    for num, cl in classes.items():
        class_folder = os.path.join(DATASET, cl)
        paths[cl] = []
        for image in os.listdir(class_folder):
            paths[cl].append(os.path.join(class_folder, image))
    
    # load actual images
    imgs = []
    true_labels = []
    for idx, data in enumerate(paths.items()):
        cl, lst = data
        imgs.extend(load_images(lst))
        true_labels.extend([idx] * len(lst))
    imgs = np.array(imgs)
    true_labels = np.array(true_labels)

    train, test, y_train, y_test = train_test_split(imgs, true_labels, train_size= .1, 
                                                    stratify= true_labels, random_state= 666)

    # fine tune model
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate= 1e-2,
        decay_steps=2,
        decay_rate=.6,
        staircase=True 
    )
    model.compile(optimizer= Adam(learning_rate=lr_schedule),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='accuracy', patience= 10,
                                mode='max', restore_best_weights=True,
                                start_from_epoch= 10)

    history = model.fit(train, y_train, 
                        epochs=20, callbacks= early_stopping
                        )
    
    _, axs = plt.subplots(1,2)
    axs[0].plot(history.history['loss'], 'b-')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[1].plot(history.history['accuracy'], 'r-')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_ylim(0,1)
    plt.tight_layout()
    plt.show()

    # test model performance
    test_loss, test_acc = model.evaluate(test,  y_test, verbose=2)
    print(f"Test loss: {test_loss, }\nTest accuracy: {test_acc}")

    # Generate the confusion matrix
    if test_acc >.5:
        y_pred = np.argmax(model.predict(test), axis=-1)
        
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= paths.keys())
        disp.plot(cmap=plt.cm.Blues, colorbar= False)
        plt.title('Confusion Matrix')
        plt.show()


def load_model():
    '''
    Lists the available models and possibly loads one

    Returns
        None, or the path to the chosen model with labels mappings
    '''

    models = os.listdir(MODELS) 
    if not models:
        print("No models are available to load")
        return None
    
    if len(models) > 1:
        # show folders containing models
        print("Available:")
        for i, model in enumerate(models):
            print(f"{i+1}. {model}")
        print(f"Other. Abort")
    
        try:
            choice = int(input(f"Choose a model (1-{len(models)}): ")) - 1
        except ValueError:
            raise ValueError(f"The choice must be a number")

        if choice < 0 or choice >= len(models):
            return
    else:
        choice = 0
        print(f"Only one model found: {models[0]}")


    # take the model path from the chosen folder
    dir = os.path.join(MODELS, models[choice])
    model = None
    for f in os.listdir(dir):
        if f.startswith("Model"):
            model = f 
            break

    if not model:
        print(f"No model found at {dir}")
        return
    
    # load model 
    model_path = os.path.join(dir, model)
    model = tf.keras.models.load_model(model_path)
    print(f"'{model_path}' loaded successfully")

    # take label mapping
    with open(os.path.join(dir, 'labels.pkl'), 'rb') as lbl:
        labels = pickle.load(lbl)
        
    return model, labels


@contextmanager
def measure_time(label):
    '''
    measures the time and prints it to the console
    '''

    start_time = time.time()
    yield
    print(f"{label} took {time.time()-start_time:.2f} seconds")


def main():

    if args.load:
        model, labels = load_model()
        test_model(model, labels)
        return 0

    with measure_time("Prepare Data"):
        training, train_classes, test_classes = prepare_data(DATASET, args.ratio)

    with measure_time("CNN creation"):
        cnn = CNN(input= (349, 128, 1), output= len(train_classes) + len(test_classes))
        #cnn.model.summary()

    print(f"\nTask batches structure:\
                    \n\tBatch size: {args.batch}\
                    \n\tClasses per batch: {args.N}\
                    \n\tSamples per class: {args.K}\n")

    # setup training
    maml = MAML(model= cnn, alpha= .001, beta= .001, steps=args.metapochs, epochs= args.epochs)

    # start training
    train_sampler = BatchSampler(training, args.N, args.K, args.batch, args.qs)
    history = maml.meta_training(train_sampler)
    print("Training finished")

    # if true save model
    model_dir = None
    if args.save:
        name = f"Model_{args.batch}-tasks_{args.N}-way_{args.K}-shot.keras"                
        dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_dir = os.path.join(MODELS, f"{dir}\\")
        os.makedirs(model_dir, exist_ok=True)
        maml.save(maml.get_model(), model_dir, name)

        # saving label mapping
        label_to_pred = train_sampler.get_labels() | {cl: idx+len(train_classes) for idx, cl in enumerate(test_classes)}
        pred_to_label = {v: k for k, v in label_to_pred.items()}
        with open(os.path.join(model_dir, "labels.pkl"), 'wb') as f:
            pickle.dump(pred_to_label, f)
    
    # training plots
    plot_history(history, model_dir)

if __name__ == '__main__':
    args = parse_args()
    main()







