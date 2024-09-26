# Music Genre Classification with a MAML trained CNN

Music genre classification using image recognition. The project implements a MAML training routine on a CNN, and the possibility to test the adaptability of a trained model.

## Usage
Start the algorithm:
```
usage: path/to/main.py [-h] [--epochs EPOCHS] [--metapochs METAPOCHS] [-K K] [-N N] [--batch BATCH] [-qs] [--ratio RATIO] [--save] [--load]
```

Usage options:
```
options:
  -h, --help            show this help message and exit

  --epochs EPOCHS, -e EPOCHS
                        number of epochs for the training loop (x >= 1)

  --metapochs METAPOCHS, -me METAPOCHS
                        number of epochs for the inner meta training loop (x >= 1)
  -K K, -k K            number of samples per class, used when building batches(from 1 to 10)

  -N N, -n N            number of classes per task, used when building batches(from 1 to 10)

  --batch BATCH, -B BATCH, -b BATCH
                        number of taks per batch, used when building batches(from 2 to 32)

  -qs                   if true query dimension is the same as support dimension, otherwise is 1

  --ratio RATIO, -rt RATIO
                        a ratio for deciding how large the training set will be(from 0.5 to 0.9)

  --save, -sv           save model after training

  --load, -l            choose model to load and test on (shows from the models folder)
```

## Repository Structure

main/\
├── dataset/\
│&emsp;&emsp;└── mel_spectrograms/&emsp;...contains original mel spectrograms images divided by class.\
│\
└── src/\
&emsp;&emsp;├── arg_parser.py&emsp;...python parser for command line arguments.\
&emsp;&emsp;├── audio_to_image.py&emsp;...function to create mel spectrograms from wav files.\
&emsp;&emsp;├── batch_sampler.py&emsp;...used to create training batches.\
&emsp;&emsp;├── cnn.py&emsp;...CNN structure.\
&emsp;&emsp;├── data_preparation.py&emsp;...initial preparation of classes and paths for files to use during training.\
&emsp;&emsp;├── main.py&emsp;...pipeline for the program(training/testing/saving depending on options).\
&emsp;&emsp;└── maml.py&emsp;...training loop.
