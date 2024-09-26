import argparse


class Range:

    def __init__(self, min=None, max=None, t= None):
        '''
        A class to handle value limits for args in input

        Args
            min: minimun accepted value
            max: maximum accepted value     
        '''
        self.min = min
        self.max = max
        self._type = t

    def __call__(self, arg):
        '''
        Checks if the value arg respects constraints

        Args
            arg: a numeric value
        '''
        try:
            value = self._type(arg)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Must be a {self._type.__name__} number")
        
        if (self.min is not None):
            assert value >= self.min, f"Accepted range: {self.min} <= x <= {self.max}"
        if (self.max is not None):
            assert value <= self.max, f"Accepted range: {self.min} <= x <= {self.max}"
        
        return value


def parse_args():
    '''
    builds a parser for 
    command line arguments

    Returns
        args value
    '''

    # define types for args check
    class_type = Range(1, 10, int)
    batch_type = Range(2, 32, int)
    ratio_type = Range(0.5, 0.9, float)
    epoch_type = Range(min=1, t= int)

    parser = argparse.ArgumentParser()

    # pipeline
    parser.add_argument("--epochs", "-e", default=1000, type= epoch_type, help=f"number of epochs for the training loop (x >= {epoch_type.min})")
    parser.add_argument("--metapochs", "-me", default=2, type= epoch_type, help=f"number of epochs for the inner meta training loop (x >= {epoch_type.min})")
    parser.add_argument("-K", "-k", default=2, type= class_type, help=f"number of samples per class, used when building batches(from {class_type.min} to {class_type.max})")
    parser.add_argument("-N", "-n", default=2, type= class_type, help=f"number of classes per task, used when building batches(from {class_type.min} to {class_type.max})")
    parser.add_argument("--batch", "-B", "-b", default=8, type= batch_type, help=f"number of taks per batch, used when building batches(from {batch_type.min} to {batch_type.max})")
    parser.add_argument("-qs", action="store_true", help=f"if true query dimension is the same as support dimension, otherwise is 1")
    parser.add_argument("--ratio", "-rt", default=0.8, type=ratio_type, help=f"a ratio for deciding how large the training set will be(from {ratio_type.min} to {ratio_type.max})")
    
    # others
    parser.add_argument("--save", "-sv", action="store_true", help="save model after training")
    parser.add_argument("--load", "-l", action="store_true", help="choose model to load and test on (shows from the models folder)")

    return parser.parse_args()