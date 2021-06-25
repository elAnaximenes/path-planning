import numpy as np
import json
import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from .base_loaders.loaders import DataLoader 

class FeedForwardValidateDataLoader(DataLoader):

    def __init__(self, numBatches, dataDirectory = './data/batches-validate/'):

        super().__init__(self, numBatches=numBatches, dataDirectory=dataDirectory)
        self.dataset = None


